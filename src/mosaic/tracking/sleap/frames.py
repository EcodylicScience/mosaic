"""Predicts extracted frames with a trained SLEAP model, in the SLEAP environment.

A request's images are written as a frames-only ``.slp`` that references each
PNG where it is. ``sleap-nn track`` predicts every frame of it without tracking.
A snippet run by the SLEAP interpreter then reads the predicted ``.slp`` back as
JSON. Each reported frame is matched to its request image by its resolved path.
Two entries' images that share a file name therefore keep their own predictions.
sleap-nn 0.2.0 and 0.3.1 report each image of the input as a frame, and a frame
without instances is written with an empty object list. So is a request image
without a reported frame.

sleap-nn builds predicted instances on the skeleton stored with the model. The
readback therefore reports the model's nodes, whatever skeleton the input file
names.

The snippet takes only argv. No path is interpolated into its source, and mosaic
never imports ``sleap_io``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)
from mosaic.core.annotations.pose_predictions import PredictedObject
from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.toolenv import (
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from mosaic.core.strict_model import StrictModel
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ResolvedModel
from mosaic.tracking.sleap.labels import write_slp
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError, run_sleap_track

logger = logging.getLogger(__name__)

__all__ = [
    "SleapFrameAdapter",
    "SleapFrameReadback",
    "SleapInstanceReadback",
    "SleapReadback",
    "read_sleap_predictions",
]

_PLACEHOLDER_SCHEMA: Final = KeypointSchema(names=("kp0",))
"""The pose of the frames-only input when the model's training recorded none."""

_FRAMES_NAME: Final = "frames.slp"
"""What the frames-only input is called, in the work directory."""

_PREDICTED_NAME: Final = "predicted.slp"
"""What sleap-nn's predictions are called, in the work directory."""

_READBACK_NAME: Final = "predicted.json"
"""What the readback of the predictions is called, in the work directory."""

_READBACK_IDLE_SECONDS: Final = 300.0
"""How long the readback may print nothing. It prints once, after loading the file."""

# Run by the SLEAP interpreter. Its only inputs are argv: the predicted .slp and
# where to write the answer. A video is one file or an image list. A frame's
# image is its video's file, or the entry at frame_idx of an image list. A NaN
# coordinate or score is written as JSON's NaN token.
_READBACK_SNIPPET: str = """
import json
import os
import sys

import sleap_io

_predicted, _out = sys.argv[1], sys.argv[2]
_labels = sleap_io.load_slp(_predicted)


def _files(video):
    name = video.filename
    return list(name) if isinstance(name, (list, tuple)) else [name]


def _image(frame):
    name = frame.video.filename
    if isinstance(name, (list, tuple)):
        name = name[frame.frame_idx]
    return os.path.realpath(name)


_skeleton = None
_frames = []
for _frame in _labels.labeled_frames:
    _instances = []
    for _instance in _frame.instances:
        if not isinstance(_instance, sleap_io.PredictedInstance):
            continue
        if _skeleton is None:
            _skeleton = _instance.skeleton
        _instances.append(
            {
                "score": float(_instance.score),
                "track": None if _instance.track is None else _instance.track.name,
                "points": _instance.numpy(scores=True).tolist(),
            }
        )
    _frames.append({"image": _image(_frame), "instances": _instances})
if _skeleton is None and _labels.skeletons:
    _skeleton = _labels.skeletons[0]

_found = {
    "nodes": [] if _skeleton is None else list(_skeleton.node_names),
    "edges": [] if _skeleton is None else [list(e) for e in _skeleton.edge_inds],
    "symmetries": (
        [] if _skeleton is None else [list(s) for s in _skeleton.symmetry_inds]
    ),
    "images": [
        os.path.realpath(name) for _video in _labels.videos for name in _files(_video)
    ],
    "frames": _frames,
}
with open(_out, "w", encoding="utf-8") as _fh:
    json.dump(_found, _fh)
print(f"read {len(_frames)} predicted frames")
"""


class SleapInstanceReadback(StrictModel):
    """Reports one instance that sleap-nn predicted in one frame.

    Attributes:
        score: The instance score.
        track: The name of the instance's track, or ``None`` when it has none.
        points: ``(x, y, score)`` of each node, in node order. A coordinate or
            a score is NaN or ``None`` where the model placed no point.
    """

    score: float | None
    track: str | None
    points: list[tuple[float | None, float | None, float | None]]


class SleapFrameReadback(StrictModel):
    """Reports one predicted frame and the instances found in it.

    Attributes:
        image: The resolved absolute path of the frame's image.
        instances: The predicted instances, in the order sleap-nn stored them.
    """

    image: str
    instances: list[SleapInstanceReadback]


class SleapReadback(StrictModel):
    """Reports a predicted ``.slp``: the model's pose, its images and its frames.

    Attributes:
        nodes: The node names of the predicted instances' skeleton, in order.
        edges: The skeleton's edges as pairs of node indices, as stored.
        symmetries: The skeleton's mirror pairs as node indices.
        images: The resolved absolute path of every image of the file's
            videos, which are the videos of the input.
        frames: Every labeled frame of the file. sleap-nn 0.2.0 and 0.3.1
            write one for each image of the input, with no instances where
            they found none.
    """

    nodes: list[str]
    edges: list[tuple[int, int]]
    symmetries: list[tuple[int, int]]
    images: list[str]
    frames: list[SleapFrameReadback]


def read_sleap_predictions(
    predicted: Path,
    answer: Path,
    *,
    cancel_check: Callable[[], bool] | None = None,
    on_activity: Callable[[str], None] | None = None,
) -> SleapReadback:
    """Read the predicted ``.slp`` at *predicted* in the SLEAP environment.

    Args:
        predicted: The ``.slp`` that ``sleap-nn track`` wrote.
        answer: Where the snippet writes its JSON answer.
        cancel_check: Polled while the snippet runs.
        on_activity: Called with each line of either output stream.

    Returns:
        The predictions, with the pose that they are built on.

    Raises:
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: The snippet exited non-zero.
        FileNotFoundError: It exited cleanly and wrote no answer.
        pydantic.ValidationError: The answer is not a readback.
    """
    invocation = tool_invocation(SLEAP_ENV, executable="python")
    cmd = [*invocation, "-c", _READBACK_SNIPPET, str(predicted), str(answer)]
    logger.info("Running: %s", command_summary(cmd, SleapError.head))
    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        idle_timeout=_READBACK_IDLE_SECONDS,
        on_activity=on_activity,
    )
    if returncode != 0:
        raise SleapError(cmd, returncode, stdout, stderr)
    if not answer.is_file():
        raise missing_output_error("SLEAP", answer, stdout, stderr)
    return SleapReadback.model_validate_json(answer.read_text(encoding="utf-8"))


def _finite(value: float | None) -> float | None:
    """Return *value*, or ``None`` when it is absent or not finite."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _keypoint(x: float | None, y: float | None, score: float | None) -> Keypoint:
    """Return a node's point, unplaced when either coordinate is not finite."""
    placed_x = _finite(x)
    placed_y = _finite(y)
    if placed_x is None or placed_y is None:
        return Keypoint.absent()
    return Keypoint(x=placed_x, y=placed_y, visibility=2, score=_finite(score))


def _predicted_object(instance: SleapInstanceReadback) -> PredictedObject:
    """Return *instance* as a predicted object with its track's name."""
    return PredictedObject(
        keypoints=tuple(_keypoint(x, y, score) for x, y, score in instance.points),
        score=_finite(instance.score),
        track_id=instance.track or None,
    )


def _objects_by_image(
    images: Sequence[FrameImage], readback: SleapReadback
) -> tuple[tuple[PredictedObject, ...], ...]:
    """Match each reported frame to its request image by resolved path.

    Each image gets the instances of its reported frame as objects. A request
    image without a reported frame gets an empty tuple when the file's videos
    list it.

    Returns:
        The objects found in each image, in request order.

    Raises:
        ValueError: A reported frame names an image outside the request, or
            names a request image a second time, or the file's videos do not
            list a request image.
    """
    resolved = [image.path.resolve() for image in images]
    position_of = {
        path: image.position for image, path in zip(images, resolved, strict=True)
    }
    found: dict[int, tuple[PredictedObject, ...]] = {}
    for frame in readback.frames:
        reported = Path(frame.image).resolve()
        position = position_of.get(reported)
        if position is None:
            message = (
                f"sleap-nn predicted the image {reported}, which is not an image "
                "of the request"
            )
            raise ValueError(message)
        if position in found:
            message = f"sleap-nn predicted the image {reported} twice"
            raise ValueError(message)
        found[position] = tuple(
            _predicted_object(instance) for instance in frame.instances
        )
    listed = {Path(path).resolve() for path in readback.images}
    unlisted = [path for path in resolved if path not in listed]
    if unlisted:
        message = (
            f"The videos of sleap-nn's predictions do not list {len(unlisted)} of "
            f"{len(images)} images of the request, first {unlisted[0]}"
        )
        raise ValueError(message)
    return tuple(found.get(image.position, ()) for image in images)


def _tool_pose(readback: SleapReadback) -> KeypointSchema:
    """Return the model's pose as the readback reports it, mirror pairs low first."""
    pairs = sorted({(min(a, b), max(a, b)) for a, b in readback.symmetries})
    return KeypointSchema(
        names=tuple(readback.nodes),
        skeleton=tuple(readback.edges),
        symmetries=tuple(pairs),
    )


def _frames_only(
    images: Sequence[FrameImage], schema: KeypointSchema, root: Path
) -> AnnotationSet:
    """Return one frame per image and no objects, rooted at the dataset *root*.

    Each frame's path is its image key, which the written COCO file names
    relative to *root*.
    """
    frames = tuple(
        AnnotationFrame(
            image_path=Path(image.key), width=image.width, height=image.height
        )
        for image in images
    )
    return AnnotationSet(schema=schema, frames=frames, image_root=root)


@dataclass(frozen=True, slots=True)
class SleapFrameAdapter:
    """Predicts extracted images with a SLEAP model, or a SLEAP top-down pair.

    Attributes:
        schema: The pose that the model's training recorded, or ``None``. The
            frames-only input is written on it, else on the one keypoint
            ``kp0``. sleap-nn predicts on the model's own skeleton either way.
        images_root: The dataset root, which each image key is relative to.
    """

    schema: KeypointSchema | None
    images_root: Path

    def __call__(
        self,
        images: Sequence[FrameImage],
        model: ResolvedModel,
        params: PredictFramesParams,
        probe: ToolProbe,
        work_dir: Path,
        *,
        on_output: Callable[[str], None],
        cancel_check: Callable[[], bool],
    ) -> AdapterResult:
        """Predict *images* with ``sleap-nn track``, and read the predictions back.

        Each object has the model's nodes with their scores, the instance score
        and the name of its track. A node that the model did not place is an
        unplaced keypoint. The tool's pose is the readback's node names, edges
        and mirror pairs.

        Args:
            images: The images, in request order.
            model: The resolved train-sleap model, or the top-down pair.
            params: The run's parameters.
            probe: The probe of the SLEAP environment.
            work_dir: The directory for the input, the predictions and the
                readback.
            on_output: Called with each line that SLEAP prints on either stream.
            cancel_check: Returns ``True`` once the run is cancelled.

        Returns:
            One tuple of objects per image, in request order, and the tool's pose.

        Raises:
            SleapNotFoundError: No SLEAP install could be located.
            SleapError: A SLEAP process exited non-zero.
            FileNotFoundError: A SLEAP process exited zero without writing its
                output.
            ValueError: The predictions name an image outside the request or
                one image twice, or their videos do not list an image of the
                request.
        """
        _ = probe
        frames_slp = write_slp(
            _frames_only(images, self.schema or _PLACEHOLDER_SCHEMA, self.images_root),
            work_dir / _FRAMES_NAME,
            cancel_check=cancel_check,
            on_output=on_output,
        )
        predicted = work_dir / _PREDICTED_NAME
        _ = run_sleap_track(
            frames_slp,
            predicted,
            model_paths=model.paths,
            tracking=False,
            peak_threshold=params.peak_threshold,
            max_instances=params.max_instances,
            batch_size=params.batch_size,
            device=params.device,
            cancel_check=cancel_check,
            on_activity=on_output,
        )
        readback = read_sleap_predictions(
            predicted,
            work_dir / _READBACK_NAME,
            cancel_check=cancel_check,
            on_activity=on_output,
        )
        return AdapterResult(
            objects=_objects_by_image(images, readback), tool_pose=_tool_pose(readback)
        )
