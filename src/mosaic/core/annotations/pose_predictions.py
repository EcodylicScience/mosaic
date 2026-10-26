"""Defines the file that a pose model's predictions on extracted frames are saved in.

One file contains one model's predictions on the frames of one entry camera of an
extraction run: every frame the model was given, the objects found in the order the
tool returned them, and the raw instance and keypoint scores on the tool's own
scale. An object's position in its frame is its instance index, which
:func:`prediction_source_ref` names. Coordinates are the pixels of the extracted
image, origin at the top-left corner.

The payload is canonical. Frames are sorted by image, skeleton edges are oriented
lower position first and sorted, keys are sorted, the form is compact, and every
coordinate and score is a float. A keypoint with a non-finite coordinate is written
unplaced, a non-finite score as null, and a box with a non-finite side as no box.
Equal predictions therefore give equal bytes, and no field depends on the time of
writing. Model references and parameters are recorded by the run that writes the
file.

:func:`accepted_object` turns one predicted object into an annotation object of a
set's pose, through a :class:`~mosaic.core.annotations.keypoint_map.KeypointMap`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, FiniteFloat

from mosaic.core.annotations.keypoint_map import KeypointMap
from mosaic.core.annotations.model import Bbox, Keypoint, KeypointSchema
from mosaic.core.annotations.pose_annotations import PoseObject, canonical_pairs
from mosaic.core.annotations.versioned_file import read_versioned_file
from mosaic.core.json_value import JsonValue
from mosaic.core.strict_model import StrictModel

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "FramePredictions",
    "PredictedFrame",
    "PredictedObject",
    "PredictedPose",
    "accepted_object",
    "finite_or_none",
    "keypoint_accepted",
    "parse_prediction_source_ref",
    "pose_predictions_json_schema",
    "pose_predictions_payload",
    "prediction_source_ref",
    "read_pose_predictions",
    "score_passes",
]

FORMAT_NAME: Final = "mosaic-pose-predictions"
"""The ``format`` every file of this kind declares."""

FORMAT_VERSION: Final = 1
"""The version this module writes, and the newest it reads."""


@dataclass(frozen=True, slots=True)
class PredictedObject:
    """Records one instance that a model found in one frame.

    Attributes:
        keypoints: One per keypoint of the model's pose, in its order. A placed
            keypoint is ``Keypoint(x, y, 2, score)`` with the tool's raw
            confidence, or a ``None`` score when the tool reports none. A
            keypoint the tool did not place is :meth:`Keypoint.absent`.
        score: The tool's instance score, or ``None`` when the tool reports none.
        track_id: The track name that a SLEAP identity model assigned, or ``None``.
        bbox: The box the tool reported, or ``None``. An accepted object does not
            take it over.
    """

    keypoints: tuple[Keypoint, ...]
    score: float | None = None
    track_id: str | None = None
    bbox: Bbox | None = None


@dataclass(frozen=True, slots=True)
class PredictedFrame:
    """Records one extracted image and the objects that a model found in it.

    Attributes:
        image: The image's dataset-relative POSIX path, its image key.
        frame_index: The frame's index on the entry's media axis.
        width: Image width in pixels.
        height: Image height in pixels.
        objects: The objects in the order the tool returned them. An object's
            position is its instance index. Empty means the frame was predicted
            and nothing was found.
    """

    image: str
    frame_index: int
    width: int
    height: int
    objects: tuple[PredictedObject, ...]


@dataclass(frozen=True, slots=True)
class PredictedPose:
    """Names the pose that a model predicts, with its keypoint layout.

    Attributes:
        id: The authoring store's id of the pose the model was trained on, or
            ``None`` when provenance does not record it.
        name: That pose's name, ``""`` when unknown.
        schema: The model's keypoint names, in the order every object stores its
            points, with the skeleton edges and mirror pairs.
    """

    id: int | None
    name: str
    schema: KeypointSchema


@dataclass(frozen=True, slots=True)
class FramePredictions:
    """Records one model's predictions on the frames of one extraction entry camera.

    Attributes:
        pose: The pose every object's keypoints are aligned to.
        frames_run_id: The extraction run that the frames come from.
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The entry's camera, ``""`` for an entry of one camera.
        frames: Every frame the model was given.
    """

    pose: PredictedPose
    frames_run_id: str
    group: str
    sequence: str
    camera: str
    frames: tuple[PredictedFrame, ...]

    def __post_init__(self) -> None:
        """Refuse an edge off the keypoints, a repeated image, or an object off layout.

        A skeleton edge may name its keypoints in either order, since the payload
        orients it, and must name two distinct positions of the pose. The reader
        then accepts every file that the writer gives.
        """
        expected = self.pose.schema.num_keypoints
        for a, b in self.pose.schema.skeleton:
            if not 0 <= min(a, b) < max(a, b) < expected:
                msg = (
                    f"skeleton edge {(a, b)} is not two distinct keypoints of "
                    f"{list(self.pose.schema.names)}"
                )
                raise ValueError(msg)
        images: set[str] = set()
        for frame in self.frames:
            if frame.image in images:
                msg = f"two frames name the image {frame.image}"
                raise ValueError(msg)
            images.add(frame.image)
            for index, obj in enumerate(frame.objects):
                if len(obj.keypoints) != expected:
                    msg = (
                        f"{frame.image} object {index} has {len(obj.keypoints)} "
                        f"keypoints, but the pose names {expected}: "
                        f"{list(self.pose.schema.names)}"
                    )
                    raise ValueError(msg)


# --- Writing ---------------------------------------------------------------------


def pose_predictions_payload(predictions: FramePredictions) -> bytes:
    """Return the canonical bytes of *predictions*: compact, keys sorted."""
    pose = predictions.pose
    edges = [(min(a, b), max(a, b)) for a, b in pose.schema.skeleton]
    document: dict[str, JsonValue] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "pose": {
            "id": pose.id,
            "name": pose.name,
            "keypoints": list(pose.schema.names),
            "skeleton": canonical_pairs(edges),
            "symmetries": canonical_pairs(pose.schema.symmetries),
        },
        "source": {
            "frames_run_id": predictions.frames_run_id,
            "group": predictions.group,
            "sequence": predictions.sequence,
            "camera": predictions.camera,
        },
        "frames": [
            _frame_document(frame)
            for frame in sorted(predictions.frames, key=lambda frame: frame.image)
        ],
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _placed(point: Keypoint) -> bool:
    """Return :attr:`Keypoint.is_placed`, false also for an infinite coordinate."""
    return point.is_placed and math.isfinite(point.x) and math.isfinite(point.y)


def _frame_document(frame: PredictedFrame) -> dict[str, JsonValue]:
    return {
        "image": frame.image,
        "frame_index": frame.frame_index,
        "width": frame.width,
        "height": frame.height,
        "objects": [_object_document(obj) for obj in frame.objects],
    }


def _box_document(box: Bbox | None) -> JsonValue:
    if box is None:
        return None
    sides = {"x": box.x, "y": box.y, "width": box.width, "height": box.height}
    if not all(math.isfinite(side) for side in sides.values()):
        return None
    return {name: float(side) for name, side in sides.items()}


def _object_document(obj: PredictedObject) -> dict[str, JsonValue]:
    return {
        "score": finite_or_none(obj.score),
        "track_id": obj.track_id,
        "bbox": _box_document(obj.bbox),
        "keypoints": [
            [float(point.x), float(point.y), finite_or_none(point.score)]
            if _placed(point)
            else None
            for point in obj.keypoints
        ],
    }


# --- Reading ---------------------------------------------------------------------


# The file is mosaic's own. A key these models do not know is a mistake, because a
# newer writer's additions come with a newer version, which read_versioned_file
# refuses before these models see the file. Their descriptions are the format's
# documentation, rendered into the reference by pose_predictions_json_schema.


class PredictedPoseRecord(StrictModel):
    """Declares the model's pose as the file stores it."""

    id: int | None = Field(
        description=(
            "The authoring store's id of the pose the model was trained on, or null "
            "when provenance does not record it."
        )
    )
    name: str = Field(description='That pose\'s name, "" when unknown.')
    keypoints: list[str] = Field(
        description=(
            "The model's keypoint names, in the order every object stores its "
            "points. kp0 to kp{n-1} when neither provenance nor the tool names them."
        )
    )
    skeleton: list[tuple[int, int]] = Field(
        default_factory=list,
        description=(
            "Undirected edges as pairs of 0-based keypoint positions, each lower "
            "first, ascending."
        ),
    )
    symmetries: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Left-right mirror pairs as 0-based positions, lower first.",
    )


class PredictionSourceRecord(StrictModel):
    """Names the extraction run and entry camera of every frame in the file."""

    frames_run_id: str = Field(description="The extraction run the frames come from.")
    group: str = Field(description="The entry's group.")
    sequence: str = Field(description="The entry's sequence.")
    camera: str = Field(description='The entry\'s camera, "" for one camera.')


class PredictedBoxRecord(StrictModel):
    """Declares a box that the tool reported, in pixels from the top-left corner."""

    x: FiniteFloat = Field(description="The left edge.")
    y: FiniteFloat = Field(description="The top edge.")
    width: FiniteFloat = Field(description="The width.")
    height: FiniteFloat = Field(description="The height.")


class PredictedObjectRecord(StrictModel):
    """Declares a predicted object as the file stores it."""

    keypoints: list[tuple[FiniteFloat, FiniteFloat, FiniteFloat | None] | None] = Field(
        description=(
            "One entry per keypoint of the pose, in its order: [x, y, score] in "
            "image pixels, or null when the tool did not place the keypoint. The "
            "score is the tool's raw confidence, null when it reports none."
        )
    )
    score: FiniteFloat | None = Field(
        default=None,
        description="The tool's instance score, or null when it reports none.",
    )
    track_id: str | None = Field(
        default=None,
        description="The track name a SLEAP identity model assigned, or null.",
    )
    bbox: PredictedBoxRecord | None = Field(
        default=None, description="The box the tool reported, or null."
    )


class PredictedFrameRecord(StrictModel):
    """Declares a frame as the file stores it."""

    image: str = Field(description="The image's dataset-relative POSIX path.")
    frame_index: int = Field(description="The frame's index on the entry's media axis.")
    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    objects: list[PredictedObjectRecord] = Field(
        description=(
            "The objects in the order the tool returned them. An object's position "
            "is its instance index. Empty when the tool found nothing."
        )
    )


class PosePredictionsFile(StrictModel):
    """Declares a ``mosaic-pose-predictions`` file."""

    format: Literal["mosaic-pose-predictions"] = Field(
        description="Always mosaic-pose-predictions."
    )
    version: int = Field(
        description="The format version the file was written in. This is version 1."
    )
    pose: PredictedPoseRecord = Field(
        description="The pose every object's keypoints are aligned to."
    )
    source: PredictionSourceRecord = Field(
        description="The extraction run and entry camera of the frames."
    )
    frames: list[PredictedFrameRecord] = Field(
        description="Every frame the model was given, ascending by image."
    )


def pose_predictions_json_schema() -> dict[str, JsonValue]:
    """Return the file's JSON Schema, which is also its documentation."""
    schema: dict[str, JsonValue] = PosePredictionsFile.model_json_schema()
    return schema


def read_pose_predictions(path: Path) -> FramePredictions:
    """Read a file that :func:`pose_predictions_payload` wrote.

    A placed keypoint reads back as ``Keypoint(x, y, 2, score)``, and an unplaced
    one as :meth:`Keypoint.absent`.

    Raises:
        ValueError: The file is not this format, is a newer version of it than
            this mosaic reads, has a skeleton edge that is not two keypoint
            positions written lower first, or holds predictions that the
            in-memory types refuse.
    """
    parsed = read_versioned_file(
        path,
        PosePredictionsFile,
        format_name=FORMAT_NAME,
        newest_version=FORMAT_VERSION,
    )
    pose = parsed.pose
    for low, high in pose.skeleton:
        if not 0 <= low < high < len(pose.keypoints):
            msg = (
                f"{path} skeleton edge {(low, high)} is not two distinct keypoints "
                f"of {pose.keypoints} written lower position first"
            )
            raise ValueError(msg)
    return FramePredictions(
        pose=PredictedPose(
            id=pose.id,
            name=pose.name,
            schema=KeypointSchema(
                names=tuple(pose.keypoints),
                skeleton=tuple(pose.skeleton),
                symmetries=tuple(pose.symmetries),
            ),
        ),
        frames_run_id=parsed.source.frames_run_id,
        group=parsed.source.group,
        sequence=parsed.source.sequence,
        camera=parsed.source.camera,
        frames=tuple(
            PredictedFrame(
                image=frame.image,
                frame_index=frame.frame_index,
                width=frame.width,
                height=frame.height,
                objects=tuple(_read_object(obj) for obj in frame.objects),
            )
            for frame in parsed.frames
        ),
    )


def _read_object(record: PredictedObjectRecord) -> PredictedObject:
    box = record.bbox
    return PredictedObject(
        keypoints=tuple(
            Keypoint.absent()
            if point is None
            else Keypoint(x=point[0], y=point[1], visibility=2, score=point[2])
            for point in record.keypoints
        ),
        score=record.score,
        track_id=record.track_id,
        bbox=None
        if box is None
        else Bbox(x=box.x, y=box.y, width=box.width, height=box.height),
    )


# --- Source refs -----------------------------------------------------------------


def prediction_source_ref(run_id: str, instance: int) -> str:
    """Return ``"<run_id>#<instance>"``, the ``source_ref`` of an accepted object.

    Args:
        run_id: The prediction run.
        instance: The object's position in its frame's objects.

    Raises:
        ValueError: *run_id* is empty or contains ``#``, or *instance* is
            negative. :func:`parse_prediction_source_ref` could not read either
            back.
    """
    if not run_id or "#" in run_id or instance < 0:
        msg = (
            f"instance {instance} of run {run_id!r} cannot be named as a prediction "
            "source ref: the run id must be non-empty and without '#', and the "
            "instance 0 or more"
        )
        raise ValueError(msg)
    return f"{run_id}#{instance}"


def parse_prediction_source_ref(ref: str) -> tuple[str, int] | None:
    """Return the run id and instance that *ref* names, or ``None`` when malformed.

    Reads only the form :func:`prediction_source_ref` writes: a run id without
    ``#``, then ``#``, then the instance in ASCII digits without a leading zero.
    """
    run_id, separator, digits = ref.partition("#")
    if not run_id or not separator or not (digits.isascii() and digits.isdigit()):
        return None
    instance = int(digits)
    if str(instance) != digits:
        return None
    return run_id, instance


# --- Accepting -------------------------------------------------------------------


def accepted_object(
    predicted: PredictedObject,
    keypoint_map: KeypointMap,
    *,
    pose_id: int,
    source_ref: str,
    min_keypoint_score: float | None = None,
) -> PoseObject:
    """Return *predicted* as an annotation object of the map's target pose.

    Each target keypoint takes the source keypoint that the map names, placed
    with visibility 2 and without its score. A target keypoint is absent when it
    is unmapped, when the tool did not place its source keypoint, or when that
    keypoint scores below *min_keypoint_score*. A keypoint without a score passes
    the threshold, because the tool reported no confidence to compare. A track
    name becomes the object's ``track_id``.

    The returned box is ``None``. A set derives it from the accepted keypoints
    under its ``bbox_policy``, and the derived box follows every keypoint edit.
    The tool's box remains on *predicted*.

    Args:
        predicted: The object, in the model's keypoint order.
        keypoint_map: Maps the model's keypoints onto the set's pose.
        pose_id: The set's pose that the object is of.
        source_ref: What produced the object, from :func:`prediction_source_ref`.
        min_keypoint_score: The lowest keypoint score accepted, or ``None`` to
            accept every placed keypoint.

    Raises:
        ValueError: No keypoint is placed after mapping and the threshold. The
            message names *source_ref*, and whether no placed keypoint mapped or
            every mapped one scored below *min_keypoint_score*. Also raised when
            *predicted* has a keypoint count other than the map's source count.
    """
    mapped = keypoint_map.apply(predicted.keypoints)
    keypoints = tuple(
        Keypoint(x=point.x, y=point.y, visibility=2)
        if keypoint_accepted(point, min_keypoint_score)
        else Keypoint.absent()
        for point in mapped
    )
    if not any(point.visibility for point in keypoints):
        if any(_placed(point) for point in mapped):
            cause = (
                f"every mapped keypoint scores below min_keypoint_score "
                f"{min_keypoint_score}"
            )
        else:
            cause = (
                f"no placed keypoint maps onto the pose's keypoints "
                f"{list(keypoint_map.target.names)}"
            )
        msg = f"the predicted object {source_ref} has nothing to accept: {cause}"
        raise ValueError(msg)
    return PoseObject(
        pose_id=pose_id,
        keypoints=keypoints,
        origin="model",
        source_ref=source_ref,
        track_id=predicted.track_id or "",
    )


def keypoint_accepted(point: Keypoint, min_keypoint_score: float | None) -> bool:
    """Return whether *point* is placed and its score passes *min_keypoint_score*.

    A placed point has finite coordinates. :func:`score_passes` applies the
    threshold.
    """
    return _placed(point) and score_passes(point.score, min_keypoint_score)


def score_passes(score: float | None, threshold: float | None) -> bool:
    """Return whether *score* is at or above *threshold*.

    A score that is absent or not finite passes, because the tool reported no
    confidence to compare. Every score passes when *threshold* is ``None``.
    """
    finite = finite_or_none(score)
    return threshold is None or finite is None or finite >= threshold


def finite_or_none(value: float | None) -> float | None:
    """Return *value* as a float, or ``None`` when it is absent or not finite."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)
