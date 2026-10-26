"""Predicts extracted frames with a trained Lightning Pose model, one image at a time.

A snippet run by the Lightning Pose interpreter loads the model once and calls
``Model.predict_frame`` on each image of a request, read as RGB. Its request is
a JSON file of ``[position, path]`` pairs in request order. Its answer contains
one result per image, keyed by the image's position, and the config's
``data.keypoint_names``. mosaic joins the results to the images by position
alone. The snippet does not use DALI, and its answer is the only file that it
writes.

``Model.predict_on_label_csv`` is not used, because it writes into the model
directory, which can belong to a linked library's dataset.

``Model.predict_frame`` is verified from Lightning Pose
:data:`LITPOSE_PREDICT_FRAME_MIN` on. :func:`probe_litpose_version` reads the
installed release before a run claims any entry, and the prediction snippet
repeats the comparison as a guard.

Both snippets take only argv. Paths are not interpolated into their source,
and mosaic does not import ``lightning_pose``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from pydantic import FiniteFloat

from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.annotations.pose_predictions import PredictedObject
from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.toolenv import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    captured_output,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from mosaic.core.strict_model import StrictModel
from mosaic.core.wording import counted
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.errors import PredictFramesUnavailable
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.litpose.run import LITPOSE_ENV, LitposeError
from mosaic.tracking.litpose.training import (
    CUDA_VISIBLE_DEVICES,
    litpose_device_placement,
)
from mosaic.tracking.model_refs import ResolvedModel

logger = logging.getLogger(__name__)

__all__ = [
    "LITPOSE_PREDICT_FRAME_MIN",
    "LitposeFrameAnswer",
    "LitposeFrameResult",
    "predict_litpose_frames",
    "probe_litpose_version",
    "require_predict_frame",
]

LITPOSE_PREDICT_FRAME_MIN: Final = "2.3.1"
"""The earliest Lightning Pose release whose ``Model.predict_frame`` is verified."""

_REQUEST_NAME: Final = "request.json"
"""What the prediction's request is called, in the work directory."""

_ANSWER_NAME: Final = "answer.json"
"""What the prediction's answer is called, in the work directory."""

_SNIPPET_IDLE_SECONDS: Final = 900.0
"""How long the prediction may print nothing. It prints nothing while the model loads."""

_TOO_OLD_EXIT: Final = 3
"""The exit status of the prediction snippet's version guard."""

_RELEASE: Final = re.compile(r"\d+(?:\.\d+)*")
"""The leading release numbers of a version, such as ``2.3.1`` of ``2.3.1.post1``."""

# Run by the Lightning Pose interpreter. It does not read argv, and prints the
# installed release.
_VERSION_SNIPPET: str = """
import lightning_pose

print(lightning_pose.__version__)
"""

# Run by the Lightning Pose interpreter. Its only inputs are argv: the model
# directory, the request, where to write the answer, the minimum release and
# the guard's exit status. The guard compares release numbers as mosaic does,
# prints the installed release and exits before the model loads.
_PREDICT_FRAME_SNIPPET: str = r"""
import json
import re
import sys

import lightning_pose

_model_dir, _request, _answer, _minimum, _too_old = sys.argv[1:6]


def _release(version):
    found = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in found.group(0).split(".")) if found else ()


_version = str(lightning_pose.__version__)
if _release(_version) < _release(_minimum):
    print(_version)
    sys.exit(int(_too_old))

import cv2
from lightning_pose.api import Model

_model = Model.from_dir(_model_dir, precision="fp32")
with open(_request, encoding="utf-8") as _fh:
    _pairs = json.load(_fh)
_results = []
for _position, _path in _pairs:
    _image = cv2.imread(_path, cv2.IMREAD_COLOR)
    if _image is None:
        sys.exit(f"cannot read the image {_path}")
    _found = _model.predict_frame(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
    _results.append(
        {
            "position": _position,
            "keypoints": _found["keypoints"].tolist(),
            "confidence": _found["confidence"].tolist(),
        }
    )
    print(f"predicted {len(_results)} of {len(_pairs)} images", flush=True)
_names = _model.cfg.data.get("keypoint_names")
with open(_answer, "w", encoding="utf-8") as _fh:
    json.dump(
        {
            "keypoint_names": [] if _names is None else [str(n) for n in _names],
            "results": _results,
        },
        _fh,
    )
"""


class LitposeFrameResult(StrictModel):
    """Reports what ``Model.predict_frame`` returned for one image.

    Attributes:
        position: The image's place in the request.
        keypoints: ``(x, y)`` of each keypoint in original-frame pixels, in the
            model's order.
        confidence: The confidence of each keypoint, in the same order.
    """

    position: int
    keypoints: list[tuple[FiniteFloat, FiniteFloat]]
    confidence: list[FiniteFloat]


class LitposeFrameAnswer(StrictModel):
    """Reports the prediction snippet's results and the model's keypoint names.

    Attributes:
        keypoint_names: The config's ``data.keypoint_names``, empty when the
            config names none.
        results: One result per image of the request, in any order.
    """

    keypoint_names: list[str]
    results: list[LitposeFrameResult]


def _release(version: str) -> tuple[int, ...]:
    """Return the leading release numbers of *version*, empty when it has none."""
    found = _RELEASE.match(version)
    if found is None:
        return ()
    return tuple(int(part) for part in found.group(0).split("."))


def require_predict_frame(version: str) -> None:
    """Refuse a Lightning Pose release older than :data:`LITPOSE_PREDICT_FRAME_MIN`.

    Args:
        version: The installed release, as Lightning Pose reports it.

    Raises:
        PredictFramesUnavailable: *version* is an older release, or names no
            release. The message names both versions.
    """
    if _release(version) >= _release(LITPOSE_PREDICT_FRAME_MIN):
        return
    message = (
        f"The Lightning Pose environment has Lightning Pose {version!r}, and "
        f"predicting single frames needs {LITPOSE_PREDICT_FRAME_MIN} or later. "
        "Install a later release in the environment that "
        f"{LITPOSE_ENV.conda_env_var} or {LITPOSE_ENV.bin_var} names."
    )
    raise PredictFramesUnavailable(message)


def _printed_version(stdout: str) -> str:
    """Return the last line that a snippet printed, ``""`` when it printed none."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def probe_litpose_version(*, cancel_check: Callable[[], bool] | None = None) -> str:
    """Return the release of the located Lightning Pose install.

    Args:
        cancel_check: Polled while the probe runs.

    Returns:
        The last line that the probe printed, ``lightning_pose.__version__``.

    Raises:
        LitposeNotFoundError: No Lightning Pose install could be located.
        LitposeError: The probe exited non-zero.
        ValueError: The probe exited zero without printing a version.
    """
    cmd = [*tool_invocation(LITPOSE_ENV, executable="python"), "-c", _VERSION_SNIPPET]
    logger.info("Running: %s", command_summary(cmd, LitposeError.head))
    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        idle_timeout=PROBE_DEADLINE_FLOOR_SECONDS,
    )
    if returncode != 0:
        raise LitposeError(cmd, returncode, stdout, stderr)
    version = _printed_version(stdout)
    if not version:
        message = (
            "The Lightning Pose environment did not print a version.\n"
            f"{captured_output(stdout, stderr)}"
        )
        raise ValueError(message)
    return version


def _predicted_object(
    result: LitposeFrameResult, names: Sequence[str]
) -> PredictedObject:
    """Return *result* as one object without a score, a track or a box.

    Raises:
        ValueError: The result has other than one point and one confidence per
            keypoint name.
    """
    expected = len(names)
    if len(result.keypoints) != expected or len(result.confidence) != expected:
        message = (
            f"Lightning Pose answered position {result.position} with "
            f"{counted(len(result.keypoints), 'point')} and "
            f"{counted(len(result.confidence), 'confidence')}, and the model's "
            f"config names {counted(expected, 'keypoint')}: {list(names)}"
        )
        raise ValueError(message)
    keypoints = tuple(
        Keypoint(x=x, y=y, visibility=2, score=confidence)
        for (x, y), confidence in zip(result.keypoints, result.confidence, strict=True)
    )
    return PredictedObject(keypoints=keypoints)


def _objects_by_position(
    images: Sequence[FrameImage], answer: LitposeFrameAnswer
) -> tuple[tuple[PredictedObject, ...], ...]:
    """Join each result of *answer* to the request image at its position.

    Returns:
        One object per image, in request order.

    Raises:
        ValueError: The config does not name the keypoints, a result names a
            position outside the request or a position a second time, an image
            lacks a result, or a result's keypoints do not match the names.
    """
    names = answer.keypoint_names
    if not names:
        message = (
            "The model's config.yaml does not set data.keypoint_names, which "
            "name the keypoints that Lightning Pose predicts"
        )
        raise ValueError(message)
    positions = {image.position for image in images}
    found: dict[int, PredictedObject] = {}
    for result in answer.results:
        if result.position not in positions:
            message = (
                f"Lightning Pose answered position {result.position}, which no "
                "image of the request has"
            )
            raise ValueError(message)
        if result.position in found:
            message = f"Lightning Pose answered position {result.position} twice"
            raise ValueError(message)
        found[result.position] = _predicted_object(result, names)
    missing = [image.position for image in images if image.position not in found]
    if missing:
        message = (
            f"Lightning Pose did not answer the image at position {missing[0]}, "
            f"one of {counted(len(missing), 'image')} without a result"
        )
        raise ValueError(message)
    return tuple((found[image.position],) for image in images)


def predict_litpose_frames(
    images: Sequence[FrameImage],
    model: ResolvedModel,
    params: PredictFramesParams,
    probe: ToolProbe,
    work_dir: Path,
    *,
    on_output: Callable[[str], None],
    cancel_check: Callable[[], bool],
) -> AdapterResult:
    """Predict a Lightning Pose model on extracted images in its environment.

    Each image has one object: the model's keypoints in original-frame pixels
    with their confidences. The object's score, track and box are null. The
    tool's pose is the config's ``data.keypoint_names``, without edges. The
    model loads at fp32, whatever the run's ``precision``. The device is passed
    to the snippet through :func:`litpose_device_placement`. ``auto`` leaves
    the environment as inherited, and CUDA indices set ``CUDA_VISIBLE_DEVICES``
    within the inherited list.

    Args:
        images: The images, in request order.
        model: The resolved train-litpose model.
        params: The run's parameters.
        probe: The version probe's answer, which the run checked before it
            took any claim.
        work_dir: The directory for the request and the answer.
        on_output: Called with each line that the snippet prints on either
            stream.
        cancel_check: Returns ``True`` once the run is cancelled.

    Returns:
        One tuple of objects per image, in request order, and the tool's pose.

    Raises:
        LitposeNotFoundError: No Lightning Pose install could be located.
        PredictFramesUnavailable: The snippet's guard found a release older
            than :data:`LITPOSE_PREDICT_FRAME_MIN`.
        LitposeError: The snippet exited non-zero otherwise.
        FileNotFoundError: The snippet exited zero without writing its answer.
        ValueError: The answer does not give each image of the request one
            result of the config's keypoints, or is malformed.
    """
    _ = probe
    placement = litpose_device_placement(
        params.device, os.environ.get(CUDA_VISIBLE_DEVICES)
    )
    request = work_dir / _REQUEST_NAME
    pairs = [[image.position, str(image.path)] for image in images]
    _ = request.write_text(json.dumps(pairs), encoding="utf-8")
    answer = work_dir / _ANSWER_NAME
    cmd = [
        *tool_invocation(LITPOSE_ENV, executable="python"),
        "-c",
        _PREDICT_FRAME_SNIPPET,
        str(model.path),
        str(request),
        str(answer),
        LITPOSE_PREDICT_FRAME_MIN,
        str(_TOO_OLD_EXIT),
    ]
    logger.info("Running: %s", command_summary(cmd, LitposeError.head))
    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(placement.env_overlay),
        cancel_check=cancel_check,
        idle_timeout=_SNIPPET_IDLE_SECONDS,
        on_activity=on_output,
    )
    if returncode == _TOO_OLD_EXIT:
        require_predict_frame(_printed_version(stdout))
    if returncode != 0:
        raise LitposeError(cmd, returncode, stdout, stderr)
    if not answer.is_file():
        raise missing_output_error("Lightning Pose", answer, stdout, stderr)
    found = LitposeFrameAnswer.model_validate_json(answer.read_text(encoding="utf-8"))
    return AdapterResult(
        objects=_objects_by_position(images, found),
        tool_pose=KeypointSchema(names=tuple(found.keypoint_names)),
    )
