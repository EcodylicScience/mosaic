"""Mosaic's side of the pose and point inference exchange.

Ultralytics is AGPL-3.0, and so is the POLO fork, so neither is imported here.
The imports live in :mod:`mosaic.tracking.external.runner.ultralytics_runner`, a
program that runs in an environment the user builds, and what crosses between the
two is a JSON request file, a JSON response file and progress lines on standard
output.

**Two environments, one program.** ``infer-pose`` runs upstream Ultralytics and
``infer-points`` runs the POLO fork, which cannot share an environment with it.
Both spawn the same runner, and which one answers is decided by the interpreter
mosaic spawns -- :data:`~mosaic.tracking.common.ultralytics_env.ULTRALYTICS_ENV`
or :data:`~mosaic.tracking.common.ultralytics_env.POLO_ENV`.

:func:`run_pose_inference_tool` and :func:`run_point_inference_tool` sit at module
scope because they are the seam the marker suites replace, which is what exercises
the whole op -- identifiers, the claim, the bridge, the index row -- with no
Ultralytics installed at all. The probe they run first is shared with training and
lives in :mod:`mosaic.tracking.common.ultralytics_env`.

:func:`predict_pose_frames` and :func:`predict_point_frames` are the frame
adapters of a predict-frames run for yolo-pose and POLO models. Each sends one
``predict-images`` request through :func:`run_predict_images_tool`, the seam beside
the inference tools, and reads the rows back as predicted objects. All three tools
share one request and response exchange.

**Every refusal is decided from what the probe reported**, before a video is
opened, so a missing environment or a checkpoint from the wrong fork is a message
rather than a traceback out of ``torch.load`` on frame zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mosaic.core.annotations.model import Bbox, Keypoint, KeypointSchema
from mosaic.core.annotations.pose_predictions import PredictedObject
from mosaic.core.pipeline.toolenv import ToolEnv, ToolExitError, missing_output_error
from mosaic.tracking.common.ultralytics_env import (
    POLO_BOOTSTRAP,
    POLO_ENV,
    ULTRALYTICS_BOOTSTRAP,
    ULTRALYTICS_ENV,
    PoloError,
    PoloNotFoundError,
    RunnerSubcommand,
    UltralyticsError,
    UltralyticsNotFoundError,
    UnsupportedModelError,
    refuse_unloadable_model,
    run_runner,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ImageSource,
    InferPointsRequest,
    InferPoseRequest,
    InferResponse,
    InferTask,
    PredictImagesRequest,
    PredictImagesResponse,
    ProbeResponse,
    request_columns,
)
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ResolvedModel

INFER_REQUEST_NAME = "infer-request.json"
"""What a request to the tool is called, inside its working directory."""

INFER_RESPONSE_NAME = "infer-response.json"
"""What the tool's reply is called, beside the request.

Both are byproducts of an attempt rather than results of one, so a re-run of the
``infer`` phase must delete them: a stale request beside fresh output is exactly
what the phase's clear globs exist to prevent. Those globs live in
:data:`~mosaic.core.pipeline.tracking_roots.TRACKING_ROOTS`, which is in ``core``
and so cannot import this module -- the two spell the same strings independently,
and ``tests/tracking/ops/test_infer_preflight.py`` holds them together.
"""

_ROWS_NAME: Final = "predict-images-rows.parquet"
"""The name of a ``predict-images`` request's rows file, in its work directory."""

_POINT_KEYPOINTS: Final = 1
"""How many keypoints a POLO object has: the one point that it locates."""


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    """What one entry's inference produced."""

    predictions_path: Path
    n_frames: int
    n_rows: int


def require_pose_model(probe: ProbeResponse, model_path: str) -> None:
    """Refuse, by name, anything ``infer-pose`` needs and the environment lacks."""
    if not probe.has_ultralytics:
        raise UltralyticsNotFoundError(
            "the Ultralytics environment resolved but holds no ultralytics. "
            f"{ULTRALYTICS_BOOTSTRAP}"
        )
    refuse_unloadable_model(probe, model_path)
    if probe.model_task != "pose":
        hint = (
            "POLO point detection has its own op: mosaic run --kind infer-points."
            if probe.model_task == "locate"
            else "'mosaic run --kind train-pose' produces the weights this op runs."
        )
        raise UnsupportedModelError(
            f"{model_path} is a {probe.model_task!r} model and 'infer-pose' runs "
            f"pose models. {hint}"
        )


def require_points_model(probe: ProbeResponse, model_path: str) -> None:
    """Refuse, by name, anything ``infer-points`` needs and the environment lacks.

    ``has_locate`` is checked **before** the weights, and that order is the whole
    value of the check. Upstream Ultralytics ships the same ``yolo`` console
    script under the same distribution name as the fork, so the last rung of the
    location ladder -- ``$PATH`` -- resolves to whichever is installed and cannot
    tell which it found. A user who never set ``MOSAIC_POLO_BIN`` would otherwise
    reach upstream here and be told only that their weights would not load.
    """
    if not probe.has_ultralytics:
        raise PoloNotFoundError(
            f"the POLO environment resolved but holds no ultralytics. {POLO_BOOTSTRAP}"
        )
    if not probe.has_locate:
        raise PoloNotFoundError(
            "the environment reached for point detection is upstream Ultralytics "
            f"({probe.ultralytics_version}), not the POLO fork: it defines no "
            "'locate' task. The two ship under one distribution name with the "
            "same console scripts, so a 'yolo' found on $PATH is as likely to be "
            f"either -- name the fork's environment explicitly. {POLO_BOOTSTRAP}"
        )
    refuse_unloadable_model(probe, model_path)
    if probe.model_task != "locate":
        raise UnsupportedModelError(
            f"{model_path} is a {probe.model_task!r} model and 'infer-points' runs "
            "point-detection models. 'mosaic run --kind train-points' produces the "
            "weights this op runs; a pose model is 'mosaic run --kind infer-pose'."
        )


def _run_runner_exchange[R: BaseModel](
    request: BaseModel,
    response_type: type[R],
    output_path: Path,
    subcommand: RunnerSubcommand,
    env: ToolEnv,
    failure: type[ToolExitError],
    *,
    work_dir: Path,
    idle_timeout: float,
    max_runtime: float | None,
    conda_env: str | None,
    bin_path: str | Path | None,
    cancel_check: Callable[[], bool] | None,
    on_output: Callable[[str], None] | None,
) -> R:
    """Send *request* to the runner's *subcommand* in *env*, and read its response.

    The request and the response are written into *work_dir*, beside the
    *output_path* that the runner publishes. An attempt's whole exchange is then
    on disk where the attempt is.

    **The caller's** ``idle_timeout`` **must exceed a cold model load.** It is
    passed through untouched, unlike the floor the probe puts under itself,
    because raising it here would blunt the one bound that supervises the run: the
    runner prints a line per decoded batch, and a value chosen so a wedged tool
    dies quickly is the whole point of the knob. The runner announces itself as
    soon as Ultralytics is imported, so the import is covered; the weights load
    runs between that line and the first batch, and is the longest silence a
    healthy run contains.

    Raises:
        ToolExitError: The subclass given as *failure*, when the runner exited
            non-zero.
        FileNotFoundError: The runner exited zero without writing its response
            or *output_path*.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / INFER_REQUEST_NAME
    response_path = work_dir / INFER_RESPONSE_NAME
    _ = request_path.write_text(request.model_dump_json())

    stdout, stderr = run_runner(
        env,
        failure,
        subcommand,
        request_path,
        response_path,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        conda_env=conda_env,
        bin_path=bin_path,
        cancel_check=cancel_check,
        on_output=on_output,
    )

    if not response_path.is_file():
        raise missing_output_error(env.tool, response_path, stdout, stderr)
    if not output_path.is_file():
        raise missing_output_error(env.tool, output_path, stdout, stderr)
    return response_type.model_validate_json(response_path.read_text())


def _inference_outcome(
    predictions_path: Path, response: InferResponse
) -> InferenceOutcome:
    """Build one entry's inference outcome from the runner's response."""
    return InferenceOutcome(
        predictions_path=predictions_path,
        n_frames=response.n_frames,
        n_rows=response.n_rows,
    )


def run_pose_inference_tool(
    request: InferPoseRequest,
    *,
    work_dir: Path,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> InferenceOutcome:
    """Run one entry through a YOLO pose model in the Ultralytics environment."""
    predictions_path = Path(request.output_parquet)
    response = _run_runner_exchange(
        request,
        InferResponse,
        predictions_path,
        "infer-pose",
        ULTRALYTICS_ENV,
        UltralyticsError,
        work_dir=work_dir,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        conda_env=conda_env,
        bin_path=bin_path,
        cancel_check=cancel_check,
        on_output=on_output,
    )
    return _inference_outcome(predictions_path, response)


def run_point_inference_tool(
    request: InferPointsRequest,
    *,
    work_dir: Path,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> InferenceOutcome:
    """Run one entry through a POLO point model in the POLO environment."""
    predictions_path = Path(request.output_parquet)
    response = _run_runner_exchange(
        request,
        InferResponse,
        predictions_path,
        "infer-points",
        POLO_ENV,
        PoloError,
        work_dir=work_dir,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        conda_env=conda_env,
        bin_path=bin_path,
        cancel_check=cancel_check,
        on_output=on_output,
    )
    return _inference_outcome(predictions_path, response)


def run_predict_images_tool(
    request: PredictImagesRequest,
    *,
    work_dir: Path,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> PredictImagesResponse:
    """Run a list of images through a pose or point model.

    A ``pose`` request runs in the Ultralytics environment and fails with
    :class:`UltralyticsError`. A ``locate`` request runs in the POLO environment
    and fails with :class:`PoloError`.
    """
    if request.task == "pose":
        env, failure = ULTRALYTICS_ENV, UltralyticsError
    else:
        env, failure = POLO_ENV, PoloError
    return _run_runner_exchange(
        request,
        PredictImagesResponse,
        Path(request.rows_path),
        "predict-images",
        env,
        failure,
        work_dir=work_dir,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        conda_env=conda_env,
        bin_path=bin_path,
        cancel_check=cancel_check,
        on_output=on_output,
    )


def _ultralytics_device(device: str) -> str:
    """Spell a predict-frames *device* for Ultralytics.

    ``auto`` is sent empty, and Ultralytics then takes the first CUDA device
    it finds, else the CPU. ``cpu`` and a CUDA index are sent as given.
    """
    return "" if device == "auto" else device


def _numeric_rows(
    rows: pd.DataFrame, columns: Sequence[str], id_column: str
) -> np.ndarray:
    """Return *rows*' *columns* as floats, sorted by image, then by *id_column*."""
    block = rows.loc[:, list(columns)].to_numpy(dtype=np.float64)
    detection = block[:, columns.index(id_column)]
    order = np.lexsort((detection, block[:, columns.index("frame")]))
    return block[order]


def _keypoint_score(score: float) -> float | None:
    """Return the tool's keypoint *score*, or None where the runner wrote NaN."""
    return None if math.isnan(score) else score


def _pose_objects(
    rows: pd.DataFrame, n_keypoints: int
) -> Mapping[int, list[PredictedObject]]:
    """Return the pose objects of each image position, in detection order.

    A keypoint is placed with the tool's score. An object's score is its
    detection's confidence, and its box spans the detection's corners.
    """
    columns = request_columns("pose", n_keypoints)
    pose_width = 2 + 3 * n_keypoints
    found: dict[int, list[PredictedObject]] = {}
    for row in _numeric_rows(rows, columns, "id"):
        values = [float(value) for value in row]
        keypoints = tuple(
            Keypoint(
                x=values[2 + 3 * k],
                y=values[3 + 3 * k],
                visibility=2,
                score=_keypoint_score(values[4 + 3 * k]),
            )
            for k in range(n_keypoints)
        )
        confidence, x1, y1, x2, y2 = values[pose_width:]
        box = Bbox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        detection = PredictedObject(keypoints=keypoints, score=confidence, bbox=box)
        found.setdefault(int(values[0]), []).append(detection)
    return found


def _point_objects(rows: pd.DataFrame) -> Mapping[int, list[PredictedObject]]:
    """Return the point objects of each image position, in detection order.

    POLO does not report a keypoint score or a box. An object's one keypoint
    therefore has a null score, and the point's confidence is the object's score.
    """
    columns = ["frame", "detection_id", "x", "y", "confidence"]
    found: dict[int, list[PredictedObject]] = {}
    for row in _numeric_rows(rows, columns, "detection_id"):
        position, _detection, x, y, confidence = (float(value) for value in row)
        point = Keypoint(x=x, y=y, visibility=2)
        detection = PredictedObject(keypoints=(point,), score=confidence)
        found.setdefault(int(position), []).append(detection)
    return found


def _predict_frames(
    task: InferTask,
    images: Sequence[FrameImage],
    model: ResolvedModel,
    params: PredictFramesParams,
    probe: ToolProbe,
    work_dir: Path,
    *,
    on_output: Callable[[str], None],
    cancel_check: Callable[[], bool],
) -> AdapterResult:
    """Predict a *task* model on *images* through :func:`run_predict_images_tool`.

    The op probes the environment, checks the model and checks its provenance
    before it takes any claim.
    """
    n_keypoints = probe.n_keypoints
    if n_keypoints is None:
        message = (
            "a yolo-pose or POLO model is predicted with the keypoint count of the "
            "Ultralytics probe, and this probe does not report one."
        )
        raise ValueError(message)
    request = PredictImagesRequest(
        model_path=str(model.path),
        task=task,
        images=[
            ImageSource(position=image.position, path=str(image.path))
            for image in images
        ],
        columns=request_columns(task, n_keypoints),
        conf=params.conf_threshold,
        iou=params.iou if task == "pose" else None,
        imgsz=params.imgsz,
        max_det=params.max_det,
        agnostic_nms=params.agnostic_nms,
        device=_ultralytics_device(params.device),
        precision=params.precision,
        batch_size=params.batch_size,
        n_keypoints=n_keypoints,
        dor=params.dor if task == "locate" else None,
        rows_path=str(work_dir / _ROWS_NAME),
    )
    _ = run_predict_images_tool(
        request, work_dir=work_dir, cancel_check=cancel_check, on_output=on_output
    )
    rows = pd.read_parquet(request.rows_path)
    if task == "pose":
        found = _pose_objects(rows, n_keypoints)
        pose_keypoints = n_keypoints
    else:
        found = _point_objects(rows)
        pose_keypoints = _POINT_KEYPOINTS
    return AdapterResult(
        objects=tuple(tuple(found.get(image.position, ())) for image in images),
        tool_pose=KeypointSchema(names=tuple(f"kp{k}" for k in range(pose_keypoints))),
    )


def predict_pose_frames(
    images: Sequence[FrameImage],
    model: ResolvedModel,
    params: PredictFramesParams,
    probe: ToolProbe,
    work_dir: Path,
    *,
    on_output: Callable[[str], None],
    cancel_check: Callable[[], bool],
) -> AdapterResult:
    """Predict a yolo-pose model on extracted images in the Ultralytics environment.

    Each object has the model's keypoints with their scores, its detection's
    confidence as its score, and its box. A keypoint's score is null when the
    model does not predict visibility. The tool's pose is ``kp0`` to ``kp{n-1}``
    for the probe's ``n`` keypoints, without edges.

    Args:
        images: The images, in request order.
        model: The resolved train-pose model.
        params: The run's parameters.
        probe: The probe of the Ultralytics environment, for its keypoint count.
        work_dir: The directory for the exchange and the rows file.
        on_output: Called with each line that the runner prints.
        cancel_check: Returns ``True`` once the run is cancelled.

    Returns:
        One tuple of objects per image, in request order, and the tool's pose.

    Raises:
        ValueError: *probe* does not report a keypoint count.
        UltralyticsError: The runner exited non-zero.
        FileNotFoundError: The runner exited zero without writing its response
            or its rows.
    """
    return _predict_frames(
        "pose",
        images,
        model,
        params,
        probe,
        work_dir,
        on_output=on_output,
        cancel_check=cancel_check,
    )


def predict_point_frames(
    images: Sequence[FrameImage],
    model: ResolvedModel,
    params: PredictFramesParams,
    probe: ToolProbe,
    work_dir: Path,
    *,
    on_output: Callable[[str], None],
    cancel_check: Callable[[], bool],
) -> AdapterResult:
    """Predict a POLO point model on extracted images in the POLO environment.

    Each object has one keypoint, the located point, with a null score. The
    object's score is the point's confidence, and its ``bbox`` is None. The
    tool's pose is the one keypoint ``kp0``.

    Args:
        images: The images, in request order.
        model: The resolved train-points model.
        params: The run's parameters.
        probe: The probe of the POLO environment, for its keypoint count.
        work_dir: The directory for the exchange and the rows file.
        on_output: Called with each line that the runner prints.
        cancel_check: Returns ``True`` once the run is cancelled.

    Returns:
        One tuple of objects per image, in request order, and the tool's pose.

    Raises:
        ValueError: *probe* does not report a keypoint count.
        PoloError: The runner exited non-zero.
        FileNotFoundError: The runner exited zero without writing its response
            or its rows.
    """
    return _predict_frames(
        "locate",
        images,
        model,
        params,
        probe,
        work_dir,
        on_output=on_output,
        cancel_check=cancel_check,
    )


__all__ = [
    "INFER_REQUEST_NAME",
    "INFER_RESPONSE_NAME",
    "InferenceOutcome",
    "predict_point_frames",
    "predict_pose_frames",
    "require_points_model",
    "require_pose_model",
    "run_point_inference_tool",
    "run_pose_inference_tool",
    "run_predict_images_tool",
]
