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

:func:`probe_inference_env`, :func:`run_pose_inference_tool` and
:func:`run_point_inference_tool` sit at module scope because they are the seam the
marker suites replace, which is what exercises the whole op -- identifiers, the
claim, the bridge, the index row -- with no Ultralytics installed at all.

**Every refusal is decided from what the probe reported**, before a video is
opened, so a missing environment or a checkpoint from the wrong fork is a message
rather than a traceback out of ``torch.load`` on frame zero.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mosaic.tracking.common.toolenv import ToolEnv, ToolExitError, missing_output_error
from mosaic.tracking.common.ultralytics_env import (
    POLO_BOOTSTRAP,
    POLO_ENV,
    PROBE_DEADLINE_FLOOR_SECONDS,
    ULTRALYTICS_BOOTSTRAP,
    ULTRALYTICS_ENV,
    PoloError,
    PoloNotFoundError,
    RunnerSubcommand,
    UltralyticsError,
    UltralyticsNotFoundError,
    refuse_unloadable_model,
    run_runner,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    InferPointsRequest,
    InferPoseRequest,
    InferRequestBase,
    InferResponse,
    ProbeRequest,
    ProbeResponse,
)

INFER_REQUEST_NAME = "infer-request.json"
"""What one entry's request to the tool is called, inside its working directory."""

INFER_RESPONSE_NAME = "infer-response.json"
"""What the tool's reply is called, beside the request.

Both are byproducts of an attempt rather than results of one, so a re-run of the
``infer`` phase must delete them: a stale request beside fresh output is exactly
what the phase's clear globs exist to prevent. Those globs live in
:data:`~mosaic.core.pipeline.tracking_roots.TRACKING_ROOTS`, which is in ``core``
and so cannot import this module -- the two spell the same strings independently,
and ``test_infer_run_markers.py`` is what holds them together.
"""


class UnsupportedModelError(ValueError):
    """The weights are not the kind of model this op runs."""


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    """What one entry's inference produced."""

    predictions_path: Path
    n_frames: int
    n_rows: int


def probe_inference_env(
    model_path: Path | str,
    *,
    env: ToolEnv,
    failure: type[ToolExitError],
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> ProbeResponse:
    """Ask one environment what it holds, and what the weights are.

    Called once per run, before anything is minted, so the keypoint count, the
    model's declared task, whether this build is the fork and whether the weights
    load at all are known before the first entry runs.

    The request and the response live in a temporary directory: the probe runs
    before the run root exists, and nothing reads either file afterwards. No
    tracker is named, because inference runs none.

    A probe writes no progress lines -- it loads the weights and answers -- so an
    inactivity bound on it is a deadline on the whole operation. *idle_timeout*
    is therefore raised to
    :data:`~mosaic.tracking.common.ultralytics_env.PROBE_DEADLINE_FLOOR_SECONDS`
    when the caller's value is shorter.

    Raises:
        ToolExitError: The subclass given as *failure*, when the runner exited
            non-zero. Weights it cannot load are **not** among the reasons: those
            are reported in ``model_load_error`` and refused by the callers below.
        FileNotFoundError: The runner exited zero having written no response.
    """
    request = ProbeRequest(model_path=str(model_path))
    with tempfile.TemporaryDirectory(prefix="mosaic-infer-probe-") as scratch:
        request_path = Path(scratch) / "probe-request.json"
        response_path = Path(scratch) / "probe-response.json"
        _ = request_path.write_text(request.model_dump_json())
        stdout, stderr = run_runner(
            env,
            failure,
            "probe",
            request_path,
            response_path,
            idle_timeout=max(idle_timeout, PROBE_DEADLINE_FLOOR_SECONDS),
            max_runtime=max_runtime,
            conda_env=conda_env,
            bin_path=bin_path,
            cancel_check=cancel_check,
            on_output=on_output,
        )
        if not response_path.is_file():
            raise missing_output_error(env.tool, response_path, stdout, stderr)
        return ProbeResponse.model_validate_json(response_path.read_text())


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


def _run_inference(
    request: InferRequestBase,
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
) -> InferenceOutcome:
    """Run one video through *env*, and read back what it did.

    The request and the response are written into *work_dir*, beside the
    predictions parquet the runner publishes, so an attempt's whole exchange is on
    disk where the attempt is.

    **The caller's** ``idle_timeout`` **must exceed a cold model load.** It is
    passed through untouched, unlike the floor the probe puts under itself,
    because raising it here would blunt the one bound that supervises the run: the
    runner prints a line per decoded batch, and a value chosen so a wedged tool
    dies quickly is the whole point of the knob. The runner announces itself as
    soon as Ultralytics is imported, so the import is covered; the weights load
    runs between that line and the first batch, and is the longest silence a
    healthy run contains.
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
    predictions_path = Path(request.output_parquet)
    if not predictions_path.is_file():
        raise missing_output_error(env.tool, predictions_path, stdout, stderr)

    response = InferResponse.model_validate_json(response_path.read_text())
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
    """Run one video through a YOLO pose model in the Ultralytics environment."""
    return _run_inference(
        request,
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
    """Run one video through a POLO point model in the POLO environment."""
    return _run_inference(
        request,
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


__all__ = [
    "INFER_REQUEST_NAME",
    "INFER_RESPONSE_NAME",
    "InferenceOutcome",
    "UnsupportedModelError",
    "probe_inference_env",
    "require_points_model",
    "require_pose_model",
    "run_point_inference_tool",
    "run_pose_inference_tool",
]
