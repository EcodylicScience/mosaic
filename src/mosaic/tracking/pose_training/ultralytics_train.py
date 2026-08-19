"""Mosaic's side of the pose and point training exchange.

Ultralytics is AGPL-3.0, and so is the POLO fork, so neither is imported here.
The imports live in :mod:`mosaic.tracking.external.runner.ultralytics_runner`, a
program that runs in an environment the user builds, and what crosses between the
two is a JSON request file, a JSON response file and progress lines on standard
output.

**Two environments, one program**, as inference already does: ``train-pose`` runs
upstream Ultralytics and ``train-points`` runs the POLO fork, which cannot share
an environment with it, and which one answers is decided by the interpreter
mosaic spawns.

:func:`run_pose_training_tool` and :func:`run_point_training_tool` sit at module
scope because they are the seam the training suites replace, which is what
exercises the whole op -- the identifier, the claim, the index row -- with no
Ultralytics installed at all.

**The exchange is per attempt**, under ``.mosaic-train/<execution_id>/`` inside
the claimed run root. Training reuses one run root across attempts by
construction -- identical params mint one identifier, and retraining writes into
the same directory -- so a fixed name would let one attempt's leftovers be read
as the next attempt's. That matters three times over here: a stale response would
be read as this run's answer by a tool that exited without writing one, and a
stale cancel sentinel would stop the next attempt at its first epoch boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mosaic.tracking.common.toolenv import ToolEnv, ToolExitError, missing_output_error
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
    ProbeResponse,
    TrainPointsRequest,
    TrainPoseRequest,
    TrainRequestBase,
    TrainResponse,
    TrainStop,
)

EXCHANGE_DIRECTORY = ".mosaic-train"
"""Where one training run's attempts keep their exchange, inside the run root.

Dot-prefixed and ``.mosaic-`` named for the reason every other marker is: it sits
in a directory Ultralytics also writes into, and the prefix is what says at a
glance which files are mosaic's. It is a sibling of the ``train/`` subdirectory
the tool owns, never inside it.
"""

TRAIN_REQUEST_NAME = "request.json"
"""What one attempt asked the tool to do."""

TRAIN_RESPONSE_NAME = "response.json"
"""What the tool reported back."""

CANCEL_SENTINEL_NAME = "cancel"
"""The file the tool stats between epochs, written when the job is cancelled.

Never written by the tool, and never written in advance. Its presence is the
whole message.
"""


class TrainingDirectoryError(RuntimeError):
    """The tool wrote its model somewhere other than where mosaic will read it."""


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """What one training run produced, and how it ended."""

    save_dir: Path
    """Where the weights and the curve are, confirmed to be where mosaic looked."""

    epochs_completed: int
    """Epochs that finished, which is not always the epochs that were asked for."""

    stop: TrainStop
    """``completed``, ``early_stopped`` or ``cancelled``.

    The one fact that cannot be recovered from disk: all three leave ``best.pt``,
    ``last.pt`` and a ``results.csv``.
    """


def attempt_directory(run_root: Path, execution_id: str) -> Path:
    """Where this attempt's request, response and sentinel live."""
    return run_root / EXCHANGE_DIRECTORY / execution_id


def require_pose_training_env(probe: ProbeResponse, base_weights: str) -> None:
    """Refuse, by name, anything ``train-pose`` needs and the environment lacks.

    The fork is **not** among them. POLO keeps every upstream task, so training a
    pose model in a POLO environment works, and refusing it would refuse a
    configuration that runs.

    An empty *base_weights* means the run starts from an asset name the
    environment resolves itself, so the probe loaded nothing and there is nothing
    to refuse about it -- only the environment is checked.
    """
    if not probe.has_ultralytics:
        raise UltralyticsNotFoundError(
            "the Ultralytics environment resolved but holds no ultralytics. "
            f"{ULTRALYTICS_BOOTSTRAP}"
        )
    if not base_weights:
        return
    refuse_unloadable_model(probe, base_weights)
    if probe.model_task != "pose":
        hint = (
            "POLO point training has its own op: mosaic run --kind train-points."
            if probe.model_task == "locate"
            else "'mosaic run --kind train-pose' produces the weights this op "
            "fine-tunes."
        )
        raise UnsupportedModelError(
            f"{base_weights} is a {probe.model_task!r} model and 'train-pose' "
            f"fine-tunes pose models. {hint}"
        )


def require_points_training_env(probe: ProbeResponse, base_weights: str) -> None:
    """Refuse, by name, anything ``train-points`` needs and the environment lacks.

    ``has_locate`` is checked **before** the weights, and that order is the whole
    value of the check: the fork and upstream ship one distribution name and the
    same console scripts, so the ``$PATH`` rung of the location ladder cannot tell
    which it found. This is what replaces the in-process import of the fork's own
    ``LocalizationModel`` -- the only thing that ever distinguished them, and one
    that could only be asked by importing an AGPL-licensed library into mosaic.
    """
    if not probe.has_ultralytics:
        raise PoloNotFoundError(
            f"the POLO environment resolved but holds no ultralytics. {POLO_BOOTSTRAP}"
        )
    if not probe.has_locate:
        raise PoloNotFoundError(
            "the environment reached for point training is upstream Ultralytics "
            f"({probe.ultralytics_version}), not the POLO fork: it defines no "
            "'locate' task. The two ship under one distribution name with the "
            "same console scripts, so a 'yolo' found on $PATH is as likely to be "
            f"either -- name the fork's environment explicitly. {POLO_BOOTSTRAP}"
        )
    if not base_weights:
        return
    refuse_unloadable_model(probe, base_weights)
    if probe.model_task != "locate":
        raise UnsupportedModelError(
            f"{base_weights} is a {probe.model_task!r} model and 'train-points' "
            "fine-tunes point-detection models. 'mosaic run --kind train-points' "
            "produces the weights this op fine-tunes; a pose model is "
            "'mosaic run --kind train-pose'."
        )


def _run_training(
    request: TrainRequestBase,
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
) -> TrainingOutcome:
    """Train one model through *env*, and read back how it ended.

    *work_dir* is this attempt's exchange directory, and it is created empty here
    rather than reused: the request, the response and the cancel sentinel are all
    byproducts of one attempt, and a leftover from the previous one would be read
    as this one's.

    **The caller's** ``cancel_check`` **decides whether a cancel is polite.**
    Handed the job's raw token, the supervisor kills the process group and the
    epoch in flight is lost; handed
    :func:`~mosaic.tracking.common.cooperative_cancel.stop_then_kill`, it writes
    the sentinel this request names and only escalates after a grace. This
    function does not choose -- it passes through what it is given -- but there is
    only one right answer for training, and the op is where it is made.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / TRAIN_REQUEST_NAME
    response_path = work_dir / TRAIN_RESPONSE_NAME
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
    response = TrainResponse.model_validate_json(response_path.read_text())

    expected = Path(request.project_dir) / request.run_name
    reported = Path(response.save_dir)
    if reported != expected:
        raise TrainingDirectoryError(
            f"{env.tool} wrote its model to {reported}, and mosaic reads "
            f"best.pt and results.csv from {expected}. Ultralytics renames a run "
            "directory that is already occupied, so continuing here would "
            "register some other attempt's weights under this run's identifier."
        )
    return TrainingOutcome(
        save_dir=reported,
        epochs_completed=response.epochs_completed,
        stop=response.stop,
    )


def run_pose_training_tool(
    request: TrainPoseRequest,
    *,
    work_dir: Path,
    idle_timeout: float,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> TrainingOutcome:
    """Train one YOLO pose model in the Ultralytics environment."""
    return _run_training(
        request,
        "train-pose",
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


def run_point_training_tool(
    request: TrainPointsRequest,
    *,
    work_dir: Path,
    idle_timeout: float,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> TrainingOutcome:
    """Train one POLO point model in the POLO environment."""
    return _run_training(
        request,
        "train-points",
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
    "CANCEL_SENTINEL_NAME",
    "EXCHANGE_DIRECTORY",
    "TRAIN_REQUEST_NAME",
    "TRAIN_RESPONSE_NAME",
    "TrainingDirectoryError",
    "TrainingOutcome",
    "attempt_directory",
    "require_points_training_env",
    "require_pose_training_env",
    "run_point_training_tool",
    "run_pose_training_tool",
]
