"""Locating the Ultralytics environment, and launching the runner program in it.

Ultralytics is AGPL-3.0, so mosaic never imports it. The imports live in
:mod:`mosaic.tracking.external.runner.ultralytics_runner`, a program that runs
in an environment the user builds, and what crosses between the two is a JSON
request file, a JSON response file and progress lines on standard output.

**Mosaic hands the tool a path**, exactly as TREx, SLEAP and Lightning Pose do,
and :func:`~mosaic.tracking.common.tool_input.resolve_tool_input` is the boundary
that keeps an imgstore recording working: a tool that opens a path cannot read a
directory of chunk files, so a store resolves to the plain video
``export-store`` wrote for it.

This module is where the tool is located and launched -- one :class:`ToolEnv`,
the argv its ladder resolves, and the calls that write a request and read a
response back. :func:`probe_ultralytics` and :func:`run_ultralytics_tool` sit at
module scope because they are the seam the marker suite replaces, which is what
exercises the whole run protocol -- identifiers, markers, reuse, the bridge --
with no Ultralytics installed at all. :func:`ultralytics_tracker_defaults` is the
third and is called by no run: it reads back what the environment's backends
ship, so mosaic's transcribed tables can be diffed against the release that will
run them.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.tracking.common.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ProbeRequest,
    ProbeResponse,
    TrackerDefaultsRequest,
    TrackerDefaultsResponse,
    TrackRequest,
    TrackResponse,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TrackerName,
    TrackerSetting,
)

logger = logging.getLogger(__name__)

ModelTask = Literal["pose", "detect"]
"""What mosaic bridges. A segment mask and a rotated box have no ``trex_v1``
mapping, POLO's point detection has its own op, and Ultralytics refuses to track
a classifier.
"""

_SUPPORTED_TASKS: Final[tuple[ModelTask, ...]] = ("pose", "detect")

_PYTHON: Final = "python"
_YOLO_SCRIPT: Final = "yolo"
_RUNNER_SCRIPT: Final = "ultralytics_runner.py"

TRACK_REQUEST_NAME: Final = "track-request.json"
"""What one entry's request to the tool is called, inside its working directory."""

TRACK_RESPONSE_NAME: Final = "track-response.json"
"""What the tool's reply is called, beside the request.

Both are byproducts of an attempt rather than results of one, so a re-run of the
``track`` phase must delete them: a stale request beside fresh output is exactly
what the phase's clear globs exist to prevent. Those globs live in
:data:`~mosaic.core.pipeline.tracking_roots.TRACKING_ROOTS`, which is in ``core``
and so cannot import this module -- the two spell the same strings
independently, and ``test_ultralytics_run_markers.py`` is what holds them
together.
"""

PROBE_DEADLINE_FLOOR_SECONDS: Final = 900.0
"""The least time a silent subcommand gets to answer, whatever the tracking bound is.

``idle_timeout`` bounds *silence*, which is the right unit for tracking: the
runner prints a line per decoded batch, so a quiet stretch means hung. A probe
prints nothing at all between spawn and answer, so the same number is a deadline
on a cold torch import and a checkpoint load off a network mount -- work
proceeding exactly as intended. A user who shortens the tracking window so a hung
tracker dies quickly must not thereby put a stopwatch on loading a model, so the
probe gets the caller's value or this floor, whichever is longer.
``tracker-defaults`` is silent for the same stretch -- the torch import is most
of what it costs -- and takes the same floor.
"""

_ENV_BOOTSTRAP: Final = (
    "Build it with 'uv sync --python 3.12' in "
    "src/mosaic/tracking/external/ultralytics-env/, then point "
    "MOSAIC_ULTRALYTICS_CONDA_ENV at a conda environment holding it or "
    "MOSAIC_ULTRALYTICS_BIN at that environment's 'yolo' script. See "
    "src/mosaic/tracking/external/README.md."
)


class UltralyticsNotFoundError(ToolNotFoundError):
    """The Ultralytics environment, its Ultralytics, or ``conda``, is not there."""

    default_message = (
        "The Ultralytics environment was not found: no 'yolo' console script on "
        "$PATH, and neither MOSAIC_ULTRALYTICS_CONDA_ENV nor "
        f"MOSAIC_ULTRALYTICS_BIN names one. {_ENV_BOOTSTRAP}"
    )


class UnsupportedTaskError(ValueError):
    """The model is not one mosaic can bridge into a ``trex_v1`` table."""


class UnsupportedTrackerError(ValueError):
    """The installed Ultralytics does not know the requested backend."""


class UltralyticsError(ToolExitError):
    """The Ultralytics runner exited with a non-zero return code."""

    tool_name = "Ultralytics"
    # The argv is ``python <runner> <subcommand> --request <path> --out <path>``,
    # so four tokens reach the subcommand and echoing further prints paths.
    head = 4


@dataclass(frozen=True, slots=True)
class UltralyticsTrackResult:
    """What one entry's tracking produced."""

    predictions_path: Path
    n_frames: int
    n_ids: int


# The runner is driven through its environment's ``python`` rather than a console
# verb, so the ``yolo`` script is what locates that interpreter: a bare
# ``python`` on ``$PATH`` would be the caller's own.
ULTRALYTICS_ENV: Final = ToolEnv(
    tool="Ultralytics",
    conda_env_var="MOSAIC_ULTRALYTICS_CONDA_ENV",
    bin_var="MOSAIC_ULTRALYTICS_BIN",
    bin_mode="sibling",
    not_found=UltralyticsNotFoundError,
    locator=_YOLO_SCRIPT,
)


# --- preflight -------------------------------------------------------------


def require_ultralytics(probe: ProbeResponse, tracker: TrackerName) -> None:
    """Refuse, by name, anything this run needs and the environment lacks.

    Decided from what the probe reported rather than by importing anything, and
    run before a video is opened, so a missing dependency or an unsupported
    backend is a message rather than a traceback from inside a callback on frame
    zero.
    """
    if not probe.has_ultralytics:
        raise UltralyticsNotFoundError(
            "the Ultralytics environment resolved but holds no ultralytics. "
            f"{_ENV_BOOTSTRAP}"
        )

    if not probe.has_lap:
        raise UltralyticsNotFoundError(
            "lap is required for Ultralytics multi-object tracking -- it is the "
            "linear-assignment solver every backend associates with. It appears "
            "in no ultralytics extra, so without it ultralytics tries to "
            "pip-install it mid-run. The environment's pyproject.toml declares "
            f"it, so rebuilding is the fix. {_ENV_BOOTSTRAP}"
        )

    if tracker not in probe.tracker_names:
        raise UnsupportedTrackerError(
            f"the ultralytics in this environment ({probe.ultralytics_version}) "
            f"knows {sorted(probe.tracker_names)}, not {tracker!r}. The four "
            "newer backends arrived in 8.4.63, which is the floor "
            "src/mosaic/tracking/external/ultralytics-env/pyproject.toml declares."
        )


def require_supported_task(task: str) -> ModelTask:
    """Narrow a model's task to one mosaic bridges, or refuse it by name."""
    if task in _SUPPORTED_TASKS:
        return "pose" if task == "pose" else "detect"
    hint = {
        "locate": "POLO point detection has its own op: mosaic run --kind infer-points.",
        "segment": "mosaic bridges boxes and keypoints; a mask has no trex_v1 mapping.",
        "obb": "mosaic bridges axis-aligned boxes; a rotated box has no trex_v1 mapping.",
        "classify": "a classifier localizes nothing, so there is nothing to track.",
    }.get(task, "")
    raise UnsupportedTaskError(
        f"the 'ultralytics' tracker runs {' and '.join(_SUPPORTED_TASKS)} models, "
        f"not {task!r}. {hint}".strip()
    )


# --- the tracker configuration file ----------------------------------------


def effective_tracker_table(
    installed: Mapping[str, TrackerSetting], resolved: Mapping[str, TrackerSetting]
) -> dict[str, TrackerSetting]:
    """The installed backend's own defaults, with mosaic's resolved values on top.

    Mosaic's table is what identity is taken over, but it must not be what gets
    *written*: each backend reads its settings off an object that raises on a
    missing attribute, so writing mosaic's table alone would fail inside
    Ultralytics -- after the model loaded and the video opened -- the first time
    an upstream release added a required setting. Merging leaves a setting mosaic
    has not transcribed at its upstream default, and the preflight drift test is
    what turns that into a decision rather than a surprise.

    *installed* arrives from the probe, which read it in the environment that
    will run: this process has no Ultralytics to read a shipped YAML from.
    """
    return {**installed, **resolved}


def write_tracker_yaml(path: Path, table: Mapping[str, TrackerSetting]) -> Path:
    """Write *table* where Ultralytics will read it, atomically.

    Four details are load-bearing. The suffix is ``.yaml`` and the path absolute,
    because Ultralytics' config check returns an existing path untouched and
    otherwise goes looking for -- or downloading -- a name. Keys are sorted, so
    the file is a function of its contents. And the write is atomic, because the
    check gates on existence alone: a truncated file would be *loaded*, its
    missing settings quietly filled from each backend's own fallbacks, and a
    differently-configured tracker would run under the right identifier.
    """
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)

    def write(temp: Path) -> None:
        _ = temp.write_text(yaml.safe_dump(dict(table), sort_keys=True))

    atomic_write(path, write)
    return path


# --- launching the runner --------------------------------------------------


def _runner_script() -> Path:
    """Where the program mosaic spawns lives, as an absolute path.

    Resolved from the package rather than joined out of strings, so a source
    checkout, an editable install and a wheel all name the same file. Importing
    the package is not importing the runner *module*: ``runner/__init__.py`` is
    a docstring, so nothing here pulls Ultralytics into this process, which is
    the whole point of the separation.
    """
    from mosaic.tracking.external import runner

    return Path(runner.__file__).parent / _RUNNER_SCRIPT


def _ultralytics_invocation(
    *,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch the Ultralytics environment's ``python``, as a prefix.

    The shared five-step ladder (:func:`tool_invocation`) applied to
    :data:`ULTRALYTICS_ENV`.
    """
    return tool_invocation(
        ULTRALYTICS_ENV,
        executable=_PYTHON,
        conda_env=conda_env,
        bin_path=bin_path,
    )


def _run_runner(
    subcommand: Literal["probe", "tracker-defaults", "track"],
    request_path: Path,
    response_path: Path,
    *,
    idle_timeout: float,
    max_runtime: float | None,
    conda_env: str | None,
    bin_path: str | Path | None,
    cancel_check: Callable[[], bool] | None,
    on_output: Callable[[str], None] | None,
) -> tuple[str, str]:
    """Run one runner subcommand and return (stdout, stderr).

    *cancel_check*, when supplied, is polled while the runner works; if it fires,
    the whole process group is killed and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    Raises:
        UltralyticsError: The runner exited with a non-zero return code.
    """
    cmd = [
        *_ultralytics_invocation(conda_env=conda_env, bin_path=bin_path),
        str(_runner_script()),
        subcommand,
        "--request",
        str(request_path),
        "--out",
        str(response_path),
    ]
    logger.info("Running: %s", " ".join(cmd[:4]) + (" ..." if len(cmd) > 4 else ""))

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
    )
    if returncode != 0:
        raise UltralyticsError(cmd, returncode, stdout, stderr)
    return stdout, stderr


def probe_ultralytics(
    model_path: Path | str,
    *,
    tracker: TrackerName,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> ProbeResponse:
    """Ask the Ultralytics environment what it holds, and what the weights are.

    Called once per run, before anything is minted, so the keypoint count, the
    backend's shipped settings, the model's declared task and the installed
    version are all known before the first entry runs.

    The request and the response live in a temporary directory: the probe runs
    before the run root exists, and nothing reads either file afterwards.

    A probe writes no progress lines -- it loads the weights and answers -- so an
    inactivity bound on it is a deadline on the whole operation. *idle_timeout*
    is therefore raised to :data:`PROBE_DEADLINE_FLOOR_SECONDS` when the caller's
    value is shorter. *max_runtime* is passed through untouched, being an
    absolute ceiling the caller asked for.

    Raises:
        UltralyticsError: The runner exited non-zero. A model it cannot load is
            the usual reason, and the captured streams say which.
        FileNotFoundError: The runner exited zero having written no response.
    """
    request = ProbeRequest(model_path=str(model_path), tracker=tracker)
    with tempfile.TemporaryDirectory(prefix="mosaic-ultralytics-probe-") as scratch:
        request_path = Path(scratch) / "probe-request.json"
        response_path = Path(scratch) / "probe-response.json"
        _ = request_path.write_text(request.model_dump_json())
        stdout, stderr = _run_runner(
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
            raise missing_output_error("Ultralytics", response_path, stdout, stderr)
        return ProbeResponse.model_validate_json(response_path.read_text())


def ultralytics_tracker_defaults(
    *,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> TrackerDefaultsResponse:
    """Read back every backend's shipped configuration table from the environment.

    Mosaic transcribes those tables rather than reading them, so that an upstream
    retune cannot silently re-mean a run identifier already on disk. This is how
    the transcription is compared against the release that will actually run it,
    which turns a moved default into a decision at upgrade time.

    Not called by a run. The request and the response live in a temporary
    directory for the same reason the probe's do: nothing reads either file
    afterwards.

    This subcommand prints nothing between spawn and answer, so *idle_timeout* is
    raised to :data:`PROBE_DEADLINE_FLOOR_SECONDS` when the caller's value is
    shorter, exactly as :func:`probe_ultralytics` does.

    Raises:
        UltralyticsError: The runner exited with a non-zero return code.
        FileNotFoundError: The runner exited zero having written no response.
    """
    request = TrackerDefaultsRequest()
    with tempfile.TemporaryDirectory(prefix="mosaic-ultralytics-defaults-") as scratch:
        request_path = Path(scratch) / "tracker-defaults-request.json"
        response_path = Path(scratch) / "tracker-defaults-response.json"
        _ = request_path.write_text(request.model_dump_json())
        stdout, stderr = _run_runner(
            "tracker-defaults",
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
            raise missing_output_error("Ultralytics", response_path, stdout, stderr)
        return TrackerDefaultsResponse.model_validate_json(response_path.read_text())


def run_ultralytics_tool(
    request: TrackRequest,
    *,
    work_dir: Path,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> UltralyticsTrackResult:
    """Track one video in the Ultralytics environment, and read back what it did.

    The request and the response are written into *work_dir*, beside the
    predictions parquet the runner publishes, so an attempt's whole exchange is
    on disk where the attempt is.

    Raises:
        UltralyticsError: The runner exited with a non-zero return code.
        FileNotFoundError: The runner exited zero having written no response, or
            no predictions table. The second is the sharper case: a run that
            reports success having written nothing would leave the reuse gate
            with no output to find, and the entry re-running forever.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / TRACK_REQUEST_NAME
    response_path = work_dir / TRACK_RESPONSE_NAME
    _ = request_path.write_text(request.model_dump_json())

    stdout, stderr = _run_runner(
        "track",
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
        raise missing_output_error("Ultralytics", response_path, stdout, stderr)
    predictions_path = Path(request.output_parquet)
    if not predictions_path.is_file():
        raise missing_output_error("Ultralytics", predictions_path, stdout, stderr)

    response = TrackResponse.model_validate_json(response_path.read_text())
    return UltralyticsTrackResult(
        predictions_path=predictions_path,
        n_frames=response.n_frames,
        n_ids=response.n_ids,
    )


__all__ = [
    "PROBE_DEADLINE_FLOOR_SECONDS",
    "TRACK_REQUEST_NAME",
    "TRACK_RESPONSE_NAME",
    "ULTRALYTICS_ENV",
    "ModelTask",
    "UltralyticsError",
    "UltralyticsNotFoundError",
    "UltralyticsTrackResult",
    "UnsupportedTaskError",
    "UnsupportedTrackerError",
    "effective_tracker_table",
    "probe_ultralytics",
    "require_supported_task",
    "require_ultralytics",
    "run_ultralytics_tool",
    "ultralytics_tracker_defaults",
    "write_tracker_yaml",
]
