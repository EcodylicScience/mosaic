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

This module is the tracker's own half of that exchange: the preflight refusals,
the merged tracker configuration, and the calls that write a request and read a
response back. Locating either environment and launching the runner in it lives
in :mod:`mosaic.tracking.common.ultralytics_env`, because pose and point
inference need the same launcher. :func:`probe_ultralytics` and :func:`run_ultralytics_tool` sit at
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
from typing import Final

from mosaic.core.pipeline._utils import atomic_write
from mosaic.tracking.common.toolenv import missing_output_error
from mosaic.tracking.common.ultralytics_env import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    ULTRALYTICS_BOOTSTRAP,
    ULTRALYTICS_ENV,
    UltralyticsError,
    UltralyticsNotFoundError,
    refuse_unloadable_model,
    run_runner,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ModelTask,
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

# `ModelTask` is re-exported rather than restated. What mosaic bridges and what
# the wire carries are the same closed set -- a segment mask and a rotated box
# have no `trex_v1` mapping, POLO's point detection has its own op, and
# Ultralytics refuses to track a classifier -- and the protocol module is where
# it is declared, because that module may take no import from mosaic while
# mosaic may import it freely.
_SUPPORTED_TASKS: Final[tuple[ModelTask, ...]] = ("pose", "detect")

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


class UnsupportedTaskError(ValueError):
    """The model is not one mosaic can bridge into a ``trex_v1`` table."""


class UnsupportedTrackerError(ValueError):
    """The installed Ultralytics does not know the requested backend."""


@dataclass(frozen=True, slots=True)
class UltralyticsTrackResult:
    """What one entry's tracking produced."""

    predictions_path: Path
    n_frames: int
    n_ids: int


# --- preflight -------------------------------------------------------------


def require_ultralytics(
    probe: ProbeResponse, tracker: TrackerName, model_path: str
) -> None:
    """Refuse, by name, anything this run needs and the environment lacks.

    Decided from what the probe reported rather than by importing anything, and
    run before a video is opened, so a missing dependency, an unsupported backend
    or a checkpoint this build cannot load is a message rather than a traceback
    from inside a callback on frame zero.

    The weights are refused **before** ``lap`` and the backend, and before
    :func:`require_supported_task` reads the task: a checkpoint that did not load
    reports no task at all, so asking about one would answer a question the probe
    never got to.
    """
    if not probe.has_ultralytics:
        raise UltralyticsNotFoundError(
            "the Ultralytics environment resolved but holds no ultralytics. "
            f"{ULTRALYTICS_BOOTSTRAP}"
        )

    refuse_unloadable_model(probe, model_path)

    if not probe.has_lap:
        raise UltralyticsNotFoundError(
            "lap is required for Ultralytics multi-object tracking -- it is the "
            "linear-assignment solver every backend associates with. It appears "
            "in no ultralytics extra, so without it ultralytics tries to "
            "pip-install it mid-run. The environment's pyproject.toml declares "
            f"it, so rebuilding is the fix. {ULTRALYTICS_BOOTSTRAP}"
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
        stdout, stderr = run_runner(
            ULTRALYTICS_ENV,
            UltralyticsError,
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
        stdout, stderr = run_runner(
            ULTRALYTICS_ENV,
            UltralyticsError,
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

    **The caller's** ``idle_timeout`` **must exceed a cold model load.** It is
    passed through untouched, unlike the floor
    :data:`PROBE_DEADLINE_FLOOR_SECONDS` puts under the two silent subcommands,
    because raising it here would blunt the one bound that supervises tracking:
    the runner prints a line per decoded batch, and a value chosen so a wedged
    tracker dies in two minutes is the whole point of the knob. What that leaves
    is one silent stretch inside a healthy run. The runner announces itself as
    soon as Ultralytics is imported, so the import is covered; loading the
    weights runs between that line and the first batch, and it is the longest
    silence a run contains -- a cold checkpoint off a network mount can take
    minutes. A bound shorter than that kills a working run and reports it as a
    tool that produced no output.

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

    stdout, stderr = run_runner(
        ULTRALYTICS_ENV,
        UltralyticsError,
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
