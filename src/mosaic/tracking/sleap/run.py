"""Run SLEAP from the command line for inference + identity tracking.

This module provides Python wrappers around the ``sleap-track`` and
``sleap-convert`` CLI console scripts, enabling headless batch pose inference +
tracking of animal videos and export of the analysis HDF5 that mosaic bridges
into standardized tracks.

Requires:
    The ``sleap-track`` / ``sleap-convert`` console scripts. SLEAP 1.6 is heavy
    (PyTorch + Qt), so it usually lives in its **own** environment rather than
    the mosaic env. Point the wrappers at it one of three ways (highest
    precedence first), via per-call args or env vars:

    * ``sleap_conda_env=`` / ``MOSAIC_SLEAP_CONDA_ENV`` -- run via
      ``conda run -n <env> sleap-track``;
    * ``sleap_bin=`` / ``MOSAIC_SLEAP_BIN`` -- a path to one SLEAP console
      script (its siblings are resolved in the same directory);
    * otherwise ``sleap-track`` / ``sleap-convert`` are looked up on ``$PATH``
      (the ``uv tool install sleap`` case).

    Unlike TRex, SLEAP inference is headless and needs no ``DISPLAY`` / ``Xvfb``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.tracking.common.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)

logger = logging.getLogger(__name__)

_SLEAP_TRACK: str = "sleap-track"
_SLEAP_CONVERT: str = "sleap-convert"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SleapNotFoundError(ToolNotFoundError):
    """Raised when a SLEAP console script (or ``conda``) cannot be located."""

    default_message = (
        "The 'sleap-track' console script was not found on $PATH.  "
        "Install SLEAP (e.g. 'uv tool install sleap[nn]') and ensure it "
        "is accessible, or point MOSAIC_SLEAP_CONDA_ENV / MOSAIC_SLEAP_BIN "
        "at it.  See https://sleap.ai for installation instructions."
    )


class SleapError(ToolExitError):
    """Raised when a SLEAP subprocess exits with a non-zero return code."""

    tool_name = "SLEAP"


# SLEAP's console scripts are always installed together, so MOSAIC_SLEAP_BIN
# pointing at any one of them names the directory the others live in. Plainly
# named because the training and label-export modules resolve through it too:
# one environment serves sleap-track, sleap-convert and sleap-nn-train alike,
# and declaring it twice is how the three drift apart.
SLEAP_ENV: Final = ToolEnv(
    tool="SLEAP",
    conda_env_var="MOSAIC_SLEAP_CONDA_ENV",
    bin_var="MOSAIC_SLEAP_BIN",
    bin_mode="sibling",
    not_found=SleapNotFoundError,
)


# The same environment addressed as an interpreter, for work that needs a SLEAP
# library rather than one of its verbs. ``locator`` is what makes that possible:
# looking up ``python`` on PATH would find the caller's, so the ladder looks up
# a script it knows SLEAP installs and takes the interpreter beside it.
SLEAP_PYTHON_ENV: Final = ToolEnv(
    tool="SLEAP",
    conda_env_var="MOSAIC_SLEAP_CONDA_ENV",
    bin_var="MOSAIC_SLEAP_BIN",
    bin_mode="sibling",
    not_found=SleapNotFoundError,
    locator=_SLEAP_TRACK,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SleapTrackResult:
    """Result of a SLEAP inference + tracking run."""

    slp_path: Path
    stdout: str
    stderr: str


@dataclass
class SleapConvertResult:
    """Result of a SLEAP ``.slp`` -> analysis HDF5 export."""

    analysis_h5_path: Path
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Invocation resolution
# ---------------------------------------------------------------------------


def _sleap_invocation(
    script: str,
    *,
    sleap_conda_env: str | None = None,
    sleap_bin: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch a SLEAP console *script*, as an argv prefix.

    The shared five-step ladder (:func:`tool_invocation`) applied to
    :data:`SLEAP_ENV`, with the script as the executable -- so one SLEAP
    environment serves both ``sleap-track`` and ``sleap-convert``.
    """
    return tool_invocation(
        SLEAP_ENV,
        executable=script,
        conda_env=sleap_conda_env,
        bin_path=sleap_bin,
    )


def _flatten_extra(extra: Mapping[str, object] | None) -> list[str]:
    """Flatten extra CLI settings into ``--key value`` pairs.

    Booleans become bare ``--key`` flags when True (omitted when False); ``None``
    values are skipped. Everything else is stringified.
    """
    args: list[str] = []
    if not extra:
        return args
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args.extend([f"--{key}", str(value)])
    return args


def _run_sleap(
    invocation: list[str],
    args: list[str],
    *,
    idle_timeout: float,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Execute a SLEAP console script with *args* and return (stdout, stderr).

    *invocation* is the argv prefix from :func:`_sleap_invocation`. *cancel_check*,
    when supplied, is polled while SLEAP runs; if it fires, the whole process
    group is killed and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    The subprocess always runs in its own process group (killable, orphan-safe)
    via :func:`run_supervised`.

    Raises :class:`SleapError` on non-zero exit.
    """
    cmd = [*invocation, *args]
    logger.info("Running: %s", " ".join(cmd))

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
    )

    if returncode != 0:
        raise SleapError(cmd, returncode, stdout, stderr)

    return stdout, stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sleap_track(
    video_path: Path | str,
    output_slp: Path | str,
    *,
    model_paths: Sequence[Path | str],
    tracking: bool = True,
    tracker: str = "flow",
    similarity: str = "instance",
    match: str = "hungarian",
    track_window: int = 5,
    max_instances: int | None = None,
    max_tracking: int | None = None,
    peak_threshold: float = 0.2,
    batch_size: int = 4,
    frames: str | None = None,
    device: str | None = None,
    extra_settings: Mapping[str, object] | None = None,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: Path | str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> SleapTrackResult:
    """Run SLEAP inference + tracking on a video, writing a ``.slp`` file.

    Invokes the classic-dialect ``sleap-track`` console script against one or
    more trained model directories (two for top-down: centroid + centered
    instance), assigning ``Track`` identities across frames when *tracking*.

    Parameters
    ----------
    video_path : path
        Input video file (or a ``.slp`` for re-tracking existing predictions).
    output_slp : path
        Destination ``.slp`` predictions file (passed as ``-o``).
    model_paths : sequence of paths
        One trained SLEAP model directory, or two for a top-down model
        (centroid, then centered-instance), passed as repeated ``-m`` flags.
    tracking : bool
        Assign cross-frame identities. When False, no tracker is attached.
    tracker : str
        Tracker flavor: ``simple`` / ``flow`` / ``simplemaxtracks`` /
        ``flowmaxtracks``.
    similarity : str
        ``--tracking.similarity`` (e.g. ``instance`` / ``centroid`` / ``iou``).
    match : str
        ``--tracking.match`` (``hungarian`` / ``greedy``).
    track_window : int
        ``--tracking.track_window`` (candidate window in frames).
    max_instances : int, optional
        ``-n`` cap on instances per frame.
    max_tracking : int, optional
        Cap on the number of maintained tracks (``--tracking.max_tracking 1`` +
        ``--tracking.max_tracks``); requires a ``*maxtracks`` tracker.
    peak_threshold : float
        Minimum confidence for a detected peak.
    batch_size : int
        Inference batch size (throughput only).
    frames : str, optional
        Frame selection, e.g. ``"0-1000"`` or ``"1,2,3"``.
    device : str, optional
        ``"cpu"`` -> ``--cpu``; a GPU index string -> ``--gpu <i>``; ``None`` /
        ``"auto"`` lets SLEAP choose.
    extra_settings : mapping, optional
        Additional ``sleap-track`` flags passed as ``--key value`` pairs.
    idle_timeout : float
        Kill the subprocess after this many seconds with no output (inactivity
        watchdog; default 900).
    max_runtime : float, optional
        Optional absolute wall-clock ceiling.
    sleap_conda_env : str, optional
        Run SLEAP inside this conda env (overrides ``MOSAIC_SLEAP_CONDA_ENV``).
    sleap_bin : path, optional
        Path to a SLEAP console script (overrides ``MOSAIC_SLEAP_BIN``).

    Returns
    -------
    SleapTrackResult
        The path to the written ``.slp`` file.

    Raises
    ------
    SleapNotFoundError
        If ``sleap-track`` cannot be located.
    SleapError
        If SLEAP exits with a non-zero return code.
    FileNotFoundError
        If the expected ``.slp`` output is not found after inference.
    ValueError
        If *model_paths* is empty.
    """
    video_path = Path(video_path)
    output_slp = Path(output_slp)
    output_slp.parent.mkdir(parents=True, exist_ok=True)
    models = [Path(m) for m in model_paths]
    if not models:
        raise ValueError("run_sleap_track requires at least one model directory")

    args: list[str] = [str(video_path), "-o", str(output_slp)]
    for model in models:
        args.extend(["-m", str(model)])
    args.extend(["--peak_threshold", str(peak_threshold)])
    args.extend(["--batch_size", str(batch_size)])
    if max_instances is not None:
        args.extend(["-n", str(max_instances)])
    if frames is not None:
        args.extend(["--frames", frames])
    if device == "cpu":
        args.append("--cpu")
    elif device is not None and device not in ("auto", "cuda"):
        args.extend(["--gpu", device])
    if tracking:
        args.extend(["--tracking.tracker", tracker])
        args.extend(["--tracking.similarity", similarity])
        args.extend(["--tracking.match", match])
        args.extend(["--tracking.track_window", str(track_window)])
        if max_tracking is not None:
            args.extend(["--tracking.max_tracking", "1"])
            args.extend(["--tracking.max_tracks", str(max_tracking)])
    args.extend(_flatten_extra(extra_settings))

    stdout, stderr = _run_sleap(
        _sleap_invocation(
            _SLEAP_TRACK, sleap_conda_env=sleap_conda_env, sleap_bin=sleap_bin
        ),
        args,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        cancel_check=cancel_check,
        on_output=on_output,
    )

    if not output_slp.exists():
        raise missing_output_error("SLEAP", output_slp, stdout, stderr)

    return SleapTrackResult(slp_path=output_slp, stdout=stdout, stderr=stderr)


def run_sleap_convert(
    slp_path: Path | str,
    output_h5: Path | str,
    *,
    idle_timeout: float = 300,
    max_runtime: float | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: Path | str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> SleapConvertResult:
    """Export a SLEAP ``.slp`` file to its analysis HDF5 form.

    Invokes ``sleap-convert --format analysis`` (the ``matlab``-layout export);
    the resulting HDF5 carries per-dataset ``dims`` attributes that the mosaic
    converter uses to normalize the layout without any SLEAP dependency.

    Parameters
    ----------
    slp_path : path
        Input ``.slp`` predictions file.
    output_h5 : path
        Destination analysis ``.h5`` (passed as ``-o``).
    idle_timeout : float
        Inactivity watchdog (default 300; the export is cheap).

    Returns
    -------
    SleapConvertResult
        The path to the written analysis HDF5.

    Raises
    ------
    SleapNotFoundError
        If ``sleap-convert`` cannot be located.
    SleapError
        If SLEAP exits with a non-zero return code.
    FileNotFoundError
        If no analysis HDF5 is found after conversion.
    """
    slp_path = Path(slp_path)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    args = [str(slp_path), "--format", "analysis", "-o", str(output_h5)]
    stdout, stderr = _run_sleap(
        _sleap_invocation(
            _SLEAP_CONVERT, sleap_conda_env=sleap_conda_env, sleap_bin=sleap_bin
        ),
        args,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        cancel_check=cancel_check,
        on_output=on_output,
    )

    produced = output_h5
    if not produced.exists():
        # sleap-convert can name the output per video (<slp>.<idx>_<stem>.h5);
        # fall back to whatever analysis HDF5 landed beside the requested path.
        candidates = sorted(output_h5.parent.glob("*.analysis.h5"))
        if not candidates:
            candidates = sorted(output_h5.parent.glob("*.h5"))
        if not candidates:
            raise missing_output_error("sleap-convert", output_h5, stdout, stderr)
        produced = candidates[0]

    return SleapConvertResult(analysis_h5_path=produced, stdout=stdout, stderr=stderr)
