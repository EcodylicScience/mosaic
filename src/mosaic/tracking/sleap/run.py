"""Run SLEAP from the command line for inference + identity tracking.

This module wraps two console scripts: ``sleap-nn track``, for headless batch
pose inference + tracking of animal videos, and ``sleap-convert``, for the
analysis HDF5 export that mosaic bridges into standardized tracks.

**Inference is sleap-nn's own CLI, not a wrapper around it.** The ``sleap``
distribution ships two inference scripts of its own, ``sleap-track`` and
``sleap-nn-track``, and both import ``sleap_nn.predict``, which sleap-nn removed
after 0.3.1. ``sleap`` 1.6.4 therefore cannot infer on the sleap-nn 0.3.3 its own
dependency range allows, and reports the ``ImportError`` as "sleap-nn is not
installed". ``sleap-nn`` (``sleap_nn.cli:cli``) ships with the modules it
imports, so the two cannot skew. ``sleap-nn-track`` is the near-miss: it looks
like that CLI and is the ``sleap`` wrapper. ``sleap-convert`` imports no
``sleap_nn`` module and is unaffected.

Requires:
    SLEAP 1.6 with its ``nn`` extra. It is heavy (PyTorch + Qt), so it usually
    lives in its **own** environment rather than the mosaic env. Point the
    wrappers at it one of three ways (highest precedence first), via per-call
    args or env vars:

    * ``sleap_conda_env=`` / ``MOSAIC_SLEAP_CONDA_ENV`` -- run via
      ``conda run -n <env> sleap-nn``;
    * ``sleap_bin=`` / ``MOSAIC_SLEAP_BIN`` -- a path to one SLEAP console
      script (its siblings are resolved in the same directory);
    * otherwise ``sleap-convert`` is looked up on ``$PATH`` and every script
      runs from the environment it belongs to. That is the ``uv tool install
      sleap[nn]`` case, which links ``sleap-convert`` into ``~/.local/bin`` but
      not ``sleap-nn``, a script of the ``sleap-nn`` distribution.

    Unlike TRex, SLEAP inference is headless and needs no ``DISPLAY`` / ``Xvfb``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

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

# sleap-nn's own CLI, shipped by the sleap-nn distribution. It dispatches on a
# subcommand, so the verb is argv[1] rather than part of the script name.
_SLEAP_TRACK_SCRIPT: Final = "sleap-nn"
_SLEAP_TRACK_SUBCOMMAND: Final = "track"
_SLEAP_CONVERT: str = "sleap-convert"

SleapCandidatesMethod = Literal["fixed_window", "local_queues"]
"""Where ``sleap-nn track`` draws match candidates from: every instance of the
last few frames, or the last few instances of each track."""

SleapFeatures = Literal["keypoints", "centroids", "bboxes", "image"]
"""What ``sleap-nn track`` compares a detection and a candidate by."""

SleapScoringMethod = Literal["oks", "cosine_sim", "iou", "euclidean_dist"]
"""How ``sleap-nn track`` scores the association between those features."""

SleapMatchingMethod = Literal["hungarian", "greedy"]
"""How ``sleap-nn track`` assigns detections to tracks from those scores."""

# The device families sleap-nn names as they are. A CUDA index is the other
# spelling mosaic accepts, and is translated.
_DEVICE_FAMILIES: Final = frozenset({"cpu", "cuda", "mps"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SleapNotFoundError(ToolNotFoundError):
    """Raised when a SLEAP console script (or ``conda``) cannot be located."""

    default_message = (
        "No SLEAP install was found: mosaic looks up 'sleap-convert' on $PATH "
        "and runs 'sleap-nn' from the same environment. Install SLEAP with its "
        "nn extra (e.g. 'uv tool install \"sleap[nn]\"'), or point "
        "MOSAIC_SLEAP_CONDA_ENV / MOSAIC_SLEAP_BIN at an environment that has "
        "it. See https://sleap.ai for installation instructions."
    )


class SleapError(ToolExitError):
    """Raised when a SLEAP subprocess exits with a non-zero return code."""

    tool_name = "SLEAP"


# SLEAP's console scripts are always installed together, so MOSAIC_SLEAP_BIN
# pointing at any one of them names the directory the others live in. Plainly
# named because the training, probe and label-export modules resolve through it
# too: one environment serves sleap-nn, sleap-convert, sleap-nn-train and its
# own python alike, and declaring it twice is how they drift apart.
#
# On $PATH the environment is found through sleap-convert rather than through
# each executable, because a `uv tool install "sleap[nn]"` links only the sleap
# distribution's scripts: sleap-nn and python are in the environment and not on
# $PATH. Looking every executable up beside one locator is also what keeps
# inference and export from answering from two different installs.
SLEAP_ENV: Final = ToolEnv(
    tool="SLEAP",
    conda_env_var="MOSAIC_SLEAP_CONDA_ENV",
    bin_var="MOSAIC_SLEAP_BIN",
    bin_mode="sibling",
    not_found=SleapNotFoundError,
    locator=_SLEAP_CONVERT,
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
    environment serves both ``sleap-nn`` and ``sleap-convert``.
    """
    return tool_invocation(
        SLEAP_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
        executable=script,
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
    on_activity: Callable[[str], None] | None = None,
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
        on_activity=on_activity,
    )

    if returncode != 0:
        raise SleapError(cmd, returncode, stdout, stderr)

    return stdout, stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sleap_track_device_args(device: str | None) -> list[str]:
    """The ``-d`` argument that puts ``sleap-nn track`` on *device*.

    sleap-nn takes one torch device string, where the rest of mosaic spells a
    device as a family name or a bare CUDA index. Sent through as it is, ``"0"``
    names no device sleap-nn knows, so an index is sent as ``cuda:<index>``.

    Args:
        device: ``None``, empty or ``auto`` to leave the choice to sleap-nn;
            ``cpu``, ``cuda`` or ``mps`` to name a device family, which then
            fails loudly where it is absent; a CUDA index such as ``0``; or
            ``cuda:<index>``.

    Returns:
        The argv tokens, empty when *device* leaves the choice open.

    Raises:
        ValueError: *device* is none of those. Raised from a field validator on
            :class:`~mosaic.tracking.sleap.params.SleapParams` too, so an
            unusable value is refused when the run is submitted rather than on a
            GPU node once it is scheduled.
    """
    if device is None or device in ("", "auto"):
        return []
    if device in _DEVICE_FAMILIES:
        return ["-d", device]
    index = device.removeprefix("cuda:")
    if index.isascii() and index.isdigit():
        return ["-d", f"cuda:{index}"]
    families = ", ".join(sorted(_DEVICE_FAMILIES))
    msg = (
        f"unusable device {device!r}: give 'auto', one of {families}, a CUDA "
        "index such as '0', or 'cuda:<index>'. sleap-nn track runs on one "
        "device, so a list of them is refused."
    )
    raise ValueError(msg)


def run_sleap_track(
    video_path: Path | str,
    output_slp: Path | str,
    *,
    model_paths: Sequence[Path | str],
    tracking: bool = True,
    use_flow: bool = True,
    candidates_method: SleapCandidatesMethod = "fixed_window",
    features: SleapFeatures = "keypoints",
    scoring_method: SleapScoringMethod = "oks",
    track_matching_method: SleapMatchingMethod = "hungarian",
    tracking_window_size: int = 5,
    max_tracks: int | None = None,
    max_instances: int | None = None,
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
    on_activity: Callable[[str], None] | None = None,
) -> SleapTrackResult:
    """Run SLEAP inference + tracking on a video, writing a ``.slp`` file.

    Invokes ``sleap-nn track`` against one or more trained model directories
    (two for top-down: centroid + centered instance), assigning ``Track``
    identities across frames when *tracking*.

    Every tracking option is sent explicitly rather than left to sleap-nn's
    defaults. Its ``track`` and ``predict`` subcommands already disagree on
    ``--candidates_method``, and a default that moves under an unchanged
    ``run_id`` is a differently configured tracker that still produces
    plausible tracks.

    Parameters
    ----------
    video_path : path
        Input video file (or a ``.slp`` for re-tracking existing predictions),
        passed as ``-i``.
    output_slp : path
        Destination ``.slp`` predictions file (passed as ``-o``).
    model_paths : sequence of paths
        One trained SLEAP model directory, or two for a top-down model
        (centroid, then centered-instance), passed as repeated ``-m`` flags in
        order.
    tracking : bool
        Assign cross-frame identities (``-t``). When False, no tracking option
        is sent at all.
    use_flow : bool
        ``--use_flow``: shift candidate poses by optical flow before matching.
    candidates_method : str
        ``--candidates_method``: ``fixed_window`` or ``local_queues``.
    features : str
        ``--features``: what a detection and a candidate are compared by.
    scoring_method : str
        ``--scoring_method``: how that comparison is scored.
    track_matching_method : str
        ``--track_matching_method``: ``hungarian`` or ``greedy``.
    tracking_window_size : int
        ``--tracking_window_size``: how many frames, or instances per track,
        are kept as candidates.
    max_tracks : int, optional
        ``--max_tracks``: cap on the number of tracks. sleap-nn applies it only
        with ``local_queues`` candidates.
    max_instances : int, optional
        ``-n`` cap on instances per frame.
    peak_threshold : float
        Minimum confidence for a detected peak.
    batch_size : int
        Inference batch size (throughput only).
    frames : str, optional
        Frame selection, e.g. ``"0-1000"`` or ``"1,2,3"``.
    device : str, optional
        Translated by :func:`sleap_track_device_args`; ``None`` / ``"auto"``
        lets sleap-nn choose.
    extra_settings : mapping, optional
        Additional ``sleap-nn track`` options passed as ``--key value`` pairs.
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
        If no SLEAP install can be located.
    SleapError
        If SLEAP exits with a non-zero return code.
    FileNotFoundError
        If the expected ``.slp`` output is not found after inference.
    ValueError
        If *model_paths* is empty, or *device* is unusable.
    """
    video_path = Path(video_path)
    output_slp = Path(output_slp)
    output_slp.parent.mkdir(parents=True, exist_ok=True)
    models = [Path(m) for m in model_paths]
    if not models:
        raise ValueError("run_sleap_track requires at least one model directory")

    # The verb first, because this CLI dispatches on a subcommand, and the video
    # by -i: a positional path is an unexpected argument to it.
    args: list[str] = [
        _SLEAP_TRACK_SUBCOMMAND,
        "-i",
        str(video_path),
        "-o",
        str(output_slp),
    ]
    for model in models:
        args.extend(["-m", str(model)])
    args.extend(["--peak_threshold", str(peak_threshold)])
    args.extend(["--batch_size", str(batch_size)])
    if max_instances is not None:
        args.extend(["-n", str(max_instances)])
    if frames is not None:
        args.extend(["--frames", frames])
    args.extend(sleap_track_device_args(device))
    if tracking:
        args.append("-t")
        if use_flow:
            args.append("--use_flow")
        args.extend(["--candidates_method", candidates_method])
        args.extend(["--features", features])
        args.extend(["--scoring_method", scoring_method])
        args.extend(["--track_matching_method", track_matching_method])
        args.extend(["--tracking_window_size", str(tracking_window_size)])
        if max_tracks is not None:
            args.extend(["--max_tracks", str(max_tracks)])
    args.extend(_flatten_extra(extra_settings))

    stdout, stderr = _run_sleap(
        _sleap_invocation(
            _SLEAP_TRACK_SCRIPT, sleap_conda_env=sleap_conda_env, sleap_bin=sleap_bin
        ),
        args,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        cancel_check=cancel_check,
        on_output=on_output,
        on_activity=on_activity,
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
    on_activity: Callable[[str], None] | None = None,
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
        on_activity=on_activity,
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
