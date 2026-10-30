"""Run SLEAP from the command line for inference + identity tracking.

This module wraps ``sleap-nn``, for headless batch pose inference + tracking of
animal videos, and the SLEAP environment's ``python``, which writes the analysis
HDF5 that mosaic bridges into standardized tracks with ``sleap-io``. Inference runs
``sleap-nn track``, or ``sleap-nn predict`` when the last model is an embedding
model, which tracks by appearance and saves each detection's vector. A snippet run
by the SLEAP interpreter reads the tracked instances of a predictions file and
their appearance vectors back (:func:`read_back_embeddings`).

**Inference is sleap-nn's own CLI, not a wrapper around it.** The ``sleap``
distribution ships two inference scripts of its own, ``sleap-track`` and
``sleap-nn-track``, and both import ``sleap_nn.predict``, which sleap-nn removed
after 0.3.1. ``sleap`` 1.6.4 therefore cannot infer on the sleap-nn 0.3.3 its own
dependency range allows, and reports the ``ImportError`` as "sleap-nn is not
installed". ``sleap-nn`` (``sleap_nn.cli:cli``) ships with the modules it
imports, so the two cannot skew. ``sleap-nn-track`` is the near-miss: it looks
like that CLI and is the ``sleap`` wrapper.

**The analysis export is sleap-io's, not sleap-convert's.** A snippet run by the
SLEAP interpreter makes the ``sleap_io.save_analysis_h5`` call that sleap 1.6.5's
``sleap-convert --format analysis`` makes (:func:`run_sleap_analysis_export`). An
environment with only sleap-nn and its sleap-io therefore exports as well.

Requires:
    sleap-nn and its sleap-io. They are heavy (PyTorch), so they usually live
    in their **own** environment rather than the mosaic env. Point the wrappers
    at it one of three ways (highest precedence first), via per-call args or
    env vars:

    * ``sleap_conda_env=`` / ``MOSAIC_SLEAP_CONDA_ENV`` -- run via
      ``conda run -n <env> sleap-nn``;
    * ``sleap_bin=`` / ``MOSAIC_SLEAP_BIN`` -- a path to one SLEAP console
      script (its siblings are resolved in the same directory);
    * otherwise the ``sleap`` distribution's ``sleap-convert`` is looked up on
      ``$PATH`` and every executable runs from the environment it belongs to.
      That is the ``uv tool install "sleap[nn]"`` case, which links
      ``sleap-convert`` into ``~/.local/bin`` but not ``sleap-nn``, a script of
      the ``sleap-nn`` distribution. mosaic runs no ``sleap-convert``; it only
      finds the environment through it.

    Unlike TRex, SLEAP inference is headless and needs no ``DISPLAY`` / ``Xvfb``.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from mosaic.core.track_library.sleap import read_appearance_readback

logger = logging.getLogger(__name__)

# sleap-nn's own CLI, shipped by the sleap-nn distribution. It dispatches on a
# subcommand, so the verb is argv[1] rather than part of the script name.
_SLEAP_TRACK_SCRIPT: Final = "sleap-nn"
_SLEAP_TRACK_SUBCOMMAND: Final = "track"
_SLEAP_PREDICT_SUBCOMMAND: Final = "predict"
_NO_DETECTIONS: Final = re.compile(
    r"No detections to embed|No detections found in .* to embed"
)
"""What ``sleap-nn predict`` 0.3.5 reports when a video has no instance to embed.

It exits with code 2 through ``click.UsageError``, with the first message when
the detection stage found none and the second when detections exist but none
receives a vector."""
SLEAP_DEFAULT_BATCH_SIZE: Final = 4
"""The detection batch ``sleap-nn`` uses when ``-b`` is not sent."""
_SLEAP_CONVERT: str = "sleap-convert"

SleapCandidatesMethod = Literal["fixed_window", "local_queues"]
"""Where the SLEAP tracker draws match candidates from: every instance of the
last few frames, or the last few instances of each track."""

SleapFeatures = Literal["keypoints", "centroids", "bboxes", "image", "embeddings"]
"""What the SLEAP tracker compares a detection and a candidate by.

``embeddings`` compares the appearance vectors of an embedding model, and is
taken only by a run whose last model is one.
"""

SleapScoringMethod = Literal["oks", "cosine_sim", "iou", "euclidean_dist"]
"""How the SLEAP tracker scores the association between those features."""

SleapMatchingMethod = Literal["hungarian", "greedy"]
"""How the SLEAP tracker assigns detections to tracks from those scores."""

# The device families sleap-nn names as they are. A CUDA index is the other
# spelling mosaic accepts, and is translated.
_DEVICE_FAMILIES: Final = frozenset({"cpu", "cuda", "mps"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SleapNotFoundError(ToolNotFoundError):
    """Raised when the SLEAP environment cannot do what a run needs of it.

    A SLEAP console script or ``conda`` cannot be located, or the environment
    resolves and its sleap-io cannot write the analysis export.
    """

    default_message = (
        "No SLEAP install was found: mosaic looks up 'sleap-convert' on $PATH "
        "and runs 'sleap-nn' from the same environment. Install SLEAP with its "
        "nn extra (e.g. 'uv tool install \"sleap[nn]\"'), or point "
        "MOSAIC_SLEAP_CONDA_ENV / MOSAIC_SLEAP_BIN at an environment that has "
        "sleap-nn, which alone is enough. See https://sleap.ai for installation "
        "instructions."
    )


class SleapError(ToolExitError):
    """Raised when a SLEAP subprocess exits with a non-zero return code."""

    tool_name = "SLEAP"


# SLEAP's console scripts are always installed together, so MOSAIC_SLEAP_BIN
# pointing at any one of them names the directory the others live in. Plainly
# named because the training, probe and label-export modules resolve through it
# too: one environment serves sleap-nn (tracking and training) and its own
# python (the export and the readbacks) alike, and declaring it twice is how
# they drift apart.
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


@dataclass(frozen=True, slots=True)
class SleapNoDetections:
    """sleap-nn's report that a video has no instance to embed.

    Attributes:
        message: The usage error that sleap-nn exited with.
        argv: The command that exited with it.
    """

    message: str
    argv: tuple[str, ...]


@dataclass
class SleapAnalysisExportResult:
    """Result of a SLEAP ``.slp`` -> analysis HDF5 export."""

    analysis_h5_path: Path
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Invocation resolution
# ---------------------------------------------------------------------------


def sleap_invocation(
    script: str,
    *,
    sleap_conda_env: str | None = None,
    sleap_bin: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch a SLEAP console *script*, as an argv prefix.

    The shared five-step ladder (:func:`tool_invocation`) applied to
    :data:`SLEAP_ENV`, with the script as the executable -- so one SLEAP
    environment serves both ``sleap-nn`` and its ``python``.
    """
    return tool_invocation(
        SLEAP_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
        executable=script,
    )


def publish_from_snippet(
    invocation: Sequence[str],
    snippet: str,
    inputs: Sequence[Path | str],
    out: Path,
    *,
    idle_timeout: float,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
    validate: Callable[[Path], object] | None = None,
) -> tuple[str, str]:
    """Run *snippet* in the SLEAP interpreter and publish the file it writes.

    The snippet's argv is *inputs* followed by ``<out>.partial``, which it
    writes. Once it exits cleanly having written it, *validate*, when given,
    checks the file, and the file is moved onto *out* with ``os.replace``.
    *out* is removed first, and a failed run leaves neither file. The staged
    file sits beside *out*, so the move is a rename within one filesystem.

    Args:
        invocation: The argv prefix that launches the SLEAP ``python``.
        snippet: The program run with ``-c``.
        inputs: The arguments before the staged path.
        out: Where the file is published.
        idle_timeout: How long the snippet may print nothing, in seconds.
        max_runtime: The longest the snippet may run, in seconds.
        cancel_check: Polled while the snippet runs.
        on_output: Called with each line of standard output.
        on_activity: Called with each line of either output stream.
        validate: Raises when the staged file is not what the snippet owes.

    Returns:
        The snippet's standard output and standard error.

    Raises:
        SleapError: The snippet exited non-zero.
        FileNotFoundError: It exited cleanly and wrote no file.
    """
    partial = out.with_name(f"{out.name}.partial")
    out.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*invocation, "-c", snippet, *(str(each) for each in inputs), str(partial)]
    logger.info("Running: %s", command_summary(cmd, SleapError.head))
    try:
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
        if not partial.is_file():
            raise missing_output_error("SLEAP", out, stdout, stderr)
        if validate is not None:
            _ = validate(partial)
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)
    return stdout, stderr


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

    *invocation* is the argv prefix from :func:`sleap_invocation`. *cancel_check*,
    when supplied, is polled while SLEAP runs; if it fires, the whole process
    group is killed and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    The subprocess always runs in its own process group (killable, orphan-safe)
    via :func:`run_supervised`.

    Raises :class:`SleapError` on non-zero exit.
    """
    cmd = [*invocation, *args]
    logger.info("Running: %s", command_summary(cmd))

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
    """The ``-d`` argument that puts ``sleap-nn track`` or ``predict`` on *device*.

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
        "index such as '0', or 'cuda:<index>'. sleap-nn runs inference on one "
        "device, so a list of them is refused."
    )
    raise ValueError(msg)


def _inference_args(
    subcommand: str,
    video_path: Path,
    output_slp: Path,
    models: Sequence[Path],
    *,
    peak_threshold: float,
    batch_args: Sequence[str],
    max_instances: int | None,
    frames: str | None,
    device: str | None,
    tracker_args: Sequence[str],
    max_tracks: int | None,
    extra_settings: Mapping[str, object] | None,
) -> list[str]:
    """Build the argv that ``sleap-nn track`` and ``sleap-nn predict`` share.

    The verb comes first, because this CLI dispatches on a subcommand, and the
    video is named by ``-i``: a positional path is an unexpected argument to it.
    *tracker_args* are the caller's tracking options, sent before *max_tracks*
    and the extra settings. *batch_args* is the batch size option with its
    value, which the two subcommands spell differently, or empty.
    """
    args: list[str] = [subcommand, "-i", str(video_path), "-o", str(output_slp)]
    for model in models:
        args.extend(["-m", str(model)])
    args.extend(["--peak_threshold", str(peak_threshold)])
    args.extend(batch_args)
    if max_instances is not None:
        args.extend(["-n", str(max_instances)])
    if frames is not None:
        args.extend(["--frames", frames])
    args.extend(sleap_track_device_args(device))
    args.extend(tracker_args)
    if max_tracks is not None:
        args.extend(["--max_tracks", str(max_tracks)])
    args.extend(_flatten_extra(extra_settings))
    return args


def run_sleap_track(
    video_path: Path | str,
    output_slp: Path | str,
    *,
    model_paths: Sequence[Path | str],
    tracking: bool = True,
    use_flow: bool = False,
    candidates_method: SleapCandidatesMethod = "fixed_window",
    features: SleapFeatures = "keypoints",
    scoring_method: SleapScoringMethod = "oks",
    track_matching_method: SleapMatchingMethod = "hungarian",
    tracking_window_size: int = 5,
    max_tracks: int | None = None,
    max_instances: int | None = None,
    peak_threshold: float = 0.2,
    batch_size: int = SLEAP_DEFAULT_BATCH_SIZE,
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

    tracker_args: list[str] = []
    if tracking:
        tracker_args.append("-t")
        if use_flow:
            tracker_args.append("--use_flow")
        tracker_args.extend(["--candidates_method", candidates_method])
        tracker_args.extend(["--features", features])
        tracker_args.extend(["--scoring_method", scoring_method])
        tracker_args.extend(["--track_matching_method", track_matching_method])
        tracker_args.extend(["--tracking_window_size", str(tracking_window_size)])
    args = _inference_args(
        _SLEAP_TRACK_SUBCOMMAND,
        video_path,
        output_slp,
        models,
        peak_threshold=peak_threshold,
        batch_args=["--batch_size", str(batch_size)],
        max_instances=max_instances,
        frames=frames,
        device=device,
        tracker_args=tracker_args,
        max_tracks=max_tracks if tracking else None,
        extra_settings=extra_settings,
    )

    stdout, stderr = _run_sleap(
        sleap_invocation(
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


def run_sleap_predict(
    video_path: Path | str,
    output_slp: Path | str,
    *,
    model_paths: Sequence[Path | str],
    track_matching_method: SleapMatchingMethod = "hungarian",
    tracking_window_size: int = 5,
    max_tracks: int | None = None,
    max_instances: int | None = None,
    peak_threshold: float = 0.2,
    batch_size: int = SLEAP_DEFAULT_BATCH_SIZE,
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
) -> SleapNoDetections | None:
    """Run ``sleap-nn predict`` with an embedding model and track by appearance.

    The last of *model_paths* is the embedding model, and the ones before it
    are the detection models. The run tracks by the cosine similarity of the
    appearance vectors, matches against the last instances of each track, and
    saves the vectors into the predictions ``.slp``. Every one of those options
    is sent. A sleap-nn release that changes a default then changes no output
    under an unchanged run id. The batch
    size is sent as ``-b`` only when it differs from
    :data:`SLEAP_DEFAULT_BATCH_SIZE`. ``-b`` also sets the embedding stage's crop
    batch, which is 64 without it. The default batch size leaves the embedder
    at 64.

    Args:
        video_path: Input video file, passed as ``-i``.
        output_slp: Destination ``.slp`` predictions file, passed as ``-o``.
        model_paths: The detection models, then the embedding model, passed as
            repeated ``-m`` flags in order.
        track_matching_method: ``--track_matching_method``.
        tracking_window_size: ``--tracking_window_size``: how many instances per
            track are kept as candidates.
        max_tracks: ``--max_tracks``, the cap on the number of tracks.
        max_instances: Cap on instances per frame.
        peak_threshold: Minimum confidence for a detected peak.
        batch_size: Inference batch size, sent as ``-b`` when it is not the
            default. It sets the detection batch and the embedding crop batch.
        frames: Frame selection, e.g. ``"0-1000"``.
        device: Translated by :func:`sleap_track_device_args`.
        extra_settings: Additional ``sleap-nn predict`` options passed as
            ``--key value`` pairs.
        idle_timeout: Kill the subprocess after this many seconds with no output.
        max_runtime: Optional absolute wall-clock ceiling.
        sleap_conda_env: Run SLEAP inside this conda env (overrides
            ``MOSAIC_SLEAP_CONDA_ENV``).
        sleap_bin: Path to a SLEAP console script (overrides
            ``MOSAIC_SLEAP_BIN``).
        cancel_check: Polled while SLEAP runs.
        on_output: Called with each chunk of output.
        on_activity: Called with each line of either output stream.

    Returns:
        sleap-nn's report when the video has no instance to embed, else
        ``None``. Then *output_slp* does not exist.

    Raises:
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: SLEAP exited non-zero for a reason other than an empty video.
        FileNotFoundError: It exited cleanly and wrote no *output_slp*.
        ValueError: *model_paths* is empty, or *device* is unusable.
    """
    video_path = Path(video_path)
    output_slp = Path(output_slp)
    output_slp.parent.mkdir(parents=True, exist_ok=True)
    models = [Path(m) for m in model_paths]
    if not models:
        raise ValueError("run_sleap_predict requires at least one model directory")

    args = _inference_args(
        _SLEAP_PREDICT_SUBCOMMAND,
        video_path,
        output_slp,
        models,
        peak_threshold=peak_threshold,
        batch_args=(
            ["-b", str(batch_size)] if batch_size != SLEAP_DEFAULT_BATCH_SIZE else []
        ),
        max_instances=max_instances,
        frames=frames,
        device=device,
        tracker_args=[
            "-t",
            "--features",
            "embeddings",
            "--scoring_method",
            "cosine_sim",
            "--candidates_method",
            "local_queues",
            "--save_embeddings",
            "slp",
            "--track_matching_method",
            track_matching_method,
            "--tracking_window_size",
            str(tracking_window_size),
        ],
        max_tracks=max_tracks,
        extra_settings=extra_settings,
    )

    try:
        stdout, stderr = _run_sleap(
            sleap_invocation(
                _SLEAP_TRACK_SCRIPT,
                sleap_conda_env=sleap_conda_env,
                sleap_bin=sleap_bin,
            ),
            args,
            idle_timeout=idle_timeout,
            max_runtime=max_runtime,
            cancel_check=cancel_check,
            on_output=on_output,
            on_activity=on_activity,
        )
    except SleapError as error:
        found = _NO_DETECTIONS.search(error.stderr) or _NO_DETECTIONS.search(
            error.stdout
        )
        if found is None:
            raise
        return SleapNoDetections(message=found.group(), argv=tuple(error.cmd))

    if not output_slp.exists():
        raise missing_output_error("SLEAP", output_slp, stdout, stderr)
    return None


# Run by the SLEAP interpreter. Its only inputs are argv: the predictions .slp
# and where to write the analysis HDF5. The call is the one sleap 1.6.5's
# sleap-convert makes for its analysis format. A video without labeled frames
# writes no file and is reported on stdout, as sleap-convert reports it.
_ANALYSIS_EXPORT_SNIPPET: str = """
import sys

import sleap_io

_predicted, _out = sys.argv[1], sys.argv[2]
_labels = sleap_io.load_slp(_predicted)
if len(_labels.videos) != 1:
    sys.exit(f"{_predicted} has {len(_labels.videos)} videos, not one")
_video = _labels.videos[0]
try:
    sleap_io.save_analysis_h5(
        _labels,
        _out,
        video=_video,
        labels_path=_predicted,
        all_frames=True,
        preset="matlab",
    )
except ValueError as _error:
    if "No labeled frames" not in str(_error):
        raise
    print(f"No labeled frames in {_video.filename}. Skipping.")
"""


def run_sleap_analysis_export(
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
) -> SleapAnalysisExportResult:
    """Export a SLEAP ``.slp`` file to its analysis HDF5 form.

    The SLEAP interpreter loads the file with ``sleap_io.load_slp`` and writes
    ``<output_h5>.partial`` with ``sleap_io.save_analysis_h5`` in the ``matlab``
    layout, every frame of the file's one video included. The file is then
    moved onto *output_h5* with ``os.replace``. Its per-dataset ``dims``
    attributes let the mosaic converter normalize the layout without any SLEAP
    dependency. *output_h5* is removed first, and a failed export leaves
    neither file.

    Args:
        slp_path: The predictions ``.slp``.
        output_h5: Where the analysis ``.h5`` is published.
        idle_timeout: How long the export may print nothing, in seconds.
        max_runtime: The longest the export may run, in seconds.
        sleap_conda_env: Run SLEAP inside this conda env (overrides
            ``MOSAIC_SLEAP_CONDA_ENV``).
        sleap_bin: Path to a SLEAP console script (overrides
            ``MOSAIC_SLEAP_BIN``).
        cancel_check: Polled while the snippet runs.
        on_output: Called with each line of standard output.
        on_activity: Called with each line of either output stream.

    Returns:
        The published path, with the snippet's output streams.

    Raises:
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: The snippet exited non-zero.
        FileNotFoundError: It exited cleanly and wrote no file, which is the
            case of a video without labeled frames.
    """
    output_h5 = Path(output_h5)
    stdout, stderr = publish_from_snippet(
        sleap_invocation(
            "python", sleap_conda_env=sleap_conda_env, sleap_bin=sleap_bin
        ),
        _ANALYSIS_EXPORT_SNIPPET,
        [slp_path],
        output_h5,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        cancel_check=cancel_check,
        on_output=on_output,
        on_activity=on_activity,
    )
    return SleapAnalysisExportResult(
        analysis_h5_path=output_h5, stdout=stdout, stderr=stderr
    )


_EMBEDDINGS_READBACK_IDLE_SECONDS: Final = 900.0
"""How long the readback may print nothing. It prints after loading the file."""

# Run by the SLEAP interpreter. Its only inputs are argv: the predictions .slp and
# where to write the .npz. One record per instance with a track, in file order. An
# instance's appearance vector is its identity embedding, and a file without any
# writes vectors with no columns. The file is opened by handle, because savez
# handed a path without the .npz suffix appends one.
_EMBEDDINGS_READBACK_SNIPPET: str = """
import sys

import numpy as np
import sleap_io

_predicted, _out = sys.argv[1], sys.argv[2]
_labels = sleap_io.load_slp(_predicted)
print(f"loaded {len(_labels.labeled_frames)} labeled frames", flush=True)

_frames, _names, _users, _found = [], [], [], []
for _frame in _labels.labeled_frames:
    for _instance in _frame.instances:
        if _instance.track is None:
            continue
        _embedding = getattr(_instance, "identity_embedding", None)
        _frames.append(int(_frame.frame_idx))
        _names.append(str(_instance.track.name))
        _users.append(not isinstance(_instance, sleap_io.PredictedInstance))
        _found.append(
            None
            if _embedding is None
            else np.asarray(_embedding.vector, dtype=np.float32).reshape(-1)
        )

_lengths = sorted({_vector.shape[0] for _vector in _found if _vector is not None})
if len(_lengths) > 1:
    sys.exit(f"appearance vectors of {_lengths} lengths in {_predicted}")
_has_vector = [_vector is not None for _vector in _found]
_vectors = np.zeros((len(_found), _lengths[0] if _lengths else 0), dtype=np.float32)
for _row, _vector in enumerate(_found):
    if _vector is not None:
        _vectors[_row] = _vector

with open(_out, "wb") as _handle:
    np.savez(
        _handle,
        frame_idx=np.asarray(_frames, dtype=np.int64),
        track_name=np.asarray(_names, dtype=np.str_),
        is_user=np.asarray(_users, dtype=np.bool_),
        has_vector=np.asarray(_has_vector, dtype=np.bool_),
        vectors=_vectors,
    )
print(f"read {len(_frames)} tracked instances, {sum(_has_vector)} with a vector")
"""


def read_back_embeddings(
    predicted: Path,
    out: Path,
    *,
    sleap_conda_env: str | None = None,
    sleap_bin: Path | str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_activity: Callable[[str], None] | None = None,
) -> Path:
    """Write the tracked instances of *predicted* and their vectors to *out*.

    The SLEAP interpreter writes ``<out>.partial``, which is checked with
    :func:`~mosaic.core.track_library.sleap.read_appearance_readback` and then
    moved onto *out* with ``os.replace``. *out* is removed first. A failed
    readback leaves neither file.

    Args:
        predicted: A predictions ``.slp``.
        out: Where the ``.npz`` is published, ``<entry_key>.embeddings.npz``.
        sleap_conda_env: Run SLEAP inside this conda env (overrides
            ``MOSAIC_SLEAP_CONDA_ENV``).
        sleap_bin: Path to a SLEAP console script (overrides
            ``MOSAIC_SLEAP_BIN``).
        cancel_check: Polled while the snippet runs.
        on_activity: Called with each line of either output stream.

    Returns:
        *out*.

    Raises:
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: The snippet exited non-zero.
        FileNotFoundError: It exited cleanly and wrote no file.
        ValueError: The file it wrote is not a readback.
    """
    _ = publish_from_snippet(
        sleap_invocation(
            "python", sleap_conda_env=sleap_conda_env, sleap_bin=sleap_bin
        ),
        _EMBEDDINGS_READBACK_SNIPPET,
        [predicted],
        out,
        idle_timeout=_EMBEDDINGS_READBACK_IDLE_SECONDS,
        cancel_check=cancel_check,
        on_activity=on_activity,
        validate=read_appearance_readback,
    )
    return out
