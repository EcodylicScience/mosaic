"""Run T-Rex from the command line for video conversion and tracking.

This module provides Python wrappers around the ``trex`` CLI binary,
enabling headless (``-nowindow``) batch conversion and tracking of
animal videos.

Requires:
    The ``trex`` binary. Because the TRex conda package pins ``python=3.11`` and
    ``numpy=1.26``, it usually lives in its **own** conda env (e.g. ``track``)
    rather than the mosaic env. Point the wrappers at it one of three ways
    (highest precedence first), via per-call args or env vars:

    * ``trex_conda_env=`` / ``MOSAIC_TREX_CONDA_ENV`` — run via
      ``conda run -n <env> trex`` (recommended for the two-env setup);
    * ``trex_bin=`` / ``MOSAIC_TREX_BIN`` — an explicit path to the binary;
    * otherwise ``trex`` is looked up on ``$PATH`` (single-env install).

    TRex initialises an OpenGL/GLFW context even headless, so on a server you
    need a display: run a virtual framebuffer (``Xvfb :99 -screen 0 ...``) and
    pass ``display=":99"`` (or set ``DISPLAY`` / ``MOSAIC_TREX_DISPLAY``). Do
    **not** wrap ``trex`` in ``xvfb-run`` on ``$PATH`` — TRex relaunches itself,
    so a per-call ``xvfb-run`` wrapper fork-bombs; one persistent ``Xvfb`` is
    correct.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Sequence

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

_TREX: Final = "trex"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TRexNotFoundError(ToolNotFoundError):
    """Raised when the ``trex`` binary (or ``conda``) cannot be located."""

    default_message = (
        "The 'trex' binary was not found on $PATH.  "
        "Install T-Rex and ensure it is accessible.  "
        "See https://trex.run for installation instructions."
    )


class TRexError(ToolExitError):
    """Raised when a T-Rex subprocess exits with a non-zero return code."""

    tool_name = "T-Rex"


# TRex is a single binary, so an explicit MOSAIC_TREX_BIN names it directly.
_TREX_ENV: Final = ToolEnv(
    tool="T-Rex",
    conda_env_var="MOSAIC_TREX_CONDA_ENV",
    bin_var="MOSAIC_TREX_BIN",
    bin_mode="direct",
    not_found=TRexNotFoundError,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TRexConvertResult:
    """Result of a T-Rex video conversion."""

    pv_path: Path
    settings_path: Path
    background_path: Path | None
    stdout: str
    stderr: str


@dataclass
class TRexTrackResult:
    """Result of a T-Rex tracking run."""

    npz_paths: list[Path] = field(default_factory=list)
    results_path: Path | None = None
    settings_path: Path = field(default_factory=Path)
    stdout: str = ""
    stderr: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trex_invocation(
    *,
    trex_conda_env: str | None = None,
    trex_bin: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch ``trex``, returned as an argv prefix.

    The shared five-step ladder (:func:`tool_invocation`) applied to
    :data:`_TREX_ENV`. ``conda run`` puts the target environment's ``bin`` first
    on ``PATH``, which matters more here than for the other tools: TRex
    relaunches itself, and a self-relaunch has to find the real binary rather
    than a wrapper.
    """
    return tool_invocation(
        _TREX_ENV,
        executable=_TREX,
        conda_env=trex_conda_env,
        bin_path=trex_bin,
    )


def _resolve_display(display: str | None) -> dict[str, str] | None:
    """Return an env overlay setting ``DISPLAY`` for the trex subprocess.

    TRex initialises a GLFW/OpenGL context even with ``-nowindow``, so it needs
    a display. On a headless host run a virtual framebuffer (``Xvfb :99 ...``)
    and either export ``DISPLAY`` (inherited automatically) or pass *display*
    here. Falls back to the ``MOSAIC_TREX_DISPLAY`` env var; ``None`` means use
    whatever ``DISPLAY`` is already in the environment.
    """
    d = display or os.environ.get("MOSAIC_TREX_DISPLAY")
    return {"DISPLAY": d} if d else None


def _is_nested(value: list[Any] | tuple[Any, ...]) -> bool:
    """Does this sequence contain a sequence or a mapping?"""
    return any(isinstance(item, (list, tuple, dict)) for item in value)


def _build_args(params: dict[str, Any]) -> list[str]:
    """Flatten a param dict into CLI ``-key value`` pairs.

    Booleans become bare flags (``-key``) when True and are omitted when
    False. ``None`` values are skipped.

    A flat sequence is written ``[a,b]``, which is what TREx's simple array
    parameters take and what mosaic has always sent for ``analysis_range``.

    A **nested** one is written as compact JSON instead, because Python's
    ``str`` of a nested list is its repr -- single quotes and spaces --
    which TREx's parameter parser does not accept. That made every nested
    parameter unreachable: ``output_fields``, which is how a user asks TREx
    to export ``tracklet_id`` or ``blobid``, is a list of ``[name, [sources]]``
    pairs, so passing it through ``track_extra_settings`` produced argv TREx
    would reject. Scoping the change to nested values leaves every flat one
    byte-identical to what it was.
    """
    args: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"-{key}")
            continue
        if isinstance(value, (list, tuple)):
            if _is_nested(value):
                args.extend([f"-{key}", json.dumps(value, separators=(",", ":"))])
            else:
                args.extend([f"-{key}", f"[{','.join(str(v) for v in value)}]"])
            continue
        if isinstance(value, dict):
            args.extend([f"-{key}", json.dumps(value, separators=(",", ":"))])
            continue
        args.extend([f"-{key}", str(value)])
    return args


def _run_trex(
    args: list[str],
    *,
    idle_timeout: float,
    max_runtime: float | None = None,
    invocation: list[str] | None = None,
    env: dict[str, str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Execute ``trex`` with *args* and return (stdout, stderr).

    *invocation* is the argv prefix from :func:`_trex_invocation` (defaults to
    the ``$PATH`` lookup). *env* is an overlay merged onto ``os.environ`` for
    the subprocess (e.g. ``{"DISPLAY": ":99"}``). *cancel_check*, when supplied,
    is polled while TRex runs; if it fires, TRex's whole process group is
    killed (it relaunches itself, so a group kill is required) and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    The subprocess always runs in its own process group (killable, orphan-safe)
    via :func:`run_supervised`.

    Raises :class:`TRexError` on non-zero exit.
    """
    cmd = [*(invocation or _trex_invocation()), *args]
    logger.info("Running: %s", " ".join(cmd))

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(env),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
    )

    if returncode != 0:
        raise TRexError(cmd, returncode, stdout, stderr)

    return stdout, stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _as_sources(video_path: Path | str | Sequence[Path | str]) -> list[Path]:
    """*video_path* as a non-empty list, whether one source was given or many.

    A bare ``str`` is a ``Sequence`` of characters, so the scalar case is tested
    for rather than fallen through to -- iterating it would turn one path into a
    list of one-character paths.
    """
    if isinstance(video_path, (str, Path)):
        return [Path(video_path)]
    sources = [Path(entry) for entry in video_path]
    if not sources:
        raise ValueError("a conversion needs at least one source video")
    return sources


def run_trex_convert(
    video_path: Path | str | Sequence[Path | str],
    output_dir: Path | str,
    *,
    output_name: str | None = None,
    detect_model: Path | str | None = None,
    detect_type: str = "yolo",
    detect_conf_threshold: float = 0.5,
    detect_iou_threshold: float = 0.1,
    track_max_individuals: int = 1,
    cm_per_pixel: float = 1.0,
    meta_encoding: str = "gray",
    extra_settings: dict[str, Any] | None = None,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> TRexConvertResult:
    """Convert a raw video to T-Rex ``.pv`` format.

    Runs T-Rex in headless mode (``-nowindow -auto_quit``) to convert a
    video file, applying the specified detection model and parameters.

    Parameters
    ----------
    video_path : path or sequence of paths
        Input video file (e.g. ``.mp4``, ``.avi``), or **several** of them. T-Rex
        takes its ``source`` as a ``PathArray`` and sums the frame counts of
        every file it names into one length, so a sequence of clips converts into
        a *single* ``.pv`` with one continuous frame index: identities never
        break at a clip boundary, and ``analysis_range`` addresses the joined
        timeline rather than any one file.

        Two properties of that join are the caller's to manage, because T-Rex
        will not. It refuses clips of differing resolution, but it takes the
        frame rate from the **first file alone** without checking the others --
        so a session whose clips were recorded at different rates converts into a
        ``.pv`` that labels all of them with the first clip's rate. Every
        per-second quantity T-Rex then reports, and its own ``time`` array, is
        wrong for the rest. See
        :func:`mosaic.core.media.timeline.concatenated_timeline` for the
        reconstruction, and :func:`mosaic.tracking.trex.joined.retime_joined_frame`
        for what mosaic does with it.
    output_dir : path
        Directory for output files (``.pv``, ``.settings``, background).
    output_name : str, optional
        Stem for the ``.pv`` T-Rex writes, passed as its ``filename`` setting.
        Left unset, T-Rex names the output itself -- after the single source's
        stem, or, for several sources sharing a parent, **after that parent
        directory**. So a joined conversion without this lands somewhere the
        caller did not choose and may not find. Not a T-Rex *setting* in mosaic's
        sense: it is a path, and paths never enter a run identifier.
    detect_model : path, optional
        Path to a YOLO ``.pt`` model file for detection/pose.
    detect_type : str
        Detection algorithm: ``"yolo"`` (default) or ``"background_subtraction"``.
    detect_conf_threshold : float
        Minimum YOLO detection confidence (default 0.5).
    detect_iou_threshold : float
        NMS IoU threshold for suppressing overlapping detections (default 0.1).
    track_max_individuals : int
        Maximum number of simultaneous individuals to track (default 1).
    cm_per_pixel : float
        Spatial calibration factor (default 1.0 = pixels).
    meta_encoding : str
        Pixel encoding: ``"gray"`` or ``"rgb8"`` (default ``"gray"``).
    extra_settings : dict, optional
        Additional T-Rex parameters passed as ``-key value`` pairs.
    idle_timeout : float
        Kill the subprocess after this many seconds with no output on either
        stream (an inactivity/hang watchdog; default 900). TRex prints progress
        while healthy, so a live long run keeps resetting it.
    max_runtime : float, optional
        Optional absolute wall-clock ceiling; ``None`` (default) imposes no
        total limit and leaves the ceiling to the caller / queue.
    trex_conda_env : str, optional
        Run ``trex`` inside this conda env via ``conda run -n <env>`` (e.g.
        ``"track"``). Use when TRex lives in a different env than the caller.
        Overrides ``MOSAIC_TREX_CONDA_ENV``. See :func:`_trex_invocation`.
    trex_bin : path, optional
        Explicit path to the ``trex`` binary (overrides ``MOSAIC_TREX_BIN``).
    display : str, optional
        ``DISPLAY`` value for the subprocess (e.g. ``":99"`` for a headless
        ``Xvfb``). Overrides ``MOSAIC_TREX_DISPLAY``; ``None`` inherits the
        current ``DISPLAY``.

    Returns
    -------
    TRexConvertResult
        Paths to the generated ``.pv``, ``.settings``, and background files.

    Raises
    ------
    TRexNotFoundError
        If the ``trex`` binary is not on ``$PATH``.
    TRexError
        If T-Rex exits with a non-zero return code.
    FileNotFoundError
        If the expected ``.pv`` output file is not found after conversion.
    """
    sources = _as_sources(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One path stays a bare string, several become T-Rex's `[a,b,c]` PathArray
    # literal -- which is what `_build_args` renders a flat list as already.
    params: dict[str, Any] = {
        "i": str(sources[0]) if len(sources) == 1 else [str(p) for p in sources],
        "task": "convert",
        "nowindow": True,
        "auto_quit": True,
        "d": str(output_dir),
        "detect_type": detect_type,
        "detect_conf_threshold": detect_conf_threshold,
        "detect_iou_threshold": detect_iou_threshold,
        "track_max_individuals": track_max_individuals,
        "cm_per_pixel": cm_per_pixel,
        "meta_encoding": meta_encoding,
    }
    if output_name is not None:
        params["filename"] = str(output_dir / f"{output_name}.pv")
    if detect_model is not None:
        params["m"] = str(detect_model)
    if extra_settings:
        params.update(extra_settings)

    stdout, stderr = _run_trex(
        _build_args(params),
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        invocation=_trex_invocation(trex_conda_env=trex_conda_env, trex_bin=trex_bin),
        env=_resolve_display(display),
        cancel_check=cancel_check,
        on_output=on_output,
    )

    # Locate output files. `output_name` pins the stem when it was given;
    # otherwise T-Rex named the output after the single source.
    stem = output_name if output_name is not None else sources[0].stem
    pv_path = output_dir / f"{stem}.pv"
    if not pv_path.exists():
        # T-Rex may place the .pv next to the source video
        pv_alt = sources[0].parent / f"{stem}.pv"
        if pv_alt.exists():
            pv_path = pv_alt
        else:
            raise missing_output_error("T-Rex", pv_path, stdout, stderr)

    settings_path = output_dir / f"{stem}.settings"
    if not settings_path.exists():
        settings_path = sources[0].parent / f"{stem}.settings"

    bg_path = output_dir / f"average_{stem}.png"
    if not bg_path.exists():
        bg_path = None

    return TRexConvertResult(
        pv_path=pv_path,
        settings_path=settings_path,
        background_path=bg_path,
        stdout=stdout,
        stderr=stderr,
    )


def run_trex_track(
    pv_path: Path | str,
    output_dir: Path | str,
    *,
    track_max_individuals: int = 1,
    track_max_speed: float = 80.0,
    track_max_reassign_time: float = 2.0,
    track_trusted_probability: float = 0.1,
    analysis_range: tuple[int, int] | None = None,
    visual_identification_model_path: Path | str | None = None,
    auto_train: bool = False,
    extra_settings: dict[str, Any] | None = None,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> TRexTrackResult:
    """Track individuals in a converted ``.pv`` video.

    Runs T-Rex in headless mode to perform tracking and (optionally)
    visual-identification training.

    Parameters
    ----------
    pv_path : path
        Converted T-Rex ``.pv`` file.
    output_dir : path
        Directory for output NPZ and results files.
    track_max_individuals : int
        Number of individuals to track (default 1).
    track_max_speed : float
        Maximum plausible speed in cm/s (default 80).
    track_max_reassign_time : float
        Seconds to wait before giving up on a lost individual (default 2.0).
    track_trusted_probability : float
        Probability threshold below which a tracklet is terminated (default 0.1).
    analysis_range : tuple of (start, end), optional
        Frame range to analyse.  ``-1`` means beginning/end of video.
    visual_identification_model_path : path, optional
        Path to pre-trained identity weights (``.pth``, without extension).
    auto_train : bool
        Automatically train visual identification after tracking (default False).
    extra_settings : dict, optional
        Additional T-Rex parameters passed as ``-key value`` pairs.
    idle_timeout : float
        Kill the subprocess after this many seconds with no output on either
        stream (an inactivity/hang watchdog; default 900). TRex prints progress
        while healthy, so a live long run keeps resetting it.
    max_runtime : float, optional
        Optional absolute wall-clock ceiling; ``None`` (default) imposes no
        total limit and leaves the ceiling to the caller / queue.
    trex_conda_env : str, optional
        Run ``trex`` inside this conda env via ``conda run -n <env>``
        (overrides ``MOSAIC_TREX_CONDA_ENV``). See :func:`_trex_invocation`.
    trex_bin : path, optional
        Explicit path to the ``trex`` binary (overrides ``MOSAIC_TREX_BIN``).
    display : str, optional
        ``DISPLAY`` for the subprocess (e.g. ``":99"`` for headless ``Xvfb``;
        overrides ``MOSAIC_TREX_DISPLAY``).

    Returns
    -------
    TRexTrackResult
        Paths to per-individual NPZ files and the results file.

    Raises
    ------
    TRexNotFoundError
        If the ``trex`` binary is not on ``$PATH``.
    TRexError
        If T-Rex exits with a non-zero return code.
    """
    pv_path = Path(pv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "i": str(pv_path),
        "task": "track",
        "nowindow": True,
        "auto_quit": True,
        "d": str(output_dir),
        "track_max_individuals": track_max_individuals,
        "track_max_speed": track_max_speed,
        "track_max_reassign_time": track_max_reassign_time,
        "track_trusted_probability": track_trusted_probability,
    }
    if analysis_range is not None:
        params["analysis_range"] = list(analysis_range)
    if visual_identification_model_path is not None:
        params["visual_identification_model_path"] = str(
            visual_identification_model_path
        )
    if auto_train:
        params["auto_train"] = True
    if extra_settings:
        params.update(extra_settings)

    stdout, stderr = _run_trex(
        _build_args(params),
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        invocation=_trex_invocation(trex_conda_env=trex_conda_env, trex_bin=trex_bin),
        env=_resolve_display(display),
        cancel_check=cancel_check,
        on_output=on_output,
    )

    # Locate output files
    data_dir = output_dir / "data"
    npz_paths: list[Path] = []
    if data_dir.is_dir():
        npz_paths = sorted(data_dir.glob("*.npz"))

    stem = pv_path.stem
    results_path = output_dir / f"{stem}.results"
    if not results_path.exists():
        results_path = None

    settings_path = output_dir / f"{stem}.settings"
    if not settings_path.exists():
        settings_path = pv_path.with_suffix(".settings")

    return TRexTrackResult(
        npz_paths=npz_paths,
        results_path=results_path,
        settings_path=settings_path,
        stdout=stdout,
        stderr=stderr,
    )


def _convert_and_track_single(
    video_path: Path,
    output_dir: Path,
    detect_model: Path | None,
    track_max_individuals: int,
    common_settings: dict[str, Any] | None,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
) -> TRexTrackResult:
    """Convert and track a single video (for use with ProcessPoolExecutor)."""
    vid_output = output_dir / video_path.stem
    convert_result = run_trex_convert(
        video_path,
        vid_output,
        detect_model=detect_model,
        track_max_individuals=track_max_individuals,
        extra_settings=common_settings,
        trex_conda_env=trex_conda_env,
        trex_bin=trex_bin,
        display=display,
    )
    return run_trex_track(
        convert_result.pv_path,
        vid_output,
        track_max_individuals=track_max_individuals,
        extra_settings=common_settings,
        trex_conda_env=trex_conda_env,
        trex_bin=trex_bin,
        display=display,
    )


def run_trex_batch(
    video_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    detect_model: Path | str | None = None,
    track_max_individuals: int = 1,
    common_settings: dict[str, Any] | None = None,
    parallel_workers: int = 1,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
) -> list[TRexTrackResult]:
    """Convert and track multiple videos.

    Each video is converted to ``.pv`` format and then tracked, with
    output placed in a per-video subdirectory under *output_dir*.

    Parameters
    ----------
    video_paths : sequence of paths
        Input video files to process.
    output_dir : path
        Root output directory.
    detect_model : path, optional
        YOLO ``.pt`` model for detection/pose.
    track_max_individuals : int
        Number of individuals per video (default 1).
    common_settings : dict, optional
        Additional T-Rex parameters applied to every video.
    parallel_workers : int
        Number of parallel workers (default 1 = sequential).
    trex_conda_env : str, optional
        Run ``trex`` inside this conda env via ``conda run -n <env>``
        (overrides ``MOSAIC_TREX_CONDA_ENV``). See :func:`_trex_invocation`.
    trex_bin : path, optional
        Explicit path to the ``trex`` binary (overrides ``MOSAIC_TREX_BIN``).
    display : str, optional
        ``DISPLAY`` for the subprocesses (e.g. ``":99"`` for headless ``Xvfb``;
        overrides ``MOSAIC_TREX_DISPLAY``).

    Returns
    -------
    list of TRexTrackResult
        One result per video, in the same order as *video_paths*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dm = Path(detect_model) if detect_model is not None else None
    paths = [Path(p) for p in video_paths]

    if parallel_workers <= 1:
        results: list[TRexTrackResult] = []
        for vp in paths:
            logger.info("Processing %s ...", vp.name)
            r = _convert_and_track_single(
                vp,
                output_dir,
                dm,
                track_max_individuals,
                common_settings,
                trex_conda_env,
                trex_bin,
                display,
            )
            results.append(r)
        return results

    with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
        futures = [
            pool.submit(
                _convert_and_track_single,
                vp,
                output_dir,
                dm,
                track_max_individuals,
                common_settings,
                trex_conda_env,
                trex_bin,
                display,
            )
            for vp in paths
        ]
        return [f.result() for f in futures]
