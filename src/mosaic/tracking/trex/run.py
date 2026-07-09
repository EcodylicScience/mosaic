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

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from mosaic.core.pipeline.subprocess_util import run_supervised

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TRexNotFoundError(FileNotFoundError):
    """Raised when the ``trex`` binary (or ``conda``) cannot be located."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "The 'trex' binary was not found on $PATH.  "
                "Install T-Rex and ensure it is accessible.  "
                "See https://trex.run for installation instructions."
            )
        )


class TRexError(RuntimeError):
    """Raised when a T-Rex subprocess exits with a non-zero return code."""

    def __init__(
        self,
        cmd: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        cmd_str = " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "")
        super().__init__(
            f"T-Rex exited with code {returncode}.\n"
            f"  Command: {cmd_str}\n"
            f"  Stderr (last 500 chars): {stderr[-500:]}"
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


def _ensure_trex() -> str:
    """Return the absolute path to the ``trex`` binary, or raise."""
    path = shutil.which("trex")
    if path is None:
        raise TRexNotFoundError()
    return path


def _conda_invocation(env_name: str) -> list[str]:
    """Return an argv prefix that runs ``trex`` inside conda env *env_name*.

    Uses ``conda run`` so the target env is fully activated — TRex's embedded
    Python and shared libraries resolve against that env's ``CONDA_PREFIX``
    (required when TRex lives in a different env than the caller, e.g. a
    py3.11 ``track`` env driven from a py3.12 ``mosaic`` kernel). ``conda run``
    puts the target env's ``bin`` first on ``PATH``, so any self-relaunch of
    ``trex`` resolves to the real binary (no wrapper recursion).
    """
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if conda is None:
        raise TRexNotFoundError(
            f"'conda' was not found on $PATH; cannot run trex in env "
            f"'{env_name}'. Set MOSAIC_TREX_BIN to an explicit trex path "
            f"instead, or make conda available."
        )
    return [conda, "run", "--no-capture-output", "-n", env_name, "trex"]


def _trex_invocation(
    *,
    trex_conda_env: str | None = None,
    trex_bin: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch ``trex``, returned as an argv prefix.

    Precedence (first match wins):

    1. ``trex_conda_env`` argument → ``conda run -n <env> trex``
    2. ``trex_bin`` argument → ``[<binary>]``
    3. ``MOSAIC_TREX_CONDA_ENV`` env var → ``conda run -n <env> trex``
    4. ``MOSAIC_TREX_BIN`` env var → ``[<binary>]``
    5. ``shutil.which("trex")`` (default; raises :class:`TRexNotFoundError`)

    The default (case 5) preserves the original single-env behaviour, so
    existing callers are unaffected.
    """
    if trex_conda_env:
        return _conda_invocation(trex_conda_env)
    if trex_bin:
        return [str(trex_bin)]
    env_conda = os.environ.get("MOSAIC_TREX_CONDA_ENV")
    if env_conda:
        return _conda_invocation(env_conda)
    env_bin = os.environ.get("MOSAIC_TREX_BIN")
    if env_bin:
        return [env_bin]
    return [_ensure_trex()]


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


def _build_args(params: dict[str, Any]) -> list[str]:
    """Flatten a param dict into CLI ``-key value`` pairs.

    Booleans become bare flags (``-key``) when True and are omitted when
    False.  ``None`` values are skipped.
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
            args.extend([f"-{key}", f"[{','.join(str(v) for v in value)}]"])
            continue
        args.extend([f"-{key}", str(value)])
    return args


def _run_trex(
    args: list[str],
    *,
    timeout: int,
    invocation: list[str] | None = None,
    env: dict[str, str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
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

    run_env = {**os.environ, **(env or {})}
    # Jupyter exports ``MPLBACKEND=module://matplotlib_inline.backend_inline`` into
    # the kernel environment; inherited by the trex subprocess it makes matplotlib
    # (imported by ultralytics inside TRex) fail at import time, which TRex's
    # pybind11 glue turns into ``terminate()`` -> SIGABRT (exit 134). TRex never
    # needs an interactive backend, so neutralise an inherited IPython ``module://``
    # backend with a headless-safe one (explicit non-module backends are kept).
    if run_env.get("MPLBACKEND", "").startswith("module://"):
        run_env["MPLBACKEND"] = "Agg"

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=run_env,
        cancel_check=cancel_check,
        timeout=timeout,
    )

    if returncode != 0:
        raise TRexError(cmd, returncode, stdout, stderr)

    return stdout, stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_trex_convert(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    detect_model: Path | str | None = None,
    detect_type: str = "yolo",
    detect_conf_threshold: float = 0.5,
    detect_iou_threshold: float = 0.1,
    track_max_individuals: int = 1,
    cm_per_pixel: float = 1.0,
    meta_encoding: str = "gray",
    extra_settings: dict[str, Any] | None = None,
    timeout: int = 600,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TRexConvertResult:
    """Convert a raw video to T-Rex ``.pv`` format.

    Runs T-Rex in headless mode (``-nowindow -auto_quit``) to convert a
    video file, applying the specified detection model and parameters.

    Parameters
    ----------
    video_path : path
        Input video file (e.g. ``.mp4``, ``.avi``).
    output_dir : path
        Directory for output files (``.pv``, ``.settings``, background).
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
    timeout : int
        Subprocess timeout in seconds (default 600).
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
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "i": str(video_path),
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
    if detect_model is not None:
        params["m"] = str(detect_model)
    if extra_settings:
        params.update(extra_settings)

    stdout, stderr = _run_trex(
        _build_args(params),
        timeout=timeout,
        invocation=_trex_invocation(trex_conda_env=trex_conda_env, trex_bin=trex_bin),
        env=_resolve_display(display),
        cancel_check=cancel_check,
    )

    # Locate output files
    stem = video_path.stem
    pv_path = output_dir / f"{stem}.pv"
    if not pv_path.exists():
        # T-Rex may place the .pv next to the source video
        pv_alt = video_path.with_suffix(".pv")
        if pv_alt.exists():
            pv_path = pv_alt
        else:
            raise FileNotFoundError(
                f"Expected .pv file not found after conversion: {pv_path}"
            )

    settings_path = output_dir / f"{stem}.settings"
    if not settings_path.exists():
        settings_path = video_path.with_suffix(".settings")

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
    timeout: int = 600,
    trex_conda_env: str | None = None,
    trex_bin: Path | str | None = None,
    display: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
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
    timeout : int
        Subprocess timeout in seconds (default 600).
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
        timeout=timeout,
        invocation=_trex_invocation(trex_conda_env=trex_conda_env, trex_bin=trex_bin),
        env=_resolve_display(display),
        cancel_check=cancel_check,
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
