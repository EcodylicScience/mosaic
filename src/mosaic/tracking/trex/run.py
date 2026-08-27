"""Run T-Rex from the command line for video conversion and tracking.

This module provides Python wrappers around the ``trex`` CLI binary,
enabling headless (``-nowindow``) batch conversion and tracking of
animal videos.

Requires:
    The ``trex`` binary. Because the TRex conda package pins ``python=3.11`` and
    ``numpy=1.26``, it usually lives in its **own** conda env (e.g. ``track``)
    rather than the mosaic env. Every wrapper takes a
    :class:`~mosaic.tracking.common.toolenv.ToolEnv` naming where it is, and
    :data:`TREX_ENV` reads that from the machine. In precedence order:

    * ``ToolEnv.conda_env`` / ``MOSAIC_TREX_CONDA_ENV`` -- run via
      ``conda run -n <env> trex`` (recommended for the two-env setup);
    * ``ToolEnv.bin_path`` / ``MOSAIC_TREX_BIN`` -- an explicit path to the
      binary;
    * otherwise ``trex`` is looked up on ``$PATH`` (single-env install).

    TRex initialises an OpenGL/GLFW context even headless, so on a server you
    need a display: run a virtual framebuffer (``Xvfb :99 -screen 0 ...``) and
    set ``ToolEnv.display`` to ``":99"`` (or set ``DISPLAY`` /
    ``MOSAIC_TREX_DISPLAY``). Do **not** wrap ``trex`` in ``xvfb-run`` on
    ``$PATH`` -- TRex relaunches itself, so a per-call ``xvfb-run`` wrapper
    fork-bombs; one persistent ``Xvfb`` is correct.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Sequence

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.common.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    display_overlay,
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
TREX_ENV: Final = ToolEnv(
    tool="T-Rex",
    conda_env_var="MOSAIC_TREX_CONDA_ENV",
    bin_var="MOSAIC_TREX_BIN",
    bin_mode="direct",
    not_found=TRexNotFoundError,
    display_var="MOSAIC_TREX_DISPLAY",
)
"""Where TREx is launched from, as the machine's variables describe it.

A caller reaching a differently placed install states it once with
``TREX_ENV.placed(conda_env="track", display=":99")`` and passes the result to
whichever wrapper it calls.
"""


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


def _trex_invocation(env: ToolEnv = TREX_ENV) -> list[str]:
    """Resolve how to launch ``trex``, returned as an argv prefix.

    The shared five-step ladder (:func:`tool_invocation`) applied to *env*.
    ``conda run`` puts the target environment's ``bin`` first on ``PATH``, which
    matters more here than for the other tools: TRex relaunches itself, and a
    self-relaunch has to find the real binary rather than a wrapper.
    """
    return tool_invocation(env, executable=_TREX)


def _is_nested(value: Sequence[JsonValue]) -> bool:
    """Does this sequence contain a sequence or a mapping?"""
    return any(isinstance(item, (list, tuple, dict)) for item in value)


def _build_args(params: dict[str, JsonValue]) -> list[str]:
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
                args.extend([f"-{key}", f"[{','.join(str(item) for item in value)}]"])
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
    env_overlay: dict[str, str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Execute ``trex`` with *args* and return (stdout, stderr).

    *invocation* is the argv prefix from :func:`_trex_invocation` (defaults to
    the ``$PATH`` lookup). *env_overlay* is merged onto ``os.environ`` for the
    subprocess (e.g. ``{"DISPLAY": ":99"}``). *cancel_check*, when supplied,
    is polled while TRex runs; if it fires, TRex's whole process group is
    killed (it relaunches itself, so a group kill is required) and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    *idle_timeout* is an inactivity watchdog rather than a runtime ceiling:
    TRex prints progress while it is healthy, so a live long run keeps resetting
    it and a hung one does not. *max_runtime* is the absolute ceiling, and
    ``None`` leaves that to the caller or the queue.

    The subprocess always runs in its own process group (killable, orphan-safe)
    via :func:`run_supervised`.

    Raises :class:`TRexError` on non-zero exit.
    """
    cmd = [*(invocation or _trex_invocation()), *args]
    logger.info("Running: %s", " ".join(cmd))

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(env_overlay),
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
    params: TrexParams,
    detect_model_path: Path | None = None,
    output_name: str | None = None,
    env: ToolEnv = TREX_ENV,
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
    params : TrexParams
        The run's parameters. Every field the ``convert`` phase declares is read
        from here, along with the phase's inactivity and runtime bounds; the
        fields the ``track`` phase declares are ignored.
        :class:`~mosaic.tracking.trex.params.TrexParams` describes each one.
    detect_model_path : path, optional
        The detection weights file to hand T-Rex. A **resolved path**, not the
        reference ``params.detect_model`` holds: identity records what a model
        *is* and the executor needs where it is, and the two are different
        values. Unset sends no model.
    output_name : str, optional
        Stem for the ``.pv`` T-Rex writes, passed as its ``filename`` setting.
        Left unset, T-Rex names the output itself -- after the single source's
        stem, or, for several sources sharing a parent, **after that parent
        directory**. So a joined conversion without this lands somewhere the
        caller did not choose and may not find. Not a T-Rex *setting* in mosaic's
        sense: it is a path, and paths never enter a run identifier.
    env : ToolEnv
        Where T-Rex is launched from -- conda environment, binary, ``DISPLAY``.
        Defaults to :data:`TREX_ENV`, which reads the machine's
        ``MOSAIC_TREX_*`` variables. Placement never enters a run identifier.
    cancel_check : callable, optional
        Polled while T-Rex runs; if it fires, T-Rex's whole process group is
        killed and ``ProcessCancelled`` propagates.
    on_output : callable, optional
        Called with each line T-Rex prints, which is the activity signal a
        progress display and an in-flight claim are refreshed from.

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
    settings: dict[str, JsonValue] = {
        "i": str(sources[0]) if len(sources) == 1 else [str(p) for p in sources],
        "task": "convert",
        "nowindow": True,
        "auto_quit": True,
        "d": str(output_dir),
        "detect_type": params.detect_type,
        "detect_conf_threshold": params.detect_conf_threshold,
        "detect_iou_threshold": params.detect_iou_threshold,
        "track_max_individuals": params.track_max_individuals,
        "cm_per_pixel": params.cm_per_pixel,
        "meta_encoding": params.meta_encoding,
    }
    if output_name is not None:
        settings["filename"] = str(output_dir / f"{output_name}.pv")
    if detect_model_path is not None:
        settings["m"] = str(detect_model_path)
    if params.convert_extra_settings:
        settings.update(params.convert_extra_settings)

    stdout, stderr = _run_trex(
        _build_args(settings),
        idle_timeout=params.idle_timeout,
        max_runtime=params.max_runtime,
        invocation=_trex_invocation(env),
        env_overlay=display_overlay(env),
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


# Each entry is the column's name beside the list of modifiers it is exported
# under. Typed as the JSON values they are, because that is what they reach the
# settings dictionary as; a list of lists is invariant and would not assign.
TREX_DEFAULT_OUTPUT_FIELDS: Final[list[JsonValue]] = [
    ["X", ["RAW", "WCENTROID"]],
    ["Y", ["RAW", "WCENTROID"]],
    ["X", ["RAW", "HEAD"]],
    ["Y", ["RAW", "HEAD"]],
    ["VX", ["RAW", "HEAD"]],
    ["VY", ["RAW", "HEAD"]],
    ["AX", ["RAW", "HEAD"]],
    ["AY", ["RAW", "HEAD"]],
    ["ANGLE", ["RAW"]],
    ["ANGULAR_V", ["RAW"]],
    ["ANGULAR_A", ["RAW"]],
    ["MIDLINE_OFFSET", ["RAW"]],
    ["normalized_midline", ["RAW"]],
    ["midline_length", ["RAW"]],
    ["midline_x", ["RAW"]],
    ["midline_y", ["RAW"]],
    ["midline_segment_length", ["RAW"]],
    ["SPEED", ["RAW", "WCENTROID"]],
    ["SPEED", ["RAW", "PCENTROID"]],
    ["SPEED", ["RAW", "HEAD"]],
    ["BORDER_DISTANCE", ["PCENTROID"]],
    ["time", []],
    ["timestamp", []],
    ["frame", []],
    ["missing", []],
    ["num_pixels", []],
    ["ACCELERATION", ["RAW", "PCENTROID"]],
    ["ACCELERATION", ["RAW", "WCENTROID"]],
]
"""TREx's own default for ``output_fields``, restated.

Restated because ``output_fields`` is a **value, not a delta**: assigning it
replaces this list rather than extending it, and TREx offers no spelling that
appends. Sending the keypoint columns alone therefore drops ``frame``, ``time``
and ``X``/``Y`` -- the columns the NPZ converter reads -- and the entry does not
survive the bridge. Measured on TREx 2.0.0: 38 exported keys become 22.

Kept in mosaic rather than read back from TREx because there is nowhere to read
it from: the value only appears in TREx's parameter reference and in a settings
file it writes after a run, neither of which is available when the argv is
built. It changes about as often as the integration version does, and a drift
shows up as a column vanishing from ``tracks/``, which
``test_trex_output_fields`` pins.
"""


def pose_output_fields(n_keypoints: int) -> list[JsonValue]:
    """The ``poseX<i>``/``poseY<i>`` entries for a model with *n_keypoints*.

    The naming is TREx's own: :func:`list_auto_pose_fields` builds exactly these
    when it can, so a table produced this way is indistinguishable from one TREx
    named itself.

    Args:
        n_keypoints: How many keypoints the detection model reports. Zero or
            fewer yields no fields.

    Returns:
        One ``[name, []]`` pair per coordinate, X before Y within a keypoint.
    """
    return [
        [f"pose{axis}{index}", []]
        for index in range(max(0, n_keypoints))
        for axis in ("X", "Y")
    ]


def run_trex_track(
    pv_path: Path | str,
    output_dir: Path | str,
    *,
    params: TrexParams,
    vi_model_path: Path | None = None,
    settings_path: Path | str | None = None,
    env: ToolEnv = TREX_ENV,
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
    params : TrexParams
        The run's parameters. Every field the ``track`` phase declares is read
        from here, along with the phase's inactivity and runtime bounds; the
        fields the ``convert`` phase declares are ignored.
        :class:`~mosaic.tracking.trex.params.TrexParams` describes each one.
    vi_model_path : path, optional
        The identity weights file to hand T-Rex. A **resolved path**, not the
        reference ``params.visual_identification_model_path`` holds, for the
        reason :func:`run_trex_convert` gives for its detection model. Unset
        sends no model.
    settings_path : path, optional
        The conversion's ``.settings`` file, passed to T-Rex as ``-s``.

        **Naming it is what carries the detection parameters into tracking.**
        Re-opening a ``.pv`` from the command line always takes T-Rex's
        restricted read-back path, which recovers only ``meta_encoding``,
        ``meta_source_path``, ``meta_video_size``, ``meta_real_width``,
        ``frame_rate``, ``cm_per_pixel`` and ``detect_type`` from the file
        itself; everything else comes from compiled defaults, from the defaults
        that ``detect_type`` implies, and from this file. T-Rex looks for one
        implicitly at ``<output_dir>/<pv stem>.settings`` and **not** beside the
        ``.pv``, so once a conversion is shared the implicit lookup finds
        nothing and says nothing. Passed as an absolute path it is honoured
        verbatim, and a named-but-missing file is an error rather than silence.
    env, cancel_check, on_output
        As :func:`run_trex_convert` takes them.

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

    settings: dict[str, JsonValue] = {
        "i": str(pv_path),
        "task": "track",
        "nowindow": True,
        "auto_quit": True,
        "d": str(output_dir),
        "track_max_individuals": params.track_max_individuals,
        "track_max_speed": params.track_max_speed,
        "track_max_reassign_time": params.track_max_reassign_time,
        "track_trusted_probability": params.track_trusted_probability,
    }
    if params.analysis_range is not None:
        settings["analysis_range"] = list(params.analysis_range)
    if vi_model_path is not None:
        settings["visual_identification_model_path"] = str(vi_model_path)
    if params.auto_train:
        settings["auto_train"] = True
    if params.detect_keypoint_count:
        # Before the track_extra_settings update below, so an explicit
        # output_fields from the caller replaces this rather than the reverse.
        settings["output_fields"] = TREX_DEFAULT_OUTPUT_FIELDS + pose_output_fields(
            params.detect_keypoint_count
        )
    if settings_path is not None:
        # Absolute, because a relative one is resolved under `output_dir` and
        # the conversion this names generally does not live there.
        settings["s"] = str(Path(settings_path).resolve())
    if params.track_extra_settings:
        settings.update(params.track_extra_settings)

    stdout, stderr = _run_trex(
        _build_args(settings),
        idle_timeout=params.idle_timeout,
        max_runtime=params.max_runtime,
        invocation=_trex_invocation(env),
        env_overlay=display_overlay(env),
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
    params: TrexParams,
    detect_model_path: Path | None,
    vi_model_path: Path | None,
    env: ToolEnv,
) -> TRexTrackResult:
    """Convert and track a single video (for use with ProcessPoolExecutor)."""
    vid_output = output_dir / video_path.stem
    convert_result = run_trex_convert(
        video_path,
        vid_output,
        params=params,
        detect_model_path=detect_model_path,
        env=env,
    )
    return run_trex_track(
        convert_result.pv_path,
        vid_output,
        params=params,
        vi_model_path=vi_model_path,
        env=env,
    )


def run_trex_batch(
    video_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    params: TrexParams,
    detect_model_path: Path | None = None,
    vi_model_path: Path | None = None,
    parallel_workers: int = 1,
    env: ToolEnv = TREX_ENV,
) -> list[TRexTrackResult]:
    """Convert and track multiple videos.

    Each video is converted to ``.pv`` format and then tracked, with
    output placed in a per-video subdirectory under *output_dir*.

    One *params* drives both phases of every video, so each phase reads the
    fields it declares: the conversion sends ``convert_extra_settings`` and the
    detection knobs, the tracking sends ``track_extra_settings`` and the
    tracking knobs, and ``track_max_individuals`` reaches both. Nothing about a
    run is per video except the video, which is what makes the parallel form
    safe.

    This wrapper writes no run index, no markers and no tracks tables. It is the
    plain loop for a notebook; :func:`mosaic.tracking.trex.run_trex` is the
    tracked, resumable, dataset-addressed form.

    Parameters
    ----------
    video_paths : sequence of paths
        Input video files to process.
    output_dir : path
        Root output directory.
    params : TrexParams
        The parameters both phases read, applied to every video.
    detect_model_path : path, optional
        Resolved detection weights, as :func:`run_trex_convert` takes them.
    vi_model_path : path, optional
        Resolved identity weights, as :func:`run_trex_track` takes them.
    parallel_workers : int
        Number of parallel workers (default 1 = sequential).
    env : ToolEnv
        Where T-Rex is launched from, as :func:`run_trex_convert` takes it.

    Returns
    -------
    list of TRexTrackResult
        One result per video, in the same order as *video_paths*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in video_paths]

    if parallel_workers <= 1:
        results: list[TRexTrackResult] = []
        for video in paths:
            logger.info("Processing %s ...", video.name)
            results.append(
                _convert_and_track_single(
                    video, output_dir, params, detect_model_path, vi_model_path, env
                )
            )
        return results

    with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
        futures = [
            pool.submit(
                _convert_and_track_single,
                video,
                output_dir,
                params,
                detect_model_path,
                vi_model_path,
                env,
            )
            for video in paths
        ]
        return [future.result() for future in futures]
