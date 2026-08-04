"""Run Lightning Pose inference from the command line.

Lightning Pose has no output-addressable console verb (``litpose predict`` writes
into the *model* directory and always computes metrics), so this wrapper drives
its Python API instead -- but from *outside* the mosaic environment. It resolves
the Lightning Pose environment's ``python`` and hands it a **static** ``-c``
snippet whose only inputs are ``sys.argv`` (model directory, video, output CSV,
precision, Hydra overrides). mosaic itself never imports ``lightning_pose``; the
import lives only in code the Lightning Pose interpreter runs, so the heavy
torch / DALI stack stays out of the mosaic env and out of the type checker.

Requires:
    The ``litpose`` console script (https://lightning-pose.readthedocs.io).
    Lightning Pose is heavy (PyTorch + Lightning + NVIDIA DALI) and video
    inference needs a Linux CUDA GPU, so it lives in its **own** environment.
    Point the wrapper at it one of three ways (highest precedence first), via
    per-call args or env vars:

    * ``litpose_conda_env=`` / ``MOSAIC_LITPOSE_CONDA_ENV`` -- run via
      ``conda run -n <env> python``;
    * ``litpose_bin=`` / ``MOSAIC_LITPOSE_BIN`` -- a path to the ``litpose``
      console script (or any script in its ``bin``), whose sibling ``python`` is
      used;
    * otherwise ``litpose`` is looked up on ``$PATH`` and its sibling ``python``
      is used (the ``uv tool install`` / activated-env case).

    Lightning Pose inference is headless and needs no ``DISPLAY``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
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

_LITPOSE_SCRIPT: str = "litpose"
_PYTHON: str = "python"

# A static program run by the Lightning Pose interpreter via ``python -c``. Its
# only inputs are argv (model_dir, video, out_csv, precision, *hydra_overrides),
# so no path is ever interpolated into the source and mosaic never imports
# lightning_pose. It runs LP's Python API with an explicit output directory and
# metrics disabled -- neither of which the ``litpose predict`` console verb
# exposes -- then renames the stem-named CSV to the requested output path.
_PREDICT_SNIPPET: str = """
import sys, shutil
from pathlib import Path
from lightning_pose.api import Model

_argv = sys.argv[1:]
_model_dir, _video, _out_csv, _precision = _argv[0], _argv[1], _argv[2], _argv[3]
_overrides = _argv[4:]
if _overrides:
    _model = Model.from_dir2(_model_dir, hydra_overrides=_overrides, precision=_precision)
else:
    _model = Model.from_dir(_model_dir, precision=_precision)
_out = Path(_out_csv)
_out.parent.mkdir(parents=True, exist_ok=True)
_model.predict_on_video_file(
    _video,
    output_dir=str(_out.parent),
    compute_metrics=False,
    generate_labeled_video=False,
)
_produced = _out.parent / (Path(_video).stem + ".csv")
if str(_produced) != str(_out):
    shutil.move(str(_produced), str(_out))
"""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LitposeNotFoundError(ToolNotFoundError):
    """Raised when the ``litpose`` console script (or ``conda``) cannot be located."""

    default_message = (
        "The 'litpose' console script was not found on $PATH.  Install "
        "Lightning Pose (e.g. 'pip install lightning-pose') in its own "
        "environment and point MOSAIC_LITPOSE_CONDA_ENV / MOSAIC_LITPOSE_BIN "
        "at it.  See https://lightning-pose.readthedocs.io for installation "
        "instructions."
    )


class LitposeError(ToolExitError):
    """Raised when a Lightning Pose subprocess exits with a non-zero return code."""

    tool_name = "Lightning Pose"
    # The argv is ``python -c <the whole predict program>``; echoing past the
    # fourth token in an error message prints the program.
    head = 4


# Lightning Pose is driven through its environment's ``python`` rather than a
# console verb, so the ``litpose`` script is what locates that interpreter.
# Plainly named because training resolves through it too: one environment
# serves prediction and training alike, and declaring it twice is how two
# copies drift.
LITPOSE_ENV: Final = ToolEnv(
    tool="Lightning Pose",
    conda_env_var="MOSAIC_LITPOSE_CONDA_ENV",
    bin_var="MOSAIC_LITPOSE_BIN",
    bin_mode="sibling",
    not_found=LitposeNotFoundError,
    locator=_LITPOSE_SCRIPT,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LitposePredictResult:
    """Result of a Lightning Pose video-inference run."""

    csv_path: Path
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Invocation resolution
# ---------------------------------------------------------------------------


def _litpose_invocation(
    *,
    litpose_conda_env: str | None = None,
    litpose_bin: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch the Lightning Pose ``python``, as an argv prefix.

    The shared five-step ladder (:func:`tool_invocation`) applied to
    :data:`LITPOSE_ENV`. What is launched is the interpreter, not a console
    verb, and what is *looked up* is the ``litpose`` script beside it -- a bare
    ``python`` on ``$PATH`` would be the caller's own.
    """
    return tool_invocation(
        LITPOSE_ENV,
        executable=_PYTHON,
        conda_env=litpose_conda_env,
        bin_path=litpose_bin,
    )


def _override_args(overrides: Mapping[str, object] | None) -> list[str]:
    """Flatten Hydra overrides into ``key=value`` argv tokens (``None`` values skipped)."""
    if not overrides:
        return []
    return [f"{key}={value}" for key, value in overrides.items() if value is not None]


def _run_litpose(
    invocation: list[str],
    args: list[str],
    *,
    idle_timeout: float,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Execute the Lightning Pose ``python`` with *args* and return (stdout, stderr).

    *invocation* is the argv prefix from :func:`_litpose_invocation`.
    *cancel_check*, when supplied, is polled while inference runs; if it fires, the
    whole process group is killed and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    The subprocess always runs in its own process group (killable, orphan-safe)
    via :func:`run_supervised`.

    Raises :class:`LitposeError` on non-zero exit.
    """
    cmd = [*invocation, *args]
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
        raise LitposeError(cmd, returncode, stdout, stderr)

    return stdout, stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_litpose_predict(
    video_path: Path | str,
    out_csv: Path | str,
    *,
    model_dir: Path | str,
    precision: str = "fp32",
    overrides: Mapping[str, object] | None = None,
    idle_timeout: float = 900,
    max_runtime: float | None = None,
    litpose_conda_env: str | None = None,
    litpose_bin: Path | str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> LitposePredictResult:
    """Run Lightning Pose video inference, writing a DeepLabCut-style CSV.

    Drives Lightning Pose's ``Model.predict_on_video_file`` in its own
    environment (with ``compute_metrics=False`` and an explicit output directory),
    then renames the produced CSV to *out_csv*. Lightning Pose is single-animal,
    per-frame: the CSV has one row per frame and ``(scorer, bodypart, coord)``
    columns, which the ``deeplabcut`` converter reads unchanged.

    Args:
        video_path: Input video file (an H.264 ``.mp4``).
        out_csv: Destination predictions CSV.
        model_dir: A trained Lightning Pose model directory (``config.yaml`` plus
            ``tb_logs/.../checkpoints/*.ckpt``).
        precision: Forward-pass precision (``fp32`` / ``fp16`` / ``bf16``).
        overrides: Hydra config overrides applied at inference time.
        idle_timeout: Kill the subprocess after this many seconds with no output.
        max_runtime: Optional absolute wall-clock ceiling.
        litpose_conda_env: Run in this conda env (overrides
            ``MOSAIC_LITPOSE_CONDA_ENV``).
        litpose_bin: Path to the ``litpose`` script (overrides
            ``MOSAIC_LITPOSE_BIN``).

    Returns:
        The path to the written CSV.

    Raises:
        LitposeNotFoundError: If Lightning Pose cannot be located.
        LitposeError: If Lightning Pose exits with a non-zero return code.
        FileNotFoundError: If the expected CSV is not found after inference.
    """
    video_path = Path(video_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-c",
        _PREDICT_SNIPPET,
        str(model_dir),
        str(video_path),
        str(out_csv),
        precision,
        *_override_args(overrides),
    ]
    stdout, stderr = _run_litpose(
        _litpose_invocation(
            litpose_conda_env=litpose_conda_env, litpose_bin=litpose_bin
        ),
        args,
        idle_timeout=idle_timeout,
        max_runtime=max_runtime,
        cancel_check=cancel_check,
        on_output=on_output,
    )

    if not out_csv.exists():
        raise missing_output_error("Lightning Pose", out_csv, stdout, stderr)

    return LitposePredictResult(csv_path=out_csv, stdout=stdout, stderr=stderr)
