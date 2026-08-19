"""Locating the Ultralytics environments, and launching the runner program in them.

Ultralytics is AGPL-3.0, and so is the `mooch443/POLO
<https://github.com/mooch443/POLO>`_ fork. A mosaic that imports either is one
work with it, so both run as a separate program in an environment the user
builds, and what crosses between mosaic and that program is a JSON request file,
a JSON response file and progress lines on standard output.

**Two environments, because the fork cannot share one.** POLO ships under the
distribution name ``ultralytics``, so an environment holds upstream or the fork
and never both. They are located by two distinct variable pairs and run the same
program: :data:`ULTRALYTICS_ENV` for the tracker and pose inference,
:data:`POLO_ENV` for point inference. The fork ships the same ``yolo`` console
script as upstream, so **which variable is set is the only thing that tells them
apart** -- the ``$PATH`` rung of the ladder resolves to whichever is on the path
and cannot know which fork it found. A caller that needs the fork specifically
must check what the environment reported rather than trust the ladder.

This module is the part both the tracker and the inference ops need: the two
:class:`~mosaic.tracking.common.toolenv.ToolEnv` values, the failure classes, the
launcher, and the reader for the progress lines the runner writes. What is
specific to one caller -- the tracker configuration, the preflight refusals, the
request each op builds -- stays with that caller.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import ValidationError

from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.tracking.common.entry import INFLIGHT_REFRESH_SECONDS
from mosaic.tracking.common.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    subprocess_env,
    tool_invocation,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ProbeResponse,
    ProgressEvent,
)

if TYPE_CHECKING:
    from mosaic.core.pipeline.job import JobContext
    from mosaic.core.pipeline.markers import PhaseName

logger = logging.getLogger(__name__)

RunnerSubcommand: TypeAlias = Literal[
    "probe",
    "tracker-defaults",
    "track",
    "infer-pose",
    "infer-points",
    "train-pose",
    "train-points",
]
"""Every subcommand the runner program answers to.

Closed rather than a bare ``str`` because the launcher passes it straight into
argv: a mistyped name would reach the runner's own parser as a usage error from
a subprocess, where the checker can refuse it here.
"""

_PYTHON: Final = "python"
_YOLO_SCRIPT: Final = "yolo"
_RUNNER_SCRIPT: Final = "ultralytics_runner.py"

PROBE_DEADLINE_FLOOR_SECONDS: Final = 900.0
"""The least time a silent subcommand gets to answer, whatever the caller's bound is.

``idle_timeout`` bounds *silence*, which is the right unit once work is under
way: the runner prints a line per decoded batch, so a quiet stretch there means
hung. A probe prints nothing at all between spawn and answer, so the same number
would be a deadline on a cold torch import and a checkpoint load off a network
mount -- work proceeding exactly as intended. A user who shortens the window so a
hung tool dies quickly must not thereby put a stopwatch on loading a model, so
the probe gets the caller's value or this floor, whichever is longer.
``tracker-defaults`` is silent for the same stretch -- the torch import is most of
what it costs -- and takes the same floor.

No such floor belongs on ``track`` or on the two inference subcommands: those
report per batch, and raising their bound would blunt the one thing supervising
them.
"""

ULTRALYTICS_BOOTSTRAP: Final = (
    "Build it with 'uv sync --python 3.12' in "
    "src/mosaic/tracking/external/ultralytics-env/, then point "
    "MOSAIC_ULTRALYTICS_CONDA_ENV at a conda environment holding it or "
    "MOSAIC_ULTRALYTICS_BIN at that environment's 'yolo' script. See "
    "src/mosaic/tracking/external/README.md."
)

POLO_BOOTSTRAP: Final = (
    "Build it with 'uv sync --python 3.12' in "
    "src/mosaic/tracking/external/polo-env/, then point "
    "MOSAIC_POLO_CONDA_ENV at a conda environment holding it or MOSAIC_POLO_BIN "
    "at that environment's 'yolo' script. It has to be its own environment: POLO "
    "ships under the distribution name 'ultralytics', so it and upstream cannot "
    "occupy one. See src/mosaic/tracking/external/README.md."
)


class UltralyticsNotFoundError(ToolNotFoundError):
    """The Ultralytics environment, its Ultralytics, or ``conda``, is not there."""

    default_message = (
        "The Ultralytics environment was not found: no 'yolo' console script on "
        "$PATH, and neither MOSAIC_ULTRALYTICS_CONDA_ENV nor "
        f"MOSAIC_ULTRALYTICS_BIN names one. {ULTRALYTICS_BOOTSTRAP}"
    )


class PoloNotFoundError(ToolNotFoundError):
    """The POLO environment, its POLO, or ``conda``, is not there.

    Its own class rather than a reuse of the Ultralytics one because the two are
    separate installations a user configures separately, and a message naming the
    wrong pair of variables sends them to the wrong directory.
    """

    default_message = (
        "The POLO environment was not found: no 'yolo' console script on $PATH, "
        "and neither MOSAIC_POLO_CONDA_ENV nor MOSAIC_POLO_BIN names one. "
        f"{POLO_BOOTSTRAP}"
    )


class UltralyticsError(ToolExitError):
    """The runner exited with a non-zero return code in the Ultralytics environment.

    The inherited ``head`` of six is what both rungs of the location ladder need.
    On the direct rung the argv is ``<python> <runner> <subcommand> --request
    <path> --out <path>``, where three tokens already reach the subcommand; on
    the conda rung ``conda run --no-capture-output -n <env> <python>`` is six
    tokens before any of that, so a shorter head elides the environment name, the
    runner and the subcommand together and leaves a message that names no
    operation at all. TREx and SLEAP take the same six for the same reason.
    """

    tool_name = "Ultralytics"


class PoloError(ToolExitError):
    """The runner exited with a non-zero return code in the POLO environment.

    Named apart from :class:`UltralyticsError` so a failure says which of the two
    environments ran, which is the first thing to establish when point inference
    fails: the two hold the same distribution under the same name.
    """

    tool_name = "POLO"


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

# Identical in shape, and deliberately not the same value. The fork installs the
# same two console scripts under the same distribution name, so nothing about a
# resolved path distinguishes the two environments -- only which variable named
# it does. Sharing one `ToolEnv` would mean one variable pointing at one of them
# and the other silently unreachable.
POLO_ENV: Final = ToolEnv(
    tool="POLO",
    conda_env_var="MOSAIC_POLO_CONDA_ENV",
    bin_var="MOSAIC_POLO_BIN",
    bin_mode="sibling",
    not_found=PoloNotFoundError,
    locator=_YOLO_SCRIPT,
)


class ModelLoadError(ValueError):
    """The environment resolved, and could not load the weights it was given."""


def refuse_unloadable_model(probe: ProbeResponse, model_path: str) -> None:
    """Refuse weights the environment could not load, naming the likely reason.

    Shared by tracking and by both inference ops, because the failure is a
    property of the environment and the checkpoint rather than of what was going
    to be done with them. POLO pickles its weights under a class upstream does
    not define, so an upstream build fails inside ``torch.load`` before the task
    the checkpoint declares can be read -- which is why the refusal that already
    routes ``locate`` weights to ``infer-points`` never used to be reachable.

    The fork paragraph is added on :attr:`ProbeResponse.has_locate` alone rather
    than by reading the traceback for a class name. What is certainly true is that
    these weights did not load *and* this build is not the fork, and that pair is
    worth telling a user whatever the exception was; matching on the text would be
    a guess that goes quietly wrong the first time either project renames
    something.
    """
    if not probe.model_load_error:
        return
    message = (
        f"{model_path} could not be loaded by the {probe.ultralytics_version} "
        f"install in this environment:\n  {probe.model_load_error}"
    )
    if not probe.has_locate:
        message += (
            "\nThis environment is upstream Ultralytics, not the POLO fork. A "
            "point-detection checkpoint pickles classes only the fork defines, so "
            "one cannot be loaded here at all -- run it with 'mosaic run --kind "
            f"infer-points' against a POLO environment. {POLO_BOOTSTRAP}"
        )
    raise ModelLoadError(message)


def runner_script() -> Path:
    """Where the program mosaic spawns lives, as an absolute path.

    Resolved from the package rather than joined out of strings, so a source
    checkout, an editable install and a wheel all name the same file. Importing
    the package is not importing the runner *module*: ``runner/__init__.py`` is
    a docstring, so nothing here pulls Ultralytics into this process, which is
    the whole point of the separation.

    One program for both environments, which is why ``runner/`` is a sibling of
    the two environment directories rather than inside either.
    """
    from mosaic.tracking.external import runner

    return Path(runner.__file__).parent / _RUNNER_SCRIPT


def runner_invocation(
    env: ToolEnv,
    *,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch *env*'s ``python``, as an argv prefix.

    The shared five-step ladder (:func:`~mosaic.tracking.common.toolenv.tool_invocation`)
    applied to one of the two environments above.
    """
    return tool_invocation(
        env,
        executable=_PYTHON,
        conda_env=conda_env,
        bin_path=bin_path,
    )


def run_runner(
    env: ToolEnv,
    failure: type[ToolExitError],
    subcommand: RunnerSubcommand,
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
    """Run one runner subcommand in *env* and return (stdout, stderr).

    *failure* is the exception raised on a non-zero exit, and travels beside
    *env* rather than hanging off it: which environment ran and what a failure in
    it is called are two facts, and ``ToolEnv`` is shared with three trackers
    that each name their own.

    *cancel_check*, when supplied, is polled while the runner works; if it fires,
    the whole process group is killed and
    :class:`mosaic.core.pipeline.subprocess_util.ProcessCancelled` propagates.

    Raises:
        ToolExitError: The subclass given as *failure*, when the runner exited
            with a non-zero return code.
    """
    cmd = [
        *runner_invocation(env, conda_env=conda_env, bin_path=bin_path),
        str(runner_script()),
        subcommand,
        "--request",
        str(request_path),
        "--out",
        str(response_path),
    ]
    # The same head the failure message uses, read off the exception class so the
    # log line and the error cannot come to elide at different points.
    head = failure.head
    logger.info(
        "Running: %s", " ".join(cmd[:head]) + (" ..." if len(cmd) > head else "")
    )

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
    )
    if returncode != 0:
        raise failure(cmd, returncode, stdout, stderr)
    return stdout, stderr


def reported_progress(line: str) -> ProgressEvent | None:
    """The ``progress`` event *line* carries, or ``None`` for anything else.

    Tolerant on purpose. This runs on the subprocess reader thread, where the
    lines are whatever the child wrote -- a torn line, a warning Ultralytics'
    own logger put on standard output, a future event kind this release does not
    know -- and none of those is a reason to raise where raising would be
    swallowed anyway. ``started`` is filtered out here rather than reported as
    ``0/0``, being liveness rather than position.
    """
    try:
        event = ProgressEvent.model_validate_json(line)
    except ValidationError:
        return None
    return event if event.event == "progress" else None


def progress_activity(
    ctx: JobContext,
    key: str,
    phase: PhaseName,
    liveness: Callable[[str], None],
) -> Callable[[str], None]:
    """Report the runner's per-batch position, keeping *liveness* intact.

    The runner counts frames per batch and writes them; without this nothing
    reads them, and the position an in-process run used to show disappears with
    it. *liveness* -- :func:`~mosaic.tracking.common.entry.phase_activity` --
    still sees every line, because what proves the phase alive is that the child
    spoke at all.

    Throttled on the same interval the claim and the heartbeat are, and for the
    same reason: a batch is a fraction of a second on short clips, and
    ``ctx.progress`` is the run-log when no callback was injected, so reporting
    per line would write one JSONL record per batch for the length of a video.
    """
    last_report = [0.0]

    def on_line(line: str) -> None:
        liveness(line)
        event = reported_progress(line)
        if event is None:
            return
        now = time.monotonic()
        if now - last_report[0] < INFLIGHT_REFRESH_SECONDS:
            return
        last_report[0] = now
        ctx.progress.on_phase(phase, f"{key} {event.done}/{event.total}")

    return on_line


__all__ = [
    "POLO_BOOTSTRAP",
    "POLO_ENV",
    "PROBE_DEADLINE_FLOOR_SECONDS",
    "ULTRALYTICS_BOOTSTRAP",
    "ULTRALYTICS_ENV",
    "PoloError",
    "PoloNotFoundError",
    "ModelLoadError",
    "RunnerSubcommand",
    "UltralyticsError",
    "UltralyticsNotFoundError",
    "progress_activity",
    "refuse_unloadable_model",
    "reported_progress",
    "run_runner",
    "runner_invocation",
    "runner_script",
]
