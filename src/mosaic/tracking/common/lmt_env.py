"""Locating the lmt-analysis environment, whose runner program rebuilds Live Mouse Tracker events.

lmt-analysis is GPL-3.0, and ``src/`` stays relicensable. Mosaic therefore never
imports it. It runs as a separate program: ``tracking/external/runner/
lmt_rebuild_runner.py`` runs in an environment that the user builds from
``tracking/external/lmt-env/``, beside a checkout of lmt-analysis that the user
clones. What crosses between mosaic and the runner is a command line, a JSON
report file, an exit status and the script's own output.

The environment is an interpreter only, so :data:`LMT_ENV` is located the way
:data:`~mosaic.behavior.feature_library.vame_feature.VAME_ENV` is: a conda
environment, a Python binary, or the ``.venv`` of the in-repository directory.
The checkout is :data:`LMT_ANALYSIS_DIR_VAR` when that variable is set, and
``lmt-env/lmt-analysis/`` otherwise. The in-repository rungs work from a source
checkout of mosaic only.

**The ladder resolves placement, never identity.** Which interpreter ran and
where the checkout lives are properties of the machine, so neither reaches a
``run_id``. The commit of lmt-analysis does, and the runner verifies it.

**What each exit of the runner means here.** :func:`probe_lmt_analysis` and
:func:`run_lmt_rebuild` read the runner's report file and map its exit status:

==== ================================== ==================================
Exit Meaning                            :func:`probe_lmt_analysis`
==== ================================== ==================================
0    done                               the report, after it is checked
                                        against the request
2    usage                              :class:`LmtRunnerDefect`
3    environment                        ``ToolUnavailable``, cause
                                        ``environment``
4    an unknown builder                 ``ValueError``
other any other status                  :class:`LmtRunnerDefect`
==== ================================== ==================================

:func:`run_lmt_rebuild` answers exits 3, 4 and 5 (``process`` raised) with
:class:`LmtRebuildError`, which fails the entry, and exit 2 and any other
status with :class:`LmtRunnerDefect`, which fails the run. A report that is
missing, is not JSON, fails :class:`LmtRunnerReport`, or states a protocol
other than 1 is a :class:`LmtRunnerDefect` whatever the exit status, and no
``ValidationError`` leaves this module.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Final

from pydantic import ConfigDict, ValidationError

from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.core.pipeline.toolenv import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    ToolEnv,
    ToolExitError,
    ToolInvocation,
    ToolNotFoundError,
    interpreter_refusal,
    locate_tool,
    subprocess_env,
)
from mosaic.core.strict_model import StrictModel, located
from mosaic.user_paths import user_path

__all__ = [
    "LMT_ANALYSIS_DIR_VAR",
    "LMT_CONDA_ENV_VAR",
    "LMT_ENV",
    "LMT_ENV_DIR",
    "LMT_PYTHON_VAR",
    "LMT_README",
    "LMT_RUNNER_PROTOCOL",
    "LMT_TOOL_NAME",
    "LmtNotFoundError",
    "LmtRebuildError",
    "LmtRunnerDefect",
    "LmtRunnerReport",
    "lmt_analysis_checkout",
    "probe_lmt_analysis",
    "read_runner_report",
    "run_lmt_rebuild",
    "runner_script",
]

LMT_TOOL_NAME: Final = "lmt-analysis"
"""The name that a :class:`~mosaic.core.pipeline.refusal.ToolUnavailable` gives the tool."""

LMT_ENV_DIR: Final = Path(__file__).parents[1] / "external" / "lmt-env"
"""The directory that holds the environment's definition and its README.

A source checkout builds the environment's ``.venv`` here, and clones
lmt-analysis into ``lmt-analysis/`` beside it.
"""

LMT_ANALYSIS_DIR_VAR: Final = "MOSAIC_LMT_ANALYSIS_DIR"
"""Names the checkout of lmt-analysis, in place of ``lmt-env/lmt-analysis/``."""

LMT_CONDA_ENV_VAR: Final = "MOSAIC_LMT_CONDA_ENV"
LMT_PYTHON_VAR: Final = "MOSAIC_LMT_PYTHON"

LMT_README: Final = "src/mosaic/tracking/external/lmt-env/README.md"
"""The environment's README, by its path in a source checkout of mosaic.

The directory is not package data, so a refusal names the path in the
repository and not a path under the installed package.
"""


class LmtNotFoundError(ToolNotFoundError):
    """The lmt-analysis environment's interpreter, or ``conda``, is not there."""

    default_message = (
        "The lmt-analysis environment was not found. lmt-analysis is GPL-3.0 and "
        "runs in a separate environment that you build. Follow "
        f"{LMT_README}, then point {LMT_PYTHON_VAR} at that "
        f"environment's python or {LMT_CONDA_ENV_VAR} at a conda environment, "
        "and name the checkout of lmt-analysis with "
        f"{LMT_ANALYSIS_DIR_VAR} unless it is in lmt-env/lmt-analysis/. See "
        "docs/installation.md#tools-that-run-in-their-own-environment."
    )


# A $PATH lookup cannot single out this interpreter. The fifth rung reads the one
# under lmt-env/.venv.
LMT_ENV: Final = ToolEnv(
    tool=LMT_TOOL_NAME,
    conda_env_var=LMT_CONDA_ENV_VAR,
    bin_var=LMT_PYTHON_VAR,
    bin_mode="direct",
    not_found=LmtNotFoundError,
    repo_venv=LMT_ENV_DIR,
)
"""Where the lmt-analysis environment's interpreter lives."""


LMT_RUNNER_PROTOCOL: Final = 1
"""The report protocol that this module reads, and the runner writes."""

_RUNNER_SCRIPT: Final = "lmt_rebuild_runner.py"
_INSTALLATION: Final = "docs/installation.md#tools-that-run-in-their-own-environment"

_MODIFIED_FILES: Final = "modified tracked files"
"""The words of the runner's report when the checkout has edited tracked files."""

_EXIT_ENVIRONMENT: Final = 3
_EXIT_INPUT: Final = 4
_EXIT_PROCESS: Final = 5


def lmt_analysis_checkout() -> Path:
    """Where the checkout of lmt-analysis is.

    :data:`LMT_ANALYSIS_DIR_VAR` when it is set, and ``lmt-analysis/`` beside
    the environment's definition otherwise. The directory is placement, and no
    identifier reads it.
    """
    named = os.environ.get(LMT_ANALYSIS_DIR_VAR, "").strip()
    return user_path(named) if named else LMT_ENV_DIR / "lmt-analysis"


def runner_script() -> Path:
    """Where the runner program lives, as an absolute path.

    Resolved from the package, as the Ultralytics runner is. Importing the
    package does not import the runner module.
    """
    from mosaic.tracking.external import runner

    return Path(runner.__file__).parent / _RUNNER_SCRIPT


class _LmtRunnerExit(ToolExitError):
    """A stop of the runner, with the sentence that says why before the generic text.

    Attributes:
        reason: The sentence, empty for none.
    """

    tool_name: ClassVar[str] = "The lmt-analysis runner"

    def __init__(
        self,
        cmd: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        *,
        reason: str = "",
    ) -> None:
        super().__init__(cmd, returncode, stdout, stderr)
        self.reason: str = reason
        if reason:
            self.args = (f"{reason} {self.args[0]}",)


class LmtRebuildError(_LmtRunnerExit):
    """The runner stopped one rebuild: the checkout moved, the input, or ``process``.

    It fails the entry that was being rebuilt. ``reason`` is the report's own
    sentence.
    """


class LmtRunnerDefect(_LmtRunnerExit):
    """The runner broke its contract, so the run fails and no entry is blamed.

    The exit status is one the contract does not give the answer for, or the
    report is missing, unreadable, of another protocol or incomplete.
    ``reason`` says which.
    """


class LmtRunnerReport(StrictModel):
    """The JSON report the runner writes, for every exit but 2.

    A report holds the keys known when the runner stopped, so each is optional
    but the protocol. ``commit`` is the checkout's HEAD once ``git rev-parse``
    ran, and ``available_builders`` the script's builders once it imported. A
    report of exit 0 holds ``builders``, ``max_frame``, ``script_max_frame``,
    ``window_frames``, ``python`` and ``packages``, and one of ``rebuild`` also
    ``last_frame`` and ``seconds``. An unknown key fails the read.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    runner_protocol: int
    error: str | None = None
    commit: str | None = None
    available_builders: tuple[str, ...] | None = None
    builders: tuple[str, ...] | None = None
    max_frame: int | None = None
    script_max_frame: int | None = None
    window_frames: int | None = None
    last_frame: int | None = None
    seconds: float | None = None
    python: str | None = None
    packages: dict[str, str] | None = None


def _completed(
    cmd: Sequence[str], stdout: str, stderr: str, returncode: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(cmd), returncode, stdout, stderr)


def _defect(
    completed: subprocess.CompletedProcess[str],
    reason: str,
    cause: BaseException | None = None,
) -> LmtRunnerDefect:
    """The defect of the finished command *completed*, chained from *cause*."""
    command = [str(part) for part in completed.args]
    defect = LmtRunnerDefect(
        command, completed.returncode, completed.stdout, completed.stderr, reason=reason
    )
    defect.__cause__ = cause
    return defect


def read_runner_report(
    path: Path, completed: subprocess.CompletedProcess[str]
) -> LmtRunnerReport:
    """Read the report that the finished command *completed* wrote to *path*.

    Args:
        path: The report file.
        completed: The finished runner, whose argv, exit status and output a
            refusal carries.

    Raises:
        LmtRunnerDefect: The file is missing, is not JSON, fails
            :class:`LmtRunnerReport`, states a protocol other than
            :data:`LMT_RUNNER_PROTOCOL`, or, for exit 0, lacks a key that every
            finished run holds.
    """
    try:
        content = path.read_bytes()
    except OSError as unreadable:
        raise _defect(
            completed, f"The runner wrote no report at {path}.", unreadable
        ) from unreadable
    try:
        report = LmtRunnerReport.model_validate_json(content)
    except ValidationError as invalid:
        raise _defect(
            completed,
            f"The report at {path} is not a valid runner report: {located(invalid)}.",
            invalid,
        ) from invalid
    if report.runner_protocol != LMT_RUNNER_PROTOCOL:
        raise _defect(
            completed,
            f"The report at {path} is of protocol {report.runner_protocol}, and "
            f"this mosaic reads protocol {LMT_RUNNER_PROTOCOL}.",
        )
    if completed.returncode == 0:
        held: dict[str, object] = {
            "builders": report.builders,
            "max_frame": report.max_frame,
            "script_max_frame": report.script_max_frame,
            "window_frames": report.window_frames,
            "python": report.python,
            "packages": report.packages,
        }
        absent = [name for name, value in held.items() if value is None]
        if absent:
            raise _defect(
                completed,
                f"The report at {path} of a finished run lacks {', '.join(absent)}.",
            )
    return report


def _interpreter() -> ToolInvocation:
    """Locate the environment's interpreter.

    Raises:
        ToolUnavailable: The ladder locates nothing, or the rung found no file.
    """
    try:
        return locate_tool(LMT_ENV, executable="python")
    except ToolNotFoundError as missing:
        raise ToolUnavailable(LMT_TOOL_NAME, "environment", str(missing)) from missing


def _unstartable(invocation: ToolInvocation, failure: OSError) -> ToolUnavailable:
    """The refusal of an interpreter that exists and does not start."""
    message = interpreter_refusal(
        LMT_ENV, invocation.argv[-1], invocation.source, "could not be started"
    )
    refusal = ToolUnavailable(LMT_TOOL_NAME, "environment", message)
    refusal.__cause__ = failure
    return refusal


def _command(
    invocation: ToolInvocation,
    subcommand: str,
    commit: str,
    builders: Sequence[str] | None,
    report_path: Path,
    *,
    database: Path | None = None,
) -> list[str]:
    command = [
        *invocation.argv,
        str(runner_script()),
        subcommand,
        "--checkout",
        str(lmt_analysis_checkout()),
        "--commit",
        commit,
        "--report",
        str(report_path),
    ]
    for builder in builders or ():
        command += ["--builder", builder]
    if database is not None:
        command += ["--database", str(database)]
    return command


def _environment_refusal(
    report: LmtRunnerReport, commit: str, completed: subprocess.CompletedProcess[str]
) -> ToolUnavailable:
    """The refusal of an exit 3: what the checkout or its imports lack."""
    checkout = lmt_analysis_checkout()
    if report.error is None:
        raise _defect(completed, "The runner reported an exit without saying why.")
    if report.commit is not None and report.commit != commit:
        message = (
            f"The lmt-analysis checkout at {checkout} is at commit {report.commit}, "
            f"and this run asks for {commit}. Run 'git -C {checkout} checkout "
            f"{commit}'. See {LMT_README}."
        )
    elif _MODIFIED_FILES in report.error:
        message = (
            f"The lmt-analysis checkout at {checkout} cannot run: {report.error} "
            f"Run 'git -C {checkout} checkout {commit} -- .' to restore them. "
            f"See {LMT_README}."
        )
    else:
        message = (
            f"The lmt-analysis checkout at {checkout} cannot run: {report.error} "
            f"See {LMT_README}."
        )
    return ToolUnavailable(LMT_TOOL_NAME, "environment", message)


def _builder_refusal(
    report: LmtRunnerReport,
    builders: Sequence[str] | None,
    completed: subprocess.CompletedProcess[str],
) -> ValueError:
    """The refusal of an exit 4 of the probe: a builder the checkout lacks."""
    available = report.available_builders
    if available is None or report.error is None:
        raise _defect(
            completed, "The runner reported an unknown builder without naming them."
        )
    unknown = [name for name in builders or () if name not in available]
    named = ", ".join(unknown) if unknown else "the builders asked for"
    return ValueError(
        f"The lmt-analysis checkout at {lmt_analysis_checkout()} has no usable "
        f"builder {named}. Its builders are {', '.join(available)}. {report.error}"
    )


def probe_lmt_analysis(
    commit: str,
    builders: Sequence[str] | None,
    report_path: Path,
    *,
    cancel_check: Callable[[], bool] | None,
) -> LmtRunnerReport:
    """Ask the runner what the checkout holds, and verify it against the request.

    The runner imports the script and resolves *builders*, which are the
    script's own without *builders*. Nothing is rebuilt.

    Args:
        commit: The commit that the checkout has to be at.
        builders: The builders asked for, or ``None`` for the script's own.
        report_path: Where the runner writes its report.
        cancel_check: Polled while the runner works.

    Returns:
        The report of an exit 0, whose ``commit`` is *commit* and whose
        ``builders`` are *builders* when they are given.

    Raises:
        ToolUnavailable: The interpreter is not located or does not start, the
            checkout directory does not exist, the commit differs, tracked
            files are modified, or the script does not import. The cause is
            ``environment``.
        ValueError: *builders* names a builder that the checkout lacks.
        LmtRunnerDefect: The runner broke its contract.
        ProcessCancelled: *cancel_check* returned true.
    """
    invocation = _interpreter()
    checkout = lmt_analysis_checkout()
    if not checkout.is_dir():
        raise ToolUnavailable(
            LMT_TOOL_NAME,
            "environment",
            f"The lmt-analysis checkout {checkout} does not exist. Clone "
            f"lmt-analysis there as {LMT_README} describes, or name an existing "
            f"checkout with {LMT_ANALYSIS_DIR_VAR}. See {_INSTALLATION}.",
        )
    command = _command(invocation, "probe", commit, builders, report_path)
    # A report left by an earlier run would be read when this one writes none.
    report_path.unlink(missing_ok=True)
    try:
        stdout, stderr, returncode = run_supervised(
            command,
            env=subprocess_env(),
            cancel_check=cancel_check,
            idle_timeout=PROBE_DEADLINE_FLOOR_SECONDS,
        )
    except OSError as failure:
        raise _unstartable(invocation, failure) from failure
    completed = _completed(command, stdout, stderr, returncode)

    if returncode == 0 or returncode in (_EXIT_ENVIRONMENT, _EXIT_INPUT):
        report = read_runner_report(report_path, completed)
        if returncode == _EXIT_ENVIRONMENT:
            raise _environment_refusal(report, commit, completed)
        if returncode == _EXIT_INPUT:
            raise _builder_refusal(report, builders, completed)
        if report.commit != commit:
            raise _defect(
                completed,
                f"The runner finished at commit {report.commit}, and the probe "
                f"asked for {commit}.",
            )
        if builders is not None and report.builders != tuple(builders):
            raise _defect(
                completed,
                f"The runner resolved the builders {report.builders}, and the "
                f"probe asked for {tuple(builders)}.",
            )
        return report
    raise _defect(completed, _undefined_exit(returncode))


def _undefined_exit(returncode: int) -> str:
    if returncode == 2:
        return "The runner rejected its command line."
    return f"The runner exited with status {returncode}, which its contract does not define."


def run_lmt_rebuild(
    database: Path,
    commit: str,
    builders: Sequence[str] | None,
    *,
    report_path: Path,
    log_path: Path,
    cancel_check: Callable[[], bool] | None,
    on_line: Callable[[str], None],
) -> LmtRunnerReport:
    """Rebuild the events of *database* in place, in the runner's environment.

    The runner has no idle timeout: a rebuild is silent for as long as the
    script takes. Each line of either stream is appended to *log_path* and
    passed to *on_line*, from the threads that read the streams.

    Args:
        database: The copy to rebuild.
        commit: The commit that the checkout has to be at.
        builders: The builders to rebuild, or ``None`` for the script's own.
        report_path: Where the runner writes its report.
        log_path: The file that receives the runner's output.
        cancel_check: Polled while the runner works.
        on_line: Called with each line of output.

    Returns:
        The report of an exit 0.

    Raises:
        ToolUnavailable: The interpreter is not located or does not start.
        LmtRebuildError: The runner exited 3, 4 or 5.
        LmtRunnerDefect: The runner broke its contract.
        ProcessCancelled: *cancel_check* returned true.
    """
    invocation = _interpreter()
    command = _command(
        invocation, "rebuild", commit, builders, report_path, database=database
    )
    report_path.unlink(missing_ok=True)
    lock = threading.Lock()
    with log_path.open("a", encoding="utf-8") as log:

        def on_activity(line: str) -> None:
            with lock:
                _ = log.write(line)
                log.flush()
            on_line(line)

        try:
            stdout, stderr, returncode = run_supervised(
                command,
                env=subprocess_env(),
                cancel_check=cancel_check,
                on_activity=on_activity,
            )
        except OSError as failure:
            raise _unstartable(invocation, failure) from failure
    completed = _completed(command, stdout, stderr, returncode)

    if returncode == 0 or returncode in (_EXIT_ENVIRONMENT, _EXIT_INPUT, _EXIT_PROCESS):
        report = read_runner_report(report_path, completed)
        if returncode == 0:
            return report
        if report.error is None:
            raise _defect(completed, "The runner stopped without saying why.")
        raise LmtRebuildError(command, returncode, stdout, stderr, reason=report.error)
    raise _defect(completed, _undefined_exit(returncode))
