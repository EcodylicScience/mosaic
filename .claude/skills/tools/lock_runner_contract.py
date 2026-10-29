"""Pass messages between the lock runner and the command that it runs.

The messages report commands that did not start, runs handed back for the
fallback of remote, and how remote started the run.
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NoReturn

from lossless_text import print_error, substitution_text
from quiet_paths import nonempty, present, read_quietly, remove_quietly, write_quietly

NOT_RUN_VARIABLE = "HEAVY_TASK_NOTRUN_FILE"
HANDOFF_VARIABLE = "HEAVY_TASK_HANDOFF_FILE"
FALLBACK_VARIABLE = "REMOTE_TASK_FALLBACK"
# Passes remote's --ignore-untracked to its fallback; commands other than
# mutate ignore it.
IGNORE_UNTRACKED_VARIABLE = "REMOTE_TASK_IGNORE_UNTRACKED"
# Task clones contain this file at their root.
TASK_CLONE_MARKER = ".remote-task-workdir"
# The lock runner's status when --wait times out.
WAIT_TIMEOUT_STATUS = 75
# The status of a command that did not start for a reason that may pass, such
# as a lock not taken in time or a task machine out of reach (EX_TEMPFAIL).
EXIT_NOT_RUN = 75
# The lock runner writes these beside a --log file: the exit code, and the
# abort marker of a command that did not start.
EXIT_MARKER_SUFFIX = ".exit"
NOT_RUN_MARKER_SUFFIX = ".notrun"

StartCase = Literal["task-clone", "fallback", "bare"]
# The lock runner's name in messages: heavy's own, or remote's when remote
# runs it.
RunnerName = Literal["heavy-task", "remote-task"]
RUNNER_NAME_VARIABLE = "HEAVY_TASK_NAME"


def runner_name(environment: Mapping[str, str]) -> RunnerName:
    """Return the name that HEAVY_TASK_NAME sets, heavy-task for a value
    outside RunnerName."""
    return "remote-task" if environment.get(RUNNER_NAME_VARIABLE) == "remote-task" else "heavy-task"


def wait_timeout_line(name: RunnerName) -> str:
    """Return the closing line of a --wait that timed out, under the lock
    runner's `name`."""
    return f"{name}: exit-status={WAIT_TIMEOUT_STATUS} (wait timeout; run not finished)"


def not_run_status_line(name: RunnerName, status: int, why: str) -> str:
    """Return the closing line of a command that did not start, under the
    lock runner's `name`."""
    return f"{name}: exit-status={status} ({why}; command not run)"


def exit_not_run(name: RunnerName, message: str, status: int, why: str) -> NoReturn:
    """Report that the command did not start, then exit with `status`."""
    print_error(f"FAILED {name}: {message}; command NOT run")
    print_error(not_run_status_line(name, status, why))
    sys.exit(status)


def mint_not_run_marker(path: Path) -> None:
    """Create the empty not-run marker on a best-effort basis.

    Pids are reused, and the lock directory outlives every run. Runs killed by
    SIGKILL leave markers that the next run drawing the same pid takes for its
    marker. The removal clears a marker left read-only, which truncation
    cannot. This function does not print a diagnostic. The marker is minted
    before the command runs, and the command's output must come first, with no
    line ahead of it that reads as the run failing.
    """
    remove_quietly(path)
    _ = write_quietly(path, "")


def mark_not_run(path: Path) -> None:
    """Record in `path` that the command never started.

    Silent when it cannot. This runs on an abort path, where the diagnostic
    about the abort must stay the only one.
    """
    _ = write_quietly(path, "notrun\n")


def command_not_run(path: Path) -> bool:
    return nonempty(path)


def not_run_marker(environment: Mapping[str, str]) -> Path | None:
    """Return the marker that the enclosing lock runner reads, None outside
    one."""
    marker = environment.get(NOT_RUN_VARIABLE)
    return Path(marker) if marker else None


def report_not_run(environment: Mapping[str, str]) -> None:
    """Tell the enclosing lock runner, if any, the command never started."""
    marker = not_run_marker(environment)
    if marker is not None:
        mark_not_run(marker)


def hand_back(path: Path, reason: str, diagnostic: bytes) -> bool:
    """Hand the run back for the fallback: the reason on the first line, then
    the diagnostic. False when not even the reason could be written."""
    try:
        _ = path.write_bytes(os.fsencode(reason) + b"\n")
    except OSError:
        return False
    try:
        with path.open("ab") as appended:
            _ = appended.write(diagnostic)
    except OSError:
        pass
    return True


def handed_back(environment: Mapping[str, str]) -> bool:
    """Return whether the command handed its run back to the asking outer
    phase."""
    path = environment.get(HANDOFF_VARIABLE)
    if not path:
        return False
    return nonempty(Path(path))


def handoff_reason(path: Path) -> tuple[str, str] | None:
    """Return the reason and diagnostic that a run was handed back with, else
    None."""
    content = (read_quietly(path) or b"").replace(b"\0", b"")
    if not content:
        return None
    reason, _, diagnostic = content.partition(b"\n")
    return os.fsdecode(reason), substitution_text(diagnostic)


def start_case(toplevel: Path | None, environment: Mapping[str, str]) -> StartCase:
    """Return how remote started the run: in a marked task clone, as its
    fallback under a lock runner, or not at all. A task clone wins over the
    fallback variables."""
    if toplevel is not None and present((toplevel / TASK_CLONE_MARKER).is_file):
        return "task-clone"
    marker = not_run_marker(environment)
    if environment.get(FALLBACK_VARIABLE) and marker is not None and present(marker.is_file):
        return "fallback"
    return "bare"
