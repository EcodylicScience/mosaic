"""Share names, state and git access between launch-gate and teardown-gate."""

import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from git_in_directory import git_text
from lexical_paths import temporary_directory
from program_messages import say

STATE_KEYS = ("repo", "worktree", "snapshot", "paths")
# The status of a gate command that refuses its arguments or its state before
# it acts.
REFUSED_STATUS = 2


def gate_file(token: str, kind: str) -> Path:
    """Return the gate's file of `kind`: its state, its worktree, or a permit
    record."""
    return temporary_directory() / f"review-prose-{token}.{kind}"


def gate_branch(token: str) -> str:
    return f"review-prose/{token}"


def peer_session(token: str, role: str) -> str:
    return f"review-prose-{token}-{role}"


def line_buffered_output() -> None:
    """Print each line ahead of the output of a command started after it."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)


class CommandFailed(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status: int = status


def require_success(status: int) -> None:
    """Raise CommandFailed for the status of a command that failed."""
    if status != 0:
        raise CommandFailed(status)


def checked(completed: subprocess.CompletedProcess[bytes]) -> subprocess.CompletedProcess[bytes]:
    require_success(completed.returncode)
    return completed


def required_git_text(directory: Path, arguments: list[str]) -> str:
    """Return the output of a git run that has to succeed."""
    status, text = git_text(directory, arguments)
    require_success(status)
    return text


def gate_status(program: str, run: Callable[[], int]) -> int:
    """Return the status of `run`: a failed command's own status, and 1 after
    reporting an OSError."""
    try:
        return run()
    except CommandFailed as failed:
        return failed.status
    except OSError as error:
        say(program, str(error))
        return 1
