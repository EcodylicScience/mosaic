"""Child processes that prove, by a file they write, whether they are still running.

A process that survives a kill can still let its parent report success. The tests
of process supervision therefore start an appender, a program that appends to a
file every 0.05 s, and assert that the file stops growing. A reaped child of the
test process is checked by its process id instead.

:class:`LineChildren` spawns the line children of one test, which the fixture
``line_children`` closes when the test ends. :func:`cancel_when` fires a cancel
token from another thread once a condition is true.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from mosaic.core.pipeline.job import CancelToken
from mosaic.core.pipeline.subprocess_util import LineChild


def appender_program(path: Path | str) -> str:
    """Return a Python program that appends to *path* every 0.05 s until it is killed.

    The program also prints ``tick`` on its standard output after each write.
    """
    return (
        "import time\n"
        f"handle = open({str(path)!r}, 'a')\n"
        "while True:\n"
        "    handle.write('x')\n"
        "    handle.flush()\n"
        "    print('tick', flush=True)\n"
        "    time.sleep(0.05)\n"
    )


def exited_pid() -> int:
    """Return the pid of a child process that has exited and been reaped."""
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    _ = finished.wait()
    return finished.pid


def wait_for(
    condition: Callable[[], bool],
    seconds: float,
    message: str = "the condition stayed false",
) -> None:
    """Poll *condition* every 0.05 s until it is True.

    The call fails with *message* when *condition* is still False after *seconds*.
    """
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(0.05)


def cancel_when(
    condition: Callable[[], bool],
    token: CancelToken,
    fired: list[float] | None = None,
    message: str = "the condition stayed false",
) -> threading.Thread:
    """Start a thread that fires *token* once *condition* is True.

    The thread polls *condition* as :func:`wait_for` does, and appends the time
    at which it fired the token to *fired*. It fails with *message* when 60 s pass
    before *condition* is True.
    """

    def cancel() -> None:
        wait_for(condition, 60.0, message)
        if fired is not None:
            fired.append(time.monotonic())
        token.cancel()

    thread = threading.Thread(target=cancel, daemon=True)
    thread.start()
    return thread


class LineChildren:
    """Spawns the line children of one test, each logging to a separate file.

    Attributes:
        directory: The directory of the log files.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._spawned: list[LineChild] = []

    def spawn(
        self, argv: Sequence[str], env: Mapping[str, str] | None = None
    ) -> LineChild:
        """Spawn *argv*, with its standard error appended to a new log file."""
        log_path = self.directory / f"child-{len(self._spawned)}.log"
        child = LineChild(argv, log_path=log_path, env=env)
        self._spawned.append(child)
        return child

    def close_all(self) -> None:
        """Close each child, allowing it 1 s to exit before each signal."""
        for child in self._spawned:
            child.close(grace=1.0)


def assert_stopped_writing(path: Path, message: str) -> None:
    """Fail with *message* when a process still appends to *path*.

    The check waits 0.3 s for a process that was just signaled to exit, then
    compares the size of *path* across 0.6 s, a dozen writes at the cadence of
    :func:`appender_program`.
    """
    time.sleep(0.3)
    settled = path.stat().st_size
    time.sleep(0.6)
    assert path.stat().st_size == settled, message


def assert_process_exited(
    pid: int, message: str = "the process is still running"
) -> None:
    """Fail with *message* unless the process *pid* is gone.

    *pid* names a reaped child of the test process. An unreaped child answers
    signal 0 after it exits.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    pytest.fail(message)
