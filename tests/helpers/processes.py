"""Child processes that prove, by a file they write, whether they are still running.

A process that survives a kill can still let its parent report success. The tests
of process supervision therefore start an appender, a program that appends to a
file every 0.05 s, and assert that the file stops growing. A reaped child of the
test process is checked by its process id instead.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest


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
