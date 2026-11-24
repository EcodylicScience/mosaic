"""Child processes that prove, by a file they write, whether they are still running.

A process that survives a kill can still let its parent report success. The tests
of process supervision therefore start an appender, a program that appends to a
file every 0.05 s, and assert that the file stops growing. A reaped child of the
test process is checked by its process id instead.

:class:`LineChildren` spawns the line children of one test, which the fixture
``line_children`` closes when the test ends. :func:`cancel_when` fires a cancel
token from another thread once a condition is true. :data:`SLEEPER_COMMAND` stands
in for a long step.

:func:`signal_group_when` runs a child in its own process group and signals the
group, or the child alone, once a condition is true.

:func:`process_alive` tells whether a process is running, and
:func:`count_process_tree_walks` counts the walks of the process tree that a test's
code makes.

:data:`CHILD_TIMEOUT_SECONDS` is the bound on every wait for a child process to
finish, and :func:`run_child` runs a command under it. :func:`read_line` reads one
line of a child's output under it. Every wait for a child takes a ``timeout``, this
constant unless the test needs a shorter one.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

import mosaic
from mosaic.core.pipeline.job import CancelToken
from mosaic.core.pipeline.subprocess_util import LineChild
from tests.helpers.paths import REPO_ROOT

CHILD_TIMEOUT_SECONDS: Final[float] = 120.0
"""The longest a test waits for a child process to finish.

A cold import of the library takes seconds on an idle machine, and several
times that on a loaded one. A child that outlasts the bound is hung, and its
test fails instead of holding the run. ``pytest-timeout`` is not a dependency,
so this bound is the only one.
"""

CLOCK_TICK_SECONDS: Final[float] = 0.01
"""The resolution of a ``/proc`` CPU time, which a figure compared across two reads may lose."""


def read_line[T: (str, bytes)](
    process: subprocess.Popen[T], timeout: float = CHILD_TIMEOUT_SECONDS
) -> T:
    """Read one line of *process*'s standard output, which the test piped.

    Args:
        process: A child started with ``stdout=subprocess.PIPE``.
        timeout: Seconds the child has to write the line.

    Returns:
        The line, with its newline, or an empty value when the child closed its
        output without writing one.

    Raises:
        AssertionError: The child wrote no line within *timeout*. The child is
            killed and reaped and its output closed first, and the message names
            the command and the timeout.
    """
    stream = process.stdout
    assert stream is not None, "the child was not started with stdout=subprocess.PIPE"
    lines: queue.Queue[T] = queue.Queue(maxsize=1)
    reader = threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True)
    reader.start()
    try:
        return lines.get(timeout=timeout)
    except queue.Empty:
        process.kill()
        _ = process.wait(timeout=CHILD_TIMEOUT_SECONDS)
        reader.join(timeout=CHILD_TIMEOUT_SECONDS)
        stream.close()
        message = f"a child process wrote no line within {timeout} s: {process.args!r}"
        raise AssertionError(message) from None


def run_child(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run *argv* to completion and return the process with its output as text.

    The exit status is the caller's to check.

    Args:
        argv: The command and its arguments.
        cwd: The working directory of the child, this process's by default.
        env: The environment of the child, this process's by default.
        timeout: Seconds the command has to finish. A longer run is killed.

    Raises:
        AssertionError: The command runs past *timeout*. The message names the
            command and the timeout.
    """
    try:
        return subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        message = f"a child process did not finish within {timeout} s: {list(argv)!r}"
        raise AssertionError(message) from expired


SLEEPER_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-c",
    "import sys, time; time.sleep(float(sys.argv[1]))",
)
"""A command that sleeps for the seconds its first argument names.

A command built on it ignores every argument after the first, and SIGTERM ends it
with the signal's default action.
"""


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
    _ = finished.wait(timeout=CHILD_TIMEOUT_SECONDS)
    return finished.pid


def process_alive(pid: int) -> bool:
    """Whether *pid* names a process that is running.

    A zombie that nobody has joined is not running, and neither is a pid that no
    process holds.
    """
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def count_process_tree_walks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record a 1 for each ``psutil.Process.children`` call in this process, from now on."""
    import psutil

    walks: list[int] = []
    children = psutil.Process.children

    def counting(self: psutil.Process, recursive: bool = False) -> list[psutil.Process]:
        walks.append(1)
        return children(self, recursive=recursive)

    monkeypatch.setattr(psutil.Process, "children", counting)
    return walks


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


@dataclass(frozen=True)
class SignalledGroup:
    """How a child that :func:`signal_group_when` signalled ended.

    Attributes:
        pid: The child's process id, which is also its process group's.
        status: The child's exit status, negative for the signal that ended it.
        signalled_at: The ``time.time()`` reading taken just before the signal.
        seconds: From the signal to the child's exit.
        errors: What the child wrote to standard error.
    """

    pid: int
    status: int
    signalled_at: float
    seconds: float
    errors: str


def signal_group_when(
    argv: Sequence[str],
    ready: Callable[[], bool],
    signum: signal.Signals,
    *,
    extra_env: Mapping[str, str],
    errors: Path,
    waiting: str,
    child_only: bool = False,
) -> SignalledGroup:
    """Run *argv* in its own process group, and signal the group once *ready* is true.

    The child runs in the repository's root with this process's environment and
    *extra_env*, and imports the ``mosaic`` that this process imported. Its
    standard error goes to *errors*. The wait for *ready*, and then for the child to
    exit, are each bounded by :data:`CHILD_TIMEOUT_SECONDS`, and the group is killed
    on the way out while the child lives. With *child_only* the signal goes to the
    child's process alone, and the rest of its group is the caller's to end.

    Args:
        argv: The command of the child.
        ready: True once the child is where the signal should find it.
        signum: The signal sent to the group.
        extra_env: Variables set in the child's environment.
        errors: The file the child's standard error is written to.
        waiting: What *ready* waits for, as a clause that follows "before" in a
            failure's message.
        child_only: Whether to signal the child alone instead of its group.

    Raises:
        AssertionError: The child exited before *ready* was true, with its
            standard error in the message, or *ready* stayed false, or the child
            outlived the wait after the signal.
    """
    source = str(Path(mosaic.__file__).parent.parent)
    inherited = os.environ.get("PYTHONPATH")
    path = source if not inherited else os.pathsep.join([source, inherited])
    environment = {**os.environ, **extra_env, "PYTHONPATH": path}
    with errors.open("w") as stream:
        child = subprocess.Popen(
            list(argv),
            env=environment,
            cwd=REPO_ROOT,
            stderr=stream,
            start_new_session=True,
        )

        def ready_or_exited() -> bool:
            if ready():
                return True
            status = child.poll()
            if status is not None:
                message = (
                    f"the child exited with status {status} before {waiting}:\n"
                    f"{errors.read_text()}"
                )
                raise AssertionError(message)
            return False

        try:
            timed_out = f"{CHILD_TIMEOUT_SECONDS} s passed before {waiting}"
            wait_for(ready_or_exited, CHILD_TIMEOUT_SECONDS, timed_out)
            signalled_at = time.time()
            signalled = time.monotonic()
            if child_only:
                os.kill(child.pid, signum)
            else:
                os.killpg(child.pid, signum)
            status = child.wait(timeout=CHILD_TIMEOUT_SECONDS)
            seconds = time.monotonic() - signalled
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                _ = child.wait(timeout=CHILD_TIMEOUT_SECONDS)
    return SignalledGroup(
        pid=child.pid,
        status=status,
        signalled_at=signalled_at,
        seconds=seconds,
        errors=errors.read_text(),
    )


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
