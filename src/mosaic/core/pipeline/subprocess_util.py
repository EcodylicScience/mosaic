"""Killable, orphan-safe supervision of external tool processes.

Two shapes spawn a child with :func:`supervised_popen_kwargs`:

* :func:`run_supervised` runs a command to its end and returns its output;
* :class:`LineChild` keeps a child running and exchanges lines with it on its
  standard input and output.

On POSIX the child leads a new process group, and :func:`terminate_group`
signals the whole group. TREx relaunches itself, and a signal to the direct
child alone leaves the relaunched process running. On Windows the direct child
alone is signaled, and a runner ends when its standard input closes. On Linux
a child receives SIGTERM when the thread that spawned it exits, and so when its
parent process dies (``PR_SET_PDEATHSIG``). Output is read on threads. The
caller polls a cancel predicate while the child runs, and a full pipe cannot
block the child.
"""

from __future__ import annotations

import ctypes
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, ClassVar, Final, TypedDict

_PR_SET_PDEATHSIG = 1  # from <sys/prctl.h>

_COMMAND_HEAD: Final = 6
"""How many argv tokens the message of a cancel or a timeout quotes."""

_LIBRARY_PATH_VAR: Final = "LD_LIBRARY_PATH"

_PYTHON_SEARCH_VARS: Final = frozenset({"PYTHONPATH", "PYTHONHOME"})


def foreign_environment() -> dict[str, str]:
    """Return this process's environment for a program of another environment.

    The copy of ``os.environ`` omits ``PYTHONPATH``, ``PYTHONHOME``, and each
    ``LD_LIBRARY_PATH`` directory inside this interpreter's prefix.

    The two Python variables name this interpreter's sources, as
    ``LD_LIBRARY_PATH`` names its libraries. With them, a tool's interpreter
    imports this environment's packages, ``mosaic`` included.

    OpenCV's loader prepends its library directory to ``LD_LIBRARY_PATH`` when
    ``cv2`` is imported, for the child processes. A child from another
    environment then loads this environment's libraries before those of the
    child's environment. SLEAP's PySide6 failed to import against mosaic's Qt
    that way. The variable is removed when every directory is left out.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _PYTHON_SEARCH_VARS
    }
    inherited = environment.get(_LIBRARY_PATH_VAR)
    if inherited is None:
        return environment
    prefix = os.path.realpath(sys.prefix)
    kept = [
        entry
        for entry in inherited.split(os.pathsep)
        if entry and os.path.commonpath([prefix, os.path.realpath(entry)]) != prefix
    ]
    if kept:
        environment[_LIBRARY_PATH_VAR] = os.pathsep.join(kept)
    else:
        del environment[_LIBRARY_PATH_VAR]
    return environment


def command_summary(cmd: Sequence[str], head: int | None = None) -> str:
    """Return *cmd* as a message or a log line quotes it.

    Any token that follows a ``-c`` token is rendered as ``<program>``, whatever
    the executable. For a Python interpreter it is an entire program, and
    printing it helps nobody. That token is third when the interpreter is run
    directly and later under ``conda run``, so no fixed count of tokens elides
    it. Then the first *head* tokens are kept, and ``...`` marks the rest.

    Args:
        cmd: The argv as run.
        head: How many tokens to keep. ``None`` keeps every token, for a log
            line of a command that it may be asked to reproduce.

    Returns:
        The kept tokens joined by spaces, with ``<program>`` in place of each
        token after ``-c``, and a space and ``...`` appended when a token was
        dropped.
    """
    shown = [
        "<program>" if i > 0 and cmd[i - 1] == "-c" else str(token)
        for i, token in enumerate(cmd)
    ]
    kept = shown if head is None else shown[:head]
    return " ".join(kept) + (" ..." if len(kept) < len(shown) else "")


class ProcessCancelled(RuntimeError):
    """Raised when a cancel predicate returned True.

    :func:`run_supervised` and :meth:`LineChild.next_line` raise it after they
    terminate the child's process group.
    """

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)
        super().__init__(
            f"subprocess cancelled: {command_summary(self.argv, _COMMAND_HEAD)}"
        )


class _LimitExpired(subprocess.TimeoutExpired):
    """A limit of :func:`run_supervised` or :meth:`LineChild.next_line` expired.

    The message summarizes the argv with :func:`command_summary`, because the
    argv's own text would quote a program passed with ``-c`` and the message is
    recorded in a run-log's ``error_json``. ``cmd`` still holds the whole argv.
    """

    expired: ClassVar[str] = ""
    """The words of the message between the command and the limit in seconds."""

    def __init__(
        self,
        argv: Sequence[str],
        timeout: float,
        output: str | None = None,
        stderr: str | None = None,
    ) -> None:
        self.argv: list[str] = list(argv)
        super().__init__(self.argv, timeout, output=output, stderr=stderr)

    def __str__(self) -> str:
        summary = command_summary(self.argv, _COMMAND_HEAD)
        return f"Command '{summary}' {self.expired} {self.timeout} seconds"


class WallClockTimeoutExpired(_LimitExpired):
    """Raised when a wall-clock limit of ``timeout`` seconds expired.

    For :func:`run_supervised` the limit bounds the whole run. For
    :meth:`LineChild.next_line` it bounds the wait of one call for a line.
    Either caller terminates the child's process group first.

    It is a :class:`subprocess.TimeoutExpired`, as ``subprocess.run`` raises for
    its *timeout*, and an ``except subprocess.TimeoutExpired`` catches it.
    """

    expired: ClassVar[str] = "timed out after"


class IdleTimeoutExpired(_LimitExpired):
    """Raised when a child was inactive for ``idle_timeout`` seconds.

    Activity differs between the two callers. For :func:`run_supervised` it is a
    line on either stream. For :meth:`LineChild.next_line` it is a line on
    standard output or a :meth:`LineChild.send`. Standard error does not count
    there, and a child that writes only to standard error is killed as inactive.
    Either caller terminates the child's process group first.

    It is a :class:`subprocess.TimeoutExpired` too, and an ``except
    subprocess.TimeoutExpired`` catches it. A caller that tells a hang apart
    from a wall-clock limit (:class:`WallClockTimeoutExpired`) or a cancel
    (:class:`ProcessCancelled`) matches this type. ``timeout`` is the idle
    window.
    """

    expired: ClassVar[str] = "produced no output for"


def set_pdeathsig() -> None:
    """Ask the kernel to send this process SIGTERM when its parent thread exits.

    The parent thread is the thread that spawned this process. It exits at the
    latest when the parent process dies. Pass this function as a subprocess
    ``preexec_fn``. It does nothing on a platform other than Linux.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


class SupervisedPopenKwargs(TypedDict):
    """The :class:`subprocess.Popen` keywords of a supervised child."""

    start_new_session: bool
    preexec_fn: Callable[[], None] | None
    encoding: str
    errors: str


def supervised_popen_kwargs() -> SupervisedPopenKwargs:
    """Return the :class:`subprocess.Popen` keywords of a supervised child.

    On POSIX the child leads a new session and a new process group, which
    :func:`terminate_group` signals as a whole. On Windows the direct child alone
    is signaled, and a runner ends when its standard input closes. On Linux the
    child also receives SIGTERM when the thread that spawned it exits
    (:func:`set_pdeathsig`). Its pipes are text, decoded as UTF-8 with each
    invalid byte replaced. A reader thread then reads a stream to its end despite
    invalid UTF-8 in it, which a tool's decoder library can print.
    """
    return SupervisedPopenKwargs(
        start_new_session=sys.platform != "win32",
        preexec_fn=set_pdeathsig if sys.platform == "linux" else None,
        encoding="utf-8",
        errors="replace",
    )


def terminate_group(proc: "subprocess.Popen[str]", *, grace: float = 5.0) -> None:
    """SIGTERM the process group, escalating to SIGKILL after *grace* seconds."""
    if proc.poll() is not None:
        return

    def _signal_group(*, hard: bool) -> None:
        # signal.SIGKILL is undefined on Windows, so it is referenced only inside
        # the POSIX branch; Windows escalates through Popen.kill/terminate.
        try:
            if sys.platform != "win32":
                sig = signal.SIGKILL if hard else signal.SIGTERM
                os.killpg(os.getpgid(proc.pid), sig)
            elif hard:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    _signal_group(hard=False)
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(hard=True)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def run_supervised(
    argv: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    timeout: float | None = None,
    idle_timeout: float | None = None,
    poll_interval: float = 0.5,
    on_output: Callable[[str], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
) -> tuple[str, str, int]:
    """Run *argv* in a new, killable process group and return its output.

    Args:
        argv: The command and its arguments.
        env: The child's whole environment. ``None`` inherits the parent's.
        cancel_check: Asked every *poll_interval* seconds. When it returns
            True, the group is terminated.
        timeout: The absolute wall-clock limit in seconds, or ``None``, the
            default, to run without a total limit.
        idle_timeout: The inactivity limit in seconds, or ``None`` to disable
            it. The group is terminated once the child has not written a line to
            either stream for this long. It suits TREx and other tools that print
            progress while healthy and run for an unpredictable time. A live long
            run keeps restarting the window, and a wedged one exceeds it.
        poll_interval: The seconds between two checks of the cancel predicate
            and the limits.
        on_output: Called with each line of standard output, for reading what
            the child says: a progress event, an epoch, a JSON response line.
            Standard output only, because a parser is written against one
            stream's format.
        on_activity: Called with each line on either stream, for a caller that
            needs to know only that the child spoke. *idle_timeout* measures the
            same signal, and a liveness hook on it (a claim refresh, a
            heartbeat) stays alive as long as the watchdog considers the child
            alive. A progress bar on standard error is not passed to
            *on_output*.

    Returns:
        The child's standard output, its standard error and its exit status.

    Raises:
        ProcessCancelled: *cancel_check* returned True.
        WallClockTimeoutExpired: The run outlasted *timeout*. It is a
            :class:`subprocess.TimeoutExpired`, as ``subprocess.run`` raises.
        IdleTimeoutExpired: The child was silent for *idle_timeout*. It is a
            :class:`subprocess.TimeoutExpired` too.
    """
    proc = subprocess.Popen(
        [str(a) for a in argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        **supervised_popen_kwargs(),
    )

    out_chunks: list[str] = []
    err_chunks: list[str] = []

    # Last instant either stream produced a line, for the inactivity watchdog.
    # A one-element list is the mutation cell shared with the poll loop; the
    # lock guards the read/write across threads (the GIL makes the assignment
    # itself atomic, but pairing it with the loop's read keeps it explicit).
    activity_lock = threading.Lock()
    last_activity = [time.monotonic()]

    def _note_activity() -> None:
        with activity_lock:
            last_activity[0] = time.monotonic()

    def _reader(
        stream: IO[str], sink: list[str], echo: Callable[[str], None] | None
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                sink.append(line)
                _note_activity()
                for callback in (echo, on_activity):
                    if callback is None:
                        continue
                    # Swallowed on purpose: this is the reader thread, so a
                    # raise here reaches nobody and would end the drain, and an
                    # undrained pipe deadlocks a chatty child.
                    try:
                        callback(line)
                    except Exception:
                        pass
        finally:
            stream.close()

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, out_chunks, on_output), daemon=True
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, err_chunks, None), daemon=True
    )
    t_out.start()
    t_err.start()

    start = time.monotonic()
    cancelled = False
    timed_out = False
    idled_out = False
    while True:
        try:
            proc.wait(timeout=poll_interval)
            break
        except subprocess.TimeoutExpired:
            pass
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        now = time.monotonic()
        if timeout is not None and (now - start) > timeout:
            timed_out = True
            break
        if idle_timeout is not None:
            with activity_lock:
                idle = now - last_activity[0]
            if idle > idle_timeout:
                idled_out = True
                break

    if cancelled or timed_out or idled_out:
        terminate_group(proc)

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    stdout = "".join(out_chunks)
    stderr = "".join(err_chunks)

    if cancelled:
        raise ProcessCancelled(argv)
    if idled_out:
        raise IdleTimeoutExpired(
            argv, idle_timeout or 0.0, output=stdout, stderr=stderr
        )
    if timed_out:
        raise WallClockTimeoutExpired(
            argv, timeout or 0.0, output=stdout, stderr=stderr
        )
    return stdout, stderr, proc.returncode


_CANCEL_POLL_SECONDS: Final = 0.5
"""How often :meth:`LineChild.next_line` asks its cancel predicate while it waits."""


class LineChild:
    """Spawns a supervised child and exchanges lines with it on its standard streams.

    The child is spawned with :func:`supervised_popen_kwargs`. Its standard input
    and output are pipes, and its standard error is appended to *log_path*. A
    daemon thread queues each line of its output as it arrives. Call the methods
    from one thread.

    On Linux the child receives SIGTERM when the thread that constructed this
    object exits. Construct it on a thread that lives as long as the child is
    used.

    Args:
        argv: The command and its arguments.
        log_path: The file that the child's standard error is appended to. Its
            directory is created when absent.
        env: The child's whole environment. ``None`` inherits this process's.

    Raises:
        FileNotFoundError: The executable does not exist.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.argv: list[str] = [str(token) for token in argv]
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            self._process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=env,
                **supervised_popen_kwargs(),
            )
        stdin, stdout = self._process.stdin, self._process.stdout
        assert stdin is not None and stdout is not None, "both are pipes"
        self._stdin = stdin
        # Each line with the monotonic time it arrived, and None at the end.
        self._lines: queue.Queue[tuple[float, str] | None] = queue.Queue()
        self._last_activity = time.monotonic()
        self._ended = False
        self._closed = False
        self._pump = threading.Thread(
            target=self._pump_lines, args=(stdout,), daemon=True
        )
        self._pump.start()

    def _pump_lines(self, stream: IO[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                self._lines.put((time.monotonic(), line.removesuffix("\n")))
        finally:
            self._lines.put(None)
            stream.close()

    @property
    def alive(self) -> bool:
        """Whether the child is still running."""
        return self._process.poll() is None

    @property
    def exit_code(self) -> int | None:
        """The child's exit status, or ``None`` while it runs.

        A child ended by a signal has the negated signal number.
        """
        return self._process.poll()

    def send(self, line: str) -> None:
        """Write *line* and a newline to the child's standard input.

        A send restarts the idle window of :meth:`next_line`. A send to a child
        that has exited does not raise. :meth:`next_line` reports the exit by
        returning ``None`` at the end of the child's output.

        Raises:
            ValueError: :meth:`close` was called, or *line* contains a line
                break, which the child reads as the end of a line.
        """
        if self._closed:
            msg = "a line cannot be sent to a child after close()"
            raise ValueError(msg)
        if "\n" in line or "\r" in line:
            msg = "a line sent to a child cannot contain a line break"
            raise ValueError(msg)
        self._last_activity = time.monotonic()
        try:
            _ = self._stdin.write(line + "\n")
            self._stdin.flush()
        except BrokenPipeError:
            pass

    def next_line(
        self,
        timeout: float | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
        idle_timeout: float | None = None,
    ) -> str | None:
        """Return the child's next line of standard output, without its newline.

        A cancel and an expired limit each terminate the child's process group
        with :func:`terminate_group` before the exception is raised.

        Args:
            timeout: The most seconds this call waits, or ``None`` to wait
                without a limit.
            cancel_check: Asked when the call starts, and again every 0.5 s
                while no line arrives. A call after the output has ended
                returns ``None`` without asking it.
            idle_timeout: The most seconds since the child's last line or the
                last :meth:`send`, whichever is later, or ``None`` to disable
                the check.

        Returns:
            The line, or ``None`` once the child's output has ended. Every later
            call returns ``None`` too.

        Raises:
            ProcessCancelled: *cancel_check* returned True.
            WallClockTimeoutExpired: No line arrived within *timeout* seconds.
            IdleTimeoutExpired: The child did not write a line, and was not sent
                one, for *idle_timeout* seconds.
        """
        if self._ended:
            return None
        started = time.monotonic()
        while True:
            if cancel_check is not None and cancel_check():
                terminate_group(self._process)
                raise ProcessCancelled(self.argv)
            now = time.monotonic()
            waits = [] if cancel_check is None else [_CANCEL_POLL_SECONDS]
            if timeout is not None:
                waits.append(started + timeout - now)
            if idle_timeout is not None:
                waits.append(self._last_activity + idle_timeout - now)
            try:
                item = self._lines.get(timeout=max(0.0, min(waits)) if waits else None)
            except queue.Empty:
                self._raise_if_expired(started, timeout, idle_timeout)
                continue
            if item is None:
                self._ended = True
                return None
            arrived, line = item
            self._last_activity = max(self._last_activity, arrived)
            return line

    def _raise_if_expired(
        self, started: float, timeout: float | None, idle_timeout: float | None
    ) -> None:
        now = time.monotonic()
        if timeout is not None and now - started >= timeout:
            terminate_group(self._process)
            raise WallClockTimeoutExpired(self.argv, timeout)
        if idle_timeout is not None and now - self._last_activity >= idle_timeout:
            terminate_group(self._process)
            raise IdleTimeoutExpired(self.argv, idle_timeout)

    def close(self, grace: float = 5.0) -> None:
        """Close the child's standard input and wait for the child to exit.

        A child still running *grace* seconds later has its process group sent
        SIGTERM, and SIGKILL if it still runs *grace* seconds after that
        (:func:`terminate_group`). Once the child has exited, any process left
        in its group, such as one that the child started and did not wait for,
        is sent SIGKILL. A second call does nothing.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.close()
        except BrokenPipeError:
            pass
        try:
            _ = self._process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            terminate_group(self._process, grace=grace)
        # The child leads a process group, whose id is the child's pid.
        _kill_group_members(self._process.pid)
        self._pump.join(timeout=grace)


def _kill_group_members(group: int) -> None:
    """Send SIGKILL to the processes of the process group *group*, if any remain.

    A process group's id is not reused while the group has a member. The group of
    a child that has been reaped can therefore be signaled.
    """
    if sys.platform == "win32":
        return
    try:
        os.killpg(group, signal.SIGKILL)
    except OSError:
        pass
