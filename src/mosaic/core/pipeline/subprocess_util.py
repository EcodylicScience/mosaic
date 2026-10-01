"""Killable, orphan-safe subprocess supervision.

A single helper for spawning external tools (TREx today; the Layer-2
``mosaic run`` executor later) so that:

* the child runs in its **own process group** -- a cooperative cancel can
  ``SIGTERM``\\ -then-``SIGKILL`` the *whole* subtree (TREx relaunches itself, so
  killing just the direct child is not enough);
* an orphaned child **self-terminates** when its parent dies
  (Linux ``PR_SET_PDEATHSIG``);
* output is drained on reader threads, so we can poll a cancel predicate while
  the child runs without deadlocking on a full pipe.

This is the parent-side supervision pattern that ``kpms`` implements ad hoc;
factoring it here lets TREx and the future executor share it.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from typing import IO, Callable, ClassVar, Final, Sequence

_PR_SET_PDEATHSIG = 1  # from <sys/prctl.h>

_COMMAND_HEAD: Final = 6
"""How many argv tokens the message of a cancel or a timeout quotes."""


def command_summary(cmd: Sequence[str], head: int | None = None) -> str:
    """*cmd* as a message or a log line quotes it.

    Any token that follows a ``-c`` token is rendered as ``<program>``, whatever
    the executable. For a Python interpreter it is an entire program, and
    printing it helps nobody. That token is third from a bin placement and later
    under conda, so no count of tokens elides it. Then the first *head* tokens
    are kept, and ``...`` marks the rest.

    Args:
        cmd: The argv as run.
        head: How many tokens to keep. ``None`` keeps every token, which is what
            a log line wants for a command it may be asked to reproduce.

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
    """Raised by :func:`run_supervised` when a cancel predicate fired."""

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)
        super().__init__(
            f"subprocess cancelled: {command_summary(self.argv, _COMMAND_HEAD)}"
        )


class _LimitExpired(subprocess.TimeoutExpired):
    """A limit of :func:`run_supervised` expired.

    The message summarizes the argv with :func:`command_summary`, because the
    argv's own text would quote a program passed with ``-c`` and the message
    reaches a run-log's ``error_json``. ``cmd`` still holds the whole argv.
    """

    expired: ClassVar[str] = ""
    """What happened, between the command and the limit in seconds."""

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
    """Raised by :func:`run_supervised` when the run outlasted ``timeout`` seconds.

    A :class:`subprocess.TimeoutExpired`, which ``subprocess.run`` raises for the
    same limit, so an ``except subprocess.TimeoutExpired`` catches it.
    """

    expired: ClassVar[str] = "timed out after"


class IdleTimeoutExpired(_LimitExpired):
    """Raised by :func:`run_supervised` when the child produced no output for
    ``idle_timeout`` seconds -- an inactivity (hang) kill.

    Subclasses :class:`subprocess.TimeoutExpired` so an existing ``except
    subprocess.TimeoutExpired`` still catches it, while a caller that wants to
    tell a hang apart from the absolute wall-clock ceiling
    (:class:`WallClockTimeoutExpired`) or a cancel (:class:`ProcessCancelled`)
    can match this type. ``self.timeout`` carries the idle window, not the
    elapsed runtime.
    """

    expired: ClassVar[str] = "produced no output for"


def set_pdeathsig() -> None:
    """Ask the kernel to signal this process when its parent dies (Linux only).

    Intended as a subprocess ``preexec_fn``. No-op on non-Linux platforms.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


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
    """Run *argv* in its own killable process group and return (stdout, stderr, rc).

    Parameters
    ----------
    argv:
        Command and arguments.
    env:
        Full environment for the child (``None`` inherits the parent's).
    cancel_check:
        Polled every ``poll_interval`` seconds; when it returns True the group is
        terminated and :class:`ProcessCancelled` is raised.
    timeout:
        Absolute wall-clock ceiling. ``None`` (the default) imposes no total
        limit; on expiry the group is terminated and
        :class:`WallClockTimeoutExpired`, a ``subprocess.TimeoutExpired``, is
        raised.
    idle_timeout:
        Inactivity limit. When set, the group is terminated once the child has
        produced *no* output -- on stdout **or** stderr -- for this many
        seconds, and :class:`IdleTimeoutExpired` is raised. This is the right
        bound for a tool whose runtime is unpredictable but which prints
        progress while healthy (e.g. TREx): a live long run keeps resetting it,
        a wedged one trips it. ``None`` disables it.
    on_output:
        Optional per-stdout-line callback, for *reading* what the child says --
        parsing a progress event, an epoch, a JSON response line. Stdout only,
        because a parser is written against one stream's format.
    on_activity:
        Optional per-line callback for every line on **either** stream, for
        callers that only need to know the child spoke. That is the same signal
        ``idle_timeout`` measures, so a caller hanging liveness on this one --
        a claim refresh, a heartbeat -- keeps it alive for exactly as long as
        the watchdog considers the child alive. A tool whose progress bar goes
        to stderr is invisible to ``on_output`` and would otherwise be read as
        silent.
    """
    popen_kwargs: dict[str, object] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True  # setsid -> own process group
        popen_kwargs["preexec_fn"] = set_pdeathsig

    # Decoded leniently: a tool's decoder library can print a byte that is not
    # UTF-8, and a strict decode would end the reader thread on it, leaving the
    # rest of that stream unread.
    proc = subprocess.Popen(
        [str(a) for a in argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        env=env,
        **popen_kwargs,  # type: ignore[arg-type]
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
