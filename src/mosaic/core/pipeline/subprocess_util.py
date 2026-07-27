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
from typing import IO, Callable, Sequence

_PR_SET_PDEATHSIG = 1  # from <sys/prctl.h>


class ProcessCancelled(RuntimeError):
    """Raised by :func:`run_supervised` when a cancel predicate fired."""

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)
        super().__init__(f"subprocess cancelled: {' '.join(map(str, argv[:4]))} ...")


class IdleTimeoutExpired(subprocess.TimeoutExpired):
    """Raised by :func:`run_supervised` when the child produced no output for
    ``idle_timeout`` seconds -- an inactivity (hang) kill.

    Subclasses :class:`subprocess.TimeoutExpired` so an existing ``except
    subprocess.TimeoutExpired`` still catches it, while a caller that wants to
    tell a hang apart from the absolute wall-clock ceiling (a plain
    ``TimeoutExpired``) or a cancel (:class:`ProcessCancelled`) can match this
    type. ``self.timeout`` carries the idle window, not the elapsed runtime.
    """

    def __str__(self) -> str:
        return f"Command '{self.cmd}' produced no output for {self.timeout} seconds"


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

    def _signal_group(sig: int) -> None:
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), sig)
            elif sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(signal.SIGKILL)
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
        ``subprocess.TimeoutExpired`` is raised (matching ``subprocess.run``).
    idle_timeout:
        Inactivity limit. When set, the group is terminated once the child has
        produced *no* output -- on stdout **or** stderr -- for this many
        seconds, and :class:`IdleTimeoutExpired` is raised. This is the right
        bound for a tool whose runtime is unpredictable but which prints
        progress while healthy (e.g. TREx): a live long run keeps resetting it,
        a wedged one trips it. ``None`` disables it.
    on_output:
        Optional per-stdout-line callback (e.g. to parse progress). Output on
        either stream counts as activity for ``idle_timeout`` regardless of
        this callback.
    """
    popen_kwargs: dict[str, object] = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True  # setsid -> own process group
        popen_kwargs["preexec_fn"] = set_pdeathsig

    proc = subprocess.Popen(
        [str(a) for a in argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
                if echo is not None:
                    try:
                        echo(line)
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
            list(argv), idle_timeout or 0.0, output=stdout, stderr=stderr
        )
    if timed_out:
        raise subprocess.TimeoutExpired(
            list(argv), timeout or 0.0, output=stdout, stderr=stderr
        )
    return stdout, stderr, proc.returncode
