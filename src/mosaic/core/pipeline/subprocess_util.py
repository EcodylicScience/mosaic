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
        Wall-clock limit; on expiry the group is terminated and
        ``subprocess.TimeoutExpired`` is raised (matching ``subprocess.run``).
    on_output:
        Optional per-stdout-line callback (e.g. to parse progress).
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

    def _reader(
        stream: IO[str], sink: list[str], echo: Callable[[str], None] | None
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                sink.append(line)
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
    while True:
        try:
            proc.wait(timeout=poll_interval)
            break
        except subprocess.TimeoutExpired:
            pass
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        if timeout is not None and (time.monotonic() - start) > timeout:
            timed_out = True
            break

    if cancelled or timed_out:
        terminate_group(proc)

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    stdout = "".join(out_chunks)
    stderr = "".join(err_chunks)

    if cancelled:
        raise ProcessCancelled(argv)
    if timed_out:
        raise subprocess.TimeoutExpired(
            list(argv), timeout or 0.0, output=stdout, stderr=stderr
        )
    return stdout, stderr, proc.returncode
