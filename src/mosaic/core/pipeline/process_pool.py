"""The process pool whose workers end with the process that built it, or with its break.

A caller killed by SIGKILL runs no cleanup. Each spawned worker holds both ends of
the executor's queue pipes, so a worker never reads end of file on its call queue
and stays alive, holding its memory, after the caller is gone.
:func:`caller_guarded_process_pool` starts each worker with a thread that ends the
worker once the caller is gone, and then runs the caller's own initializer.

The thread waits on the parent sentinel of :func:`multiprocessing.parent_process`,
a pipe whose write end the caller holds under every start method. The kernel closes
it when the caller dies, so the thread reads no clock and no process id, and a
process that reuses the caller's id cannot be mistaken for it. A worker's parent
process is the forkserver under ``forkserver``, which is why the thread does not
compare process ids.

The guard does not use ``PR_SET_PDEATHSIG``, which fires when the thread that
started the worker ends, and the executor starts workers from ``submit``, which
runs in the caller's thread. A caller that ends its thread and keeps its process
would lose its workers.

A worker that ignores SIGTERM (:func:`ignore_stop_signals`) survives the executor's
end of a broken pool, so each :class:`CallerGuardedPool` kills its live workers
once it breaks.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from types import FrameType
from typing import Final, override

__all__ = [
    "CallerGuardedPool",
    "caller_guarded_process_pool",
    "ignore_stop_signals",
    "kill_guarded_workers",
]

_WATCH_SECONDS: Final = 0.05

_POOLS: Final[weakref.WeakSet[CallerGuardedPool]] = weakref.WeakSet()
"""The pools this process built, for :func:`kill_guarded_workers`."""

_SHUT_DOWN: Final[set[CallerGuardedPool]] = set()
"""The pools shut down whose watcher still runs, kept alive until it returns."""


def _exit_when_caller_ends() -> None:
    """Start the thread that ends this worker once the caller is gone.

    The thread ends the worker with ``os._exit``, which runs no cleanup. A worker
    ended while it writes through ``write_parquet_atomic`` or ``atomic_write`` leaves
    a temporary file that no reader takes as output.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, name="caller-guard", daemon=True).start()


def _guard_then_initialize(
    initializer: Callable[..., object] | None, initargs: tuple[object, ...]
) -> None:
    """The executor's initializer in each worker: the guard first, then *initializer*."""
    _exit_when_caller_ends()
    if initializer is not None:
        _ = initializer(*initargs)


class CallerGuardedPool(ProcessPoolExecutor):
    """A process pool that kills its live workers once it breaks.

    On a break the executor's manager thread fails the pending calls, sends SIGTERM
    to each worker and joins it, holding the executor's shutdown lock. A worker that
    ignores SIGTERM keeps that join waiting, and with it every later ``submit`` and
    ``shutdown``, which take the lock, and the interpreter's exit, which joins the
    manager thread. A kill from a thread that waits for the lock never runs.

    The pool therefore starts a watcher thread after its first ``submit`` returns,
    which polls the executor's broken flag and calls :meth:`kill_workers` once the
    flag is set, whichever thread of the caller sees the break, or none. The
    executor forks every worker of a ``fork`` pool in that first ``submit``, so no
    fork of this pool runs while its watcher lives. The watcher takes no lock of the
    executor. It returns after the kill, or once the pool was shut down or
    collected and every worker it started has exited, so a break during a shutdown
    still finds it running. A pool shut down stays alive until its watcher returns,
    however its caller drops it. Before its shutdown the watcher holds it weakly, so
    that a pool dropped without a shutdown is collected and its workers end, as the
    executor's own manager thread arranges, and a break after that is not watched.

    The watcher also wakes the manager thread when a worker has exited while the
    pool is neither broken nor shut down. CPython's ``submit`` wakes the manager
    before it starts the worker it adds, and the manager lists the workers'
    sentinels before it waits. A worker that this ``submit`` started is then missing
    from the wait, and its death before any later result or submission would leave
    the pool unbroken and its calls waiting forever. Woken, the manager lists the
    sentinels again and breaks the pool.
    """

    def __init__(
        self,
        max_workers: int,
        mp_context: BaseContext | None,
        initializer: Callable[..., object] | None,
        initargs: tuple[object, ...],
    ) -> None:
        super().__init__(
            max_workers,
            mp_context,
            initializer=_guard_then_initialize,
            initargs=(initializer, initargs),
        )
        # The executor drops its own reference at shutdown and keeps the map.
        self._started: Final = self._processes
        self._watcher_lock: Final = threading.Lock()
        self._watcher: threading.Thread | None = None
        self._watcher_done: bool = False
        self._shutdown_called: bool = False
        _POOLS.add(self)

    @override
    def submit[**P, T](
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future = super().submit(fn, *args, **kwargs)
        with self._watcher_lock:
            if self._watcher is None:
                self._watcher = threading.Thread(
                    target=CallerGuardedPool._watch,
                    args=(weakref.ref(self), self._started),
                    name="caller-guarded-pool-watcher",
                    daemon=True,
                )
                self._watcher.start()
        return future

    @override
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        # The executor's own flag does not serve, because a break sets it too.
        self._shutdown_called = True
        with self._watcher_lock:
            watcher = self._watcher
            if watcher is not None and not self._watcher_done:
                _SHUT_DOWN.add(self)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait and watcher is not None:
            # Every worker has exited, so the watcher returns within its poll.
            watcher.join()

    def kill_workers(self) -> None:
        """Send SIGKILL to each worker this pool started that is still alive.

        A worker is killed through its own process object, which signals it only
        while it has not been reaped, so a process that reused an exited worker's id
        is never reached. ``is_alive`` reaps an exited worker first, which the
        manager thread's join then reads as done. A worker that cannot be checked or
        killed does not stop the kill of the others. The call takes no lock, and
        works after a shutdown.
        """
        for process in tuple(self._started.values()):
            try:
                if process.is_alive():
                    process.kill()
            except Exception:
                continue

    def _wake_manager(self) -> None:
        wakeup = self._executor_manager_thread_wakeup
        # A shutdown clears the attribute, and the wakeup itself does nothing once
        # the manager has closed it.
        try:
            wakeup.wakeup()
        except AttributeError:
            return

    @staticmethod
    def _watch(
        pool_reference: weakref.ref[CallerGuardedPool],
        started: Mapping[int, BaseProcess],
    ) -> None:
        try:
            while True:
                pool = pool_reference()
                if pool is not None and pool._broken:
                    pool.kill_workers()
                    return
                ending = pool is None or pool._shutdown_called
                if pool is not None and not ending and _any_exited(started):
                    pool._wake_manager()
                del pool
                if ending and _all_exited(started):
                    return
                time.sleep(_WATCH_SECONDS)
        finally:
            pool = pool_reference()
            if pool is not None:
                with pool._watcher_lock:
                    pool._watcher_done = True
                    _SHUT_DOWN.discard(pool)


def _exited_count(started: Mapping[int, BaseProcess]) -> tuple[int, int]:
    """How many of the *started* workers have exited, and how many there are."""
    sentinels = [process.sentinel for process in tuple(started.values())]
    return len(connection.wait(sentinels, timeout=0)), len(sentinels)


def _any_exited(started: Mapping[int, BaseProcess]) -> bool:
    return _exited_count(started)[0] > 0


def _all_exited(started: Mapping[int, BaseProcess]) -> bool:
    exited, count = _exited_count(started)
    return exited == count


def caller_guarded_process_pool(
    max_workers: int,
    mp_context: BaseContext | None = None,
    *,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[object, ...] = (),
) -> CallerGuardedPool:
    """Return a process pool whose workers exit when the caller's process ends.

    Each worker starts the caller guard, then calls ``initializer(*initargs)``, so
    an initializer that hangs or raises still leaves a worker that ends with the
    caller. The start method is the caller's choice: *mp_context*, else the
    interpreter default.
    """
    return CallerGuardedPool(max_workers, mp_context, initializer, initargs)


def kill_guarded_workers() -> None:
    """Kill the live workers of every pool this process built.

    The caller is the handler of a second stop signal to the process, so the pools
    are listed first: a pool may be built or collected while the handler runs.
    """
    for pool in list(_POOLS):
        pool.kill_workers()


def _do_nothing(_signum: int, _frame: FrameType | None) -> None:
    return None


def ignore_stop_signals() -> None:
    """Make SIGTERM and SIGINT do nothing in this process.

    A worker built so finishes its call when a stop signal reaches its process
    group, and its caller decides what the signal ends. A handler that does nothing
    is installed instead of ``SIG_IGN``, because an ignored disposition survives
    ``exec`` and a program the worker starts would ignore the signals too, whereas
    a handler resets to the default.

    Python's handler makes blocking system calls return ``EINTR``, so each handler
    is followed by ``signal.siginterrupt(sig, False)``, which restarts the calls
    that ``SA_RESTART`` covers, such as ``read`` and ``write``. ``select``,
    ``poll``, ``nanosleep`` and the other calls it does not restart still return
    ``EINTR``, which Python's own calls retry and a C extension may not. Windows has
    no ``siginterrupt``, and the call is skipped there.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        _ = signal.signal(sig, _do_nothing)
        if sys.platform != "win32":
            signal.siginterrupt(sig, False)
