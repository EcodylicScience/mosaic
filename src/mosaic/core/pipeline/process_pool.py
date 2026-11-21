"""The process pool whose workers end with the process that built it.

A caller killed by SIGKILL runs no cleanup. Each spawned worker holds both ends of
the executor's queue pipes, so a worker never reads end of file on its call queue
and stays alive, holding its memory, after the caller is gone.
:func:`caller_guarded_process_pool` starts each worker with a thread that ends the
worker once the caller is gone.

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
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext

__all__ = ["caller_guarded_process_pool"]


def _exit_when_caller_ends() -> None:
    """Start the thread that ends this worker once the caller is gone.

    The executor runs this in each worker as its initializer. The thread ends the
    worker with ``os._exit``, which runs no cleanup. A worker ended while it writes
    through ``write_parquet_atomic`` or ``atomic_write`` leaves a temporary file that
    no reader takes as output.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, name="caller-guard", daemon=True).start()


def caller_guarded_process_pool(
    max_workers: int, mp_context: BaseContext | None = None
) -> ProcessPoolExecutor:
    """Return a process pool whose workers exit when the caller's process ends.

    The start method is the caller's choice: *mp_context*, else the interpreter
    default.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_exit_when_caller_ends,
    )
