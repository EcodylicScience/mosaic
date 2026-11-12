"""The pool that computes one run's entries, in turn or several at once.

A run hands the pool one call per entry and hears the outcome of each through two
callbacks. The pool owns what every such run shares: the executor (spawned processes
or threads), the limit on calls in flight, the host's memory floor, the memory
measurement a run records, and the two ways a host ends a run, a worker it kills and a
floor reached with nothing in flight to free memory.

One worker computes each call in the caller's thread, so a run reads the same whatever
its worker count. The caller decides what a call is, checks for a cancel before it
submits, and writes what a result holds. The pool does none of that.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import pickle
import traceback
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .memory_watch import MemoryWatch
from .parallelism import (
    MemoryFloorReached,
    ParallelMode,
    ParallelSettings,
    WorkerProcessDied,
    memory_use_text,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "EntryFailure",
    "EntryPool",
    "OrderedResults",
    "ResourceReporter",
    "dataset_ref",
    "worker_dataset",
]


class ResourceReporter(Protocol):
    """Where a pool records its workers and the memory measured so far.

    :meth:`mosaic.core.pipeline.job.JobContext.resources` is one.
    """

    def __call__(
        self,
        *,
        workers: int,
        parallel_mode: str,
        peak_rss_bytes: int,
        min_available_bytes: int,
        total_memory_bytes: int,
    ) -> None: ...


class EntryPool[Tag, Result]:
    """Compute calls with a bounded number in flight, and hand back each outcome.

    Args:
        workers: How many calls are computed at once. 1 computes each call inside
            :meth:`submit`, with no executor.
        mode: ``process`` spawns interpreters, and a call and its arguments must
            pickle. ``thread`` shares the caller's process.
        min_free_memory_fraction: The share of the host's memory that must stay
            available while several workers run. 0 turns the check off, and one
            worker never checks.
        on_result: Called as ``on_result(tag, result)`` for each call that
            returned, in the caller's thread.
        on_error: Called as ``on_error(tag, exc)`` for each call that raised. It
            may raise to end the run, and the calls not yet started are then
            cancelled.
        on_resources: Where the workers and the memory measured are recorded.
    """

    def __init__(
        self,
        *,
        workers: int,
        mode: ParallelMode,
        on_result: Callable[[Tag, Result], None],
        on_error: Callable[[Tag, Exception], None],
        min_free_memory_fraction: float = 0.0,
        on_resources: ResourceReporter | None = None,
    ) -> None:
        self.workers: int = max(1, workers)
        self.mode: ParallelMode = mode
        self._floor: float = min_free_memory_fraction
        self._on_result: Callable[[Tag, Result], None] = on_result
        self._on_error: Callable[[Tag, Exception], None] = on_error
        self._on_resources: ResourceReporter | None = on_resources
        self._watch: MemoryWatch = MemoryWatch()
        self._pending: dict[Future[Result], Tag] = {}
        self._executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        if self.workers > 1:
            if mode == "process":
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=mp.get_context("spawn")
                )
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.workers)

    @classmethod
    def for_settings(
        cls,
        settings: ParallelSettings,
        *,
        on_result: Callable[[Tag, Result], None],
        on_error: Callable[[Tag, Exception], None],
        on_resources: ResourceReporter | None = None,
    ) -> EntryPool[Tag, Result]:
        """Build the pool that *settings* describe."""
        return cls(
            workers=settings.workers,
            mode=settings.mode,
            min_free_memory_fraction=settings.min_free_memory_fraction,
            on_result=on_result,
            on_error=on_error,
            on_resources=on_resources,
        )

    def submit[*Args](
        self, tag: Tag, fn: Callable[[*Args], Result], *args: *Args
    ) -> None:
        """Compute ``fn(*args)`` and report its outcome under *tag*.

        With several workers the call waits for a free worker and for the memory
        floor, and the outcome of this or an earlier call arrives here or in a later
        call of this pool. With one worker the outcome has arrived on return.

        Raises:
            WorkerProcessDied: A process worker ended without returning.
            MemoryFloorReached: Memory is under the floor with no call in flight.
        """
        executor = self._executor
        if executor is None:
            try:
                result = fn(*args)
            except Exception as exc:
                self._on_error(tag, exc)
                return
            self._on_result(tag, result)
            self.report()
            return
        while len(self._pending) >= self.workers:
            self._drain()
        self._wait_for_memory()
        try:
            self._pending[executor.submit(fn, *args)] = tag
        except BrokenProcessPool as exc:
            self.report(final=True)
            raise WorkerProcessDied(self._memory_use()) from exc
        self.report()

    def finish(self) -> None:
        """Wait for every call in flight, then shut the executor down."""
        while self._pending:
            self._drain()
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def settle(self) -> None:
        """Wait for the calls in flight and hand back each outcome, whatever it was.

        For a caller that is ending on an exception and keeps what the calls in
        flight finish: a table a worker wrote is indexed, and is not left for a
        rerun to skip as already current. An exception that a callback raises while
        the calls settle is dropped, because the caller is already raising.
        """
        while self._pending:
            done, _ = wait(self._pending)
            for future in done:
                tag = self._pending.pop(future)
                if future.cancelled():
                    continue
                try:
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._on_error(tag, exc)
                    else:
                        self._on_result(tag, result)
                except Exception:
                    continue

    def close(self) -> None:
        """Shut the executor down without waiting, and record the final measurement.

        Call it on every exit. A pool left running holds its workers' memory, and a
        broken one is of no further use. Closing a pool that finished does nothing
        more than record the measurement.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.report(final=True)

    def report(self, *, final: bool = False) -> None:
        """Record the workers and the memory measured so far.

        Throttled to the sampler's interval unless *final*. The final report is
        written on every exit, a failure included, because the reader of a failed
        run needs the memory it reached.
        """
        measured = (
            self._watch.sample() is not None if final else self._watch.sample_if_due()
        )
        if not measured or self._on_resources is None:
            return
        self._on_resources(
            workers=self.workers,
            parallel_mode=self.mode if self.workers > 1 else "",
            peak_rss_bytes=self._watch.peak_rss_bytes,
            min_available_bytes=self._watch.min_available_bytes,
            total_memory_bytes=self._watch.total_bytes,
        )

    def _memory_use(self) -> str:
        return memory_use_text(
            workers=self.workers,
            mode=self.mode,
            peak_rss_bytes=self._watch.peak_rss_bytes,
            total_bytes=self._watch.total_bytes,
        )

    def _drain(self) -> None:
        """Wait for one call to finish and hand back every call that has."""
        done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
        for future in done:
            tag = self._pending.pop(future)
            try:
                result = future.result()
            except BrokenProcessPool as exc:
                # Every other pending future of a broken pool fails the same way, so
                # this is the run's failure and not one entry's.
                self.report(final=True)
                raise WorkerProcessDied(self._memory_use()) from exc
            except Exception as exc:
                try:
                    self._on_error(tag, exc)
                except BaseException:
                    # A future already running cannot be cancelled. The rest never
                    # start.
                    for waiting in self._pending:
                        _ = waiting.cancel()
                    raise
                continue
            self._on_result(tag, result)
        self.report()

    def _wait_for_memory(self) -> None:
        """Hold dispatch while the host's available memory is under the floor.

        Waits for in-flight calls, whose tables free memory as they finish. With
        none in flight it collects garbage once, and raises when the host is still under the floor.
        """
        if self._floor <= 0.0:
            return
        collected = False
        while True:
            free = self._watch.free_fraction()
            if free is None or free >= self._floor:
                return
            if not self._pending:
                if not collected:
                    _ = gc.collect()
                    collected = True
                    continue
                self.report(final=True)
                raise MemoryFloorReached(
                    self._memory_use(), free_fraction=free, floor=self._floor
                )
            self._drain()


class OrderedResults[Item]:
    """Hand items to a callback in index order, whatever order they arrive in.

    Calls that finish in the order of their workers would write an index in an
    order that depends on the worker count. Items numbered from 0 as they were
    submitted are emitted as soon as every item before them has been, which is the
    order one worker emits them in, and each item is durable once its turn comes.

    Args:
        emit: Called with each item, in index order. An exception it raises
            propagates from :meth:`put`, and the item is not emitted again.
    """

    def __init__(self, emit: Callable[[Item], None]) -> None:
        self._emit: Callable[[Item], None] = emit
        self._next: int = 0
        self._held: dict[int, Item] = {}

    def put(self, index: int, item: Item) -> None:
        """Take the item at *index*, and emit it and the held items that follow it."""
        self._held[index] = item
        while self._next in self._held:
            ready = self._held.pop(self._next)
            self._next += 1
            self._emit(ready)

    def held(self) -> list[Item]:
        """Return the items still waiting for an earlier one, in index order."""
        return [self._held[index] for index in sorted(self._held)]


@dataclass(frozen=True)
class EntryFailure:
    """One entry's failure, carried from a worker to the caller as a value.

    A worker that handles a failure itself returns what it had done before the
    failure beside it, which an exception cannot carry. The error and its traceback
    text survive the trip from a worker process, where the traceback object does
    not.

    Attributes:
        key: What the failure is recorded under, such as a path or an entry key.
        error: The exception, or a :class:`RuntimeError` naming its type and
            message when the exception cannot be pickled.
        trace: The traceback text as the worker formatted it.
    """

    key: str
    error: Exception
    trace: str

    @classmethod
    def capture(cls, key: str, error: Exception) -> EntryFailure:
        """Capture *error*. Call it from inside the ``except`` block that caught it."""
        try:
            _ = pickle.loads(pickle.dumps(error))
        except Exception:
            error = RuntimeError(f"{type(error).__name__}: {error}")
        return cls(key=key, error=error, trace=traceback.format_exc())

    def record(self, entry_failed: Callable[[str, BaseException], None]) -> None:
        """Call ``entry_failed(key, error)`` with the error raised and handled.

        Recorders such as :meth:`mosaic.core.pipeline.job.JobContext.entry_failed`
        read the traceback of the exception being handled, so the error is raised
        again for them. An error that crossed a process boundary has no traceback
        of its own, and the worker's text is attached to it as a note.
        """
        remote = self.error.__traceback__ is None
        try:
            raise self.error
        except Exception as exc:
            if remote:
                exc.add_note(f"Raised in a worker:\n{self.trace}")
            entry_failed(self.key, exc)


_OPENED: dict[str, Dataset] = {}
"""The datasets a worker process opened, by manifest path."""


def dataset_ref(ds: Dataset, mode: ParallelMode) -> Dataset | str:
    """Return what a call hands a worker so that it can use *ds*.

    A thread shares the dataset. A process receives the manifest path and opens
    its own, because a dataset is not meant to be pickled.
    """
    return str(ds.manifest_path) if mode == "process" else ds


def worker_dataset(ref: Dataset | str) -> Dataset:
    """Return the dataset that :func:`dataset_ref` named, opened once per process."""
    if not isinstance(ref, str):
        return ref
    opened = _OPENED.get(ref)
    if opened is None:
        from mosaic.core.dataset import open_dataset

        opened = open_dataset(ref)
        _OPENED[ref] = opened
    return opened
