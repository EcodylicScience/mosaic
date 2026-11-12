"""Measures the memory and CPU that a run and its worker processes use, and what the limit leaves.

A run with several workers is where memory runs out: each process-mode worker is an
interpreter with its own copy of the feature stack and of every table it is handed.
:class:`MemoryWatch` reports the resident memory of the process tree, which contains
the workers, the memory the host still has available, and the memory limit that
applies. In a container that is the cgroup's limit, which the kernel enforces and
``psutil`` does not show, and on a host with no limit it is the host's memory
(:mod:`mosaic.core.pipeline.memory_limit`). The free fraction a floor reads is the
share of that limit still available.

Resident memory counts a page once for every process that maps it. Thread mode lives in
one process and is exact. Process mode overstates a little where workers share
libraries.

The watch also measures a run's cost from the moment :meth:`MemoryWatch.begin` is
called: the CPU time of the process tree and the wall time since then. A sampling
thread (:meth:`MemoryWatch.start_sampling`) measures between the run's own events, so
a long fit or a long entry is measured too.

:func:`max_rss_bytes` reads the process's ``ru_maxrss``, a high-water mark over the
whole life of the process, which in a pipeline run, the agent server or a notebook
carries earlier work. A run reads it when measuring begins and uses a later reading
only when it has risen since. Linux reports the figure in kilobytes and macOS in
bytes, and the function returns bytes.

Imports the standard library only. ``psutil`` is imported inside the methods, as
``markers`` does, and a host where it cannot be imported reports nothing.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from .memory_limit import MemoryLimit, read_memory_limit

__all__ = [
    "ESTIMATE_SAMPLE_INTERVAL_SECONDS",
    "SAMPLE_INTERVAL_SECONDS",
    "SAMPLER_THREAD_NAME",
    "MemorySample",
    "MemoryWatch",
    "max_rss_bytes",
    "process_rss_bytes",
]

SAMPLE_INTERVAL_SECONDS: Final = 2.0
"""The shortest time between two measurements that :meth:`MemoryWatch.sample_if_due` takes."""

ESTIMATE_SAMPLE_INTERVAL_SECONDS: Final = 0.5
"""The time between two measurements of the sampling thread."""

SAMPLER_THREAD_NAME: Final = "mosaic-memory-sampler"
"""The name of the thread :meth:`MemoryWatch.start_sampling` starts."""

_SPAWNED_WORKER_MARK: Final = "spawn_main"
"""What the command line of a worker that ``multiprocessing`` spawned contains."""


@dataclass(frozen=True)
class MemorySample:
    """One measurement of memory, in bytes.

    Attributes:
        tree_rss_bytes: The resident memory of this process and every process
            below it.
        available_bytes: The memory the host can give a process without swapping.
        total_bytes: The host's physical memory.
        limit: The memory limit that applies, with its source and use.
        spawned_workers: The children of this process that ``multiprocessing``
            spawned as workers since :meth:`MemoryWatch.begin`, which are a
            process-mode run's live workers. A resource tracker is not one, nor is
            a process a worker forked, which keeps the worker's command line.
    """

    tree_rss_bytes: int
    available_bytes: int
    total_bytes: int
    limit: MemoryLimit
    spawned_workers: int = 0


def max_rss_bytes() -> int:
    """Return this process's high-water resident memory in bytes, 0 where unknown."""
    try:
        import resource
    except ImportError:
        return 0
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def process_rss_bytes() -> int:
    """Return this process's resident memory in bytes, 0 where ``psutil`` is absent."""
    try:
        import psutil
    except ImportError:
        return 0
    try:
        return int(psutil.Process().memory_info().rss)
    except psutil.Error:
        return 0


def _child_pids() -> frozenset[int]:
    """The process ids below this process, empty where ``psutil`` is absent."""
    try:
        import psutil
    except ImportError:
        return frozenset()
    try:
        return frozenset(
            child.pid for child in psutil.Process().children(recursive=True)
        )
    except psutil.Error:
        return frozenset()


def _tree_cpu_seconds() -> float:
    """The user and system time of this process, its reaped children and its live ones."""
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        root = psutil.Process()
        own = root.cpu_times()
        total = own.user + own.system + own.children_user + own.children_system
        children = root.children(recursive=True)
    except psutil.Error:
        return 0.0
    for child in children:
        try:
            times = child.cpu_times()
        except psutil.Error:
            continue
        total += times.user + times.system
    return float(total)


class MemoryWatch:
    """Measure memory repeatedly and keep the extremes.

    Every read and update of the extremes holds the watch's lock, because the
    sampling thread and the run's thread both measure.

    Attributes:
        peak_rss_bytes: The largest ``tree_rss_bytes`` measured, 0 before the first.
        min_available_bytes: The smallest ``available_bytes`` measured, 0 before the
            first.
        total_bytes: The host's physical memory at the last measurement.
        limit: The memory limit at the last measurement, ``None`` before the first.
        min_available_in_limit_bytes: The least memory that was available under the
            limit, 0 before the first measurement.
        limit_peak_bytes: The largest high-water mark the limit's accounting
            reported, 0 when it reports none.
    """

    def __init__(self, *, interval_seconds: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self._interval: float = interval_seconds
        self._last: float | None = None
        self._lock: threading.Lock = threading.Lock()
        self._began: float | None = None
        self._cpu_at_begin: float = 0.0
        self._max_rss_at_begin: int = 0
        self._children_at_begin: frozenset[int] = frozenset()
        self._stop: threading.Event = threading.Event()
        self._sampler: threading.Thread | None = None
        self.peak_rss_bytes: int = 0
        self.min_available_bytes: int = 0
        self.total_bytes: int = 0
        self.limit: MemoryLimit | None = None
        self.min_available_in_limit_bytes: int = 0
        self.limit_peak_bytes: int = 0

    def begin(self) -> None:
        """Start measuring cost: CPU, wall time and the high-water mark count from now.

        The child processes alive now are earlier work, and no sample counts them as
        this run's workers.
        """
        self._children_at_begin = _child_pids()
        self._cpu_at_begin = _tree_cpu_seconds()
        self._max_rss_at_begin = max_rss_bytes()
        self._began = time.monotonic()

    def cpu_seconds(self) -> float:
        """Return the tree's user and system time since :meth:`begin`, 0 before it."""
        if self._began is None:
            return 0.0
        return max(0.0, _tree_cpu_seconds() - self._cpu_at_begin)

    def wall_seconds(self) -> float:
        """Return the wall time since :meth:`begin`, 0 before it."""
        if self._began is None:
            return 0.0
        return time.monotonic() - self._began

    def high_water_mark(self) -> int:
        """Return the process's high-water resident memory now, in bytes."""
        return max_rss_bytes()

    def high_water_mark_since(self, since: int | None = None) -> int:
        """Return the process's high-water mark when it has risen, else 0.

        Args:
            since: The reading the mark must have risen above, the reading at
                :meth:`begin` when ``None``.

        Returns:
            The current high-water mark in bytes when it is above *since*, so it
            was reached after that reading, else 0.
        """
        floor = self._max_rss_at_begin if since is None else since
        now = max_rss_bytes()
        return now if now > floor else 0

    def sample(self) -> MemorySample | None:
        """Measure now, and return the measurement.

        Returns:
            The measurement, or ``None`` where ``psutil`` is not importable.
        """
        try:
            import psutil
        except ImportError:
            return None
        host = psutil.virtual_memory()
        limit = read_memory_limit()
        if limit is None:
            return None
        try:
            root = psutil.Process()
            members = [root, *root.children(recursive=True)]
            # Workers are direct children: a fork of a worker keeps its command line.
            direct = root.children(recursive=False)
        except psutil.Error:
            members = []
            direct = []
        resident = 0
        for member in members:
            try:
                resident += int(member.memory_info().rss)
            except psutil.Error:
                continue
        workers = 0
        for child in direct:
            if child.pid in self._children_at_begin:
                continue
            try:
                command = child.cmdline()
            except psutil.Error:
                continue
            if any(_SPAWNED_WORKER_MARK in part for part in command):
                workers += 1
        sample = MemorySample(
            tree_rss_bytes=resident,
            available_bytes=int(host.available),
            total_bytes=int(host.total),
            limit=limit,
            spawned_workers=workers,
        )
        with self._lock:
            self.peak_rss_bytes = max(self.peak_rss_bytes, sample.tree_rss_bytes)
            if self.min_available_bytes == 0:
                self.min_available_bytes = sample.available_bytes
            else:
                self.min_available_bytes = min(
                    self.min_available_bytes, sample.available_bytes
                )
            self.total_bytes = sample.total_bytes
            if self.limit is None:
                self.min_available_in_limit_bytes = limit.available_bytes
            else:
                self.min_available_in_limit_bytes = min(
                    self.min_available_in_limit_bytes, limit.available_bytes
                )
            self.limit_peak_bytes = max(self.limit_peak_bytes, limit.peak_bytes)
            self.limit = limit
        return sample

    def sample_if_due(self) -> bool:
        """Measure when the interval has passed since the last measurement this took.

        The sampling thread's measurements do not count, so a caller that reports
        on this schedule still reports once an interval.

        Returns:
            Whether a measurement was taken.
        """
        now = time.monotonic()
        with self._lock:
            if self._last is not None and now - self._last < self._interval:
                return False
            self._last = now
        return self.sample() is not None

    def free_fraction(self) -> float | None:
        """Return the share of the memory limit that is available now.

        The limit is the container's where a cgroup sets one, else the host's
        memory. It reads the limit afresh and records nothing.

        Returns:
            A fraction from 0 to 1, or ``None`` where ``psutil`` is not importable.
        """
        limit = read_memory_limit()
        if limit is None or limit.limit_bytes <= 0:
            return None
        return limit.available_bytes / limit.limit_bytes

    def start_sampling(
        self,
        on_sample: Callable[[MemorySample], None],
        *,
        interval: float = ESTIMATE_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        """Measure every *interval* seconds on a daemon thread, until :meth:`stop_sampling`.

        Each measurement is handed to *on_sample*. An error it raises is swallowed:
        a measurement is a report, and a failed report must not fail the run. A
        second call starts nothing.
        """
        if self._sampler is not None:
            return
        self._stop.clear()
        self._sampler = threading.Thread(
            target=self._keep_sampling,
            args=(on_sample, interval),
            name=SAMPLER_THREAD_NAME,
            daemon=True,
        )
        self._sampler.start()

    def stop_sampling(self) -> None:
        """Stop the sampling thread and wait for it to end."""
        sampler, self._sampler = self._sampler, None
        if sampler is None:
            return
        self._stop.set()
        sampler.join()

    def _keep_sampling(
        self, on_sample: Callable[[MemorySample], None], interval: float
    ) -> None:
        while not self._stop.wait(interval):
            try:
                sample = self.sample()
                if sample is not None:
                    on_sample(sample)
            except Exception:
                continue
