"""Measures the memory and CPU that a run and its worker processes use, and what the limit leaves.

A run with several workers is where memory runs out: each process-mode worker is an
interpreter with its own copy of the feature stack and of every table it is handed.
:class:`MemoryWatch` reports the resident memory of the process tree, which contains the
workers, its non-reclaimable memory, the memory the host still has available, and the
memory limit that applies. In a container that is the cgroup's limit, which the kernel
enforces and ``psutil`` does not show, and on a host with no limit it is the host's
memory (:mod:`mosaic.core.pipeline.memory_limit`). The free fraction a floor reads is
the share of that limit still available.

The watch knows the processes below this one from its last walk of the tree, which it
repeats at most once an interval and after a pool's submission that may have started a
worker. A sample between walks measures the known processes only.

Resident memory counts a page once for every process that maps it. Thread mode lives in
one process and is exact. Process mode overstates a little where workers share
libraries.

The non-reclaimable figure is the memory the kernel cannot reclaim, which is what a
memory limit kills a job for, and the admission gate reads it in place of resident
memory where a run measures it. Under a cgroup limit it is the cgroup's own figure, read
once for the whole tree (:attr:`MemoryLimit.nonreclaimable_bytes`): anonymous and shared
memory, plus kernel memory under cgroup v2, and it counts every process in the cgroup.
Where no limit applies it is the sum over the tree of ``RssAnon + RssShmem`` from
``/proc/<pid>/status``, which leaves out kernel memory. Where there is no ``/proc`` it
is not measured, and a run gates on resident memory. Neither figure counts the page
cache or GPU memory. A worker process cannot read its share of the cgroup, so
:func:`peak_nonreclaimable_while` samples the worker's own status file, which leaves out
the worker's kernel memory and misses a spike shorter than its interval.

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

import os
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .memory_limit import MemoryLimit, read_memory_limit

if TYPE_CHECKING:
    import psutil

__all__ = [
    "ESTIMATE_SAMPLE_INTERVAL_SECONDS",
    "SAMPLE_INTERVAL_SECONDS",
    "SAMPLER_THREAD_NAME",
    "WORKER_SAMPLER_THREAD_NAME",
    "MemorySample",
    "MemoryWatch",
    "NonreclaimablePeak",
    "max_rss_bytes",
    "now",
    "peak_nonreclaimable_while",
    "process_nonreclaimable_bytes",
    "process_rss_bytes",
]

SAMPLE_INTERVAL_SECONDS: Final = 2.0
"""The shortest time between two measurements that :meth:`MemoryWatch.sample_if_due` takes."""

ESTIMATE_SAMPLE_INTERVAL_SECONDS: Final = 0.5
"""The time between two measurements of the sampling thread."""

SAMPLER_THREAD_NAME: Final = "mosaic-memory-sampler"
"""The name of the thread :meth:`MemoryWatch.start_sampling` starts."""

WORKER_SAMPLER_THREAD_NAME: Final = "mosaic-worker-memory-sampler"
"""The name of the thread :func:`peak_nonreclaimable_while` starts."""

_DEFAULT_PROC_ROOT: Final = Path("/proc")

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
            a process a worker forked, which keeps the worker's command line. The
            count comes from the last walk of the tree, which may be up to one
            interval old between a pool's submissions.
        nonreclaimable_bytes: The memory the kernel cannot reclaim: the limit's
            figure where a cgroup limit applies, else the sum over the process tree
            of ``RssAnon + RssShmem``. ``None`` where ``/proc`` cannot be read,
            where a process that is still alive had no readable figure, or where a
            cgroup's figure is 0, which a cgroup that holds a process never has.
        started_at: ``time.monotonic()`` when the measurement began, before its
            reads and before any walk of the tree it made, so an entry that
            started after it is not in this sample. ``None`` where the sample was
            not taken by a :class:`MemoryWatch`.
    """

    tree_rss_bytes: int
    available_bytes: int
    total_bytes: int
    limit: MemoryLimit
    spawned_workers: int = 0
    nonreclaimable_bytes: int | None = None
    started_at: float | None = None


def now() -> float:
    """The monotonic time that the age of a watch's membership reads.

    A function of this module, read on each use, so that a test can give every
    watch one clock that it controls.
    """
    return time.monotonic()


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


def process_nonreclaimable_bytes(
    pid: int | None = None, *, proc_root: Path = _DEFAULT_PROC_ROOT
) -> int | None:
    """Return the anonymous and shared resident memory of a process, in bytes.

    Args:
        pid: The process to read, this process when ``None``.
        proc_root: Where ``/proc`` is mounted.

    Returns:
        ``RssAnon + RssShmem`` from the process's ``status`` file, or ``None``
        where the file is missing, unreadable or lacks either line, as it does
        without ``/proc`` and for a process that has gone.
    """
    name = "self" if pid is None else str(pid)
    try:
        text = (proc_root / name / "status").read_text()
    except (OSError, ValueError):
        return None
    kilobytes: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key not in ("RssAnon", "RssShmem"):
            continue
        fields = value.split()
        if len(fields) != 2 or fields[1] != "kB":
            return None
        try:
            kilobytes[key] = int(fields[0])
        except ValueError:
            return None
    if len(kilobytes) != 2:
        return None
    return sum(kilobytes.values()) * 1024


@dataclass(slots=True)
class NonreclaimablePeak:
    """The largest figure that :func:`peak_nonreclaimable_while` has read.

    Attributes:
        peak: The largest read in bytes, ``None`` while no read has succeeded.
    """

    peak: int | None = None

    def record(self, figure: int | None) -> None:
        """Keep *figure* when it is larger than the peak so far."""
        if figure is not None and (self.peak is None or figure > self.peak):
            self.peak = figure


@contextmanager
def peak_nonreclaimable_while(
    *,
    interval: float = ESTIMATE_SAMPLE_INTERVAL_SECONDS,
    proc_root: Path = _DEFAULT_PROC_ROOT,
) -> Generator[NonreclaimablePeak]:
    """Track this process's peak non-reclaimable memory while the block runs.

    The peak is the largest of a read on entry, a read every *interval* seconds on
    a daemon thread named :data:`WORKER_SAMPLER_THREAD_NAME`, and a read on exit.
    The thread stops and is joined on exit, whether the block raised or not.

    Args:
        interval: The seconds between two reads of the sampling thread.
        proc_root: Where ``/proc`` is mounted.

    Yields:
        A holder whose ``peak`` is ``None`` when no read succeeded.
    """
    holder = NonreclaimablePeak()
    holder.record(process_nonreclaimable_bytes(proc_root=proc_root))
    stop = threading.Event()

    def keep_reading() -> None:
        while not stop.wait(interval):
            holder.record(process_nonreclaimable_bytes(proc_root=proc_root))

    sampler = threading.Thread(
        target=keep_reading, name=WORKER_SAMPLER_THREAD_NAME, daemon=True
    )
    sampler.start()
    try:
        yield holder
    finally:
        stop.set()
        sampler.join()
        holder.record(process_nonreclaimable_bytes(proc_root=proc_root))


def _process_has_gone(pid: int, proc_root: Path) -> bool:
    """Whether *pid* has no ``status`` file or is a zombie or dead."""
    try:
        text = (proc_root / str(pid) / "status").read_text()
    except (OSError, ValueError):
        return True
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key == "State":
            return value.strip()[:1] in ("Z", "X")
    return False


def _own_cpu_seconds() -> float:
    """The user and system time of this process and of its reaped children.

    ``resource.getrusage`` reads it to the microsecond, where ``os.times()`` rounds
    to 10 ms on macOS, and counts reaped children on macOS, where ``psutil``
    reports 0. ``os.times()`` reads it where ``resource`` cannot be imported
    (Windows).
    """
    try:
        import resource
    except ImportError:
        times = os.times()
        return times.user + times.system + times.children_user + times.children_system
    own = resource.getrusage(resource.RUSAGE_SELF)
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + reaped.ru_utime + reaped.ru_stime


def _process_cpu_seconds(process: psutil.Process) -> float:
    """The user and system time of *process*, without its children's."""
    times = process.cpu_times()
    return float(times.user + times.system)


@dataclass(slots=True)
class _Member:
    """What the walks of a :class:`MemoryWatch` know of one process below this one.

    Attributes:
        direct: Whether the watching process is its parent.
        found_at_begin: Whether the walk of :meth:`MemoryWatch.begin` found it.
        first_walk: The number of the walk that first found it.
        cpu_base: Its user and system time when first found.
        cpu_last: Its user and system time at the last read that succeeded.
        spawned: Whether its command line, read at the last walk, marks it a
            worker that ``multiprocessing`` spawned. Read for a direct child only.
    """

    direct: bool
    found_at_begin: bool
    first_walk: int
    cpu_base: float
    cpu_last: float
    spawned: bool = False


def _is_spawned_worker(process: psutil.Process) -> bool:
    """Whether *process*'s command line marks a worker that ``multiprocessing`` spawned."""
    import psutil

    try:
        command = process.cmdline()
    except psutil.Error:
        return False
    return any(_SPAWNED_WORKER_MARK in part for part in command)


class MemoryWatch:
    """Measure memory repeatedly and keep the extremes.

    Every read and update of the extremes holds the watch's lock, because the
    sampling thread and the run's thread both measure.

    The processes below this one are known from the last walk of the tree, which a
    read repeats once it is ``ESTIMATE_SAMPLE_INTERVAL_SECONDS`` old and
    :meth:`sight_workers` repeats when a pool may have started a worker. A walk and
    a read's update of the members hold the membership lock, which is never held
    while another lock is taken. A member whose pid another process took after it
    ended is read as that process until the next walk, at most an interval later.

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
        nonreclaimable_bytes: The latest sample's non-reclaimable figure, 0 when
            that sample could not measure it.
        peak_nonreclaimable_bytes: The largest non-reclaimable figure measured, 0
            when none was.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
        proc_root: Path = _DEFAULT_PROC_ROOT,
    ) -> None:
        self._interval: float = interval_seconds
        self._proc_root: Path = proc_root
        self._last: float | None = None
        self._lock: threading.Lock = threading.Lock()
        self._began: float | None = None
        self._cpu_at_begin: float = 0.0
        self._max_rss_at_begin: int = 0
        self._membership: threading.Lock = threading.Lock()
        self._members: dict[psutil.Process, _Member] = {}
        self._walks: int = 0
        self._walked_at: float | None = None
        self._reaped_bases: float = 0.0
        self._stop: threading.Event = threading.Event()
        self._sampler: threading.Thread | None = None
        self.peak_rss_bytes: int = 0
        self.min_available_bytes: int = 0
        self.total_bytes: int = 0
        self.limit: MemoryLimit | None = None
        self.min_available_in_limit_bytes: int = 0
        self.limit_peak_bytes: int = 0
        self.nonreclaimable_bytes: int = 0
        self.peak_nonreclaimable_bytes: int = 0

    def begin(self) -> None:
        """Start measuring cost: CPU, wall time and the high-water mark count from now.

        The child processes alive now are earlier work, and no sample counts them as
        this run's workers. A process found now counts its CPU from now.
        """
        with self._membership:
            self._members = {}
            self._reaped_bases = 0.0
            self._walked_at = None
            _ = self._walk(at_begin=True)
        self._cpu_at_begin = _own_cpu_seconds()
        self._max_rss_at_begin = max_rss_bytes()
        self._began = time.monotonic()

    def cpu_seconds(self) -> float:
        """Return the tree's user and system time since :meth:`begin`, 0 before it.

        A process below this one counts its CPU from the walk that first finds it,
        also after it is reaped. The figure is this process's time and its reaped
        children's, plus each member's time since it was first found, less the
        times at first sighting of the direct members that have been reaped, whose
        whole time the reaped children's now holds, less the same figure at
        :meth:`begin`.

        It walks the tree when the membership is stale, then reads the members, and
        then this process's own time, under the membership lock that every leave
        holds, so a member that has left was reaped before that read and its whole
        time is in it. A direct member reaped between its read and the last is
        counted twice by that one read, which the next read corrects once the
        member leaves. The other order would undercount instead, and a span whose
        start read undercounted would take the difference.

        Returns:
            The seconds, never below 0, and 0 where ``psutil`` is not importable.
        """
        if self._began is None:
            return 0.0
        try:
            import psutil
        except ImportError:
            return 0.0
        with self._membership:
            if self._membership_is_stale():
                _ = self._walk()
            known = list(self._members)
        read: dict[psutil.Process, float] = {}
        gone: list[psutil.Process] = []
        for process in known:
            try:
                read[process] = _process_cpu_seconds(process)
            except psutil.ZombieProcess:
                pass
            except psutil.NoSuchProcess:
                gone.append(process)
            except psutil.Error:
                pass
        with self._membership:
            self._leave(gone)
            own = _own_cpu_seconds()
            for process, cpu in read.items():
                member = self._members.get(process)
                if member is not None:
                    member.cpu_last = cpu
            members = sum(
                member.cpu_last - member.cpu_base for member in self._members.values()
            )
            total = own + members - self._reaped_bases
        return max(0.0, total - self._cpu_at_begin)

    @property
    def walks(self) -> int:
        """How many walks of the process tree have succeeded."""
        return self._walks

    def sight_workers(self, *, since: int, running: int) -> None:
        """Walk the tree when fewer workers are known than calls are running.

        A pool calls it after a submission, and walks only when the submission may
        have started a worker.

        Args:
            since: The walk count when the pool was made. The workers first found
                by that walk or an earlier one belong to an earlier pool.
            running: The pool's calls not yet done.
        """
        with self._membership:
            known = sum(
                1
                for member in self._members.values()
                if member.first_walk > since and self._is_worker(member)
            )
            if known < running:
                _ = self._walk()

    def _walk(self, *, at_begin: bool = False) -> bool:
        """List the processes below this one, keeping the record of each known one.

        The caller holds the membership lock. A walk that fails, or a host without
        ``psutil``, changes nothing. A new member's CPU now is its base, and a direct
        member that the walk no longer finds has been reaped and leaves.

        Returns:
            Whether the walk succeeded.
        """
        try:
            import psutil
        except ImportError:
            return False
        try:
            root = psutil.Process()
            found = root.children(recursive=True)
        except psutil.Error:
            return False
        walk = self._walks + 1
        members: dict[psutil.Process, _Member] = {}
        for process in found:
            member = self._members.get(process)
            if member is None:
                try:
                    direct = process.ppid() == root.pid
                    cpu = _process_cpu_seconds(process)
                except psutil.Error:
                    continue
                member = _Member(
                    direct=direct,
                    found_at_begin=at_begin,
                    first_walk=walk,
                    cpu_base=cpu,
                    cpu_last=cpu,
                )
            if member.direct and not member.found_at_begin:
                member.spawned = _is_spawned_worker(process)
            members[process] = member
        self._leave([process for process in self._members if process not in members])
        self._members = members
        self._walks = walk
        self._walked_at = now()
        return True

    def _leave(self, gone: list[psutil.Process]) -> None:
        """Drop the members in *gone*, which have been reaped.

        The caller holds the membership lock. A direct member's CPU base is
        subtracted from then on, because the reaped children's time holds its
        whole CPU.
        """
        for process in gone:
            member = self._members.pop(process, None)
            if member is not None and member.direct:
                self._reaped_bases += member.cpu_base

    def _membership_is_stale(self) -> bool:
        """Whether the last walk is an interval old, or there was none."""
        return (
            self._walked_at is None
            or now() - self._walked_at >= ESTIMATE_SAMPLE_INTERVAL_SECONDS
        )

    @staticmethod
    def _is_worker(member: _Member) -> bool:
        return member.direct and member.spawned and not member.found_at_begin

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

        Walks the tree only when the membership is stale. A member that has ended
        adds nothing and leaves the members.

        Returns:
            The measurement, or ``None`` where ``psutil`` is not importable.
        """
        try:
            import psutil
        except ImportError:
            return None
        started_at = time.monotonic()
        host = psutil.virtual_memory()
        limit = read_memory_limit()
        if limit is None:
            return None
        with self._membership:
            if self._membership_is_stale():
                _ = self._walk()
            known = list(self._members)
        resident = 0
        pids: list[int] = []
        gone: list[psutil.Process] = []
        for process in [psutil.Process(), *known]:
            try:
                resident += int(process.memory_info().rss)
            except psutil.ZombieProcess:
                pass
            except psutil.NoSuchProcess:
                gone.append(process)
                continue
            except psutil.Error:
                pass
            pids.append(process.pid)
        with self._membership:
            self._leave(gone)
            workers = sum(
                1 for member in self._members.values() if self._is_worker(member)
            )
        sample = MemorySample(
            tree_rss_bytes=resident,
            available_bytes=int(host.available),
            total_bytes=int(host.total),
            limit=limit,
            spawned_workers=workers,
            nonreclaimable_bytes=self._tree_nonreclaimable_bytes(limit, pids),
            started_at=started_at,
        )
        measured = sample.nonreclaimable_bytes or 0
        with self._lock:
            self.nonreclaimable_bytes = measured
            self.peak_nonreclaimable_bytes = max(
                self.peak_nonreclaimable_bytes, measured
            )
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

    def _tree_nonreclaimable_bytes(
        self, limit: MemoryLimit, pids: list[int]
    ) -> int | None:
        """Return the non-reclaimable memory of the process tree, ``None`` if not read.

        A cgroup limit's figure already holds every process in the cgroup and is not
        summed per process, and a figure of 0 means its ``memory.stat`` was not read.
        Without a cgroup the status files of *pids* are summed. A process that has
        gone since *pids* was listed, or is a zombie, adds nothing, and a live one
        with no readable figure leaves the whole figure out.
        """
        if limit.source != "host":
            return limit.nonreclaimable_bytes or None
        if not pids or not self._proc_root.is_dir():
            return None
        total = 0
        for pid in pids:
            figure = process_nonreclaimable_bytes(pid, proc_root=self._proc_root)
            if figure is not None:
                total += figure
            elif not _process_has_gone(pid, self._proc_root):
                return None
        return total

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
