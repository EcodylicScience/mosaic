"""The memory a feature run's entries cost per input row, learned while it runs.

The run fixes one measure at its base sample, and the estimate, the budget, the bases
and the peak sample are all in it. The measure is the memory the kernel cannot reclaim
(:attr:`MemorySample.nonreclaimable_bytes`) where the base sample measured it, which is
what a memory limit kills a job for. Where it did not, as on macOS, the measure is the
resident memory of the process tree. The two never mix in one estimate: a process
worker that returns no non-reclaimable figure in a non-reclaimable run adds no
observation and sets no worker base.

:class:`RowCostEstimate` keeps the memory of the run's process tree in that measure
beside the input rows in flight at each measurement, and takes the most memory a row
has cost over the tree's fixed part: the parent's base and, in process mode, a worker's
own memory for each worker process alive at the measurement. Several entries in one
process cannot be told apart. The figure is per row of everything in flight.

In process mode a worker also reports its own memory around each entry. In the
non-reclaimable measure it samples its anonymous and shared memory while it applies,
and its growth over its base, per row of that entry, is a further observation. In the
resident measure it reports its high-water mark before and after the entry. When the
mark rose during the entry, the peak was that entry's, and its growth over the worker's
base, per row of that entry, is a further observation. A mark that did not rise belongs
to an earlier entry on the same worker and is left out. The estimate is the largest
observed. An underestimate needs every observation to be low. When every observation
fell at or under the fixed part there is no estimate, rather than an estimate of 0
bytes a row.

What the estimate cannot see:

- memory that does not grow with input rows, such as a feature that reads video
  frames, or a pair feature whose cost grows with the square of its individuals;
- allocator memory kept after an entry ends, which inflates the estimate and so errs
  on the safe side;
- a peak between two observations in thread mode, in process mode one that a worker
  reaches before it has measured its base, and a spike of a worker shorter than the
  sampling interval;
- the page cache and GPU memory, which neither measure counts;
- kernel memory, which only a cgroup v2 figure counts. A cgroup v1 figure and a
  ``/proc`` sum leave it out. A worker's own figure leaves it out too, and the
  worker's page tables count as growth of the entry, on the safe side;
- a page several workers map, which the resident measure and a ``/proc`` sum count
  once per worker;
- memory outside this run's process tree, which a cgroup figure counts when it is in
  the same cgroup, and a process there that grows during the run raises the estimate.
  A ``/proc`` figure leaves it out, and the budget takes it from the limit's available
  memory instead;
- a child process of earlier work in the same process, which the tree counts as this
  run's. One that exits or grows during the run moves the figure.

Imports the standard library and :mod:`mosaic.runlog`, which imports the standard library
only.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from mosaic.runlog import FlightSpan, InFlightLevel, sweep_levels

__all__ = [
    "KEPT_OBSERVATIONS",
    "EntriesInFlight",
    "EntrySpan",
    "PeakSample",
    "RowCostEstimate",
    "WorkerBase",
    "WorkerBases",
    "WorkerMeasure",
    "admission_budget",
    "grown_over_bases",
    "larger_peak",
]

KEPT_OBSERVATIONS: Final = 10_000
"""How many observations an estimate keeps, so a long run's memory is fixed."""


@dataclass(frozen=True, slots=True)
class WorkerMeasure:
    """What one process worker measured of its own memory, in bytes.

    Attributes:
        base_rss_bytes: Its resident memory after it rebuilt the feature and loaded
            its state, before ``apply``, less the entry's table it was handed: the
            memory a worker holds with no table, as the probe measures it.
        peak_before_rss_bytes: Its high-water resident memory at the same moment,
            which holds the peaks of the entries it applied before.
        peak_rss_bytes: Its high-water resident memory after ``apply``.
        base_nonreclaimable_bytes: Its anonymous and shared resident memory at the
            same moment as ``base_rss_bytes``, less the table, ``None`` where it
            could not be read.
        peak_nonreclaimable_bytes: The largest anonymous and shared resident memory
            its sampler read from before ``apply`` to after it, ``None`` where none
            could be read.
    """

    base_rss_bytes: int
    peak_before_rss_bytes: int
    peak_rss_bytes: int
    base_nonreclaimable_bytes: int | None = None
    peak_nonreclaimable_bytes: int | None = None

    @property
    def grown_bytes(self) -> int | None:
        """What the worker's memory grew over its base while it applied one entry.

        ``None`` where either figure could not be read, so a growth is never
        taken against a base that was not measured.
        """
        if (
            self.base_nonreclaimable_bytes is None
            or self.peak_nonreclaimable_bytes is None
        ):
            return None
        return max(0, self.peak_nonreclaimable_bytes - self.base_nonreclaimable_bytes)


@dataclass(frozen=True, slots=True)
class WorkerBase:
    """What one process worker holds before it is handed a table, in bytes.

    Attributes:
        rss_bytes: Its resident memory, 0 where it could not be measured.
        nonreclaimable_bytes: Its anonymous and shared resident memory, ``None``
            where it could not be read.
    """

    rss_bytes: int
    nonreclaimable_bytes: int | None


@dataclass(slots=True)
class WorkerBases:
    """The largest base any process worker has reported, in bytes.

    Attributes:
        rss_bytes: The largest resident base, 0 until a worker reports one.
        nonreclaimable_bytes: The largest anonymous and shared base, 0 until a
            worker reports one.
    """

    rss_bytes: int = 0
    nonreclaimable_bytes: int = 0

    def learn(self, worker: WorkerMeasure, *, nonreclaimable: bool) -> int | None:
        """Keep the largest bases seen, and return the one in the run's measure.

        Args:
            worker: What one process worker measured of itself.
            nonreclaimable: Whether the run's measure is the non-reclaimable
                figure.

        Returns:
            The largest base in the run's measure, ``None`` when the run is
            non-reclaimable and *worker* has no such figure, so the estimate
            takes no worker base from a worker that cannot give it.
        """
        self.rss_bytes = max(self.rss_bytes, worker.base_rss_bytes)
        if worker.base_nonreclaimable_bytes is not None:
            self.nonreclaimable_bytes = max(
                self.nonreclaimable_bytes, worker.base_nonreclaimable_bytes
            )
        if not nonreclaimable:
            return self.rss_bytes
        if worker.base_nonreclaimable_bytes is None:
            return None
        return self.nonreclaimable_bytes


@dataclass(frozen=True, slots=True)
class PeakSample:
    """The sample of a run with the largest non-reclaimable memory while entries ran.

    Attributes:
        bytes_: The non-reclaimable memory of the sample.
        entries: The entries in flight at the sample, at least 1.
        rows: Their input rows.
        pair_frames: Their ordered pairs summed over frames, an entry without the
            figure adding 0.
        pair_frames_bound: Whether any of them carried a lower bound in place of a
            measured figure.
        live_workers: The process workers alive at the sample, 0 in thread mode.
    """

    bytes_: int
    entries: int
    rows: int
    pair_frames: int
    pair_frames_bound: bool
    live_workers: int


def larger_peak(
    held: PeakSample | None,
    *,
    bytes_: int,
    entries: int,
    rows: int,
    pair_frames: int,
    pair_frames_bound: bool,
    live_workers: int,
) -> PeakSample | None:
    """Return the peak sample once a sample with these figures is considered.

    A sample with no entry in flight is never the peak, so a figure the fit or the
    run's idle time reached cannot be paired with entries it did not hold.

    Args:
        held: The peak sample so far, ``None`` before the first.
        bytes_: The sample's non-reclaimable memory.
        entries: The entries in flight at the sample.
        rows: Their input rows.
        pair_frames: Their ordered pairs summed over frames.
        pair_frames_bound: Whether any of them carried a lower bound.
        live_workers: The process workers alive at the sample.

    Returns:
        A new sample when *entries* is at least 1 and *bytes_* exceeds the held
        figure, else *held*.
    """
    if entries < 1 or (held is not None and bytes_ <= held.bytes_):
        return held
    return PeakSample(
        bytes_=bytes_,
        entries=entries,
        rows=rows,
        pair_frames=pair_frames,
        pair_frames_bound=pair_frames_bound,
        live_workers=live_workers,
    )


def grown_over_bases(
    peak: PeakSample | None, *, base_bytes: int, worker_base_bytes: int
) -> int:
    """Return what the peak sample held over the run's base and its workers' bases.

    The workers' bases are those of the workers alive at the sample, so a worker
    base learned after it applies, as in :class:`RowCostEstimate`.

    Args:
        peak: The run's peak sample, ``None`` without one.
        base_bytes: The run's own memory before the first entry loaded, 0 where it
            was not measured.
        worker_base_bytes: The largest base any process worker reported, 0 where
            none was.

    Returns:
        The growth in bytes, never negative. 0 without a peak sample, and 0 when a
        base the sample needs was not measured, rather than a figure taken against
        a base of 0.
    """
    if peak is None or base_bytes <= 0:
        return 0
    if peak.live_workers > 0 and worker_base_bytes <= 0:
        return 0
    fixed = base_bytes + peak.live_workers * worker_base_bytes
    return max(0, peak.bytes_ - fixed)


@dataclass(frozen=True, slots=True)
class _Observation:
    measured_bytes: int
    in_flight_rows: int
    live_workers: int


@dataclass
class RowCostEstimate:
    """The most memory an input row in flight has cost, over the tree's fixed part.

    Every figure is in the run's measure: the memory the kernel cannot reclaim where
    the run's base sample measured it, else the resident memory of the process tree.

    Every method holds the estimate's lock: observations arrive from a sampling
    thread and from the threads that end an entry's apply, while the run's thread
    reads the estimate.
    """

    _base: int = 0
    _worker_base: int = 0
    _per_row: int = 0
    _worker_peak_per_row: int = 0
    _observations: deque[_Observation] = field(
        default_factory=lambda: deque(maxlen=KEPT_OBSERVATIONS)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def parent_base(self, bytes_: int) -> None:
        """Set the run's own fixed memory: interpreter, imports and the feature's state.

        Args:
            bytes_: The run's own memory in the run's measure.
        """
        with self._lock:
            self._base = bytes_
            self._recompute()

    def worker_base(self, bytes_: int) -> None:
        """Set one process worker's own memory, and recompute from the kept observations.

        Observations taken before a worker reported counted its base as rows.

        Args:
            bytes_: One worker's own memory in the run's measure.
        """
        with self._lock:
            self._worker_base = bytes_
            self._recompute()

    def observe(
        self, measured_bytes: int, in_flight_rows: int, live_workers: int
    ) -> None:
        """Keep one measurement of the tree beside the rows and workers live at it.

        An observation with no rows in flight is left out.

        Args:
            measured_bytes: The memory of the run's process tree in the run's measure.
            in_flight_rows: The input rows of the entries in flight.
            live_workers: The run's worker processes alive at the measurement, 0
                in thread mode.
        """
        if in_flight_rows <= 0:
            return
        observation = _Observation(measured_bytes, in_flight_rows, live_workers)
        with self._lock:
            self._observations.append(observation)
            self._per_row = max(self._per_row, self._cost(observation))

    def observe_worker(
        self, measure: WorkerMeasure, rows: int, *, nonreclaimable: bool
    ) -> None:
        """Keep a process worker's growth over its own base while it applied one entry.

        Resident memory has a high-water mark over the worker's whole life, so it
        is the entry's peak only when it rose past the mark read before the entry.
        Otherwise an earlier entry on the same worker set it, and the observation
        is left out. The non-reclaimable peak is the largest figure the worker read
        during this entry alone and needs no such test.

        Args:
            measure: What the worker measured of itself around the entry.
            rows: The entry's input rows. An entry without rows is left out.
            nonreclaimable: Whether the run's measure is the non-reclaimable
                figure. Then a measure without both of its figures is left out,
                and otherwise the resident pair is used.
        """
        if rows <= 0:
            return
        if nonreclaimable:
            grown = measure.grown_bytes
            if grown is None:
                return
        else:
            if measure.peak_rss_bytes <= measure.peak_before_rss_bytes:
                return
            grown = max(0, measure.peak_rss_bytes - measure.base_rss_bytes)
        cost = math.ceil(grown / rows)
        with self._lock:
            self._worker_peak_per_row = max(self._worker_peak_per_row, cost)

    @property
    def bytes_per_row(self) -> int | None:
        """The estimate, ``None`` until an observation with rows exceeds the fixed part."""
        with self._lock:
            return self._estimate()

    def predicted_peak(
        self, rows_in_flight: int, rows_next: int, workers_after: int, margin: float
    ) -> int:
        """Predict the tree's peak if an entry of *rows_next* rows starts now.

        Args:
            rows_in_flight: The input rows of the entries already in flight.
            rows_next: The input rows of the entry that would start.
            workers_after: The process workers alive once it starts, 0 in thread
                mode.
            margin: The factor on the rows' memory.

        Returns:
            The fixed part plus *margin* times the rows' estimated memory. Without
            an estimate the rows count nothing.
        """
        with self._lock:
            per_row = self._estimate() or 0
            fixed = self._base + workers_after * self._worker_base
        return fixed + int(margin * per_row * (rows_in_flight + rows_next))

    def _cost(self, observation: _Observation) -> int:
        fixed = self._base + observation.live_workers * self._worker_base
        grown = max(0, observation.measured_bytes - fixed)
        return math.ceil(grown / observation.in_flight_rows)

    def _recompute(self) -> None:
        costs = [self._cost(observation) for observation in self._observations]
        self._per_row = max(costs, default=0)

    def _estimate(self) -> int | None:
        per_row = max(self._per_row, self._worker_peak_per_row)
        return per_row if per_row > 0 else None


def admission_budget(
    *, available_bytes: int, held_bytes: int, limit_bytes: int, floor: float
) -> int:
    """The memory a run's process tree may grow to before the floor is reached.

    What the tree holds now and what is still available under the limit, never
    more than the limit, less the floor. Memory that something outside the tree
    uses, another process on a host or another container under a shared limit, is
    in neither, so it narrows the budget.

    Args:
        available_bytes: The memory available under the limit now.
        held_bytes: The memory of the run's process tree now, in the run's measure.
        limit_bytes: The memory limit that applies.
        floor: The share of the limit that must stay available.

    Returns:
        The budget in bytes, which may be negative when the floor is already
        reached.
    """
    reachable = min(available_bytes + held_bytes, limit_bytes)
    return reachable - int(floor * limit_bytes)


@dataclass(frozen=True, slots=True)
class EntrySpan:
    """One entry's time in flight, from the start of its load to its exit.

    Attributes:
        rows: Its input rows.
        most_in_flight: The most entries in flight at any moment of its span,
            itself included.
        load_started: The monotonic time its load began.
        cpu_at_load: The run's CPU seconds when its load began.
        apply_seconds: From its dispatch to the end of its apply, 0 when either
            was not stamped.
        peak_nonreclaimable_bytes: The largest non-reclaimable memory of any sample
            taken while it was in flight, ``None`` when it saw no sample.
    """

    rows: int
    most_in_flight: int
    load_started: float
    cpu_at_load: float
    apply_seconds: float
    peak_nonreclaimable_bytes: int | None

    def grown_bytes(self, *, base_bytes: int) -> int | None:
        """Return what the entry added over the run's base when it ran alone.

        Args:
            base_bytes: The run's own memory before the first entry loaded, 0
                where it was not measured.

        Returns:
            The largest sampled figure less *base_bytes*, never negative. ``None``
            when the entry shared its span with another, which cannot be told
            apart in one process, when it saw no sample, and when the base was
            not measured.
        """
        if (
            self.most_in_flight != 1
            or self.peak_nonreclaimable_bytes is None
            or base_bytes <= 0
        ):
            return None
        return max(0, self.peak_nonreclaimable_bytes - base_bytes)


@dataclass(slots=True)
class _Flight:
    rows: int
    pair_frames: int | None
    pair_frames_bound: bool
    most: int
    load_started: float
    cpu_at_load: float
    dispatched: float | None = None
    applied: float | None = None
    peak_nonreclaimable: int | None = None


@dataclass(frozen=True, slots=True)
class _WrittenFlight:
    """A flight that ended with its output written, with the tree's CPU at both ends."""

    started: float
    ended: float
    rows: int
    cpu_at_start: float
    cpu_at_end: float


class EntriesInFlight:
    """The entries of a run from the start of their load to their exit, with their rows.

    Every exit of an entry, an output written, a failure or a skip, calls
    :meth:`leave`, and leaving twice is harmless. Every method holds the record's
    lock, because an apply ends on a worker thread. An entry that leaves with its
    output written keeps its span and rows, which :meth:`levels` reads.

    Args:
        cpu_seconds: The CPU seconds of the run's process tree now, read where an
            entry leaves with its output written.
        clock: The clock the entries' ``started`` comes from.
    """

    def __init__(
        self,
        *,
        cpu_seconds: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cpu_seconds: Callable[[], float] = cpu_seconds
        self._clock: Callable[[], float] = clock
        self._lock: threading.Lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}
        self._written: list[_WrittenFlight] = []

    def enter(
        self,
        key: str,
        rows: int,
        *,
        pair_frames: int | None,
        started: float,
        cpu: float,
        pair_frames_bound: bool = False,
    ) -> None:
        """Record that the entry *key* of *rows* input rows starts to load.

        Args:
            key: The entry's key.
            rows: Its input rows.
            pair_frames: Its ordered pairs summed over frames, ``None`` where it
                has no figure.
            started: When its load began, on the clock of this record.
            cpu: The tree's CPU seconds when its load began.
            pair_frames_bound: Whether *pair_frames* is a lower bound in place of a
                measured figure.
        """
        with self._lock:
            self._flights[key] = _Flight(
                rows=rows,
                pair_frames=pair_frames,
                pair_frames_bound=pair_frames_bound,
                most=0,
                load_started=started,
                cpu_at_load=cpu,
            )
            count = len(self._flights)
            for flight in self._flights.values():
                flight.most = max(flight.most, count)

    def observe_memory(self, bytes_: int, *, sampled_from: float | None = None) -> None:
        """Raise each entry in flight's largest sample to *bytes_* when it is lower.

        Args:
            bytes_: The sampled figure.
            sampled_from: When the measurement began, on the clock the entries'
                ``started`` came from. An entry that started after it is left out,
                because the figure was read before the entry held anything. ``None``
                credits every entry in flight.
        """
        with self._lock:
            for flight in self._flights.values():
                if sampled_from is not None and flight.load_started > sampled_from:
                    continue
                if (
                    flight.peak_nonreclaimable is None
                    or bytes_ > flight.peak_nonreclaimable
                ):
                    flight.peak_nonreclaimable = bytes_

    def dispatched(self, key: str, at: float) -> None:
        """Stamp the moment the entry *key* is handed to ``apply``."""
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None:
                flight.dispatched = at

    def applied(self, key: str, at: float) -> None:
        """Stamp the moment the entry *key*'s ``apply`` ended. A later stamp is ignored."""
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None and flight.applied is None:
                flight.applied = at

    def leave(self, key: str, *, written: bool = False) -> EntrySpan | None:
        """Remove the entry *key*, and return its span, ``None`` when it was not in flight.

        Args:
            key: The entry's key.
            written: Whether the entry leaves with its output written. Its start,
                end, rows and the tree's CPU at both ends are then kept for
                :meth:`levels`. An entry that failed or was skipped adds nothing.
        """
        ended = self._clock() if written else 0.0
        cpu_at_end = self._cpu_seconds() if written else 0.0
        with self._lock:
            flight = self._flights.pop(key, None)
            if flight is not None and written:
                self._written.append(
                    _WrittenFlight(
                        started=flight.load_started,
                        ended=ended,
                        rows=flight.rows,
                        cpu_at_start=flight.cpu_at_load,
                        cpu_at_end=cpu_at_end,
                    )
                )
        if flight is None:
            return None
        apply_seconds = (
            flight.applied - flight.dispatched
            if flight.applied is not None and flight.dispatched is not None
            else 0.0
        )
        return EntrySpan(
            rows=flight.rows,
            most_in_flight=flight.most,
            load_started=flight.load_started,
            cpu_at_load=flight.cpu_at_load,
            apply_seconds=max(0.0, apply_seconds),
            peak_nonreclaimable_bytes=flight.peak_nonreclaimable,
        )

    def levels(self) -> tuple[InFlightLevel, ...]:
        """What the run did at each count of entries in flight together.

        Only entries that left with their output written count, so an open flight
        and a failed or skipped entry add neither time nor rows. A level's CPU is
        the difference of the tree's CPU stamps at the two ends of each of its
        segments, summed.

        Returns:
            The levels ascending by ``in_flight``, without level 0.
        """
        with self._lock:
            kept = list(self._written)
        stamps: dict[float, float] = {}
        for flight in kept:
            stamps[flight.started] = max(
                stamps.get(flight.started, 0.0), flight.cpu_at_start
            )
            stamps[flight.ended] = max(stamps.get(flight.ended, 0.0), flight.cpu_at_end)

        def cpu_of_segment(start: float, end: float, flights: tuple[int, ...]) -> float:
            return max(0.0, stamps[end] - stamps[start])

        return sweep_levels(
            [FlightSpan(f.started, f.ended, f.rows) for f in kept], cpu_of_segment
        )

    def rows_of(self, key: str) -> int:
        """The input rows of the entry *key*, 0 when it is not in flight."""
        with self._lock:
            flight = self._flights.get(key)
            return 0 if flight is None else flight.rows

    def rows(self) -> int:
        """The input rows of the entries in flight now."""
        with self._lock:
            return sum(flight.rows for flight in self._flights.values())

    def count(self) -> int:
        """How many entries are in flight now."""
        with self._lock:
            return len(self._flights)

    def totals(
        self, *, sampled_from: float | None = None
    ) -> tuple[int, int, int, bool]:
        """The entries in flight now, their rows and pair frames, read at one moment.

        Args:
            sampled_from: When a measurement began, on the clock the entries'
                ``started`` came from. An entry that started after it is not
                counted, so the totals describe what the measurement saw.
                ``None`` counts every entry in flight.

        Returns:
            The entries, their input rows, their pair frames (an entry without the
            figure adding 0) and whether any of them carried a lower bound.
        """
        with self._lock:
            held = [
                flight
                for flight in self._flights.values()
                if sampled_from is None or flight.load_started <= sampled_from
            ]
            return (
                len(held),
                sum(flight.rows for flight in held),
                sum(flight.pair_frames or 0 for flight in held),
                any(flight.pair_frames_bound for flight in held),
            )
