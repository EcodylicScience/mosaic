"""The memory a feature run's entries cost per input row, learned while it runs.

:class:`RowCostEstimate` keeps the resident memory of the run's process tree beside
the input rows in flight at each measurement, and takes the most memory a row has
cost over the tree's fixed part: the parent's base and, in process mode, a worker's
own memory for each worker process alive at the measurement. Several entries in one
process cannot be told apart, so the figure is per row of everything in flight. In
process mode a worker also reports its high-water mark before and after each entry.
When the mark rose during the entry, the peak was that entry's, and its growth over
the worker's base, per row of that entry, is a further observation, which sees a
peak between two measurements. A mark that did not rise belongs to an earlier entry
on the same worker and is left out. The estimate is the largest observed, so an
underestimate needs every observation to be low. When every observation fell at or
under the fixed part there is no estimate, rather than an estimate of 0 bytes a row.

What the estimate cannot see:

- memory that does not grow with input rows, such as a feature that reads video
  frames, or a pair feature whose cost grows with the square of its individuals;
- allocator memory kept after an entry ends, which inflates the estimate and so
  errs on the safe side;
- a peak between two observations in thread mode, and in process mode one that a
  worker reaches before it has measured its base;
- pages in ``/dev/shm``, the page cache and GPU memory, which a cgroup counts and
  resident memory does not;
- a page several workers map, which is counted once per worker;
- memory outside this run's process tree, which a budget takes from the limit's
  available memory instead;
- a child process of earlier work in the same process, which the tree counts as
  this run's, so one that exits or grows during the run moves the figure.

Imports the standard library only.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "KEPT_OBSERVATIONS",
    "EntriesInFlight",
    "EntrySpan",
    "RowCostEstimate",
    "WorkerMeasure",
    "admission_budget",
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
    """

    base_rss_bytes: int
    peak_before_rss_bytes: int
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class _Observation:
    tree_rss: int
    in_flight_rows: int
    live_workers: int


@dataclass
class RowCostEstimate:
    """The most memory an input row in flight has cost, over the tree's fixed part.

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
        """Set the run's own fixed memory: interpreter, imports and the feature's state."""
        with self._lock:
            self._base = bytes_
            self._recompute()

    def worker_base(self, bytes_: int) -> None:
        """Set one process worker's own memory, and recompute from the kept observations.

        Observations taken before a worker reported counted its base as rows.
        """
        with self._lock:
            self._worker_base = bytes_
            self._recompute()

    def observe(self, tree_rss: int, in_flight_rows: int, live_workers: int) -> None:
        """Keep one measurement of the tree beside the rows and workers live at it.

        An observation with no rows in flight is left out.

        Args:
            tree_rss: The resident memory of the run's process tree.
            in_flight_rows: The input rows of the entries in flight.
            live_workers: The run's worker processes alive at the measurement, 0
                in thread mode.
        """
        if in_flight_rows <= 0:
            return
        observation = _Observation(tree_rss, in_flight_rows, live_workers)
        with self._lock:
            self._observations.append(observation)
            self._per_row = max(self._per_row, self._cost(observation))

    def observe_worker(self, measure: WorkerMeasure, rows: int) -> None:
        """Keep a process worker's growth over its own base while it applied one entry.

        The high-water mark is the worker's whole life, so it is the entry's peak
        only when it rose past the mark read before the entry. Otherwise an earlier
        entry on the same worker set it, and the observation is left out.

        Args:
            measure: What the worker measured of itself around the entry.
            rows: The entry's input rows. An entry without rows is left out.
        """
        if rows <= 0 or measure.peak_rss_bytes <= measure.peak_before_rss_bytes:
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
        grown = max(0, observation.tree_rss - fixed)
        return math.ceil(grown / observation.in_flight_rows)

    def _recompute(self) -> None:
        costs = [self._cost(observation) for observation in self._observations]
        self._per_row = max(costs, default=0)

    def _estimate(self) -> int | None:
        per_row = max(self._per_row, self._worker_peak_per_row)
        return per_row if per_row > 0 else None


def admission_budget(
    *, available_bytes: int, tree_rss_bytes: int, limit_bytes: int, floor: float
) -> int:
    """The memory a run's process tree may grow to before the floor is reached.

    What the tree holds now and what is still available under the limit, never
    more than the limit, less the floor. Memory that something outside the tree
    uses, another process on a host or another container under a shared limit, is
    in neither, so it narrows the budget.

    Args:
        available_bytes: The memory available under the limit now.
        tree_rss_bytes: The resident memory of the run's process tree now.
        limit_bytes: The memory limit that applies.
        floor: The share of the limit that must stay available.

    Returns:
        The budget in bytes, which may be negative when the floor is already
        reached.
    """
    reachable = min(available_bytes + tree_rss_bytes, limit_bytes)
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
    """

    rows: int
    most_in_flight: int
    load_started: float
    cpu_at_load: float
    apply_seconds: float


@dataclass(slots=True)
class _Flight:
    rows: int
    most: int
    load_started: float
    cpu_at_load: float
    dispatched: float | None = None
    applied: float | None = None


class EntriesInFlight:
    """The entries of a run from the start of their load to their exit, with their rows.

    Every exit of an entry, an output written, a failure or a skip, calls
    :meth:`leave`, and leaving twice is harmless. Every method holds the record's
    lock, because an apply ends on a worker thread.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}

    def enter(self, key: str, rows: int, *, started: float, cpu: float) -> None:
        """Record that the entry *key* of *rows* input rows starts to load."""
        with self._lock:
            self._flights[key] = _Flight(
                rows=rows, most=0, load_started=started, cpu_at_load=cpu
            )
            count = len(self._flights)
            for flight in self._flights.values():
                flight.most = max(flight.most, count)

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

    def leave(self, key: str) -> EntrySpan | None:
        """Remove the entry *key*, and return its span, ``None`` when it was not in flight."""
        with self._lock:
            flight = self._flights.pop(key, None)
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
