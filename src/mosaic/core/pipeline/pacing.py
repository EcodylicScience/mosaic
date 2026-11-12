"""Work that a run repeats, paced by what each repetition costs.

A pass over everything a run holds, such as writing the index it appends to or
collecting garbage, costs more as the run grows. Done after each entry it costs more
than the entries do, so a run over thousands of entries is dominated by it. A
:class:`Pacer` lets a pass run again only after a multiple of what the last one cost,
which keeps the passes to a fixed share of a run however large the run is.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

__all__ = ["PacedCollector", "Pacer", "now"]


def now() -> float:
    """The monotonic time that a pacer made without a clock reads.

    A function of this module, read when a pacer is made, so that a test can give
    every pacer one clock that it controls.
    """
    return time.monotonic()


class Pacer:
    """Say when a repeated pass is due, from what the last pass cost.

    A pass is due once the time since the last one ended is at least *interval*
    and at least *cost_factor* times what the last one took, so the passes take at
    most ``1 / (1 + cost_factor)`` of the time once they cost more than the
    interval. The first pass is due *interval* seconds after the pacer is made.

    Args:
        interval: The least time between the end of one pass and the next.
        cost_factor: How many times the cost of a pass must pass before the next.
        clock: The clock that times passes. ``None`` reads :func:`now`.
    """

    def __init__(
        self,
        *,
        interval: float,
        cost_factor: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._interval: float = interval
        self._cost_factor: float = cost_factor
        self._clock: Callable[[], float] = now if clock is None else clock
        self._last_end: float = self._clock()
        self._last_cost: float = 0.0

    def due(self) -> bool:
        """Whether the next pass should run now."""
        return self._clock() - self._last_end >= max(
            self._interval, self._cost_factor * self._last_cost
        )

    def timed[Result](self, action: Callable[[], Result]) -> Result:
        """Run *action* as a pass, and record what it cost."""
        started = self._clock()
        try:
            return action()
        finally:
            self._last_end = self._clock()
            self._last_cost = self._last_end - started


class PacedCollector:
    """Collect garbage after finished entries, no more often than it pays for.

    A full collection walks every live object, so one after each of thousands of
    entries costs more than the entries do. This collects when the outputs finished
    since the last collection are large, and otherwise when the collections would
    take a tenth of the time.

    Args:
        byte_limit: The bytes of finished outputs after which the next entry
            collects, whatever the time since the last collection.
        clock: The clock that times collections. ``None`` reads :func:`now`.
        collect: What a collection runs. ``None`` runs ``gc.collect``.
    """

    def __init__(
        self,
        *,
        byte_limit: int = 256 * 1024 * 1024,
        clock: Callable[[], float] | None = None,
        collect: Callable[[], object] | None = None,
    ) -> None:
        self._byte_limit: int = byte_limit
        self._collect: Callable[[], object] = gc.collect if collect is None else collect
        self._pacer: Pacer = Pacer(interval=0.0, cost_factor=9.0, clock=clock)
        self._uncollected: int = 0

    def finished(self, nbytes: int) -> bool:
        """Note an output of *nbytes* that was just written, and collect if it is time.

        Returns:
            Whether a collection ran.
        """
        self._uncollected += nbytes
        if self._uncollected < self._byte_limit and not self._pacer.due():
            return False
        self._pacer.timed(self._collect)
        self._uncollected = 0
        return True
