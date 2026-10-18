"""Measures the memory that a run and its worker processes use, and the host's remainder.

A run with several workers is where memory runs out: each process-mode worker is an
interpreter with its own copy of the feature stack and of every table it is handed.
:class:`MemoryWatch` reports the resident memory of the process tree, which contains
the workers, and the memory the host still has available.

Resident memory counts a page once for every process that maps it. Thread mode lives in
one process and is exact. Process mode overstates a little where workers share
libraries.

Imports the standard library only. ``psutil`` is imported inside the methods, as
``markers`` does, and a host where it cannot be imported reports nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

__all__ = [
    "SAMPLE_INTERVAL_SECONDS",
    "MemorySample",
    "MemoryWatch",
    "host_memory_bytes",
]

SAMPLE_INTERVAL_SECONDS: Final = 2.0
"""The shortest time between two measurements that :meth:`MemoryWatch.sample_if_due` takes."""


def host_memory_bytes() -> tuple[int, int]:
    """Return the host's physical and available memory, in bytes.

    Returns:
        ``(total, available)``, or ``(0, 0)`` where ``psutil`` is not importable.
    """
    try:
        import psutil
    except ImportError:
        return 0, 0
    host = psutil.virtual_memory()
    return int(host.total), int(host.available)


@dataclass(frozen=True)
class MemorySample:
    """One measurement of memory, in bytes.

    Attributes:
        tree_rss_bytes: The resident memory of this process and every process
            below it.
        available_bytes: The memory the host can give a process without swapping.
        total_bytes: The host's physical memory.
    """

    tree_rss_bytes: int
    available_bytes: int
    total_bytes: int


class MemoryWatch:
    """Measure memory repeatedly and keep the extremes.

    Attributes:
        peak_rss_bytes: The largest ``tree_rss_bytes`` measured, 0 before the first.
        min_available_bytes: The smallest ``available_bytes`` measured, 0 before the
            first.
        total_bytes: The host's physical memory at the last measurement.
    """

    def __init__(self, *, interval_seconds: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self._interval: float = interval_seconds
        self._last: float | None = None
        self.peak_rss_bytes: int = 0
        self.min_available_bytes: int = 0
        self.total_bytes: int = 0

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
        try:
            root = psutil.Process()
            members = [root, *root.children(recursive=True)]
        except psutil.Error:
            members = []
        resident = 0
        for member in members:
            try:
                resident += int(member.memory_info().rss)
            except psutil.Error:
                continue
        sample = MemorySample(
            tree_rss_bytes=resident,
            available_bytes=int(host.available),
            total_bytes=int(host.total),
        )
        self.peak_rss_bytes = max(self.peak_rss_bytes, sample.tree_rss_bytes)
        if self.min_available_bytes == 0:
            self.min_available_bytes = sample.available_bytes
        else:
            self.min_available_bytes = min(
                self.min_available_bytes, sample.available_bytes
            )
        self.total_bytes = sample.total_bytes
        self._last = time.monotonic()
        return sample

    def sample_if_due(self) -> bool:
        """Measure when the interval has passed since the last measurement.

        Returns:
            Whether a measurement was taken.
        """
        if self._last is not None and time.monotonic() - self._last < self._interval:
            return False
        return self.sample() is not None

    def free_fraction(self) -> float | None:
        """Return the share of the host's memory that is available now.

        Returns:
            A fraction from 0 to 1, or ``None`` where ``psutil`` is not importable.
        """
        try:
            import psutil
        except ImportError:
            return None
        host = psutil.virtual_memory()
        if host.total <= 0:
            return None
        return float(host.available) / float(host.total)
