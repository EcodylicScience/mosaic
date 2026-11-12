"""The bounds of a row filter on a run: a time window, and the one-per-boundary rule.

A leaf with no imports of its own, because a recipe's ``run`` block validates
against it and a status read must not load pandas for that.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["TimeWindow", "check_one_filter_per_boundary"]


class TimeWindow(NamedTuple):
    """A window of ``time`` in seconds, half-open: ``start <= time < end``.

    A bound left ``None`` is open. The window filters rows on the ``time`` column
    and is never converted to frames, so it needs no frame rate and keeps exactly
    the rows whose ``time`` falls in it on a table that dropped frames.
    """

    start: float | None = None
    end: float | None = None

    @property
    def is_set(self) -> bool:
        """Whether either bound is set."""
        return self.start is not None or self.end is not None


def check_one_filter_per_boundary(
    start_frame: int | None,
    end_frame: int | None,
    start_time: float | None,
    end_time: float | None,
) -> None:
    """Refuse a boundary that is set as a frame and as a time.

    Raises:
        ValueError: If a frame and a time are both set for the start, or both
            for the end.
    """
    for boundary, frame, time in (
        ("start", start_frame, start_time),
        ("end", end_frame, end_time),
    ):
        if frame is not None and time is not None:
            msg = f"set one of filter_{boundary}_frame and filter_{boundary}_time"
            raise ValueError(msg)
