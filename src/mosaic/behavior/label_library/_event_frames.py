"""Where an event stamped in seconds lands on an entry's frames.

BORIS and the custom template stamp events in seconds. When the entry has
tracks, a second is the last frame whose track ``time`` is at or before it,
because a table that dropped frames is not timed by ``frame / fps``. Without
tracks, a second is ``int(seconds * fps)``.
"""

from __future__ import annotations

import math

import numpy as np

from mosaic.core.label_converter import TrackClock

__all__ = ["EventFrames"]


class EventFrames:
    """One entry's mapping of event seconds to frames.

    Args:
        fps: The rate of an entry without tracks.
        clock: The dataset's tracks clock, or ``None`` outside a dataset conversion.
        group: The entry's group.
        sequence: The entry's sequence.
    """

    def __init__(
        self, fps: float, clock: TrackClock | None, group: str, sequence: str
    ) -> None:
        self._fps = fps
        self._clock = clock
        self._entry = (group, sequence)

    def span(self, start: float, stop: float) -> tuple[int, int] | None:
        """The first and last frame of an event from *start* to *stop* seconds.

        With tracks, each second is the last track frame at or before it. An event
        that starts before the first track time starts on the first frame, and one
        that stops before it, or starts after the last track time, has no frame
        and gives ``None``. A stop after the last track time ends on the last
        frame.
        """
        bounds = None if self._clock is None else self._clock.bounds(*self._entry)
        if self._clock is None or bounds is None:
            return int(start * self._fps), int(stop * self._fps)
        if start > bounds.last_time or stop < bounds.first_time:
            return None
        found = self._clock.frames_at_or_before(
            *self._entry, np.array([start, stop], dtype=np.float64)
        )
        if found is None:
            return None
        first = bounds.first_frame if math.isnan(found[0]) else int(found[0])
        return first, int(found[1])

    def n_frames(self, last_second: float) -> int:
        """The frames a dense label array spans, up to *last_second* without tracks."""
        bounds = None if self._clock is None else self._clock.bounds(*self._entry)
        if bounds is None:
            return int(np.ceil(last_second * self._fps)) + 1
        return bounds.last_frame + 1
