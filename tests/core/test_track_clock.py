"""``TrackClock.frames_at_or_after``: the first frame at or after an event second."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mosaic.core.label_converter import TrackClock

_TRACKS = (
    np.arange(10, dtype=np.float64) * 2.0 + 100.0,
    np.arange(10, dtype=np.float64) * 0.1,
)


def _lookup(
    group: str, sequence: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    return _TRACKS if (group, sequence) == ("g", "with-tracks") else None


def test_the_clock_returns_the_frames_and_marks_the_entry_used() -> None:
    clock = TrackClock(_lookup)
    frames = clock.frames_at_or_after("g", "with-tracks", np.array([0.3, 0.35, 5.0]))
    assert frames is not None
    assert frames[0] == 106.0
    assert frames[1] == 108.0
    assert np.isnan(frames[2])
    assert clock.used == {("g", "with-tracks")}


def test_an_entry_without_tracks_answers_none_and_stays_unused() -> None:
    clock = TrackClock(_lookup)
    assert clock.frames_at_or_after("g", "bare", np.array([0.3])) is None
    assert clock.used == set()


def test_a_skipped_entry_answers_none_and_stays_unused() -> None:
    clock = TrackClock(_lookup, skip=lambda group, sequence: sequence == "with-tracks")
    assert clock.frames_at_or_after("g", "with-tracks", np.array([0.3])) is None
    assert clock.used == set()
