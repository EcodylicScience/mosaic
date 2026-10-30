"""Training windows are planned from one entry's tracked instances by frame index."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from mosaic.core.track_library.sleap import AppearanceReadback
from mosaic.tracking.sleap.embedding_windows import (
    EmbeddingWindow,
    EntryWindows,
    WindowInstance,
    plan_windows,
)


def _presence(pairs: Sequence[tuple[int, str]]) -> AppearanceReadback:
    n = len(pairs)
    return AppearanceReadback(
        frame_idx=np.asarray([frame for frame, _ in pairs], dtype=np.int64),
        track_name=np.asarray([track for _, track in pairs], dtype=np.str_),
        is_user=np.zeros(n, dtype=np.bool_),
        has_vector=np.zeros(n, dtype=np.bool_),
        vectors=np.zeros((n, 0), dtype=np.float32),
    )


def _everywhere(frames: Iterable[int], tracks: Sequence[str]) -> list[tuple[int, str]]:
    return [(frame, track) for frame in frames for track in tracks]


def _plan(
    pairs: Sequence[tuple[int, str]],
    *,
    window_frames: int = 10,
    frames_per_window: int = 16,
    min_tracks_present: int = 2,
    split_tracks_at_gap: int | None = None,
) -> EntryWindows:
    return plan_windows(
        _presence(pairs),
        entry_key="cam1",
        window_frames=window_frames,
        frames_per_window=frames_per_window,
        min_tracks_present=min_tracks_present,
        split_tracks_at_gap=split_tracks_at_gap,
    )


def test_a_frame_qualifies_with_enough_tracks_present() -> None:
    pairs = _everywhere(range(10), ["a"]) + [(2, "b"), (3, "b"), (5, "b")]

    plan = _plan(pairs)

    assert plan == EntryWindows(
        entry_key="cam1",
        windows=(
            EmbeddingWindow(
                index=0,
                frames=(2, 3, 5),
                instances=tuple(
                    WindowInstance(
                        frame=frame, track=track, window_track=f"cam1/w00000/{track}"
                    )
                    for frame in (2, 3, 5)
                    for track in ("a", "b")
                ),
            ),
        ),
    )


def test_a_higher_minimum_leaves_fewer_frames() -> None:
    pairs = _everywhere(range(10), ["a", "b"]) + [(4, "c"), (7, "c")]

    plan = _plan(pairs, min_tracks_present=3)

    assert [window.frames for window in plan.windows] == [(4, 7)]


def test_frames_are_spaced_evenly_over_the_qualifying_frames() -> None:
    pairs = _everywhere(range(100), ["a", "b"])

    plan = _plan(pairs, window_frames=100, frames_per_window=4)

    assert [window.frames for window in plan.windows] == [(0, 33, 66, 99)]


def test_spacing_follows_the_qualifying_frames_rather_than_the_window() -> None:
    qualifying = [0, 1, 2, 3, 50, 51, 52, 53, 54]
    pairs = _everywhere(range(100), ["a"]) + [(frame, "b") for frame in qualifying]

    plan = _plan(pairs, window_frames=100, frames_per_window=3)

    assert [window.frames for window in plan.windows] == [(0, 50, 54)]


def test_a_window_without_a_qualifying_frame_is_skipped() -> None:
    pairs = (
        _everywhere(range(10), ["a", "b"])
        + _everywhere(range(10, 20), ["a"])
        + _everywhere(range(20, 30), ["a", "b"])
    )

    plan = _plan(pairs, frames_per_window=2)

    assert [(window.index, window.frames) for window in plan.windows] == [
        (0, (0, 9)),
        (2, (20, 29)),
    ]


def test_an_entry_without_a_qualifying_frame_plans_no_window() -> None:
    assert _plan(_everywhere(range(30), ["a"])).windows == ()
    assert _plan([]).windows == ()


def test_a_gap_splits_a_track_into_pieces_inside_one_window() -> None:
    pairs = (
        _everywhere([0, 1, 2, 3, 8, 9], ["a"])
        + _everywhere(range(10), ["b"])
        + _everywhere(range(10, 20), ["a", "b"])
    )

    plan = _plan(pairs, split_tracks_at_gap=2)

    first, second = plan.windows
    assert {(i.frame, i.window_track) for i in first.instances if i.track == "a"} == {
        (0, "cam1/w00000/a.0"),
        (1, "cam1/w00000/a.0"),
        (2, "cam1/w00000/a.0"),
        (3, "cam1/w00000/a.0"),
        (8, "cam1/w00000/a.1"),
        (9, "cam1/w00000/a.1"),
    }
    assert {i.window_track for i in first.instances if i.track == "b"} == {
        "cam1/w00000/b"
    }
    assert {i.window_track for i in second.instances} == {
        "cam1/w00001/a",
        "cam1/w00001/b",
    }


def test_a_gap_within_the_limit_keeps_a_track_whole() -> None:
    pairs = _everywhere([0, 1, 2, 3, 8, 9], ["a"]) + _everywhere(range(10), ["b"])

    for gap in (None, 5):
        plan = _plan(pairs, split_tracks_at_gap=gap)
        names = {i.window_track for i in plan.windows[0].instances}
        assert names == {"cam1/w00000/a", "cam1/w00000/b"}


def test_track_names_are_unique_across_windows() -> None:
    pairs = _everywhere(range(40), ["a", "b"])

    plan = _plan(pairs, frames_per_window=3)

    names = [{i.window_track for i in window.instances} for window in plan.windows]
    assert names == [{f"cam1/w{w:05d}/a", f"cam1/w{w:05d}/b"} for w in range(4)]
    assert len(frozenset[str]().union(*names)) == 8


def test_the_plan_does_not_depend_on_the_row_order() -> None:
    pairs = (
        _everywhere([0, 1, 2, 3, 8, 9], ["a"])
        + _everywhere(range(25), ["b", "c"])
        + [(14, "a"), (21, "a")]
    )
    reordered = pairs[::-1]
    interleaved = pairs[1::2] + pairs[::2]

    plans = [
        _plan(rows, frames_per_window=4, split_tracks_at_gap=2)
        for rows in (pairs, reordered, interleaved)
    ]

    assert plans[0] == plans[1] == plans[2]
    for window in plans[0].windows:
        keys = [(i.frame, i.track) for i in window.instances]
        assert keys == sorted(keys)
