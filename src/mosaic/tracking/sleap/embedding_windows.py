"""Plans the training windows of one entry from its tracked instances.

Window ``w`` covers frames ``w * window_frames`` to ``(w + 1) * window_frames - 1``.
A frame of a window qualifies when at least ``min_tracks_present`` tracks have an
instance in it, and a window takes ``frames_per_window`` of its qualifying frames,
evenly spaced, or all of them when it has fewer. A window with no qualifying frame
is left out. :func:`keep_evenly_spaced_windows` then thins an entry's windows to a
count, evenly spaced by position in window order, for an entry that needs fewer.

Every instance of a chosen frame is kept under a window track name,
``{entry_key}/w{w:05d}/{track}``, which no other window shares. With
``split_tracks_at_gap`` set, a track whose frames inside a window are more than
that many frames apart is cut there into pieces, and piece ``k`` is named
``{track}.{k}``.

Selection reads frame indices alone and draws nothing. Every order is sorted, so
the plan does not depend on the order of the readback's rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from mosaic.core.track_library.sleap import AppearanceReadback

__all__ = [
    "EmbeddingWindow",
    "EntryWindows",
    "WindowInstance",
    "keep_evenly_spaced_windows",
    "plan_windows",
]


@dataclass(frozen=True, slots=True)
class WindowInstance:
    """One tracked instance kept for training.

    Attributes:
        frame: The instance's frame index.
        track: The tracker's name for its track.
        window_track: The name of its track in the training labels.
    """

    frame: int
    track: str
    window_track: str


@dataclass(frozen=True, slots=True)
class EmbeddingWindow:
    """The frames chosen from one window and the instances kept in them.

    Attributes:
        index: The window's index ``w``.
        frames: The chosen frames, ascending.
        instances: Every instance of a chosen frame, ordered by frame, then track.
    """

    index: int
    frames: tuple[int, ...]
    instances: tuple[WindowInstance, ...]


@dataclass(frozen=True, slots=True)
class EntryWindows:
    """The planned windows of one entry.

    Attributes:
        entry_key: The entry the windows are cut from.
        windows: The windows with a qualifying frame, by ascending index.
    """

    entry_key: str
    windows: tuple[EmbeddingWindow, ...]


def plan_windows(
    presence: AppearanceReadback,
    *,
    entry_key: str,
    window_frames: int,
    frames_per_window: int,
    min_tracks_present: int,
    split_tracks_at_gap: int | None,
) -> EntryWindows:
    """Plan the training windows of one entry.

    Args:
        presence: The entry's tracked instances. Only their frames and track
            names are read.
        entry_key: The entry's key, which begins every window track name.
        window_frames: The length of one window.
        frames_per_window: How many qualifying frames a window takes.
        min_tracks_present: How many tracks a qualifying frame holds at least.
        split_tracks_at_gap: The largest step between two frames of one track
            inside a window that keeps the track whole, or ``None`` to keep
            every track whole.

    Returns:
        The windows that have a qualifying frame.
    """
    tracks_by_frame: dict[int, set[str]] = defaultdict(set)
    for frame, track in zip(
        presence.frame_idx.tolist(), presence.track_name.tolist(), strict=True
    ):
        tracks_by_frame[int(frame)].add(str(track))

    frames_by_window: dict[int, list[int]] = defaultdict(list)
    for frame in sorted(tracks_by_frame):
        frames_by_window[frame // window_frames].append(frame)

    windows: list[EmbeddingWindow] = []
    for index in sorted(frames_by_window):
        frames = frames_by_window[index]
        qualifying = [
            frame
            for frame in frames
            if len(tracks_by_frame[frame]) >= min_tracks_present
        ]
        if not qualifying:
            continue
        chosen = _evenly_spaced(qualifying, frames_per_window)
        names = _window_track_names(
            frames, tracks_by_frame, split_tracks_at_gap, f"{entry_key}/w{index:05d}"
        )
        instances = tuple(
            WindowInstance(frame=frame, track=track, window_track=names[frame, track])
            for frame in chosen
            for track in sorted(tracks_by_frame[frame])
        )
        windows.append(EmbeddingWindow(index=index, frames=chosen, instances=instances))
    return EntryWindows(entry_key=entry_key, windows=tuple(windows))


def keep_evenly_spaced_windows(plan: EntryWindows, count: int) -> EntryWindows:
    """Thin an entry's windows to *count*, evenly spaced by position in window order.

    The windows are ordered by ascending index. The first and the last are kept,
    and the others are spaced evenly by position among the windows present, so a
    gap in the indices does not shorten the result. An entry with *count* windows
    or fewer is returned whole. Nothing is drawn at random.

    Args:
        plan: The entry's planned windows.
        count: How many windows to keep, at least one.

    Returns:
        The kept windows, still by ascending index.
    """
    return EntryWindows(
        entry_key=plan.entry_key, windows=_evenly_spaced(plan.windows, count)
    )


def _evenly_spaced[T](items: Sequence[T], count: int) -> tuple[T, ...]:
    """Take *count* of the ascending *items*, the first and last among them.

    Position ``i`` of ``count`` is ``i * (n - 1) // (count - 1)``. When there are
    more items than *count*, each step exceeds one position and no item is taken
    twice.
    """
    n = len(items)
    if n <= count:
        return tuple(items)
    if count == 1:
        return (items[0],)
    return tuple(items[i * (n - 1) // (count - 1)] for i in range(count))


def _window_track_names(
    frames: list[int],
    tracks_by_frame: dict[int, set[str]],
    split_tracks_at_gap: int | None,
    prefix: str,
) -> dict[tuple[int, str], str]:
    """Name every ``(frame, track)`` of one window's ascending *frames*."""
    frames_by_track: dict[str, list[int]] = defaultdict(list)
    for frame in frames:
        for track in tracks_by_frame[frame]:
            frames_by_track[track].append(frame)

    names: dict[tuple[int, str], str] = {}
    for track, track_frames in frames_by_track.items():
        pieces = _pieces(track_frames, split_tracks_at_gap)
        for k, piece in enumerate(pieces):
            name = f"{prefix}/{track}.{k}" if len(pieces) > 1 else f"{prefix}/{track}"
            for frame in piece:
                names[frame, track] = name
    return names


def _pieces(frames: list[int], split_at_gap: int | None) -> list[list[int]]:
    """Cut the ascending *frames* wherever two neighbors are more than the gap apart."""
    if split_at_gap is None:
        return [frames]
    pieces = [[frames[0]]]
    for previous, frame in zip(frames, frames[1:], strict=False):
        if frame - previous > split_at_gap:
            pieces.append([])
        pieces[-1].append(frame)
    return pieces
