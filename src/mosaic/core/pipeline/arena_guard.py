"""Refuse a feature that relates individuals across arenas on an entry with several.

A feature declares ``relates`` (:data:`~mosaic.core.pipeline.types.feature.RelatesLevel`).
One that is ``"arena-blind"`` measures across individuals without
:func:`~mosaic.core.individuals.individual_pairs`, such as a nearest neighbor,
so on a camera of several tanks it relates animals that never met.
``run_feature`` asks :func:`refuse_arena_blind` of each entry before ``apply``,
and serves no cached output of an entry whose layout names several arenas,
because declaring a layout changes no tracks composition.

The count of an entry's arenas is the ``n_arenas`` of the arenas index rows that
:func:`indexed_arena_rows` reads, so the guard opens no layout file and a layout
file that cannot be read does not refuse a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.arenas import ArenaKey
from mosaic.core.pipeline.arena_index import ArenaIndexError, cached_arena_lookup
from mosaic.core.recording import RecordingKey
from mosaic.core.schema import ARENA_COLUMN

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.arena_index import IndexedLayout

__all__ = [
    "ArenaBlindFeature",
    "entry_arena_count",
    "indexed_arena_rows",
    "refuse_arena_blind",
]


class ArenaBlindFeature(ValueError):
    """An arena-blind feature was asked to relate the individuals of several arenas."""


def indexed_arena_rows(ds: Dataset) -> Mapping[ArenaKey, IndexedLayout]:
    """Return the arenas index's row of each key, without opening a layout file.

    Raises:
        ArenaIndexError: If the arenas index cannot be read.
    """
    lookup = cached_arena_lookup(ds)
    if lookup.refusal is not None:
        raise ArenaIndexError(lookup.refusal)
    return lookup.rows


def entry_arena_count(
    ds: Dataset,
    group: str,
    sequence: str,
    rows: Mapping[ArenaKey, IndexedLayout],
) -> int:
    """Return the most arenas that a layout of the entry's recording names.

    Every camera's layout counts, and a ``(G, "")`` layout counts only for a
    sequence of the continuous group ``G``. ``0`` when no layout applies. The
    count is each row's ``n_arenas``.
    """
    own = RecordingKey(group, sequence)
    recording = ds.recording_of(group, sequence)
    return max(
        (
            row.n_arenas
            for key, row in rows.items()
            if key.recording == own or (not key.sequence and key.recording == recording)
        ),
        default=0,
    )


def refuse_arena_blind(name: str, df: pd.DataFrame, arena_count: int) -> None:
    """Refuse an arena-blind feature's input of one entry that spans arenas.

    Args:
        name: The feature, for the message.
        df: The entry's input.
        arena_count: What :func:`entry_arena_count` returns for the entry.

    Raises:
        ArenaBlindFeature: If *df* holds two or more arenas, or holds no
            ``arena`` while the recording's layout names two or more.
    """
    if ARENA_COLUMN in df.columns:
        held = sorted({str(arena) for arena in df[ARENA_COLUMN].dropna()})
        if len(held) < 2:
            return
        message = (
            f"{name} relates individuals across arenas, and this entry holds "
            f"{len(held)} ({', '.join(held[:4])}{'...' if len(held) > 4 else ''}). "
            "Its measures would pair animals of different arenas. Use a feature "
            "that relates individuals within one arena, or one entry per arena."
        )
        raise ArenaBlindFeature(message)
    if arena_count >= 2:
        message = (
            f"{name} relates individuals across arenas, and this entry's "
            f"recording has a layout of {arena_count} arenas while its input "
            "names none. Read tracks that assign-arenas wrote, and a feature that "
            "relates individuals within one arena."
        )
        raise ArenaBlindFeature(message)
