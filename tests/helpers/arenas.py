"""Count the arenas index reads of a run, and make the index unreadable.

The lookup of the arenas root reads the index and the media cameras once per
process for each state of the two files. :class:`Counted` counts those reads for
a test to assert, and :func:`refuse_index` damages the index as a hand edit can.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mosaic.core.pipeline.arena_index as arena_index_module
from mosaic.core.arenas import ArenaKey
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.arena_index import IndexedLayout, arena_index_path

__all__ = ["Counted", "refuse_index"]


class Counted:
    """Counts the calls of the two index reads the arena lookup makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows = 0
        self.cameras = 0
        read_rows = arena_index_module.read_arena_index_rows
        read_cameras = arena_index_module.media_cameras

        def rows(ds: Dataset) -> dict[ArenaKey, IndexedLayout]:
            self.rows += 1
            return read_rows(ds)

        def cameras(ds: Dataset) -> dict[Entry, set[str]]:
            self.cameras += 1
            return read_cameras(ds)

        monkeypatch.setattr(arena_index_module, "read_arena_index_rows", rows)
        monkeypatch.setattr(arena_index_module, "media_cameras", cameras)


def refuse_index(ds: Dataset) -> None:
    """Make the arenas index unreadable, as a hand edit can."""
    path = arena_index_path(ds)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows["n_arenas"] = "many"
    rows.to_csv(path, index=False)
