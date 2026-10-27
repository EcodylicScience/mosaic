"""A minimal stand-in for :class:`~mosaic.core.dataset.Dataset`.

The pipeline reaches a dataset through a handful of methods -- root lookup, the
two halves of stored-path resolution, and the metadata accessors -- so a run can
be exercised against a directory tree without building a real dataset.

One class rather than one per test module. Two modules used to keep their own
near-identical copies, and adding a method to the real dataset's surface then
broke whichever one was forgotten; the copies could also drift apart on what
they claimed the dataset does.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from mosaic.core.arenas import ArenaKey
from mosaic.core.entry import Entry
from mosaic.core.pipeline.arena_index import DeclaredLayout
from mosaic.core.pipeline.entry_keys import index_entries
from mosaic.core.recording import (
    OrderFacts,
    Recording,
    RecordingKey,
    RecordingKind,
    recording_kind,
    recording_of,
    recordings,
)


class MockDataset:
    """A dataset-shaped object over *root*, with no manifest behind it."""

    def __init__(
        self,
        root: Path,
        continuous_groups: tuple[str, ...] = (),
        division_facts: Mapping[Entry, OrderFacts] | None = None,
    ) -> None:
        self._root = root
        self._continuous_groups = continuous_groups
        self._division_facts = dict(division_facts or {})
        for directory in ("tracks", "features"):
            (root / directory).mkdir(parents=True, exist_ok=True)

    @property
    def continuous_groups(self) -> tuple[str, ...]:
        return self._continuous_groups

    def recording_kind(self, group: str) -> RecordingKind:
        return recording_kind(self._continuous_groups, group)

    def recording_of(self, group: str, sequence: str) -> RecordingKey:
        return recording_of(self._continuous_groups, group, sequence)

    def division_facts(self, entries: Iterable[Entry]) -> dict[Entry, OrderFacts]:
        """Return the division facts given at construction for *entries*."""
        return {
            entry: self._division_facts[entry]
            for entry in entries
            if entry in self._division_facts
        }

    def recordings(
        self,
        entries: Iterable[Entry] | None = None,
        *,
        order_facts: Mapping[Entry, OrderFacts] | None = None,
    ) -> tuple[Recording, ...]:
        """Return the recordings of *entries*, or of every entry the tracks index names."""
        universe = (
            index_entries(self._root / "tracks" / "index.csv")
            if entries is None
            else entries
        )
        return recordings(self._continuous_groups, universe, order_facts)

    def arena_layouts(self) -> dict[ArenaKey, DeclaredLayout]:
        """Return no arena layout: a stand-in declares none."""
        return {}

    @property
    def base_dir(self) -> Path:
        return self._root

    def get_root(self, key: str) -> Path:
        return self._root / key

    def resolve_path(self, stored_path: object, anchor: object = None) -> Path:
        _ = anchor
        path = Path(str(stored_path))
        return path if path.is_absolute() else self._root / path

    def relative_to_root(self, path: object) -> str:
        try:
            return str(Path(str(path)).resolve().relative_to(self._root.resolve()))
        except ValueError:
            return str(path)

    @property
    def meta(self) -> dict[str, object]:
        return {"fps_default": 30.0}

    def meta_section(self, key: str) -> dict[str, object]:
        value = self.meta.get(key)
        return value if isinstance(value, dict) else {}

    def meta_float(self, key: str, default: float | None = None) -> float | None:
        value = self.meta.get(key)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return default
