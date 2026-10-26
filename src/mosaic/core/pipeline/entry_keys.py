"""The entries an index names, read for the key-collision check at a write.

A write refuses a new entry whose key names a different entry already in its
own index or in the media index, or that takes a continuous group's name
(``Dataset.refuse_entry_key_collision``).
The two columns are read as text with the standard ``csv`` module, which keeps
a zero-padded name as written and costs one pass over the file.

:func:`append_refusing_key_collision` reads the target index under its lock,
so two writers of one index cannot both pass. The media index is read without
its lock, and a media write racing an index write can still pass both checks.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.entry import Entry
from mosaic.core.pipeline.index_csv import IndexCSV, SchemaRowBase, index_records
from mosaic.core.pipeline.index_lock import index_lock

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "append_refusing_key_collision",
    "entries_kept_beside",
    "entries_of_rows",
    "index_entries",
    "key_collision_at_write",
    "media_index_entries",
    "root_index_entries",
]


def entries_of_rows(rows: Iterable[Mapping[str, object]]) -> set[Entry]:
    """Return the ``(group, sequence)`` of each row, a blank cell read as ``""``."""
    return {(_text(row.get("group")), _text(row.get("sequence"))) for row in rows}


def entries_kept_beside(
    written: Iterable[Entry],
    committed: Iterable[Mapping[str, object]],
    preserved: Iterable[Mapping[str, object]],
    *,
    held_elsewhere: Iterable[Entry],
) -> set[Entry]:
    """Return the entries that remain beside *written* once a replacing write lands.

    A replacing write keeps the *preserved* rows of *committed* and drops the
    rest. An entry of a dropped row leaves the index, so a written entry may
    take its key, unless *held_elsewhere*, the entries of the dataset's other
    indexes, still names it. A committed entry that *written* rewrites stays
    existing, which keeps a collision recorded before the check accepted on a
    rewrite.
    """
    remaining = set(written) | set(held_elsewhere)
    return entries_of_rows(preserved) | (entries_of_rows(committed) & remaining)


def _text(value: object) -> str:
    """Return a cell as text, with an absent or NaN cell as ``""``."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def index_entries(path: Path) -> set[Entry]:
    """Return every entry the ``index.csv`` at *path* names, empty when absent."""
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sequence" not in reader.fieldnames:
            return set()
        return entries_of_rows(reader)


def root_index_entries(ds: Dataset, root_key: str) -> set[Entry]:
    """Return every entry of the index at the top of *root_key*, empty when undeclared."""
    if not ds.has_root(root_key):
        return set()
    return index_entries(ds.get_root(root_key) / "index.csv")


def media_index_entries(ds: Dataset) -> set[Entry]:
    """Return every entry of the originals media index."""
    return root_index_entries(ds, ds.resolve_media_root())


def key_collision_at_write(
    ds: Dataset, written: Iterable[Entry], index_path: Path
) -> str | None:
    """Return why a new entry of *written* would name a different indexed entry.

    The existing entries are those of the index at *index_path* and of the
    media index, read without a lock. ``None`` when no new entry collides.
    """
    existing = index_entries(index_path) | media_index_entries(ds)
    return ds.entry_key_collision(written, existing)


def append_refusing_key_collision[RowT: SchemaRowBase](
    ds: Dataset, index: IndexCSV[RowT], rows: list[RowT], written: Iterable[Entry]
) -> None:
    """Append *rows* to *index*, refusing a new entry of *written* that collides.

    The check runs once before the lock, because taking the lock creates an
    absent index and a refused first write would leave it empty. It runs again
    inside the hold of the lock that appends, so a row another writer appends
    first is part of the check.

    Raises:
        EntryKeyCollisionError: If a new entry shares its key with an entry of
            *index* or of the media index, or takes a continuous group's name.
    """
    entries = list(written)
    ds.refuse_entry_key_collision(
        entries, index_entries(index.path) | media_index_entries(ds)
    )
    with index_lock(index.path):
        existing = entries_of_rows(index_records(index.read_holding_lock()))
        ds.refuse_entry_key_collision(entries, existing | media_index_entries(ds))
        index.append_holding_lock(rows)
