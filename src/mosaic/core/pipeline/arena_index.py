"""The arenas root: one layout file per ``(recording, camera)`` and their index.

``arenas/index.csv`` has one row per layout key, naming the layout's file and
the digest of its content. A file is named
``<recording_storage_name>[.<camera>].json``. :func:`write_arena_layout` writes
the file and the row under the index lock. :func:`read_arena_index_rows` reads
the rows, :func:`entry_arena_row` finds the row that one entry and its cameras
look up, and :func:`read_arena_layout` reads that row's file. None of them takes
the lock or writes.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.arenas import ArenaKey, ArenaLayout, layout_lookup_order
from mosaic.core.entry import Entry
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    IndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.recording import RecordingKey, recording_storage_name
from mosaic.core.strict_model import terse

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "ARENAS_ROOT",
    "ArenaIndexError",
    "ArenaLayoutRow",
    "DeclaredLayout",
    "IndexedLayout",
    "arena_index",
    "arena_index_path",
    "arena_layout_file_name",
    "entry_arena_row",
    "media_cameras",
    "read_arena_index_rows",
    "read_arena_layout",
    "read_arena_layouts",
    "read_declared_layout",
    "write_arena_layout",
]

ARENAS_ROOT: Final = "arenas"
"""The root key, which the default manifest also uses as the directory name."""

_KEY_COLUMNS: Final = ("group", "sequence", "camera")


class ArenaIndexError(ValueError):
    """The arenas index holds two layouts of one key, or a file it cannot use."""


@dataclass(frozen=True, slots=True)
class ArenaLayoutRow(IndexRowBase):
    """Records one declared layout.

    ``abs_path`` names the layout's file, dataset-relative.

    Attributes:
        group: The recording key's group.
        sequence: The recording key's sequence, ``""`` for a divided recording.
        camera: The camera, ``""`` for every camera.
        digest: :meth:`ArenaLayout.digest` of the file's layout.
        n_arenas: The layout's arenas.
    """

    group: str
    sequence: str
    camera: str
    digest: str
    n_arenas: int


_COLUMNS: Final = tuple(field.name for field in fields(ArenaLayoutRow))


@dataclass(frozen=True, slots=True)
class IndexedLayout:
    """One row of the arenas index as stored, before its file is opened.

    It checks no path, so a row that names a file by the absolute path of
    another machine reads as a row, and :func:`read_arena_layout` resolves the
    path through the dataset's ``path_map``.

    Attributes:
        group: The recording key's group.
        sequence: The recording key's sequence, ``""`` for a divided recording.
        camera: The camera, ``""`` for every camera.
        stored_path: The layout file's path as the row stores it.
        digest: :meth:`ArenaLayout.digest` of the file's layout.
        n_arenas: The number of arenas in the layout.
    """

    group: str
    sequence: str
    camera: str
    stored_path: str
    digest: str
    n_arenas: int

    @property
    def key(self) -> ArenaKey:
        """The key the row declares its layout under."""
        return ArenaKey(self.group, self.sequence, self.camera)


@dataclass(frozen=True, slots=True)
class DeclaredLayout:
    """A layout as the arenas index declares it.

    Attributes:
        key: The key it is declared under.
        layout: The layout.
        digest: Its digest, which equals ``layout.digest()``.
        path: Its file.
    """

    key: ArenaKey
    layout: ArenaLayout
    digest: str
    path: Path


def arena_index_path(ds: Dataset) -> Path:
    """Return the arenas index, which may not exist yet."""
    return ds.get_root(ARENAS_ROOT) / "index.csv"


def arena_index(path: Path) -> IndexCSV[ArenaLayoutRow]:
    """Open the arenas index at *path*, one row per layout key."""
    return IndexCSV(
        path,
        ArenaLayoutRow,
        dedup_keys=list(_KEY_COLUMNS),
        adopt=partial(project_to_schema, columns=_COLUMNS),
    )


register_reconcilable_index(ARENAS_ROOT, arena_index)


def arena_layout_file_name(key: ArenaKey) -> str:
    """Return the file name of the layout declared under *key*."""
    stem = recording_storage_name(key.recording)
    return f"{stem}.{key.camera}.json" if key.camera else f"{stem}.json"


def _row_key(record: Mapping[str, str]) -> ArenaKey:
    """Return the key that one index row declares."""
    return ArenaKey(
        record.get("group", ""), record.get("sequence", ""), record.get("camera", "")
    )


def write_arena_layout(ds: Dataset, key: ArenaKey, layout: ArenaLayout) -> Path:
    """Write *layout* under *key*, replacing the layout the key had.

    The file and the row are written under the index lock. The caller checks
    the key against the dataset first (``Dataset.set_arena_layout``).

    Returns:
        The layout's file.

    Raises:
        ArenaIndexError: If another key's layout has the file name of *key*,
            as the recording ``a.b`` and the recording ``a`` of camera ``b`` do.
    """
    index = arena_index(arena_index_path(ds))
    target = ds.get_root(ARENAS_ROOT) / arena_layout_file_name(key)
    relative = Path(ds.relative_to_root(target))
    with index_lock(index.path):
        held = index.read_holding_lock()
        for record in index_records(held):
            other = _row_key(record)
            if other != key and Path(record.get("abs_path", "")) == relative:
                message = (
                    f"the layout of {key} would be written to {relative}, which "
                    f"holds the layout of {other}. Name the camera or the "
                    "recording without the dot that joins them."
                )
                raise ArenaIndexError(message)
        text = layout.model_dump_json(indent=2) + "\n"
        atomic_write(target, lambda path: path.write_text(text, encoding="utf-8"))
        index.append_holding_lock(
            [
                ArenaLayoutRow(
                    abs_path=relative,
                    group=key.group,
                    sequence=key.sequence,
                    camera=key.camera,
                    digest=layout.digest(),
                    n_arenas=len(layout.arenas),
                )
            ]
        )
    return target


def read_arena_index_rows(ds: Dataset) -> dict[ArenaKey, IndexedLayout]:
    """Return the index's row of each key, without opening a layout file.

    An absent root or index declares none. Reads without the lock and writes
    nothing.

    Raises:
        ArenaIndexError: If the index cannot be read or two rows declare one
            key.
    """
    if not ds.has_root(ARENAS_ROOT):
        return {}
    path = arena_index_path(ds)
    if not path.is_file():
        return {}
    try:
        records = index_records(arena_index(path).read())
        built = [
            IndexedLayout(
                group=record.get("group", ""),
                sequence=record.get("sequence", ""),
                camera=record.get("camera", ""),
                stored_path=record.get("abs_path", ""),
                digest=record.get("digest", ""),
                n_arenas=int(record.get("n_arenas", "0")),
            )
            for record in records
        ]
    except (OSError, ValueError) as exc:
        message = (
            f"{path} cannot be read: {terse(exc)}. Remove it and set the layouts again."
        )
        raise ArenaIndexError(message) from exc
    rows: dict[ArenaKey, IndexedLayout] = {}
    for row in built:
        if row.key in rows:
            message = f"{path} declares two layouts of {row.key}. Set it again."
            raise ArenaIndexError(message)
        rows[row.key] = row
    return rows


def media_cameras(ds: Dataset) -> dict[Entry, set[str]]:
    """Return the cameras of each entry's media rows, from one read of the index."""
    cameras: dict[Entry, set[str]] = {}
    for row in ds.read_media_index():
        entry = (row.get("group", ""), row.get("sequence", ""))
        cameras.setdefault(entry, set()).add(row.get("camera", ""))
    return cameras


def _row_for_camera(
    ds: Dataset, entry: Entry, camera: str, rows: Mapping[ArenaKey, IndexedLayout]
) -> IndexedLayout | None:
    """Return the row *entry* finds under *camera*, ``None`` without one.

    The keys are tried in the order of
    :func:`~mosaic.core.arenas.layout_lookup_order`. A sequence-less key of a
    group that is not a divided recording names no recording and is skipped.
    """
    for key in layout_lookup_order(
        RecordingKey(*entry), ds.recording_of(*entry), camera
    ):
        if not key.sequence and ds.recording_kind(key.group) != "divided":
            continue
        found = rows.get(key)
        if found is not None:
            return found
    return None


def entry_arena_row(
    ds: Dataset,
    entry: Entry,
    cameras: Collection[str],
    rows: Mapping[ArenaKey, IndexedLayout],
) -> IndexedLayout | None:
    """Return the one row of *entry*'s cameras, ``None`` without one.

    A tracks table names no camera, so an entry of several cameras is assigned
    when every camera that finds a row finds the same digest. A camera that
    finds none is passed over. An entry without media rows looks up camera
    ``""``.

    Args:
        ds: The dataset, which says what recording the entry belongs to.
        entry: The entry's ``(group, sequence)``.
        cameras: The cameras of the entry's media rows.
        rows: The index's rows, :func:`read_arena_index_rows`.

    Raises:
        ValueError: If the entry's cameras find rows of different digests.
    """
    found = [
        row
        for camera in sorted(cameras or {""})
        if (row := _row_for_camera(ds, entry, camera, rows)) is not None
    ]
    digests = {row.digest for row in found}
    if len(digests) > 1:
        message = (
            f"the cameras of {entry!r} find {len(digests)} arena layouts, and its "
            "tracks name no camera. Declare one layout for every camera."
        )
        raise ValueError(message)
    return found[0] if found else None


def read_arena_layout(ds: Dataset, row: IndexedLayout) -> ArenaLayout:
    """Return the layout in *row*'s file, checked against the row's digest.

    Raises:
        ArenaIndexError: If the file is missing, does not parse as a layout, or
            holds content whose digest is not the recorded one.
    """
    file = ds.resolve_path(row.stored_path)
    index = arena_index_path(ds)
    try:
        layout = ArenaLayout.model_validate(json.loads(file.read_text("utf-8")))
    except (OSError, ValueError) as exc:
        message = (
            f"{file}, which {index} records for {row.key}, cannot be read as a "
            f"layout: {terse(exc)}. Set the layout again."
        )
        raise ArenaIndexError(message) from exc
    if layout.digest() != row.digest:
        message = (
            f"{file} holds a layout whose digest is not the {row.digest} that "
            f"{index} records for {row.key}. Set the layout again."
        )
        raise ArenaIndexError(message)
    return layout


def read_declared_layout(ds: Dataset, row: IndexedLayout) -> DeclaredLayout:
    """Return *row*'s layout with its key, digest and file.

    Raises:
        ArenaIndexError: As :func:`read_arena_layout`.
    """
    layout = read_arena_layout(ds, row)
    return DeclaredLayout(row.key, layout, row.digest, ds.resolve_path(row.stored_path))


def read_arena_layouts(ds: Dataset) -> dict[ArenaKey, DeclaredLayout]:
    """Return every declared layout by its key, reading without the lock.

    An absent root or index declares none.

    Raises:
        ArenaIndexError: If the index cannot be read, two rows declare one
            key, or a layout's file is missing, does not parse as a layout, or
            holds content whose digest is not the recorded one.
    """
    return {
        key: read_declared_layout(ds, row)
        for key, row in read_arena_index_rows(ds).items()
    }
