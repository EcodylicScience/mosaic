"""The arenas root: one layout file per ``(recording, camera)`` and their index.

``arenas/index.csv`` has one row per layout key, naming the layout's file and
the digest of its content. A file is named
``<recording_storage_name>[.<camera>].json``. :func:`write_arena_layout` writes
the file and the row under the index lock. :func:`read_arena_index_rows` reads
the rows, :func:`entry_arena_row` finds the row that one entry and its cameras
look up, and :func:`read_arena_layout` reads that row's file. None of them takes
the lock or writes.

:func:`arenas_compositions` gives each entry's layout digest, the composition of
the root ``arenas``. It is read only for a run whose feature declares the root in
``consumed_roots``, through :func:`cached_arena_lookup`, which reads the index
and the media cameras once per process for each state of the two files.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Collection, Iterable, Mapping
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
from mosaic.core.pipeline.sequence_index import read_entry_compositions
from mosaic.core.recording import RecordingKey, recording_storage_name
from mosaic.core.strict_model import terse

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "ARENAS_ROOT",
    "ARENA_LOOKUP_CACHE",
    "ArenaIndexError",
    "ArenaLayoutRow",
    "ArenaLookup",
    "DeclaredLayout",
    "IndexedLayout",
    "arena_index",
    "arena_index_path",
    "arena_layout_file_name",
    "arenas_compositions",
    "cached_arena_lookup",
    "entry_arena_row",
    "entry_compositions",
    "media_cameras",
    "read_arena_index_rows",
    "read_arena_lookup",
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


@dataclass(frozen=True, slots=True)
class ArenaLookup:
    """What an entry's layout lookup reads: the index rows and the media cameras.

    Attributes:
        rows: The index's row of each key, :func:`read_arena_index_rows`.
        cameras: The cameras of each entry's media rows, :func:`media_cameras`.
        refusal: Why the whole index is refused, ``None`` when it reads.
    """

    rows: Mapping[ArenaKey, IndexedLayout]
    cameras: Mapping[Entry, Collection[str]]
    refusal: str | None = None

    def row(self, ds: Dataset, entry: Entry) -> IndexedLayout | None:
        """Return *entry*'s row, ``None`` without one (:func:`entry_arena_row`).

        Raises:
            ArenaIndexError: If the whole index is refused.
            ValueError: If the entry's cameras find rows of different digests.
        """
        if self.refusal is not None:
            raise ArenaIndexError(self.refusal)
        return entry_arena_row(ds, entry, self.cameras.get(entry, ()), self.rows)


def read_arena_lookup(ds: Dataset) -> ArenaLookup:
    """Read the arenas index and, when it declares a layout, the media cameras.

    A refused index is printed once on stderr and held as the lookup's refusal.
    Reads without the lock and writes nothing.
    """
    try:
        rows = read_arena_index_rows(ds)
    except ArenaIndexError as exc:
        refusal = str(exc)
        print(
            f"[arenas] {refusal} Until then no entry reads an arena layout.",
            file=sys.stderr,
        )
        return ArenaLookup(rows={}, cameras={}, refusal=refusal)
    return ArenaLookup(rows=rows, cameras=media_cameras(ds) if rows else {})


type _FileStamp = tuple[int, int, int] | None
"""A file's ``st_mtime_ns``, ``st_size`` and ``st_ino``, ``None`` when absent."""


@dataclass(frozen=True, slots=True)
class _CachedLookup:
    stamps: tuple[_FileStamp, _FileStamp]
    lookup: ArenaLookup


ARENA_LOOKUP_CACHE: Final[dict[tuple[Path, Path | None], _CachedLookup]] = {}
"""This process's lookup of each dataset, keyed by its arenas and media indexes."""

_CACHE_LOCK: Final = threading.Lock()


def _stamp(path: Path | None) -> _FileStamp:
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def cached_arena_lookup(ds: Dataset) -> ArenaLookup:
    """Return *ds*'s :class:`ArenaLookup`, read again only when an index changed.

    The lookup is held per process, keyed by the arenas index and the media
    index (:meth:`Dataset.read_media_index`), beside each file's modification
    time, size and inode. A dataset without an arenas root reads nothing.
    """
    if not ds.has_root(ARENAS_ROOT):
        return ArenaLookup(rows={}, cameras={})
    media_root = ds.resolve_media_root()
    media = ds.get_root(media_root) / "index.csv" if ds.has_root(media_root) else None
    key = (arena_index_path(ds), media)
    with _CACHE_LOCK:
        stamps = (_stamp(key[0]), _stamp(media))
        cached = ARENA_LOOKUP_CACHE.get(key)
        if cached is not None and cached.stamps == stamps:
            return cached.lookup
        lookup = read_arena_lookup(ds)
        ARENA_LOOKUP_CACHE[key] = _CachedLookup(stamps, lookup)
        return lookup


def arenas_compositions(
    ds: Dataset, entries: Iterable[Entry], *, lookup: ArenaLookup | None = None
) -> dict[Entry, str]:
    """Return the layout digest of each of *entries* that has one.

    An entry without a layout, or whose cameras find different layouts, is
    absent, as is every entry while the index is refused.

    Args:
        ds: The dataset.
        entries: The entries to look up.
        lookup: The lookup to read, :func:`cached_arena_lookup` when ``None``.
    """
    found = cached_arena_lookup(ds) if lookup is None else lookup
    if found.refusal is not None or not found.rows:
        return {}
    digests: dict[Entry, str] = {}
    for entry in entries:
        try:
            row = found.row(ds, entry)
        except ValueError:
            continue
        if row is not None:
            digests[entry] = row.digest
    return digests


def entry_compositions(
    ds: Dataset,
    entries: Iterable[Entry],
    roots: Collection[str],
    *,
    lookup: ArenaLookup | None = None,
) -> dict[Entry, dict[str, str]]:
    """Return each entry's compositions, with its layout digest when *roots* asks.

    Every source root's composition is :func:`read_entry_compositions`. When
    *roots* holds ``arenas``, the digest of :func:`arenas_compositions` joins
    each entry that has a layout, under ``arenas``.

    Args:
        ds: The dataset.
        entries: The entries to read.
        roots: The roots in play, such as a feature's ``consumed_roots``.
        lookup: The lookup to read, :func:`cached_arena_lookup` when ``None``.
    """
    wanted = list(entries)
    compositions = read_entry_compositions(ds, wanted)
    if ARENAS_ROOT not in roots:
        return compositions
    for entry, digest in arenas_compositions(ds, wanted, lookup=lookup).items():
        compositions.setdefault(entry, {})[ARENAS_ROOT] = digest
    return compositions
