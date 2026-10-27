"""The arenas root: one layout file per ``(recording, camera)`` and their index.

``arenas/index.csv`` has one row per layout key, naming the layout's file and
the digest of its content. A file is named
``<recording_storage_name>[.<camera>].json``. :func:`write_arena_layout` writes
the file and the row under the index lock, and :func:`read_arena_layouts` reads
them without the lock and writes nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.arenas import ArenaKey, ArenaLayout
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    IndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.recording import recording_storage_name
from mosaic.core.strict_model import terse

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "ARENAS_ROOT",
    "ArenaIndexError",
    "ArenaLayoutRow",
    "DeclaredLayout",
    "arena_index",
    "arena_index_path",
    "arena_layout_file_name",
    "read_arena_layouts",
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


def read_arena_layouts(ds: Dataset) -> dict[ArenaKey, DeclaredLayout]:
    """Return every declared layout by its key, reading without the lock.

    An absent root or index declares none.

    Raises:
        ArenaIndexError: If the index cannot be read, two rows declare one
            key, or a layout's file is missing, does not parse as a layout, or
            holds content whose digest is not the recorded one.
    """
    if not ds.has_root(ARENAS_ROOT):
        return {}
    path = arena_index_path(ds)
    if not path.is_file():
        return {}
    try:
        records = index_records(arena_index(path).read())
    except (OSError, ValueError) as exc:
        message = (
            f"{path} cannot be read: {terse(exc)}. Remove it and set the layouts again."
        )
        raise ArenaIndexError(message) from exc
    declared: dict[ArenaKey, DeclaredLayout] = {}
    for record in records:
        key = _row_key(record)
        if key in declared:
            message = f"{path} declares two layouts of {key}. Set it again."
            raise ArenaIndexError(message)
        file = ds.resolve_path(record.get("abs_path", ""))
        try:
            layout = ArenaLayout.model_validate(json.loads(file.read_text("utf-8")))
        except (OSError, ValueError) as exc:
            message = (
                f"{file}, which {path} records for {key}, cannot be read as a "
                f"layout: {terse(exc)}. Set the layout again."
            )
            raise ArenaIndexError(message) from exc
        digest = record.get("digest", "")
        if layout.digest() != digest:
            message = (
                f"{file} holds a layout whose digest is not the {digest} that "
                f"{path} records for {key}. Set the layout again."
            )
            raise ArenaIndexError(message)
        declared[key] = DeclaredLayout(key, layout, digest, file)
    return declared
