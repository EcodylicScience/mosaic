"""The appearance table: one embedding vector per tracked detection.

A tracks variant stores the table beside each entry's tracks parquet, at
``<variant>/appearance/<entry_key>.parquet``, and the entry's tracks-index row
records its path in ``appearance_abs_path``. The table is keyed by the tracks
table's ``(frame, id)`` and has one float32 column ``emb_<k>`` per vector
component. This module defines the contract and imports no tracker code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.entry import Entry
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.parquet_reader import read_parquet_column_names
from mosaic.core.pipeline.tracks_index import CellRow, variant_entry_rows
from mosaic.core.pipeline.writers import write_parquet_atomic

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

APPEARANCE_PREFIX: Final = "emb_"
"""The prefix of the vector columns."""

NORM_TOLERANCE: Final = 1e-3
"""How far an L2 norm may differ from 1."""

_KEY_COLUMNS: Final = ("frame", "id")
_LABEL_COLUMNS: Final = ("group", "sequence")


class AppearanceTableError(ValueError):
    """An appearance table breaks the contract. The message names the source."""


def appearance_columns(dim: int) -> tuple[str, ...]:
    """Returns the vector column names ``emb_0`` to ``emb_{dim-1}``."""
    return tuple(f"{APPEARANCE_PREFIX}{position}" for position in range(dim))


def appearance_path(variant_root: Path, entry_key: str) -> Path:
    """Returns the appearance table of *entry_key* under *variant_root*."""
    return variant_root / "appearance" / f"{entry_key}.parquet"


@dataclass(frozen=True, slots=True)
class AppearanceTable:
    """The vectors of one entry, row for row.

    Attributes:
        frame: The source-space frame of each row.
        id: The track identity of each row.
        vectors: A C-contiguous float32 array of shape ``(n, D)``.
    """

    frame: npt.NDArray[np.int64]
    id: npt.NDArray[np.int64]
    vectors: npt.NDArray[np.float32]


def _vector_columns(columns: Iterable[object], source: str) -> tuple[str, ...]:
    """Returns the ``emb_`` names among *columns*, which must run from ``emb_0``."""
    names = {str(name) for name in columns if str(name).startswith(APPEARANCE_PREFIX)}
    if not names:
        msg = f"{source}: the table has no {APPEARANCE_PREFIX} columns"
        raise AppearanceTableError(msg)
    columns = appearance_columns(len(names))
    if names != set(columns):
        msg = (
            f"{source}: the {APPEARANCE_PREFIX} columns are not contiguous from "
            f"{columns[0]}: found {sorted(names)}"
        )
        raise AppearanceTableError(msg)
    return columns


def _require_columns(
    df: pd.DataFrame, names: tuple[str, ...], source: str, *, table: str = "the table"
) -> None:
    """Raises when *df*, described as *table*, lacks one of *names*."""
    for column in names:
        if column not in df.columns:
            msg = f"{source}: {table} lacks the {column} column"
            raise AppearanceTableError(msg)


def _require_dtypes(df: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    """Raises when a column of *df* does not have the dtype the contract fixes."""
    for column in _KEY_COLUMNS:
        if df[column].dtype != np.int64:
            msg = f"{source}: the {column} column is {df[column].dtype}, not int64"
            raise AppearanceTableError(msg)
    for column in _LABEL_COLUMNS:
        if not pd.api.types.is_string_dtype(df[column]):
            msg = f"{source}: the {column} column is {df[column].dtype}, not a string"
            raise AppearanceTableError(msg)
    for column in columns:
        if df[column].dtype != np.float32:
            msg = f"{source}: the {column} column is {df[column].dtype}, not float32"
            raise AppearanceTableError(msg)


def validate_appearance(df: pd.DataFrame, tracks: pd.DataFrame, *, source: str) -> int:
    """Checks *df* against the appearance contract and returns its dimension ``D``.

    Args:
        df: The appearance table of one entry.
        tracks: The tracks table of the same entry and variant.
        source: What the table was read from or is destined for, named in the
            error.

    Returns:
        The number of vector columns.

    Raises:
        AppearanceTableError: A column of ``group``, ``sequence``, ``frame`` or
            ``id`` is missing from *df*, ``frame`` or ``id`` is missing from
            *tracks*, the vector columns are missing or not contiguous from
            ``emb_0``, a column does not have its dtype (int64 ``frame`` and
            ``id``, string ``group`` and ``sequence``, float32 ``emb_``
            columns), a vector contains NaN, a ``(frame, id)`` repeats or is
            absent from *tracks*, or a vector's L2 norm differs from 1 by more
            than 1e-3. The message names the first violation.
    """
    _require_columns(tracks, _KEY_COLUMNS, source, table="the tracks table")
    _require_columns(df, (*_LABEL_COLUMNS, *_KEY_COLUMNS), source)
    columns = _vector_columns(df.columns, source)
    _require_dtypes(df, columns, source)
    vectors = df[list(columns)].to_numpy(dtype=np.float64)
    bad_nan = np.isnan(vectors)
    if bad_nan.any():
        row, component = np.argwhere(bad_nan)[0]
        msg = f"{source}: NaN in {columns[component]} at row {row}"
        raise AppearanceTableError(msg)
    repeated = df.duplicated(list(_KEY_COLUMNS), keep="first")
    if repeated.any():
        first = df.loc[repeated, list(_KEY_COLUMNS)].iloc[0]
        msg = (
            f"{source}: (frame, id) is not unique: "
            f"({first['frame']}, {first['id']}) repeats"
        )
        raise AppearanceTableError(msg)
    known = pd.MultiIndex.from_frame(tracks[list(_KEY_COLUMNS)])
    keyed = pd.MultiIndex.from_frame(df[list(_KEY_COLUMNS)])
    absent = ~keyed.isin(known)
    if absent.any():
        frame, identity = keyed[absent][0]
        msg = f"{source}: (frame, id) ({frame}, {identity}) is not in the tracks table"
        raise AppearanceTableError(msg)
    norms = np.linalg.norm(vectors, axis=1)
    off = np.abs(norms - 1.0) > NORM_TOLERANCE
    if off.any():
        row = int(np.flatnonzero(off)[0])
        msg = (
            f"{source}: the L2 norm of row {row} is {norms[row]:.6f}, "
            f"more than {NORM_TOLERANCE} from 1"
        )
        raise AppearanceTableError(msg)
    return len(columns)


def write_appearance(path: Path, df: pd.DataFrame) -> None:
    """Writes *df* to *path* atomically, creating the parent directory."""
    _ = write_parquet_atomic(df, path)


def recorded_appearance_path(ds: Dataset, row: CellRow) -> Path | None:
    """Returns the appearance table that a tracks-index *row* records.

    Returns ``None`` for a blank ``appearance_abs_path`` cell.
    """
    cell = text_cell(row.get("appearance_abs_path"))
    return ds.resolve_path(cell) if cell else None


def read_appearance_dim(path: Path) -> int:
    """Reads the dimension ``D`` of the appearance table at *path* from its footer.

    Raises:
        FileNotFoundError: The file is missing.
        AppearanceTableError: The footer does not parse, or the file has no
            contiguous ``emb_`` columns.
    """
    names = read_parquet_column_names(path)
    if names is None:
        if not path.is_file():
            msg = f"the appearance table is missing: {path}"
            raise FileNotFoundError(msg)
        msg = f"{path}: the parquet footer does not parse"
        raise AppearanceTableError(msg)
    return len(_vector_columns(names, str(path)))


def read_appearance_file(path: Path, *, what: str) -> AppearanceTable:
    """Reads the appearance table at *path*, which belongs to *what*.

    Raises:
        FileNotFoundError: The file is missing. The message names *what*.
        AppearanceTableError: The file lacks ``frame`` or ``id`` or has no
            contiguous ``emb_`` columns.
    """
    if not path.is_file():
        msg = f"the appearance table of {what} is missing: {path}"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(path)
    _require_columns(df, _KEY_COLUMNS, str(path))
    columns = _vector_columns(df.columns, str(path))
    return AppearanceTable(
        frame=df["frame"].to_numpy(dtype=np.int64),
        id=df["id"].to_numpy(dtype=np.int64),
        vectors=np.ascontiguousarray(df[list(columns)].to_numpy(dtype=np.float32)),
    )


def read_appearance(ds: Dataset, variant: str, entry: Entry) -> AppearanceTable | None:
    """Reads the appearance table that *variant* records for *entry*.

    Writes nothing and creates no directory.

    Args:
        ds: The dataset.
        variant: The tracks variant.
        entry: The ``(group, sequence)`` entry.

    Returns:
        The table, or ``None`` when the variant has no row for *entry* or the
        row's ``appearance_abs_path`` cell is blank.

    Raises:
        FileNotFoundError: The recorded file is missing.
        AppearanceTableError: The file lacks ``frame`` or ``id`` or has no
            contiguous ``emb_`` columns.
    """
    hit = variant_entry_rows(ds, variant, entry)
    if hit.empty:
        return None
    path = recorded_appearance_path(ds, hit.iloc[-1])
    if path is None:
        return None
    return read_appearance_file(path, what=f"{entry} in {variant}")
