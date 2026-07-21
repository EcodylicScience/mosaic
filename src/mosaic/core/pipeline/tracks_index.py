"""The single writer of the raw-tracks index CSV.

The raw-tracks index (``tracks_raw/index.csv``) is a mosaic-defined artifact
with invariants mosaic owns: the ``TRACKS_RAW_INDEX_COLUMNS`` schema, an
``abs_path`` stored root-relative when the file lives inside the dataset tree
(absolute otherwise), and atomic writes. This module owns the Dataset-agnostic
serialization of that artifact so a single body of code enforces those
invariants for every caller (today: ``Dataset.index_tracks_raw``).

It is the tracks sibling of :mod:`mosaic.core.pipeline.media_index`, but
simpler: tracks have no within-sequence ordering (no ``video_order``), no
ffprobe metadata, and no camera axis. Only a raw file's *format* and *grouping*
vary, and that variation is decided by the scanning caller, not here.

Like the media module, it stays Dataset-agnostic: the one place that needs
dataset context -- turning an absolute path into its stored form -- is supplied
by the caller as a ``to_store_path`` callable (``Dataset.relative_to_root``),
mirroring the resolver callable :meth:`IndexCSV.prune_missing` takes.

Reserved extension (not built yet): an assignment-driven projection
``Dataset.write_tracks_index(scopes)`` + ``TracksRawIndexScope``, the tracks
mirror of ``Dataset.write_media_index`` for a future API track-import flow that
uploads raw tracker files, assigns each an explicit ``(group, sequence,
src_format)``, and imports them. It would (re)stat each file under a scope
directory, stamp the caller's identity, preserve rows outside the scopes, and
atomic-write -- but without the media densifier, since tracks are unordered.
Its first caller is that API flow, which does not exist yet.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pandas as pd

from mosaic.core.pipeline._utils import atomic_write

# Column schema and order for ``tracks_raw/index.csv``. Owned here (not in the
# ``core/media`` leaf like ``MEDIA_INDEX_COLUMNS``) because raw-tracks columns
# are pipeline-owned and carry no media-I/O dependency.
TRACKS_RAW_INDEX_COLUMNS = [
    "group",
    "sequence",
    "abs_path",
    "src_format",
    "size_bytes",
    "mtime_iso",
    "md5",
]

# The only numeric raw-tracks column; every other column is a text cell that
# must round trip through CSV as an empty string rather than a float NaN.
TRACKS_RAW_NUMERIC_COLUMNS = frozenset({"size_bytes"})


class TracksRawIndexRow(TypedDict):
    """One ``tracks_raw/index.csv`` row -- the typed surface a schema change
    cannot silently break.

    ``group``/``sequence`` are the raw file's assigned identity, ``src_format``
    names its converter, ``abs_path`` is the stored (root-relative or absolute)
    path, and ``md5`` is empty unless the caller opted into hashing.
    """

    group: str
    sequence: str
    abs_path: str
    src_format: str
    size_bytes: int
    mtime_iso: str
    md5: str


def _mtime_iso(timestamp: float) -> str:
    """UTC ISO-8601 string for a filesystem mtime."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def read_tracks_index(index_path: Path) -> list[dict[str, str]]:
    """Read a raw-tracks index CSV as string-cell records (empty list if absent).

    The record form (string cells, not pandas) is what identity-reading callers
    want; it is the drop-in a future API reader uses instead of its own
    ``csv.DictReader``.
    """
    if not index_path.exists():
        return []
    with index_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_tracks_index_frame(index_path: Path) -> pd.DataFrame:
    """Read a raw-tracks index CSV into the full schema, text cells as object ``""``.

    Missing columns are added, and text columns (everything non-numeric) are
    coerced to an object dtype with NaN replaced by ``""`` so later cell writes
    do not trip pandas' incompatible-dtype warning on all-empty float columns.
    """
    if index_path.exists():
        df = pd.read_csv(index_path)
    else:
        df = pd.DataFrame(columns=TRACKS_RAW_INDEX_COLUMNS)
    for column in TRACKS_RAW_INDEX_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        if column not in TRACKS_RAW_NUMERIC_COLUMNS:
            df[column] = df[column].astype("object").where(df[column].notna(), "")
    return df


def frame_from_rows(rows: list[TracksRawIndexRow]) -> pd.DataFrame:
    """Build a raw-tracks-index DataFrame with exactly ``TRACKS_RAW_INDEX_COLUMNS``."""
    return pd.DataFrame(rows, columns=TRACKS_RAW_INDEX_COLUMNS)


def write_tracks_index_rows(index_path: Path, df: pd.DataFrame) -> None:
    """Atomically write *df* projected onto ``TRACKS_RAW_INDEX_COLUMNS``.

    The projection fixes column order and drops any transient column; the atomic
    write guarantees a concurrent reader never sees a partial file.
    """
    atomic_write(
        index_path, lambda p: df[TRACKS_RAW_INDEX_COLUMNS].to_csv(p, index=False)
    )


def build_tracks_raw_row(
    *,
    path: Path,
    stat: os.stat_result,
    to_store_path: Callable[[Path], str],
    group: str,
    sequence: str,
    src_format: str,
    md5: str = "",
) -> TracksRawIndexRow:
    """Assemble one raw-tracks-index row.

    *stat* is the file's ``os.stat_result`` (size and mtime come from it).
    ``abs_path`` is produced by *to_store_path*, so the in-tree-relative /
    out-of-tree-absolute rule is enforced in one place. *md5* is passed by the
    caller (empty unless hashing was requested) rather than computed here, so
    this stays a stat-only, read-light assembler.
    """
    return {
        "group": group,
        "sequence": sequence,
        "abs_path": to_store_path(path),
        "src_format": src_format,
        "size_bytes": stat.st_size,
        "mtime_iso": _mtime_iso(stat.st_mtime),
        "md5": md5,
    }
