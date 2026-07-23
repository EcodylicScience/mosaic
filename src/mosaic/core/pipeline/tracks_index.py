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

The assignment-driven projection ``Dataset.write_tracks_index(scopes)`` (with
``TracksRawIndexScope``) is the tracks mirror of ``Dataset.write_media_index``
for a future API track-import flow that uploads raw tracker files, assigns each
an explicit ``(group, sequence, src_format)``, and imports them: it (re)stats
each file under a scope directory, stamps the caller's identity, preserves rows
outside the scopes, and atomic-writes -- without the media densifier, since
today's tracks are unordered. No API caller exists yet.

Reserved axes (additive, not built): multi-camera / time-chunked raw tracks add
a scope-level ``camera`` and a per-file time-order (concatenation) axis -- the
tracks mirror of ``MediaIndexScope``'s ``camera`` + ``order_by_name`` ->
``video_order``. Both would reuse the shared ``assign_video_order`` ranker and
land as new keyword scope fields plus reserved columns
(``write_tracks_index_rows`` already drops transient columns), so nothing here
needs a redesign when they arrive.
"""

from __future__ import annotations

import csv
import fnmatch
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TracksRawIndexScope:
    """One affected ``(group, sequence)`` directory to (re)scan and assign.

    ``directory`` is the ``tracks_raw`` subdir scanned for this sequence's raw
    files; every matching file found is assigned ``group``/``sequence`` and
    stamped with ``src_format`` (the converter name) -- the assignment-driven
    counterpart to ``Dataset.index_tracks_raw``'s scan-and-derive. All files in
    one scope dir take that single identity, so multiple per-id raw files (e.g.
    TREx ``seq_fish0``/``seq_fish1``) become one multi-file sequence *without*
    the ``_fishN`` strip ``index_tracks_raw`` applies -- here the caller owns the
    grouping. This is the tracks sibling of :class:`MediaIndexScope`; it has no
    ``camera`` or ordering axis today (tracks are unordered). Multi-camera /
    time-chunked raw tracks would add those as new keyword fields, mirroring
    ``MediaIndexScope``'s ``camera`` + ``order_by_name`` (see the module
    docstring).
    """

    directory: Path
    group: str
    sequence: str
    src_format: str


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


def frame_from_rows(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Build a raw-tracks-index DataFrame with exactly ``TRACKS_RAW_INDEX_COLUMNS``.

    Accepts both freshly built :class:`TracksRawIndexRow` rows (a ``TypedDict``,
    hence a ``Mapping[str, object]``) and preserved string-cell records from
    :func:`read_tracks_index`, so the projection can merge the two without
    per-row normalization.
    """
    return pd.DataFrame(rows, columns=TRACKS_RAW_INDEX_COLUMNS)


def iter_track_files(
    search_dirs: Iterable[Path],
    patterns: Sequence[str],
    *,
    recursive: bool = True,
    exclude_patterns: Sequence[str] = (),
) -> list[tuple[Path, os.stat_result]]:
    """Scan *search_dirs* for raw-track files, deterministically.

    Yields ``(path, stat)`` for every file under *search_dirs* matching any glob
    in *patterns*, skipping macOS resource forks (``._*``) and any basename
    matching *exclude_patterns*. A file matched by several patterns appears once
    (deduped by resolved path), and the result is sorted by resolved path so the
    written index is order-stable regardless of filesystem iteration order. The
    identity-free scan shared by :meth:`Dataset.index_tracks_raw` (scan-and-
    derive) and :meth:`Dataset.write_tracks_index` (assignment-driven).
    """
    by_resolved: dict[Path, tuple[Path, os.stat_result]] = {}
    for directory in search_dirs:
        for pattern in patterns:
            matches = directory.rglob(pattern) if recursive else directory.glob(pattern)
            for path in matches:
                if not path.is_file():
                    continue
                if path.name.startswith("._"):
                    continue
                if any(fnmatch.fnmatch(path.name, ex) for ex in exclude_patterns):
                    continue
                resolved = path.resolve()
                if resolved in by_resolved:
                    continue
                by_resolved[resolved] = (path, path.stat())
    return [by_resolved[key] for key in sorted(by_resolved)]


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
