"""The single writer of the media index CSV.

The media index (``media_raw/index.csv`` for raw originals, ``media/index.csv``
for derivatives) is a mosaic-defined artifact with invariants mosaic owns:
the ``MEDIA_INDEX_COLUMNS`` schema, safe-name encoding, ``video_order``
semantics, an ``abs_path`` stored root-relative when the file lives inside the
dataset tree (absolute otherwise), and atomic writes. This module owns the
Dataset-agnostic serialization of that artifact so a single body of code
enforces those invariants for every caller (``Dataset.index_media``,
``Dataset.write_media_index``, and the transcode derivative-link writers).

It stays Dataset-agnostic: the one place that needs dataset context -- turning
an absolute path into its stored form -- is supplied by the caller as a
``to_store_path`` callable (``Dataset.relative_to_root``), mirroring the
resolver callable :meth:`IndexCSV.prune_missing` takes.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import pandas as pd

from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.pipeline._utils import atomic_write

# Numeric media-index columns; every other column is a text cell that must round
# trip through CSV as an empty string rather than a float NaN.
MEDIA_NUMERIC_COLUMNS = frozenset(
    {"size_bytes", "width", "height", "fps", "frame_count", "video_order"}
)

# Sort rank for a pre-existing row whose prior ``video_order`` is unknown: it
# sorts after every recorded prior order, then falls back to name.
_UNORDERED_PRIOR_RANK = 2**31


def _empty_order() -> dict[str, int]:
    """Typed default for :attr:`MediaIndexScope.order_by_name` (bare ``dict`` as a
    default_factory infers ``dict[Unknown, Unknown]`` under strict typing)."""
    return {}


@dataclass(frozen=True)
class MediaIndexScope:
    """One affected ``(group, sequence)`` to re-probe and reorder.

    ``directory`` is the media_raw subdir scanned for this sequence's files;
    every file found is assigned ``group``/``sequence`` (not derived from a
    track keymap). ``order_by_name`` maps a file's basename to its arranged
    linear position within the sequence; a file absent from the map keeps its
    prior ``video_order`` and sorts before the arranged ones. ``camera`` is
    reserved for multi-camera recordings; it is ``""`` today and only takes
    effect once the media index gains a ``camera`` column.
    """

    directory: Path
    group: str
    sequence: str
    order_by_name: Mapping[str, int] = field(default_factory=_empty_order)
    camera: str = ""


def _mtime_iso(timestamp: float) -> str:
    """UTC ISO-8601 string for a filesystem mtime."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def read_media_index(index_path: Path) -> list[dict[str, str]]:
    """Read a media index CSV as string-cell records (empty list if absent).

    The record form (string cells, not pandas) is what order-reading callers
    want: they read ``video_order`` as ``str`` and strip/int it themselves.
    """
    if not index_path.exists():
        return []
    with index_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_media_index_frame(index_path: Path) -> pd.DataFrame:
    """Read a media index CSV into the full schema, text cells as object ``""``.

    Missing columns are added, and text columns (everything non-numeric) are
    coerced to an object dtype with NaN replaced by ``""`` so later cell writes
    do not trip pandas' incompatible-dtype warning on all-empty float columns.
    """
    if index_path.exists():
        df = pd.read_csv(index_path)
    else:
        df = pd.DataFrame(columns=MEDIA_INDEX_COLUMNS)
    for column in MEDIA_INDEX_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        if column not in MEDIA_NUMERIC_COLUMNS:
            df[column] = df[column].astype("object").where(df[column].notna(), "")
    return df


def frame_from_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a media-index DataFrame with exactly ``MEDIA_INDEX_COLUMNS``."""
    return pd.DataFrame(rows, columns=MEDIA_INDEX_COLUMNS)


def write_media_index_rows(index_path: Path, df: pd.DataFrame) -> None:
    """Atomically write *df* projected onto ``MEDIA_INDEX_COLUMNS``.

    The projection drops any transient column (e.g. a future ``camera`` before
    it is a schema column) and fixes column order; the atomic write guarantees
    a concurrent reader never sees a partial file.
    """
    atomic_write(index_path, lambda p: df[MEDIA_INDEX_COLUMNS].to_csv(p, index=False))


def build_media_index_row(
    *,
    path: Path,
    stat: object,
    to_store_path: Callable[[Path], str],
    group: str,
    sequence: str,
    group_safe: str,
    sequence_safe: str,
    probe: Mapping[str, object],
    name: str | None = None,
    camera: str = "",
    media_type: str = "video",
    source_path: str | None = None,
    video_order: int = 0,
) -> dict[str, object]:
    """Assemble one media-index row.

    *stat* is the file's ``os.stat_result`` (size and mtime come from it).
    *probe* is the width/height/fps/codec plus the injectable MediaFacts cells
    (a :class:`ProbeMetadata`). ``abs_path`` is produced by *to_store_path*, so
    the in-tree-relative / out-of-tree-absolute rule is enforced in one place.
    *source_path* overrides the empty ``source_path`` the probe carries (used by
    a derivative's back-link). *camera* is not persisted until the media index
    gains a ``camera`` column, but is carried for :func:`densify_video_order`.
    """
    size_bytes = getattr(stat, "st_size")
    mtime = getattr(stat, "st_mtime")
    row: dict[str, object] = {
        "name": name if name is not None else path.name,
        "group": group,
        "sequence": sequence,
        "group_safe": group_safe,
        "sequence_safe": sequence_safe,
        "abs_path": to_store_path(path),
        "size_bytes": size_bytes,
        "mtime_iso": _mtime_iso(mtime),
        "media_type": media_type,
        "video_order": video_order,
        **probe,
    }
    if camera:
        row["camera"] = camera
    if source_path is not None:
        row["source_path"] = source_path
    return row


def build_prior_order(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str], int]:
    """Map ``(sequence, basename)`` to prior ``video_order`` for existing rows.

    Rows with a missing or blank ``video_order`` are skipped so they fall back
    to name ordering in :func:`densify_video_order`.
    """
    prior: dict[tuple[str, str], int] = {}
    for row in rows:
        raw = str(row.get("video_order", "")).strip()
        if not raw or raw.lower() == "nan":
            continue
        try:
            order = int(float(raw))
        except ValueError:
            continue
        prior[(str(row["sequence"]), Path(str(row["abs_path"])).name)] = order
    return prior


T = TypeVar("T")


@dataclass(frozen=True)
class VideoOrderKey:
    """The ordering inputs for one item in :func:`assign_video_order`.

    ``group``/``sequence``/``camera`` place the item in its dense-counter group.
    ``name`` is the final sort tie-break only (a display basename), never a lookup
    key. ``prior_order`` is the item's existing ``video_order`` (``None`` when it
    has none, so it sorts after every recorded prior order). ``session_position``
    is the item's arranged position among this session's additions (``None`` when
    the item is not a session addition).
    """

    group: str
    sequence: str
    camera: str
    name: str
    prior_order: int | None
    session_position: int | None


def assign_video_order(
    items: Sequence[T], key_of: Callable[[T], VideoOrderKey]
) -> list[tuple[T, int]]:
    """Arrange *items* and assign a dense ``video_order`` per (group, sequence, camera).

    Within each group the order is: items with no ``session_position`` (already
    present) first, by ``prior_order`` (``None`` -> a sentinel past every real
    order), then this session's additions by ``session_position``; ``name`` breaks
    ties. A single stable sort over the total key makes this deterministic, and
    the dense counter restarts at 0 for each ``(group, sequence, camera)`` so
    parallel cameras are numbered independently rather than as temporal chunks of
    one camera. Returns ``(item, video_order)`` pairs in the arranged order.

    This is the one place the ``video_order`` ranking lives:
    :func:`densify_video_order` (media-index rows) and the mosaic-api sequence
    importer (``Video.idx``) both project their inputs into a
    :class:`VideoOrderKey` and call it, so the numbering cannot diverge across the
    two call sites.
    """

    def sort_key(
        pair: tuple[T, VideoOrderKey],
    ) -> tuple[str, str, str, int, int, str]:
        key = pair[1]
        if key.session_position is not None:
            rank_class, rank_value = 1, key.session_position
        else:
            rank_class = 0
            rank_value = (
                key.prior_order
                if key.prior_order is not None
                else _UNORDERED_PRIOR_RANK
            )
        return (key.group, key.sequence, key.camera, rank_class, rank_value, key.name)

    keyed = sorted(((item, key_of(item)) for item in items), key=sort_key)
    counters: dict[tuple[str, str, str], int] = {}
    result: list[tuple[T, int]] = []
    for item, key in keyed:
        group_key = (key.group, key.sequence, key.camera)
        position = counters.get(group_key, 0)
        counters[group_key] = position + 1
        result.append((item, position))
    return result


def densify_video_order(
    rows: list[dict[str, object]],
    *,
    session_positions: Mapping[tuple[str, str], int],
    prior_order: Mapping[tuple[str, str], int],
) -> list[dict[str, object]]:
    """Re-number ``video_order`` as a dense counter per ``(group, sequence, camera)``.

    A thin adapter over :func:`assign_video_order`: it maps each media-index row
    into a :class:`VideoOrderKey`. Within each group the order is pre-existing
    videos first, keeping their prior ``video_order`` (*prior_order*), then this
    session's videos by arranged position (*session_positions*); the ``name``
    column breaks ties. Both maps are keyed ``(sequence, basename)`` where the
    basename comes from ``abs_path``; a session video is one present in
    *session_positions*. Keying the dense counter on ``camera`` (``""`` for every
    row today) makes this per-``(group, sequence)`` now and per-camera once a
    ``camera`` column exists. Mutates each row's ``video_order`` in place and
    returns the rows in the assigned order.
    """

    def key_of(row: dict[str, object]) -> VideoOrderKey:
        lookup = (str(row["sequence"]), Path(str(row["abs_path"])).name)
        return VideoOrderKey(
            group=str(row["group"]),
            sequence=str(row["sequence"]),
            camera=str(row.get("camera", "") or ""),
            name=str(row["name"]),
            prior_order=prior_order.get(lookup),
            session_position=session_positions.get(lookup),
        )

    ordered = assign_video_order(rows, key_of)
    result: list[dict[str, object]] = []
    for row, order in ordered:
        row["video_order"] = order
        result.append(row)
    return result
