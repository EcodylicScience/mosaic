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
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import pandas as pd
from mosaic_media import MediaFacts

from mosaic.core.manifest import MediaDivision
from mosaic.core.media.facts_columns import (
    MEDIA_INDEX_COLUMNS,
    AssignmentSource,
    media_row_uuid,
)
from mosaic.core.pipeline.composition import MediaMember
from mosaic.core.pipeline._utils import atomic_write

# Numeric media-index columns; every other column is a text cell that must round
# trip through CSV as an empty string rather than a float NaN.
MEDIA_NUMERIC_COLUMNS = frozenset(
    {"size_bytes", "width", "height", "fps", "frame_count", "video_order"}
)

# Sort rank for a pre-existing row whose prior ``video_order`` is unknown: it
# sorts after every recorded prior order, then falls back to name.
_UNORDERED_PRIOR_RANK = 2**31


class StoreGroupingError(ValueError):
    """Imgstores whose metadata leaves their recording undecided.

    A media index write raises it before it writes anything, when stores of
    different recordings would join one entry or a synchronized set cannot tell
    its cameras apart.
    """


def unnamed_cameras_error(sync_uuid: str, paths: Sequence[str]) -> StoreGroupingError:
    """The refusal of a synchronized recording with stores that record no serial.

    Args:
        sync_uuid: The recording's ``sync_uuid``.
        paths: The stores without a ``camera_serial``.
    """
    message = (
        f"imgstore recording {sync_uuid!r} has stores with no camera_serial "
        f"{list(paths)}. Its cameras cannot be told apart. Record each "
        "camera's camera_serial in its store's metadata.yaml."
    )
    return StoreGroupingError(message)


def refuse_merged_store_recordings(
    rows: Iterable[Mapping[str, object]],
    *,
    beside: Iterable[Mapping[str, object]] = (),
) -> None:
    """Refuse the stores a scan derived into one entry that it cannot group.

    Only imgstore rows whose entry is still derived (``scan-imgstore-sync``) are
    compared. An entry that a caller assigned is kept. A store's recording is
    its ``sync_uuid``, else the store itself. Two stores of one camera that
    derive one entry are therefore refused too. A synchronized recording of two or more
    stores is refused when one of them names no camera: its ``camera_serial``
    and ``camera`` are both empty (an index older than ``camera_serial`` names
    the camera in ``camera``). *beside* holds the index rows the write keeps,
    compared only on an entry that *rows* write.

    Raises:
        StoreGroupingError: Naming the entry and the stores of each recording,
            or the recording and its stores without a camera.
    """
    fresh = list(rows)
    written = {
        (str(row.get("group", "")), str(row.get("sequence", ""))) for row in fresh
    }
    stores: dict[tuple[str, str], dict[str, list[str]]] = {}
    unnamed: dict[tuple[tuple[str, str], str], list[str]] = {}
    for row in [*beside, *fresh]:
        if row.get("media_type") != "imgstore":
            continue
        if row.get("assignment_source") != "scan-imgstore-sync":
            continue
        entry = (str(row.get("group", "")), str(row.get("sequence", "")))
        if entry not in written:
            continue
        path = str(row.get("abs_path", ""))
        sync_uuid = str(row.get("sync_uuid", ""))
        stores.setdefault(entry, {}).setdefault(sync_uuid or path, []).append(path)
        if sync_uuid and not (row.get("camera_serial") or row.get("camera")):
            unnamed.setdefault((entry, sync_uuid), []).append(path)
    for entry, recordings in sorted(stores.items()):
        for sync_uuid, members in sorted(recordings.items()):
            if len(members) > 1 and (entry, sync_uuid) in unnamed:
                paths = sorted(unnamed[entry, sync_uuid])
                raise unnamed_cameras_error(sync_uuid, paths)
        group, sequence = entry
        if len(recordings) < 2:
            continue
        paths = sorted(path for members in recordings.values() for path in members)
        message = (
            f"imgstores {paths} belong to {len(recordings)} recordings but all "
            f"derive the entry ({group!r}, {sequence!r}). Rename the store "
            "directories apart, or assign each recording its entry with "
            "write_media_index."
        )
        raise StoreGroupingError(message)


def _empty_order() -> dict[str, int]:
    """Typed default for :attr:`MediaIndexScope.order_by_name` (bare ``dict`` as a
    default_factory infers ``dict[Unknown, Unknown]`` under strict typing)."""
    return {}


def _empty_facts() -> dict[str, MediaFacts]:
    """Typed default for :attr:`MediaIndexScope.facts_by_name` (a bare ``dict``
    infers ``dict[Unknown, Unknown]`` under strict typing)."""
    return {}


@dataclass(frozen=True)
class MediaIndexScope:
    """One affected ``(group, sequence)`` to re-probe and reorder.

    ``directory`` is the media_raw subdir scanned for this sequence's files;
    every file found is assigned ``group``/``sequence`` (not derived from a
    track keymap). ``order_by_name`` maps a file's basename to its arranged
    linear position within the sequence; a file absent from the map keeps its
    prior ``video_order`` and sorts before the arranged ones. ``facts_by_name``
    maps a file's basename to an already-measured ``MediaFacts``; a file present
    in it is turned into its row from those facts instead of being re-probed.
    ``camera`` is an explicit override for a plain video that the caller tags
    with a camera axis. A probed imgstore supplies ``camera_serial`` and
    ``sync_uuid`` from store metadata, and its serial names its camera. The
    override applies only when the probe records no serial. Unsynchronized stores
    under one scope must share one ``camera_serial``, and stores of one camera
    join the sequence as clips.

    ``division`` divides the one imgstore that ``directory`` is into sequences
    of whole chunks. ``sequence`` is then ``""``, and ``group`` is the
    recording's name, or ``""`` to derive it from the store. A recording by
    camera is the stores of several scopes that share one ``group``.
    """

    directory: Path
    group: str
    sequence: str
    order_by_name: Mapping[str, int] = field(default_factory=_empty_order)
    facts_by_name: Mapping[str, MediaFacts] = field(default_factory=_empty_facts)
    camera: str = ""
    division: MediaDivision | None = None


def mtime_iso(timestamp: float) -> str:
    """UTC ISO-8601 string for a filesystem mtime."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class StoreStat:
    """The size and modification time that a media index row records for an imgstore.

    Attributes:
        size: The bytes of the store's files that
            :func:`~mosaic.core.media.imgstore_io.store_stat` covers.
        mtime: The newest modification time among those files, in seconds since
            the epoch.
    """

    size: int
    mtime: float


type RowStat = os.stat_result | StoreStat
"""What a media index row's ``size_bytes`` and ``mtime_iso`` are read from."""


def stat_cells(stat: RowStat) -> tuple[int, str]:
    """Return the ``size_bytes`` and ``mtime_iso`` cells of *stat*."""
    if isinstance(stat, StoreStat):
        return stat.size, mtime_iso(stat.mtime)
    return stat.st_size, mtime_iso(stat.st_mtime)


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
    read as text and held as an object dtype with NaN replaced by ``""``, so
    later cell writes do not trip pandas' incompatible-dtype warning on
    all-empty float columns, and a text cell such as a division's
    ``chunk_first`` of ``0`` is written back as it was read.

    An empty file reads as an empty frame, not as an error. ``index_lock``
    creates the index it is asked to hold when it does not exist yet -- a lock
    needs an inode -- so a caller that acquires the lock before its first read
    finds a zero-byte file where it expected none. Nothing else produces one:
    ``atomic_write`` renames a fully-written temp into place, so a zero-byte
    index is always a freshly-locked absent one rather than a truncated one.
    """
    if index_path.exists() and index_path.stat().st_size > 0:
        df = pd.read_csv(
            index_path,
            dtype={
                column: str
                for column in MEDIA_INDEX_COLUMNS
                if column not in MEDIA_NUMERIC_COLUMNS
            },
        )
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

    The projection drops any column outside the schema and fixes column order;
    the atomic write guarantees a concurrent reader never sees a partial file.
    """
    atomic_write(index_path, lambda p: df[MEDIA_INDEX_COLUMNS].to_csv(p, index=False))


def build_media_index_row(
    *,
    path: Path,
    stat: RowStat,
    to_store_path: Callable[[Path], str],
    group: str,
    sequence: str,
    group_safe: str,
    sequence_safe: str,
    probe: Mapping[str, object],
    name: str | None = None,
    camera: str = "",
    sync_uuid: str = "",
    camera_serial: str = "",
    media_type: str = "video",
    assignment_source: AssignmentSource = "",
    source_path: str | None = None,
    source_video_uuid: str | None = None,
    recipe_hash: str | None = None,
    encoder: str | None = None,
    video_order: int = 0,
) -> dict[str, object]:
    """Assemble one media-index row.

    *stat* is the file's ``os.stat_result``, or a store's :class:`StoreStat`
    (size and mtime come from it).
    *probe* is the width/height/fps/codec plus the injectable MediaFacts cells
    (a :class:`ProbeMetadata`). ``abs_path`` is produced by *to_store_path*, so
    the in-tree-relative / out-of-tree-absolute rule is enforced in one place.
    *source_path* overrides the empty ``source_path`` the probe carries (used by
    a derivative's back-link); *source_video_uuid* likewise overrides the empty
    ``source_video_uuid`` the probe carries with the source's ``video_uuid`` --
    the rename-resilient form of the same back-link, populated on a derivative
    row. *recipe_hash* likewise overrides the empty ``recipe_hash`` the probe
    carries with the recipe the derivative was produced under. *encoder*
    likewise overrides the empty ``encoder`` the probe carries with the video
    encoder that produced the derivative, which is empty for a copy remux --
    permitting hardware does not decide it, since a machine whose device cannot
    open the hardware encoder encodes on the CPU instead. *camera* is the
    within-sequence camera axis (``""`` for single-camera
    media) and *sync_uuid* the recording id that groups a recording's cameras;
    both feed :func:`densify_video_order` (which numbers ``video_order`` per
    ``(group, sequence, camera)``) and persist as schema columns.
    *camera_serial* is the serial an imgstore's metadata records, empty for a
    plain video. It is written as a fact and enters no composition.
    *assignment_source* records how this row's ``(group, sequence)`` was arrived
    at; it defaults to ``""`` so a caller that has not been taught the column
    writes an honest unknown rather than a wrong claim.
    """
    size_bytes, modified = stat_cells(stat)
    row: dict[str, object] = {
        "name": name if name is not None else path.name,
        "group": group,
        "sequence": sequence,
        "group_safe": group_safe,
        "sequence_safe": sequence_safe,
        "camera": camera,
        "sync_uuid": sync_uuid,
        "camera_serial": camera_serial,
        "abs_path": to_store_path(path),
        "size_bytes": size_bytes,
        "mtime_iso": modified,
        "media_type": media_type,
        "video_order": video_order,
        "assignment_source": assignment_source,
        **probe,
    }
    if source_path is not None:
        row["source_path"] = source_path
    if source_video_uuid is not None:
        row["source_video_uuid"] = source_video_uuid
    if recipe_hash is not None:
        row["recipe_hash"] = recipe_hash
    if encoder is not None:
        row["encoder"] = encoder
    return row


def build_prior_order(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str, str], int]:
    """Map ``(group, sequence, basename)`` to prior ``video_order`` for existing rows.

    Rows with a missing or blank ``video_order`` are skipped so they fall back
    to name ordering in :func:`densify_video_order`.
    """
    prior: dict[tuple[str, str, str], int] = {}
    for row in rows:
        raw = str(row.get("video_order", "")).strip()
        if not raw or raw.lower() == "nan":
            continue
        try:
            order = int(float(raw))
        except ValueError:
            continue
        prior[
            (
                str(row["group"]),
                str(row["sequence"]),
                Path(str(row["abs_path"])).name,
            )
        ] = order
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
    session_positions: Mapping[tuple[str, str, str], int],
    prior_order: Mapping[tuple[str, str, str], int],
) -> list[dict[str, object]]:
    """Re-number ``video_order`` as a dense counter per ``(group, sequence, camera)``.

    A thin adapter over :func:`assign_video_order`: it maps each media-index row
    into a :class:`VideoOrderKey`. Within each group the order is pre-existing
    videos first, keeping their prior ``video_order`` (*prior_order*), then this
    session's videos by arranged position (*session_positions*); the ``name``
    column breaks ties. Both maps are keyed ``(group, sequence, basename)`` where
    the basename comes from ``abs_path``; a session video is one present in
    *session_positions*. Keying the dense counter on ``camera`` makes this
    per-``(group, sequence, camera)``: a probed imgstore supplies a real
    ``camera``, so the parallel cameras of one recording -- which share a
    sequence -- are numbered independently rather than as temporal chunks of
    one camera. Mutates each row's ``video_order`` in place and returns the
    rows in the assigned order.
    """

    def key_of(row: dict[str, object]) -> VideoOrderKey:
        lookup = (
            str(row["group"]),
            str(row["sequence"]),
            Path(str(row["abs_path"])).name,
        )
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


def media_members_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str], list[MediaMember]]:
    """Group media-index rows into per-sequence composition members.

    Lives here rather than in ``composition`` because it reads the media schema,
    and ``composition`` is deliberately free of it -- the same split that keeps
    the hashing half testable with no fixture.

    Grouped on the exact ``(group, sequence)`` cells, the way
    ``resolve_media_scope`` groups, and never through ``_match_media_rows``,
    whose last tier answers case-insensitively for a sequence carrying no row of
    its own: a composition is built from the rows an entry actually has, so a
    match reached by that tier has nothing to contribute here.

    ``video_uuid`` is read through :func:`media_row_uuid`, which collapses ``""``,
    the string ``"nan"`` and a float NaN to one absent form. A raw cell read would
    hash the literal ``"nan"`` for a row a CSV round-trip had emptied.
    """
    members: dict[tuple[str, str], list[MediaMember]] = {}
    for row in rows:
        key = (str(row.get("group", "") or ""), str(row.get("sequence", "") or ""))
        raw_order = str(row.get("video_order", "") or "").strip()
        try:
            order = int(float(raw_order)) if raw_order else 0
        except ValueError:
            order = 0
        members.setdefault(key, []).append(
            MediaMember(
                camera=str(row.get("camera", "") or ""),
                video_order=order,
                uid=media_row_uuid(row),
            )
        )
    return members
