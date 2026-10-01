"""Rebuild what a tracks row's producer read of its entry, from the dataset's records.

A tracker or inference op records ``media_frames`` and ``frames_read`` as it
publishes. A row published before the cells existed has only its records.

- ``media_frames`` comes from the
  :class:`~mosaic.core.pipeline.placement.EntryAxis` of what the run read. The
  variant's ``params.json`` says whether the run read a media variant and whether
  a frame window narrowed it, and the media index says what the entry's media is.
  :func:`recorded_axis` rebuilds the same :class:`EntryAxis` from them, so a row
  filled later answers by the rule a row filled at publication answers by.
- ``frames_read`` is the tool's own count, which only something the run left on
  disk can tell: TRex's ``.pv``, a runner's response. Each producer that leaves
  such a thing registers a reader (:func:`register_frames_read_reader`), because
  ``core`` does not import ``tracking``. A producer without one leaves the cell
  blank.

Only a tool's read is rebuilt. A converted, resampled or upgraded table was made
from another table and read no media, and its producer is no tracking root.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.media_input import MediaInputParams
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.core.pipeline.tracks_identity import (
    VariantSidecar,
    read_tracks_variant,
    recorded_media,
    recorded_op_params,
)
from mosaic.core.pipeline.variant_source import VariantLookup

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "FramesReadReader",
    "recorded_axis",
    "recorded_frames_read",
    "register_frames_read_reader",
]

type FramesReadReader = Callable[["Dataset", Path, Path | None], int | None]
"""Return how many frames a producer's tool read for one published table, or ``None``.

Called with the dataset, the table, and the directory its producer's output was
read from (the row's ``source_abs_path``), or ``None`` when the row records none.
"""

_FRAMES_READ_READERS: Final[dict[str, FramesReadReader]] = {}


def recorded_axis(
    ds: Dataset, *, producer: str, variant: str, group: str, sequence: str
) -> EntryAxis | None:
    """Return the axis of what *producer* read for one tracks row, or ``None``.

    Args:
        ds: The dataset.
        producer: The row's producer.
        variant: The row's tracks variant, whose record says what the run read.
        group: The row's group.
        sequence: The row's sequence.

    Returns:
        The axis of the entry media, under the run's frame window, or of the media
        variant that the run read. ``None`` when the row is not a tool's read, or
        when what it read cannot be established: the variant has no record, the
        op that made it is not registered (``mosaic.tracking.register_ops``
        registers the tracking ops), the entry's media does not resolve, or the
        media variant's file for the entry is missing or has drifted.
    """
    if producer not in TRACKING_ROOTS or not variant:
        return None
    sidecar = read_tracks_variant(ds.get_root("tracks"), variant)
    if sidecar is None:
        return None
    windowed = _windowed(producer, sidecar)
    if windowed is None:
        return None
    try:
        scope = ds.resolve_media_scope([(group, sequence)], errors={})
    except FileNotFoundError:
        return None
    kept = one_camera_per_entry(producer, scope, report_skipped=False)
    if len(kept) != 1:
        return None
    (entry,) = kept
    media = recorded_media(sidecar)
    if not media:
        return EntryAxis.of_entry_media(entry.resolved.facts, windowed=windowed)
    try:
        source = VariantLookup.read(ds, media, [(group, sequence)]).resolve(ds, entry)
    except (MediaVariantMissingError, MediaVariantDriftedError, ValueError):
        return None
    return EntryAxis.of_variant(source.mapping())


def _windowed(producer: str, sidecar: VariantSidecar) -> bool | None:
    """Whether a frame window narrowed the run that *sidecar* records, or ``None``.

    The op's own parameters decide, from the values the run recorded
    (:meth:`~mosaic.core.pipeline.media_input.MediaInputParams.frame_window_of`).
    ``None`` when *producer* names no registered op.
    """
    op = OPS.get(producer)
    if op is None:
        return None
    params = op.Params
    if not issubclass(params, MediaInputParams):
        return False
    return bool(params.frame_window_of(recorded_op_params(sidecar)))


def register_frames_read_reader(producer: str, reader: FramesReadReader) -> None:
    """Declare how the frames that *producer*'s tool read are found after the run.

    Called at module scope by the producer, so importing it is what makes the
    reader available. Registering twice replaces.
    """
    _FRAMES_READ_READERS[producer] = reader


def recorded_frames_read(
    ds: Dataset, *, producer: str, table: Path, source: Path | None
) -> int | None:
    """Return how many frames *producer*'s tool read for *table*, from what it left.

    ``None`` when *producer* registered no reader, or its reader finds nothing.
    """
    reader = _FRAMES_READ_READERS.get(producer)
    return None if reader is None else reader(ds, table, source)
