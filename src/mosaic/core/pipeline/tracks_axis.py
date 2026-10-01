"""Rebuild what a tracks row's producer read of its entry, from the dataset's records.

A tracker or inference op records ``media_frames`` and ``frames_read`` as it
publishes. A row published before the cells existed has only its records.

- ``media_frames`` comes from the
  :class:`~mosaic.core.pipeline.placement.EntryAxis` of what the run read. The
  variant's ``params.json`` says whether the run read a media variant and whether
  a frame window narrowed it, and the media index says what the entry's media is.
  :func:`recorded_media_frames` rebuilds the same :class:`EntryAxis` from them, so
  a row filled later answers by the rule a row filled at publication answers by.
- ``frames_read`` is the tool's own count, which only something the run left on
  disk can tell: TRex's ``.pv``, a runner's response. Each producer that leaves
  such a thing registers a reader (:func:`register_frames_read_reader`), because
  ``core`` does not import ``tracking``.
- ``known_tail_loss`` is what the producer's rule (``TrackingRoot.tail_loss``)
  gives the header of the file its tool read. Only the run knows which file that
  was, so the producer registers a reader of it too
  (:func:`register_tail_loss_reader`), which applies :func:`tail_loss_of_files`.

Each answers in one of four ways, and a caller rewriting a cell needs all four
apart. A :class:`RuleValue` with a count is the cell's value. A :class:`RuleValue`
with none is a cell the rule leaves blank: a converted, resampled or upgraded table
was made from another table and read no media, its producer is no tracking root,
and neither count applies to it, and a run that read less on purpose has no media
length. The two :data:`Unanswered` values leave the cell as it is:
``"not-established"`` is a run that cannot be established from what is on disk
now, because its record, its media or its files are missing or have changed, and
``"unregistered"`` is a producer whose op this process has not registered, so
nothing about its run can be read here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from mosaic_media import MediaProbeError
from mosaic_media.probe.ffprobe import read_header

from mosaic.core.params import Params
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.consumed_camera import one_camera_per_entry
from mosaic.core.pipeline.media_input import MediaInputParams
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
)
from mosaic.core.pipeline.sequence_index import media_compositions_for
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
    "BLANK",
    "RecordedCountReader",
    "RuleValue",
    "Unanswered",
    "recorded_frames_read",
    "recorded_media_frames",
    "recorded_tail_loss",
    "register_frames_read_reader",
    "register_tail_loss_reader",
    "tail_loss_of_files",
]

type RecordedCountReader = Callable[["Dataset", Path, Path | None], int | None]
"""Return one count that a producer's run left on disk for one published table.

Called with the dataset, the table, and the directory its producer's output was
read from (the row's ``source_abs_path``), or ``None`` when the row records none.
``None`` when what the run left does not tell.
"""

_FRAMES_READ_READERS: Final[dict[str, RecordedCountReader]] = {}
_TAIL_LOSS_READERS: Final[dict[str, RecordedCountReader]] = {}


@dataclass(frozen=True, slots=True)
class RuleValue:
    """The value that the publication rule gives one count cell of a tracks row.

    Attributes:
        count: The cell's count, or ``None`` where the rule leaves it blank.
    """

    count: int | None


BLANK: Final = RuleValue(None)
"""A cell the rule leaves blank."""

type Unanswered = Literal["not-established", "unregistered"]
"""Why a rule gives one cell no value, so that the cell keeps what it holds.

``"not-established"``: what the row's run read cannot be established from what
is on disk now. ``"unregistered"``: the row's producer is a tracking op that this
process has not registered. Importing ``mosaic.tracking`` registers every
tracking op, and with it every reader of frames read, so a process that imported
only ``mosaic.core`` answers this for every tracker row.
"""


def recorded_media_frames(
    ds: Dataset,
    *,
    producer: str,
    variant: str,
    group: str,
    sequence: str,
    consumed_media: str,
) -> RuleValue | Unanswered:
    """Return the ``media_frames`` that the bridge's rule gives one tracks row.

    Args:
        ds: The dataset.
        producer: The row's producer.
        variant: The row's tracks variant, whose record says what the run read.
        group: The row's group.
        sequence: The row's sequence.
        consumed_media: The row's ``consumed_media_composition``: what the entry's
            media was when the run read it, or ``""`` when not recorded.

    Returns:
        The frames of the entry media, or of the media variant, that the run was
        to read. :data:`BLANK` for a producer that is no tracking root, a run
        under a frame window, and a trimmed or decimated media variant.
        ``"unregistered"`` when the op that made the row is not registered in
        this process. ``"not-established"`` when what the run read cannot be
        established: the variant has no record, the entry's media has changed
        since the run or does not resolve, a clip's frame count is unknown, or
        the media variant's file for the entry is missing or has drifted.
    """
    if producer not in TRACKING_ROOTS:
        return BLANK
    op = OPS.get(producer)
    if op is None:
        return "unregistered"
    if not variant:
        return "not-established"
    sidecar = read_tracks_variant(ds.get_root("tracks"), variant)
    if sidecar is None:
        return "not-established"
    if _windowed(op.Params, sidecar):
        return BLANK
    if _media_changed(ds, group, sequence, consumed_media):
        return "not-established"
    entry_key = (group, sequence)
    try:
        scope = ds.resolve_media_scope([entry_key], errors={})
    except FileNotFoundError:
        return "not-established"
    kept = one_camera_per_entry(producer, scope, report_skipped=False)
    if len(kept) != 1:
        return "not-established"
    (entry,) = kept
    media = recorded_media(sidecar)
    if not media:
        axis = EntryAxis.of_entry_media(entry.resolved.facts, windowed=False)
    else:
        try:
            lookup = VariantLookup.read(ds, media, [entry_key])
            source = lookup.resolve(ds, entry)
        except (MediaVariantMissingError, MediaVariantDriftedError, ValueError):
            return "not-established"
        axis = EntryAxis.of_variant(source.mapping())
    if not axis.reads_every_frame:
        return BLANK
    count = axis.media_frames
    return "not-established" if count is None else RuleValue(count)


def _media_changed(ds: Dataset, group: str, sequence: str, consumed_media: str) -> bool:
    """Whether the entry's media has changed since the run that read it.

    *consumed_media* is the row's ``consumed_media_composition``. A blank one, or
    an entry whose composition is not projected, is unknown and not a change
    (:func:`~mosaic.core.pipeline.composition.compositions_disagree`).
    """
    entry_key = (group, sequence)
    now = media_compositions_for(ds, [entry_key]).get(entry_key, "")
    return compositions_disagree(consumed_media, now)


def _windowed(params: type[Params], sidecar: VariantSidecar) -> bool:
    """Whether a frame window narrowed the run that *sidecar* records.

    The op's own parameters, *params*, decide from the values the run recorded
    (:meth:`~mosaic.core.pipeline.media_input.MediaInputParams.frame_window_of`).
    """
    if not issubclass(params, MediaInputParams):
        return False
    return bool(params.frame_window_of(recorded_op_params(sidecar)))


def register_frames_read_reader(producer: str, reader: RecordedCountReader) -> None:
    """Declare how the frames that *producer*'s tool read are found after the run.

    Called at module scope by the producer, so importing it is what makes the
    reader available. Registering twice replaces.
    """
    _FRAMES_READ_READERS[producer] = reader


def recorded_frames_read(
    ds: Dataset, *, producer: str, table: Path, source: Path | None
) -> RuleValue | Unanswered:
    """Return the ``frames_read`` that what *producer*'s run left gives *table*.

    Returns:
        How many frames the tool read. :data:`BLANK` for a producer that is no
        tracking root, whose table no tool made from media. ``"unregistered"``
        when *producer*'s op is not registered in this process, which has then
        not imported its reader either. ``"not-established"`` when the
        registered *producer* has no reader, because its run leaves nothing
        that tells, or its reader finds nothing: a missing file establishes
        nothing.
    """
    if producer not in TRACKING_ROOTS:
        return BLANK
    if producer not in OPS:
        return "unregistered"
    reader = _FRAMES_READ_READERS.get(producer)
    count = None if reader is None else reader(ds, table, source)
    return "not-established" if count is None else RuleValue(count)


def tail_loss_of_files(producer: str, files: Sequence[str]) -> int | None:
    """How many frames short of the end of *files* *producer*'s tool is known to stop.

    *files* are what the tool read, as its run recorded them. One file answers
    by the rule its producer's root declares (``TrackingRoot.tail_loss``), from
    the file's header: a header read, not a scan of every packet. Several files
    answer 0: a tool reading several files loses frames at each boundary, which
    moves every frame after it, so no shortfall of theirs is a loss at the end.

    Returns:
        The known loss, or ``None`` when *producer* declares none, no file is
        recorded, or the file's header cannot be read.
    """
    root = TRACKING_ROOTS.get(producer)
    loss = None if root is None else root.tail_loss
    if loss is None or not files:
        return None
    if len(files) > 1:
        return 0
    try:
        header = read_header(Path(files[0]))
    except (MediaProbeError, OSError):
        return None
    return loss.of_file(header)


def register_tail_loss_reader(producer: str, reader: RecordedCountReader) -> None:
    """Declare how the known tail loss of *producer*'s tool is found after the run.

    The reader finds the file the tool read and applies :func:`tail_loss_of_files`.
    Called at module scope by the producer, as :func:`register_frames_read_reader`
    is. Registering twice replaces.
    """
    _TAIL_LOSS_READERS[producer] = reader


def recorded_tail_loss(
    ds: Dataset,
    *,
    producer: str,
    table: Path,
    source: Path | None,
    group: str,
    sequence: str,
    consumed_media: str,
) -> RuleValue | Unanswered:
    """Return the ``known_tail_loss`` that what *producer*'s run left gives *table*.

    The reader reads the header of the file at the path the run recorded. Once
    the entry's media has changed since the run, another file may be there, so
    such a row is not established, as it is by :func:`recorded_media_frames`.

    Args:
        ds: The dataset.
        producer: The row's producer.
        table: The row's table.
        source: The directory the table was read from, or ``None``.
        group: The row's group.
        sequence: The row's sequence.
        consumed_media: The row's ``consumed_media_composition``.

    Returns:
        The loss its tool is known to have at the end of the file it read.
        :data:`BLANK` for a producer that is no tracking root, or whose root
        declares no loss. ``"unregistered"`` when *producer*'s op is not
        registered in this process. ``"not-established"`` when the entry's media
        has changed since the run, its reader finds no file, or the file's
        header cannot be read.
    """
    root = TRACKING_ROOTS.get(producer)
    if root is None or root.tail_loss is None:
        return BLANK
    if producer not in OPS:
        return "unregistered"
    if _media_changed(ds, group, sequence, consumed_media):
        return "not-established"
    reader = _TAIL_LOSS_READERS.get(producer)
    count = None if reader is None else reader(ds, table, source)
    return "not-established" if count is None else RuleValue(count)
