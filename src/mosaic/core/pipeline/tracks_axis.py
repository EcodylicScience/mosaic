"""Rebuild what a tracks row's producer read of its entry, from the dataset's records.

A tracker or inference op records ``media_frames``, ``frames_read`` and
``known_tail_loss`` as it publishes. A row published before the cells existed has
only its records.

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
with none is a cell the rule leaves blank. Every cell of a converted, resampled or
upgraded table is blank, because its producer is no tracking root and it was made
from another table without reading media. ``media_frames`` is blank for a run
that read less on purpose, and ``known_tail_loss`` for a producer that declares no
loss. The two :data:`Unanswered` values leave the cell as it is.
``"not-established"`` is a run that cannot be established from what is on disk
now, because its record, its media or its files are missing or have changed.
``"unregistered"`` is a producer whose op this process has not registered, so
none of its run's records can be read here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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
    media_variant_rows,
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
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.entry import Entry

__all__ = [
    "BLANK",
    "BackfillReads",
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
"""Why a rule gives one cell no value, so that the cell keeps its current value.

``"not-established"``: what the row's run read cannot be established from what
is on disk now. ``"unregistered"``: the row's producer is a tracking op that this
process has not registered. Importing ``mosaic.tracking`` registers every
tracking op, and with it every reader that a producer registers, so a process
that imported only ``mosaic.core`` answers this for every tracker row.
"""


class BackfillReads:
    """The dataset records that a pass over the tracks index reads once for every row.

    A rule asked about one row reads the entry's media, its media composition,
    its variant's record, and the rows of the media variant its run read. Each
    is read here on first use and kept for the rest of the pass. The media and
    the compositions are read for every entry of the pass at once, so a pass
    reads the media index and the sequence projection once rather than once per
    row.
    """

    def __init__(self, ds: Dataset, entries: Iterable[Entry]) -> None:
        """Prepare to read, for the entries *entries* of *ds*."""
        self._ds: Final = ds
        self._entries: Final = tuple(dict.fromkeys(entries))
        self._compositions: Mapping[Entry, str] | None = None
        self._scope_read = False
        self._scope: Mapping[Entry, tuple[ResolvedScopeEntry, ...]] | None = None
        self._sidecars: Final[dict[str, VariantSidecar | None]] = {}
        self._lookups: Final[dict[str, VariantLookup]] = {}

    def media_changed(self, entry: Entry, consumed_media: str) -> bool:
        """Whether *entry*'s media has changed since the run that read it.

        *consumed_media* is the row's ``consumed_media_composition``. A blank one,
        or an entry whose composition is not projected, is unknown and not a change
        (:func:`~mosaic.core.pipeline.composition.compositions_disagree`).
        """
        return compositions_disagree(
            consumed_media, self._current_compositions().get(entry, "")
        )

    def scope(self, entry: Entry) -> tuple[ResolvedScopeEntry, ...] | None:
        """*entry*'s media, one item for each camera, or ``None`` without a media index.

        An entry whose media cannot be routed has no items.
        """
        if not self._scope_read:
            self._scope_read = True
            try:
                resolved = self._ds.resolve_media_scope(self._entries, errors={})
            except FileNotFoundError:
                resolved = None
            if resolved is not None:
                by_entry: dict[Entry, list[ResolvedScopeEntry]] = {}
                for item in resolved:
                    by_entry.setdefault((item.group, item.sequence), []).append(item)
                self._scope = {key: tuple(items) for key, items in by_entry.items()}
        if self._scope is None:
            return None
        return self._scope.get(entry, ())

    def sidecar(self, variant: str) -> VariantSidecar | None:
        """The record of tracks variant *variant*, or ``None`` when it has none."""
        if variant not in self._sidecars:
            tracks = self._ds.get_root("tracks")
            self._sidecars[variant] = read_tracks_variant(tracks, variant)
        return self._sidecars[variant]

    def lookup(self, media: str) -> VariantLookup:
        """The rows of media variant *media*, against the entries' compositions."""
        lookup = self._lookups.get(media)
        if lookup is None:
            lookup = VariantLookup(
                run_id=media,
                rows=media_variant_rows(self._ds, media),
                compositions=self._current_compositions(),
            )
            self._lookups[media] = lookup
        return lookup

    def _current_compositions(self) -> Mapping[Entry, str]:
        """The current media composition of every entry of the pass."""
        if self._compositions is None:
            self._compositions = media_compositions_for(self._ds, self._entries)
        return self._compositions


def recorded_media_frames(
    ds: Dataset,
    *,
    producer: str,
    variant: str,
    group: str,
    sequence: str,
    consumed_media: str,
    reads: BackfillReads,
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
        reads: The records that the pass reads once for all its rows.

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
    sidecar = reads.sidecar(variant)
    if sidecar is None:
        return "not-established"
    if _windowed(op.Params, sidecar):
        return BLANK
    entry_key = (group, sequence)
    if reads.media_changed(entry_key, consumed_media):
        return "not-established"
    scope = reads.scope(entry_key)
    if scope is None:
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
            source = reads.lookup(media).resolve(ds, entry)
        except (MediaVariantMissingError, MediaVariantDriftedError, ValueError):
            return "not-established"
        axis = EntryAxis.of_variant(source.mapping())
    if not axis.reads_every_frame:
        return BLANK
    count = axis.media_frames
    return "not-established" if count is None else RuleValue(count)


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

    Called at module scope by the producer, so importing the producer registers
    the reader. Registering twice replaces.
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
    reads: BackfillReads,
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
        reads: The records that the pass reads once for all its rows.

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
    if reads.media_changed((group, sequence), consumed_media):
        return "not-established"
    reader = _TAIL_LOSS_READERS.get(producer)
    count = None if reader is None else reader(ds, table, source)
    return "not-established" if count is None else RuleValue(count)
