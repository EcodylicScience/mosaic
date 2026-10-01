"""Publishing a tracker's output as a standardized tracks table.

Every tracker ends the same way: read the tool's own output through a registered
converter, validate it against the standard schema, write one parquet under the
tracks variant, and record the row that says where it came from.

**Conversion is the tracker's, publication is shared.** The three read different
formats -- an analysis HDF5, a DeepLabCut-style CSV, a set of per-individual NPZ
files merged on their column union -- and choose different converters and
converter params to do it. That part stays with the tracker. What happens to the
frame afterwards was written three times and is here once.

**The skip is variant-scoped, not path-scoped.** ``tracks/<variant>/`` names the
recipe, so asking whether the table exists asks whether *these settings* already
produced it, not whether any settings did. Before variants, two tracker runs with
different settings targeted one path behind an ``exists()`` check and the second
was discarded with a success return.

**Which schema is the producer's to declare.** It is read per tracker from its
``TrackingRoot.output_schema``, not spelled once here for all of them. As a
module constant it left a tracker whose columns genuinely differ nowhere to say
so, and it silently outranked every other spelling of the same question: because
all four trackers publish through this module, ``meta.tracks.standard_format``
had no effect on any tracked table at all.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import (
    FrameAxisVerdict,
    consumed_roots_for,
    frame_axis_verdict,
    tail_allowance,
    write_tracks_row,
)
from mosaic.core.pipeline.tracking_roots import tracking_root
from mosaic.core.schema import ensure_track_schema

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "BridgeCounts",
    "readable_tracks_table",
    "frame_counts",
    "publish_or_record",
    "publish_tracks_table",
    "tracks_table_path",
]


@dataclass(frozen=True, slots=True)
class BridgeCounts:
    """What one published tracks table holds.

    ``n_ids`` is the number of distinct ``id`` values, which for a tracker that
    maintains identities is not a count of animals -- see
    :class:`~mosaic.tracking.common.index.TrackerRunRowBase`.

    ``frames_read`` and ``media_frames`` are the two ends of one comparison: how
    many frames the producer's tool read, and how many it should have read
    (:attr:`~mosaic.core.pipeline.placement.EntryAxis.media_frames`). Both default
    to ``None``, which means not known and never zero. A reused table makes no
    measurement. ``producer`` is the tracking root that published the table,
    whose declaration the comparison reads, and ``""`` for counts of a table
    read back from disk. ``known_tail_loss`` is how many frames short of the end
    of the file it read the producer's tool is known to stop, ``None`` when
    unknown.

    ``dropped`` names the columns removed before the table was published because
    they do not map onto the source media's pixels, frames or clock, in the
    table's column order: those that a joined entry's retiming or a media
    variant's mapping removed. It is empty when the table kept every column.
    """

    n_rows: int
    n_ids: int
    frames_read: int | None = None
    media_frames: int | None = None
    dropped: tuple[str, ...] = ()
    producer: str = ""
    known_tail_loss: int | None = None

    @property
    def frame_axis(self) -> FrameAxisVerdict | None:
        """How the frames the tool read disagree with its media's, or ``None``.

        :func:`~mosaic.core.pipeline.tracks_index.frame_axis_verdict` decides, as
        it does for the recorded cells. ``None`` when either count is unknown as
        well as when they agree. The index comparison follows the same rule,
        because one measurement cannot disagree with an absent one.
        """
        if self.frames_read is None or self.media_frames is None:
            return None
        return frame_axis_verdict(
            self.producer,
            read=self.frames_read,
            media=self.media_frames,
            known_tail_loss=self.known_tail_loss,
        )


def tracks_table_path(ds: Dataset, tracks_variant: str, key: str) -> Path:
    """Where the table for one entry of one variant lives."""
    return tracks_variant_root(ds.get_root("tracks"), tracks_variant) / f"{key}.parquet"


def frame_counts(df: pd.DataFrame) -> BridgeCounts:
    """``(rows, distinct ids)`` for a tracks frame."""
    n_ids = int(df["id"].nunique()) if "id" in df.columns and len(df) else 0
    return BridgeCounts(n_rows=int(len(df)), n_ids=n_ids)


def readable_tracks_table(path: Path) -> BridgeCounts | None:
    """``(rows, distinct ids)`` for a table on disk, or ``None`` if unreadable.

    Three answers, not two, and the third is the whole point. A table that reads
    and holds no rows is a *legitimate* result -- a video in which the tracker
    found no individuals -- and the marker rules declare it reusable, so it must
    stay reusable. A table that cannot be read at all is not a result.

    This used to be ``existing_counts``, which collapsed the two: it caught
    ``(OSError, ValueError, KeyError)`` -- and pyarrow raises ``ArrowInvalid``, a
    ``ValueError`` subclass, on a truncated file -- and returned
    ``BridgeCounts(0, 0)``. A torn table was therefore adopted as a valid empty
    one, its zero written into the index row, and the run reported success. The
    old docstring even argued the correct case and then did the opposite: "a reuse
    run that returned nothing would replace a good row with a zero" is precisely
    what the ``except`` did.

    Reads only the ``id`` column, so a reuse check pays a column read rather than a
    full table load. That is enough to catch a torn file, because a parquet is
    unreadable without its footer and the footer is written last.

    ``None`` for an absent path too: a caller asking whether it can reuse a table
    wants one answer for "there is nothing usable here", not two.
    """
    try:
        existing = pd.read_parquet(path, columns=["id"])
    except (OSError, ValueError, KeyError):
        return None
    return frame_counts(existing)


def publish_tracks_table(
    ds: Dataset,
    df: pd.DataFrame,
    *,
    kind: str,
    group: str,
    sequence: str,
    tracks_variant: str,
    producer_run_id: str,
    source: Path,
    consumed: Sequence[Path],
    axis: EntryAxis,
    frames_read: int | None,
    known_tail_loss: int | None = None,
    strict: bool = False,
) -> BridgeCounts:
    """Write one converted frame as this variant's table for one entry.

    Every tracker and inference op publishes through here, so every table is
    placed on its entry's axes the same way, and every row records how many
    frames its tool read beside how many it should have read. The producer says
    only what it read, as *axis* and *frames_read*, and cannot leave either out.

    Args:
        ds: The dataset.
        df: The converted frame, already in standardized columns, on the axes
            of the file that the tool read.
        kind: The producing tracker, recorded as the row's ``producer``.
        group: The entry's group, which may be empty.
        sequence: The entry's sequence.
        tracks_variant: What names the recipe these tables belong to. Names the
            directory as well as the row.
        producer_run_id: The tracker run that produced the tool output.
        source: The directory the tool output was read from.
        consumed: Every file this table was derived from -- the tool's output,
            the video, and any model files. Only those under a dataset root
            contribute; an external model directory sits under none, which is
            correct, because its identity is already in the run identifier.
        axis: What the tool read of the entry's media. *df* is placed on the
            entry's axes first (:meth:`EntryAxis.place`): a media variant's
            table is mapped into source space, and a table from the join of
            several clips is timed by the clips. The table validated, written
            and counted is the placed one. The row records
            :attr:`EntryAxis.media_frames`, or a blank when it is ``None``.
        frames_read: How many frames the tool read for *df*, on the axis of the
            file it read, or ``None`` when the producer cannot know. The row
            records it. It is not the table's extent: a table with rows only
            where something was detected ends at its last detection.
        known_tail_loss: How many frames short of the end of the file it read
            the tool is known to stop, from that file's header by the rule of
            the producer's root (``TrackingRoot.tail_loss``). The row records
            it. ``None`` for a producer that declares no loss, and where the
            file's header was not read: a shortfall is then allowed the most the
            producer loses on any file.
        strict: Raise when the table lacks a column that its schema requires,
            rather than printing the report and publishing it.

    Returns:
        Counts of the published table, and every column dropped from it.

    Raises:
        UnclassifiedColumnError: If *axis* reads a variant and the table has a
            numeric column that the mapping cannot classify.
        ValueError: If *axis* reads a variant and a frame column contains a
            value that is not a whole frame number, or a clip of *axis* reports
            no frame rate.
        ForbiddenTrackColumnError: If the table has a column that its schema
            forbids, whatever *strict* says.
        TrackSchemaError: If *strict* and the table lacks a required column.
    """
    root = tracking_root(kind)
    placed = axis.place(df)
    df = placed.frame
    media_frames = axis.media_frames
    out_path = tracks_table_path(ds, tracks_variant, make_entry_key(group, sequence))
    std_format = root.output_schema
    ensure_track_schema(df, std_format, strict=strict, source=f"{group}/{sequence}")

    _ = write_parquet_atomic(df, out_path)

    counts = frame_counts(df)
    write_tracks_row(
        ds,
        run_id=tracks_variant,
        group=group,
        sequence=sequence,
        out_path=out_path,
        producer=kind,
        std_format=std_format,
        n_rows=counts.n_rows,
        producer_run_id=producer_run_id,
        source=source,
        consumed_source_roots=consumed_roots_for(ds, list(consumed)),
        media_frames=media_frames,
        frames_read=frames_read,
        known_tail_loss=known_tail_loss,
        # A bridge opens the entry's media, so its row records what that
        # media was. The variant identity has no term for the pixels, so
        # this cell is the only thing that notices a re-transcode.
        records_media=True,
    )
    return replace(
        counts,
        frames_read=frames_read,
        media_frames=media_frames,
        dropped=placed.dropped,
        producer=kind,
        known_tail_loss=known_tail_loss,
    )


def publish_or_record(
    ctx: JobContext,
    key: str,
    publish: Callable[[], BridgeCounts | None],
    *,
    kind: str,
) -> BridgeCounts | None:
    """Run one entry's bridge, recording a failure instead of hiding it.

    A tracker run's declared output is a tracks table. When the conversion of a
    finished entry raises, the run has lost that entry's entire published result
    -- and every tracker used to answer that by printing to stderr and returning
    ``None``, which its caller discarded. The run then reported ``finished`` and
    exited 0 having published nothing, and under ``mosaic-queue``, which gives
    the child's stderr to ``DEVNULL``, the message did not exist at all. A real
    TREx run tracked a 3.5-hour session and produced no table this way; it was
    found by noticing ``tracks/`` was empty.

    So the failure goes to the run-log through :meth:`JobContext.entry_failed`,
    which is the channel that survives the queue: it appends to ``failed_keys``
    and emits an ``entry_error`` event that ``reduce_run_log`` folds into
    ``entries_failed``, and the CLI reports the attempt as ``partial``. That is
    the invariant the feature pipeline already keeps -- *a run reports what it
    lost* -- applied to the layer that never adopted it.

    **The entry's index row is still written by the caller.** The tool output is
    real and durable, and recording it is what lets a re-run adopt the finished
    directory and redo only the bridge -- seconds rather than hours. A failed
    bridge means the publication was lost, not the tracking.

    **A frame-axis mismatch is reported here too, and is not a failure.** When the
    tool read a different number of frames than the media it was given holds, the
    entry succeeded: the table is schema-valid, and every quantity computed
    inside it is right. What may be wrong is the correspondence between a
    ``frame`` in that table and a frame of the video, so `overlay`, the crop
    features and frame extraction can read the wrong image. How far out, and
    where, depends on which frames the tool missed. Frames missed at the end
    move nothing, and frames missed earlier move every frame after them.
    mosaic knows only the size of the gap. It reports that and says so.

    A shortfall within what the producer's tool is known to lose at the end of
    the file it read (``TrackingRoot.tail_loss``) is reported apart, as a
    ``frame_tail_short`` event. It is the loss that tool is known to have on that
    file, and still a count that cannot show where the frames went.

    Raising instead was considered and rejected. It would be permanent: the
    condition is deterministic, so every re-run and every TRex republish would
    fail the same entry, and the other trackers cannot re-publish a table without
    re-tracking at all -- their bridge serves an existing parquet before it
    converts anything, and ``overwrite`` clears the whole working tree. A defect
    that spoils registration would then cost the analyses that never depended on
    registration.

    **Dropped columns are reported the same way.** A table mapped from a media
    variant into source space publishes without the columns that the mapping
    cannot correct, and a table from the join of an entry's clips without the
    columns computed against its single frame rate. The entry succeeded, and the
    names go to the run-log as a ``columns_dropped`` event and to stderr.

    Args:
        ctx: The attempt's Job Contract, which owns the run-log.
        key: The entry's ``<group>__<sequence>`` key, named in the event.
        publish: The tracker's own bridge call, deferred so this can wrap it.
        kind: The tracker, for the stderr line that accompanies the record.

    Returns:
        Whatever *publish* returned, or ``None`` when it raised.
    """
    try:
        counts = publish()
    except Exception as exc:  # noqa: BLE001 - recorded on the attempt, not hidden
        # Inside the except block on purpose: entry_failed captures
        # traceback.format_exc(), which only has a traceback while one is being
        # handled. Called before the print for the same reason -- the record is
        # the point, and the print is the convenience.
        ctx.entry_failed(key, exc)
        print(
            f"[{kind}] publishing {key} failed: {type(exc).__name__}: {exc}; "
            f"the tracker output is kept, so a re-run will retry the conversion",
            file=sys.stderr,
        )
        return None
    if counts is not None:
        _report_frame_axis(ctx, key, counts, kind=kind)
    if counts is not None and counts.dropped:
        ctx.columns_dropped(key, counts.dropped)
        # Emit the event and the line, as for a frame-axis mismatch above.
        verb = "does" if len(counts.dropped) == 1 else "do"
        print(
            f"[{kind}] {key}: published without {', '.join(counts.dropped)}, "
            f"which {verb} not map onto the source media's pixels, frames or "
            f"clock.",
            file=sys.stderr,
        )
    return counts


def _report_frame_axis(
    ctx: JobContext, key: str, counts: BridgeCounts, *, kind: str
) -> None:
    """Record how the frames one entry's tool read disagree with its media's.

    Emits both the event and the line, for the reason ``entry_failed`` keeps both.
    The event is the record that survives a queue sending stderr to DEVNULL, and
    the line is the one a person running this in a terminal sees.
    """
    verdict = counts.frame_axis
    if verdict is None or counts.frames_read is None or counts.media_frames is None:
        return
    read, media = counts.frames_read, counts.media_frames
    if verdict == "tail_short":
        ctx.frame_tail_short(key, read=read, media=media)
        allowance = tail_allowance(counts.producer, counts.known_tail_loss)
        print(
            f"[{kind}] {key}: the tool read {read} of {media} frames, within "
            f"{allowance.describe(counts.producer)}. A count cannot show that "
            f"the missing frames are at the end.",
            file=sys.stderr,
        )
        return
    ctx.frame_axis_mismatch(key, read=read, media=media)
    print(
        f"[{kind}] {key}: the tool read {read} of {media} frames. Unless the "
        f"difference is all at the end, a frame of this table may be up to "
        f"{abs(media - read)} frames from the video frame of that number. "
        f"Check any overlay or crop from this entry before trusting it. "
        f"Everything computed inside the table is unaffected.",
        file=sys.stderr,
    )
