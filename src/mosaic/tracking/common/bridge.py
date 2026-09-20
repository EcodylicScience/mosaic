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
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.pipeline.tracking_roots import tracking_output_schema
from mosaic.core.pipeline.types.data_config import COLUMNS
from mosaic.core.schema import ensure_track_schema

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "BridgeCounts",
    "readable_tracks_table",
    "frame_counts",
    "frame_span",
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

    ``frame_span`` and ``media_frames`` are the two ends of one comparison: how
    far the table's own frame axis reaches, and how long the axis of the media it
    came from was. Both default to ``None``, which means *not measured* and never
    zero -- a reused table makes no measurement, and a producer that does not
    join a session's clips has no second axis to compare against.
    """

    n_rows: int
    n_ids: int
    frame_span: tuple[int, int] | None = None
    media_frames: int | None = None

    @property
    def frame_axis_mismatch(self) -> tuple[int, int] | None:
        """``(tracked, media)`` when the two axes disagree, else ``None``.

        ``None`` when either is unknown as well as when they agree: the honest
        -empty rule the index comparison follows, because one measurement cannot
        disagree with an absent one.
        """
        if self.frame_span is None or self.media_frames is None:
            return None
        tracked = self.frame_span[1] + 1
        return None if tracked == self.media_frames else (tracked, self.media_frames)


def tracks_table_path(ds: Dataset, tracks_variant: str, key: str) -> Path:
    """Where the table for one entry of one variant lives."""
    return tracks_variant_root(ds.get_root("tracks"), tracks_variant) / f"{key}.parquet"


def frame_counts(df: pd.DataFrame) -> BridgeCounts:
    """``(rows, distinct ids)`` for a tracks frame."""
    n_ids = int(df["id"].nunique()) if "id" in df.columns and len(df) else 0
    return BridgeCounts(n_rows=int(len(df)), n_ids=n_ids)


def frame_span(df: pd.DataFrame) -> tuple[int, int] | None:
    """The ``(min, max)`` of a frame's ``frame`` column, in memory.

    The sibling of
    :func:`~mosaic.core.pipeline.tracks_index.frame_extent`, which measures the
    same thing off the parquet. Two spellings, deliberately: that one is what the
    index writer uses, and its rule that the extent is *measured from the file
    rather than passed in* is what keeps six call sites from each being able to
    record a false one. This one answers for the caller that is holding the frame
    anyway and wants the number before it is written.

    ``None`` when the answer is unknown -- no ``frame`` column, or every value
    null -- which is not the same claim as ``(0, 0)``.
    """
    if COLUMNS.frame_col not in df.columns:
        return None
    frames = pd.to_numeric(df[COLUMNS.frame_col], errors="coerce").dropna()
    if frames.empty:
        return None
    return int(frames.min()), int(frames.max())


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
    media_frames: int | None = None,
) -> BridgeCounts:
    """Write one converted frame as this variant's table for one entry.

    Args:
        ds: The dataset.
        df: The converted frame, already in standardized columns.
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
        media_frames: How long the media axis this table's frames are supposed to
            address was. ``None`` -- the default every producer takes until it
            says otherwise -- records a blank cell and makes no comparison.

            The producer's to supply, not this function's to derive, for the same
            reason ``records_media`` is: only the caller knows what it resolved
            and how much of it the tool was asked to read. Which producers can
            answer is declared knowledge --
            :attr:`~mosaic.core.pipeline.tracking_roots.TrackingRoot.joins_sources`
            marks the ones whose tool is handed a whole session, and those are
            the ones where the two axes can come apart.
    """
    out_path = tracks_table_path(ds, tracks_variant, make_entry_key(group, sequence))
    std_format = tracking_output_schema(kind)
    ensure_track_schema(df, std_format, strict=False, source=f"{group}/{sequence}")

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
        # A bridge opens the entry's media, so its row records what that
        # media was. The variant identity has no term for the pixels, so
        # this cell is the only thing that notices a re-transcode.
        records_media=True,
    )
    return replace(counts, frame_span=frame_span(df), media_frames=media_frames)


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
    published table's frame axis is not as long as the media it was made from,
    the entry succeeded: the table is schema-valid, its rows are dense, and every
    quantity computed inside it is right. What is wrong is the correspondence
    between a ``frame`` in that table and a frame of the video -- so `overlay`,
    the crop features and frame extraction can read the wrong image. How far
    out, and where, depends on what the tool did with the frames it missed, and
    mosaic knows only the size of the gap. It reports that and says so.

    Raising instead was considered and rejected. It would be permanent: the
    condition is deterministic, so every re-run would fail the same entry, and
    there is no way to re-publish a table without re-tracking -- the bridge
    serves an existing parquet before it converts anything, and ``overwrite``
    clears the whole working tree. A defect that spoils registration would then
    cost the analyses that never depended on registration.

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
    if counts is not None and (mismatch := counts.frame_axis_mismatch) is not None:
        tracked, media = mismatch
        ctx.frame_axis_mismatch(key, tracked=tracked, media=media)
        # Both the event and the line, for the reason `entry_failed` keeps both:
        # the event is the record that survives a queue sending stderr to
        # DEVNULL, and the line is what a person running this in a terminal sees.
        print(
            f"[{kind}] {key}: this table spans {tracked} frames but its media "
            f"holds {media}, so the two are not one axis and a frame read for "
            f"this table may be up to {abs(media - tracked)} frames out. Check "
            f"any overlay or crop from this entry before trusting it. "
            f"Everything computed inside the table is unaffected.",
            file=sys.stderr,
        )
    return counts
