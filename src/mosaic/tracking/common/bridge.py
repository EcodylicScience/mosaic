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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.pipeline.tracking_roots import tracking_output_schema
from mosaic.core.schema import ensure_track_schema

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "BridgeCounts",
    "readable_tracks_table",
    "frame_counts",
    "publish_tracks_table",
    "tracks_table_path",
]


@dataclass(frozen=True, slots=True)
class BridgeCounts:
    """What one published tracks table holds.

    ``n_ids`` is the number of distinct ``id`` values, which for a tracker that
    maintains identities is not a count of animals -- see
    :class:`~mosaic.tracking.common.index.TrackerRunRowBase`.
    """

    n_rows: int
    n_ids: int


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
    )
    return counts
