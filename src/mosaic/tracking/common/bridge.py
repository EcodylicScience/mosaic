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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import consumed_roots_for, write_tracks_row
from mosaic.core.schema import ensure_track_schema

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "BridgeCounts",
    "existing_counts",
    "frame_counts",
    "publish_tracks_table",
    "tracks_table_path",
]

# Every tracker bridges into the one generic standardized schema. A tracker whose
# columns genuinely differ would register its own; none does, and registering a
# near-copy would fragment what downstream features read.
STANDARD_FORMAT = "trex_v1"


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


def existing_counts(path: Path) -> BridgeCounts:
    """``(rows, distinct ids)`` read back from a table already on disk.

    Reads only the ``id`` column, so a reuse run pays a column read rather than a
    full table load. Re-derived rather than reported as unknown: the index row
    carries the count, and a reuse run that returned nothing would replace a good
    row with a zero.
    """
    try:
        existing = pd.read_parquet(path, columns=["id"])
    except (OSError, ValueError, KeyError):
        return BridgeCounts(n_rows=0, n_ids=0)
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
    ensure_track_schema(df, STANDARD_FORMAT, strict=False, source=f"{group}/{sequence}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    counts = frame_counts(df)
    write_tracks_row(
        ds,
        run_id=tracks_variant,
        group=group,
        sequence=sequence,
        out_path=out_path,
        producer=kind,
        std_format=STANDARD_FORMAT,
        n_rows=counts.n_rows,
        producer_run_id=producer_run_id,
        source=source,
        consumed_source_roots=consumed_roots_for(ds, list(consumed)),
    )
    return counts
