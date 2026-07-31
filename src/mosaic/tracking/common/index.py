"""The run index every integrated tracker keeps, and the columns they share.

Each tracker records one row per entry under ``_tracking/<kind>/index.csv``: what
run it belonged to, which entry, which video, and where its outputs landed. Three
row classes declared the same four columns before adding their own, and three
copies of the accessors around them.

**Renaming the count column.** SLEAP called it ``n_tracks`` and TREx and Lightning
Pose called it ``n_individuals``, for what a reader would take to be one quantity.
It is not: a SLEAP ``Track`` is an identity the tracker maintained, and
fragmentation across an occlusion splits one animal into several, which is what
``--tracking.clean_instance_count`` exists to repair. TREx's was a count of
per-individual files. So the shared column is ``n_ids`` and means exactly what it
can mean for all three -- how many distinct ``id`` values the produced tracks
table holds -- and the docstring says plainly that it is not a headcount. A name
that overstated it is worse than a name that does not try.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pandas as pd

from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase, project_to_schema

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "TrackerRunRowBase",
    "list_tracker_runs",
    "tracker_index",
    "tracker_index_path",
]

RowT = TypeVar("RowT", bound="TrackerRunRowBase")

# Legacy spellings of the count column, mapped onto n_ids when an index written
# before the rename is next appended to. Kept as data so the mapping is readable
# and so a root that never had either column is untouched.
_LEGACY_COUNT_COLUMNS: tuple[str, ...] = ("n_tracks", "n_individuals")


@dataclass(frozen=True, slots=True)
class TrackerRunRowBase(RunIndexRowBase):
    """The run-index columns every integrated tracker carries.

    ``video_abs_path`` -- like every path column a subclass adds -- is stored the
    way ``abs_path`` is: dataset-root-relative when the file is inside the
    dataset, absolute when it is not, and resolved by
    :meth:`Dataset.resolve_path`. A stored absolute would never match a freshly
    resolved one after a move or a sync between machines, which is the comparison
    the reuse guard makes, so it would invert into a permanent recompute. A path
    column a subclass adds must also be declared in that root's ``path_columns``
    in :data:`~mosaic.core.pipeline.tracking_roots.TRACKING_ROOTS`, which is where
    the portability passes read it from.

    ``n_ids`` is how many distinct ``id`` values the produced tracks table holds.
    For a tracker that maintains identities it is **not** a count of animals: an
    identity lost across an occlusion and re-acquired is two ids, so the number is
    an upper bound on the population and a lower bound on how fragmented the
    tracking was.
    """

    group: str
    sequence: str
    video_abs_path: str
    params_hash: str
    n_ids: int = 0


def tracker_index_path(ds: Dataset, kind: str) -> Path:
    """Where tracker *kind* keeps its run index."""
    return ds.get_root(kind) / "index.csv"


def _adopt_for(row_cls: type[RowT]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build the ``adopt`` hook that carries an older index onto *row_cls*.

    Runs inside the write lock, so a read-only dataset and one that is never
    appended to are both left alone: an index is brought forward the first time
    something writes to it, not the first time something looks.
    """
    columns = [f.name for f in dataclasses.fields(row_cls)]

    def adopt(df: pd.DataFrame) -> pd.DataFrame:
        for legacy in _LEGACY_COUNT_COLUMNS:
            if legacy in df.columns and "n_ids" not in df.columns:
                df = df.rename(columns={legacy: "n_ids"})
                break
        return project_to_schema(df, columns)

    return adopt


def tracker_index(path: Path, row_cls: type[RowT]) -> IndexCSV[RowT]:
    """The run index for one tracker root, keyed one row per (run, entry).

    The ``adopt`` hook carries an index written before the count column was
    renamed: ``n_tracks`` or ``n_individuals`` becomes ``n_ids`` in memory inside
    the write lock, so the counts a user already has survive rather than reading
    back as zero.
    """
    return IndexCSV(
        path,
        row_cls,
        dedup_keys=["run_id", "group", "sequence"],
        adopt=_adopt_for(row_cls),
    )


def list_tracker_runs(ds: Dataset, kind: str, row_cls: type[RowT]) -> pd.DataFrame:
    """Every run recorded in one tracker's index, empty-but-typed when absent.

    Read through the typed index rather than ``pd.read_csv`` so a caller sees the
    current columns whatever the file on disk still spells them, which is what
    makes the count rename invisible to anything downstream.
    """
    columns: pd.Index = pd.Index([f.name for f in dataclasses.fields(row_cls)])
    if not ds.has_root(kind):
        return pd.DataFrame(columns=columns)
    path = tracker_index_path(ds, kind)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return _adopt_for(row_cls)(pd.read_csv(path))
