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

from mosaic.core.entry import Entry
from mosaic.core.pipeline.composition import composition_drift
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    Coverage,
    InventoryScope,
    TrackerRunRef,
    classify,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

if TYPE_CHECKING:
    from mosaic.core.pipeline.inventory._read import IndexReader
    from mosaic.core.dataset import Dataset

__all__ = [
    "TrackerRunRowBase",
    "drifted_media_entries",
    "list_tracker_runs",
    "tracker_index",
    "tracker_index_path",
    "tracker_kinds",
    "tracker_run_records",
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
    consumed_media_composition: str = ""
    """What this entry's media was when the run read it. Compared, never hashed.

    A tracker's identity is its settings, with no term for the media it read --
    which file it opens is decided at run time from the routing verdict on the
    media index. So re-transcoding an entry and re-linking the derivative leaves
    every tracker run's identifier exactly where it was, and a reader asking
    "is this run current" got yes over pixels from a different encode.

    Recorded rather than hashed, the rule ``consumed_tracks_composition``
    already follows: one tracker identity covers many entries, so folding a
    per-entry value into the digest would rename a whole directory because one
    other entry's source moved. Empty means not establishable -- a run written
    before this column, or a dataset with no ``media_raw`` -- and never "no
    media", which is how every other composition cell here reads.
    """
    media: str = ""
    """The media variant that this entry was read from, or ``""`` for the entry media.

    The tool's output under the working directory is in the variant's pixels and
    frames, and only its published table is mapped back into source space. This
    cell records that for a run whose output is read again later.
    """


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


# --- What the inventory says about tracker runs -----------------------------


def tracker_kinds() -> tuple[str, ...]:
    """Every tracker root, sorted: the roots whose retention class is ``tracker``.

    Read from :data:`~mosaic.core.pipeline.tracking_roots.TRACKING_ROOTS`, which
    declares a row for every tracker. A reader of every tracker's index therefore
    imports none of the trackers. Each index contains rows of a
    :class:`TrackerRunRowBase` subclass, and these readers project the rows onto
    the shared columns.
    """
    return tuple(
        sorted(
            key for key, root in TRACKING_ROOTS.items() if root.retention == "tracker"
        )
    )


def drifted_media_entries(
    ds: Dataset, kind: str, run_id: str
) -> tuple[tuple[str, str], ...]:
    """Entries whose media moved under a recorded tracker run.

    **Compared, not hashed**, which is what makes it useful at all: a tracker's
    identity is its settings, with no term for the media it read. Re-transcode
    an entry, re-link the derivative, and the run identifier does not move -- so
    a reuse gate keyed on identity alone serves the old run over pixels from a
    different encode, and reports the work done.

    The comparison uses :func:`~mosaic.core.pipeline.composition.composition_drift`,
    under which a blank cell on either side is not drift.
    """
    from mosaic.core.pipeline.sequence_index import media_compositions_for

    if kind not in tracker_kinds() or not ds.has_root(kind):
        return ()
    frame = list_tracker_runs(ds, kind, TrackerRunRowBase)
    if frame.empty or "consumed_media_composition" not in frame.columns:
        return ()
    recorded = {
        (record.get("group", ""), record.get("sequence", "")): record.get(
            "consumed_media_composition", ""
        )
        for record in index_records(frame)
        if record.get("run_id", "") == run_id
    }
    if not recorded:
        return ()
    return composition_drift(recorded, media_compositions_for(ds, recorded))


def tracker_run_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[Entry]]:
    """Every recorded tracker and inference run, as inventory records.

    A run covers the entries its own rows name -- not the dataset's universe --
    because a tracker is legitimately pointed at a subset, and measuring one
    against everything would report a finished run as short. What makes an entry
    *covered* is that its recorded working directory is still there: the sweeper
    reclaims those, so a swept run is honestly no longer holding its outputs.
    """
    records: list[ArtifactRecord[Entry]] = []
    for kind in tracker_kinds():
        if not ds.has_root(kind):
            continue
        index_path = tracker_index_path(ds, kind)
        reader.note(index_path)
        frame = reader.frame(
            index_path, lambda k=kind: list_tracker_runs(ds, k, TrackerRunRowBase)
        )
        if frame.empty or "run_id" not in frame.columns:
            continue
        rows_by_run: dict[str, set[Entry]] = {}
        present_by_run: dict[str, set[Entry]] = {}
        finished_by_run: dict[str, str] = {}
        started_by_run: dict[str, str] = {}
        for record in index_records(frame):
            run_id = record.get("run_id", "")
            entry = (record.get("group", ""), record.get("sequence", ""))
            rows_by_run.setdefault(run_id, set()).add(entry)
            started_by_run.setdefault(run_id, record.get("started_at", ""))
            if record.get("finished_at", ""):
                finished_by_run.setdefault(run_id, record.get("finished_at", ""))
            stored = record.get("abs_path", "")
            if stored and ds.resolve_path(stored).exists():
                present_by_run.setdefault(run_id, set()).add(entry)
        for run_id in sorted(rows_by_run):
            rows = frozenset(rows_by_run[run_id])
            wanted = scope.selector.entry_pairs
            target = frozenset(rows if wanted is None else rows & wanted)
            present = frozenset(present_by_run.get(run_id, set()))
            coverage = Coverage(target=target, present=present)
            finished = finished_by_run.get(run_id, "")
            drift = drifted_media_entries(ds, kind, run_id)
            records.append(
                ArtifactRecord[Entry](
                    ref=TrackerRunRef(root_key=kind, run_id=run_id),
                    name=kind,
                    run_id=run_id,
                    coverage=coverage,
                    status=classify(
                        satisfied=coverage.is_satisfied,
                        any_covered=bool(coverage.covered),
                        orphan_rows=bool(rows - present),
                        orphan_files=False,
                        drifted=bool(drift),
                        finished=bool(finished),
                    ),
                    index_path=index_path,
                    rows=rows,
                    orphan_rows=rows - present,
                    drift=drift,
                    started_at=started_by_run.get(run_id, ""),
                    finished_at=finished,
                )
            )
    return records
