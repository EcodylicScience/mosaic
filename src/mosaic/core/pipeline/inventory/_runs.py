"""Read an index's runs and each run's finish state from its rows.

Every run index records a run's rows with ``run_id``, ``started_at`` and
``finished_at``. The feature scan and the media-variant records read both facts
here, the same way. The record builders of the indexes that read like them live
here too: the row-keyed one of frame runs, frame predictions and frame
selections, and the run-keyed one of prediction scores, prepared datasets and
trained models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    ArtifactRef,
    Coverage,
    classify,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.pipeline.inventory._read import IndexReader
    from mosaic.core.pipeline.inventory.model import InventoryScope

__all__ = ["finish_state", "keyed_run_records", "run_ids", "run_keyed_record"]


def finish_state(frame: pd.DataFrame, run_id: str) -> tuple[str, str, bool]:
    """Return ``(started_at, finished_at, finished)`` for one run, from its rows.

    Each value is the first non-empty cell of its column. A run re-entered for more
    entries reads as finished once any of its rows recorded a finish.
    """
    started, finished = "", ""
    for record in index_records(frame):
        if str(record.get("run_id", "")) != run_id:
            continue
        started = started or str(record.get("started_at", ""))
        finished = finished or str(record.get("finished_at", ""))
    return started, finished, bool(finished)


def run_ids(frame: pd.DataFrame) -> list[str]:
    """Return every run identifier in an index, sorted, or an empty list."""
    if frame.empty or "run_id" not in frame.columns:
        return []
    return sorted({record.get("run_id", "") for record in index_records(frame)})


def keyed_run_records[KeyT](
    ds: Dataset,
    scope: InventoryScope,
    reader: IndexReader,
    *,
    root_key: str,
    kinds: Sequence[str] | None,
    read_index: Callable[[Dataset, str], pd.DataFrame],
    key_of: Callable[[dict[str, str]], KeyT],
    make_ref: Callable[[str, str, pd.DataFrame], ArtifactRef],
    drifted: Callable[[str, pd.DataFrame], tuple[Entry, ...]],
) -> list[ArtifactRecord[KeyT]]:
    """One record per run of every ``<root_key>/<kind>/index.csv``, keyed by row.

    A run's coverage target is the keys of its rows inside the scope, and a key is
    present when its row's ``abs_path`` exists. A row without its file is an
    orphan row. A key none of whose rows records a finish is an unfinished row.

    Args:
        ds: The dataset.
        scope: What was asked about. Its entry selector drops rows outside it
            before the drift check sees them.
        reader: The scan's reader, which notes each index and reads it once.
        root_key: The root whose child directories are the kinds.
        kinds: The kinds to read, or ``None`` for every child directory.
        read_index: Reads one kind's index when the reader has not.
        key_of: Maps an index record to its coverage key.
        make_ref: Builds a run's ref from its kind, run id and index rows.
        drifted: Receives a run id and that run's rows and returns the sorted
            entries whose rows drifted. The record's status and ``drift`` follow.

    Returns:
        The records, ordered by kind and then run id.
    """
    records: list[ArtifactRecord[KeyT]] = []
    wanted = scope.selector.entry_pairs
    names = root_subdirectories(ds, root_key) if kinds is None else list(kinds)
    for kind in names:
        index_path = ds.get_root(root_key) / kind / "index.csv"
        reader.note(index_path)
        frame = reader.frame(index_path, lambda k=kind: read_index(ds, k))
        if frame.empty or "run_id" not in frame.columns:
            continue
        rows_by_run: dict[str, set[KeyT]] = {}
        present_by_run: dict[str, set[KeyT]] = {}
        finished_by_run: dict[str, set[KeyT]] = {}
        positions: dict[str, list[int]] = {}
        started: dict[str, str] = {}
        finished: dict[str, str] = {}
        for position, record in enumerate(index_records(frame)):
            run_id = record.get("run_id", "")
            if (
                wanted is not None
                and (
                    record.get("group", ""),
                    record.get("sequence", ""),
                )
                not in wanted
            ):
                continue
            key = key_of(record)
            rows_by_run.setdefault(run_id, set()).add(key)
            positions.setdefault(run_id, []).append(position)
            started.setdefault(run_id, record.get("started_at", ""))
            if record.get("finished_at", ""):
                finished.setdefault(run_id, record.get("finished_at", ""))
                finished_by_run.setdefault(run_id, set()).add(key)
            stored = record.get("abs_path", "")
            if stored and ds.resolve_path(stored).exists():
                present_by_run.setdefault(run_id, set()).add(key)
        for run_id in sorted(rows_by_run):
            rows = frozenset(rows_by_run[run_id])
            present = frozenset(present_by_run.get(run_id, set()))
            run_rows = frame.iloc[positions[run_id]]
            drift = drifted(run_id, run_rows)
            coverage = Coverage(target=rows, present=present)
            records.append(
                ArtifactRecord[KeyT](
                    ref=make_ref(kind, run_id, run_rows),
                    name=kind,
                    run_id=run_id,
                    coverage=coverage,
                    status=classify(
                        satisfied=coverage.is_satisfied,
                        any_covered=bool(coverage.covered),
                        orphan_rows=bool(rows - present),
                        orphan_files=False,
                        drifted=bool(drift),
                        finished=bool(finished.get(run_id, "")),
                    ),
                    index_path=index_path,
                    rows=rows,
                    orphan_rows=rows - present,
                    unfinished_rows=rows - finished_by_run.get(run_id, set()),
                    drift=drift,
                    started_at=started.get(run_id, ""),
                    finished_at=finished.get(run_id, ""),
                )
            )
    return records


def run_keyed_record(
    *,
    ref: ArtifactRef,
    name: str,
    run_id: str,
    row: Mapping[str, str],
    recorded_finished: bool,
    complete: bool,
    index_path: Path,
    run_root: Path,
    upstreams: tuple[str, ...] = (),
) -> ArtifactRecord[str]:
    """The record of one run that is a single unit, keyed by its own run id.

    A trained model, a prepared dataset and a score run cover themselves. The run
    is present when it finished and its output is still there (*complete*), and a
    finished row without its output is an orphan row.

    Args:
        ref: The run's ref.
        name: The human-facing name, the run's kind.
        run_id: The run identifier.
        row: The run's first index row, read for its times.
        recorded_finished: Whether the row records the run as finished. The
            ``status`` of a trained-model or prepared-dataset row records it,
            and the ``finished_at`` of a score row.
        complete: Whether the run finished and its output exists.
        index_path: The index the row came from.
        run_root: The run's directory.
        upstreams: The run ids this run was built from.

    Returns:
        The run's record, covering itself.
    """
    return ArtifactRecord[str](
        ref=ref,
        name=name,
        run_id=run_id,
        coverage=Coverage(
            target=frozenset({run_id}),
            present=frozenset({run_id} if complete else ()),
        ),
        status=classify(
            satisfied=complete,
            any_covered=complete,
            orphan_rows=recorded_finished and not complete,
            orphan_files=False,
            drifted=False,
            finished=bool(row.get("finished_at", "")),
        ),
        run_root=run_root,
        index_path=index_path,
        rows=frozenset({run_id}),
        started_at=row.get("started_at", ""),
        finished_at=row.get("finished_at", ""),
        upstreams=upstreams,
    )
