"""Defines the selections index, its inventory records and the frames it lists.

``selections/select-frames/index.csv`` has one :class:`FrameSelectionRow` per run
and entry, written under ``index_lock`` through
:class:`~mosaic.core.pipeline.index_csv.IndexCSV`. Each row names the entry's
:data:`SELECTION_FILE_NAME`. Reconcile opens the index through
:func:`frame_selection_index`, the inventory reads it through
:func:`frame_selection_records`, and :func:`selected_frames` returns a run's
frames as the ``frames`` argument of a ``list`` extraction. A run's
:data:`EXTRACT_PARAMS_FILE_NAME` contains the same frames as the params of that
extraction.

The selections have a separate root, because the inventory reads every child of
the frames root as an extraction method. This module leaves the extraction, tool
and trainer code unimported.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from mosaic.core.pipeline.dataset_indexes import (
    register_reconcilable_index,
    root_subdirectories,
)
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    finished_row_file,
    finished_rows,
    index_records,
    project_to_schema,
    row_file_exists,
)
from mosaic.core.pipeline.inventory._runs import keyed_run_records
from mosaic.core.pipeline.inventory.model import ArtifactRecord, FrameSelectionRef

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.pipeline.inventory._read import IndexReader
    from mosaic.core.pipeline.inventory.model import InventoryScope

__all__ = [
    "EXTRACT_PARAMS_FILE_NAME",
    "FRAME_SELECTION_COLUMNS",
    "SELECTIONS_ROOT",
    "SELECTION_FILE_NAME",
    "SELECTION_KEY_COLUMNS",
    "SELECT_FRAMES_KIND",
    "FrameSelectionRow",
    "frame_selection_index",
    "frame_selection_records",
    "run_selected_frames",
    "selected_frames",
    "selection_index_path",
    "selections_kind_root",
]

SELECTIONS_ROOT: Final = "selections"
"""The root key, which the default manifest also uses as the directory name."""

SELECT_FRAMES_KIND: Final = "select-frames"
"""The kind directory of the root, named for the op that writes its runs."""

SELECTION_FILE_NAME: Final = "selection.parquet"
"""What an entry's selection is called, in the entry's directory of its run."""

EXTRACT_PARAMS_FILE_NAME: Final = "extract-frames.params.json"
"""The params of the ``list`` extraction of a run's frames, in the run's directory."""

SELECTION_KEY_COLUMNS: Final = ("group", "sequence")
"""The columns that key a row of the selections index within its run."""


@dataclass(frozen=True, slots=True)
class FrameSelectionRow(RunIndexRowBase):
    """Records one entry that a select-frames run selected frames from.

    ``abs_path`` names the entry's :data:`SELECTION_FILE_NAME`, dataset-relative.
    No other column is a path.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        source_tracks_run_id: The tracks variant that the run read.
        producer: The ``producer`` that the variant's tracks-index row records.
        n_candidates: The candidate frames that the entry's events offered.
        n_selected: The frames chosen from them.
        skipped_criteria: The criteria left out for the producer, a JSON list.
        body_length_px: The body length used, in pixels, ``""`` when unknown.
        params_hash: ``hash_params`` of the run's identity payload.
    """

    group: str
    sequence: str
    source_tracks_run_id: str
    producer: str
    n_candidates: int
    n_selected: int
    skipped_criteria: str
    body_length_px: str
    params_hash: str


FRAME_SELECTION_COLUMNS: Final = tuple(
    field.name for field in fields(FrameSelectionRow)
)
"""The select-frames index schema, in CSV order."""


def selections_kind_root(ds: Dataset) -> Path:
    """Return the directory that contains the select-frames runs and their index."""
    return ds.get_root(SELECTIONS_ROOT) / SELECT_FRAMES_KIND


def selection_index_path(ds: Dataset) -> Path:
    """Return the index of the select-frames runs, which may not exist yet."""
    return selections_kind_root(ds) / "index.csv"


def frame_selection_index(path: Path) -> IndexCSV[FrameSelectionRow]:
    """Open a select-frames index, one row per run and entry.

    Reconcile opens every index of the selections root through this function.

    Raises:
        ValueError: *path* is in a directory other than the
            :data:`SELECT_FRAMES_KIND` directory.
    """
    if path.parent.name != SELECT_FRAMES_KIND:
        msg = (
            f"{path} is in the directory {path.parent.name!r}, and the "
            f"{SELECTIONS_ROOT} root has one index, in its {SELECT_FRAMES_KIND} "
            "directory."
        )
        raise ValueError(msg)
    return IndexCSV(
        path,
        FrameSelectionRow,
        dedup_keys=["run_id", *SELECTION_KEY_COLUMNS],
        adopt=partial(project_to_schema, columns=FRAME_SELECTION_COLUMNS),
    )


def frame_selection_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[Entry]]:
    """Return every select-frames run, keyed by ``(group, sequence)``.

    A run is covered by the entries of its rows. A row whose file is gone is an
    orphan row. A run does not drift, because its rows do not record a media
    composition.

    Args:
        ds: The dataset, which is read and not written.
        scope: What was asked about. Its entry selector drops rows outside it.
        reader: The scan's reader, which reads each index once.

    Returns:
        The records, ordered by run id. Empty when the root is undeclared or
        absent.
    """
    kinds = (
        [SELECT_FRAMES_KIND]
        if SELECT_FRAMES_KIND in root_subdirectories(ds, SELECTIONS_ROOT)
        else []
    )
    return keyed_run_records(
        ds,
        scope,
        reader,
        root_key=SELECTIONS_ROOT,
        kinds=kinds,
        read_index=_read_selection_index,
        key_of=lambda record: (record.get("group", ""), record.get("sequence", "")),
        make_ref=lambda _kind, run_id, rows: FrameSelectionRef(
            run_id=run_id,
            source_tracks_run_id=index_records(rows)[0].get("source_tracks_run_id", ""),
        ),
        drifted=lambda _run_id, _rows: (),
    )


def _read_selection_index(ds: Dataset, _kind: str) -> pd.DataFrame:
    """Read the select-frames index, which raises when it is absent."""
    return frame_selection_index(selection_index_path(ds)).read()


def selected_frames(ds: Dataset, run_id: str) -> dict[Entry, tuple[int, ...]]:
    """Return the frames that a select-frames run chose, by entry.

    Reads the file of each finished row of *run_id*. The result is the
    ``frames`` argument of ``extract_frames(ds, method="list", frames=...)``
    unchanged. The read does not create or write a file, and it does not take a
    lock.

    Args:
        ds: The dataset that contains the run.
        run_id: The select-frames run.

    Returns:
        Each entry's frames, sorted and distinct, with the entries in sorted
        order. An entry is left out when its selection chose zero frames,
        because a ``list`` extraction refuses an entry without frames. Empty
        when the root is undeclared or absent, or when the run does not have a
        finished row.

    Raises:
        FileNotFoundError: A finished row names a file that is not on disk.
    """
    if SELECT_FRAMES_KIND not in root_subdirectories(
        ds, SELECTIONS_ROOT, require_index=True
    ):
        return {}
    index = frame_selection_index(selection_index_path(ds))
    return run_selected_frames(ds, index.read(), run_id, skip_missing=False)


def run_selected_frames(
    ds: Dataset, rows: pd.DataFrame, run_id: str, *, skip_missing: bool
) -> dict[Entry, tuple[int, ...]]:
    """Return the frames that the finished rows of *run_id* in *rows* name, by entry.

    Args:
        ds: The dataset that contains the run.
        rows: The select-frames index, as it reads.
        run_id: The select-frames run.
        skip_missing: Leave out a finished row whose file is not on disk,
            instead of raising.

    Returns:
        Each entry's frames, sorted and distinct, with the entries in sorted
        order. An entry whose selection chose zero frames is left out.

    Raises:
        FileNotFoundError: A finished row names a file that is not on disk,
            and *skip_missing* is false.
    """
    records = index_records(project_to_schema(rows, FRAME_SELECTION_COLUMNS))
    finished = finished_rows(records, run_id, SELECTION_KEY_COLUMNS)
    chosen: dict[Entry, tuple[int, ...]] = {}
    for key in sorted(finished):
        record = finished[key]
        if skip_missing and not row_file_exists(ds, record):
            continue
        path = finished_row_file(
            ds, record, run_kind=SELECT_FRAMES_KIND, key_columns=SELECTION_KEY_COLUMNS
        )
        table = pd.read_parquet(path, columns=["frame"])
        frames = np.unique(table["frame"].to_numpy(dtype=np.int64))
        if frames.size:
            chosen[record["group"], record["sequence"]] = tuple(
                int(frame) for frame in frames
            )
    return chosen


register_reconcilable_index(SELECTIONS_ROOT, frame_selection_index)
