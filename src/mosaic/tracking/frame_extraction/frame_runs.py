"""The frames index, and what the inventory says about frame-extraction runs.

Each extraction method keeps one index under frames/<method>/index.csv, one
row per (run, group, sequence, camera). This module reads and writes it and
imports no tracker, trainer or extraction code, and loads only the media modules
that ``mosaic.core.pipeline.inventory`` loads, so an inventory reads frame runs
without loading what writes them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase, index_records
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    CameraEntry,
    Coverage,
    FrameRunRef,
    InventoryScope,
    classify,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.inventory._read import IndexReader

__all__ = [
    "FRAMES_INDEX_COLUMNS",
    "FramesIndexRow",
    "adopt_frames_columns",
    "frame_run_records",
    "frames_index",
    "frames_index_path",
    "frames_run_root",
    "list_frame_runs",
]


def frames_run_root(ds: Dataset, method: str, run_id: str) -> Path:
    return ds.get_root("frames") / method / run_id


def frames_index_path(ds: Dataset, method: str) -> Path:
    return ds.get_root("frames") / method / "index.csv"


@dataclass(frozen=True, slots=True)
class FramesIndexRow(RunIndexRowBase):
    """Typed row for the frames index CSV.

    ``video_uuids`` and ``media_composition`` are item 5.1's frames half: what
    this frame set was cut from, recorded and never hashed. The extraction
    identifier is frozen (see :func:`frames_run_id`), so this is the only place
    the answer can be written down.

    **Both, not either.** ``video_uuids`` names the individual files, which is
    what item 5.1's table asks for and what survives the sequence being
    rearranged around them. ``media_composition`` names the arrangement, and it
    is the one that matters here: ``_extract_frames_multi`` pools candidates
    across a camera's clips using *global* frame indices, so a frame set is per
    camera over an ordered clip list and a reorder genuinely changes what a
    stored index refers to. Item 5.1's "extracted frames are already per-video on
    disk" is true only of a single-clip camera.

    ``video_uuids`` is in ``video_order`` and is **never sorted** -- it is the
    arrangement, and sorting it would make two orderings record alike. Contrast
    ``consumed_source_roots`` on the tracks row, which is a set and is sorted.
    Empty means not establishable, never "no videos".
    """

    method: str
    group: str
    sequence: str
    camera: str
    video_abs_path: str
    params_hash: str
    n_frames_extracted: int = 0
    n_frames_requested: int = 0
    video_uuids: str = ""
    media_composition: str = ""


FRAMES_INDEX_COLUMNS: Final[list[str]] = [
    field.name for field in fields(FramesIndexRow)
]
"""The schema, in CSV order. Derived from the row so the two cannot drift."""


def adopt_frames_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a frames index read off disk up to the current schema, in memory.

    The same hook ``tracks_index`` and the feature index carry, added here with
    the columns above rather than after someone meets the failure. Without it the
    first append to a pre-Stage-5 index concatenates a frame missing two columns
    with one that has them, and every older row's ``n_frames_extracted`` is
    widened to float on the way through -- ``40`` reaching disk as ``40.0``,
    which is the trap ``index_csv`` documents and this repo has already paid for
    once.

    Must return a **schema-complete** frame: filling only the new columns would
    leave ``list_runs`` raising on ``finished_at``.
    """
    out = pd.DataFrame(index=df.index)
    for column in FRAMES_INDEX_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    return out.reset_index(drop=True)


def frames_index(path: Path) -> IndexCSV[FramesIndexRow]:
    return IndexCSV(
        path,
        FramesIndexRow,
        # camera is part of the identity: the cameras of one recording share a
        # (run_id, group, sequence), so without it a partial re-run of one camera
        # would dedup away the other camera's row.
        dedup_keys=["run_id", "group", "sequence", "camera"],
        adopt=adopt_frames_columns,
    )


def list_frame_runs(ds: Dataset, method: str | None = None) -> pd.DataFrame:
    """
    List all frame extraction runs tracked in the frames index.

    Parameters
    ----------
    method : str, optional
        Filter to a specific method ("uniform", "kmeans" or "list").
        If None, returns runs across all methods.

    Returns
    -------
    pd.DataFrame
        Index rows for matching extraction runs.
    """
    frames_root = ds.get_root("frames")
    if not frames_root.exists():
        return pd.DataFrame(columns=FRAMES_INDEX_COLUMNS)

    methods = [method] if method else root_subdirectories(ds, "frames")
    dfs: list[pd.DataFrame] = []
    for m in methods:
        idx_path = frames_root / m / "index.csv"
        if idx_path.exists():
            # Through the typed reader, not a bare read_csv: inference reads an
            # all-digit cell as int64 and an empty one as NaN, so a numeric-looking
            # ``video_uuids`` or ``params_hash`` would come back as a number and a
            # blank as the float NaN that ``adopt_frames_columns`` exists to keep
            # off this frame. The same defect item 0.2 fixed for four other
            # readers, in the one that was missed.
            dfs.append(frames_index(idx_path).read())
    if not dfs:
        return pd.DataFrame(columns=FRAMES_INDEX_COLUMNS)
    return pd.concat(dfs, ignore_index=True)


def frame_run_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[CameraEntry]]:
    """Every frame-extraction run, keyed by ``(group, sequence, camera)``.

    The camera is part of the key rather than a detail. The cameras of one
    recording share a ``(group, sequence)``, which is why this index dedups on
    the four-part key -- and without the camera here, a run that extracted one
    camera would read as covering the entry and the other would never be seen as
    missing.
    """
    records: list[ArtifactRecord[CameraEntry]] = []
    wanted = scope.selector.entry_pairs
    for method in root_subdirectories(ds, "frames"):
        index_path = frames_index_path(ds, method)
        reader.note(index_path)
        frame = reader.frame(index_path, lambda m=method: list_frame_runs(ds, m))
        if frame.empty or "run_id" not in frame.columns:
            continue
        rows_by_run: dict[str, set[CameraEntry]] = {}
        present_by_run: dict[str, set[CameraEntry]] = {}
        started: dict[str, str] = {}
        finished: dict[str, str] = {}
        for record in index_records(frame):
            run_id = record.get("run_id", "")
            key: CameraEntry = (
                record.get("group", ""),
                record.get("sequence", ""),
                record.get("camera", ""),
            )
            if wanted is not None and (key[0], key[1]) not in wanted:
                continue
            rows_by_run.setdefault(run_id, set()).add(key)
            started.setdefault(run_id, record.get("started_at", ""))
            if record.get("finished_at", ""):
                finished.setdefault(run_id, record.get("finished_at", ""))
            stored = record.get("abs_path", "")
            if stored and ds.resolve_path(stored).exists():
                present_by_run.setdefault(run_id, set()).add(key)
        for run_id in sorted(rows_by_run):
            rows = frozenset(rows_by_run[run_id])
            present = frozenset(present_by_run.get(run_id, set()))
            coverage = Coverage(target=rows, present=present)
            records.append(
                ArtifactRecord[CameraEntry](
                    ref=FrameRunRef(method=method, run_id=run_id),
                    name=method,
                    run_id=run_id,
                    coverage=coverage,
                    status=classify(
                        satisfied=coverage.is_satisfied,
                        any_covered=bool(coverage.covered),
                        orphan_rows=bool(rows - present),
                        orphan_files=False,
                        drifted=False,
                        finished=bool(finished.get(run_id, "")),
                    ),
                    index_path=index_path,
                    rows=rows,
                    orphan_rows=rows - present,
                    started_at=started.get(run_id, ""),
                    finished_at=finished.get(run_id, ""),
                )
            )
    return records
