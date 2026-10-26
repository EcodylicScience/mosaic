"""Reads the predict-frames runs of a dataset.

Each reader writes nothing, takes no lock and works on a dataset opened
read-only. An undeclared or absent root reads as empty. A file is opened through
its index row's ``abs_path``, and no path is built from an entry name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    read_pose_predictions,
)
from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.index_csv import index_records, project_to_schema
from mosaic.tracking.frame_predictions.index import (
    FRAME_PREDICTIONS_COLUMNS,
    FRAME_PREDICTIONS_ROOT,
    predict_frames_index,
    prediction_index_path,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["list_prediction_runs", "read_entry_predictions"]


def list_prediction_runs(
    ds: Dataset, *, frames_run_id: str | None = None
) -> pd.DataFrame:
    """List the predict-frames index rows, one per run and entry camera.

    Every row is listed, finished or not. A caller that wants finished work keeps
    the rows whose ``finished_at`` is not blank.

    Args:
        ds: The dataset that contains the runs.
        frames_run_id: Keep only the rows of this extraction run.

    Returns:
        The rows, with every column of
        :class:`~mosaic.tracking.frame_predictions.index.FramePredictionsRow`.
        Empty, with those columns, when the root is undeclared or absent or
        contains no predict-frames index.
    """
    kinds = root_subdirectories(ds, FRAME_PREDICTIONS_ROOT, require_index=True)
    if "predict-frames" not in kinds:
        return project_to_schema(pd.DataFrame(), FRAME_PREDICTIONS_COLUMNS)
    index = predict_frames_index(prediction_index_path(ds, "predict-frames"))
    rows = project_to_schema(index.read(), FRAME_PREDICTIONS_COLUMNS)
    if frames_run_id is None:
        return rows
    return rows.loc[rows["frames_run_id"] == frames_run_id].reset_index(drop=True)


def read_entry_predictions(
    ds: Dataset, run_id: str, group: str, sequence: str, camera: str = ""
) -> FramePredictions | None:
    """Read one entry camera's predictions from a predict-frames run.

    Args:
        ds: The dataset that contains the run.
        run_id: The predict-frames run.
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The entry's camera, ``""`` for single-camera media.

    Returns:
        The predictions in the file that the entry camera's finished row names,
        or ``None`` when the run has no finished row for the entry camera.

    Raises:
        FileNotFoundError: The finished row names a file that is not on disk.
        ValueError: The file is not a predictions file that this mosaic reads.
    """
    wanted = (run_id, group, sequence, camera)
    for record in index_records(list_prediction_runs(ds)):
        key = (record["run_id"], record["group"], record["sequence"], record["camera"])
        if key != wanted or not record["finished_at"]:
            continue
        path = ds.resolve_path(record["abs_path"])
        if not path.is_file():
            msg = (
                f"predict-frames run {run_id!r} records entry (group={group!r}, "
                f"sequence={sequence!r}, camera={camera!r}) as finished in {path}, "
                "and that file is not on disk. `mosaic reindex --apply` drops a row "
                "whose file is gone."
            )
            raise FileNotFoundError(msg)
        return read_pose_predictions(path)
    return None
