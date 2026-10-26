"""Reads the predict-frames and score-predictions runs of a dataset.

Each reader writes nothing, takes no lock and works on a dataset opened
read-only. An undeclared or absent root reads as empty. A file is opened through
its index row's ``abs_path``, and no path is built from an entry name.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.annotations.scoring import (
    DEFAULT_PCK_ALPHAS,
    PoseScoreSummary,
    empty_frame_table,
)
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    read_pose_predictions,
)
from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.index_csv import index_records, project_to_schema
from mosaic.tracking.frame_predictions.index import (
    FRAME_PREDICTIONS_COLUMNS,
    FRAME_PREDICTIONS_ROOT,
    SCORE_RUN_COLUMNS,
    predict_frames_index,
    prediction_index_path,
    score_predictions_index,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mosaic.core.dataset import Dataset

__all__ = [
    "list_prediction_runs",
    "list_score_runs",
    "predictions_file",
    "read_entry_predictions",
    "read_score_frames",
    "read_score_summary",
]


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
        return read_pose_predictions(predictions_file(ds, record))
    return None


def predictions_file(ds: Dataset, record: Mapping[str, str]) -> Path:
    """Return the predictions file that a finished predict-frames row names.

    Args:
        ds: The dataset that contains the run.
        record: The row, as :func:`list_prediction_runs` lists it.

    Raises:
        FileNotFoundError: The file is not on disk. The message names the run,
            the entry camera and the path.
    """
    path = ds.resolve_path(record["abs_path"])
    if path.is_file():
        return path
    msg = (
        f"predict-frames run {record['run_id']!r} records entry "
        f"(group={record['group']!r}, sequence={record['sequence']!r}, "
        f"camera={record['camera']!r}) as finished in {path}, and that file is "
        "not on disk. `mosaic reindex --apply` drops a row whose file is gone."
    )
    raise FileNotFoundError(msg)


def list_score_runs(ds: Dataset, *, prediction_run: str | None = None) -> pd.DataFrame:
    """List the score-predictions index rows, one per run.

    Every row is listed, finished or not. A caller that wants finished work keeps
    the rows whose ``finished_at`` is not blank.

    Args:
        ds: The dataset that contains the runs.
        prediction_run: Keep only the runs that scored this predict-frames run.

    Returns:
        The rows, with every column of
        :class:`~mosaic.tracking.frame_predictions.index.ScoreRunRow`. Empty,
        with those columns, when the root is undeclared or absent or contains no
        score-predictions index.
    """
    kinds = root_subdirectories(ds, FRAME_PREDICTIONS_ROOT, require_index=True)
    if "score-predictions" not in kinds:
        return project_to_schema(pd.DataFrame(), SCORE_RUN_COLUMNS)
    index = score_predictions_index(prediction_index_path(ds, "score-predictions"))
    rows = project_to_schema(index.read(), SCORE_RUN_COLUMNS)
    if prediction_run is None:
        return rows
    kept = [
        prediction_run in json.loads(text) if text else False
        for text in rows["prediction_runs"]
    ]
    return rows.loc[kept].reset_index(drop=True)


def read_score_summary(ds: Dataset, run_id: str) -> PoseScoreSummary | None:
    """Read one score-predictions run's ``summary.json``.

    Args:
        ds: The dataset that contains the run.
        run_id: The score-predictions run.

    Returns:
        The summary, or ``None`` when the run has no finished row.

    Raises:
        FileNotFoundError: The finished row's run has no ``summary.json``.
        ValueError: The file is not a summary that this mosaic reads.
    """
    record = _finished_score_row(ds, run_id)
    if record is None:
        return None
    path = _score_output(ds, record, "summary.json")
    return PoseScoreSummary.model_validate_json(path.read_bytes())


def read_score_frames(ds: Dataset, run_id: str) -> pd.DataFrame:
    """Read one score-predictions run's ``frames.parquet``.

    Args:
        ds: The dataset that contains the run.
        run_id: The score-predictions run.

    Returns:
        The frames table. Without a finished row, an empty table with the
        columns that a score of the default ``pck_alphas`` writes, the only
        alphas known without a finished run.

    Raises:
        FileNotFoundError: The finished row's run has no ``frames.parquet``.
    """
    record = _finished_score_row(ds, run_id)
    if record is None:
        return empty_frame_table(DEFAULT_PCK_ALPHAS)
    return pd.read_parquet(_score_output(ds, record, "frames.parquet"))


def _finished_score_row(ds: Dataset, run_id: str) -> Mapping[str, str] | None:
    """Return the finished row of *run_id*, or ``None`` when it has none."""
    for record in index_records(list_score_runs(ds)):
        if record["run_id"] == run_id and record["finished_at"]:
            return record
    return None


def _score_output(ds: Dataset, record: Mapping[str, str], name: str) -> Path:
    """Return the file *name* in a finished score run's directory.

    Raises:
        FileNotFoundError: The file is not on disk. The message names the run and
            the path.
    """
    path = ds.resolve_path(record["abs_path"]) / name
    if path.is_file():
        return path
    msg = (
        f"score-predictions run {record['run_id']!r} is recorded as finished, "
        f"and {path} is not on disk. `mosaic reindex --apply` drops a row whose "
        "files are gone."
    )
    raise FileNotFoundError(msg)
