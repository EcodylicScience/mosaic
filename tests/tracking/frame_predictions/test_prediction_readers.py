"""The readers of predict-frames runs.

Each reader takes the file from the index row's ``abs_path``, answers empty when the
root or the index is absent, and creates nothing, on a dataset opened read-only too.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.annotations.scoring import (
    DEFAULT_PCK_ALPHAS,
    PoseScoreSummary,
    frame_table_columns,
)
from mosaic.tracking.frame_predictions.index import (
    FRAME_PREDICTIONS_COLUMNS,
    SCORE_RUN_COLUMNS,
)
from mosaic.tracking.frame_predictions.read import (
    list_prediction_runs,
    list_score_runs,
    read_entry_predictions,
    read_score_frames,
    read_score_summary,
)
from tests.helpers import (
    assert_tree_unchanged,
    entry_predictions,
    make_dataset,
    record_entry_predictions,
    record_score_run,
    tree_snapshot,
)

SCORE_ID = "score-predictions.0.1-0123456789"
UNFINISHED_SCORE_ID = "score-predictions.0.1-abcdef0123"
RUN_ID = "predict-frames.0.1-0123456789"
UNFINISHED_RUN_ID = "predict-frames.0.1-abcdef0123"
FRAMES_RUN_ID = "uniform-0123456789"
OTHER_FRAMES_RUN_ID = "kmeans-0123456789"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _record(
    ds: Dataset,
    *,
    run_id: str = RUN_ID,
    frames_run_id: str = FRAMES_RUN_ID,
    camera: str = "",
    finished: bool = True,
) -> Path:
    predictions = entry_predictions(
        frames_run_id=frames_run_id, group="G", sequence="S", camera=camera
    )
    return record_entry_predictions(ds, predictions, run_id=run_id, finished=finished)


def _reopened(ds: Dataset) -> Dataset:
    return open_dataset(ds.manifest_path, ensure_roots=False)


def _undeclared(base: Path) -> Dataset:
    ds = open_dataset(base, ensure_roots=False)
    _ = ds.roots.pop("frame_predictions")
    return ds


# --- list_prediction_runs ------------------------------------------------------


def test_every_row_is_listed_finished_or_not(dataset: Dataset) -> None:
    _ = _record(dataset, camera="cam1")
    _ = _record(dataset, camera="cam2")
    _ = _record(dataset, run_id=UNFINISHED_RUN_ID, finished=False)

    runs = list_prediction_runs(dataset).sort_values(["run_id", "camera"])

    assert list(runs.columns) == list(FRAME_PREDICTIONS_COLUMNS)
    assert list(zip(runs["run_id"], runs["camera"], strict=True)) == [
        (RUN_ID, "cam1"),
        (RUN_ID, "cam2"),
        (UNFINISHED_RUN_ID, ""),
    ]
    assert [bool(cell) for cell in runs["finished_at"]] == [True, True, False]


def test_the_rows_of_one_extraction_run_are_listed(dataset: Dataset) -> None:
    _ = _record(dataset)
    _ = _record(dataset, run_id=UNFINISHED_RUN_ID, frames_run_id=OTHER_FRAMES_RUN_ID)

    runs = list_prediction_runs(dataset, frames_run_id=OTHER_FRAMES_RUN_ID)

    assert runs["run_id"].tolist() == [UNFINISHED_RUN_ID]
    assert list_prediction_runs(dataset, frames_run_id="absent").empty


def test_a_dataset_without_an_index_lists_the_schema_alone(dataset: Dataset) -> None:
    runs = list_prediction_runs(dataset)

    assert runs.empty
    assert list(runs.columns) == list(FRAME_PREDICTIONS_COLUMNS)


def test_a_read_only_listing_creates_nothing(dataset: Dataset) -> None:
    _ = _record(dataset)
    reopened = _reopened(dataset)
    before = tree_snapshot(dataset.base_dir)

    runs = list_prediction_runs(reopened)

    assert runs["run_id"].tolist() == [RUN_ID]
    assert_tree_unchanged(dataset.base_dir, before)


def test_an_absent_root_lists_the_schema_alone(tmp_path: Path) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    runs = list_prediction_runs(open_dataset(base, ensure_roots=False))

    assert runs.empty
    assert list(runs.columns) == list(FRAME_PREDICTIONS_COLUMNS)
    assert_tree_unchanged(base, before)


def test_an_undeclared_root_lists_the_schema_alone(tmp_path: Path) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    runs = list_prediction_runs(_undeclared(base))

    assert runs.empty
    assert list(runs.columns) == list(FRAME_PREDICTIONS_COLUMNS)
    assert_tree_unchanged(base, before)


# --- read_entry_predictions ----------------------------------------------------


def test_a_finished_entry_camera_reads_back(dataset: Dataset) -> None:
    _ = _record(dataset, camera="cam1")
    _ = _record(dataset, camera="cam2")

    found = read_entry_predictions(dataset, RUN_ID, "G", "S", "cam2")

    assert found == entry_predictions(
        frames_run_id=FRAMES_RUN_ID, group="G", sequence="S", camera="cam2"
    )


def test_the_camera_defaults_to_the_single_camera_entry(dataset: Dataset) -> None:
    _ = _record(dataset)

    found = read_entry_predictions(dataset, RUN_ID, "G", "S")

    assert found is not None
    assert found.camera == ""


def test_the_file_is_the_one_the_row_names(dataset: Dataset) -> None:
    """The reader opens the row's ``abs_path`` and builds no path from the entry."""
    elsewhere = dataset.get_root("frame_predictions") / "moved" / "file.json"
    predictions = entry_predictions(
        frames_run_id=FRAMES_RUN_ID, group="G", sequence="S"
    )
    _ = record_entry_predictions(dataset, predictions, run_id=RUN_ID, path=elsewhere)

    assert read_entry_predictions(dataset, RUN_ID, "G", "S") == predictions


def test_no_row_reads_as_none(dataset: Dataset) -> None:
    _ = _record(dataset, camera="cam1")

    assert read_entry_predictions(dataset, RUN_ID, "G", "S", "cam2") is None
    assert read_entry_predictions(dataset, UNFINISHED_RUN_ID, "G", "S", "cam1") is None
    assert read_entry_predictions(dataset, RUN_ID, "G", "T", "cam1") is None


def test_an_unfinished_row_reads_as_none(dataset: Dataset) -> None:
    _ = _record(dataset, run_id=UNFINISHED_RUN_ID, finished=False)

    assert read_entry_predictions(dataset, UNFINISHED_RUN_ID, "G", "S") is None


def test_a_finished_row_without_its_file_names_the_run_and_path(
    dataset: Dataset,
) -> None:
    written = _record(dataset)
    written.unlink()

    with pytest.raises(FileNotFoundError, match=RUN_ID) as raised:
        _ = read_entry_predictions(dataset, RUN_ID, "G", "S")

    assert str(written) in str(raised.value)


def test_a_read_only_entry_read_creates_nothing(dataset: Dataset) -> None:
    _ = _record(dataset)
    reopened = _reopened(dataset)
    before = tree_snapshot(dataset.base_dir)

    found = read_entry_predictions(reopened, RUN_ID, "G", "S")

    assert found is not None
    assert_tree_unchanged(dataset.base_dir, before)


def test_an_absent_root_reads_as_none(tmp_path: Path) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    reopened = open_dataset(base, ensure_roots=False)

    assert read_entry_predictions(reopened, RUN_ID, "G", "S") is None
    assert_tree_unchanged(base, before)


def test_an_undeclared_root_reads_as_none(tmp_path: Path) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    assert read_entry_predictions(_undeclared(base), RUN_ID, "G", "S") is None
    assert_tree_unchanged(base, before)


# --- the score readers ---------------------------------------------------------

OTHER_RUN_ID = "predict-frames.0.1-fedcba9876"


def test_every_score_row_is_listed_finished_or_not(dataset: Dataset) -> None:
    _ = record_score_run(dataset, SCORE_ID)
    _ = record_score_run(dataset, UNFINISHED_SCORE_ID, finished=False)

    runs = list_score_runs(dataset).sort_values("run_id")

    assert list(runs.columns) == list(SCORE_RUN_COLUMNS)
    assert runs["run_id"].tolist() == [SCORE_ID, UNFINISHED_SCORE_ID]
    assert [bool(cell) for cell in runs["finished_at"]] == [True, False]


def test_score_runs_are_filtered_by_a_scored_prediction_run(dataset: Dataset) -> None:
    _ = record_score_run(dataset, SCORE_ID, prediction_runs=(RUN_ID,))
    _ = record_score_run(
        dataset, UNFINISHED_SCORE_ID, prediction_runs=(RUN_ID, OTHER_RUN_ID)
    )

    assert list_score_runs(dataset, prediction_run=RUN_ID)["run_id"].tolist() == [
        SCORE_ID,
        UNFINISHED_SCORE_ID,
    ]
    assert list_score_runs(dataset, prediction_run=OTHER_RUN_ID)["run_id"].tolist() == [
        UNFINISHED_SCORE_ID
    ]
    assert list_score_runs(dataset, prediction_run="absent").empty


def test_no_score_index_lists_the_schema_alone(dataset: Dataset) -> None:
    runs = list_score_runs(dataset)

    assert runs.empty
    assert list(runs.columns) == list(SCORE_RUN_COLUMNS)


def test_a_finished_score_summary_reads_back(dataset: Dataset) -> None:
    _ = record_score_run(dataset, SCORE_ID, prediction_runs=(RUN_ID,))

    summary = read_score_summary(dataset, SCORE_ID)

    assert isinstance(summary, PoseScoreSummary)
    assert summary.prediction_runs == [RUN_ID]


def test_no_finished_score_row_reads_a_summary_as_none(dataset: Dataset) -> None:
    _ = record_score_run(dataset, UNFINISHED_SCORE_ID, finished=False)

    assert read_score_summary(dataset, UNFINISHED_SCORE_ID) is None
    assert read_score_summary(dataset, SCORE_ID) is None


def test_a_finished_score_row_without_its_summary_raises(dataset: Dataset) -> None:
    run_root = record_score_run(dataset, SCORE_ID, outputs=("frames.parquet",))

    with pytest.raises(FileNotFoundError, match=SCORE_ID) as raised:
        _ = read_score_summary(dataset, SCORE_ID)

    assert str(run_root / "summary.json") in str(raised.value)


def test_a_finished_score_frames_table_reads_back(dataset: Dataset) -> None:
    """The table has an alpha column and rows that the default schema lacks."""
    columns = frame_table_columns((0.5,))
    written = pd.DataFrame(
        {
            name: pd.Series([1, 2] if kind == "int64" else [None, None], dtype=kind)
            for name, kind in columns.items()
        }
    )
    written["n_pck_0p5"] = pd.Series([3, 4], dtype="int64")
    _ = record_score_run(dataset, SCORE_ID, frames=written)

    frames = read_score_frames(dataset, SCORE_ID)

    assert "n_pck_0p5" in frames.columns
    pd.testing.assert_frame_equal(frames, written)


def test_no_finished_score_row_reads_the_default_frames_schema(
    dataset: Dataset,
) -> None:
    _ = record_score_run(dataset, UNFINISHED_SCORE_ID, finished=False)
    expected = frame_table_columns(DEFAULT_PCK_ALPHAS)

    for run_id in (UNFINISHED_SCORE_ID, SCORE_ID):
        frames = read_score_frames(dataset, run_id)

        assert frames.empty
        assert list(frames.columns) == list(expected)
        assert {name: str(kind) for name, kind in frames.dtypes.items()} == {
            name: str(kind) for name, kind in expected.items()
        }
    assert {"n_pck_0p05", "n_pck_0p1", "n_pck_0p2"} <= set(frames.columns)


def test_a_finished_score_row_without_its_frames_raises(dataset: Dataset) -> None:
    run_root = record_score_run(dataset, SCORE_ID, outputs=("summary.json",))

    with pytest.raises(FileNotFoundError, match=SCORE_ID) as raised:
        _ = read_score_frames(dataset, SCORE_ID)

    assert str(run_root / "frames.parquet") in str(raised.value)


def test_the_score_readers_create_nothing_on_a_read_only_dataset(
    dataset: Dataset,
) -> None:
    _ = record_score_run(dataset, SCORE_ID)
    reopened = _reopened(dataset)
    before = tree_snapshot(dataset.base_dir)

    assert list_score_runs(reopened)["run_id"].tolist() == [SCORE_ID]
    assert read_score_summary(reopened, SCORE_ID) is not None
    assert not read_score_frames(reopened, SCORE_ID).columns.empty

    assert_tree_unchanged(dataset.base_dir, before)


@pytest.mark.parametrize("declared", [False, True])
def test_the_score_readers_of_an_absent_or_undeclared_root_read_empty(
    tmp_path: Path, declared: bool
) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)
    ds = open_dataset(base, ensure_roots=False) if declared else _undeclared(base)

    runs = list_score_runs(ds)

    assert runs.empty
    assert list(runs.columns) == list(SCORE_RUN_COLUMNS)
    assert read_score_summary(ds, SCORE_ID) is None
    assert read_score_frames(ds, SCORE_ID).empty
    assert_tree_unchanged(base, before)
