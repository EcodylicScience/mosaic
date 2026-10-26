"""The frame-predictions indexes: their rows, their dedup and their reconcile opener.

The frame_predictions root contains one index per kind directory, and the opener that
reconcile registers for the root reads the kind from that directory's name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.dataset_indexes import reconcilable_index
from mosaic.tracking.frame_predictions.index import (
    FRAME_PREDICTIONS_COLUMNS,
    SCORE_RUN_COLUMNS,
    FramePredictionsRow,
    ScoreRunRow,
    frame_predictions_index,
    predict_frames_index,
    prediction_index_path,
    score_predictions_index,
)
from tests.helpers import entry_predictions, make_dataset, record_entry_predictions

RUN_ID = "predict-frames.0.1-0123456789"
OTHER_RUN_ID = "predict-frames.0.1-abcdef0123"
SCORE_RUN_ID = "score-predictions.0.1-0123456789"
FRAMES_RUN_ID = "uniform-0123456789"
MODEL = "train-pose.0.1-abcdef0123"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _row(*, camera: str, n_frames: int, run_id: str = RUN_ID) -> FramePredictionsRow:
    return FramePredictionsRow(
        run_id=run_id,
        abs_path=Path(f"frame_predictions/predict-frames/{run_id}/G__S/{camera}"),
        group="G",
        sequence="S",
        camera=camera,
        frames_method="uniform",
        frames_run_id=FRAMES_RUN_ID,
        model=json.dumps([MODEL]),
        model_kind="train-pose",
        model_id=MODEL,
        n_frames=n_frames,
        n_objects=2 * n_frames,
        consumed_frames_composition="",
        tool_version="8.3.0",
        params_hash="h",
    )


def _score_row(*, n_scored: int, run_id: str = SCORE_RUN_ID) -> ScoreRunRow:
    return ScoreRunRow(
        run_id=run_id,
        abs_path=Path(f"frame_predictions/score-predictions/{run_id}"),
        prediction_runs=json.dumps([RUN_ID]),
        model=json.dumps([MODEL]),
        model_kind="train-pose",
        model_id=MODEL,
        consumed_sets="[]",
        pose_id="3",
        n_scored=n_scored,
        n_excluded_training=0,
        n_not_covered=1,
        training_frames_known=True,
        params_hash="h",
    )


def test_a_predictions_row_has_the_documented_columns() -> None:
    assert FRAME_PREDICTIONS_COLUMNS == (
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "group",
        "sequence",
        "camera",
        "frames_method",
        "frames_run_id",
        "model",
        "model_kind",
        "model_id",
        "n_frames",
        "n_objects",
        "consumed_frames_composition",
        "tool_version",
        "params_hash",
    )


def test_a_score_row_has_the_documented_columns() -> None:
    assert SCORE_RUN_COLUMNS == (
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "prediction_runs",
        "model",
        "model_kind",
        "model_id",
        "consumed_sets",
        "pose_id",
        "n_scored",
        "n_excluded_training",
        "n_not_covered",
        "training_frames_known",
        "params_hash",
    )


def test_each_index_is_in_its_kind_directory(dataset: Dataset) -> None:
    root = dataset.get_root("frame_predictions")

    assert root == dataset.base_dir / "frame_predictions"
    assert prediction_index_path(dataset, "predict-frames") == (
        root / "predict-frames" / "index.csv"
    )
    assert prediction_index_path(dataset, "score-predictions") == (
        root / "score-predictions" / "index.csv"
    )


@pytest.mark.parametrize(
    ("kind", "row_cls"),
    [("predict-frames", FramePredictionsRow), ("score-predictions", ScoreRunRow)],
)
def test_the_opener_reads_a_kind_directory_with_its_row(
    tmp_path: Path, kind: str, row_cls: type[FramePredictionsRow | ScoreRunRow]
) -> None:
    index = frame_predictions_index(tmp_path / "frame_predictions" / kind / "index.csv")

    assert index.row_cls is row_cls


def test_the_opener_refuses_an_index_in_any_other_directory(tmp_path: Path) -> None:
    stray = tmp_path / "frame_predictions" / "infer-pose"
    stray.mkdir(parents=True)
    _ = (stray / "index.csv").write_text("run_id\n")

    with pytest.raises(ValueError, match=r"infer-pose") as raised:
        _ = frame_predictions_index(stray / "index.csv")

    assert str(stray) in str(raised.value)


def test_the_root_registers_the_opener_for_reconcile() -> None:
    assert reconcilable_index("frame_predictions") is frame_predictions_index


def test_a_predictions_row_replaces_the_row_of_its_run_and_entry_camera(
    dataset: Dataset,
) -> None:
    index = predict_frames_index(prediction_index_path(dataset, "predict-frames"))
    index.append([_row(camera="cam1", n_frames=3), _row(camera="cam2", n_frames=4)])
    index.append([_row(camera="cam1", n_frames=5)])
    index.append([_row(camera="cam1", n_frames=6, run_id=OTHER_RUN_ID)])

    rows = index.read().sort_values(["run_id", "camera"])

    assert list(zip(rows["run_id"], rows["camera"], rows["n_frames"], strict=True)) == [
        (RUN_ID, "cam1", 5),
        (RUN_ID, "cam2", 4),
        (OTHER_RUN_ID, "cam1", 6),
    ]


def test_a_score_row_replaces_the_row_of_its_run(dataset: Dataset) -> None:
    index = score_predictions_index(prediction_index_path(dataset, "score-predictions"))
    index.append([_score_row(n_scored=3)])
    index.append([_score_row(n_scored=7)])
    index.append([_score_row(n_scored=2, run_id="score-predictions.0.1-abcdef0123")])

    rows = index.read()

    assert sorted(zip(rows["run_id"], rows["n_scored"], strict=True)) == [
        (SCORE_RUN_ID, 7),
        ("score-predictions.0.1-abcdef0123", 2),
    ]
    assert rows["training_frames_known"].tolist() == [True, True]


def test_an_append_projects_an_older_index_onto_the_schema(dataset: Dataset) -> None:
    path = prediction_index_path(dataset, "predict-frames")
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "abs_path": "frame_predictions/predict-frames/old/S",
                "run_id": OTHER_RUN_ID,
                "group": "",
                "sequence": "S",
                "camera": "",
                "retired": "x",
            }
        ]
    ).to_csv(path, index=False)

    predict_frames_index(path).append([_row(camera="cam1", n_frames=1)])

    written = pd.read_csv(path, keep_default_na=False)
    assert list(written.columns) == list(FRAME_PREDICTIONS_COLUMNS)
    assert written["run_id"].tolist() == [OTHER_RUN_ID, RUN_ID]
    assert written["tool_version"].tolist() == ["", "8.3.0"]


def test_reindex_drops_the_row_of_a_missing_predictions_file(
    dataset: Dataset,
) -> None:
    kept = record_entry_predictions(
        dataset,
        entry_predictions(
            frames_run_id=FRAMES_RUN_ID, group="G", sequence="S", camera="cam1"
        ),
        run_id=RUN_ID,
    )
    gone = record_entry_predictions(
        dataset,
        entry_predictions(
            frames_run_id=FRAMES_RUN_ID, group="G", sequence="S", camera="cam2"
        ),
        run_id=RUN_ID,
    )
    gone.unlink()
    path = prediction_index_path(dataset, "predict-frames")

    dropped = dataset.reindex("frame_predictions", dry_run=False)

    assert dropped == {str(path): 1}
    assert predict_frames_index(path).read()["abs_path"].tolist() == [
        dataset.relative_to_root(kept)
    ]
