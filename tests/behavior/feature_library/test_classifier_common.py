"""What the global classifiers share: the output, the standardizer, the folds and the bundle.

``xgboost``, ``lightning-action`` and ``sequence-classifier`` assemble their output,
standardize, deal their cross-validation folds and write their bundle through this
module. Their own suites cover the models. These tests cover the shared parts on
their own, so that a classifier built on them inherits the guarantees.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.classifier_common import (
    FitReporting,
    RunningMoments,
    assemble_prediction,
    check_cross_validation,
    check_folds,
    deal_folds,
    fold_map,
    fold_rows,
    fold_table,
    predicted_labels,
    standardize,
    write_bundle,
)
from mosaic.behavior.feature_library.helpers import iter_series_positions
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.progress import NullProgressCallback

# --- Prediction ---

CLASSES = [0, 2, 5]


def _probs(*rows: list[float]) -> npt.NDArray[np.float64]:
    return np.array(rows, dtype=np.float64)


def test_without_a_threshold_a_row_takes_its_highest_probability() -> None:
    probs = _probs([0.1, 0.7, 0.2], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8])

    labels = predicted_labels(probs, CLASSES, None, default_class=0)

    assert labels.tolist() == [2, 0, 5]


def test_a_row_with_a_nan_probability_takes_the_default_class_in_both_modes() -> None:
    probs = _probs([0.1, 0.7, 0.2], [np.nan, np.nan, np.nan], [0.1, np.nan, 0.8])

    assert predicted_labels(probs, CLASSES, None, default_class=5).tolist() == [2, 5, 5]
    assert predicted_labels(probs, CLASSES, 0.5, default_class=5).tolist() == [2, 5, 5]


def test_a_threshold_zeroes_a_probability_under_it_and_defaults_an_empty_row() -> None:
    probs = _probs([0.3, 0.4, 0.3], [0.2, 0.9, 0.1], [0.6, 0.2, 0.2])

    labels = predicted_labels(probs, CLASSES, 0.5, default_class=5)

    assert labels.tolist() == [5, 2, 0]


def test_a_threshold_per_class_reads_a_class_it_omits_as_zero() -> None:
    probs = _probs([0.3, 0.4, 0.3], [0.2, 0.1, 0.1])

    # Class 0 needs 0.35 and class 2 needs 0.5. Class 5 is not listed, so it has no bar.
    labels = predicted_labels(probs, CLASSES, {0: 0.35, 2: 0.5}, default_class=9)

    assert labels.tolist() == [5, 5]


def test_the_prediction_carries_the_identity_of_the_rows_it_describes() -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 1, 0, 1],
            "id1": [3, 3, 3, 3],
            "id2": [4, 4, 4, 4],
            "perspective": [0, 0, 1, 1],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    probs = _probs([0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1])

    result = assemble_prediction(df, probs, CLASSES, None, default_class=0)

    assert list(result.columns) == [
        "frame",
        "id1",
        "id2",
        "perspective",
        "prob_0",
        "prob_2",
        "prob_5",
        "predicted_label",
    ]
    assert result["perspective"].tolist() == [0, 0, 1, 1]
    assert result["predicted_label"].tolist() == [0, 2, 5, 0]
    assert result["prob_2"].tolist() == [0.05, 0.8, 0.2, 0.2]


# --- Parameters ---


def test_cross_validation_needs_folds_for_a_seed_and_templates_for_folds() -> None:
    check_cross_validation(None, 0, model_given=False)
    check_cross_validation(None, 0, model_given=True)
    check_cross_validation(5, 3, model_given=False)

    with pytest.raises(ValueError, match="cv_seed.*cv_folds"):
        check_cross_validation(None, 3, model_given=False)
    with pytest.raises(ValueError, match="cv_folds.*templates.*model"):
        check_cross_validation(5, 0, model_given=True)


# --- Standardization ---


def test_moments_added_in_pieces_equal_the_moments_of_the_whole() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(loc=3.0, scale=2.0, size=(500, 4))
    matrix[::7, 1] = np.nan
    moments = RunningMoments(4)

    for piece in np.array_split(matrix, [37, 190, 330]):
        moments.add(piece)
    mean, std = moments.standardizer()

    np.testing.assert_allclose(mean, np.nanmean(matrix, axis=0))
    np.testing.assert_allclose(std, np.nanstd(matrix, axis=0))


def test_a_column_without_values_or_spread_standardizes_without_dividing_by_zero() -> (
    None
):
    matrix = np.column_stack([np.full(10, np.nan), np.full(10, 4.0), np.arange(10.0)])
    moments = RunningMoments(3)
    moments.add(matrix)

    mean, std = moments.standardizer()

    assert mean[0] == 0.0 and std[0] == 1.0
    assert mean[1] == 4.0 and std[1] == 1.0
    standardized = standardize(matrix, mean, std)
    assert np.isfinite(standardized).all()
    assert standardized[:, 0].tolist() == [0.0] * 10


def test_a_missing_value_standardizes_to_the_mean_and_a_clip_bounds_the_rest() -> None:
    mean = np.array([10.0, 0.0])
    std = np.array([2.0, 1.0])
    matrix = np.array([[np.nan, 50.0], [12.0, -50.0], [10.0, 0.5]])

    plain = standardize(matrix, mean, std)
    clipped = standardize(matrix, mean, std, clip=10.0)

    assert plain[0, 0] == 0.0
    assert plain[:, 1].tolist() == [50.0, -50.0, 0.5]
    assert clipped[:, 1].tolist() == [10.0, -10.0, 0.5]
    assert clipped[1, 0] == 1.0


# --- Folds ---

TRAIN = [("g", f"s{i}") for i in range(6)]


def test_dealing_names_the_classifier_in_a_refusal() -> None:
    assert sorted(deal_folds("clf", TRAIN, 3, 0).values()) == [0, 0, 1, 1, 2, 2]

    with pytest.raises(ValueError, match=r"^\[clf\] .*6 train entries into 7"):
        deal_folds("clf", TRAIN, 7, 0)


def test_the_fold_of_each_entry_survives_a_bundle_round_trip() -> None:
    folds = deal_folds("clf", TRAIN, 3, 4)

    rows = fold_rows(folds)

    assert rows == sorted(rows)
    assert fold_map(rows) == folds


def _counts(per_entry: dict[tuple[str, str], dict[int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(per_entry, orient="index").fillna(0).astype(int)
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["group", "sequence"])
    return frame


def test_the_fold_table_lists_every_dealt_entry_with_its_rows_per_class() -> None:
    fold_of = {("g", "a"): 0, ("g", "b"): 1, ("g", "c"): 0}
    counts = _counts({("g", "a"): {0: 5, 2: 1}, ("g", "b"): {0: 4}})

    table = fold_table(fold_of, counts, [0, 2, 5])

    assert list(table.columns) == [
        "group",
        "sequence",
        "fold",
        "n_rows",
        "rows_0",
        "rows_2",
        "rows_5",
    ]
    assert table["sequence"].tolist() == ["a", "b", "c"]
    assert table["fold"].tolist() == [0, 1, 0]
    assert table["n_rows"].tolist() == [6, 4, 0]
    assert table["rows_5"].tolist() == [0, 0, 0]
    assert all(table[name].dtype == np.int64 for name in table.columns[2:])


def _four_entry_table(rows_in_fold_one: dict[int, int]) -> pd.DataFrame:
    fold_of = {("g", "a"): 0, ("g", "b"): 0, ("g", "c"): 1, ("g", "d"): 1}
    counts = _counts(
        {
            ("g", "a"): {0: 10, 1: 10},
            ("g", "b"): {0: 10, 1: 10},
            ("g", "c"): rows_in_fold_one,
            ("g", "d"): rows_in_fold_one,
        }
    )
    return fold_table(fold_of, counts, [0, 1])


def test_a_fold_whose_training_rows_hold_one_class_is_refused() -> None:
    table = _four_entry_table({0: 10})

    with pytest.raises(
        ValueError, match=r"^\[clf\] cv_folds=2: .*fold 0 hold 1 of the 2 classes"
    ):
        check_folds("clf", 2, table, [0, 1], rows="template rows")


def test_a_check_of_the_classifiers_own_runs_after_the_class_count_of_each_fold() -> (
    None
):
    table = _four_entry_table({0: 10, 1: 10})
    seen: list[tuple[int, list[int], list[int]]] = []

    def record(
        fold: int, present: npt.NDArray[np.intp], counts: npt.NDArray[np.intp]
    ) -> None:
        seen.append((fold, present.tolist(), counts.tolist()))

    check_folds("clf", 2, table, [0, 1], rows="template rows", per_fold=record)

    assert seen == [(0, [0, 1], [20, 20]), (1, [0, 1], [20, 20])]


def test_a_class_in_fewer_sequences_than_folds_is_reported_with_the_rows_it_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fold_of = {("g", "a"): 0, ("g", "b"): 1, ("g", "c"): 2}
    counts = _counts(
        {("g", "a"): {0: 9, 1: 9}, ("g", "b"): {0: 9, 1: 9}, ("g", "c"): {0: 9}}
    )
    table = fold_table(fold_of, counts, [0, 1])
    caplog.set_level(logging.WARNING)

    check_folds("clf", 3, table, [0, 1], rows="labeled frames")

    assert (
        "class 1 has labeled frames in 2 sequences, fewer than cv_folds=3"
        in caplog.text
    )


def test_a_fold_that_leaves_under_half_of_a_class_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fold_of = {("g", "a"): 0, ("g", "b"): 1, ("g", "c"): 1}
    counts = _counts(
        {
            ("g", "a"): {0: 10, 1: 90},
            ("g", "b"): {0: 10, 1: 5},
            ("g", "c"): {0: 10, 1: 5},
        }
    )
    table = fold_table(fold_of, counts, [0, 1])
    caplog.set_level(logging.WARNING)

    check_folds("clf", 2, table, [0, 1], rows="template rows")

    assert "under half of" in caplog.text
    assert "fold 0 class 1 (10 of 100)" in caplog.text


# --- Fitting and saving ---


class _Recorder(NullProgressCallback):
    def __init__(self) -> None:
        self.phases: list[tuple[str, str]] = []
        self.epochs: list[tuple[int, int]] = []

    def on_phase(self, phase: str, message: str) -> None:
        self.phases.append((phase, message))

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        self.epochs.append((epoch, total_epochs))


class _Fitter(FitReporting):
    """A classifier's fit loop, reduced to the calls it makes around each model."""

    def fit_all(self, total: int) -> None:
        for done in range(total):
            self._before_fit(done, total, whole="all runs")
            self._after_fit(done + 1, total)


def test_a_fit_names_each_fold_and_then_all_rows_and_counts_them_done() -> None:
    fitter = _Fitter()
    recorder = _Recorder()
    fitter.progress_callback = recorder

    fitter.fit_all(3)

    assert recorder.phases == [
        ("fit", "fold 1 of 2"),
        ("fit", "fold 2 of 2"),
        ("fit", "all runs"),
    ]
    assert recorder.epochs == [(0, 3), (1, 3), (2, 3)]


def test_a_single_fit_reports_nothing_and_a_cancel_stops_before_the_next() -> None:
    fitter = _Fitter()
    recorder = _Recorder()
    fitter.progress_callback = recorder
    fitter.fit_all(1)
    assert recorder.phases == []

    token = CancelToken()
    fitter.cancel_token = token
    token.cancel()
    with pytest.raises(Cancelled):
        fitter.fit_all(4)
    assert recorder.phases == []


def test_a_bundle_is_written_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run" / "model.joblib"
    write_bundle(path, {"version": "1"})
    assert joblib.load(path) == {"version": "1"}

    def fail(value: object, filename: object) -> None:
        Path(str(filename)).write_bytes(b"partial")
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr(joblib, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        write_bundle(path, {"version": "2"})

    monkeypatch.undo()
    assert joblib.load(path) == {"version": "1"}
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.joblib"]


# --- Series positions ---


def test_the_positions_of_a_pair_table_come_by_series_and_by_frame() -> None:
    df = pd.DataFrame(
        {
            "frame": [2, 2, 1, 1],
            "perspective": [1, 0, 1, 0],
            "id1": [9, 4, 9, 4],
            "id2": [4, 9, 4, 9],
        }
    )

    series = [positions.tolist() for positions in iter_series_positions(df)]

    # (4, 9, 0) first: frame 1 is row 3 and frame 2 is row 1. Then (9, 4, 1).
    assert series == [[3, 1], [2, 0]]


def test_equal_frames_keep_their_row_order_and_a_row_without_a_key_is_in_no_series() -> (
    None
):
    df = pd.DataFrame(
        {
            "frame": [1, 1, 0, 0],
            "id": [7.0, 7.0, 7.0, np.nan],
        }
    )

    series = [positions.tolist() for positions in iter_series_positions(df)]

    assert series == [[2, 0, 1]]


def test_a_table_without_identity_is_one_series() -> None:
    df = pd.DataFrame({"frame": [3, 1, 2]})

    series = [positions.tolist() for positions in iter_series_positions(df)]

    assert series == [[1, 2, 0]]
