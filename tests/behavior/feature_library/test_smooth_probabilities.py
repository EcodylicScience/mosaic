"""``smooth-probabilities``: smoothing a classifier's probabilities over consecutive frames.

The step reads one classifier run. These tests hand it the table the engine would,
with a fold table beside the input's outputs where a case needs one. The recipe
over a cross-validated classifier is in ``test_smooth_probabilities_pipeline.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d, median_filter

from mosaic.behavior.feature_library.classifier_common import (
    THRESHOLDS_FILE,
    predicted_labels,
)
from mosaic.behavior.feature_library.smooth_probabilities import (
    SmoothProbabilitiesFeature,
)
from mosaic.core.pipeline.types import DependencyLookup, InputStream, Result

GROUP = "g"
SEQUENCE = "s"
Entry = tuple[str, str]
INPUT = Result(feature="xgboost", run_id="0.4-aaa")


def _feature(**params: object) -> SmoothProbabilitiesFeature:
    return SmoothProbabilitiesFeature(
        SmoothProbabilitiesFeature.Inputs((INPUT,)),
        params={"default_class": 0, **params},
    )


def _table(
    probs: npt.NDArray[np.float64],
    frames: Sequence[int] | npt.NDArray[np.int64] | None = None,
    *,
    individual: int = 0,
) -> pd.DataFrame:
    """One classifier's output for one entry: identity, ``prob_<class>``, ``predicted_label``."""
    frame = np.arange(len(probs)) if frames is None else np.asarray(frames)
    table = pd.DataFrame(
        {"frame": frame, "id": individual, "group": GROUP, "sequence": SEQUENCE}
    )
    for column in range(probs.shape[1]):
        table[f"prob_{column}"] = probs[:, column]
    table["predicted_label"] = np.argmax(probs, axis=1)
    return table


def _probs(table: pd.DataFrame) -> npt.NDArray[np.float64]:
    columns = sorted(c for c in table.columns if c.startswith("prob_"))
    return table[columns].to_numpy(dtype=np.float64)


def _random(rows: int, classes: int = 3, seed: int = 0) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).dirichlet(np.ones(classes), size=rows)


def _ready(
    feature: SmoothProbabilitiesFeature, tmp_path: Path
) -> SmoothProbabilitiesFeature:
    """Load *feature* over an input with no fold table, as a run would."""
    _ = feature.load_state(tmp_path / "run", {}, {})
    return feature


def _input_run(
    root: Path,
    table: pd.DataFrame | None,
    entries: Sequence[Entry] = ((GROUP, SEQUENCE),),
) -> dict[str, DependencyLookup]:
    """The lookups of an input run with one output per entry and its fold table."""
    run = root / "features" / "xgboost" / "0.4-aaa"
    run.mkdir(parents=True, exist_ok=True)
    lookup: DependencyLookup = {}
    for group, sequence in entries:
        output = run / f"{group}__{sequence}.parquet"
        output.write_bytes(b"")
        lookup[(group, sequence)] = output
    if table is not None:
        table.to_parquet(run / "cv_folds.parquet")
    return {"_input_0": lookup}


def _blocks(*blocks: tuple[int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": GROUP,
                "sequence": SEQUENCE,
                "start": start,
                "stop": stop,
                "fold": fold,
                "n_rows": stop - start,
            }
            for start, stop, fold in blocks
        ]
    )


def _by_sequence(fold: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [{"group": GROUP, "sequence": SEQUENCE, "fold": fold, "n_rows": 10}]
    )


# --- Smoothing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "params", [{}, {"method": "median", "window": 7}, {"sigma": 5.0}]
)
def test_constant_probabilities_come_out_unchanged(
    tmp_path: Path, params: dict[str, object]
) -> None:
    probs = np.tile([0.2, 0.5, 0.3], (40, 1))
    feature = _ready(_feature(**params), tmp_path)

    result = feature.apply(_table(probs))

    np.testing.assert_allclose(_probs(result), probs, rtol=0, atol=1e-15)


def test_a_step_is_smoothed_as_the_gaussian_filter_smooths_it(tmp_path: Path) -> None:
    probs = np.array([[0.9, 0.1]] * 12 + [[0.2, 0.8]] * 12)
    feature = _ready(_feature(sigma=1.5), tmp_path)

    result = feature.apply(_table(probs))

    expected = gaussian_filter1d(probs, 1.5, axis=0, mode="nearest", truncate=4.0)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(_probs(result), expected, rtol=0, atol=1e-15)


def test_a_step_is_smoothed_as_the_median_filter_smooths_it(tmp_path: Path) -> None:
    probs = np.array([[0.7, 0.2, 0.1]] * 9 + [[0.1, 0.3, 0.6]] * 9)
    probs[4] = [0.1, 0.1, 0.8]
    feature = _ready(_feature(method="median", window=5), tmp_path)

    result = feature.apply(_table(probs))

    expected = median_filter(probs, size=(5, 1), mode="nearest")
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(_probs(result), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("params", [{}, {"method": "median"}])
def test_every_finite_output_row_sums_to_one(
    tmp_path: Path, params: dict[str, object]
) -> None:
    feature = _ready(_feature(**params), tmp_path)

    result = feature.apply(_table(_random(200)))

    np.testing.assert_allclose(_probs(result).sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_method_none_leaves_the_probabilities_as_they_are(tmp_path: Path) -> None:
    probs = _random(30)
    feature = _ready(_feature(method="none"), tmp_path)

    result = feature.apply(_table(probs))

    np.testing.assert_array_equal(_probs(result), probs)


def test_the_output_is_the_identity_the_probabilities_and_a_prediction(
    tmp_path: Path,
) -> None:
    feature = _ready(_feature(), tmp_path)
    table = _table(_random(20))
    table["speed"] = 1.0

    result = feature.apply(table)

    assert sorted(result.columns) == sorted(
        [
            "frame",
            "id",
            "group",
            "sequence",
            "prob_0",
            "prob_1",
            "prob_2",
            "predicted_label",
        ]
    )


def test_the_prediction_is_the_argmax_of_the_smoothed_probabilities(
    tmp_path: Path,
) -> None:
    feature = _ready(_feature(), tmp_path)
    table = _table(_random(60))

    result = feature.apply(table)

    expected = predicted_labels(_probs(result), [0, 1, 2], None, 0)
    assert result["predicted_label"].tolist() == expected.tolist()


# --- What cuts a segment --------------------------------------------------------


def _moved_after(table: pd.DataFrame, frame: int) -> pd.DataFrame:
    """*table* with the probabilities of every row past *frame* replaced."""
    moved = table.copy()
    later = moved["frame"] > frame
    replacement = _random(int(later.sum()), seed=9)
    for column in range(replacement.shape[1]):
        moved.loc[later, f"prob_{column}"] = replacement[:, column]
    return moved


def _first_rows(
    feature: SmoothProbabilitiesFeature, table: pd.DataFrame, through: int
) -> npt.NDArray[np.float64]:
    result = feature.apply(table)
    return _probs(result[result["frame"] <= through])


def test_a_frame_gap_cuts_the_smoothing(tmp_path: Path) -> None:
    table = _table(_random(20), [*range(10), *range(15, 25)])
    feature = _ready(_feature(sigma=3.0), tmp_path)

    before = _first_rows(feature, table, 9)
    after = _first_rows(feature, _moved_after(table, 9), 9)

    np.testing.assert_array_equal(after, before)


def test_without_a_cut_the_rows_across_reach_the_row(tmp_path: Path) -> None:
    table = _table(_random(20))
    feature = _ready(_feature(sigma=3.0), tmp_path)

    before = _first_rows(feature, table, 9)
    after = _first_rows(feature, _moved_after(table, 9), 9)

    assert not np.array_equal(after, before)


def test_a_block_edge_of_the_fold_table_cuts_the_smoothing(tmp_path: Path) -> None:
    table = _table(_random(20))
    feature = _feature(sigma=3.0)
    lookups = _input_run(tmp_path, _blocks((0, 10, 0), (10, 20, 1)))
    _ = feature.load_state(tmp_path / "run", {}, lookups)

    before = _first_rows(feature, table, 9)
    after = _first_rows(feature, _moved_after(table, 9), 9)

    np.testing.assert_array_equal(after, before)


def test_frames_outside_every_block_are_a_segment_of_their_own(tmp_path: Path) -> None:
    table = _table(_random(20))
    feature = _feature(sigma=3.0)
    lookups = _input_run(tmp_path, _blocks((10, 20, 0)))
    _ = feature.load_state(tmp_path / "run", {}, lookups)

    before = _first_rows(feature, table, 9)
    after = _first_rows(feature, _moved_after(table, 9), 9)

    np.testing.assert_array_equal(after, before)


def test_a_fold_table_by_sequence_cuts_nothing_within_an_entry(tmp_path: Path) -> None:
    table = _table(_random(20))
    feature = _feature(sigma=3.0)
    lookups = _input_run(tmp_path, _by_sequence())
    _ = feature.load_state(tmp_path / "run", {}, lookups)

    smoothed = _probs(feature.apply(table))

    expected = gaussian_filter1d(
        _probs(table), 3.0, axis=0, mode="nearest", truncate=4.0
    )
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(smoothed, expected, rtol=0, atol=1e-15)


def test_the_two_perspectives_of_a_pair_are_two_series(tmp_path: Path) -> None:
    probs = _random(40)
    pair = pd.DataFrame(
        {
            "frame": np.tile(np.arange(20), 2),
            "id1": 0,
            "id2": 1,
            "perspective": np.repeat([0, 1], 20),
            "group": GROUP,
            "sequence": SEQUENCE,
        }
    )
    for column in range(3):
        pair[f"prob_{column}"] = probs[:, column]
    moved = pair.copy()
    other = moved["perspective"] == 1
    replacement = _random(20, seed=4)
    for column in range(3):
        moved.loc[other, f"prob_{column}"] = replacement[:, column]
    feature = _ready(_feature(sigma=3.0), tmp_path)

    before = feature.apply(pair)
    after = feature.apply(moved)

    first = before["perspective"] == 0
    np.testing.assert_array_equal(_probs(after[first]), _probs(before[first]))
    expected = gaussian_filter1d(probs[:20], 3.0, axis=0, mode="nearest", truncate=4.0)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(_probs(before[first]), expected, rtol=0, atol=1e-15)


def test_a_row_without_probabilities_is_written_as_it_came_and_cuts(
    tmp_path: Path,
) -> None:
    table = _table(_random(20))
    table.loc[10, ["prob_0", "prob_1", "prob_2"]] = np.nan
    feature = _ready(_feature(default_class=2, sigma=3.0), tmp_path)

    result = feature.apply(table)
    moved = feature.apply(_moved_after(table, 10))

    assert np.isnan(_probs(result)[10]).all()
    assert result["predicted_label"].iloc[10] == 2
    np.testing.assert_array_equal(_probs(moved)[:10], _probs(result)[:10])


# --- The frame axis -------------------------------------------------------------


def test_a_table_of_every_second_frame_is_refused_naming_the_step(
    tmp_path: Path,
) -> None:
    feature = _ready(_feature(), tmp_path)

    with pytest.raises(ValueError, match="step of 2"):
        feature.apply(_table(_random(10), np.arange(0, 20, 2)))


def test_a_series_of_one_row_passes(tmp_path: Path) -> None:
    feature = _ready(_feature(), tmp_path)
    probs = _random(1)

    result = feature.apply(_table(probs, [7]))

    np.testing.assert_allclose(_probs(result), probs, rtol=0, atol=1e-15)


def test_frames_that_are_not_whole_numbers_are_refused(tmp_path: Path) -> None:
    feature = _ready(_feature(), tmp_path)
    table = _table(_random(4))
    table["frame"] = [0.0, 1.0, 2.5, 3.0]

    with pytest.raises(ValueError, match="whole"):
        feature.apply(table)


def test_a_table_without_frames_is_refused(tmp_path: Path) -> None:
    feature = _ready(_feature(), tmp_path)

    with pytest.raises(ValueError, match="frame"):
        feature.apply(_table(_random(4)).drop(columns="frame"))


def test_repeated_rows_of_one_series_are_refused(tmp_path: Path) -> None:
    feature = _ready(_feature(), tmp_path)

    with pytest.raises(ValueError, match="more than one row"):
        feature.apply(_table(_random(4), [0, 1, 1, 2]))


def test_a_table_without_probabilities_is_refused(tmp_path: Path) -> None:
    feature = _ready(_feature(), tmp_path)

    with pytest.raises(ValueError, match="prob_"):
        feature.apply(_table(_random(4)).drop(columns=["prob_0", "prob_1", "prob_2"]))


def test_two_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="one classifier run"):
        _ = SmoothProbabilitiesFeature.Inputs(
            (INPUT, Result(feature="xgboost", run_id="0.4-bbb"))
        )


# --- Bouts ----------------------------------------------------------------------


def _bouts(*bouts: tuple[int, list[float]]) -> npt.NDArray[np.float64]:
    """Rows of probabilities: *count* rows of each given row."""
    return np.array([row for count, row in bouts for _ in range(count)])


def _merged(probs: npt.NDArray[np.float64], tmp_path: Path) -> list[int]:
    feature = _ready(_feature(method="none", min_bout_frames=3), tmp_path)
    return feature.apply(_table(probs))["predicted_label"].tolist()


def test_a_short_bout_takes_the_neighbor_class_of_the_higher_mean(
    tmp_path: Path,
) -> None:
    probs = _bouts((6, [0.8, 0.1, 0.1]), (2, [0.25, 0.45, 0.3]), (6, [0.1, 0.1, 0.8]))

    assert _merged(probs, tmp_path) == [0] * 6 + [2] * 8


def test_a_tie_between_the_neighbors_goes_to_the_earlier(tmp_path: Path) -> None:
    probs = _bouts(
        (6, [0.1, 0.1, 0.8]), (2, [0.275, 0.45, 0.275]), (6, [0.8, 0.1, 0.1])
    )

    assert _merged(probs, tmp_path) == [2] * 8 + [0] * 6


def test_a_bout_at_the_edge_of_a_segment_takes_its_one_neighbor(
    tmp_path: Path,
) -> None:
    probs = _bouts((2, [0.1, 0.5, 0.4]), (6, [0.8, 0.1, 0.1]))

    assert _merged(probs, tmp_path) == [0] * 8


def test_neighbors_of_one_class_join_once_the_bout_between_them_merges(
    tmp_path: Path,
) -> None:
    """C(10) A(2) B(1) A(2) D(10): B joins the two A bouts into one of five frames.

    The rows of the second A bout hold more of D than of A, and D's threshold
    leaves them at A. Were that bout still a bout of two, it would take D.
    """
    probs = _bouts(
        (10, [0.9, 0.03, 0.03, 0.04]),
        (2, [0.05, 0.6, 0.3, 0.05]),
        (1, [0.05, 0.3, 0.6, 0.05]),
        (2, [0.05, 0.4, 0.05, 0.5]),
        (10, [0.03, 0.03, 0.04, 0.9]),
    )
    feature = _ready(
        _feature(method="none", min_bout_frames=3, decision_threshold={3: 0.6}),
        tmp_path,
    )

    result = feature.apply(_table(probs))

    assert result["predicted_label"].tolist() == [0] * 10 + [1] * 5 + [3] * 10


def test_a_segment_of_one_bout_stays_as_it_is(tmp_path: Path) -> None:
    probs = _bouts((2, [0.1, 0.8, 0.1]))

    assert _merged(probs, tmp_path) == [1, 1]


def test_merging_leaves_the_probabilities_as_they_are(tmp_path: Path) -> None:
    probs = _bouts((6, [0.8, 0.1, 0.1]), (2, [0.25, 0.45, 0.3]), (6, [0.1, 0.1, 0.8]))
    feature = _ready(_feature(method="none", min_bout_frames=3), tmp_path)

    result = feature.apply(_table(probs))

    np.testing.assert_array_equal(_probs(result), probs)


def test_a_bout_does_not_merge_across_a_cut(tmp_path: Path) -> None:
    probs = _bouts((6, [0.8, 0.1, 0.1]), (2, [0.1, 0.8, 0.1]))
    feature = _ready(_feature(method="none", min_bout_frames=3), tmp_path)

    result = feature.apply(_table(probs, [*range(6), 20, 21]))

    assert result["predicted_label"].tolist() == [0] * 6 + [1, 1]


def test_shuffling_the_rows_changes_no_output_row(tmp_path: Path) -> None:
    tables = [
        _table(_random(120, seed=individual), individual=individual)
        for individual in (0, 1)
    ]
    table = pd.concat(tables, ignore_index=True)
    shuffled = table.sample(frac=1.0, random_state=3)
    feature = _ready(_feature(min_bout_frames=4), tmp_path)

    key = ["id", "frame"]
    ordered = feature.apply(table).sort_values(key, ignore_index=True)
    reordered = feature.apply(shuffled).sort_values(key, ignore_index=True)

    pd.testing.assert_frame_equal(reordered, ordered)


# --- Thresholds -----------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.6, {0: 1.0, 1: 0.2}])
def test_a_float_and_a_mapping_threshold_apply(
    tmp_path: Path, threshold: float | dict[int, float]
) -> None:
    feature = _ready(_feature(decision_threshold=threshold, default_class=2), tmp_path)

    result = feature.apply(_table(_random(50)))

    expected = predicted_labels(_probs(result), [0, 1, 2], threshold, 2)
    assert result["predicted_label"].tolist() == expected.tolist()


_TUNED: dict[str, object] = {
    "feature": "tune-threshold",
    "run_id": "0.1-0123456789",
    "pattern": THRESHOLDS_FILE,
}


def _tuned_run(root: Path, read: Result) -> Path:
    """A ``tune-threshold`` run that tuned on *read*: its thresholds and ``params.json``."""
    run = root / "tune"
    run.mkdir(parents=True)
    pd.DataFrame(
        {"class": np.array([0, 1, 2], dtype=np.int64), "threshold": [1.0, 0.3, 0.4]}
    ).to_parquet(run / THRESHOLDS_FILE)
    document = {"_inputs": [{"feature": read.feature, "run_id": read.run_id}]}
    (run / "params.json").write_text(json.dumps(document))
    return run / THRESHOLDS_FILE


def test_a_tuned_threshold_applies_with_method_none(tmp_path: Path) -> None:
    thresholds = _tuned_run(tmp_path, INPUT)
    feature = _feature(method="none", decision_threshold=_TUNED)
    _ = feature.load_state(tmp_path / "run", {"decision_threshold": thresholds}, {})
    table = _table(_random(50))

    result = feature.apply(table)

    expected = predicted_labels(_probs(table), [0, 1, 2], {0: 1.0, 1: 0.3, 2: 0.4}, 0)
    assert result["predicted_label"].tolist() == expected.tolist()


@pytest.mark.parametrize("method", ["gaussian", "median"])
def test_a_tuned_threshold_is_refused_where_the_step_smooths(method: str) -> None:
    with pytest.raises(ValueError, match="method none"):
        _ = _feature(method=method, decision_threshold=_TUNED)


def test_a_threshold_tuned_on_another_run_is_refused(tmp_path: Path) -> None:
    thresholds = _tuned_run(tmp_path, Result(feature="xgboost", run_id="0.4-bbb"))
    feature = _feature(method="none", decision_threshold=_TUNED)

    with pytest.raises(ValueError, match="tuned the thresholds on xgboost 0.4-bbb"):
        _ = feature.load_state(tmp_path / "run", {"decision_threshold": thresholds}, {})


def test_a_tuned_threshold_that_did_not_resolve_is_refused(tmp_path: Path) -> None:
    feature = _feature(method="none", decision_threshold=_TUNED)

    with pytest.raises(FileNotFoundError, match="holds no"):
        _ = feature.load_state(tmp_path / "run", {}, {})


# --- Params ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"method": "median", "sigma": 3.0},
        {"method": "none", "sigma": 3.0},
        {"window": 7},
        {"method": "none", "window": 7},
    ],
)
def test_a_setting_of_another_method_is_refused(params: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="only applies with method"):
        _ = _feature(**params)


@pytest.mark.parametrize("window", [1, 4])
def test_a_median_window_is_odd_and_at_least_three(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        _ = _feature(method="median", window=window)


def test_default_class_is_required() -> None:
    with pytest.raises(ValueError, match="default_class"):
        _ = SmoothProbabilitiesFeature(SmoothProbabilitiesFeature.Inputs((INPUT,)))


@pytest.mark.parametrize(
    ("params", "other"),
    [
        ({}, {"sigma": 3.0}),
        ({"method": "median"}, {"method": "median", "window": 7}),
        ({}, {"min_bout_frames": 3}),
        ({}, {"decision_threshold": 0.4}),
        ({}, {"method": "none"}),
    ],
)
def test_a_setting_changes_the_identity(
    params: dict[str, object], other: dict[str, object]
) -> None:
    assert (
        _feature(**params).params.identity_dump()
        != _feature(**other).params.identity_dump()
    )


def test_spellings_of_one_setting_give_one_identity() -> None:
    assert (
        _feature(sigma=2).params.identity_dump()
        == _feature(sigma=2.0).params.identity_dump()
    )
    assert (
        _feature(decision_threshold={"1": 0.2}).params.identity_dump()
        == _feature(decision_threshold={1: 0.2}).params.identity_dump()
    )


# --- The fold table -------------------------------------------------------------


def test_the_input_fold_table_is_copied_to_the_run(tmp_path: Path) -> None:
    table = _blocks((0, 10, 0), (10, 20, 1))
    lookups = _input_run(tmp_path, table)
    run = tmp_path / "run"
    feature = _feature()

    assert not feature.load_state(run, {}, lookups)
    feature.fit(InputStream(lambda: iter(()), n_entries=0))
    feature.save_state(run)

    copied = pd.read_parquet(run / "cv_folds.parquet")
    pd.testing.assert_frame_equal(copied, pd.read_parquet(_cv_path(lookups)))
    assert _feature().load_state(run, {}, lookups)


def test_an_input_without_a_fold_table_leaves_nothing_to_write(tmp_path: Path) -> None:
    lookups = _input_run(tmp_path, None)
    run = tmp_path / "run"

    assert _feature().load_state(run, {}, lookups)
    assert not (run / "cv_folds.parquet").exists()


def _cv_path(lookups: dict[str, DependencyLookup]) -> Path:
    first = next(iter(lookups["_input_0"].values()))
    return first.parent / "cv_folds.parquet"


def test_the_step_learns_nothing_and_hashes_no_scope() -> None:
    feature = _feature()

    assert feature.fits is False
    assert feature.scope_dependent is False
    assert SmoothProbabilitiesFeature.category == "per-frame"
