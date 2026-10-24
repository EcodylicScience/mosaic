"""``tune-threshold``: choosing a classifier's decision thresholds from out-of-fold probabilities.

The step reads one classifier run through ``inputs``, so these tests build the
table the engine would hand it from real parquet files, and a fold table and
label files beside the outputs. The recipe that fits a classifier, tunes it and
applies the thresholds is in ``test_tune_threshold_pipeline.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.classifier_common import (
    THRESHOLDS_FILE,
    ThresholdsArtifact,
    predicted_labels,
    read_thresholds,
)
from mosaic.behavior.feature_library.probability_ensemble import (
    ProbabilityEnsembleFeature,
)
from mosaic.behavior.feature_library.tune_threshold import (
    DEFAULT_GRID,
    REPORT_FILE,
    TuneThresholdFeature,
)
from mosaic.core.pipeline.types import DependencyLookup, InputStream, Result
from tests.helpers import Event, write_sparse_labels

GROUP = "g"
FRAMES = 240
Entry = tuple[str, str]


def _inputs() -> TuneThresholdFeature.Inputs:
    return TuneThresholdFeature.Inputs((Result(feature="xgboost", run_id="0.4-aaa"),))


def _feature(**params: object) -> TuneThresholdFeature:
    return TuneThresholdFeature(_inputs(), params={"default_class": 0, **params})


def _output(probs: npt.NDArray[np.float64], group: str, sequence: str) -> pd.DataFrame:
    """One classifier's output for one entry: identity, ``prob_<class>``, ``predicted_label``."""
    table = pd.DataFrame(
        {"frame": np.arange(len(probs)), "id": 0, "group": group, "sequence": sequence}
    )
    for column in range(probs.shape[1]):
        table[f"prob_{column}"] = probs[:, column]
    table["predicted_label"] = np.argmax(probs, axis=1)
    return table


def _binary(rng: np.random.Generator) -> tuple[npt.NDArray[np.float64], np.ndarray]:
    """A rare positive class that the classifier scores low: the best cut is well under 0.5."""
    truth = (rng.random(FRAMES) < 0.15).astype(np.int64)
    positive = np.where(truth == 1, rng.beta(4, 6, FRAMES), rng.beta(1, 8, FRAMES))
    return np.column_stack([1.0 - positive, positive]), truth


def _three_classes(
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], np.ndarray]:
    """Class 0 is common and classes 1 and 2 are rare and scored low."""
    truth = rng.choice(3, size=FRAMES, p=[0.7, 0.18, 0.12]).astype(np.int64)
    logits = rng.normal(0.0, 1.0, (FRAMES, 3))
    logits[np.arange(FRAMES), truth] += np.array([2.2, 1.4, 1.4])[truth]
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True), truth


class _Study:
    """Entries with probabilities, labels and a fold table, as the engine would hand them."""

    def __init__(
        self,
        root: Path,
        *,
        entries: int = 6,
        folds: int = 3,
        classes: int = 2,
        seed: int = 5,
    ) -> None:
        self.root = root
        self.tables: dict[Entry, pd.DataFrame] = {}
        self.truth: dict[Entry, np.ndarray] = {}
        self.files: DependencyLookup = {}
        rng = np.random.default_rng(seed)
        make = _binary if classes == 2 else _three_classes
        for index in range(entries):
            entry = (GROUP, f"s{index}")
            probs, truth = make(rng)
            self.tables[entry] = _output(probs, *entry)
            self.truth[entry] = truth
            events: list[Event] = [
                (frame, 0, -1, int(label)) for frame, label in enumerate(truth)
            ]
            self.files[entry] = write_sparse_labels(
                root / "labels" / f"{entry[1]}.npz", events
            )
        self.folds = {entry: index % folds for index, entry in enumerate(self.tables)}

    def add(
        self, entry: Entry, probs: npt.NDArray[np.float64], truth: np.ndarray
    ) -> None:
        """Add an entry that is not in the fold table."""
        self.tables[entry] = _output(probs, *entry)
        self.truth[entry] = truth
        events: list[Event] = [(f, 0, -1, int(label)) for f, label in enumerate(truth)]
        self.files[entry] = write_sparse_labels(
            self.root / "labels" / f"{entry[1]}.npz", events
        )

    def lookups(
        self,
        *,
        fold_table: bool = True,
        blocks: Sequence[tuple[int, int, int]] | None = None,
    ) -> dict[str, DependencyLookup]:
        """The lookups of the input, with a fold table of entries or of *blocks*.

        *blocks* are ``(start, stop, fold)`` cut out of every entry: a table of
        blocks is what a classifier fitted with ``cv_unit="block"`` writes.
        """
        run = self.root / "classifier"
        run.mkdir(parents=True, exist_ok=True)
        lookup: DependencyLookup = {}
        for group, sequence in self.tables:
            output = run / f"{group}__{sequence}.parquet"
            output.write_bytes(b"")
            lookup[(group, sequence)] = output
        if fold_table and blocks is not None:
            pd.DataFrame(
                [
                    {
                        "group": g,
                        "sequence": s,
                        "start": start,
                        "stop": stop,
                        "fold": fold,
                        "n_rows": stop - start,
                    }
                    for g, s in sorted(self.tables)
                    for start, stop, fold in blocks
                ]
            ).to_parquet(run / "cv_folds.parquet")
        elif fold_table:
            pd.DataFrame(
                [
                    {"group": g, "sequence": s, "fold": fold, "n_rows": FRAMES}
                    for (g, s), fold in sorted(self.folds.items())
                ]
            ).to_parquet(run / "cv_folds.parquet")
        return {"_input_0": lookup, "labels": self.files}

    def stream(self, order: Sequence[Entry] | None = None) -> InputStream:
        keys = list(self.tables) if order is None else list(order)
        items = [(f"{g}__{s}", self.tables[(g, s)]) for g, s in keys]
        return InputStream(lambda: iter(items), n_entries=len(items))

    def rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The probabilities, truth and fold of every fold entry, pooled."""
        entries = sorted(self.folds)
        probs = np.concatenate(
            [
                self.tables[e][
                    [c for c in self.tables[e].columns if c.startswith("prob_")]
                ]
                for e in entries
            ]
        )
        truth = np.concatenate([self.truth[e] for e in entries])
        fold = np.concatenate([np.full(FRAMES, self.folds[e]) for e in entries])
        return probs, truth, fold


def _tuned(
    study: _Study,
    name: str = "run",
    blocks: Sequence[tuple[int, int, int]] | None = None,
    **params: object,
) -> tuple[TuneThresholdFeature, Path]:
    """Fit a step over *study* and save its state under ``study.root / name``."""
    feature = _feature(**params)
    run = study.root / name
    assert not feature.load_state(run, {}, study.lookups(blocks=blocks))
    feature.fit(study.stream())
    feature.save_state(run)
    return feature, run


def _report(run: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((run / REPORT_FILE).read_text())
    return report


def _binary_f1(truth: np.ndarray, positive: np.ndarray) -> float:
    hits = int(((truth == 1) & positive).sum())
    predicted, support = int(positive.sum()), int((truth == 1).sum())
    if hits == 0:
        return 0.0
    precision, recall = hits / predicted, hits / support
    return 2 * precision * recall / (precision + recall)


def _best_cut(probs1: np.ndarray, truth: np.ndarray, grid: Sequence[float]) -> float:
    """The first grid threshold of the highest F1 of the positive class, by brute force."""
    best, best_value = grid[0], -1.0
    for limit in grid:
        value = _binary_f1(truth, probs1 >= limit)
        if value > best_value:
            best, best_value = limit, value
    return best


# --- The grid and the rule ------------------------------------------------------


def test_the_positive_class_of_a_binary_problem_gets_the_cut_a_brute_force_search_picks(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    _, run = _tuned(study, metric="pooled_f1", classes=[1])

    probs, truth, _ = study.rows()
    expected = _best_cut(probs[:, 1], truth, DEFAULT_GRID)
    assert read_thresholds(run / THRESHOLDS_FILE) == {0: 1.0, 1: expected}
    assert expected < 0.5, "the problem is built so that argmax is the wrong cut"
    report = _report(run)
    plain = report["plain"]
    assert plain["value"] == pytest.approx(_binary_f1(truth, probs[:, 1] >= expected))
    assert plain["value"] > _binary_f1(truth, probs[:, 1] > probs[:, 0])


def test_a_tie_goes_to_the_lowest_threshold(tmp_path: Path) -> None:
    """Every cut between the two scores gives the same predictions, so the lowest wins."""
    study = _Study(tmp_path)
    for entry, table in study.tables.items():
        positive = study.truth[entry] == 1
        table["prob_1"] = np.where(positive, 0.9, 0.1)
        table["prob_0"] = 1.0 - table["prob_1"]

    _, run = _tuned(study, metric="pooled_f1", classes=[1])

    assert read_thresholds(run / THRESHOLDS_FILE)[1] == 0.15


def test_a_grid_point_is_tried_and_the_curve_is_reported(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    grid = [0.1, 0.2, 0.3, 0.4, 0.5]

    _, run = _tuned(study, metric="pooled_f1", classes=[1], grid=grid)

    plain = _report(run)["plain"]
    curve = plain["curves"]["1"]
    probs, truth, _ = study.rows()
    assert [point["threshold"] for point in curve] == grid
    for point in curve:
        assert point["value"] == pytest.approx(
            _binary_f1(truth, probs[:, 1] >= point["threshold"])
        )
    assert plain["value"] == max(point["value"] for point in curve)


def test_the_default_class_keeps_out_of_the_competition_and_an_unlisted_class_gets_zero(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path, classes=3)

    _, run = _tuned(study, classes=[1])

    thresholds = read_thresholds(run / THRESHOLDS_FILE)
    assert thresholds[0] == 1.0
    assert thresholds[2] == 0.0
    assert thresholds[1] in DEFAULT_GRID
    report = _report(run)
    assert report["classes_tuned"] == [1]


def test_every_class_but_the_default_is_tuned_when_none_are_listed(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path, classes=3)

    _, run = _tuned(study)

    thresholds = read_thresholds(run / THRESHOLDS_FILE)
    assert thresholds[0] == 1.0
    assert thresholds[1] in DEFAULT_GRID and thresholds[2] in DEFAULT_GRID
    assert _report(run)["classes_tuned"] == [1, 2]


def test_several_classes_do_no_worse_than_the_best_shared_threshold(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path, classes=3)
    _, run = _tuned(study, average_over=[1, 2], nested=False)

    probs, truth, _ = study.rows()

    def macro(limits: dict[int, float]) -> float:
        predicted = predicted_labels(probs, [0, 1, 2], limits, 0)
        scores = [
            2
            * ((predicted == c) & (truth == c)).sum()
            / max(int((predicted == c).sum() + (truth == c).sum()), 1)
            for c in (1, 2)
        ]
        return float(np.mean(scores))

    shared = max(macro({0: 1.0, 1: t, 2: t}) for t in DEFAULT_GRID)
    chosen = read_thresholds(run / THRESHOLDS_FILE)
    plain = _report(run)["plain"]
    assert plain["value"] == pytest.approx(macro(chosen))
    assert plain["value"] >= shared - 1e-12


def test_the_thresholds_do_not_depend_on_the_order_entries_arrive_in(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path, classes=3)
    _, forward_run = _tuned(study, "forward")
    backward = _feature()
    backward_run = tmp_path / "backward"
    assert not backward.load_state(backward_run, {}, study.lookups())
    backward.fit(study.stream(list(reversed(study.tables))))
    backward.save_state(backward_run)

    assert read_thresholds(forward_run / THRESHOLDS_FILE) == read_thresholds(
        backward_run / THRESHOLDS_FILE
    )
    assert _report(forward_run) == _report(backward_run)


# --- Plain and nested estimates -------------------------------------------------


def _noise(root: Path) -> _Study:
    """Probabilities that say nothing about the labels, so any threshold is luck."""
    study = _Study(root, entries=6, folds=3, seed=11)
    rng = np.random.default_rng(2)
    for table in study.tables.values():
        table["prob_1"] = rng.random(FRAMES)
        table["prob_0"] = 1.0 - table["prob_1"]
    return study


def test_the_nested_estimate_is_the_score_of_each_fold_under_thresholds_from_the_others(
    tmp_path: Path,
) -> None:
    study = _noise(tmp_path)

    _, run = _tuned(study, metric="pooled_f1", classes=[1])

    probs, truth, fold = study.rows()
    positive = np.zeros(len(truth), dtype=bool)
    chosen: dict[int, float] = {}
    for k in sorted(set(fold)):
        cut = _best_cut(probs[fold != k, 1], truth[fold != k], DEFAULT_GRID)
        chosen[int(k)] = cut
        positive[fold == k] = probs[fold == k, 1] >= cut
    report = _report(run)
    nested = report["nested"]
    assert nested["value"] == pytest.approx(_binary_f1(truth, positive))
    assert {f["fold"]: f["thresholds"]["1"] for f in nested["folds"]} == chosen


def test_the_plain_estimate_flatters_a_threshold_chosen_on_noise(
    tmp_path: Path,
) -> None:
    study = _noise(tmp_path)

    _, run = _tuned(study, metric="pooled_f1", classes=[1])

    report = _report(run)
    plain, nested = report["plain"], report["nested"]
    assert plain["value"] > nested["value"]


def test_without_nested_the_report_has_the_plain_estimate_only(tmp_path: Path) -> None:
    study = _Study(tmp_path)

    _, run = _tuned(study, nested=False)

    report = _report(run)
    assert report["nested"] is None
    assert "value" in report["plain"]


def test_the_report_names_what_neither_estimate_corrects(tmp_path: Path) -> None:
    _, run = _tuned(_Study(tmp_path))

    assert "hyperparameters" in str(_report(run)["caveat"])


# --- Rows with no probability ---------------------------------------------------


def test_a_labeled_row_with_no_probability_is_scored_as_the_default_class(
    tmp_path: Path,
) -> None:
    """The consumer assigns it default_class and evaluate-labels counts it, so it counts here."""
    study = _Study(tmp_path)
    for table in study.tables.values():
        table.loc[table["frame"] % 6 == 0, ["prob_0", "prob_1"]] = np.nan

    _, run = _tuned(study, metric="pooled_f1", classes=[1], nested=False)

    probs, truth, _ = study.rows()
    thresholds = read_thresholds(run / THRESHOLDS_FILE)
    predicted = predicted_labels(probs, [0, 1], thresholds, 0)
    report = _report(run)
    assert report["n_rows"] == len(truth)
    assert report["plain"]["value"] == pytest.approx(_binary_f1(truth, predicted == 1))
    # The cut that is best over every labeled row, the unscored ones included.
    best = max(
        DEFAULT_GRID,
        key=lambda limit: (
            _binary_f1(
                truth, predicted_labels(probs, [0, 1], {0: 1.0, 1: limit}, 0) == 1
            ),
            -limit,
        ),
    )
    assert thresholds[1] == best


def test_the_plain_estimate_is_what_the_consumer_output_scores_with_missing_rows(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    for table in study.tables.values():
        table.loc[table["frame"] < 40, ["prob_0", "prob_1"]] = np.nan
    feature, run = _tuned(study, metric="pooled_f1", classes=[1], nested=False)

    truth = np.concatenate([study.truth[e] for e in sorted(study.folds)])
    applied = np.concatenate(
        [
            feature.apply(study.tables[e])["predicted_label"].to_numpy()
            for e in sorted(study.folds)
        ]
    )

    assert _report(run)["plain"]["value"] == pytest.approx(
        _binary_f1(truth, applied == 1)
    )


# --- The edge of the grid ---------------------------------------------------------


def test_a_threshold_on_the_lowest_grid_point_is_reported_as_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    study = _Study(tmp_path)
    caplog.set_level("WARNING")

    # The best cut of this problem is under 0.4, so the grid's edge wins.
    _, run = _tuned(study, metric="pooled_f1", classes=[1], grid=[0.4, 0.5, 0.6])

    assert read_thresholds(run / THRESHOLDS_FILE)[1] == 0.4
    (warning,) = _report(run)["warnings"]
    assert "class 1 is 0.4, the lowest value of grid" in warning
    assert "below it" in warning
    assert warning in caplog.text


def test_a_threshold_on_the_highest_grid_point_is_reported_as_a_warning(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    for entry, table in study.tables.items():
        positive = study.truth[entry] == 1
        table["prob_1"] = np.where(positive, 0.99, 0.5)
        table["prob_0"] = 1.0 - table["prob_1"]

    _, run = _tuned(study, metric="pooled_f1", classes=[1], grid=[0.1, 0.3, 0.6])

    # Every cut up to 0.5 keeps the negatives that score 0.5, and only a higher
    # one separates the classes, so the answer is the top of the grid.
    assert read_thresholds(run / THRESHOLDS_FILE)[1] == 0.6
    (warning,) = _report(run)["warnings"]
    assert "the highest value of grid" in warning
    assert "above it" in warning


def test_a_threshold_inside_the_grid_is_not_a_warning(tmp_path: Path) -> None:
    _, run = _tuned(_Study(tmp_path), metric="pooled_f1", classes=[1])

    assert _report(run)["warnings"] == []


def test_a_one_point_grid_has_no_edge(tmp_path: Path) -> None:
    _, run = _tuned(_Study(tmp_path), metric="pooled_f1", classes=[1], grid=[0.3])

    assert _report(run)["warnings"] == []


# --- Block folds -----------------------------------------------------------------

BLOCKS = [(0, 100, 0), (140, 200, 1)]
"""Two blocks of each entry, in different folds, with frames 100-139 and 200+ in none."""


def _in_blocks(frames: np.ndarray) -> np.ndarray:
    return np.isin(frames, np.arange(0, 100)) | np.isin(frames, np.arange(140, 200))


def test_a_fold_table_of_blocks_tunes_on_the_rows_inside_a_block(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    # Outside every block the positives score lowest: a row that leaked in
    # would pull the cut to where it is wrong.
    for entry, table in study.tables.items():
        outside = ~_in_blocks(table["frame"].to_numpy())
        positive = study.truth[entry] == 1
        table.loc[outside, "prob_1"] = np.where(positive[outside], 0.0, 0.99)
        table["prob_0"] = 1.0 - table["prob_1"]

    _, run = _tuned(study, blocks=BLOCKS, metric="pooled_f1", classes=[1])

    kept = np.concatenate(
        [_in_blocks(study.tables[e]["frame"].to_numpy()) for e in sorted(study.tables)]
    )
    probs, truth, _ = study.rows()
    expected = _best_cut(probs[kept, 1], truth[kept], DEFAULT_GRID)
    report = _report(run)
    assert read_thresholds(run / THRESHOLDS_FILE)[1] == expected
    assert report["n_rows"] == int(kept.sum()) == 6 * 160
    assert report["plain"]["value"] == pytest.approx(
        _binary_f1(truth[kept], probs[kept, 1] >= expected)
    )


def test_the_nested_estimate_leaves_out_the_blocks_of_the_fold_it_scores(
    tmp_path: Path,
) -> None:
    study = _noise(tmp_path)

    _, run = _tuned(study, blocks=BLOCKS, metric="pooled_f1", classes=[1])

    probs, truth, _ = study.rows()
    frame = np.concatenate([np.arange(FRAMES)] * len(study.tables))
    fold = np.where(frame < 100, 0, np.where((frame >= 140) & (frame < 200), 1, -1))
    kept = fold >= 0
    positive = np.zeros(len(truth), dtype=bool)
    for k in (0, 1):
        train = kept & (fold != k)
        cut = _best_cut(probs[train, 1], truth[train], DEFAULT_GRID)
        positive[fold == k] = probs[fold == k, 1] >= cut
    nested = _report(run)["nested"]
    assert nested["n_rows"] == int(kept.sum())
    assert nested["value"] == pytest.approx(_binary_f1(truth[kept], positive[kept]))
    assert sorted(f["fold"] for f in nested["folds"]) == [0, 1]


# --- Entries -------------------------------------------------------------------


def test_an_entry_outside_the_fold_table_does_not_choose_the_thresholds(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    _, plain_run = _tuned(study, "plain")
    # A test entry on which every cut is wrong: the positives score lowest.
    rng = np.random.default_rng(0)
    truth = (rng.random(FRAMES) < 0.5).astype(np.int64)
    positive = np.where(truth == 1, 0.02, 0.95)
    study.add(("test", "t0"), np.column_stack([1.0 - positive, positive]), truth)

    _, wider_run = _tuned(study, "wider")

    assert read_thresholds(wider_run / THRESHOLDS_FILE) == read_thresholds(
        plain_run / THRESHOLDS_FILE
    )
    assert _report(wider_run) == _report(plain_run)


def test_a_fold_entry_that_is_not_in_the_scope_is_refused(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    feature = _feature()
    assert not feature.load_state(tmp_path / "run", {}, study.lookups())
    kept = [entry for entry in study.tables if entry != (GROUP, "s3")]

    with pytest.raises(ValueError, match=r"1 entries of the input's folds.*g/s3"):
        feature.fit(study.stream(kept))


def test_an_input_without_a_fold_table_is_refused(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    feature = _feature()
    assert not feature.load_state(tmp_path / "run", {}, study.lookups(fold_table=False))

    with pytest.raises(ValueError, match="input writes no cv_folds.parquet"):
        feature.fit(study.stream())


def test_only_the_rows_of_the_chosen_perspective_are_tuned_on(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    probs, truth, _ = study.rows()
    for entry, table in list(study.tables.items()):
        # A second perspective whose probabilities are wrong about everything.
        mirror = table.copy()
        mirror["prob_1"] = 1.0 - table["prob_1"]
        mirror["prob_0"] = 1.0 - mirror["prob_1"]
        study.tables[entry] = pd.concat(
            [table.assign(perspective=0), mirror.assign(perspective=1)]
        ).reset_index(drop=True)
        events: list[Event] = [
            (frame, 0, -1, int(label)) for frame, label in enumerate(study.truth[entry])
        ]
        study.files[entry] = write_sparse_labels(
            tmp_path / "labels" / f"{entry[1]}.npz", events
        )

    _, run = _tuned(study, metric="pooled_f1", classes=[1], perspective=0)

    expected = _best_cut(probs[:, 1], truth, DEFAULT_GRID)
    assert read_thresholds(run / THRESHOLDS_FILE)[1] == expected


# --- State and apply ------------------------------------------------------------


def test_apply_assigns_each_row_by_the_rule_a_consumer_would_apply(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path, classes=3)
    feature, run = _tuned(study)
    thresholds = read_thresholds(run / THRESHOLDS_FILE)
    table = study.tables[(GROUP, "s0")]

    result = feature.apply(table)

    probs = table[["prob_0", "prob_1", "prob_2"]].to_numpy()
    expected = predicted_labels(probs, [0, 1, 2], thresholds, 0)
    assert result["predicted_label"].tolist() == expected.tolist()
    assert set(result.columns) == set(table.columns)


def test_a_saved_run_is_a_cache_that_applies_the_same_thresholds(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    feature, run = _tuned(study)
    again = _feature()

    assert again.load_state(run, {}, study.lookups())

    table = study.tables[(GROUP, "s1")]
    pd.testing.assert_frame_equal(again.apply(table), feature.apply(table))


def test_apply_refuses_an_input_with_other_classes(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    feature, _ = _tuned(study)
    other = study.tables[(GROUP, "s0")].assign(prob_2=0.0)

    with pytest.raises(ValueError, match="tuned for classes"):
        feature.apply(other)


def test_the_thresholds_are_written_last(tmp_path: Path) -> None:
    """The thresholds' presence is what makes the run a cache, so the report is first."""
    study = _Study(tmp_path)
    _, run = _tuned(study)

    assert (run / REPORT_FILE).stat().st_mtime_ns <= (
        run / THRESHOLDS_FILE
    ).stat().st_mtime_ns


# --- Parameters -----------------------------------------------------------------


def test_the_step_reads_one_run() -> None:
    with pytest.raises(ValueError, match="reads one classifier run"):
        TuneThresholdFeature.Inputs(
            (
                Result(feature="xgboost", run_id="0.4-aaa"),
                Result(feature="xgboost", run_id="0.4-bbb"),
            )
        )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"grid": []}, "lists no threshold"),
        ({"grid": [0.5, 0.2]}, "strictly ascending"),
        ({"grid": [0.2, 0.2]}, "strictly ascending"),
        ({"grid": [0.5, 1.5]}, "between 0 and 1"),
        ({"classes": []}, "lists no class"),
        ({"classes": [1, 1]}, "twice"),
        ({"classes": [0, 1]}, "default_class 0"),
        ({"metric": "pooled_f1", "classes": [1, 2]}, "exactly one"),
        ({"metric": "pooled_f1", "average_over": [1]}, "only applies with metric"),
    ],
)
def test_a_choice_that_cannot_be_searched_is_refused(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _feature(**params)


def test_pooled_f1_needs_one_class_to_tune(tmp_path: Path) -> None:
    study = _Study(tmp_path, classes=3)
    feature = _feature(metric="pooled_f1")
    assert not feature.load_state(tmp_path / "run", {}, study.lookups())

    with pytest.raises(ValueError, match="pooled_f1 scores one class"):
        feature.fit(study.stream())


def test_a_listed_class_the_input_does_not_write_is_refused(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    feature = _feature(classes=[4])
    assert not feature.load_state(tmp_path / "run", {}, study.lookups())

    with pytest.raises(ValueError, match=r"lists \[4\], which the input does not"):
        feature.fit(study.stream())


def test_the_step_fits_once_and_hashes_no_scope() -> None:
    feature = _feature()

    assert TuneThresholdFeature.scope_dependent is False
    assert TuneThresholdFeature.fit_set_pinned_by == "inputs"
    assert feature.fits is True
    assert feature.scope_dependent is False


# --- The consumer ---------------------------------------------------------------


def _ensemble(**params: object) -> ProbabilityEnsembleFeature:
    return ProbabilityEnsembleFeature(
        ProbabilityEnsembleFeature.Inputs(
            (Result(feature="xgboost", run_id="0.4-aaa"),)
        ),
        params={"default_class": 0, **params},
    )


_THRESHOLDS_REFERENCE: dict[str, object] = {
    "feature": "tune-threshold",
    "run_id": "0.1-0123456789",
    "pattern": THRESHOLDS_FILE,
}


def test_an_ensemble_applies_the_thresholds_of_a_tuned_run(tmp_path: Path) -> None:
    study = _Study(tmp_path, classes=3)
    _, run = _tuned(study)
    ensemble = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE)
    assert isinstance(ensemble.params.decision_threshold, ThresholdsArtifact)
    paths = {"decision_threshold": run / THRESHOLDS_FILE}
    assert not ensemble.load_state(tmp_path / "ensemble", paths, {})
    ensemble.fit(InputStream(lambda: iter(()), n_entries=0))
    table = study.tables[(GROUP, "s2")]

    result = ensemble.apply(table)

    probs = table[["prob_0", "prob_1", "prob_2"]].to_numpy()
    expected = predicted_labels(
        probs, [0, 1, 2], read_thresholds(run / THRESHOLDS_FILE), 0
    )
    assert result["predicted_label"].tolist() == expected.tolist()


def test_an_ensemble_cached_with_its_weights_still_reads_the_thresholds(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    _, run = _tuned(study)
    paths = {"decision_threshold": run / THRESHOLDS_FILE}
    first = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE)
    assert not first.load_state(tmp_path / "ensemble", paths, {})
    first.fit(InputStream(lambda: iter(()), n_entries=0))
    first.save_state(tmp_path / "ensemble")
    second = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE)

    assert second.load_state(tmp_path / "ensemble", paths, {})

    table = study.tables[(GROUP, "s0")]
    pd.testing.assert_frame_equal(second.apply(table), first.apply(table))


def test_an_ensemble_naming_a_run_with_no_thresholds_does_not_fall_back_to_the_argmax(
    tmp_path: Path,
) -> None:
    ensemble = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE)

    with pytest.raises(FileNotFoundError, match="holds no 'thresholds.parquet'"):
        ensemble.load_state(tmp_path / "ensemble", {}, {})


def test_an_ensemble_whose_default_class_the_tuning_tuned_as_a_class_is_refused(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path, classes=3)
    _, run = _tuned(study)
    ensemble = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE, default_class=1)

    with pytest.raises(ValueError, match="has the tuned threshold"):
        ensemble.load_state(
            tmp_path / "ensemble", {"decision_threshold": run / THRESHOLDS_FILE}, {}
        )


def _recorded_inputs(run: Path, *inputs: tuple[str, str]) -> None:
    """Write the ``params.json`` that ``run_feature`` leaves beside a run's state."""
    document = {"_inputs": [{"feature": f, "run_id": r} for f, r in inputs]}
    (run / "params.json").write_text(json.dumps(document))


def test_an_ensemble_over_the_run_that_was_tuned_applies_its_thresholds(
    tmp_path: Path,
) -> None:
    _, run = _tuned(_Study(tmp_path))
    _recorded_inputs(run, ("xgboost", "0.4-aaa"))
    ensemble = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE)

    assert not ensemble.load_state(
        tmp_path / "ensemble", {"decision_threshold": run / THRESHOLDS_FILE}, {}
    )


def test_an_ensemble_of_other_runs_than_the_tuned_one_is_refused(
    tmp_path: Path,
) -> None:
    """The estimates of the report describe the output of what was tuned, not a blend."""
    _, run = _tuned(_Study(tmp_path))
    _recorded_inputs(run, ("xgboost", "0.4-bbb"))
    ensemble = _ensemble(decision_threshold=_THRESHOLDS_REFERENCE)

    with pytest.raises(ValueError, match=r"tuned the thresholds on xgboost 0.4-bbb"):
        ensemble.load_state(
            tmp_path / "ensemble", {"decision_threshold": run / THRESHOLDS_FILE}, {}
        )


def test_an_ensemble_blending_more_than_the_tuned_run_is_refused(
    tmp_path: Path,
) -> None:
    _, run = _tuned(_Study(tmp_path))
    _recorded_inputs(run, ("xgboost", "0.4-aaa"))
    blend = ProbabilityEnsembleFeature(
        ProbabilityEnsembleFeature.Inputs(
            (
                Result(feature="xgboost", run_id="0.4-aaa"),
                Result(feature="xgboost", run_id="0.4-bbb"),
            )
        ),
        params={"default_class": 0, "decision_threshold": _THRESHOLDS_REFERENCE},
    )

    with pytest.raises(ValueError, match="blends xgboost 0.4-aaa, xgboost 0.4-bbb"):
        blend.load_state(
            tmp_path / "ensemble", {"decision_threshold": run / THRESHOLDS_FILE}, {}
        )


def test_an_ensemble_still_takes_a_float_and_a_mapping() -> None:
    assert _ensemble(decision_threshold=0.3).params.decision_threshold == 0.3
    mapping = _ensemble(decision_threshold={"0": 1.0, "1": 0.2}).params
    assert mapping.decision_threshold == {0: 1.0, 1: 0.2}
