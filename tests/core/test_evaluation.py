"""Scoring a run against ground truth: the scorers and the row alignment.

The cluster scorer reported a Hungarian accuracy that a clustering finer than the
vocabulary cannot score well on, counted unannotated frames as class 0, and keyed
its entries by sequence alone, so two groups sharing a sequence name read each
other's labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset
import pytest

from mosaic.core.evaluation import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    ClassScore,
    EvaluationReport,
    aggregate_reports,
    align_predictions_to_labels,
    bootstrap_sequences,
    classification_scores,
    cluster_contingency,
    cluster_label_scores,
    entry_counts,
    majority_mapping,
    perspective_consistency,
    pool_entry_counts,
    pool_reports,
    select_scored_rows,
)
from mosaic.core.pipeline.types import ResultColumn
from tests.helpers import (
    PairClusterFeature,
    add_labels,
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
)


def test_majority_accuracy_is_purity_and_k_above_classes_is_not_one_to_one() -> None:
    """Four clean clusters of two classes: every cluster is right, two pairings are."""
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 2, 2, 3, 3])

    scores = cluster_label_scores(y_true, y_pred)

    assert scores["majority_accuracy"] == scores["purity"] == 1.0
    assert scores["hungarian_one_to_one"] is False
    hungarian = scores["hungarian_accuracy"]
    assert isinstance(hungarian, float) and hungarian < 1.0
    assert scores["majority_mapping"] == {0: 0, 1: 0, 2: 1, 3: 1}
    assert majority_mapping(y_true, y_pred) == {0: 0, 1: 0, 2: 1, 3: 1}


def test_unlabeled_rows_are_dropped_and_counted() -> None:
    scores = cluster_label_scores(np.array([-1, -1, 0, 1]), np.array([7, 7, 0, 1]))

    assert scores["n_unlabeled"] == 2
    assert scores["n_samples"] == 2
    assert scores["n_clusters"] == 2, "a cluster seen only on unlabeled rows is out"
    assert scores["purity"] == 1.0


def test_perspective_consistency_counts_both_rows_of_a_pair() -> None:
    """Frame 0 agrees, frame 1 does not, frame 2 holds one perspective only."""
    frame = pd.DataFrame(
        {
            "group": "g",
            "sequence": "s",
            "frame": [0, 0, 1, 1, 2],
            "id1": [0, 1, 0, 1, 0],
            "id2": [1, 0, 1, 0, 1],
            "perspective": [0, 1, 0, 1, 0],
            "cluster": [3, 3, 4, 5, 6],
        }
    )

    assert perspective_consistency(frame, "cluster") == (0.5, 2)
    assert perspective_consistency(frame.drop(columns="perspective"), "cluster") is None


def _two_groups_one_sequence(tmp_path: Path) -> tuple[Dataset, ResultColumn]:
    """Groups ``g1`` and ``g2`` both hold sequence ``s``, labeled differently."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g1", "s"), ("g2", "s"), n_rows=3, n_ids=2)
    for group, label in (("g1", 1), ("g2", 2)):
        _ = add_labels(ds, group, "s", [(f, 0, 1, label) for f in range(3)])
    run_id = ds.run_feature(PairClusterFeature()).run_id
    column = ResultColumn(
        feature="prov-pair-cluster__from__tracks", run_id=run_id, column="cluster"
    )
    return ds, column


def test_alignment_keys_by_group_and_sequence(tmp_path: Path) -> None:
    """Each group's rows take that group's labels, though the sequence name is shared."""
    ds, column = _two_groups_one_sequence(tmp_path)

    aligned = align_predictions_to_labels(ds, column)

    labels = aligned.frame.groupby("group")[LABEL_COLUMN].unique()
    assert labels["g1"].tolist() == [1]
    assert labels["g2"].tolist() == [2]
    assert aligned.n_rows == 12
    assert aligned.n_unlabeled == 6, "the intruder-focal rows carry no label"
    assert aligned.n_missing_prediction == 0
    assert set(aligned.frame["perspective"]) == {0}
    assert PREDICTION_COLUMN in aligned.frame.columns


def test_a_perspective_filter_counts_only_its_rows(tmp_path: Path) -> None:
    ds, column = _two_groups_one_sequence(tmp_path)

    aligned = align_predictions_to_labels(ds, column, perspective=1)

    assert aligned.n_rows == 6
    assert aligned.n_unlabeled == 6
    assert aligned.frame.empty


def test_a_reconverted_dataset_scores_the_tracks_variant_the_run_read(
    tmp_path: Path,
) -> None:
    """Two tracks variants of one entry make reading its rows ambiguous.

    The run being scored chose one when it ran, so the alignment reads that one
    rather than refusing, with no keyword the caller could answer it by.
    """
    ds = make_dataset(tmp_path / "ds")
    first, second = "convert-a.0.1-aaaaaaaaaa", "convert-b.0.1-bbbbbbbbbb"
    add_tracks_variant(ds, first, "s", n_rows=3)
    add_tracks_variant(ds, second, "s", n_rows=3)
    _ = add_labels(ds, "", "s", [(f, 0, 1, 1) for f in range(3)])
    run_id = ds.run_feature(PairClusterFeature(), tracks_run_id=first).run_id
    column = ResultColumn(
        feature="prov-pair-cluster__from__tracks", run_id=run_id, column="cluster"
    )

    inferred = align_predictions_to_labels(ds, column, perspective=0)
    named = align_predictions_to_labels(ds, column, perspective=0, tracks_run_id=first)

    assert inferred.frame[LABEL_COLUMN].tolist() == [1, 1, 1]
    assert inferred.frame.equals(named.frame)


# --- classification scores --------------------------------------------------


def test_classification_scores_per_class_and_macro_subset() -> None:
    """Macro F1 averages over the classes asked for, such as all but ``other``."""
    true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])

    scores = classification_scores(true, pred, average_over=[0, 1, 2])

    per_class = {row["label"]: row for row in scores["per_class"]}
    assert scores["accuracy"] == pytest.approx(6 / 8)
    assert per_class[0]["precision"] == pytest.approx(1 / 2)
    assert per_class[0]["recall"] == pytest.approx(1 / 2)
    assert per_class[1]["precision"] == pytest.approx(2 / 3)
    assert per_class[1]["recall"] == pytest.approx(1.0)
    f1 = {k: per_class[k]["f1"] for k in (0, 1, 2)}
    assert scores["macro_f1"] == pytest.approx(np.mean(list(f1.values())))
    assert scores["average_over"] == [0, 1, 2]


def test_confusion_rows_are_true_columns_predicted() -> None:
    scores = classification_scores(
        np.array([0, 0, 1]), np.array([1, 1, 1]), classes=[0, 1, 2]
    )

    assert scores["classes"] == [0, 1, 2]
    assert np.array_equal(
        np.asarray(scores["confusion"]), [[0, 2, 0], [0, 1, 0], [0, 0, 0]]
    )


def test_unlabeled_rows_are_dropped_and_counted_by_the_classifier_score() -> None:
    scores = classification_scores(np.array([-1, 0, 1, -1]), np.array([0, 0, 1, 1]))

    assert (scores["n_samples"], scores["n_unlabeled"]) == (2, 2)
    assert scores["accuracy"] == 1.0


# --- which rows a pair output scores --------------------------------------------


def _pair_rows() -> pd.DataFrame:
    """Two frames of one pair, each written focal on either individual."""
    return pd.DataFrame(
        {
            "group": ["g"] * 4,
            "sequence": ["s"] * 4,
            "frame": [0, 0, 1, 1],
            "id1": [0, 1, 0, 1],
            "id2": [1, 0, 1, 0],
            "perspective": [0, 1, 0, 1],
            PREDICTION_COLUMN: [1, 0, 0, 0],
            LABEL_COLUMN: [1, -1, 0, -1],
            "prob_0": [0.4, 0.9, 0.6, 0.7],
            "prob_1": [0.6, 0.1, 0.4, 0.3],
        }
    )


def test_select_rows_by_perspective() -> None:
    rows = _pair_rows()

    resident = select_scored_rows(rows, perspective=0)
    pooled = select_scored_rows(rows, perspective="pooled")

    assert resident.frame["id1"].tolist() == [0, 0]
    assert (resident.n_rows, len(pooled.frame)) == (4, 4)


def test_mean_probabilities_averages_perspectives() -> None:
    """Frame 0: mean P(1) is 0.35, so the pair is class 0 against its label 1."""
    scored = select_scored_rows(_pair_rows(), perspective="mean_probabilities")

    assert scored.frame[PREDICTION_COLUMN].tolist() == [0, 0]
    assert scored.frame[LABEL_COLUMN].tolist() == [1, 0]
    assert scored.frame["prob_1"].tolist() == pytest.approx([0.35, 0.35])
    assert scored.n_conflicting == 0


def test_mean_probabilities_drops_a_pair_whose_rows_disagree_on_the_label() -> None:
    rows = _pair_rows()
    rows.loc[1, LABEL_COLUMN] = 0

    scored = select_scored_rows(rows, perspective="mean_probabilities")

    assert scored.n_conflicting == 1
    assert scored.frame["frame"].tolist() == [1]


def test_multi_perspective_input_without_a_choice_is_refused() -> None:
    with pytest.raises(ValueError, match="more than one perspective"):
        _ = select_scored_rows(_pair_rows(), perspective=None)

    rows = _pair_rows()
    one = rows.loc[rows["perspective"] == 0]
    assert len(select_scored_rows(one, perspective=None).frame) == 2


def test_majority_map_ties_break_to_the_lowest_class() -> None:
    assert majority_mapping(np.array([2, 1, 0, 0]), np.array([5, 5, 6, 6])) == {
        5: 1,
        6: 0,
    }


def test_the_contingency_has_a_row_per_cluster_and_a_column_per_class() -> None:
    """Unlabeled rows are no class, and a cluster they alone fill has no row."""
    y_true = np.array([0, 0, 1, 1, 1, -1, -1])
    y_pred = np.array([3, 3, 3, 4, 4, 4, 9])

    contingency = cluster_contingency(y_true, y_pred)

    assert contingency.to_dict("list") == {
        "cluster": [3, 4],
        "class_0": [2, 0],
        "class_1": [1, 2],
    }
    assert np.array_equal(
        contingency[["class_0", "class_1"]].to_numpy().T,
        np.asarray(cluster_label_scores(y_true, y_pred)["confusion"]),
    )


def test_class_balanced_weighting_follows_the_share_of_a_class_not_the_count() -> None:
    """Class 1 is rare: cluster 5 holds all of it, and more rows of the common class."""
    y_true = np.array([0] * 30 + [1] * 8 + [0] * 62)
    y_pred = np.array([5] * 38 + [6] * 62)

    assert majority_mapping(y_true, y_pred) == {5: 0, 6: 0}
    assert majority_mapping(y_true, y_pred, weighting="count") == {5: 0, 6: 0}
    assert majority_mapping(y_true, y_pred, weighting="class_balanced") == {
        5: 1,
        6: 0,
    }


def test_class_balanced_ties_break_to_the_lowest_class_and_skip_unlabeled() -> None:
    """Cluster 9 holds half of each class's four rows. Unlabeled rows are no class."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1])
    y_pred = np.array([1, 1, 9, 9, 3, 3, 9, 9, 9, 9, 7])

    mapping = majority_mapping(y_true, y_pred, weighting="class_balanced")

    assert mapping == {1: 0, 3: 1, 9: 0}


def test_an_unknown_weighting_is_refused() -> None:
    with pytest.raises(ValueError, match="weighting"):
        _ = majority_mapping(
            np.array([0, 1]),
            np.array([0, 1]),
            weighting="prevalence",  # pyright: ignore[reportArgumentType]
        )


# --- the report, pooled and aggregated -------------------------------------------


def _report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average_over: list[int] | None = None,
    n_entries: int = 1,
) -> EvaluationReport:
    """The report a classifier run over *y_true* and *y_pred* writes."""
    scores = classification_scores(y_true, y_pred, average_over=average_over)
    return EvaluationReport(
        mode="classifier",
        n_entries=n_entries,
        n_scored=scores["n_samples"],
        n_unlabeled=scores["n_unlabeled"],
        n_missing_prediction=0,
        accuracy=scores["accuracy"],
        macro_f1=scores["macro_f1"],
        average_over=scores["average_over"],
        classes=scores["classes"],
        per_class=[ClassScore(**row) for row in scores["per_class"]],
        confusion=scores["confusion"].tolist(),
    )


def _noisy(
    seed: int, n: int, n_classes: int = 4, error: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """Labels of uneven prevalence, and predictions that are wrong at rate *error*."""
    rng = np.random.default_rng(seed)
    weights = np.arange(n_classes, 0, -1, dtype=float)
    y_true = rng.choice(n_classes, size=n, p=weights / weights.sum())
    wrong = rng.random(n) < error
    y_pred = np.where(wrong, rng.integers(0, n_classes, size=n), y_true)
    return y_true, y_pred


def test_a_report_written_before_label_map_source_still_loads() -> None:
    old = _report(np.array([0, 1]), np.array([0, 1])).model_dump(mode="json")
    del old["label_map_source"]

    report = EvaluationReport.model_validate_json(json.dumps(old))

    assert report.label_map_source == ""


def test_pooling_reports_of_disjoint_entries_equals_scoring_them_together() -> None:
    """Folds with different classes pool to the report of all their rows at once."""
    true_a, pred_a = _noisy(1, 300)
    true_b, pred_b = _noisy(2, 200, n_classes=3)
    true_c, pred_c = _noisy(3, 250, n_classes=5)
    folds = [(true_a, pred_a), (true_b, pred_b), (true_c, pred_c)]

    pooled = pool_reports([_report(*fold) for fold in folds])
    whole = _report(
        np.concatenate([fold[0] for fold in folds]),
        np.concatenate([fold[1] for fold in folds]),
        n_entries=3,
    )

    assert pooled == whole


def test_pooling_reports_that_list_other_classes_orders_them_as_one_run_would() -> None:
    """A fold with classes 0 and 2 and one with 0 and 1 pool to the order 0, 1, 2."""
    true_a, pred_a = np.array([0, 2, 2, 0]), np.array([0, 2, 0, 0])
    true_b, pred_b = np.array([0, 1, 1]), np.array([0, 1, 0])

    pooled = pool_reports([_report(true_a, pred_a), _report(true_b, pred_b)])
    whole = _report(
        np.concatenate([true_a, true_b]), np.concatenate([pred_a, pred_b]), n_entries=2
    )

    assert pooled.classes == [0, 1, 2]
    assert pooled == whole


def test_pooling_reports_keeps_the_average_over_they_share() -> None:
    true_a, pred_a = _noisy(4, 200)
    true_b, pred_b = _noisy(5, 200)
    reports = [
        _report(true_a, pred_a, average_over=[2, 0, 1]),
        _report(true_b, pred_b, average_over=[2, 0, 1]),
    ]

    pooled = pool_reports(reports)

    assert pooled.average_over == [2, 0, 1]
    whole = _report(
        np.concatenate([true_a, true_b]),
        np.concatenate([pred_a, pred_b]),
        average_over=[2, 0, 1],
        n_entries=2,
    )
    assert pooled.macro_f1 == whole.macro_f1


def test_pooling_refuses_nothing_and_mixed_modes() -> None:
    clusters = _report(np.array([0, 1]), np.array([0, 1])).model_copy(
        update={"mode": "clusters"}
    )

    with pytest.raises(ValueError, match="at least one report"):
        _ = pool_reports([])
    with pytest.raises(ValueError, match="mode"):
        _ = pool_reports([_report(np.array([0]), np.array([0])), clusters])


def test_pooling_cluster_reports_drops_what_cannot_be_added() -> None:
    """ARI and purity are not additive, and a map fitted per fold is not one map."""
    first = _report(np.array([0, 1]), np.array([0, 1])).model_copy(
        update={
            "mode": "clusters",
            "clusters": {"ARI": 1.0},
            "label_map": {7: 0, 9: 1},
            "label_map_source": "fit-label-map",
        }
    )
    second = first.model_copy(update={"label_map": {7: 1, 9: 0}})

    pooled = pool_reports([first, second])

    assert pooled.clusters == {}
    assert pooled.label_map == {}
    assert pooled.label_map_source == "fit-label-map"
    assert pool_reports([first, first]).label_map == {7: 0, 9: 1}


def test_aggregating_reports_gives_mean_and_sd_per_class_and_for_macro_f1() -> None:
    reports = [_report(*_noisy(seed, 150, error=0.2 + 0.1 * seed)) for seed in range(3)]

    spread = aggregate_reports(reports)

    macro = [report.macro_f1 for report in reports]
    assert spread["n_reports"] == 3
    assert spread["macro_f1"]["mean"] == pytest.approx(np.mean(macro))
    assert spread["macro_f1"]["sd"] == pytest.approx(np.std(macro, ddof=1))
    accuracy = [report.accuracy for report in reports]
    assert spread["accuracy"]["sd"] == pytest.approx(np.std(accuracy, ddof=1))
    f1_of_zero = [report.per_class[0].f1 for report in reports]
    first = spread["per_class"][0]
    assert first["label"] == 0
    assert first["n_reports"] == 3
    assert first["f1"]["mean"] == pytest.approx(np.mean(f1_of_zero))
    assert first["f1"]["sd"] == pytest.approx(np.std(f1_of_zero, ddof=1))


def test_aggregating_averages_a_class_over_the_reports_that_list_it() -> None:
    """A class one fold never saw is not a zero in the mean of the others."""
    with_class = _report(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]))
    without = _report(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))

    spread = aggregate_reports([with_class, without])

    by_label = {row["label"]: row for row in spread["per_class"]}
    assert by_label[1]["n_reports"] == 1
    assert by_label[1]["f1"]["mean"] == with_class.per_class[1].f1
    assert np.isnan(by_label[1]["f1"]["sd"])
    assert by_label[0]["n_reports"] == 2


def test_a_single_report_has_no_sd() -> None:
    spread = aggregate_reports([_report(np.array([0, 1]), np.array([0, 1]))])

    assert spread["macro_f1"]["mean"] == 1.0
    assert np.isnan(spread["macro_f1"]["sd"])


# --- per-entry counts ---------------------------------------------------------------


def _entry_table(
    true: np.ndarray, pred: np.ndarray, sizes: list[int], classes: list[int]
) -> pd.DataFrame:
    """One row per entry of the counts ``evaluate-labels`` writes, from slices of the rows."""
    rows: list[dict[str, int]] = []
    start = 0
    for size in sizes:
        scores = classification_scores(
            true[start : start + size], pred[start : start + size], classes=classes
        )
        rows.append({"n_scored": scores["n_samples"], **entry_counts(scores, classes)})
        start += size
    return pd.DataFrame(rows)


def test_entry_counts_name_each_class_once() -> None:
    scores = classification_scores(
        np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 2]), classes=[0, 1, 2]
    )

    assert entry_counts(scores, [0, 1, 2]) == {
        "tp_0": 1,
        "fp_0": 0,
        "fn_0": 1,
        "tp_1": 2,
        "fp_1": 1,
        "fn_1": 1,
        "tp_2": 0,
        "fp_2": 1,
        "fn_2": 0,
    }


def test_a_pooled_score_from_entry_counts_equals_the_pooled_rows_exactly() -> None:
    """Exactly, not approximately: both take the same sums through the same formula."""
    true, pred = _noisy(7, 900, n_classes=5, error=0.45)
    sizes = [100, 250, 50, 300, 200]
    classes = [0, 1, 2, 3, 4]

    pooled = pool_entry_counts(_entry_table(true, pred, sizes, classes))
    whole = classification_scores(true, pred, classes=classes)

    assert pooled["macro_f1"] == whole["macro_f1"]
    assert pooled["accuracy"] == whole["accuracy"]
    assert pooled["average_over"] == whole["average_over"]
    assert pooled["per_class"] == whole["per_class"]


def test_pooling_counts_takes_average_over_from_the_caller() -> None:
    true, pred = _noisy(8, 400, n_classes=4)
    table = _entry_table(true, pred, [200, 200], [0, 1, 2, 3])

    pooled = pool_entry_counts(table, average_over=[1, 2])

    whole = classification_scores(true, pred, classes=[0, 1, 2, 3], average_over=[1, 2])
    assert pooled["macro_f1"] == whole["macro_f1"]


def test_a_table_without_counts_is_refused() -> None:
    with pytest.raises(ValueError, match="tp_<class>"):
        _ = pool_entry_counts(pd.DataFrame({"n_scored": [3]}))
    with pytest.raises(ValueError, match="fp_1"):
        _ = pool_entry_counts(pd.DataFrame({"tp_1": [1], "fn_1": [0]}))


def _bootstrap_table() -> pd.DataFrame:
    true, pred = _noisy(9, 1200, n_classes=3, error=0.3)
    return _entry_table(true, pred, [100] * 12, [0, 1, 2])


def test_a_bootstrap_over_entries_is_repeatable_from_its_random_state() -> None:
    table = _bootstrap_table()

    first = bootstrap_sequences(table, n_boot=200, random_state=3)
    again = bootstrap_sequences(table, n_boot=200, random_state=3)
    other = bootstrap_sequences(table, n_boot=200, random_state=4)

    assert first["macro_f1"] == again["macro_f1"]
    assert np.array_equal(
        first["replicates"]["macro_f1"], again["replicates"]["macro_f1"]
    )
    assert not np.array_equal(
        first["replicates"]["macro_f1"], other["replicates"]["macro_f1"]
    )


def test_a_bootstrap_spanning_several_draw_chunks_keeps_its_exact_result() -> None:
    """600 entries and 8,000 resamples draw more than one chunk of the multinomial."""
    true, pred = _noisy(5, 3000, n_classes=3, error=0.3)
    table = _entry_table(true, pred, [5] * 600, [0, 1, 2])

    scores = bootstrap_sequences(table, n_boot=8000, random_state=11)

    def approx(value: float) -> object:
        return pytest.approx(value, rel=1e-12)

    assert scores["n_entries"] == 600
    assert scores["average_over"] == [0, 1, 2]
    assert scores["accuracy"] == {
        "estimate": approx(0.784),
        "lower": approx(0.7696666666666667),
        "upper": approx(0.7983333333333333),
    }
    assert scores["macro_f1"] == {
        "estimate": approx(0.7590187466712971),
        "lower": approx(0.7428674341369436),
        "upper": approx(0.7749001695671882),
    }
    assert scores["per_class"] == [
        {
            "label": 0,
            "f1": {
                "estimate": approx(0.8329246935201402),
                "lower": approx(0.8182146729309723),
                "upper": approx(0.847067236204099),
            },
        },
        {
            "label": 1,
            "f1": {
                "estimate": approx(0.7862137862137862),
                "lower": approx(0.7669320907584135),
                "upper": approx(0.805247225025227),
            },
        },
        {
            "label": 2,
            "f1": {
                "estimate": approx(0.657917760279965),
                "lower": approx(0.62488805245008),
                "upper": approx(0.6888694483938512),
            },
        },
    ]
    replicates = scores["replicates"]
    assert replicates["accuracy"].shape == (8000,)
    assert replicates["f1"].shape == (8000, 3)
    assert float(replicates["accuracy"].sum()) == approx(6271.944666666666)
    assert float(replicates["macro_f1"].sum()) == approx(6071.311093137474)
    assert float(replicates["f1"].sum()) == approx(18213.93327941242)
    assert float(replicates["macro_f1"][-1]) == approx(0.7580767768247768)


def test_a_bootstrap_interval_brackets_the_pooled_estimate() -> None:
    table = _bootstrap_table()

    result = bootstrap_sequences(table, n_boot=500, random_state=0, confidence=0.9)

    pooled = pool_entry_counts(table)
    macro = result["macro_f1"]
    assert macro["estimate"] == pooled["macro_f1"]
    assert macro["lower"] < macro["estimate"] < macro["upper"]
    assert result["n_entries"] == 12
    assert result["confidence"] == 0.9
    assert result["replicates"]["f1"].shape == (500, 3)
    assert [row["label"] for row in result["per_class"]] == [0, 1, 2]
    for row in result["per_class"]:
        assert row["f1"]["lower"] <= row["f1"]["estimate"] <= row["f1"]["upper"]


def test_a_bootstrap_of_identical_entries_has_no_width() -> None:
    row = {"tp_0": 8, "fp_0": 2, "fn_0": 1, "tp_1": 5, "fp_1": 1, "fn_1": 2}
    table = pd.DataFrame([row] * 6)

    result = bootstrap_sequences(table, n_boot=50, random_state=1)

    macro = result["macro_f1"]
    assert macro["lower"] == macro["estimate"] == macro["upper"]


def test_a_bootstrap_resamples_only_the_entries_that_scored_rows() -> None:
    table = _bootstrap_table()
    empty = pd.DataFrame([{column: 0 for column in table.columns}] * 3)

    plain = bootstrap_sequences(table, n_boot=100, random_state=5)
    padded = bootstrap_sequences(
        pd.concat([table, empty], ignore_index=True), n_boot=100, random_state=5
    )

    assert padded["n_entries"] == 12
    assert padded["macro_f1"] == plain["macro_f1"]


def test_counts_of_runs_that_listed_other_classes_are_refused_until_filled() -> None:
    """A concatenation leaves NaN where a run listed no such class. It is not a count."""
    first = pd.DataFrame({"tp_0": [3], "fp_0": [1], "fn_0": [1]})
    second = pd.DataFrame(
        {
            "tp_0": [2],
            "fp_0": [0],
            "fn_0": [0],
            "tp_1": [4],
            "fp_1": [1],
            "fn_1": [0],
        }
    )
    both = pd.concat([first, second], ignore_index=True)

    with pytest.raises(ValueError, match="tp_1"):
        _ = pool_entry_counts(both)
    with pytest.raises(ValueError, match="missing"):
        _ = bootstrap_sequences(both, n_boot=5, random_state=0)
    pooled = pool_entry_counts(both.fillna(0))
    assert [row["support"] for row in pooled["per_class"]] == [6, 4]


def _hit_or_miss_table() -> pd.DataFrame:
    """One entry that scored one row right, and one that scored one row wrong."""
    return pd.DataFrame(
        [
            {"tp_0": 1, "fp_0": 0, "fn_0": 0, "tp_1": 0, "fp_1": 0, "fn_1": 0},
            {"tp_0": 0, "fp_0": 0, "fn_0": 0, "tp_1": 0, "fp_1": 0, "fn_1": 1},
        ]
    )


def test_a_bootstrap_draws_as_many_entries_as_it_has() -> None:
    """Two entries, one right and one wrong: a resample of two is 0, 1 or 2 right."""
    result = bootstrap_sequences(_hit_or_miss_table(), n_boot=4000, random_state=0)

    accuracy = result["replicates"]["accuracy"]
    values, counts = np.unique(accuracy, return_counts=True)
    assert values.tolist() == [0.0, 0.5, 1.0]
    assert (counts / accuracy.size).tolist() == pytest.approx(
        [0.25, 0.5, 0.25], abs=0.03
    )
    assert accuracy.mean() == pytest.approx(result["accuracy"]["estimate"], abs=0.03)


def test_a_bootstrap_interval_is_the_percentile_of_its_replicates() -> None:
    table = _bootstrap_table()

    result = bootstrap_sequences(table, n_boot=300, random_state=2, confidence=0.8)

    replicates = result["replicates"]["macro_f1"]
    assert result["macro_f1"]["lower"] == np.quantile(replicates, 0.1)
    assert result["macro_f1"]["upper"] == np.quantile(replicates, 0.9)
    f1_of_zero = result["replicates"]["f1"][:, 0]
    assert result["per_class"][0]["f1"]["lower"] == np.quantile(f1_of_zero, 0.1)


def test_a_bootstrap_averages_the_classes_it_was_given_in_every_resample() -> None:
    """Class 1 is absent from some resamples and is not the macro average's to drop."""
    table = _hit_or_miss_table()

    only_zero = bootstrap_sequences(table, n_boot=200, random_state=3, average_over=[0])
    both = bootstrap_sequences(table, n_boot=200, random_state=3)

    assert np.array_equal(
        only_zero["replicates"]["macro_f1"], only_zero["replicates"]["f1"][:, 0]
    )
    assert only_zero["average_over"] == [0]
    assert np.array_equal(
        both["replicates"]["macro_f1"], both["replicates"]["f1"].mean(axis=1)
    )
    assert both["average_over"] == [0, 1]


def test_a_bootstrap_needs_entries_and_a_sensible_confidence() -> None:
    table = _bootstrap_table()

    with pytest.raises(ValueError, match="no entry scored"):
        _ = bootstrap_sequences(table.iloc[0:0], n_boot=10, random_state=0)
    with pytest.raises(ValueError, match="confidence"):
        _ = bootstrap_sequences(table, n_boot=10, random_state=0, confidence=1.0)
    with pytest.raises(ValueError, match="n_boot"):
        _ = bootstrap_sequences(table, n_boot=0, random_state=0)
