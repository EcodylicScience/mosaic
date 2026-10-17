"""Scoring a run against ground truth: the scorers and the row alignment.

The cluster scorer reported a Hungarian accuracy that a clustering finer than the
vocabulary cannot score well on, counted unannotated frames as class 0, and keyed
its entries by sequence alone, so two groups sharing a sequence name read each
other's labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset
import pytest

from mosaic.core.evaluation import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    align_predictions_to_labels,
    classification_scores,
    cluster_label_scores,
    majority_mapping,
    perspective_consistency,
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
