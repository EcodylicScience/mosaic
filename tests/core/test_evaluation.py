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
from mosaic.core.evaluation import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    align_predictions_to_labels,
    cluster_label_scores,
    majority_mapping,
    perspective_consistency,
)
from mosaic.core.pipeline.types import ResultColumn
from tests.helpers import (
    PairClusterFeature,
    add_labels,
    add_track_sequences,
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
