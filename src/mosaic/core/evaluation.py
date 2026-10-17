"""Scoring a run's predictions against ground-truth labels.

Two halves, kept apart so either can be used without the other:

- **Scorers** take arrays or frames and return plain dicts. They read no dataset,
  so a feature's ``apply`` can call them on the rows it was handed.
- **Alignment** reads a run's output and the labels of the same entries from a
  dataset and pairs them row by row (:func:`align_predictions_to_labels`).

Both drop a row whose label is :data:`~mosaic.core.helpers.UNLABELED` and count
it, so a score never treats "nobody annotated this frame" as a class, and a
caller can report how many rows it lost.

sklearn, scipy and the pipeline engine are imported inside the functions that
use them, so importing this module stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .helpers import UNLABELED

if TYPE_CHECKING:
    from .dataset import Dataset
    from .pipeline.types import NNResult, ResultColumn
    from .scope import Scope

__all__ = [
    "PREDICTION_COLUMN",
    "LABEL_COLUMN",
    "AlignedLabels",
    "align_predictions_to_labels",
    "cluster_label_scores",
    "majority_mapping",
    "perspective_consistency",
]

PREDICTION_COLUMN = "prediction"
"""The column of :attr:`AlignedLabels.frame` holding the run's value."""

LABEL_COLUMN = "label"
"""The column of :attr:`AlignedLabels.frame` holding the ground-truth class."""


# --- Scorers -----------------------------------------------------------------


def _as_int_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.int64).ravel()
    pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return true, pred


def _contingency(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(classes, clusters, counts)``, with rows the classes and columns the clusters."""
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    counts = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(
        counts, (np.searchsorted(classes, y_true), np.searchsorted(clusters, y_pred)), 1
    )
    return classes, clusters, counts


def majority_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, int]:
    """Each cluster's most frequent class, as ``{cluster: class}``.

    Many clusters may map to one class, which is what a clustering finer than the
    behavior vocabulary needs. Rows labeled :data:`~mosaic.core.helpers.UNLABELED`
    are left out. A tie goes to the lower class id.

    Args:
        y_true: Ground-truth class per row.
        y_pred: Cluster per row.

    Returns:
        The mapping, empty when no labeled row remains.
    """
    true, pred = _as_int_arrays(y_true, y_pred)
    labeled = true != UNLABELED
    true, pred = true[labeled], pred[labeled]
    if true.size == 0:
        return {}
    classes, clusters, counts = _contingency(true, pred)
    best = counts.argmax(axis=0)
    return {int(clusters[c]): int(classes[best[c]]) for c in range(clusters.size)}


def cluster_label_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """How well a clustering agrees with ground-truth classes.

    Rows labeled :data:`~mosaic.core.helpers.UNLABELED` are dropped first and
    counted in ``n_unlabeled``.

    **Which accuracy to read.** ``majority_accuracy`` maps each cluster to its
    most frequent class (``majority_mapping``) and is the same number as
    ``purity``. ``hungarian_accuracy`` pairs classes and clusters one to one, so
    when there are more clusters than classes every cluster beyond the pairing
    counts as wrong, and a clustering that splits each behavior finely scores
    low however clean its clusters are. ``hungarian_one_to_one`` says whether the
    pairing covered every cluster; when it is False, read
    ``majority_accuracy``.

    Args:
        y_true: Ground-truth class per row.
        y_pred: Cluster per row.

    Returns:
        ``n_samples``, ``n_unlabeled``, ``n_classes``, ``n_clusters``, ``ARI``,
        ``AMI``, ``homogeneity``, ``completeness``, ``v_measure``, ``purity``,
        ``majority_accuracy``, ``majority_mapping`` (cluster to class),
        ``hungarian_accuracy``, ``mapping`` (the Hungarian class to cluster
        pairing), ``hungarian_one_to_one`` and ``confusion`` (rows the classes,
        columns the clusters, both ascending).
    """
    from scipy.optimize import linear_sum_assignment

    # sklearn's metric signatures leave their array parameters untyped.
    from sklearn.metrics import (
        adjusted_mutual_info_score,  # pyright: ignore[reportUnknownVariableType]
        adjusted_rand_score,  # pyright: ignore[reportUnknownVariableType]
        homogeneity_completeness_v_measure,  # pyright: ignore[reportUnknownVariableType]
    )

    true, pred = _as_int_arrays(y_true, y_pred)
    labeled = true != UNLABELED
    n_unlabeled = int((~labeled).sum())
    true, pred = true[labeled], pred[labeled]

    n_samples = int(true.shape[0])
    if n_samples == 0:
        return {
            "n_samples": 0,
            "n_unlabeled": n_unlabeled,
            "n_classes": 0,
            "n_clusters": 0,
            "ARI": 0.0,
            "AMI": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "purity": 0.0,
            "majority_accuracy": 0.0,
            "majority_mapping": {},
            "hungarian_accuracy": 0.0,
            "mapping": {},
            "hungarian_one_to_one": True,
            "confusion": np.zeros((1, 1), dtype=np.int64),
        }

    classes, clusters, counts = _contingency(true, pred)
    total = int(counts.sum())
    purity = float(counts.max(axis=0).sum()) / total

    rows, cols = linear_sum_assignment(counts.max() - counts)
    hungarian = float(counts[rows, cols].sum()) / total
    pairing = {int(classes[r]): int(clusters[c]) for r, c in zip(rows, cols)}
    best = counts.argmax(axis=0)

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true, pred
    )
    return {
        "n_samples": n_samples,
        "n_unlabeled": n_unlabeled,
        "n_classes": int(classes.size),
        "n_clusters": int(clusters.size),
        "ARI": float(adjusted_rand_score(true, pred)),
        "AMI": float(adjusted_mutual_info_score(true, pred)),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "purity": purity,
        "majority_accuracy": purity,
        "majority_mapping": {
            int(clusters[c]): int(classes[best[c]]) for c in range(clusters.size)
        },
        "hungarian_accuracy": hungarian,
        "mapping": pairing,
        "hungarian_one_to_one": bool(clusters.size <= classes.size),
        "confusion": counts,
    }


_PAIR_KEY = ["group", "sequence", "frame", "_low", "_high"]


def perspective_consistency(
    frame: pd.DataFrame, column: str
) -> tuple[float, int] | None:
    """How often the two perspectives of one pair at one frame agree on *column*.

    A pair feature writes each pair twice per frame, once focal on each
    individual (``perspective``). For a quantity that describes the pair rather
    than the focal, the two rows should agree; a clustering that splits them is
    describing the focal.

    Args:
        frame: Pair rows with ``group``, ``sequence``, ``frame``, ``id1``,
            ``id2`` and ``perspective``.
        column: The column to compare, such as a cluster.

    Returns:
        ``(fraction, n_pairs)``: the share of ``(group, sequence, frame, pair)``
        keys holding both perspectives whose rows agree, and how many such keys
        there are (the fraction is NaN when there are none). ``None`` when
        *frame* has no ``perspective`` column.
    """
    if "perspective" not in frame.columns:
        return None
    ids = frame[["id1", "id2"]].to_numpy()
    keyed = pd.DataFrame(
        {
            "group": frame["group"].to_numpy(),
            "sequence": frame["sequence"].to_numpy(),
            "frame": frame["frame"].to_numpy(),
            "_low": ids.min(axis=1),
            "_high": ids.max(axis=1),
            "perspective": frame["perspective"].to_numpy(),
            "value": frame[column].to_numpy(),
        }
    )
    per_key = keyed.groupby(_PAIR_KEY, sort=False).agg(
        perspectives=("perspective", "nunique"),
        values=("value", "nunique"),
    )
    both = per_key[per_key["perspectives"] >= 2]
    n_pairs = int(both.shape[0])
    if n_pairs == 0:
        return float("nan"), 0
    return float((both["values"] == 1).sum()) / n_pairs, n_pairs


# --- Alignment ---------------------------------------------------------------


@dataclass(frozen=True)
class AlignedLabels:
    """A run's predictions beside the labels of the same rows.

    Attributes:
        frame: One row per aligned output row: the identity columns the run
            wrote (``group``, ``sequence``, ``frame``, and ``id`` or ``id1``,
            ``id2`` and ``perspective``), :data:`PREDICTION_COLUMN` and
            :data:`LABEL_COLUMN`.
        n_rows: Rows read, after the perspective filter and before any drop.
        n_unlabeled: Rows whose label is ``-1``. Dropped from *frame* unless the
            caller kept them.
        n_missing_prediction: Rows with no prediction (NaN), always dropped.
    """

    frame: pd.DataFrame
    n_rows: int
    n_unlabeled: int
    n_missing_prediction: int


def align_predictions_to_labels(
    ds: Dataset,
    prediction: ResultColumn,
    *,
    label_kind: str = "behavior",
    scope: Scope | None = None,
    labels_run_id: str | None = None,
    pair_filter: NNResult | None = None,
    perspective: int | None = None,
    drop_unlabeled: bool = True,
) -> AlignedLabels:
    """Pair each output row of a run with the ground-truth label of that row.

    Built on :func:`~mosaic.core.pipeline.run.load_values`, so the entries are
    group-qualified, one labels variant is selected (``labels_run_id`` when the
    dataset holds two), and each row is labeled by its own frame and individuals:
    a pair row takes the label of its own direction.

    Args:
        ds: The dataset holding the run and the labels.
        prediction: The run and the column to score. Pin ``run_id`` to score a
            run other than the latest.
        label_kind: The labels kind, ``behavior`` by default.
        scope: The entries to read. ``None`` reads every entry of the run.
        labels_run_id: The labels variant, when the kind has more than one.
        pair_filter: Keep only the pair rows this nearest-neighbor run selects.
        perspective: Keep only the rows of this ``perspective``, when the run
            writes one. ``None`` keeps both.
        drop_unlabeled: Drop the rows labeled ``-1``. They are counted either
            way.

    Returns:
        The aligned rows and what was dropped.

    Raises:
        ValueError: A row with a label file has no ``frame`` to align it by.
    """
    from .pipeline.run import load_values
    from .pipeline.types import GroundTruthLabelsSource, LabelsSource

    labels = (
        GroundTruthLabelsSource()
        if label_kind == "behavior"
        else LabelsSource(kind=label_kind)
    )
    values = load_values(
        ds,
        [prediction, labels],
        scope=scope,
        pair_filter=pair_filter,
        labels_run_id=labels_run_id,
    )
    values = values.rename(
        columns={
            prediction.column: PREDICTION_COLUMN,
            f"labels-{label_kind}": LABEL_COLUMN,
        }
    )
    if perspective is not None and "perspective" in values.columns:
        values = values.loc[values["perspective"] == perspective]
    n_rows = int(values.shape[0])

    missing = values[PREDICTION_COLUMN].isna().to_numpy()
    values = values.loc[~missing]
    unlabeled = (values[LABEL_COLUMN] == UNLABELED).to_numpy()
    if drop_unlabeled:
        values = values.loc[~unlabeled]
    return AlignedLabels(
        frame=values.reset_index(drop=True),
        n_rows=n_rows,
        n_unlabeled=int(unlabeled.sum()),
        n_missing_prediction=int(missing.sum()),
    )
