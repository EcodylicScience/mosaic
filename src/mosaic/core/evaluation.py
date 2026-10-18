"""Scoring a run's predictions against ground-truth labels.

Three parts, kept apart so any can be used without the others:

- **Scorers** take arrays or frames and return plain dicts. They read no dataset,
  so a feature's ``apply`` can call them on the rows it was handed:
  :func:`classification_scores` for a classifier's predicted classes,
  :func:`cluster_label_scores` for a clustering, and :func:`select_scored_rows`
  for which perspective of a pair row is scored.
- **Pooling** combines scores that were taken over separate entries, with no
  dataset and no random state but the one a caller passes: :func:`pool_reports`
  and :func:`aggregate_reports` over the reports of several runs, and
  :func:`pool_entry_counts` and :func:`bootstrap_sequences` over the per-entry
  counts that ``evaluate-labels`` writes (``tp_<class>``, ``fp_<class>`` and
  ``fn_<class>`` per entry). A score pooled from the counts is the score of the
  pooled rows, to the last bit.
- **Alignment** reads a run's output and the labels of the same entries from a
  dataset and pairs them row by row (:func:`align_predictions_to_labels`).

:class:`EvaluationReport` is what the ``evaluate-labels`` feature writes, and
reads the same whichever part produced its numbers.

Both drop a row whose label is :data:`~mosaic.core.helpers.UNLABELED` and count
it, so a score never treats "nobody annotated this frame" as a class, and a
caller can report how many rows it lost.

sklearn, scipy and the pipeline engine are imported inside the functions that
use them, so importing this module stays cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import pandas as pd

from .helpers import UNLABELED
from .strict_model import StrictModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .dataset import Dataset
    from .pipeline.types import NNResult, ResultColumn
    from .scope import Scope

__all__ = [
    "PREDICTION_COLUMN",
    "LABEL_COLUMN",
    "AlignedLabels",
    "BootstrapScores",
    "ClassInterval",
    "ClassScore",
    "ClassScoreRow",
    "ClassSpread",
    "ClassificationScores",
    "CountScores",
    "EvaluationReport",
    "Interval",
    "LabelMapSource",
    "MetricSpread",
    "Perspective",
    "ReportAggregate",
    "Replicates",
    "ScoredRows",
    "Weighting",
    "aggregate_reports",
    "align_predictions_to_labels",
    "bootstrap_sequences",
    "classification_scores",
    "cluster_contingency",
    "cluster_label_scores",
    "entry_counts",
    "majority_mapping",
    "perspective_consistency",
    "pool_entry_counts",
    "pool_reports",
    "select_scored_rows",
]

type Perspective = int | Literal["pooled", "mean_probabilities"] | None
"""Which rows of a pair output are scored.

An ``int`` keeps the rows of that ``perspective``. ``"pooled"`` scores every row.
``"mean_probabilities"`` averages the class probabilities of a pair's two rows
at one frame and scores the class they then favor. ``None`` scores the rows as
they are and refuses an output holding more than one perspective.
"""

type Weighting = Literal["count", "class_balanced"]
"""How :func:`majority_mapping` picks the class a cluster stands for.

``"count"`` takes the class with the most rows in the cluster. ``"class_balanced"``
takes the class whose rows the cluster holds the largest share of.
"""

type LabelMapSource = Literal["", "scored_rows", "fit-label-map"]
"""Where the cluster to class map of an :class:`EvaluationReport` was fitted.

``""`` is no map, as for a classifier. ``"scored_rows"`` is the majority map of
the rows the report scored, which makes the accuracy the clustering's purity and
leaves the map not held out. ``"fit-label-map"`` is a map fitted on other
entries and referenced through ``label_map``.
"""

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


def majority_mapping(
    y_true: np.ndarray, y_pred: np.ndarray, *, weighting: Weighting = "count"
) -> dict[int, int]:
    """Each cluster's most frequent class, as ``{cluster: class}``.

    Many clusters may map to one class, which is what a clustering finer than the
    behavior vocabulary needs. Rows labeled :data:`~mosaic.core.helpers.UNLABELED`
    are left out. A tie goes to the lower class id.

    **Which weighting to read.** ``"count"`` follows prevalence: a cluster goes to
    whichever class has the most rows in it, so a common class takes every
    cluster it holds more rows of than a rare class does, however small a share
    of its own rows that is, and a rare behavior can end up with no cluster at
    all. ``"class_balanced"`` gives a cluster to the class for which it holds the
    largest share of that class's rows, so each class is weighted equally
    whatever its size.

    Args:
        y_true: Ground-truth class per row.
        y_pred: Cluster per row.
        weighting: See above.

    Returns:
        The mapping, empty when no labeled row remains.

    Raises:
        ValueError: *weighting* is neither of its two values.
    """
    if weighting not in ("count", "class_balanced"):
        msg = f"weighting must be 'count' or 'class_balanced', not {weighting!r}."
        raise ValueError(msg)
    true, pred = _as_int_arrays(y_true, y_pred)
    labeled = true != UNLABELED
    true, pred = true[labeled], pred[labeled]
    if true.size == 0:
        return {}
    classes, clusters, counts = _contingency(true, pred)
    if weighting == "class_balanced":
        # Every class listed holds a labeled row, so no row sums to zero.
        best = (counts / counts.sum(axis=1, keepdims=True)).argmax(axis=0)
    else:
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


def cluster_contingency(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """How many rows of each class fall in each cluster.

    Rows labeled :data:`~mosaic.core.helpers.UNLABELED` are left out, so a
    cluster seen only on them has no row. This is the ``confusion`` of
    :func:`cluster_label_scores` with its axes named and transposed.

    Args:
        y_true: Ground-truth class per row.
        y_pred: Cluster per row.

    Returns:
        One row per cluster, ascending: ``cluster``, then ``class_<k>`` for each
        class ``k`` a labeled row holds, ascending.
    """
    true, pred = _as_int_arrays(y_true, y_pred)
    labeled = true != UNLABELED
    true, pred = true[labeled], pred[labeled]
    classes, clusters, counts = _contingency(true, pred)
    frame = pd.DataFrame(
        counts.T.astype(np.int64),
        columns=pd.Index([f"class_{value}" for value in classes]),
    )
    frame.insert(0, "cluster", clusters.astype(np.int64))
    return frame


class ClassScoreRow(TypedDict):
    """One class's row of :class:`ClassificationScores`."""

    label: int
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


class ClassificationScores(TypedDict):
    """What :func:`classification_scores` returns."""

    n_samples: int
    n_unlabeled: int
    accuracy: float
    macro_f1: float
    average_over: list[int]
    classes: list[int]
    per_class: list[ClassScoreRow]
    confusion: np.ndarray


def _class_row(label: int, *, hit: int, support: int, predicted: int) -> ClassScoreRow:
    """One class's scores from its counts.

    The one formula behind :func:`classification_scores` and the pooling helpers,
    so a score pooled from counts equals the score of the pooled rows exactly.
    """
    precision = hit / predicted if predicted else 0.0
    recall = hit / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted": predicted,
    }


def _macro_f1(rows: Sequence[ClassScoreRow], averaged: Sequence[int]) -> float:
    """The mean F1 over *averaged*, a class with no row scoring 0."""
    f1_of = {row["label"]: row["f1"] for row in rows}
    if not averaged:
        return 0.0
    return float(np.mean([f1_of.get(value, 0.0) for value in averaged]))


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: Sequence[int] | None = None,
    average_over: Sequence[int] | None = None,
) -> ClassificationScores:
    """How well predicted classes agree with ground-truth classes.

    Rows labeled :data:`~mosaic.core.helpers.UNLABELED` are dropped first and
    counted in ``n_unlabeled``. Precision, recall and F1 are per class, and a
    class nothing was predicted as has a precision of 0.

    Args:
        y_true: Ground-truth class per row.
        y_pred: Predicted class per row.
        classes: The classes to report, in order. Unset, every class a row
            holds or is predicted as. A class neither holds is reported with
            zeros, and one either holds outside this list is appended.
        average_over: The classes ``macro_f1`` averages over, such as every
            behavior but ``other``. Unset, every class a labeled row holds.

    Returns:
        ``n_samples``, ``n_unlabeled``, ``accuracy``, ``macro_f1``,
        ``average_over``, ``classes``, ``per_class`` (one row per class of
        ``label``, ``precision``, ``recall``, ``f1``, ``support`` and
        ``predicted``) and ``confusion`` (rows the true classes, columns the
        predicted ones, both in ``classes`` order).
    """
    true, pred = _as_int_arrays(y_true, y_pred)
    labeled = true != UNLABELED
    n_unlabeled = int((~labeled).sum())
    true, pred = true[labeled], pred[labeled]
    observed = sorted({int(value) for value in np.concatenate([true, pred])})
    listed = [int(value) for value in classes] if classes is not None else []
    order = listed + [value for value in observed if value not in listed]
    position = {value: index for index, value in enumerate(order)}

    confusion = np.zeros((len(order), len(order)), dtype=np.int64)
    if true.size:
        rows = np.array([position[int(value)] for value in true], dtype=np.int64)
        cols = np.array([position[int(value)] for value in pred], dtype=np.int64)
        np.add.at(confusion, (rows, cols), 1)

    per_class = [
        _class_row(
            value,
            hit=int(confusion[index, index]),
            support=int(confusion[index].sum()),
            predicted=int(confusion[:, index].sum()),
        )
        for index, value in enumerate(order)
    ]
    averaged = (
        [int(value) for value in average_over]
        if average_over is not None
        else sorted({int(value) for value in true})
    )
    return {
        "n_samples": int(true.size),
        "n_unlabeled": n_unlabeled,
        "accuracy": float((true == pred).mean()) if true.size else 0.0,
        "macro_f1": _macro_f1(per_class, averaged),
        "average_over": averaged,
        "classes": order,
        "per_class": per_class,
        "confusion": confusion,
    }


@dataclass(frozen=True)
class ScoredRows:
    """The rows :func:`select_scored_rows` keeps, and what it set aside.

    Attributes:
        frame: The rows to score.
        n_rows: Rows given.
        n_conflicting: Pairs whose two rows carry different labels, which
            ``"mean_probabilities"`` cannot score as one and drops.
    """

    frame: pd.DataFrame
    n_rows: int
    n_conflicting: int = 0


def select_scored_rows(
    frame: pd.DataFrame,
    *,
    perspective: Perspective,
    prediction_column: str = PREDICTION_COLUMN,
    label_column: str = LABEL_COLUMN,
    probability_prefix: str = "prob_",
) -> ScoredRows:
    """Keep the rows of a pair output that one score is taken over.

    A pair feature writes each pair twice per frame, once focal on each
    individual, and a classifier then predicts each direction. Scoring both
    rows counts every frame twice; :data:`Perspective` says which to score.

    Args:
        frame: Aligned rows, with ``group``, ``sequence``, ``frame``, the
            prediction and the label, and ``perspective``, ``id1`` and ``id2``
            for a pair output.
        perspective: Which rows to keep. See :data:`Perspective`.
        prediction_column: The predicted class.
        label_column: The ground-truth class.
        probability_prefix: The prefix of the per-class probability columns
            ``"mean_probabilities"`` averages, as ``prob_<class>``.

    Returns:
        The kept rows. Under ``"mean_probabilities"`` one row per pair and
        frame, whose prediction is the class of the highest mean probability
        and whose label is that of the pair's labeled rows.

    Raises:
        ValueError: *perspective* is ``None`` and *frame* holds more than one
            perspective, an ``int`` names a perspective on an output without
            one, or ``"mean_probabilities"`` finds no probability column.
    """
    n_rows = int(frame.shape[0])
    has_perspective = "perspective" in frame.columns
    if perspective is None:
        if has_perspective and frame["perspective"].nunique() > 1:
            raise ValueError(
                "these rows hold more than one perspective, so each pair is "
                "scored twice. Choose a perspective, 'pooled' to score both "
                "rows, or 'mean_probabilities'."
            )
        return ScoredRows(frame=frame, n_rows=n_rows)
    if perspective == "pooled":
        return ScoredRows(frame=frame, n_rows=n_rows)
    if isinstance(perspective, int):
        if not has_perspective:
            raise ValueError(
                f"perspective {perspective} names a perspective, and these rows "
                "have no perspective column."
            )
        kept = frame.loc[frame["perspective"] == perspective]
        return ScoredRows(frame=kept.reset_index(drop=True), n_rows=n_rows)
    return _mean_probabilities(
        frame,
        prediction_column=prediction_column,
        label_column=label_column,
        probability_prefix=probability_prefix,
    )


def _mean_probabilities(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    label_column: str,
    probability_prefix: str,
) -> ScoredRows:
    """One row per pair and frame, scored by the mean of its rows' probabilities."""
    probability = {
        int(column.removeprefix(probability_prefix)): column
        for column in frame.columns
        if column.startswith(probability_prefix)
        and column.removeprefix(probability_prefix).lstrip("-").isdigit()
    }
    if not probability:
        raise ValueError(
            f"'mean_probabilities' averages the {probability_prefix}<class> "
            "columns, and these rows have none."
        )
    key = ["group", "sequence", "frame"]
    keyed = frame[[*key, label_column]].copy()
    if {"id1", "id2"} <= set(frame.columns):
        ids = frame[["id1", "id2"]].to_numpy()
        keyed["_low"] = ids.min(axis=1)
        keyed["_high"] = ids.max(axis=1)
        key = [*key, "_low", "_high"]
    classes = sorted(probability)
    for value in classes:
        keyed[f"_p{value}"] = frame[probability[value]].to_numpy(dtype=float)
    labeled = keyed[label_column].where(keyed[label_column] != UNLABELED)
    keyed["_label_min"] = labeled
    keyed["_label_max"] = labeled
    grouped = keyed.groupby(key, sort=True, dropna=False).agg(
        **{f"_p{value}": (f"_p{value}", "mean") for value in classes},
        _label_min=("_label_min", "min"),
        _label_max=("_label_max", "max"),
    )
    conflicting = (grouped["_label_min"] != grouped["_label_max"]) & grouped[
        "_label_min"
    ].notna()
    grouped = grouped.loc[~conflicting].reset_index()
    means = grouped[[f"_p{value}" for value in classes]].to_numpy()
    best = np.asarray(classes, dtype=np.int64)[means.argmax(axis=1)]
    label = grouped["_label_min"].fillna(UNLABELED).to_numpy(dtype=np.int64)
    out = grouped[["group", "sequence", "frame"]].copy()
    out[prediction_column] = best
    out[label_column] = label
    for value in classes:
        out[probability[value]] = grouped[f"_p{value}"].to_numpy()
    return ScoredRows(
        frame=out.reset_index(drop=True),
        n_rows=int(frame.shape[0]),
        n_conflicting=int(conflicting.sum()),
    )


class ClassScore(StrictModel):
    """One class's score in an :class:`EvaluationReport`."""

    label: int
    name: str = ""
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


class EvaluationReport(StrictModel):
    """What ``evaluate-labels`` measured, over every entry it scored.

    Attributes:
        mode: ``classifier`` for predicted classes, ``clusters`` for a
            clustering scored against the classes.
        n_entries: Entries scored.
        n_scored: Rows scored.
        n_unlabeled: Rows dropped because no label covers them.
        n_missing_prediction: Rows dropped because they hold no prediction.
        n_conflicting: Pairs dropped because their two rows carry different
            labels, under ``mean_probabilities``.
        perspective: Which rows were scored (:data:`Perspective`).
        accuracy: The share of scored rows predicted correctly. In clusters
            mode, after the label map, or the majority mapping without one.
        macro_f1: Mean F1 over ``average_over``.
        average_over: The classes ``macro_f1`` averages over.
        classes: Every class reported, in confusion order.
        per_class: One score per class in ``classes``.
        confusion: Rows the true classes, columns the predicted ones.
        clusters: In clusters mode, the scores of :func:`cluster_label_scores`
            that a clustering has and a classifier does not: ``ARI``, ``AMI``,
            ``homogeneity``, ``completeness``, ``v_measure``, ``purity``,
            ``hungarian_accuracy``, ``hungarian_one_to_one``, ``n_clusters``.
            These compare the clusters with the classes and go through no map.
        label_map: In clusters mode, the cluster to class map the predictions
            went through: the map that was given, or else the majority map of
            the rows scored. Empty for a classifier, and for a report pooled
            from reports that went through different maps.
        label_map_source: Where ``label_map`` came from
            (:data:`LabelMapSource`). Read it before reading ``accuracy``. From
            ``"scored_rows"`` the map was fitted on the rows scored, so
            ``accuracy`` equals the clustering's purity and no row was held out;
            only ``"fit-label-map"`` measures how well the clusters predict
            behavior on other entries. ``""`` in a report saved before the field
            existed, which cannot say.
    """

    mode: Literal["classifier", "clusters"]
    n_entries: int
    n_scored: int
    n_unlabeled: int
    n_missing_prediction: int
    n_conflicting: int = 0
    perspective: int | Literal["pooled", "mean_probabilities"] | None = None
    accuracy: float
    macro_f1: float
    average_over: list[int]
    classes: list[int]
    per_class: list[ClassScore]
    confusion: list[list[int]]
    clusters: dict[str, float | int | bool] = {}
    label_map: dict[int, int] = {}
    label_map_source: LabelMapSource = ""


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


# --- Pooling and spread --------------------------------------------------------------

_COUNT_KINDS = ("tp", "fp", "fn")
_DRAW_CELLS = 1 << 22
"""How many resample weights :func:`bootstrap_sequences` holds at once."""
_COUNT_COLUMN = re.compile(r"^(tp|fp|fn)_(-?[0-9]+)$")


def entry_counts(
    scores: ClassificationScores, classes: Sequence[int]
) -> dict[str, int]:
    """One entry's true positives, false positives and false negatives per class.

    The columns ``evaluate-labels`` writes beside each entry's score:
    ``tp_<class>`` rows predicted and labeled as the class, ``fp_<class>`` rows
    predicted as it and labeled otherwise, and ``fn_<class>`` rows labeled as it
    and predicted otherwise. They are all that precision, recall and F1 need, and
    they add across entries. A row predicted as a class outside *classes*, such
    as a cluster the label map does not name, counts for its true class as a
    false negative and for no class as a false positive.

    Args:
        scores: The entry's :func:`classification_scores`.
        classes: The classes to name, in column order.

    Returns:
        ``tp_<class>``, ``fp_<class>`` and ``fn_<class>`` for each class, 0 for
        a class *scores* does not list.
    """
    position = {value: index for index, value in enumerate(scores["classes"])}
    confusion = np.asarray(scores["confusion"])
    by_label = {row["label"]: row for row in scores["per_class"]}
    counts: dict[str, int] = {}
    for value in classes:
        row = by_label.get(value)
        hit = int(confusion[position[value], position[value]]) if row else 0
        counts[f"tp_{value}"] = hit
        counts[f"fp_{value}"] = row["predicted"] - hit if row else 0
        counts[f"fn_{value}"] = row["support"] - hit if row else 0
    return counts


class CountScores(TypedDict):
    """What :func:`pool_entry_counts` returns."""

    n_samples: int
    accuracy: float
    macro_f1: float
    average_over: list[int]
    classes: list[int]
    per_class: list[ClassScoreRow]


def _count_classes(table: pd.DataFrame) -> list[int]:
    """The classes whose counts *table* holds, ascending."""
    found: dict[int, set[str]] = {}
    for column in table.columns:
        match = _COUNT_COLUMN.match(str(column))
        if match is not None:
            found.setdefault(int(match[2]), set()).add(match[1])
    if not found:
        msg = (
            "the table has no tp_<class>, fp_<class> and fn_<class> columns. "
            "They are the columns of an evaluate-labels run's output."
        )
        raise ValueError(msg)
    for value in sorted(found):
        for kind in _COUNT_KINDS:
            if kind not in found[value]:
                msg = f"the table has counts for class {value} but no {kind}_{value} column."
                raise ValueError(msg)
    return sorted(found)


def _count_array(table: pd.DataFrame, classes: Sequence[int]) -> np.ndarray:
    """*table*'s counts as an array: entry, class, then ``tp``, ``fp``, ``fn``."""
    counts = np.zeros((table.shape[0], len(classes), len(_COUNT_KINDS)), dtype=np.int64)
    for column, value in enumerate(classes):
        for kind, name in enumerate(_COUNT_KINDS):
            counts[:, column, kind] = table[f"{name}_{value}"].to_numpy(dtype=np.int64)
    return counts


def _score_totals(
    totals: np.ndarray, classes: Sequence[int], average_over: Sequence[int] | None
) -> CountScores:
    """The scores of counts already summed over entries, one ``(tp, fp, fn)`` per class."""
    per_class = [
        _class_row(
            value,
            hit=int(totals[index, 0]),
            support=int(totals[index, 0] + totals[index, 2]),
            predicted=int(totals[index, 0] + totals[index, 1]),
        )
        for index, value in enumerate(classes)
    ]
    n_samples = sum(row["support"] for row in per_class)
    hits = int(totals[:, 0].sum())
    averaged = (
        [int(value) for value in average_over]
        if average_over is not None
        else [row["label"] for row in per_class if row["support"] > 0]
    )
    return {
        "n_samples": n_samples,
        "accuracy": hits / n_samples if n_samples else 0.0,
        "macro_f1": _macro_f1(per_class, averaged),
        "average_over": averaged,
        "classes": [int(value) for value in classes],
        "per_class": per_class,
    }


def pool_entry_counts(
    table: pd.DataFrame, *, average_over: Sequence[int] | None = None
) -> CountScores:
    """The score of all the rows of several entries, from their counts.

    Adds the ``tp_<class>``, ``fp_<class>`` and ``fn_<class>`` columns over the
    rows of *table* and scores the sums with the formula of
    :func:`classification_scores`. The result equals scoring the rows
    themselves, bit for bit, which averaging per-entry scores does not: an
    entry's score weighs a short sequence like a long one.

    Args:
        table: One row per entry with the count columns, as ``evaluate-labels``
            writes them. Other columns are ignored. The rows of every entry or
            of a chosen subset, such as the test groups.
        average_over: The classes ``macro_f1`` averages over. Unset, every
            class with a row labeled as it. Pass the report's ``average_over``
            to match a run that set it.

    Returns:
        ``n_samples``, ``accuracy``, ``macro_f1``, ``average_over``, ``classes``
        and ``per_class`` (see :class:`ClassScoreRow`).

    Raises:
        ValueError: *table* holds no count columns, or only some of a class's.
    """
    classes = _count_classes(table)
    totals = _count_array(table, classes).sum(axis=0)
    return _score_totals(totals, classes, average_over)


class Interval(TypedDict):
    """A score and the percentile interval of its bootstrap."""

    estimate: float
    lower: float
    upper: float


class ClassInterval(TypedDict):
    """One class's F1 interval in :class:`BootstrapScores`."""

    label: int
    f1: Interval


class Replicates(TypedDict):
    """The bootstrap replicates behind :class:`BootstrapScores`, one row each."""

    accuracy: np.ndarray
    macro_f1: np.ndarray
    f1: np.ndarray


class BootstrapScores(TypedDict):
    """What :func:`bootstrap_sequences` returns."""

    n_entries: int
    n_boot: int
    confidence: float
    random_state: int
    average_over: list[int]
    accuracy: Interval
    macro_f1: Interval
    per_class: list[ClassInterval]
    replicates: Replicates


def bootstrap_sequences(
    table: pd.DataFrame,
    *,
    random_state: int,
    n_boot: int = 1000,
    confidence: float = 0.95,
    average_over: Sequence[int] | None = None,
) -> BootstrapScores:
    """How much a pooled score moves when the entries drawn are other entries.

    Resamples the rows of *table*, the entries, with replacement, pools the
    counts of each resample as :func:`pool_entry_counts` does, and reads the
    percentile interval of the replicates. The unit is the entry, not the frame:
    frames of one sequence are not independent, so a bootstrap over rows would
    be too narrow. An entry that scored no row is left out before the draw.

    A fixed *random_state* and the same number of scored entries draw the same
    resamples, so two systems scored on the same entries, in the same order,
    can be compared replicate by replicate through ``replicates``. The test of
    their difference is not made here.

    Args:
        table: One row per entry with the count columns that ``evaluate-labels``
            writes.
        random_state: The seed. Required, so a run is repeated by passing the
            same value.
        n_boot: How many resamples.
        confidence: The width of the interval, between 0 and 1 exclusive.
        average_over: The classes ``macro_f1`` averages over, fixed across
            replicates. Unset, every class with a row labeled as it in *table*.

    Returns:
        ``n_entries`` (those resampled), the arguments, ``average_over``,
        ``accuracy`` and ``macro_f1`` as an :class:`Interval`, ``per_class``
        (each class's F1 interval) and ``replicates``: ``accuracy`` and
        ``macro_f1`` with one value per resample, and ``f1`` with a column per
        class of ``per_class``.

    Raises:
        ValueError: *n_boot* is below 1, *confidence* is outside (0, 1), *table*
            holds no count columns, or no entry scored a row.
    """
    if n_boot < 1:
        msg = f"n_boot must be at least 1, not {n_boot}."
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie between 0 and 1, not {confidence}."
        raise ValueError(msg)
    classes = _count_classes(table)
    counts = _count_array(table, classes)
    counts = counts[counts.sum(axis=(1, 2)) > 0]
    n_entries = counts.shape[0]
    if n_entries == 0:
        msg = "no entry scored a row, so there is nothing to resample."
        raise ValueError(msg)

    point = _score_totals(counts.sum(axis=0), classes, average_over)
    averaged = point["average_over"]
    rng = np.random.default_rng(random_state)
    chances = np.full(n_entries, 1.0 / n_entries)
    flat = counts.reshape(n_entries, -1)
    step = max(1, _DRAW_CELLS // n_entries)
    pooled = np.concatenate(
        [
            rng.multinomial(n_entries, chances, size=min(step, n_boot - start)) @ flat
            for start in range(0, n_boot, step)
        ]
    ).reshape(n_boot, len(classes), 3)

    hit = pooled[:, :, 0].astype(np.float64)
    predicted = hit + pooled[:, :, 1]
    support = hit + pooled[:, :, 2]
    precision = np.divide(hit, predicted, out=np.zeros_like(hit), where=predicted > 0)
    recall = np.divide(hit, support, out=np.zeros_like(hit), where=support > 0)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros_like(hit), where=both > 0)
    padded = np.concatenate([f1, np.zeros((n_boot, 1))], axis=1)
    column = {value: index for index, value in enumerate(classes)}
    macro = (
        padded[:, [column.get(value, len(classes)) for value in averaged]].mean(axis=1)
        if averaged
        else np.zeros(n_boot)
    )
    labeled = support.sum(axis=1)
    accuracy = np.divide(
        hit.sum(axis=1), labeled, out=np.zeros(n_boot), where=labeled > 0
    )

    tail = (1.0 - confidence) / 2.0

    def interval(estimate: float, replicates: np.ndarray) -> Interval:
        lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
        return {"estimate": estimate, "lower": float(lower), "upper": float(upper)}

    return {
        "n_entries": n_entries,
        "n_boot": n_boot,
        "confidence": confidence,
        "random_state": random_state,
        "average_over": averaged,
        "accuracy": interval(point["accuracy"], accuracy),
        "macro_f1": interval(point["macro_f1"], macro),
        "per_class": [
            {
                "label": row["label"],
                "f1": interval(row["f1"], f1[:, index]),
            }
            for index, row in enumerate(point["per_class"])
        ],
        "replicates": {"accuracy": accuracy, "macro_f1": macro, "f1": f1},
    }


def pool_reports(reports: Sequence[EvaluationReport]) -> EvaluationReport:
    """One report for the entries of several, as if they were scored in one run.

    Sums the confusion matrices over the union of the classes and scores the
    sums, so the result equals the report of a run over every entry at once. It
    pools the runs of a cross-validation, whose entry sets are disjoint and so
    cannot be one run. A report names no entries, so *reports* that overlap are
    not detected and count a shared entry twice.

    A clustering's agreement scores (``ARI``, ``purity`` and the rest) are not
    sums, so the pooled report holds none. Its ``label_map`` is kept only when
    every report went through one map, and its ``label_map_source`` only when
    they share one source.

    Args:
        reports: The reports to pool, of one ``mode`` and ``perspective``.

    Returns:
        The pooled report. Its ``average_over`` is the one the reports share, or
        else the sorted union of theirs.

    Raises:
        ValueError: *reports* is empty, or differs in ``mode`` or
            ``perspective``.
    """
    if not reports:
        msg = "pool_reports needs at least one report."
        raise ValueError(msg)
    first = reports[0]
    for report in reports[1:]:
        for name in ("mode", "perspective"):
            if getattr(report, name) != getattr(first, name):
                msg = (
                    f"cannot pool reports with another {name} "
                    f"({getattr(first, name)!r} and {getattr(report, name)!r})."
                )
                raise ValueError(msg)

    classes: list[int] = []
    for report in reports:
        classes.extend(value for value in report.classes if value not in classes)
    position = {value: index for index, value in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    support = np.zeros(len(classes), dtype=np.int64)
    predicted = np.zeros(len(classes), dtype=np.int64)
    names: dict[int, str] = {}
    for report in reports:
        own = [position[value] for value in report.classes]
        confusion[np.ix_(own, own)] += np.asarray(
            report.confusion, dtype=np.int64
        ).reshape(len(own), len(own))
        for score in report.per_class:
            support[position[score.label]] += score.support
            predicted[position[score.label]] += score.predicted
            if score.name and score.label not in names:
                names[score.label] = score.name

    rows = [
        _class_row(
            value,
            hit=int(confusion[index, index]),
            support=int(support[index]),
            predicted=int(predicted[index]),
        )
        for index, value in enumerate(classes)
    ]
    n_scored = sum(report.n_scored for report in reports)
    shared = all(report.average_over == first.average_over for report in reports)
    averaged = (
        list(first.average_over)
        if shared
        else sorted({value for report in reports for value in report.average_over})
    )
    same_map = all(report.label_map == first.label_map for report in reports)
    same_source = all(
        report.label_map_source == first.label_map_source for report in reports
    )
    return EvaluationReport(
        mode=first.mode,
        n_entries=sum(report.n_entries for report in reports),
        n_scored=n_scored,
        n_unlabeled=sum(report.n_unlabeled for report in reports),
        n_missing_prediction=sum(report.n_missing_prediction for report in reports),
        n_conflicting=sum(report.n_conflicting for report in reports),
        perspective=first.perspective,
        accuracy=int(np.trace(confusion)) / n_scored if n_scored else 0.0,
        macro_f1=_macro_f1(rows, averaged),
        average_over=averaged,
        classes=classes,
        per_class=[ClassScore(name=names.get(row["label"], ""), **row) for row in rows],
        confusion=confusion.tolist(),
        label_map=dict(first.label_map) if same_map else {},
        label_map_source=first.label_map_source if same_source else "",
    )


class MetricSpread(TypedDict):
    """A score's mean over reports, and its sample standard deviation."""

    mean: float
    sd: float


class ClassSpread(TypedDict):
    """One class's spread in :class:`ReportAggregate`."""

    label: int
    name: str
    n_reports: int
    precision: MetricSpread
    recall: MetricSpread
    f1: MetricSpread


class ReportAggregate(TypedDict):
    """What :func:`aggregate_reports` returns."""

    n_reports: int
    accuracy: MetricSpread
    macro_f1: MetricSpread
    per_class: list[ClassSpread]


def _spread(values: Sequence[float]) -> MetricSpread:
    """The mean and sample standard deviation of *values*, NaN for one value."""
    array = np.asarray(values, dtype=np.float64)
    sd = float(np.std(array, ddof=1)) if array.size > 1 else float("nan")
    return {"mean": float(array.mean()), "sd": sd}


def aggregate_reports(reports: Sequence[EvaluationReport]) -> ReportAggregate:
    """The mean and spread of the scores of several runs of one experiment.

    For the reports of repeated runs, such as seeds or folds, whose variation is
    the quantity wanted. To score the rows of all of them as one set, use
    :func:`pool_reports`. The standard deviation is the sample one (``ddof=1``)
    and NaN where one report gives a value.

    A class is averaged over the reports that list it: a fold that never saw a
    class has no score for it, which is not a score of 0.

    Args:
        reports: The reports to summarize.

    Returns:
        The spread of ``accuracy`` and ``macro_f1``, and of the precision,
        recall and F1 of each class, the classes ascending.

    Raises:
        ValueError: *reports* is empty.
    """
    if not reports:
        msg = "aggregate_reports needs at least one report."
        raise ValueError(msg)
    scores: dict[int, list[ClassScore]] = {}
    for report in reports:
        for score in report.per_class:
            scores.setdefault(score.label, []).append(score)
    return {
        "n_reports": len(reports),
        "accuracy": _spread([report.accuracy for report in reports]),
        "macro_f1": _spread([report.macro_f1 for report in reports]),
        "per_class": [
            {
                "label": label,
                "name": next((score.name for score in held if score.name), ""),
                "n_reports": len(held),
                "precision": _spread([score.precision for score in held]),
                "recall": _spread([score.recall for score in held]),
                "f1": _spread([score.f1 for score in held]),
            }
            for label, held in sorted(scores.items())
        ],
    }


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
    tracks_run_id: str | None = None,
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
        tracks_run_id: The tracks variant the entries resolve to, when an entry
            has more than one. ``None`` takes the one variant the run read,
            directly or through its upstreams, when its lineage names exactly
            one.
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
    if tracks_run_id is None:
        tracks_run_id = _tracks_variant_read(ds, prediction)
    values = load_values(
        ds,
        [prediction, labels],
        scope=scope,
        tracks_run_id=tracks_run_id,
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


def _tracks_variant_read(ds: Dataset, prediction: ResultColumn) -> str | None:
    """The one tracks variant *prediction*'s run read, or ``None``.

    Reading a run's rows builds a manifest, which resolves each entry to one
    tracks variant. A dataset holding two variants of an entry -- after a
    reconversion -- refuses to choose, though the run being scored already chose
    when it ran. ``None`` when the run is unpinned, its lineage is unknown, or it
    names no variant or several.
    """
    from .pipeline.inventory.params import run_tracks_variants

    if prediction.run_id is None:
        return None
    variants = run_tracks_variants(ds, prediction.feature, prediction.run_id)
    if variants is None or len(variants) != 1:
        return None
    return next(iter(variants))
