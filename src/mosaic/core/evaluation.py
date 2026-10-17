"""Scoring a run's predictions against ground-truth labels.

Two halves, kept apart so either can be used without the other:

- **Scorers** take arrays or frames and return plain dicts. They read no dataset,
  so a feature's ``apply`` can call them on the rows it was handed:
  :func:`classification_scores` for a classifier's predicted classes,
  :func:`cluster_label_scores` for a clustering, and :func:`select_scored_rows`
  for which perspective of a pair row is scored.
- **Alignment** reads a run's output and the labels of the same entries from a
  dataset and pairs them row by row (:func:`align_predictions_to_labels`).

:class:`EvaluationReport` is what the ``evaluate-labels`` feature writes, and
reads the same whichever half produced its numbers.

Both drop a row whose label is :data:`~mosaic.core.helpers.UNLABELED` and count
it, so a score never treats "nobody annotated this frame" as a class, and a
caller can report how many rows it lost.

sklearn, scipy and the pipeline engine are imported inside the functions that
use them, so importing this module stays cheap.
"""

from __future__ import annotations

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
    "ClassScore",
    "ClassScoreRow",
    "ClassificationScores",
    "EvaluationReport",
    "Perspective",
    "ScoredRows",
    "align_predictions_to_labels",
    "classification_scores",
    "cluster_label_scores",
    "majority_mapping",
    "perspective_consistency",
    "select_scored_rows",
]

type Perspective = int | Literal["pooled", "mean_probabilities"] | None
"""Which rows of a pair output are scored.

An ``int`` keeps the rows of that ``perspective``. ``"pooled"`` scores every row.
``"mean_probabilities"`` averages the class probabilities of a pair's two rows
at one frame and scores the class they then favor. ``None`` scores the rows as
they are and refuses an output holding more than one perspective.
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

    per_class: list[ClassScoreRow] = []
    for index, value in enumerate(order):
        hit = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = hit / predicted if predicted else 0.0
        recall = hit / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class.append(
            {
                "label": value,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted": predicted,
            }
        )

    averaged = (
        [int(value) for value in average_over]
        if average_over is not None
        else sorted({int(value) for value in true})
    )
    f1_of = {row["label"]: row["f1"] for row in per_class}
    macro = (
        float(np.mean([f1_of.get(value, 0.0) for value in averaged]))
        if averaged
        else 0.0
    )
    return {
        "n_samples": int(true.size),
        "n_unlabeled": n_unlabeled,
        "accuracy": float((true == pred).mean()) if true.size else 0.0,
        "macro_f1": macro,
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
            ``purity``, ``hungarian_accuracy``, ``hungarian_one_to_one``,
            ``n_clusters``.
        label_map: The cluster to class map the predictions went through, empty
            when none was given.
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
