"""What the global classifiers have in common.

``xgboost``, ``lightning-action`` and ``sequence-classifier`` fit on the train
entries of a labeled-templates run and write one probability per class and a
``predicted_label`` per row. This module holds the parts of that which do not
depend on the model: the output assembly, the training-frame standardizer, the
cross-validation folds and the bundle write. A classifier supplies a model and
a fit loop.

Imports no machine-learning package, so every classifier and the ensemble step
can use it without one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Container, Mapping
from typing import TYPE_CHECKING, Annotated, Final

import numpy as np
import pandas as pd

from mosaic.core.params import Declared
from mosaic.core.pipeline._utils import atomic_write

from .helpers import cv_fold_assignment, meta_columns

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from mosaic.core.pipeline.job import CancelToken
    from mosaic.core.pipeline.progress import ProgressCallback

__all__ = [
    "FIT_SCOPE_REMEDY",
    "CV_SEED_DESCRIPTION",
    "DECISION_THRESHOLD_DESCRIPTION",
    "DEFAULT_CLASS_DESCRIPTION",
    "CvSeed",
    "DecisionThreshold",
    "FitReporting",
    "RunningMoments",
    "assemble_prediction",
    "balanced_class_weights",
    "check_cross_validation",
    "check_folds",
    "deal_folds",
    "fold_map",
    "fold_rows",
    "fold_table",
    "predicted_labels",
    "refuse_entries_outside_scope",
    "standardize",
    "write_bundle",
]

log = logging.getLogger(__name__)


# --- Parameters ---

FIT_SCOPE_REMEDY: Final = (
    "The step's first run must be asked for over a scope that holds the training "
    "entries, because it fits on them. Once it has fitted, a wider scope applies "
    "the stored fit to the new entries and fits nothing again. To apply a fit "
    "from elsewhere, name it through `model`."
)

DECISION_THRESHOLD_DESCRIPTION: Final = (
    "Unset, the predicted class is the class with the highest probability. "
    "A float applies one probability threshold to every class, and a "
    "mapping from class to threshold applies a per-class one. Any row "
    "where no class clears its threshold is assigned default_class."
)

DEFAULT_CLASS_DESCRIPTION: Final = (
    "The class assigned to a row where decision_threshold is set and no "
    "class clears its threshold."
)

CV_SEED_DESCRIPTION: Final = (
    "The seed that deals the training sequences into folds. It changes the "
    "folds and nothing else, so runs that differ in random_state are "
    "compared on the same folds. Other than 0 only with cv_folds."
)

DecisionThreshold = Annotated[
    float | Mapping[int, float] | None, Declared(DECISION_THRESHOLD_DESCRIPTION)
]
"""The type of a classifier's ``decision_threshold`` parameter."""

CvSeed = Annotated[int, Declared(CV_SEED_DESCRIPTION)]
"""The type of a classifier's ``cv_seed`` parameter."""


def check_cross_validation(
    cv_folds: int | None, cv_seed: int, *, model_given: bool
) -> None:
    """Refuse a ``cv_folds`` and ``cv_seed`` that cannot describe one fit.

    ``cv_folds`` fits fold models from templates, and ``cv_seed`` only deals
    their folds. A ``cv_seed`` other than 0 without folds would give the same
    output a second run id, since the id hashes both.

    Args:
        cv_folds: The number of folds, unset for none.
        cv_seed: The seed that deals them.
        model_given: Whether the parameters name a pre-fitted ``model``.

    Raises:
        ValueError: ``cv_seed`` is set without ``cv_folds``, or ``cv_folds``
            is set beside a ``model``.
    """
    if cv_folds is None:
        if cv_seed != 0:
            msg = "cv_seed deals the folds of cv_folds, which is not set"
            raise ValueError(msg)
        return
    if model_given:
        msg = (
            "cv_folds fits fold models from templates, so it needs "
            "templates and is refused beside model"
        )
        raise ValueError(msg)


# --- Prediction ---


def predicted_labels(
    probs: npt.NDArray[np.float64],
    classes: list[int],
    threshold: float | Mapping[int, float] | None,
    default_class: int,
) -> npt.NDArray[np.intp]:
    """Return the class each row of *probs* is assigned to.

    Without a *threshold* a row takes the class of its highest probability.
    With one, a probability under its class's threshold counts as 0, and a row
    in which every class counts as 0 takes *default_class*. A row with a NaN
    probability, one of no series, takes *default_class* in both cases.

    Args:
        probs: One row per input row, one column per class in *classes*.
        classes: The class labels of the columns.
        threshold: One threshold for every class, a threshold per class (0 for a
            class it does not list), or ``None``.
        default_class: The class of a row that no class claims.
    """
    labels = np.asarray(classes, dtype=np.intp)
    if threshold is None:
        assigned = labels[np.argmax(np.nan_to_num(probs, nan=-1.0), axis=1)]
    else:
        masked = probs.copy()
        for column, cls in enumerate(classes):
            limit = (
                threshold.get(cls, 0.0) if isinstance(threshold, Mapping) else threshold
            )
            masked[:, column] = np.where(
                probs[:, column] >= limit, probs[:, column], 0.0
            )
        assigned = labels[np.argmax(masked, axis=1)]
        assigned[masked.sum(axis=1) == 0] = default_class
    assigned[np.isnan(probs).any(axis=1)] = default_class
    return assigned


def assemble_prediction(
    df: pd.DataFrame,
    probs: npt.NDArray[np.float64],
    classes: list[int],
    threshold: float | Mapping[int, float] | None,
    default_class: int,
) -> pd.DataFrame:
    """Return the output table of a classifier: identity, ``prob_<class>``, ``predicted_label``.

    The whole identity of *df* travels with the predictions, ``perspective``
    included: a classifier is routinely fed pair-level input, and without the
    identity the output is two rows per frame with nothing to tell them apart,
    unjoinable against the very features it was trained on.

    Args:
        df: The input table. Left unchanged.
        probs: One row per row of *df*, one column per class in *classes*.
        classes: The class labels of the columns.
        threshold: As for :func:`predicted_labels`.
        default_class: As for :func:`predicted_labels`.
    """
    result = df[meta_columns(df)].copy()
    for column, cls in enumerate(classes):
        result[f"prob_{cls}"] = probs[:, column]
    result["predicted_label"] = predicted_labels(
        probs, classes, threshold, default_class
    )
    return result


# --- Standardization ---


class RunningMoments:
    """Per-column count, mean and sum of squared deviations over the finite values.

    Runs are added one at a time and combined with the pairwise update, so the
    statistics of every training row exist without holding the rows together.
    """

    def __init__(self, n_columns: int) -> None:
        self.count: npt.NDArray[np.int64] = np.zeros(n_columns, dtype=np.int64)
        self.mean: npt.NDArray[np.float64] = np.zeros(n_columns)
        self.squares: npt.NDArray[np.float64] = np.zeros(n_columns)

    def add(self, matrix: npt.NDArray[np.float64]) -> None:
        """Fold the finite values of *matrix*, one row per frame, into the statistics."""
        finite = np.isfinite(matrix)
        data = np.where(finite, matrix, 0.0)
        count = finite.sum(axis=0)
        mean = np.divide(
            data.sum(axis=0), count, out=np.zeros(data.shape[1]), where=count > 0
        )
        squares = np.where(finite, data - mean, 0.0)
        squares = (squares * squares).sum(axis=0)
        total = self.count + count
        share = np.divide(count, total, out=np.zeros(data.shape[1]), where=total > 0)
        delta = mean - self.mean
        self.squares = self.squares + squares + delta * delta * self.count * share
        self.mean = self.mean + delta * share
        self.count = total

    def standardizer(
        self, min_std: float = 0.0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the mean and the standard deviation to standardize by.

        A column with no finite value has mean 0, and one whose standard
        deviation is not above *min_std* has a standard deviation of 1, so that
        standardizing leaves neither undefined.

        Args:
            min_std: The spread under which a column counts as constant.
        """
        variance = np.divide(
            self.squares,
            self.count,
            out=np.ones(self.count.shape[0]),
            where=self.count > 0,
        )
        std = np.sqrt(variance)
        return self.mean, np.where(std > min_std, std, 1.0)


def standardize(
    matrix: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    std: npt.NDArray[np.float64],
    clip: float | None = None,
) -> npt.NDArray[np.float64]:
    """Standardize *matrix* by the training statistics, reading a missing value as the mean.

    Args:
        matrix: One row per frame.
        mean: The training mean of each column.
        std: The training standard deviation of each column.
        clip: When set, the bound that standardized values are limited to,
            on both sides.
    """
    standardized = (matrix - mean) / std
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    if clip is None:
        return standardized
    return np.clip(standardized, -clip, clip)


# --- Cross-validation folds ---


def deal_folds(
    feature: str, train_entries: list[tuple[str, str]], folds: int, seed: int
) -> dict[tuple[str, str], int]:
    """Deal the train entries into *folds* folds, naming *feature* in a refusal.

    Wraps :func:`~mosaic.behavior.feature_library.helpers.cv_fold_assignment`,
    so that every classifier deals alike.

    Raises:
        ValueError: Naming *feature*, when *folds* is below 2 or above the
            number of entries.
    """
    try:
        return cv_fold_assignment(train_entries, folds, seed)
    except ValueError as exc:
        msg = f"[{feature}] {exc}"
        raise ValueError(msg) from exc


def fold_rows(fold_of: Mapping[tuple[str, str], int]) -> list[tuple[str, str, int]]:
    """Return the ``(group, sequence, fold)`` of every dealt entry, sorted, as a bundle stores them."""
    return [
        (group, sequence, fold) for (group, sequence), fold in sorted(fold_of.items())
    ]


def fold_map(rows: list[tuple[str, str, int]]) -> dict[tuple[str, str], int]:
    """Return the fold of each entry from the rows :func:`fold_rows` wrote."""
    return {(group, sequence): fold for group, sequence, fold in rows}


def fold_table(
    fold_of: Mapping[tuple[str, str], int],
    counts: pd.DataFrame,
    classes: list[int],
) -> pd.DataFrame:
    """Return ``cv_folds.parquet``: the fold and the rows of each dealt entry.

    Args:
        fold_of: The fold of each entry that was dealt.
        counts: The rows of an entry per class, indexed by ``(group, sequence)``,
            one column per class label. An entry or a class it lacks counts 0.
        classes: The class labels, in the order of their columns.

    Returns:
        One row per dealt entry, sorted by ``(group, sequence)``, with ``group``,
        ``sequence``, ``fold``, ``n_rows`` and a ``rows_<class>`` column per class.
    """
    entries = pd.MultiIndex.from_tuples(sorted(fold_of), names=["group", "sequence"])
    per_class = counts.reindex(index=entries, columns=classes, fill_value=0)
    table = per_class.reset_index()
    table.columns = pd.Index(["group", "sequence", *[f"rows_{c}" for c in classes]])
    table.insert(2, "fold", [fold_of[(g, s)] for g, s in entries])
    table.insert(3, "n_rows", per_class.sum(axis=1).to_numpy())
    return table.astype(
        {name: np.int64 for name in table.columns if name not in ("group", "sequence")}
    )


def check_folds(
    feature: str,
    folds: int,
    table: pd.DataFrame,
    classes: list[int],
    *,
    rows: str,
    per_fold: Callable[[int, npt.NDArray[np.intp], npt.NDArray[np.intp]], None]
    | None = None,
) -> None:
    """Refuse a fold that cannot be fitted without, and report one that will fit thinly.

    A fold whose training rows contain fewer than two classes is refused. A
    class that is in fewer sequences than there are folds, and a fold whose
    training rows are under half of a class's rows, are warned about, because
    the model of that fold sees the class thinly or not at all.

    Args:
        feature: The classifier, which a refusal and a warning name.
        folds: The number of folds.
        table: The result of :func:`fold_table`.
        classes: The class labels.
        rows: What *table* counts, such as ``"template rows"``, for the messages.
        per_fold: Called for each fold, after its two-class check, with the
            fold, the classes present outside it and their row counts. It
            raises to refuse a fold for a reason of the classifier's own.

    Raises:
        ValueError: A fold's training rows contain fewer than two classes, or
            *per_fold* raises.
    """
    for fold in range(folds):
        outside = table.loc[table["fold"] != fold, [f"rows_{c}" for c in classes]]
        counts = outside.sum(axis=0).to_numpy(dtype=np.intp)
        present = np.asarray(classes, dtype=np.intp)[counts > 0]
        if len(present) < 2:
            msg = (
                f"[{feature}] cv_folds={folds}: the train rows outside fold {fold} "
                f"hold {len(present)} of the {len(classes)} classes "
                f"({present.tolist()}), and a classifier needs two. Lower "
                "cv_folds, or draw the templates from more sequences of "
                "each class"
            )
            raise ValueError(msg)
        if per_fold is not None:
            per_fold(fold, present, counts[counts > 0])

    thin: list[str] = []
    for cls in classes:
        column = table[f"rows_{cls}"]
        n_sequences = int((column > 0).sum())
        if n_sequences < folds:
            log.warning(
                "[%s] class %d has %s in %d sequence%s, fewer than "
                "cv_folds=%d, so the models of some fold never see it and "
                "give it probability 0 there (rows_%d of cv_folds.parquet "
                "shows where)",
                feature,
                cls,
                rows,
                n_sequences,
                "" if n_sequences == 1 else "s",
                folds,
                cls,
            )
        total = int(column.sum())
        for fold in range(folds):
            outside_rows = total - int(column[table["fold"] == fold].sum())
            if total > 0 and 2 * outside_rows < total:
                thin.append(f"fold {fold} class {cls} ({outside_rows} of {total})")
    if thin:
        log.warning(
            "[%s] the training rows of a fold hold under half of a class's "
            "rows: %s. That fold's model is fitted on a thin remainder "
            "of the class, so its score is noisy",
            feature,
            ", ".join(thin),
        )


# --- Fitting and saving ---


class FitReporting:
    """Gives a classifier the progress and cancel hooks that ``run_feature`` wires.

    ``run_feature`` sets ``progress_callback`` and ``cancel_token`` on a feature
    that has them, so that a fit reports each model it trains and stops between
    two. A classifier that fits several models calls :meth:`_before_fit` and
    :meth:`_after_fit` around each.
    """

    progress_callback: ProgressCallback | None = None
    cancel_token: CancelToken | None = None

    def _before_fit(self, done: int, total: int, *, whole: str) -> None:
        """Stop if a cancel was asked for, and name the fit that starts.

        Args:
            done: The fits finished so far.
            total: The fits in all, the last of them on every row.
            whole: What the last fit is called.
        """
        if self.cancel_token is not None:
            self.cancel_token.raise_if_cancelled()
        if self.progress_callback is not None and total > 1:
            what = f"fold {done + 1} of {total - 1}" if done < total - 1 else whole
            self.progress_callback.on_phase("fit", what)

    def _after_fit(self, done: int, total: int) -> None:
        """Report that *done* of *total* fits have finished."""
        if self.progress_callback is not None and total > 1:
            self.progress_callback.on_epoch_end(done - 1, total, {})


def write_bundle(path: Path, bundle: object) -> None:
    """Write a classifier's bundle to *path* atomically.

    The bundle goes last, after the files that belong to it, because its presence
    makes the run a cache. A bundle of several models takes a while to write, and
    a kill or a second invocation of the run must not leave ``load_state`` a
    partial file.
    """
    import joblib

    atomic_write(path, lambda temporary: joblib.dump(bundle, temporary))


# --- Class weights ---


def balanced_class_weights(counts: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
    """Return the inverse-frequency weight of each class: ``total / (classes * count)``.

    A class with no row has weight 0.

    Args:
        counts: The rows of each class, in the order of the classes.
    """
    return np.where(
        counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0
    )


# --- Entries ---


def refuse_entries_outside_scope(
    feature: str,
    who: str,
    expected: Collection[tuple[str, str]],
    labeled: Container[tuple[str, str]],
    seen: Collection[tuple[str, str]],
    *,
    consequence: str,
    remedy: str,
) -> None:
    """Refuse a fit that is missing an entry it was fitted for.

    A fit that streams its inputs sees the entries of its scope. An entry that the
    fit names, that has labels, and that the stream did not deliver would leave a
    model fitted on fewer entries than its inputs say.

    Args:
        feature: The feature, which the refusal names.
        who: What *expected* are, as the refusal says "N <who> have labels".
        expected: The entries the fit is for.
        labeled: The entries that have a label file.
        seen: The entries that the stream delivered.
        consequence: What the fit would then do.
        remedy: What to do about it.

    Raises:
        ValueError: An entry of *expected* has labels and is not in *seen*.
    """
    absent = sorted(
        entry for entry in expected if entry in labeled and entry not in seen
    )
    if not absent:
        return
    shown = ", ".join(f"{group}/{sequence}" for group, sequence in absent[:5])
    msg = (
        f"[{feature}] {len(absent)} {who} have labels and are not in this step's "
        f"scope: {shown}. {consequence} {remedy}"
    )
    raise ValueError(msg)
