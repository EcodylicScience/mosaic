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
import re
from dataclasses import dataclass
from collections.abc import (
    Callable,
    Collection,
    Container,
    Iterable,
    Mapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    TypedDict,
)

import numpy as np
import pandas as pd
from pydantic import AfterValidator, Field

from mosaic.core.entry import Entry, entry_token, parse_entry_tokens
from mosaic.core.evaluation import CountScores, pool_entry_counts
from mosaic.core.params import Declared
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.types import COLUMNS, ParquetArtifact, ParquetLoadSpec
from mosaic.core.track_quality import consecutive_runs

from .helpers import (
    cv_block_assignment,
    cv_block_bounds,
    cv_fold_assignment,
    meta_columns,
    out_of_fold_choice,
    scored_entry,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from mosaic.core.pipeline.job import CancelToken
    from mosaic.core.pipeline.progress import ProgressCallback

__all__ = [
    "DERIVED_FIT_SCOPE_REMEDY",
    "FIT_SCOPE_REMEDY",
    "CV_SEED_DESCRIPTION",
    "DECISION_THRESHOLD_DESCRIPTION",
    "DEFAULT_CLASS_DESCRIPTION",
    "PROBABILITY_COLUMN",
    "THRESHOLDS_FILE",
    "TUNED_DECISION_THRESHOLD_DESCRIPTION",
    "CV_BLOCKS_PER_SEQUENCE_DESCRIPTION",
    "CV_GAP_FRAMES_DESCRIPTION",
    "CV_GROUP_BY_DESCRIPTION",
    "CV_UNIT_DESCRIPTION",
    "CvBlock",
    "CvBlocksPerSequence",
    "CvRecord",
    "CvGapFrames",
    "CvGroupBy",
    "CvSeed",
    "CvSettings",
    "CvUnit",
    "DecisionThreshold",
    "FitReporting",
    "FoldLookup",
    "LabeledRun",
    "RunFolds",
    "RunningMoments",
    "ThresholdsArtifact",
    "TunedDecisionThreshold",
    "assemble_prediction",
    "balanced_class_weights",
    "block_fold_table",
    "block_list",
    "block_rows",
    "check_cross_validation",
    "check_folds",
    "cv_record",
    "deal_blocks",
    "deal_folds",
    "deal_runs",
    "drop_unused_cv_settings",
    "fold_map",
    "fold_rows",
    "fold_table",
    "held_out_frames",
    "note_labeled_range",
    "pooled_class_scores",
    "predicted_labels",
    "probability_classes",
    "read_fold_table",
    "read_thresholds",
    "refuse_entries_outside_scope",
    "scoring_folds",
    "standardize",
    "stored_folds",
    "write_bundle",
]

log = logging.getLogger(__name__)

PROBABILITY_COLUMN: Final = re.compile(r"^prob_(-?[0-9]+)$")
"""The name of a class probability column, ``prob_<class>``."""


def probability_classes(columns: Iterable[object]) -> list[int]:
    """Return the classes of the ``prob_<class>`` columns of *columns*, ascending.

    A member column of an input merge, ``prob_<class>__<i>``, does not match.
    Returns ``[]`` when no column matches.
    """
    return sorted(
        {
            int(match[1])
            for name in columns
            if (match := PROBABILITY_COLUMN.match(str(name))) is not None
        }
    )


# --- Parameters ---

FIT_SCOPE_REMEDY: Final = (
    "The step's first run must be asked for over a scope that holds the training "
    "entries, because it fits on them. Once it has fitted, a wider scope applies "
    "the stored fit to the new entries and fits nothing again. To apply a fit "
    "from elsewhere, name it through `model`."
)

DERIVED_FIT_SCOPE_REMEDY: Final = (
    "The step's first run must be asked for over a scope that holds the "
    "entries of its input's cv_folds.parquet, because it fits on their "
    "out-of-fold output. Once it has fitted, a wider scope applies the stored "
    "result to the new entries and fits nothing again."
)

DECISION_THRESHOLD_DESCRIPTION: Final = (
    "Unset, the predicted class is the class with the highest probability. "
    "A float applies one probability threshold to every class, and a "
    "mapping from class to threshold applies a per-class one. A probability "
    "under its class's threshold counts as 0, so a mapping that omits a "
    "class gives that class threshold 0, which it always clears. Any row "
    "where no class clears its threshold is assigned default_class. To "
    "predict the positive class of a binary problem from probability t up, "
    'give the other class threshold 1.0: {"0": 1.0, "1": t}. {"1": t} alone '
    "changes nothing for a t under 0.5, because class 0 still wins wherever "
    "its probability is the higher."
)

DEFAULT_CLASS_DESCRIPTION: Final = (
    "The class assigned to a row where decision_threshold is set and no "
    "class clears its threshold. A mapping that omits a class gives it "
    "threshold 0, so only a class listed with a threshold above 0 can fail "
    "to claim a row. A row with no probabilities takes it too, whether or "
    "not decision_threshold is set."
)

TUNED_DECISION_THRESHOLD_DESCRIPTION: Final = (
    DECISION_THRESHOLD_DESCRIPTION
    + " A {step: <id>} reference names a tune-threshold step and applies the "
    "thresholds it saved, which keep default_class out of the competition for a "
    "row. Give default_class the value that step has. That step reads a classifier "
    "run, so a classifier cannot name its own tuning."
)

THRESHOLDS_FILE: Final = "thresholds.parquet"
"""What ``tune-threshold`` calls the thresholds it saves, one row per class."""

CV_SEED_DESCRIPTION: Final = (
    "The seed that deals the training sequences into folds. It changes the "
    "folds and nothing else, so runs that differ in random_state are "
    "compared on the same folds. Other than 0 only with cv_folds."
)

DecisionThreshold = Annotated[
    float | Mapping[int, float] | None, Declared(DECISION_THRESHOLD_DESCRIPTION)
]
"""The type of a classifier's ``decision_threshold`` parameter."""


class ThresholdsArtifact(ParquetArtifact):
    """The ``thresholds.parquet`` of a ``tune-threshold`` run: a ``class`` and a ``threshold`` per row."""

    feature: str = "tune-threshold"
    pattern: str = THRESHOLDS_FILE
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


TunedDecisionThreshold = Annotated[
    float | Mapping[int, float] | ThresholdsArtifact | None,
    Declared(TUNED_DECISION_THRESHOLD_DESCRIPTION),
]
"""The type of a ``decision_threshold`` that may also name a ``tune-threshold`` run."""

CvSeed = Annotated[int, Declared(CV_SEED_DESCRIPTION)]
"""The type of a classifier's ``cv_seed`` parameter."""

CV_UNIT_DESCRIPTION: Final = (
    "What cross-validation holds out. sequence holds out whole training "
    "sequences, so a fold's models never see a held-out sequence. block cuts "
    "each labeled sequence into cv_blocks_per_sequence contiguous blocks of "
    "frames, deals the blocks into the folds, and holds one block out at a "
    "time, so that a dataset with one training sequence still has a "
    "cross-validated score. A block's frames are not independent of the "
    "frames beside it, so block is optimistic where sequences are the unit "
    "you generalize over; use cv_gap_frames to drop the frames around a "
    "held-out block from training. Only with cv_folds."
)

CV_BLOCKS_PER_SEQUENCE_DESCRIPTION: Final = (
    "With cv_unit block, the blocks that each training sequence is cut into, "
    "at least 2. A sequence's blocks span its labeled frames, from the first "
    "labeled frame of any of its series to the last, and differ in length by "
    "at most one frame. Unset, the number of cv_folds, so that a single "
    "training sequence has one block per fold. The blocks of all the sequences "
    "are dealt into the folds together, so with several sequences a fold holds "
    "a near-equal count of blocks, not one block of each sequence. Only with "
    "cv_unit block."
)

CV_GAP_FRAMES_DESCRIPTION: Final = (
    "With cv_unit block, the width of the margin on each side of a held-out "
    "block that the model of that fold does not train on, so that a window or a "
    "network receptive field reaching across the block's edge does not carry its "
    "labels into training. The model of a fold trains on no frame of its "
    "held-out blocks or of this margin, and scores only what lies inside its "
    "own blocks. Only with cv_unit block."
)

CvUnit = Annotated[Literal["sequence", "block"], Declared(CV_UNIT_DESCRIPTION)]
"""The type of a classifier's ``cv_unit`` parameter."""

CvBlocksPerSequence = Annotated[
    int | None, Field(ge=2), Declared(CV_BLOCKS_PER_SEQUENCE_DESCRIPTION)
]
"""The type of a classifier's ``cv_blocks_per_sequence`` parameter."""

CvGapFrames = Annotated[
    int, Field(ge=0), Declared(CV_GAP_FRAMES_DESCRIPTION, unit="frames")
]
"""The type of a classifier's ``cv_gap_frames`` parameter."""

CV_GROUP_BY_DESCRIPTION: Final = (
    "Ties training entries together, so that a fold holds out together the "
    "entries of one animal, one session or one recording. A mapping from an "
    "entry, group:sequence or a bare sequence of the empty group, to one tie "
    "key or a list of them. Entries that share a key fall into one fold, and so "
    "do entries linked through a chain of shared keys among the training "
    "entries. A training entry the mapping does not list is a tie by itself, "
    "and an entry outside the training set may be listed and is ignored, with a "
    "warning counting such entries. The per-entry animal and session metadata "
    "lives in the sequence tags of mosaic-api, and this mapping passes it to the "
    "folds. Each tie goes whole to the fold with the fewest entries so far, the "
    "largest first, so folds may differ in size by more than one entry, and "
    "cv_folds may be at most the number of ties. Only with cv_folds and cv_unit "
    "sequence."
)


def _canonical_group_by(
    value: dict[str, str | list[str]] | None,
) -> dict[str, str | list[str]] | None:
    """Return *value* in one spelling: entry tokens sorted, each with a sorted list of its keys.

    ``:s`` and ``s`` name one entry, so their keys merge. An empty mapping ties
    nothing and reads as unset.
    """
    if not value:
        return None
    merged: dict[str, set[str]] = {}
    for token, keys in value.items():
        (entry,) = parse_entry_tokens([token])
        merged.setdefault(entry_token(*entry), set()).update(
            [keys] if isinstance(keys, str) else keys
        )
    return {token: sorted(merged[token]) for token in sorted(merged)}


CvGroupBy = Annotated[
    dict[str, str | list[str]] | None,
    AfterValidator(_canonical_group_by),
    Declared(CV_GROUP_BY_DESCRIPTION),
]
"""The type of a classifier's ``cv_group_by`` parameter."""


def drop_unused_cv_settings(
    dumped: dict[str, object],
    cv_unit: Literal["sequence", "block"],
    cv_group_by: Mapping[str, str | list[str]] | None,
) -> dict[str, object]:
    """Return *dumped* without the cross-validation settings that the run leaves unused.

    A classifier's ``identity_dump`` calls this, so that a run hashes as it did
    before a setting it does not use existed: the block settings when the unit is
    the sequence, and ``cv_group_by`` when it is unset. ``check_cross_validation``
    leaves the block settings at their defaults under the sequence unit, so
    leaving them out loses nothing.

    Args:
        dumped: The hashed params of the classifier, which this changes in place.
        cv_unit: The classifier's ``cv_unit``.
        cv_group_by: The classifier's ``cv_group_by``.

    Returns:
        *dumped*.
    """
    if cv_unit == "sequence":
        for name in ("cv_unit", "cv_blocks_per_sequence", "cv_gap_frames"):
            del dumped[name]
    if cv_group_by is None:
        del dumped["cv_group_by"]
    return dumped


def check_cross_validation(
    cv_folds: int | None,
    cv_seed: int,
    *,
    model_given: bool,
    cv_unit: Literal["sequence", "block"] = "sequence",
    cv_blocks_per_sequence: int | None = None,
    cv_gap_frames: int = 0,
    cv_group_by: Mapping[str, str | list[str]] | None = None,
) -> int | None:
    """Refuse cross-validation settings that cannot describe one fit.

    ``cv_folds`` fits fold models from templates, and ``cv_seed`` only deals
    their folds. A ``cv_seed`` other than 0 without folds would give the same
    output a second run id, since the id hashes both, and so would a setting of
    the block unit that the sequence unit ignores.

    Args:
        cv_folds: The number of folds, unset for none.
        cv_seed: The seed that deals them.
        model_given: Whether the parameters name a pre-fitted ``model``.
        cv_unit: What a fold holds out: whole sequences or blocks of frames.
        cv_blocks_per_sequence: The blocks that a sequence is cut into, unset
            for ``cv_folds`` blocks.
        cv_gap_frames: The frames around a held-out block that its fold does
            not train on.
        cv_group_by: The tie keys of each entry, which keep tied entries in one
            fold of whole entries.

    Returns:
        The blocks per sequence the fit cuts: the one given, else ``cv_folds``
        under the block unit, ``None`` under the sequence unit. A caller stores it
        back, so that naming the default and leaving it out are one run id.

    Raises:
        ValueError: ``cv_seed``, a block setting or ``cv_group_by`` is set
            without what it needs, ``cv_group_by`` is set under the block unit,
            or ``cv_folds`` is set beside a ``model``.
    """
    block_settings = [
        name
        for name, value in (
            ("cv_blocks_per_sequence", cv_blocks_per_sequence),
            ("cv_gap_frames", cv_gap_frames),
        )
        if value not in (None, 0)
    ]
    if cv_folds is None:
        if cv_seed != 0:
            msg = "cv_seed deals the folds of cv_folds, which is not set"
            raise ValueError(msg)
        unset = ["cv_unit"] if cv_unit != "sequence" else block_settings
        if unset:
            msg = f"{', '.join(unset)} cut the folds of cv_folds, which is not set"
            raise ValueError(msg)
        if cv_group_by is not None:
            msg = (
                "cv_group_by ties entries into the folds of cv_folds, which is not set"
            )
            raise ValueError(msg)
        return None
    if model_given:
        msg = (
            "cv_folds fits fold models from templates, so it needs "
            "templates and is refused beside model"
        )
        raise ValueError(msg)
    if cv_unit == "sequence":
        if block_settings:
            msg = f"{', '.join(block_settings)} only apply with cv_unit block"
            raise ValueError(msg)
        return None
    if cv_group_by is not None:
        msg = (
            "cv_group_by ties whole entries into one fold, and cv_unit block deals "
            "blocks of frames. Use cv_group_by with cv_unit sequence"
        )
        raise ValueError(msg)
    return cv_folds if cv_blocks_per_sequence is None else cv_blocks_per_sequence


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


def read_thresholds(path: Path) -> dict[int, float]:
    """Return the ``{class: threshold}`` mapping a ``tune-threshold`` run saved at *path*.

    Args:
        path: A ``thresholds.parquet`` with a ``class`` and a ``threshold`` column.
    """
    frame = pd.read_parquet(path)
    return {
        int(cls): float(limit)
        for cls, limit in zip(frame["class"], frame["threshold"], strict=True)
    }


def pooled_class_scores(
    truth: npt.NDArray[np.intp],
    predicted: npt.NDArray[np.intp],
    labels: Sequence[int],
    average_over: Sequence[int] | None,
) -> CountScores:
    """Score predictions by their pooled counts, with the formula of ``evaluate-labels``.

    Args:
        truth: The position in *labels* of each row's true class.
        predicted: The position in *labels* of each row's predicted class.
        labels: Every class a row can hold or be predicted as, in order.
        average_over: The classes ``macro_f1`` averages over. Unset, every class
            with a true row.

    Returns:
        The result of :func:`~mosaic.core.evaluation.pool_entry_counts` over the
        rows: ``macro_f1``, ``accuracy`` and ``per_class`` with each class's ``f1``.
    """
    size = len(labels)
    confusion = np.bincount(truth * size + predicted, minlength=size * size).reshape(
        size, size
    )
    hits = np.diag(confusion)
    counts: dict[str, int] = {}
    for position, label in enumerate(labels):
        counts[f"tp_{label}"] = int(hits[position])
        counts[f"fp_{label}"] = int(confusion[:, position].sum() - hits[position])
        counts[f"fn_{label}"] = int(confusion[position].sum() - hits[position])
    return pool_entry_counts(pd.DataFrame([counts]), average_over=average_over)


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


def _tie_components(
    feature: str,
    train_entries: Iterable[Entry],
    group_by: Mapping[str, str | list[str]],
) -> dict[Entry, Entry]:
    """Return the tie of each train entry: the smallest entry linked to it by shared keys.

    Two train entries that share a key are linked, and so are the entries of a
    chain of links. An entry outside the train entries links nothing, and the
    entries the mapping names outside them are counted in a warning.
    """
    root = {entry: entry for entry in train_entries}

    def find(entry: Entry) -> Entry:
        while root[entry] != entry:
            entry = root[entry]
        return entry

    first_of_key: dict[str, Entry] = {}
    outside: set[Entry] = set()
    for token, keys in group_by.items():
        (entry,) = parse_entry_tokens([token])
        if entry not in root:
            outside.add(entry)
            continue
        for key in [keys] if isinstance(keys, str) else keys:
            mine, theirs = find(entry), find(first_of_key.setdefault(key, entry))
            root[max(mine, theirs)] = min(mine, theirs)
    if outside:
        log.warning(
            "[%s] cv_group_by names %d entries outside the train entries, and "
            "they tie no train entries together",
            feature,
            len(outside),
        )
    return {entry: find(entry) for entry in root}


def deal_folds(
    feature: str,
    train_entries: list[Entry],
    folds: int,
    seed: int,
    group_by: Mapping[str, str | list[str]] | None = None,
) -> dict[Entry, int]:
    """Deal the train entries into *folds* folds, keeping tied entries together.

    Wraps :func:`~mosaic.behavior.feature_library.helpers.cv_fold_assignment`,
    so that every classifier deals alike. The ties are the components of the
    train entries that *group_by* links through shared keys.

    Args:
        feature: The classifier, which a refusal and a warning name.
        train_entries: The entries to deal.
        folds: The number of folds.
        seed: The seed that deals them.
        group_by: The tie keys of each entry token, as ``cv_group_by`` holds
            them. Unset, every entry is a tie by itself.

    Raises:
        ValueError: Naming *feature*, when *folds* is below 2 or above the
            number of ties.
    """
    ties = (
        None if group_by is None else _tie_components(feature, train_entries, group_by)
    )
    try:
        return cv_fold_assignment(train_entries, folds, seed, ties)
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


class CvBlock(NamedTuple):
    """One block of a sequence that cross-validation holds out: its frames and its fold.

    The block is the frames ``start`` up to ``stop`` (half-open, in the ``frame``
    column) of the entry ``(group, sequence)``, and ``block`` is its index among
    the blocks the sequence was cut into.
    """

    group: str
    sequence: str
    block: int
    start: int
    stop: int
    fold: int


def deal_blocks(
    feature: str,
    ranges: Mapping[tuple[str, str], tuple[int, int]],
    blocks_per_sequence: int,
    folds: int,
    seed: int,
) -> list[CvBlock]:
    """Cut each entry's labeled frames into blocks and deal the blocks into *folds* folds.

    Wraps :func:`~mosaic.behavior.feature_library.helpers.cv_block_bounds` and
    :func:`~mosaic.behavior.feature_library.helpers.cv_block_assignment`, so that
    every classifier cuts and deals alike. The result depends on the ranges, the
    block count, the fold count and the seed alone.

    Args:
        feature: The classifier, which a refusal names.
        ranges: The labeled frames of each entry to cut, as ``(first, stop)``: the
            first labeled frame of any of its series and one past the last.
        blocks_per_sequence: The blocks each entry is cut into.
        folds: The number of folds.
        seed: The seed that deals the blocks.

    Returns:
        Every block, sorted by ``(group, sequence, block)``.

    Raises:
        ValueError: Naming *feature*, when *folds* is below 2 or above the number
            of blocks.
    """
    bounds = {
        (group, sequence, index): (start, stop)
        for (group, sequence), (first, last) in ranges.items()
        for index, start, stop in cv_block_bounds(first, last, blocks_per_sequence)
    }
    try:
        fold_of = cv_block_assignment(bounds, folds, seed)
    except ValueError as exc:
        msg = f"[{feature}] {exc}"
        raise ValueError(msg) from exc
    return [
        CvBlock(group, sequence, index, start, stop, fold_of[(group, sequence, index)])
        for (group, sequence, index), (start, stop) in sorted(bounds.items())
    ]


def block_rows(blocks: Sequence[CvBlock]) -> list[tuple[str, str, int, int, int, int]]:
    """Return the blocks as plain tuples, ``(group, sequence, block, start, stop, fold)``, as a bundle stores them."""
    return [(b.group, b.sequence, b.block, b.start, b.stop, b.fold) for b in blocks]


def block_list(rows: Sequence[Sequence[object]]) -> list[CvBlock]:
    """Return the blocks of the rows :func:`block_rows` wrote."""
    return [CvBlock(*row) for row in rows]  # pyright: ignore[reportArgumentType]


def held_out_frames(
    blocks: Sequence[CvBlock], fold: int, gap: int
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Return the frames a fold's model does not train on, per entry, as half-open ranges.

    Each block of *fold* is held out together with *gap* frames on both sides.

    Args:
        blocks: Every block, with its fold.
        fold: The fold whose model is fitted.
        gap: The frames either side of a held-out block that are held out too.
    """
    held: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for block in blocks:
        if block.fold == fold:
            held.setdefault((block.group, block.sequence), []).append(
                (block.start - gap, block.stop + gap)
            )
    return held


def block_fold_table(
    blocks: Sequence[CvBlock], counts: pd.DataFrame, classes: list[int]
) -> pd.DataFrame:
    """Return ``cv_folds.parquet`` of block cross-validation: the fold and the rows of each block.

    Args:
        blocks: Every block, sorted by ``(group, sequence, block)``.
        counts: The rows of a block per class, indexed by ``(group, sequence,
            block)``, one column per class label. A block or a class it lacks
            counts 0.
        classes: The class labels, in the order of their columns.

    Returns:
        One row per block with ``group``, ``sequence``, ``block``, ``start``,
        ``stop``, ``fold``, ``n_rows`` and a ``rows_<class>`` column per class.
        ``start`` and ``stop`` are the frames of the block, half-open.
    """
    index = pd.MultiIndex.from_tuples(
        [(b.group, b.sequence, b.block) for b in blocks],
        names=["group", "sequence", "block"],
    )
    per_class = counts.reindex(index=index, columns=classes, fill_value=0)
    table = per_class.reset_index()
    table.columns = pd.Index(
        ["group", "sequence", "block", *[f"rows_{c}" for c in classes]]
    )
    table.insert(3, "start", [b.start for b in blocks])
    table.insert(4, "stop", [b.stop for b in blocks])
    table.insert(5, "fold", [b.fold for b in blocks])
    table.insert(6, "n_rows", per_class.sum(axis=1).to_numpy())
    return table.astype(
        {name: np.int64 for name in table.columns if name not in ("group", "sequence")}
    )


class LabeledRun(Protocol):
    """A stretch of consecutive labeled frames of one series, as the fold tables count it."""

    @property
    def entry(self) -> tuple[str, str]:
        """The ``(group, sequence)`` it comes from."""
        ...

    @property
    def y(self) -> npt.NDArray[np.intp]:
        """The class index of each frame, in the order of the classes."""
        ...

    @property
    def first_frame(self) -> int | None:
        """The ``frame`` of its first row, set where runs are cut by block."""
        ...


def _runs_fold_table(
    fold_of: Mapping[tuple[str, str], int],
    runs: Iterable[LabeledRun],
    classes: list[int],
) -> pd.DataFrame:
    """Return ``cv_folds.parquet`` by sequence: the fold of every train entry and its labeled frames.

    Args:
        fold_of: The fold of each train entry.
        runs: The training runs, whose frames are counted.
        classes: The class labels, which the runs' class indices point into.
    """
    frame = pd.DataFrame(
        [
            (run.entry[0], run.entry[1], classes[int(c)], int(n))
            for run in runs
            for c, n in zip(*np.unique(run.y, return_counts=True))
        ],
        columns=["group", "sequence", "label", "count"],
    )
    counts = (
        frame.groupby(["group", "sequence", "label"])["count"]
        .sum()
        .unstack(fill_value=0)
    )
    return fold_table(fold_of, counts, classes)


def _runs_block_table(
    blocks: Sequence[CvBlock], runs: Iterable[LabeledRun], classes: list[int]
) -> pd.DataFrame:
    """Return ``cv_folds.parquet`` by block: the fold of every block and its labeled frames.

    Args:
        blocks: Every block, sorted by ``(group, sequence, block)``.
        runs: The training runs, each with its ``first_frame``.
        classes: The class labels, which the runs' class indices point into.
    """
    in_entry: dict[tuple[str, str], list[CvBlock]] = {}
    for block in blocks:
        in_entry.setdefault((block.group, block.sequence), []).append(block)
    counts = {
        (block.group, block.sequence, block.block): np.zeros(
            len(classes), dtype=np.int64
        )
        for block in blocks
    }
    for run in runs:
        assert run.first_frame is not None
        of_entry = in_entry[run.entry]
        starts = np.array([block.start for block in of_entry], dtype=np.int64)
        frames = run.first_frame + np.arange(len(run.y))
        where = np.clip(np.searchsorted(starts, frames, side="right") - 1, 0, None)
        tally = np.bincount(
            where * len(classes) + run.y, minlength=len(of_entry) * len(classes)
        ).reshape(len(of_entry), len(classes))
        for position, block in enumerate(of_entry):
            counts[(block.group, block.sequence, block.block)] += tally[position]
    frame = pd.DataFrame(
        list(counts.values()),
        index=pd.MultiIndex.from_tuples(
            list(counts), names=["group", "sequence", "block"]
        ),
        columns=classes,
    )
    return block_fold_table(blocks, frame, classes)


def _pieces_outside(
    first_frame: int,
    length: int,
    held: Sequence[tuple[int, int]],
    min_length: int,
) -> list[tuple[int, int]]:
    """Return the pieces of a run that lie outside the held-out frames.

    Args:
        first_frame: The ``frame`` of the run's first row, which the rows then
            follow one by one.
        length: The rows of the run.
        held: The frames a fold's model does not train on, as half-open ranges.
        min_length: The fewest rows a piece keeps.

    Returns:
        Each piece of *min_length* rows or more, as half-open row positions in
        the run.
    """
    frames = first_frame + np.arange(length)
    keep = np.ones(length, dtype=bool)
    for first, stop in held:
        keep &= (frames < first) | (frames >= stop)
    return [
        (start, stop)
        for start, stop in consecutive_runs(frames, keep)
        if stop - start >= min_length
    ]


def note_labeled_range(
    ranges: dict[tuple[str, str], tuple[int, int]],
    entry: tuple[str, str],
    frames: npt.NDArray[np.int64],
    labeled: npt.NDArray[np.bool_],
) -> None:
    """Widen *entry*'s labeled frames in *ranges* to cover the labeled *frames* of one series.

    A range is ``(first, stop)``: the first labeled frame of any of the entry's
    series and one past the last, as :func:`deal_blocks` cuts it.

    Args:
        ranges: The labeled frames of each entry so far, which this changes in place.
        entry: The ``(group, sequence)`` the series belongs to.
        frames: The ``frame`` of each row of the series.
        labeled: Which rows a label covers.
    """
    if not labeled.any():
        return
    hit = frames[labeled]
    first, stop = ranges.get(entry, (int(hit.min()), 0))
    ranges[entry] = (min(first, int(hit.min())), max(stop, int(hit.max()) + 1))


_BLOCK_FOLD_REMEDY: Final = (
    "Lower cv_gap_frames or cv_folds, or cut fewer, larger blocks"
)
"""What to change when a block fold's training runs hold fewer than two classes."""


class CvSettings(Protocol):
    """The cross-validation parameters of a classifier that fits on runs of frames."""

    @property
    def cv_folds(self) -> int | None:
        """The number of folds, unset for none."""
        ...

    @property
    def cv_seed(self) -> int:
        """The seed that deals the folds."""
        ...

    @property
    def cv_unit(self) -> Literal["sequence", "block"]:
        """What a fold holds out."""
        ...

    @property
    def cv_blocks_per_sequence(self) -> int | None:
        """The blocks each entry is cut into, set under the block unit."""
        ...

    @property
    def cv_gap_frames(self) -> int:
        """The frames on each side of a held-out block that its fold does not train on."""
        ...

    @property
    def cv_group_by(self) -> Mapping[str, str | list[str]] | None:
        """The tie keys of each entry."""
        ...


@dataclass(frozen=True)
class RunFolds[R: LabeledRun]:
    """The folds a fit on runs of frames deals, and what each fold's model trains on.

    ``fold_of`` holds the fold of each entry by sequence, and ``blocks`` the
    folded blocks by block. ``training[k]`` holds the pieces of runs that the
    model without fold ``k`` trains on, each as ``(run, start, stop)`` with
    half-open row positions in the run. ``table`` is ``cv_folds.parquet``.
    """

    table: pd.DataFrame
    fold_of: dict[tuple[str, str], int]
    blocks: list[CvBlock]
    training: list[list[tuple[R, int, int]]]


def deal_runs[R: LabeledRun](
    feature: str,
    settings: CvSettings,
    runs: Sequence[R],
    classes: list[int],
    *,
    train_entries: list[Entry],
    ranges: Mapping[tuple[str, str], tuple[int, int]],
    min_length: int,
    multiple: int = 1,
    block_remedy: str = _BLOCK_FOLD_REMEDY,
) -> RunFolds[R]:
    """Deal the folds of a fit on runs of labeled frames, and name what each fold's model trains on.

    By sequence, a fold's model trains on the whole runs of every entry outside
    the fold. By block, it trains on every entry's runs less the frames of its
    own blocks and ``cv_gap_frames`` on each side of them. A piece of a run that
    this leaves is cut to a whole number of *multiple* rows from its first row,
    and kept when it holds *min_length* rows or more.

    Args:
        feature: The classifier, which a refusal and a warning name.
        settings: The classifier's params, with ``cv_folds`` set.
        runs: The training runs, each with the class index of its frames, and
            with its ``first_frame`` under the block unit.
        classes: The class labels, which the runs' class indices point into.
        train_entries: The train entries to deal by sequence.
        ranges: The labeled frames of each entry to cut into blocks, as
            :func:`note_labeled_range` gathers them, under the block unit.
        min_length: The fewest rows a piece of a run keeps.
        multiple: The rows a piece is a whole number of, 1 to keep it as cut.
        block_remedy: What the refusal of a block fold left with fewer than two
            classes tells the user to change.

    Raises:
        ValueError: The folds cannot be dealt, or a fold's training frames hold
            fewer than two classes.
    """
    folds = settings.cv_folds
    assert folds is not None
    if settings.cv_unit == "sequence":
        fold_of = deal_folds(
            feature, train_entries, folds, settings.cv_seed, settings.cv_group_by
        )
        table = _runs_fold_table(fold_of, runs, classes)
        check_folds(feature, folds, table, classes, rows="labeled frames")
        training = [
            [(run, 0, len(run.y)) for run in runs if fold_of[run.entry] != fold]
            for fold in range(folds)
        ]
        return RunFolds(table, fold_of, [], training)

    assert settings.cv_blocks_per_sequence is not None
    blocks = deal_blocks(
        feature, ranges, settings.cv_blocks_per_sequence, folds, settings.cv_seed
    )
    table = _runs_block_table(blocks, runs, classes)
    training: list[list[tuple[R, int, int]]] = []
    for fold in range(folds):
        held = held_out_frames(blocks, fold, settings.cv_gap_frames)
        pieces: list[tuple[R, int, int]] = []
        for run in runs:
            held_frames = held.get(run.entry)
            if not held_frames:
                pieces.append((run, 0, len(run.y)))
                continue
            assert run.first_frame is not None
            for start, stop in _pieces_outside(
                run.first_frame, len(run.y), held_frames, min_length
            ):
                stop = start + (stop - start) // multiple * multiple
                if stop - start >= min_length:
                    pieces.append((run, start, stop))
        training.append(pieces)

    def enough_classes(
        fold: int, present: npt.NDArray[np.intp], _: npt.NDArray[np.intp]
    ) -> None:
        kept = np.unique(
            np.concatenate(
                [run.y[start:stop] for run, start, stop in training[fold]]
                or [np.empty(0, np.intp)]
            )
        )
        if len(kept) < 2:
            msg = (
                f"[{feature}] holding out the blocks of fold {fold} and "
                f"{settings.cv_gap_frames} frames around them leaves training runs "
                f"of {len(kept)} class (of {len(present)} outside the blocks), and "
                f"a classifier needs two. {block_remedy}"
            )
            raise ValueError(msg)

    check_folds(
        feature,
        folds,
        table,
        classes,
        rows="labeled frames",
        unit="block",
        per_fold=enough_classes,
    )
    return RunFolds(table, {}, blocks, training)


def read_fold_table(path: Path) -> pd.DataFrame:
    """Return the columns of a ``cv_folds.parquet`` that say which rows are out of fold.

    ``group``, ``sequence`` and ``fold``, and ``start`` and ``stop`` when the file
    holds blocks, sorted by entry and block. Two classifiers that deal alike
    return equal tables, whatever they count in the other columns.

    Args:
        path: A ``cv_folds.parquet``.
    """
    table = pd.read_parquet(path)
    by_block = "start" in table.columns
    columns = ["group", "sequence", "fold", *(["start", "stop"] if by_block else [])]
    return table[columns].sort_values(
        ["group", "sequence", *(["start"] if by_block else [])], ignore_index=True
    )


class FoldLookup:
    """The fold of each row of an entry that ``cv_folds.parquet`` deals out.

    A table of entries gives every row of an entry its fold. A table of blocks
    gives a row the fold of the block its ``frame`` falls in, and no fold to a
    frame outside every block. Build one with :meth:`from_table`,
    :meth:`by_entry` or :meth:`by_block`.
    """

    def __init__(
        self,
        entries: Mapping[tuple[str, str], int],
        blocks: Mapping[
            tuple[str, str],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
        ],
    ) -> None:
        self._entries = dict(entries)
        self._blocks = dict(blocks)

    @classmethod
    def by_entry(cls, fold_of: Mapping[tuple[str, str], int]) -> FoldLookup:
        """Return the lookup of whole entries: *fold_of* gives each its fold."""
        return cls(fold_of, {})

    @classmethod
    def by_block(cls, blocks: Iterable[Sequence[object]]) -> FoldLookup:
        """Return the lookup of blocks from ``(group, sequence, ..., start, stop, fold)`` rows.

        A row's first two items name the entry and its last three are the block's
        ``start``, ``stop`` and ``fold``; the items between are ignored.
        """
        grouped: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
        for row in blocks:
            entry = (str(row[0]), str(row[1]))
            grouped.setdefault(entry, []).append(
                (int(row[-3]), int(row[-2]), int(row[-1]))  # pyright: ignore[reportArgumentType]
            )
        arrays: dict[
            tuple[str, str],
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
        ] = {}
        for entry, rows in grouped.items():
            start, stop, fold = (
                np.array(column, dtype=np.int64) for column in zip(*sorted(rows))
            )
            arrays[entry] = (start, stop, fold)
        return cls({}, arrays)

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> FoldLookup:
        """Return the lookup that a ``cv_folds.parquet`` (or :func:`read_fold_table`) describes."""
        if "start" in table.columns:
            return cls.by_block(
                zip(
                    table["group"],
                    table["sequence"],
                    table["start"],
                    table["stop"],
                    table["fold"],
                    strict=True,
                )
            )
        return cls.by_entry(
            {
                (str(group), str(sequence)): int(fold)
                for group, sequence, fold in zip(
                    table["group"], table["sequence"], table["fold"], strict=True
                )
            }
        )

    @property
    def entries(self) -> set[tuple[str, str]]:
        """The entries that have a fold, or a block with one."""
        return set(self._entries) | set(self._blocks)

    def folds(
        self, entry: tuple[str, str], frames: npt.ArrayLike
    ) -> npt.NDArray[np.intp]:
        """Return the fold of each frame of *entry*, ``-1`` for a frame in no fold.

        Args:
            entry: The ``(group, sequence)`` the frames belong to.
            frames: The ``frame`` of each row, in any order.
        """
        frame = np.asarray(frames, dtype=np.int64)
        if entry in self._entries:
            return np.full(frame.shape, self._entries[entry], dtype=np.intp)
        blocks = self._blocks.get(entry)
        if blocks is None:
            return np.full(frame.shape, -1, dtype=np.intp)
        starts, stops, folds = blocks
        at = np.maximum(np.searchsorted(starts, frame, side="right") - 1, 0)
        inside = (frame >= starts[at]) & (frame < stops[at])
        return np.where(inside, folds[at], -1).astype(np.intp)

    def scoring_folds(self, df: pd.DataFrame, feature: str) -> npt.NDArray[np.intp]:
        """Return the fold of each row of the one entry *df* holds, ``-1`` for the full model.

        Args:
            df: One entry's table, with ``group``, ``sequence`` and ``frame``.
            feature: The feature that scores, which a refusal names.

        Raises:
            ValueError: *df* lacks ``group`` or ``sequence``, or holds more than one entry.
        """
        entry = scored_entry(df, feature)
        return self.folds(entry, df[COLUMNS.frame_col].to_numpy())


class CvRecord(TypedDict):
    """The folds of a cross-validated fit, as a classifier's bundle stores them.

    By sequence, ``fold_of`` holds ``(group, sequence, fold)`` for every train entry.
    By block, the keys ``unit``, ``blocks_per_sequence``, ``gap_frames`` and
    ``blocks`` are written too, ``fold_of`` is empty, and ``blocks`` holds
    ``(group, sequence, block, start, stop, fold)`` for every block. A classifier
    whose bundle keeps its fold models beside the folds extends this.
    """

    folds: int
    seed: int
    fold_of: list[tuple[str, str, int]]
    unit: NotRequired[Literal["block"]]
    blocks_per_sequence: NotRequired[int]
    gap_frames: NotRequired[int]
    blocks: NotRequired[list[tuple[str, str, int, int, int, int]]]


def cv_record(
    folds: int,
    seed: int,
    fold_of: Mapping[tuple[str, str], int],
    blocks: Sequence[CvBlock],
    *,
    blocks_per_sequence: int | None,
    gap_frames: int,
) -> CvRecord:
    """Return the folds a fit dealt, as a bundle stores them.

    Args:
        folds: ``cv_folds``.
        seed: ``cv_seed``.
        fold_of: The fold of each train entry, empty by block.
        blocks: Every block and its fold, empty by sequence.
        blocks_per_sequence: ``cv_blocks_per_sequence``, set by block.
        gap_frames: ``cv_gap_frames``.
    """
    cv: CvRecord = {"folds": folds, "seed": seed, "fold_of": fold_rows(fold_of)}
    if blocks and blocks_per_sequence is not None:
        cv["unit"] = "block"
        cv["blocks_per_sequence"] = blocks_per_sequence
        cv["gap_frames"] = gap_frames
        cv["blocks"] = block_rows(blocks)
    return cv


def stored_folds(
    feature: str,
    cv: CvRecord | None,
    cv_unit: Literal["sequence", "block"],
    bundle_file: str,
) -> tuple[dict[tuple[str, str], int], FoldLookup | None]:
    """Return the folds of a cached bundle that a run with ``cv_folds`` scores by.

    Args:
        feature: The classifier, which a refusal names.
        cv: The bundle's folds, ``None`` when it has none.
        cv_unit: The run's ``cv_unit``.
        bundle_file: The bundle's file name, which a refusal says to remove.

    Returns:
        The fold of each train entry and no block lookup by sequence, or no entry
        folds and the block lookup by block.

    Raises:
        RuntimeError: The bundle has no folds, or was dealt by the other unit.
    """
    if cv is None:
        msg = (
            f"[{feature}] cv_folds is set and the cached model has no folds. "
            f"Remove {bundle_file} from the run directory to fit again"
        )
        raise RuntimeError(msg)
    by_block = cv_unit == "block"
    if by_block != (cv.get("unit") == "block"):
        msg = (
            f"[{feature}] cv_unit is {cv_unit!r} and the cached model was fitted "
            f"with the other unit. Remove {bundle_file} from the run directory to "
            "fit again"
        )
        raise RuntimeError(msg)
    if by_block:
        return {}, FoldLookup.by_block(block_list(cv.get("blocks", [])))
    return fold_map(cv["fold_of"]), None


def scoring_folds(
    df: pd.DataFrame,
    fold_of: Mapping[tuple[str, str], int],
    blocks: FoldLookup | None,
    feature: str,
) -> npt.NDArray[np.intp]:
    """Return the fold whose model scores each row of the entry *df* holds, ``-1`` for the full model.

    By sequence an entry of a fold is scored whole by that fold's model, and any
    other entry by the full model. By block a row is scored by the model of the
    fold whose block holds its frame, and a row in no block by the full model.
    Without folds every row is the full model's.

    Args:
        df: One entry's table.
        fold_of: The fold of each train entry, empty by block or without folds.
        blocks: The block lookup by block, ``None`` otherwise.
        feature: The classifier, which a refusal names.

    Raises:
        ValueError: *df* lacks ``group`` or ``sequence``, or holds more than one entry.
    """
    if blocks is not None:
        return blocks.scoring_folds(df, feature)
    if not fold_of:
        return np.full(len(df), -1, dtype=np.intp)
    fold = out_of_fold_choice(df, fold_of, feature)
    return np.full(len(df), -1 if fold is None else fold, dtype=np.intp)


def check_folds(
    feature: str,
    folds: int,
    table: pd.DataFrame,
    classes: list[int],
    *,
    rows: str,
    unit: Literal["sequence", "block"] = "sequence",
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
        unit: What a row of *table* is, ``"sequence"`` or ``"block"``, for the messages.
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
                + (
                    "cv_folds, or label each class in more of the blocks (a longer "
                    "labeled stretch or more sequences)"
                    if unit == "block"
                    else "cv_folds, or draw the templates from more sequences of "
                    "each class"
                )
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
                "[%s] class %d has %s in %d %s%s, fewer than "
                "cv_folds=%d, so the models of some fold never see it and "
                "give it probability 0 there (rows_%d of cv_folds.parquet "
                "shows where)",
                feature,
                cls,
                rows,
                n_sequences,
                unit,
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
