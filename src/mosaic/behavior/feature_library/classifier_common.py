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
from collections.abc import (
    Callable,
    Collection,
    Container,
    Iterable,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.evaluation import CountScores, pool_entry_counts
from mosaic.core.params import Declared
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.types import COLUMNS, ParquetArtifact, ParquetLoadSpec

from .helpers import (
    cv_block_assignment,
    cv_block_bounds,
    cv_fold_assignment,
    meta_columns,
    scored_entry,
)

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
    "THRESHOLDS_FILE",
    "TUNED_DECISION_THRESHOLD_DESCRIPTION",
    "CV_BLOCKS_PER_SEQUENCE_DESCRIPTION",
    "CV_GAP_FRAMES_DESCRIPTION",
    "CV_UNIT_DESCRIPTION",
    "CvBlock",
    "CvBlocksPerSequence",
    "CvGapFrames",
    "CvSeed",
    "CvUnit",
    "DecisionThreshold",
    "FitReporting",
    "FoldLookup",
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
    "deal_blocks",
    "deal_folds",
    "fold_map",
    "fold_rows",
    "fold_table",
    "held_out_frames",
    "pooled_class_scores",
    "predicted_labels",
    "read_fold_table",
    "read_thresholds",
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
    "at most one frame. Unset, the number of cv_folds, so that every fold "
    "holds out one block of each sequence. Only with cv_unit block."
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


def check_cross_validation(
    cv_folds: int | None,
    cv_seed: int,
    *,
    model_given: bool,
    cv_unit: Literal["sequence", "block"] = "sequence",
    cv_blocks_per_sequence: int | None = None,
    cv_gap_frames: int = 0,
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

    Returns:
        The blocks per sequence the fit cuts: the one given, else ``cv_folds``
        under the block unit, ``None`` under the sequence unit. A caller stores it
        back, so that naming the default and leaving it out are one run id.

    Raises:
        ValueError: ``cv_seed`` or a block setting is set without what it needs,
            or ``cv_folds`` is set beside a ``model``.
    """
    if cv_folds is None:
        if cv_seed != 0:
            msg = "cv_seed deals the folds of cv_folds, which is not set"
            raise ValueError(msg)
        unset = (
            ["cv_unit"]
            if cv_unit != "sequence"
            else [
                name
                for name, value in (
                    ("cv_blocks_per_sequence", cv_blocks_per_sequence),
                    ("cv_gap_frames", cv_gap_frames),
                )
                if value not in (None, 0)
            ]
        )
        if unset:
            msg = f"{', '.join(unset)} cut the folds of cv_folds, which is not set"
            raise ValueError(msg)
        return None
    if model_given:
        msg = (
            "cv_folds fits fold models from templates, so it needs "
            "templates and is refused beside model"
        )
        raise ValueError(msg)
    if cv_unit == "sequence":
        unused = [
            name
            for name, value in (
                ("cv_blocks_per_sequence", cv_blocks_per_sequence),
                ("cv_gap_frames", cv_gap_frames),
            )
            if value not in (None, 0)
        ]
        if unused:
            msg = f"{', '.join(unused)} only apply with cv_unit block"
            raise ValueError(msg)
        return None
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
    def by_blocks(self) -> bool:
        """Whether a row's fold depends on its frame."""
        return bool(self._blocks)

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


def check_folds(
    feature: str,
    folds: int,
    table: pd.DataFrame,
    classes: list[int],
    *,
    rows: str,
    unit: str = "sequence",
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
                    "cv_folds, or label each class in more of the sequence"
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
