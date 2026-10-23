"""``probability-ensemble``: average or blend the class probabilities of several runs.

The members are classifier runs read through ``inputs`` (``xgboost``,
``lightning-action``, ``sequence-classifier``, or another ensemble). The step
writes the weighted mean of their ``prob_<class>`` columns and the class of its
highest value, with the columns and identity of any classifier.

**Weights.** ``uniform`` averages, as a seed average of one model does. A list
gives one weight per input, in the order of ``inputs``. ``fit`` chooses the
weights that maximize the pooled macro F1 of the blend over the training
entries, from the members' out-of-fold probabilities, on a grid over the simplex
(the first maximum in the grid's order wins, so the result does not depend on
the order entries arrive in). The score that the report gives for those weights
is optimistic by the freedom of the weights, which is one fewer than the number
of members.

**Out-of-fold members.** A member fitted with ``cv_folds`` writes
``cv_folds.parquet`` beside its output, and scores each training entry with a
model that did not see it. This step writes the same fold table when every
member has one and the tables agree on ``group``, ``sequence`` and ``fold``,
because an average of out-of-fold probabilities is still out-of-fold, and a
further ensemble finds the table where it finds a classifier's. It writes none
when no member has one. The count columns of a member's table are not compared:
``xgboost`` counts sampled template rows and ``sequence-classifier`` counts
labeled frames.

``uniform`` and listed weights fit nothing and need no fold table, so members of
which some have one and some do not, or whose folds differ, are averaged all the
same, and the step writes no fold table for them. ``weights="fit"`` refuses
such members, and a member without a fold table, so that an in-sample probability
never chooses a weight.

**Scope.** The fitted weights come from the members' fold entries, so the run
id does not cover the step's scope: the step fits once, and a wider scope applies
the stored weights to the new entries. The first run of ``weights="fit"`` has to
be asked for over a scope that holds the fold entries.

**Column names.** An input merge keeps the first input's ``prob_<class>`` and
numbers the same column of input *i* ``prob_<class>__<i>``
(:func:`~mosaic.core.pipeline.loading.merge_naming`). The members must write the
same classes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.evaluation import pool_entry_counts
from mosaic.core.helpers import UNLABELED
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    InputStream,
    Result,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.strict_model import StrictModel

from .classifier_common import (
    FIT_SCOPE_REMEDY,
    DEFAULT_CLASS_DESCRIPTION,
    DecisionThreshold,
    assemble_prediction,
    refuse_entries_outside_scope,
)
from .helpers import ensure_columns, entry_identity
from .registry import register_feature

__all__ = [
    "MAX_GRID_POINTS",
    "ProbabilityEnsembleFeature",
    "member_probabilities",
    "member_probability_column",
    "merged_classes",
    "simplex_grid",
]

MAX_GRID_POINTS: Final = 20_000
"""The most weight vectors that ``weights="fit"`` searches."""

_PROBABILITY_COLUMN: Final = re.compile(r"^prob_(-?[0-9]+)$")
_FOLD_COLUMNS: Final = ["group", "sequence", "fold"]

_WEIGHTS_DESCRIPTION = (
    "How the members' probabilities are combined. uniform averages them. A "
    "list gives one non-negative weight per input, in the order of inputs, "
    "and the weights are scaled to sum to 1. fit chooses the weights that "
    "maximize the pooled macro F1 of the blend over the training entries of "
    "the members' cross-validation folds, which needs labels and members fitted "
    "with cv_folds."
)

_LABELS_DESCRIPTION = (
    "The ground-truth labels that fit scores the blend against. Only with weights fit."
)

_SCORE_CLASSES_DESCRIPTION = (
    "The classes that the macro F1 of fit averages over, such as every "
    "behavior but other. Unset, every class that a labeled frame has. Only with "
    "weights fit."
)

_GRID_STEP_DESCRIPTION = (
    "The spacing of the weights that fit tries. It must divide 1, and the "
    "search refuses a grid of more than 20,000 weight vectors, which a "
    "coarser step or fewer members avoids. Only with weights fit."
)


class _EnsembleState(StrictModel):
    """What a run keeps of its fit: the weights its inputs are combined by."""

    weights: list[float]


def member_probability_column(label: int, position: int) -> str:
    """Return the column that input *position* contributes for class *label*.

    The first input keeps ``prob_<class>``, and a later one is numbered by the
    merge of the step's inputs.
    """
    return f"prob_{label}" if position == 0 else f"prob_{label}__{position}"


def merged_classes(columns: Sequence[str]) -> list[int]:
    """Return the classes of the first input, from the ``prob_<class>`` columns.

    Raises:
        ValueError: *columns* hold no ``prob_<class>`` column.
    """
    found = sorted(
        {
            int(match[1])
            for name in columns
            if (match := _PROBABILITY_COLUMN.match(name)) is not None
        }
    )
    if not found:
        msg = (
            "[probability-ensemble] the inputs have no prob_<class> columns. "
            "Its inputs are classifier runs"
        )
        raise ValueError(msg)
    return found


def member_probabilities(
    df: pd.DataFrame, classes: Sequence[int], members: int
) -> npt.NDArray[np.float64]:
    """Return the probabilities of every member, one row per row of *df*.

    Args:
        df: The merged inputs of the step.
        classes: The classes, in column order.
        members: The number of inputs.

    Returns:
        An array of shape ``(rows, members, classes)``.

    Raises:
        ValueError: A member lacks the probability column of a class.
    """
    names = [
        [member_probability_column(label, position) for label in classes]
        for position in range(members)
    ]
    missing = [
        f"member {position} ({name})"
        for position, row in enumerate(names)
        for name in row
        if name not in df.columns
    ]
    if missing:
        msg = (
            "[probability-ensemble] the members do not write the same classes: "
            f"no column for {', '.join(missing[:5])}"
        )
        raise ValueError(msg)
    return np.stack([df[row].to_numpy(dtype=np.float64) for row in names], axis=1)


def simplex_grid(members: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every way to split *parts* units among *members* non-negative integers.

    The order is lexicographic, ``(0, ..., 0, parts)`` first, so a tie between
    two weight vectors is broken the same way on every machine.
    """
    if members == 1:
        yield (parts,)
        return
    for first in range(parts + 1):
        for rest in simplex_grid(members - 1, parts - first):
            yield (first, *rest)


def _pooled_macro_f1(
    truth: npt.NDArray[np.intp],
    predicted: npt.NDArray[np.intp],
    labels: Sequence[int],
    average_over: Sequence[int] | None,
) -> float:
    """Score predictions by the pooled counts, with the formula of ``evaluate-labels``."""
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
    scores = pool_entry_counts(pd.DataFrame([counts]), average_over=average_over)
    return scores["macro_f1"]


@final
@register_feature
class ProbabilityEnsembleFeature:
    """Average or blend the class probabilities of several classifier runs.

    Reads two or more runs that write ``prob_<class>`` columns and writes their
    weighted mean as a classifier would: ``prob_<class>`` and
    ``predicted_label``. The weights are uniform, listed, or fitted on the
    members' out-of-fold probabilities against labels. When every member writes
    the same ``cv_folds.parquet`` this step writes it too, so the blend is
    out-of-fold wherever its members are.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.probability_ensemble.ProbabilityEnsembleFeature.Params`.
    """

    category = "global"
    costly_fit = False  # a fit scores a grid of weights and reads no model
    name = "probability-ensemble"
    # 0.2: the run id no longer covers the step's scope. The members' fold table
    # names the entries the weights are chosen on, so a wider scope applies the
    # weights it has. Fixed weights no longer check the members' fold tables.
    version = "0.2"
    parallelizable = True
    # ``weights="fit"`` streams its inputs but scores only the entries of the
    # members' ``cv_folds.parquet``, and skips the others before touching state.
    scope_dependent = False
    fit_set_pinned_by: ClassVar[str] = "inputs"
    # Every other mode fits nothing, and ``__init__`` clears this for them.
    fits = True  # learns its weights in fit mode
    accepts_overlap = False  # computes within a row
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    requires: ClassVar[tuple[Requirement, ...]] = (Prefix("prob_"),)
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("prob_"), Column("predicted_label")),
        conditional=META_COLUMNS_KEPT,
    )

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        weights: Annotated[
            list[float] | Literal["uniform", "fit"], Declared(_WEIGHTS_DESCRIPTION)
        ] = "uniform"
        labels: Annotated[
            GroundTruthLabelsSource | None, Declared(_LABELS_DESCRIPTION)
        ] = None
        score_classes: Annotated[
            list[int] | None, Declared(_SCORE_CLASSES_DESCRIPTION)
        ] = None
        grid_step: Annotated[
            float, Field(gt=0.0, le=0.5), Declared(_GRID_STEP_DESCRIPTION)
        ] = 0.05
        decision_threshold: DecisionThreshold = None
        default_class: Annotated[int, Declared(DEFAULT_CLASS_DESCRIPTION)]

        @model_validator(mode="after")
        def _only_a_fit_takes_its_settings(self) -> Self:
            """Name the settings of a fit, and of no other mode."""
            if isinstance(self.weights, list):
                if not self.weights:
                    msg = "weights lists no weight"
                    raise ValueError(msg)
                if not all(math.isfinite(w) and w >= 0.0 for w in self.weights):
                    msg = "weights must be finite and not negative"
                    raise ValueError(msg)
                if sum(self.weights) <= 0.0:
                    msg = "weights must not all be 0"
                    raise ValueError(msg)
            if self.weights == "fit":
                if self.labels is None:
                    msg = (
                        "labels is required with weights fit: the weights are "
                        "chosen against them"
                    )
                    raise ValueError(msg)
                return self
            fit_only = [
                name
                for name in ("labels", "score_classes", "grid_step")
                if name in self.model_fields_set and getattr(self, name) is not None
            ]
            if fit_only:
                msg = f"{', '.join(fit_only)} only apply with weights fit"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: ProbabilityEnsembleFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.fits = self.params.weights == "fit"
        self._members = len(inputs.root)
        if (
            isinstance(self.params.weights, list)
            and len(self.params.weights) != self._members
        ):
            msg = (
                f"[{self.name}] weights lists {len(self.params.weights)} "
                f"weight{'' if len(self.params.weights) == 1 else 's'} and the "
                f"step has {self._members} input{'' if self._members == 1 else 's'}"
            )
            raise ValueError(msg)

        self._weights: npt.NDArray[np.float64] | None = None
        self._report: dict[str, object] | None = None
        self._cv_table: pd.DataFrame | None = None
        self._label_files: DependencyLookup = {}

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._weights = None
        self._report = None
        self._label_files = {}
        self._cv_table = self._members_fold_table(dependency_lookups)

        cached = run_root / "ensemble.json"
        if cached.exists():
            stored = _EnsembleState.model_validate_json(cached.read_text())
            self._weights = np.asarray(stored.weights, dtype=np.float64)
            return True
        self._label_files = dependency_lookups.get("labels", {})
        return False

    def fit(self, inputs: InputStream) -> None:
        weights = self.params.weights
        if weights == "uniform":
            chosen = np.ones(self._members)
            self._report = {"weights": "uniform"}
        elif isinstance(weights, list):
            chosen = np.asarray(weights, dtype=np.float64)
            self._report = {"weights": "listed"}
        else:
            chosen, self._report = self._fit_weights(inputs)
        self._weights = chosen / chosen.sum()
        self._report["weights_used"] = [float(w) for w in self._weights]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._weights is None:
            msg = f"[{self.name}] no weights -- call fit() or load_state() first"
            raise RuntimeError(msg)
        ensure_columns(df, ["frame"])
        classes = merged_classes([str(c) for c in df.columns])
        probs = member_probabilities(df, classes, self._members)
        active = self._weights > 0.0
        # A member with no weight does not decide a row, whatever it holds.
        blended = np.einsum("m,nmc->nc", self._weights[active], probs[:, active, :])
        return assemble_prediction(
            df,
            blended,
            classes,
            self.params.decision_threshold,
            self.params.default_class,
        )

    def save_state(self, run_root: Path) -> None:
        if self._weights is None or self._report is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        if self._cv_table is not None:
            _ = write_parquet_atomic(self._cv_table, run_root / "cv_folds.parquet")
        report = json.dumps(self._report, indent=2)
        atomic_write(
            run_root / "ensemble_report.json",
            lambda temporary: temporary.write_text(report),
        )
        # The state goes last: its presence is what makes the run a cache.
        state = _EnsembleState(weights=[float(w) for w in self._weights])
        atomic_write(
            run_root / "ensemble.json",
            lambda temporary: temporary.write_text(state.model_dump_json()),
        )

    # --- Folds ---

    def _members_fold_table(
        self, dependency_lookups: dict[str, DependencyLookup]
    ) -> pd.DataFrame | None:
        """Return the fold table that every member shares, ``None`` when none has one.

        Only ``weights="fit"`` needs the table, so only it is refused members that
        cannot give one. Fixed weights fit nothing: members of which some have a
        table and some do not, or whose folds differ, give ``None`` and no table is
        written for the blend, because it is not out of fold wherever they differ.

        Raises:
            ValueError: ``weights="fit"`` and an input resolved no output, some
                members have a table and some do not, or the tables disagree on
                the fold of an entry.
        """
        fitting = self.params.weights == "fit"
        tables: list[pd.DataFrame | None] = []
        unresolved: list[str] = []
        for position in range(self._members):
            lookup = dependency_lookups.get(f"_input_{position}", {})
            if not lookup:
                unresolved.append(str(position))
                tables.append(None)
                continue
            path = next(iter(lookup.values())).parent / "cv_folds.parquet"
            tables.append(
                pd.read_parquet(path)[_FOLD_COLUMNS].sort_values(
                    _FOLD_COLUMNS[:2], ignore_index=True
                )
                if path.exists()
                else None
            )
        if unresolved and fitting:
            msg = (
                f"[{self.name}] inputs {', '.join(unresolved)} resolved no output "
                "entry, so their cv_folds.parquet cannot be read. Run the members "
                "first"
            )
            raise ValueError(msg)
        present = [table for table in tables if table is not None]
        if not present:
            return None
        without = [str(i) for i, table in enumerate(tables) if table is None]
        if without:
            if not fitting:
                return None
            msg = (
                f"[{self.name}] inputs {', '.join(without)} write no cv_folds.parquet "
                "and the others do. Average runs that were all fitted with cv_folds, "
                "or none of them"
            )
            raise ValueError(msg)
        first = present[0]
        for position, table in enumerate(present[1:], start=1):
            if not table.equals(first):
                if not fitting:
                    return None
                msg = (
                    f"[{self.name}] input {position} deals its cross-validation "
                    "folds differently from input 0, so no entry is out of fold "
                    "for all members. Fit the members with one templates run and "
                    "one cv_seed"
                )
                raise ValueError(msg)
        return first

    # --- Fitting the weights ---

    def _fit_weights(
        self, inputs: InputStream
    ) -> tuple[npt.NDArray[np.float64], dict[str, object]]:
        """Choose the weights that maximize the pooled macro F1 over the fold entries."""
        parts = round(1.0 / self.params.grid_step)
        if abs(parts * self.params.grid_step - 1.0) > 1e-9:
            msg = f"[{self.name}] grid_step {self.params.grid_step} does not divide 1"
            raise ValueError(msg)
        points = math.comb(parts + self._members - 1, self._members - 1)
        if points > MAX_GRID_POINTS:
            msg = (
                f"[{self.name}] a grid of step {self.params.grid_step} over "
                f"{self._members} members has {points:,} weight vectors, over the "
                f"limit of {MAX_GRID_POINTS:,}. Raise grid_step, or average the "
                "runs of one model first and blend the averages"
            )
            raise ValueError(msg)

        if self._cv_table is None:
            msg = (
                f"[{self.name}] weights fit chooses the weights on out-of-fold "
                "probabilities, and no input writes cv_folds.parquet. Fit the "
                "members with cv_folds"
            )
            raise ValueError(msg)
        fit_entries = set(zip(self._cv_table["group"], self._cv_table["sequence"]))
        seen: set[tuple[str, str]] = set()
        classes: list[int] | None = None
        probability_blocks: list[npt.NDArray[np.float64]] = []
        truth_blocks: list[npt.NDArray[np.int64]] = []
        for _, df in inputs():
            entry = entry_identity(df)
            if entry not in fit_entries:
                continue
            seen.add(entry)
            label_file = self._label_files.get(entry)
            if label_file is None:
                continue
            ensure_columns(df, ["frame"])
            if classes is None:
                classes = merged_classes([str(c) for c in df.columns])
            probs = member_probabilities(df, classes, self._members)
            labels = labels_for_rows(label_file, df)
            keep = (labels != UNLABELED) & ~np.isnan(probs).any(axis=(1, 2))
            probability_blocks.append(probs[keep])
            truth_blocks.append(labels[keep])

        refuse_entries_outside_scope(
            self.name,
            "entries of the members' folds",
            fit_entries,
            self._label_files,
            seen,
            consequence=(
                "The weights would be chosen on fewer entries than the members "
                "were fitted on."
            ),
            remedy=FIT_SCOPE_REMEDY,
        )
        if classes is None or not probability_blocks:
            msg = (
                f"[{self.name}] no labeled frame of the members' fold entries "
                "reached the fit"
            )
            raise ValueError(msg)

        probabilities = np.concatenate(probability_blocks)
        truth = np.concatenate(truth_blocks)
        if truth.size == 0:
            msg = (
                f"[{self.name}] the labels cover no frame of the members' fold entries"
            )
            raise ValueError(msg)
        labels_all = sorted(set(classes) | {int(v) for v in np.unique(truth)})
        position_of = {label: index for index, label in enumerate(labels_all)}
        truth_index = np.array([position_of[int(v)] for v in truth], dtype=np.intp)
        class_index = np.array([position_of[label] for label in classes], dtype=np.intp)
        stacked = np.ascontiguousarray(probabilities.transpose(1, 0, 2))
        average_over = self.params.score_classes

        def score(weights: npt.NDArray[np.float64]) -> float:
            blended = np.tensordot(weights, stacked, axes=([0], [0]))
            predicted = class_index[np.argmax(blended, axis=1)]
            return _pooled_macro_f1(truth_index, predicted, labels_all, average_over)

        best_score = -1.0
        best = np.full(self._members, 1.0 / self._members)
        for units in simplex_grid(self._members, parts):
            weights = np.asarray(units, dtype=np.float64) / parts
            value = score(weights)
            if value > best_score:
                best_score, best = value, weights
        singles = [score(np.eye(self._members)[i]) for i in range(self._members)]
        report: dict[str, object] = {
            "weights": "fit",
            "objective": best_score,
            "member_objectives": singles,
            "uniform_objective": score(np.full(self._members, 1.0 / self._members)),
            "score_classes": None if average_over is None else list(average_over),
            "grid_step": self.params.grid_step,
            "grid_points": points,
            "n_entries": len(seen),
            "n_frames": int(truth.size),
        }
        return best, report
