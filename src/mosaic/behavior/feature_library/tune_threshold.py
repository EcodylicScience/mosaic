"""``tune-threshold``: choose the decision thresholds of a classifier from its out-of-fold probabilities.

The input is one classifier run, or an ensemble of them, fitted with ``cv_folds``:
it writes ``cv_folds.parquet`` beside its output and scores every training entry
with a model that did not see it. This step reads those out-of-fold
probabilities and the ground-truth labels, and picks the probability threshold of
each class that maximizes one score, with the rule a classifier's
``decision_threshold`` applies
(:func:`~mosaic.behavior.feature_library.classifier_common.predicted_labels`). It
saves them as ``thresholds.parquet``, which a ``probability-ensemble`` step reads
through ``decision_threshold: {"step": <this step>}``. A one-member ensemble is
the step that applies a threshold to a classifier's output.

**The rule that is tuned.** ``default_class`` is the class of a row that no other
class claims, so it takes no threshold of its own: it is saved with threshold
1.0, which keeps it out of the competition for a row, and every other listed class
gets a tuned threshold. A class left out of ``classes`` keeps threshold 0. For a
binary problem this is the ``{"0": 1.0, "1": t}`` that predicts the positive class
from probability ``t`` up.

**The search.** One threshold shared by all tuned classes is tried on every point of
``grid``. Several classes are then improved one at a time, each over the whole grid
with the others held, in ascending class order, for up to
:data:`MAX_PASSES` passes or until a pass changes nothing. With one tuned class
the first sweep is the whole search. A tie goes to the lowest threshold, and a pass
that does not improve the score leaves the thresholds as they are, so the result
does not depend on the order entries arrive in.

**Two estimates.** The thresholds are chosen on the same out-of-fold rows that the
plain estimate scores, so the plain estimate is optimistic by the freedom of the
thresholds. With ``nested`` the report also gives the estimate in which the
thresholds that score the rows of fold ``k`` were chosen on the rows of the other
folds, pooled over the folds. The gap between the two is what the tuning flatters.
Both still use probabilities from models whose hyperparameters were chosen on these
rows, which neither estimate corrects.

**Scope.** The thresholds come from the entries of the input's ``cv_folds.parquet``,
so the run id does not cover the step's scope: the step tunes once, and a wider
scope applies the stored thresholds to the new entries. An entry that is not in the
fold table is never tuned on, and is not an error. The first run has to be asked for
over a scope that holds the fold entries. A classifier cannot take its own tuned
thresholds: the step needs the classifier's run, so naming it in that classifier's
``decision_threshold`` is a cycle. The step that applies them is downstream.

**Rows with no probability.** A labeled row whose probabilities are NaN, such as one of a
frame the classifier has no series for, is scored as a prediction of ``default_class``,
the class the rule assigns it and evaluate-labels counts. It is not left out, because
the estimates would then be better than the output they describe.

**Blocks.** An input fitted with ``cv_unit="block"`` writes a fold table with one row per
block, and a row's fold is the fold of the block its frame falls in. A labeled row in
no block is left out of the tuning, and the nested estimate leaves out a whole fold's
rows as it does for entries.

**Grid edge.** A tuned threshold equal to the lowest or highest point of ``grid`` is
reported in ``warnings`` of the report and logged: the optimum may lie beyond what was
tried, and the default grid starts at 0.01, so a class cut there is a rare class whose
best cut the grid cannot see.

**Output.** Each entry's table is the input's ``prob_<class>`` columns with
``predicted_label`` chosen by the saved thresholds. On the fold entries that is the
plain estimate's prediction, which the thresholds were chosen on.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

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

from .classifier_common import (
    FIT_SCOPE_REMEDY,
    THRESHOLDS_FILE,
    FoldLookup,
    assemble_prediction,
    pooled_class_scores,
    predicted_labels,
    read_fold_table,
    read_thresholds,
    refuse_entries_outside_scope,
)
from .helpers import ensure_columns, entry_identity
from .registry import register_feature

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GRID",
    "MAX_PASSES",
    "REPORT_FILE",
    "TuneThresholdFeature",
]

REPORT_FILE: Final = "tune_report.json"
"""What the step calls its report, beside ``thresholds.parquet``."""

MAX_PASSES: Final = 5
"""The most passes of one-class-at-a-time improvement over several classes."""

DEFAULT_GRID: Final[tuple[float, ...]] = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
)
"""The thresholds tried when ``grid`` is not set: fine below 0.1, where a rare class lives."""

_PROBABILITY_COLUMN: Final = re.compile(r"^prob_(-?[0-9]+)$")

_CAVEAT: Final = (
    "The probabilities are out of fold, but the models that wrote them had their "
    "hyperparameters chosen on these same rows, which neither estimate corrects."
)

_LABELS_DESCRIPTION = (
    "The ground-truth labels the thresholds are scored against, resolved from "
    "labels/<kind>/index.csv. A {step: <id>} reference names a convert-labels "
    "step and pins the variant it writes."
)

_CLASSES_DESCRIPTION = (
    "The classes that get a tuned threshold, other than default_class, which "
    "takes the rows that no class claims. Unset, every class of the input but "
    "default_class. A class left out keeps threshold 0, which it always clears."
)

_METRIC_DESCRIPTION = (
    "What the thresholds maximize, over every labeled out-of-fold row pooled "
    "together. macro_f1 is the mean F1 of the classes in average_over. pooled_f1 "
    "is the F1 of the one class in classes, such as the positive class of a binary "
    "classifier."
)

_AVERAGE_OVER_DESCRIPTION = (
    "The classes that macro_f1 averages over, such as every behavior but other. "
    "Unset, every class that a labeled row has. Only with macro_f1."
)

_GRID_DESCRIPTION = (
    "The thresholds tried for each class, ascending and between 0 and 1. A tie "
    "goes to the lowest. The default is fine below 0.1 and steps by 0.05 above, "
    "because a rare class is best cut low."
)

_NESTED_DESCRIPTION = (
    "Also estimate how the tuning generalizes, by choosing the thresholds of each "
    "fold on the rows of the other folds and scoring that fold with them. The "
    "report then gives this nested estimate beside the plain one, which is "
    "optimistic because the thresholds were chosen on the rows it scores."
)

_DEFAULT_CLASS_DESCRIPTION = (
    "The class of a row that no tuned class claims, as in the decision_threshold "
    "of a classifier. It takes no threshold of its own: it is saved with 1.0, "
    "which keeps it out of the competition for a row. The step that applies the "
    "thresholds takes the same value as its default_class."
)

_PERSPECTIVE_DESCRIPTION = (
    "Which rows of a pair output to tune on: the rows of this perspective, as "
    "evaluate-labels scores one. Unset, every row."
)


def _edge_warnings(
    chosen: Mapping[int, float], tuned: Sequence[int], grid: Sequence[float]
) -> list[str]:
    """Return a sentence for each tuned class whose threshold is on the grid's edge.

    A threshold at the lowest or highest point tried says the optimum may lie
    beyond it, which the grid cannot show. A one-point grid has no edge.
    """
    if len(grid) < 2:
        return []
    found: list[str] = []
    for cls in tuned:
        limit = chosen[cls]
        if limit == grid[0]:
            side, bound = "lowest", "below"
        elif limit == grid[-1]:
            side, bound = "highest", "above"
        else:
            continue
        found.append(
            f"The threshold of class {cls} is {limit}, the {side} value of grid. "
            f"The best threshold may lie {bound} it. Widen grid before reading "
            "this as a tuned value"
        )
    return found


def _classes_of(columns: Sequence[object]) -> list[int]:
    """Return the classes of the ``prob_<class>`` columns, ascending.

    Raises:
        ValueError: *columns* hold no ``prob_<class>`` column.
    """
    found = sorted(
        {
            int(match[1])
            for name in columns
            if (match := _PROBABILITY_COLUMN.match(str(name))) is not None
        }
    )
    if not found:
        msg = (
            "[tune-threshold] the input has no prob_<class> columns. Its input is "
            "a classifier run or an ensemble of them"
        )
        raise ValueError(msg)
    return found


class _Rows:
    """The labeled out-of-fold rows of every fold entry, pooled.

    Attributes:
        probabilities: One row per labeled row, one column per class.
        truth: The true class of each row, as its position in the labels.
        fold: The fold of the entry each row came from.
    """

    def __init__(
        self,
        probabilities: npt.NDArray[np.float64],
        truth: npt.NDArray[np.intp],
        fold: npt.NDArray[np.int64],
    ) -> None:
        self.probabilities = probabilities
        self.truth = truth
        self.fold = fold

    def where(self, mask: npt.NDArray[np.bool_]) -> _Rows:
        """Return the rows of *mask*."""
        return _Rows(self.probabilities[mask], self.truth[mask], self.fold[mask])


class _Tuner:
    """Scores a rule and searches the thresholds that maximize it.

    The rule is :func:`predicted_labels`, the one a classifier's
    ``decision_threshold`` applies, so the score of a threshold here is the score a
    consumer of it gets.
    """

    def __init__(
        self,
        *,
        classes: list[int],
        tuned: list[int],
        labels: list[int],
        default_class: int,
        metric: str,
        average_over: list[int] | None,
        grid: list[float],
    ) -> None:
        self.classes = classes
        self.tuned = tuned
        self.labels = labels
        self.default_class = default_class
        self.metric = metric
        self.average_over = average_over
        self.grid = grid
        self._label_array = np.asarray(labels, dtype=np.intp)

    def rule(self, tuned: Mapping[int, float]) -> dict[int, float]:
        """Return every class's threshold given the tuned ones.

        ``default_class`` gets 1.0, so it takes only the rows no class claims, and
        a class that is not tuned gets 0.
        """
        return {
            cls: (
                tuned[cls]
                if cls in tuned
                else 1.0
                if cls == self.default_class
                else 0.0
            )
            for cls in self.classes
        }

    def predict(self, rows: _Rows, tuned: Mapping[int, float]) -> npt.NDArray[np.intp]:
        """Return the position in the labels of the class each row is assigned to."""
        assigned = predicted_labels(
            rows.probabilities, self.classes, self.rule(tuned), self.default_class
        )
        return np.searchsorted(self._label_array, assigned).astype(np.intp)

    def value_of(
        self, truth: npt.NDArray[np.intp], predicted: npt.NDArray[np.intp]
    ) -> float:
        """Return the metric of *predicted* against *truth*."""
        scores = pooled_class_scores(truth, predicted, self.labels, self.average_over)
        if self.metric == "macro_f1":
            return scores["macro_f1"]
        label = self.tuned[0]
        return next(row["f1"] for row in scores["per_class"] if row["label"] == label)

    def score(self, rows: _Rows, tuned: Mapping[int, float]) -> float:
        """Return the metric of the rule with *tuned* thresholds over *rows*."""
        return self.value_of(rows.truth, self.predict(rows, tuned))

    def search(self, rows: _Rows) -> tuple[dict[int, float], float, int]:
        """Return the thresholds that maximize the metric over *rows*.

        Returns:
            The tuned thresholds, their metric over *rows* and the number of
            one-class-at-a-time passes that ran.
        """
        best_shared, best = self.grid[0], -math.inf
        for limit in self.grid:
            value = self.score(rows, dict.fromkeys(self.tuned, limit))
            if value > best:
                best_shared, best = limit, value
        current = dict.fromkeys(self.tuned, best_shared)
        passes = 0
        if len(self.tuned) > 1:
            for _ in range(MAX_PASSES):
                passes += 1
                moved = False
                for cls in self.tuned:
                    chosen, chosen_value = current[cls], best
                    for limit in self.grid:
                        value = self.score(rows, {**current, cls: limit})
                        if value > chosen_value:
                            chosen, chosen_value = limit, value
                    if chosen != current[cls]:
                        current[cls], best, moved = chosen, chosen_value, True
                if not moved:
                    break
        return current, best, passes

    def curves(
        self, rows: _Rows, tuned: Mapping[int, float]
    ) -> dict[str, list[dict[str, float]]]:
        """Return each tuned class's metric over the grid, the others held at *tuned*."""
        return {
            str(cls): [
                {"threshold": limit, "value": self.score(rows, {**tuned, cls: limit})}
                for limit in self.grid
            ]
            for cls in self.tuned
        }


@final
@register_feature
class TuneThresholdFeature:
    """Choose the probability thresholds of a cross-validated classifier.

    Reads one classifier run, or an ensemble of them, fitted with ``cv_folds``, and
    chooses each class's threshold on its out-of-fold probabilities against labels.
    Saves them as ``thresholds.parquet`` and a report of the plain and the nested
    estimate. Apply writes the input's probabilities with ``predicted_label`` chosen
    by those thresholds.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.tune_threshold.TuneThresholdFeature.Params`.
    """

    category = "global"
    costly_fit = False  # a fit scores a grid of thresholds and reads no model
    name = "tune-threshold"
    version = "0.1"
    parallelizable = True
    # fit() streams the input but scores only the entries of its ``cv_folds.parquet``
    # and skips the others before touching state.
    scope_dependent = False
    fit_set_pinned_by: ClassVar[str] = "inputs"
    fits = True  # learns its thresholds from the input's out-of-fold probabilities
    temporal_reach = 0  # assigns each row from its own probabilities
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

        @model_validator(mode="after")
        def _one_classifier(self) -> Self:
            """Tune the thresholds of one run, whose fold table names the entries."""
            if len(self.root) != 1:
                msg = (
                    "tune-threshold reads one classifier run, or one ensemble of "
                    f"them, and got {len(self.root)} inputs"
                )
                raise ValueError(msg)
            return self

    class Params(Params):
        labels: Annotated[GroundTruthLabelsSource, Declared(_LABELS_DESCRIPTION)] = (
            Field(default_factory=GroundTruthLabelsSource)
        )
        classes: Annotated[list[int] | None, Declared(_CLASSES_DESCRIPTION)] = None
        metric: Annotated[
            Literal["macro_f1", "pooled_f1"], Declared(_METRIC_DESCRIPTION)
        ] = "macro_f1"
        average_over: Annotated[
            list[int] | None, Declared(_AVERAGE_OVER_DESCRIPTION)
        ] = None
        grid: Annotated[list[float], Declared(_GRID_DESCRIPTION)] = Field(
            default_factory=lambda: list(DEFAULT_GRID)
        )
        nested: Annotated[bool, Declared(_NESTED_DESCRIPTION)] = True
        perspective: Annotated[int | None, Declared(_PERSPECTIVE_DESCRIPTION)] = None
        default_class: Annotated[int, Declared(_DEFAULT_CLASS_DESCRIPTION)]

        @model_validator(mode="after")
        def _consistent(self) -> Self:
            """Refuse a grid or a choice of classes that cannot be searched."""
            if not self.grid:
                msg = "grid lists no threshold"
                raise ValueError(msg)
            if not all(math.isfinite(t) and 0.0 <= t <= 1.0 for t in self.grid):
                msg = "grid thresholds must lie between 0 and 1"
                raise ValueError(msg)
            if any(b <= a for a, b in zip(self.grid, self.grid[1:], strict=False)):
                msg = (
                    "grid must be strictly ascending, so that a tie goes to the lowest"
                )
                raise ValueError(msg)
            if self.classes is not None:
                if not self.classes:
                    msg = "classes lists no class"
                    raise ValueError(msg)
                if len(set(self.classes)) != len(self.classes):
                    msg = "classes lists a class twice"
                    raise ValueError(msg)
                if self.default_class in self.classes:
                    msg = (
                        f"classes lists default_class {self.default_class}, which "
                        "takes the rows that no listed class claims and so has no "
                        "threshold to tune"
                    )
                    raise ValueError(msg)
            if self.metric == "pooled_f1":
                if self.average_over is not None:
                    msg = "average_over only applies with metric macro_f1"
                    raise ValueError(msg)
                if self.classes is not None and len(self.classes) != 1:
                    msg = (
                        "metric pooled_f1 scores one class, so classes lists "
                        f"exactly one, and {len(self.classes)} are listed. Use "
                        "macro_f1 to tune several"
                    )
                    raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: TuneThresholdFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._thresholds: dict[int, float] | None = None
        self._report: dict[str, object] | None = None
        self._fold_table: pd.DataFrame | None = None
        self._label_files: DependencyLookup = {}

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._thresholds = None
        self._report = None
        self._label_files = {}
        cached = run_root / THRESHOLDS_FILE
        if cached.exists():
            self._thresholds = read_thresholds(cached)
            return True
        self._fold_table = self._input_fold_table(dependency_lookups)
        self._label_files = dependency_lookups.get("labels", {})
        return False

    def fit(self, inputs: InputStream) -> None:
        self._thresholds, self._report = self._tune(inputs)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._thresholds is None:
            msg = f"[{self.name}] no thresholds -- call fit() or load_state() first"
            raise RuntimeError(msg)
        ensure_columns(df, ["frame"])
        classes = _classes_of(list(df.columns))
        if classes != sorted(self._thresholds):
            msg = (
                f"[{self.name}] the thresholds were tuned for classes "
                f"{sorted(self._thresholds)} and this input writes {classes}"
            )
            raise ValueError(msg)
        probs = df[[f"prob_{cls}" for cls in classes]].to_numpy(dtype=np.float64)
        return assemble_prediction(
            df, probs, classes, self._thresholds, self.params.default_class
        )

    def save_state(self, run_root: Path) -> None:
        if self._thresholds is None or self._report is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        report = json.dumps(self._report, indent=2)
        atomic_write(
            run_root / REPORT_FILE, lambda temporary: temporary.write_text(report)
        )
        table = pd.DataFrame(
            {
                "class": np.asarray(sorted(self._thresholds), dtype=np.int64),
                "threshold": [self._thresholds[c] for c in sorted(self._thresholds)],
            }
        )
        # The thresholds go last: their presence is what makes the run a cache.
        _ = write_parquet_atomic(table, run_root / THRESHOLDS_FILE)

    # --- The input's folds ---

    def _input_fold_table(
        self, dependency_lookups: dict[str, DependencyLookup]
    ) -> pd.DataFrame | None:
        """Return the input's ``cv_folds.parquet``, ``None`` when it writes none."""
        lookup = dependency_lookups.get("_input_0", {})
        if not lookup:
            return None
        path = next(iter(lookup.values())).parent / "cv_folds.parquet"
        if not path.exists():
            return None
        return read_fold_table(path)

    # --- Tuning ---

    def _collect(
        self, inputs: InputStream, folds: FoldLookup
    ) -> tuple[list[int], _Rows, int]:
        """Read the labeled out-of-fold rows of the fold entries.

        A fold table by block gives each row the fold of the block its frame falls
        in, and a row in no block is left out.

        Returns:
            The classes of the input, the pooled rows with their true class as a
            label (not yet a position) and the number of entries read.
        """
        fold_entries = folds.entries
        seen: set[tuple[str, str]] = set()
        classes: list[int] | None = None
        probability_blocks: list[npt.NDArray[np.float64]] = []
        truth_blocks: list[npt.NDArray[np.int64]] = []
        fold_blocks: list[npt.NDArray[np.int64]] = []
        for _, df in inputs():
            entry = entry_identity(df)
            if entry not in fold_entries:
                continue
            seen.add(entry)
            label_file = self._label_files.get(entry)
            if label_file is None:
                continue
            ensure_columns(df, ["frame"])
            if classes is None:
                classes = _classes_of(list(df.columns))
            perspective = self.params.perspective
            if perspective is not None:
                ensure_columns(df, ["perspective"])
                df = df.loc[df["perspective"] == perspective]
            probs = df[[f"prob_{cls}" for cls in classes]].to_numpy(dtype=np.float64)
            labels = labels_for_rows(label_file, df)
            row_folds = folds.folds(entry, df["frame"].to_numpy())
            # A labeled row with a NaN probability stays: the rule assigns it
            # default_class, and evaluate-labels counts it as that prediction.
            keep = (labels != UNLABELED) & (row_folds >= 0)
            probability_blocks.append(probs[keep])
            truth_blocks.append(labels[keep])
            fold_blocks.append(row_folds[keep].astype(np.int64))

        refuse_entries_outside_scope(
            self.name,
            "entries of the input's folds",
            fold_entries,
            self._label_files,
            seen,
            consequence=(
                "The thresholds would be chosen on fewer entries than the input "
                "was fitted on."
            ),
            remedy=FIT_SCOPE_REMEDY,
        )
        if classes is None or not probability_blocks:
            msg = (
                f"[{self.name}] no labeled frame of the input's fold entries "
                "reached the fit"
            )
            raise ValueError(msg)
        truth = np.concatenate(truth_blocks)
        if truth.size == 0:
            msg = f"[{self.name}] the labels cover no frame of the input's fold entries"
            raise ValueError(msg)
        rows = _Rows(
            np.concatenate(probability_blocks),
            truth.astype(np.intp),
            np.concatenate(fold_blocks),
        )
        return classes, rows, len(seen)

    def _tune(self, inputs: InputStream) -> tuple[dict[int, float], dict[str, object]]:
        """Choose the thresholds, and report the plain and the nested estimate."""
        params = self.params
        if self._fold_table is None:
            msg = (
                f"[{self.name}] the thresholds are chosen on out-of-fold "
                "probabilities, and the input writes no cv_folds.parquet. Fit the "
                "classifier, or every member of the ensemble, with cv_folds"
            )
            raise ValueError(msg)
        classes, rows, n_entries = self._collect(
            inputs, FoldLookup.from_table(self._fold_table)
        )

        tuned = (
            list(params.classes)
            if params.classes is not None
            else [c for c in classes if c != params.default_class]
        )
        unknown = [c for c in tuned if c not in classes]
        if unknown:
            msg = (
                f"[{self.name}] classes lists {unknown}, which the input does not "
                f"write. It writes {classes}"
            )
            raise ValueError(msg)
        if not tuned:
            msg = (
                f"[{self.name}] the input writes only default_class "
                f"{params.default_class}, so there is no class to tune"
            )
            raise ValueError(msg)
        if params.metric == "pooled_f1" and len(tuned) != 1:
            msg = (
                f"[{self.name}] metric pooled_f1 scores one class and the input "
                f"has {len(tuned)} to tune ({sorted(tuned)}). List the one in "
                "classes, or use macro_f1"
            )
            raise ValueError(msg)
        tuned.sort()

        present = {int(v) for v in np.unique(rows.truth)}
        labels = sorted(set(classes) | present | {params.default_class})
        position = {label: index for index, label in enumerate(labels)}
        pooled = _Rows(
            rows.probabilities,
            np.array([position[int(v)] for v in rows.truth], dtype=np.intp),
            rows.fold,
        )
        tuner = _Tuner(
            classes=classes,
            tuned=tuned,
            labels=labels,
            default_class=params.default_class,
            metric=params.metric,
            average_over=params.average_over,
            grid=list(params.grid),
        )

        chosen, plain, passes = tuner.search(pooled)
        edge = _edge_warnings(chosen, tuned, params.grid)
        report: dict[str, object] = {
            "metric": params.metric,
            "classes_tuned": tuned,
            "average_over": (
                None if params.average_over is None else list(params.average_over)
            ),
            "default_class": params.default_class,
            "perspective": params.perspective,
            "grid": list(params.grid),
            "n_entries": n_entries,
            "n_rows": int(pooled.truth.size),
            "thresholds": {str(c): chosen[c] for c in tuned},
            "passes": passes,
            "plain": {"value": plain, "curves": tuner.curves(pooled, chosen)},
            "nested": None,
            "warnings": edge,
            "caveat": _CAVEAT,
        }
        for sentence in edge:
            log.warning("[%s] %s", self.name, sentence)
        if params.nested:
            report["nested"] = self._nested(tuner, pooled)
        return tuner.rule(chosen), report

    def _nested(self, tuner: _Tuner, pooled: _Rows) -> dict[str, object]:
        """Score each fold with thresholds chosen on the other folds, pooled over folds."""
        predicted = np.zeros(pooled.truth.size, dtype=np.intp)
        covered = np.zeros(pooled.truth.size, dtype=np.bool_)
        folds: list[dict[str, object]] = []
        for fold in sorted(int(f) for f in np.unique(pooled.fold)):
            inside = pooled.fold == fold
            others = pooled.where(~inside)
            if others.truth.size == 0:
                continue
            thresholds, value, _ = tuner.search(others)
            predicted[inside] = tuner.predict(pooled.where(inside), thresholds)
            covered |= inside
            folds.append(
                {
                    "fold": fold,
                    "thresholds": {str(c): thresholds[c] for c in tuner.tuned},
                    "value_on_other_folds": value,
                    "n_rows": int(inside.sum()),
                }
            )
        if not folds:
            msg = (
                f"[{self.name}] nested needs rows in at least two folds, and "
                "the labeled rows are all in one"
            )
            raise ValueError(msg)
        value = tuner.value_of(pooled.truth[covered], predicted[covered])
        return {"value": value, "folds": folds, "n_rows": int(covered.sum())}
