"""``evaluate-labels``: score a run's predictions against ground-truth labels.

The step a supervised chain ends with. Under a step scope naming the test
groups, it scores a classifier fitted on the training groups, in the same
request that fitted it. Its report is ``metrics.json``, an
:class:`~mosaic.core.evaluation.EvaluationReport` over every entry it scored,
beside ``confusion.parquet``; each output row is one entry's own score, and its
true positives, false positives and false negatives per class
(``tp_<class>``, ``fp_<class>`` and ``fn_<class>``). Those counts add across
entries, so the score of any set of entries, and a bootstrap over them, follow
from the output (:func:`~mosaic.core.evaluation.pool_entry_counts` and
:func:`~mosaic.core.evaluation.bootstrap_sequences`).

**Which rows are scored.** Each row is labeled by its own frame and individuals
(:func:`~mosaic.core.pipeline.loading.labels_for_rows`), so a pair row takes the
label of its own direction, and a row no label covers is unlabeled and dropped,
never class 0. A pair output holds each pair twice per frame, and
``perspective`` says which rows to score (:data:`~mosaic.core.evaluation.Perspective`).

**Two modes.** ``classifier`` scores predicted classes. ``clusters`` scores a
clustering: through ``label_map`` when one is given, a map fitted on other
entries by ``fit-label-map``, and otherwise through the majority mapping of the
scored rows themselves, whose accuracy is the clustering's purity and which no
row is held out of. The report's ``label_map_source`` says which, and the
cluster by class counts are ``cluster_confusion.parquet``.

**Several inputs are one stack.** Give the step several classifier runs over
disjoint entries, such as one run per fold, and it scores their union: each
entry is read from the one input that holds it, and the same prediction column
is scored throughout, so the report equals scoring the concatenated rows. An
entry that two inputs both hold is refused, because it would count twice. Other
steps join several inputs column-wise instead, and this one declares
``stacks_inputs``.

The labels come from the run's dependency lookup, so a ``labels`` reference to a
``convert-labels`` step scores against the variant that step wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, ClassVar, Literal, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.evaluation import (
    ClassScore,
    EvaluationReport,
    LabelMapSource,
    classification_scores,
    cluster_contingency,
    cluster_label_scores,
    entry_counts,
    majority_mapping,
    select_scored_rows,
)
from mosaic.core.helpers import UNLABELED
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.column_contract import (
    Column,
    ParamColumn,
    Prefix,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    Result,
)
from mosaic.core.pipeline.writers import write_parquet_atomic

from .fit_label_map import LabelMapRef, read_label_map
from .helpers import ensure_columns, entry_identity
from .registry import register_feature

METRICS_FILE = "metrics.json"
"""The report a run writes, an :class:`~mosaic.core.evaluation.EvaluationReport`."""

CONFUSION_FILE = "confusion.parquet"
"""The confusion matrix: a row per true class, a ``predicted_<k>`` column per class."""

CLUSTER_CONFUSION_FILE = "cluster_confusion.parquet"
"""In clusters mode, a row per cluster and a ``class_<k>`` column per class."""

_CLUSTER_KEYS = (
    "ARI",
    "AMI",
    "homogeneity",
    "completeness",
    "v_measure",
    "purity",
    "hungarian_accuracy",
    "hungarian_one_to_one",
    "n_clusters",
)
"""The scores of :func:`cluster_label_scores` that the report keeps."""

_WRONG: int = -2
"""What a cluster outside the label map scores as: a class no label holds."""

_LABELS_DESCRIPTION = (
    "The ground-truth labels to score against, resolved from "
    "labels/<kind>/index.csv. A {step: <id>} reference names a convert-labels "
    "step and pins the variant it writes."
)
_MODE_DESCRIPTION = (
    "classifier scores predicted classes. clusters scores a clustering, mapped "
    "to classes through label_map, or through the majority class of each "
    "cluster among the scored rows when label_map is unset."
)
_PREDICTION_COLUMN_DESCRIPTION = (
    "The column holding each row's prediction: a class in classifier mode, "
    "a cluster in clusters mode."
)
_PROBABILITY_PREFIX_DESCRIPTION = (
    "The prefix of the per-class probability columns that perspective "
    "'mean_probabilities' averages, as prob_<class>."
)
_PERSPECTIVE_DESCRIPTION = (
    "Which rows of a pair output to score: the rows of one perspective, "
    "'pooled' for both, or 'mean_probabilities' to average a pair's two rows. "
    "Unset scores the rows as they are, and refuses rows of more than one "
    "perspective."
)
_MISSING_LABEL_DESCRIPTION = (
    "What an entry with no label file does: drop leaves it out of the score, "
    "error fails the run."
)
_CLASSES_DESCRIPTION = (
    "The classes to report, in order. Unset reports every class a row holds or "
    "is predicted as."
)
_AVERAGE_OVER_DESCRIPTION = (
    "The classes macro_f1 averages over, such as every behavior but other. "
    "Unset averages over every class a scored row holds."
)
_LABEL_MAP_DESCRIPTION = (
    "In clusters mode, a cluster to class map fitted on other entries by "
    "fit-label-map, as {step: <id>}."
)


@final
@register_feature
class EvaluateLabels:
    """Score a run's predictions or clusters against ground-truth labels.

    Writes ``metrics.json`` (accuracy, macro F1, per-class precision, recall
    and F1, and the clustering scores in clusters mode) and
    ``confusion.parquet`` over every entry it scored, ``cluster_confusion.parquet``
    in clusters mode, and one row per entry with its score and its per-class
    counts. Several inputs stack: their entries are unioned, and an entry held by
    more than one input is refused.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.evaluate_labels.EvaluateLabels.Params`.
    """

    category = "summary"
    name = "evaluate-labels"
    version = "0.2"
    parallelizable = False
    scope_dependent = True
    fits = False  # it scores a model's output and learns nothing from it
    stacks_inputs = True  # several inputs are runs over disjoint entries
    relates: RelatesLevel = "none"
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"
    report_file: ClassVar[str] = METRICS_FILE
    """The file of a run that the MCP report resource serves."""
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        ParamColumn("prediction_column"),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column("n_scored"),
            Column("n_unlabeled"),
            Column("accuracy"),
            Column("macro_f1"),
            Prefix("tp_"),
            Prefix("fp_"),
            Prefix("fn_"),
        ),
        conditional=kept(C.group_col, C.seq_col),
    )

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        labels: Annotated[GroundTruthLabelsSource, Declared(_LABELS_DESCRIPTION)] = (
            Field(default_factory=GroundTruthLabelsSource)
        )
        mode: Annotated[
            Literal["classifier", "clusters"], Declared(_MODE_DESCRIPTION)
        ] = "classifier"
        prediction_column: Annotated[str, Declared(_PREDICTION_COLUMN_DESCRIPTION)] = (
            "predicted_label"
        )
        probability_prefix: Annotated[
            str, Declared(_PROBABILITY_PREFIX_DESCRIPTION)
        ] = "prob_"
        perspective: Annotated[
            int | Literal["pooled", "mean_probabilities"] | None,
            Declared(_PERSPECTIVE_DESCRIPTION),
        ] = None
        missing_label: Annotated[
            Literal["drop", "error"], Declared(_MISSING_LABEL_DESCRIPTION)
        ] = "drop"
        classes: Annotated[list[int] | None, Declared(_CLASSES_DESCRIPTION)] = None
        average_over: Annotated[
            list[int] | None, Declared(_AVERAGE_OVER_DESCRIPTION)
        ] = None
        label_map: Annotated[LabelMapRef | None, Declared(_LABEL_MAP_DESCRIPTION)] = (
            None
        )

    def __init__(
        self, inputs: EvaluateLabels.Inputs, params: dict[str, object] | None = None
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._labels_lookup: DependencyLookup = {}
        self._label_map: dict[int, int] | None = None
        self._report: EvaluationReport | None = None
        self._contingency: pd.DataFrame | None = None
        self._names: dict[int, str] = {}

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._labels_lookup = dependency_lookups.get("labels", {})
        self._label_map = None
        self._report = None
        self._contingency = None
        self._names = {}
        if self.params.label_map is not None and "label_map" in artifact_paths:
            self._label_map = read_label_map(artifact_paths["label_map"])
        path = run_root / METRICS_FILE
        if path.exists():
            self._report = EvaluationReport.model_validate_json(path.read_text())
            if self._label_map is None and self._report.label_map:
                self._label_map = dict(self._report.label_map)
            return True
        return False

    def fit(self, inputs: InputStream) -> None:
        if self.params.mode == "clusters" and self.params.label_map is not None:
            if self._label_map is None:
                msg = (
                    "[evaluate-labels] label_map names no saved map. Reference "
                    "a fit-label-map step that has run."
                )
                raise RuntimeError(msg)
        true: list[np.ndarray] = []
        pred: list[np.ndarray] = []
        n_entries = n_unlabeled = n_missing = n_conflicting = 0
        for _key, df in inputs():
            scored = self._scored(df)
            if scored is None:
                continue
            n_entries += 1
            frame, unlabeled, missing, conflicting = scored
            n_unlabeled += unlabeled
            n_missing += missing
            n_conflicting += conflicting
            true.append(frame["_label"].to_numpy(dtype=np.int32))
            pred.append(frame["_prediction"].to_numpy(dtype=np.int32))
        y_true = np.concatenate(true) if true else np.empty(0, dtype=np.int32)
        y_pred = np.concatenate(pred) if pred else np.empty(0, dtype=np.int32)
        self._report = self._evaluate(
            y_true,
            y_pred,
            n_entries=n_entries,
            n_unlabeled=n_unlabeled,
            n_missing=n_missing,
            n_conflicting=n_conflicting,
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._report is None:
            msg = "evaluate-labels has not scored its entries yet."
            raise RuntimeError(msg)
        group, sequence = entry_identity(df)
        scored = self._scored(df)
        row: dict[str, object] = {
            C.group_col: group,
            C.seq_col: sequence,
            "n_scored": 0,
            "n_unlabeled": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
        y_true = y_pred = np.empty(0, dtype=np.int64)
        if scored is not None:
            frame, unlabeled, _, _ = scored
            y_true = frame["_label"].to_numpy(dtype=np.int64)
            y_pred = self._mapped(frame["_prediction"].to_numpy(dtype=np.int64))
            row["n_unlabeled"] = unlabeled
        # Every entry names the report's classes, so the rows stack into one table.
        classes = self._report.classes
        scores = classification_scores(
            y_true, y_pred, classes=classes, average_over=self.params.average_over
        )
        row["n_scored"] = scores["n_samples"]
        if scores["n_samples"]:
            row["accuracy"] = scores["accuracy"]
            row["macro_f1"] = scores["macro_f1"]
        row.update(entry_counts(scores, classes))
        return pd.DataFrame([row])

    def save_state(self, run_root: Path) -> None:
        if self._report is None:
            return
        report = self._report
        atomic_write(
            run_root / METRICS_FILE,
            lambda tmp: tmp.write_text(report.model_dump_json(indent=2) + "\n"),
        )
        classes = report.classes
        confusion = pd.DataFrame(
            np.asarray(report.confusion, dtype=np.int64).reshape(
                len(classes), len(classes)
            ),
            columns=[f"predicted_{value}" for value in classes],
        )
        confusion.insert(0, "label", classes)
        _ = write_parquet_atomic(confusion, run_root / CONFUSION_FILE)
        if self._contingency is not None:
            _ = write_parquet_atomic(
                self._contingency, run_root / CLUSTER_CONFUSION_FILE
            )

    # --- scoring ---

    def _scored(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int] | None:
        """One entry's scored rows: ``_label`` and ``_prediction`` beside its keys.

        Returns the rows and how many were unlabeled, held no prediction and
        were dropped as conflicting pairs, or ``None`` for an entry with no
        label file under ``missing_label="drop"``.
        """
        column = self.params.prediction_column
        ensure_columns(df, [C.frame_col, column])
        group, sequence = entry_identity(df)
        path = self._labels_lookup.get((group, sequence))
        if path is None:
            if self.params.missing_label == "error":
                msg = (
                    f"[evaluate-labels] entry ({group!r}, {sequence!r}) has no "
                    "label file, and missing_label is 'error'."
                )
                raise ValueError(msg)
            return None
        if not self._names:
            self._names = _class_names(path)
        rows = df.assign(_label=labels_for_rows(path, df))
        selected = select_scored_rows(
            rows,
            perspective=self.params.perspective,
            prediction_column=column,
            label_column="_label",
            probability_prefix=self.params.probability_prefix,
        )
        frame = selected.frame
        missing = frame[column].isna().to_numpy()
        frame = frame.loc[~missing]
        unlabeled = (frame["_label"] == UNLABELED).to_numpy()
        frame = frame.loc[~unlabeled]
        out = pd.DataFrame(
            {
                "_label": frame["_label"].to_numpy(dtype=np.int64),
                "_prediction": frame[column].to_numpy(dtype=np.int64),
            }
        )
        return out, int(unlabeled.sum()), int(missing.sum()), selected.n_conflicting

    def _mapped(self, predictions: np.ndarray) -> np.ndarray:
        """*predictions* as classes: unchanged for a classifier, mapped for clusters."""
        if self.params.mode == "classifier" or self._label_map is None:
            return predictions
        mapping = self._label_map
        return np.array(
            [mapping.get(int(value), _WRONG) for value in predictions], dtype=np.int64
        )

    def _evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        n_entries: int,
        n_unlabeled: int,
        n_missing: int,
        n_conflicting: int,
    ) -> EvaluationReport:
        """The report over every scored row of every entry."""
        clusters: dict[str, float | int | bool] = {}
        source: LabelMapSource = ""
        if self.params.mode == "clusters":
            if self.params.label_map is not None:
                source = "fit-label-map"
            else:
                source = "scored_rows"
                self._label_map = majority_mapping(y_true, y_pred)
            scored = cluster_label_scores(y_true, y_pred)
            for key in _CLUSTER_KEYS:
                value = scored[key]
                if isinstance(value, bool | int | float):
                    clusters[key] = value
            self._contingency = cluster_contingency(y_true, y_pred)
        mapped = self._mapped(y_pred)
        classes = self.params.classes
        if self.params.mode == "clusters":
            known = sorted({int(value) for value in y_true})
            classes = classes if classes is not None else known
        scores = classification_scores(
            y_true, mapped, classes=classes, average_over=self.params.average_over
        )
        listed = [value for value in scores["classes"] if value != _WRONG]
        keep = [scores["classes"].index(value) for value in listed]
        confusion = np.asarray(scores["confusion"])[np.ix_(keep, keep)]
        return EvaluationReport(
            mode=self.params.mode,
            n_entries=n_entries,
            n_scored=scores["n_samples"],
            n_unlabeled=n_unlabeled,
            n_missing_prediction=n_missing,
            n_conflicting=n_conflicting,
            perspective=self.params.perspective,
            accuracy=scores["accuracy"],
            macro_f1=scores["macro_f1"],
            average_over=scores["average_over"],
            classes=listed,
            per_class=[
                ClassScore(
                    label=row["label"],
                    name=self._names.get(row["label"], ""),
                    precision=row["precision"],
                    recall=row["recall"],
                    f1=row["f1"],
                    support=row["support"],
                    predicted=row["predicted"],
                )
                for row in scores["per_class"]
                if row["label"] != _WRONG
            ],
            confusion=confusion.astype(int).tolist(),
            clusters=clusters,
            label_map=dict(self._label_map or {})
            if self.params.mode == "clusters"
            else {},
            label_map_source=source,
        )


def _class_names(path: Path) -> dict[int, str]:
    """The class names a label file records, by class id, empty when it records none."""
    with np.load(path, allow_pickle=True) as npz:
        if "label_ids" not in npz.files or "label_names" not in npz.files:
            return {}
        ids = np.asarray(npz["label_ids"]).ravel()
        names = np.asarray(npz["label_names"]).ravel()
    return {int(i): str(n) for i, n in zip(ids, names, strict=False)}


def read_report(run_root: Path) -> EvaluationReport | None:
    """The report an ``evaluate-labels`` run saved, or ``None`` when it has none."""
    path = run_root / METRICS_FILE
    if not path.exists():
        return None
    return EvaluationReport.model_validate(json.loads(path.read_text()))
