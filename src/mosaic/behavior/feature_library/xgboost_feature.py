from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypedDict, final

import numpy as np
import pandas as pd
from pydantic import Field
from xgboost import XGBClassifier

from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    GlobalModelParams,
    LabeledTemplatesRef,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)
from mosaic.core.params import Declared

from .helpers import ensure_columns, feature_columns, meta_columns
from .registry import register_feature

_TEMPLATES_DESCRIPTION = (
    "The labeled templates artifact to train on. Mutually exclusive with model."
)

_MODEL_DESCRIPTION = (
    "A pre-fitted XgboostModelArtifact to load, skipping training. Mutually "
    "exclusive with templates."
)

_STRATEGY_DESCRIPTION = (
    "The classification strategy. multiclass fits one multi-class model. "
    "one_vs_rest fits one binary classifier per class."
)

_DECISION_THRESHOLD_DESCRIPTION = (
    "Unset, the predicted class is the class with the highest probability. "
    "A float applies one probability threshold to every class, and a "
    "mapping from class to threshold applies a per-class one. Any row "
    "where no class clears its threshold is assigned default_class."
)

_DEFAULT_CLASS_DESCRIPTION = (
    "The class assigned to a row where decision_threshold is set and no "
    "class clears its threshold."
)

_CLASS_WEIGHT_DESCRIPTION = (
    "Unset, every training sample is weighted equally. balanced weights "
    "each sample inversely to its class's frequency."
)

_USE_SMOTE_DESCRIPTION = (
    "Apply SMOTE oversampling to the training set, after any "
    "undersample_ratio undersampling."
)

_UNDERSAMPLE_RATIO_DESCRIPTION = (
    "Unset, no class is undersampled before training. Set, every class "
    "larger than the minority class is undersampled to this many times "
    "the minority class's size, never below the minority class's own "
    "size."
)

_N_ESTIMATORS_DESCRIPTION = "The number of boosting rounds the model fits."

_MAX_DEPTH_DESCRIPTION = "The maximum depth of each boosted tree."

_LEARNING_RATE_DESCRIPTION = (
    "The boosting learning rate (XGBoost's eta), scaling each tree's "
    "contribution to the ensemble."
)

_SUBSAMPLE_DESCRIPTION = (
    "The fraction of training rows sampled for each boosting round."
)

_COLSAMPLE_BYTREE_DESCRIPTION = "The fraction of feature columns sampled for each tree."

_RANDOM_STATE_DESCRIPTION = (
    "The random seed for XGBoost training and, when enabled, for "
    "undersampling and SMOTE."
)


class XgboostModelBundle(TypedDict):
    model: XGBClassifier | list[XGBClassifier]
    feature_columns: list[str]
    classes: list[int]
    strategy: str
    version: str


class XgboostModelArtifact(JoblibArtifact[XgboostModelBundle]):
    """Fitted XGBoost model bundle (xgboost_model.joblib)."""

    feature: str = "xgboost"
    pattern: str = "xgboost_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


@final
@register_feature
class XgboostFeature:
    """XGBoost behavior classifier as a pipeline feature.

    Trains on labeled templates (from ExtractLabeledTemplates) and runs
    per-sequence inference. Supports multiclass and one-vs-rest strategies.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.xgboost_feature.XgboostFeature.Params`.
    """

    category = "global"
    name = "xgboost"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    ModelArtifact = XgboostModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(GlobalModelParams[XgboostModelArtifact, LabeledTemplatesRef]):
        templates: Annotated[
            LabeledTemplatesRef | None, Declared(_TEMPLATES_DESCRIPTION)
        ] = None
        model: Annotated[XgboostModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = (
            Field(default_factory=XgboostModelArtifact)
        )
        strategy: Annotated[
            Literal["multiclass", "one_vs_rest"], Declared(_STRATEGY_DESCRIPTION)
        ] = "multiclass"
        decision_threshold: Annotated[
            float | Mapping[int, float] | None,
            Declared(_DECISION_THRESHOLD_DESCRIPTION),
        ] = None
        default_class: Annotated[int, Declared(_DEFAULT_CLASS_DESCRIPTION)]
        class_weight: Annotated[
            Literal["balanced"] | None, Declared(_CLASS_WEIGHT_DESCRIPTION)
        ] = "balanced"
        use_smote: Annotated[bool, Declared(_USE_SMOTE_DESCRIPTION)] = False
        undersample_ratio: Annotated[
            float | None, Declared(_UNDERSAMPLE_RATIO_DESCRIPTION)
        ] = None
        n_estimators: Annotated[int, Declared(_N_ESTIMATORS_DESCRIPTION)] = Field(
            default=100, ge=1
        )
        max_depth: Annotated[int, Declared(_MAX_DEPTH_DESCRIPTION)] = Field(
            default=6, ge=1
        )
        learning_rate: Annotated[float, Declared(_LEARNING_RATE_DESCRIPTION)] = Field(
            default=0.1, gt=0
        )
        subsample: Annotated[float, Declared(_SUBSAMPLE_DESCRIPTION)] = Field(
            default=0.8, gt=0, le=1
        )
        colsample_bytree: Annotated[float, Declared(_COLSAMPLE_BYTREE_DESCRIPTION)] = (
            Field(default=0.8, gt=0, le=1)
        )
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42

    def __init__(
        self,
        inputs: XgboostFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._model: XGBClassifier | list[XGBClassifier] | None = None
        self._classes: list[int] | None = None
        self._templates: pd.DataFrame | None = None
        self._metrics: dict[str, object] | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._feature_columns = None
        self._model = None
        self._classes = None
        self._templates = None
        self._metrics = None

        # Branch 1: cached model in run_root
        cached_path = run_root / "xgboost_model.joblib"
        if cached_path.exists():
            bundle: XgboostModelBundle = XgboostModelArtifact().from_path(cached_path)
            self._model = bundle["model"]
            self._feature_columns = bundle["feature_columns"]
            self._classes = bundle["classes"]
            return True

        # Branch 2: pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._model = bundle["model"]
            self._feature_columns = bundle["feature_columns"]
            self._classes = bundle["classes"]
            return True

        # Branch 3: labeled templates to fit from
        if self.params.templates is not None and "templates" in artifact_paths:
            self._templates = self.params.templates.from_path(
                artifact_paths["templates"]
            )
            self._feature_columns = feature_columns(self._templates)
            return False

        return False

    def _undersample_then_smote(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multiclass-aware undersampling with optional SMOTE oversampling."""
        from imblearn.over_sampling import SMOTE  # type: ignore[import-untyped]

        classes, counts = np.unique(labels, return_counts=True)
        minority_count = int(counts.min())

        # Undersample majority classes
        if self.params.undersample_ratio is not None:
            ratio = self.params.undersample_ratio
            keep_indices: list[np.intp] = []
            rng = np.random.default_rng(self.params.random_state)
            for cls, count in zip(classes, counts):
                cls_indices = np.where(labels == cls)[0]
                target = max(minority_count, min(count, int(minority_count * ratio)))
                if count > target:
                    chosen = rng.choice(cls_indices, size=target, replace=False)
                    keep_indices.extend(chosen.tolist())
                else:
                    keep_indices.extend(cls_indices.tolist())
            indices_array = np.array(keep_indices)
            features = features[indices_array]
            labels = labels[indices_array]

        # SMOTE oversampling
        if self.params.use_smote:
            smote = SMOTE(random_state=self.params.random_state)
            resampled = smote.fit_resample(features, labels)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            features = np.asarray(resampled[0])  # pyright: ignore[reportUnknownArgumentType]
            labels = np.asarray(resampled[1])  # pyright: ignore[reportUnknownArgumentType]

        return features, labels

    def _to_index(self, labels: np.ndarray) -> np.ndarray:
        """Class labels -> contiguous 0-based column indices."""
        assert self._classes is not None
        lookup = {c: i for i, c in enumerate(self._classes)}
        return np.array([lookup[int(v)] for v in labels], dtype=np.intp)

    def _from_index(self, indices: np.ndarray) -> np.ndarray:
        """Column indices -> the class labels they stand for."""
        assert self._classes is not None
        return np.array([self._classes[int(i)] for i in indices], dtype=np.intp)

    def fit(self, inputs: InputStream) -> None:
        if self._templates is None:
            msg = "No templates loaded -- call load_state first"
            raise RuntimeError(msg)
        if self._feature_columns is None:
            msg = "No feature columns determined"
            raise RuntimeError(msg)

        templates = self._templates
        ensure_columns(templates, ["label", "split"])
        train_mask = templates["split"] == "train"
        test_mask = templates["split"] == "test"

        train_df = templates.loc[train_mask]
        test_df = templates.loc[test_mask]

        x_train = train_df[self._feature_columns].to_numpy(dtype=np.float64)
        y_train = train_df["label"].to_numpy(dtype=np.intp)
        self._classes = sorted(int(c) for c in np.unique(y_train))

        # Optional rebalancing
        if self.params.use_smote or self.params.undersample_ratio is not None:
            x_train, y_train = self._undersample_then_smote(x_train, y_train)

        if self.params.strategy == "multiclass":
            # XGBoost's multiclass objective requires labels 0..num_class-1, while
            # these are behaviour ids -- and a training split need not contain every
            # one. A class missing here (an interaction that does not occur in the
            # training recordings) would otherwise leave `num_class` and the label
            # values disagreeing, and XGBClassifier refuses the fit outright. Index
            # space is what the rest of this class already speaks: `apply` maps a
            # probability column back through `self._classes`, and so does the
            # one-vs-rest branch below.
            y_fit = self._to_index(y_train)
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(self._classes),
                n_estimators=self.params.n_estimators,
                max_depth=self.params.max_depth,
                learning_rate=self.params.learning_rate,
                subsample=self.params.subsample,
                colsample_bytree=self.params.colsample_bytree,
                random_state=self.params.random_state,
                eval_metric="mlogloss",
            )
            if self.params.class_weight == "balanced":
                class_counts = np.bincount(y_fit, minlength=len(self._classes))
                weights = np.where(
                    class_counts > 0,
                    len(y_fit) / (len(self._classes) * class_counts),
                    0.0,
                )
                sample_weight = weights[y_fit]
                model.fit(x_train, y_fit, sample_weight=sample_weight)
            else:
                model.fit(x_train, y_fit)
            self._model = model

        else:
            # one_vs_rest strategy
            models: list[XGBClassifier] = []
            for cls in self._classes:
                binary_y = (y_train == cls).astype(int)
                ovr_model = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=self.params.n_estimators,
                    max_depth=self.params.max_depth,
                    learning_rate=self.params.learning_rate,
                    subsample=self.params.subsample,
                    colsample_bytree=self.params.colsample_bytree,
                    random_state=self.params.random_state,
                    eval_metric="logloss",
                )
                if self.params.class_weight == "balanced":
                    n_pos = int(binary_y.sum())
                    n_neg = len(binary_y) - n_pos
                    scale = n_neg / max(n_pos, 1)
                    ovr_model.set_params(scale_pos_weight=scale)
                ovr_model.fit(x_train, binary_y)
                models.append(ovr_model)
            self._model = models

        # Evaluate on test split if available
        if len(test_df) > 0:
            from sklearn.metrics import classification_report as _cr  # pyright: ignore[reportUnknownVariableType]

            x_test = test_df[self._feature_columns].to_numpy(dtype=np.float64)
            y_test = test_df["label"].to_numpy(dtype=np.intp)

            if self.params.strategy == "multiclass":
                assert isinstance(self._model, XGBClassifier)
                # The model was fitted in index space, so its predictions are
                # column indices; the report has to be in label space.
                y_pred: np.ndarray = self._from_index(
                    np.asarray(self._model.predict(x_test))
                )
            else:
                assert isinstance(self._model, list)
                test_probs = np.column_stack(
                    [m.predict_proba(x_test)[:, 1] for m in self._model]
                )
                y_pred = self._from_index(np.argmax(test_probs, axis=1))

            # `zero_division=0` because the splits are by sequence, so the test
            # split can hold a class the training one did not -- and precision or
            # recall for a class the model was never given is genuinely 0, not an
            # undefined value worth warning about on every fit.
            report: dict[str, object] = _cr(  # pyright: ignore[reportUnknownVariableType, reportAssignmentType]
                y_test, y_pred, output_dict=True, zero_division=0
            )
            self._metrics = report

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._feature_columns is None:
            msg = "No feature columns -- model not loaded"
            raise RuntimeError(msg)
        if self._model is None:
            msg = "No model fitted -- call fit() or load_state() first"
            raise RuntimeError(msg)
        if self._classes is None:
            msg = "No classes determined -- model not loaded"
            raise RuntimeError(msg)

        ensure_columns(df, self._feature_columns)

        feat_matrix = df[self._feature_columns].to_numpy(dtype=np.float64)

        # Get probabilities
        if self.params.strategy == "multiclass":
            assert isinstance(self._model, XGBClassifier)
            probs = self._model.predict_proba(feat_matrix)
        else:
            assert isinstance(self._model, list)
            probs = np.column_stack(
                [m.predict_proba(feat_matrix)[:, 1] for m in self._model]
            )

        # Apply thresholds
        threshold = self.params.decision_threshold
        if threshold is not None:
            masked_probs = probs.copy()
            for col_idx, cls in enumerate(self._classes):
                if isinstance(threshold, Mapping):
                    thresh_val = threshold.get(cls, 0.0)
                else:
                    thresh_val = threshold
                masked_probs[:, col_idx] = np.where(
                    probs[:, col_idx] >= thresh_val,
                    probs[:, col_idx],
                    0.0,
                )
            # Rows where all probs are zeroed out -> default_class
            all_zero = masked_probs.sum(axis=1) == 0
            predicted_labels = self._from_index(np.argmax(masked_probs, axis=1))
            predicted_labels[all_zero] = self.params.default_class
        else:
            predicted_labels = self._from_index(np.argmax(probs, axis=1))

        # Build output DataFrame. The whole identity travels with the predictions,
        # `perspective` included: this feature is routinely fed pair-level input,
        # and without it the output is two rows per frame with nothing to tell them
        # apart -- unjoinable against the very features it was trained on.
        result = df[meta_columns(df)].copy()
        for col_idx, cls in enumerate(self._classes):
            result[f"prob_{cls}"] = probs[:, col_idx]
        result["predicted_label"] = predicted_labels

        return result

    def save_state(self, run_root: Path) -> None:
        if (
            self._model is None
            or self._feature_columns is None
            or self._classes is None
        ):
            return
        run_root.mkdir(parents=True, exist_ok=True)

        import joblib

        bundle: XgboostModelBundle = {
            "model": self._model,
            "feature_columns": self._feature_columns,
            "classes": self._classes,
            "strategy": self.params.strategy,
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "xgboost_model.joblib")

        if self._metrics is not None:
            import json

            (run_root / "reports.json").write_text(json.dumps(self._metrics, indent=2))
            summary = pd.DataFrame(self._metrics).T
            summary.to_csv(run_root / "summary.csv")
