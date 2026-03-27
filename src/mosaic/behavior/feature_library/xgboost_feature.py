from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal, TypedDict, final

import numpy as np
import pandas as pd
from pydantic import Field
from xgboost import XGBClassifier

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    GlobalModelParams,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)

from .helpers import ensure_columns, feature_columns
from .registry import register_feature


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

    Params:
        model: Pre-fitted XgboostModelArtifact to load (skip training).
            Default: XgboostModelArtifact().
        strategy: Classification strategy — "multiclass" trains a single
            multi-class model; "one_vs_rest" trains one binary classifier
            per class. Default: "multiclass".
        decision_threshold: Probability threshold(s) for positive
            prediction. A float applies to all classes; a dict maps
            class -> threshold. None uses argmax. Default: None.
        default_class: Class label assigned when no class exceeds the
            decision threshold (required).
        class_weight: If "balanced", adjust sample weights inversely
            proportional to class frequency. Default: "balanced".
        use_smote: If True, apply SMOTE oversampling to the training
            set. Default: False.
        undersample_ratio: If set, undersample majority classes to this
            ratio relative to the minority class before SMOTE.
            Default: None.
        n_estimators: Number of boosting rounds. Default: 100.
        max_depth: Maximum tree depth. Default: 6.
        learning_rate: Boosting learning rate. Default: 0.1.
        subsample: Fraction of training samples used per tree.
            Default: 0.8.
        colsample_bytree: Fraction of features used per tree.
            Default: 0.8.
        random_state: Random seed for reproducibility. Default: 42.
    """

    name = "xgboost"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    ModelArtifact = XgboostModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(GlobalModelParams[XgboostModelArtifact]):
        model: XgboostModelArtifact | None = Field(
            default_factory=XgboostModelArtifact
        )
        strategy: Literal["multiclass", "one_vs_rest"] = "multiclass"
        decision_threshold: float | Mapping[int, float] | None = None
        default_class: int
        class_weight: Literal["balanced"] | None = "balanced"
        use_smote: bool = False
        undersample_ratio: float | None = None
        n_estimators: int = Field(default=100, ge=1)
        max_depth: int = Field(default=6, ge=1)
        learning_rate: float = Field(default=0.1, gt=0)
        subsample: float = Field(default=0.8, gt=0, le=1)
        colsample_bytree: float = Field(default=0.8, gt=0, le=1)
        random_state: int = 42

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
            bundle: XgboostModelBundle = XgboostModelArtifact().from_path(
                cached_path
            )
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
                class_counts = np.bincount(
                    y_train, minlength=max(self._classes) + 1
                )
                weights = np.where(
                    class_counts > 0,
                    len(y_train) / (len(self._classes) * class_counts),
                    0.0,
                )
                sample_weight = weights[y_train]
                model.fit(x_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(x_train, y_train)
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
                y_pred: np.ndarray = np.asarray(self._model.predict(x_test))
            else:
                assert isinstance(self._model, list)
                test_probs = np.column_stack([
                    m.predict_proba(x_test)[:, 1] for m in self._model
                ])
                pred_indices = np.argmax(test_probs, axis=1)
                y_pred = np.array(
                    [self._classes[int(i)] for i in pred_indices], dtype=np.intp
                )

            report: dict[str, object] = _cr(  # pyright: ignore[reportUnknownVariableType, reportAssignmentType]
                y_test, y_pred, output_dict=True
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
            probs = np.column_stack([
                m.predict_proba(feat_matrix)[:, 1] for m in self._model
            ])

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
            predicted_indices = np.argmax(masked_probs, axis=1)
            predicted_labels = np.array(
                [self._classes[int(i)] for i in predicted_indices],
                dtype=np.intp,
            )
            predicted_labels[all_zero] = self.params.default_class
        else:
            predicted_indices = np.argmax(probs, axis=1)
            predicted_labels = np.array(
                [self._classes[int(i)] for i in predicted_indices],
                dtype=np.intp,
            )

        # Build output DataFrame
        meta_cols = [
            c
            for c in [C.frame_col, C.time_col, C.id_col, C.group_col, C.seq_col]
            if c in df.columns
        ]
        result = df[meta_cols].copy()
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

            (run_root / "reports.json").write_text(
                json.dumps(self._metrics, indent=2)
            )
            summary = pd.DataFrame(self._metrics).T
            summary.to_csv(run_root / "summary.csv")
