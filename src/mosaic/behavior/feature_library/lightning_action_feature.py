"""Lightning-action supervised temporal action segmentation feature.

Wraps the ``lightning-action`` package (Paninski lab, MIT license) as a
mosaic global feature.  Trains a temporal neural network classifier
(DilatedTCN, RNN, or TemporalMLP) on labeled templates and predicts
per-frame action probabilities with temporal context.

Requires the optional ``lightning-action`` package::

    pip install lightning-action

Or install mosaic with the extra::

    pip install mosaic-behavior[lightning-action]
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field

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

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# --- Import guard ---


def _check_lightning_action() -> None:
    """Raise a helpful ImportError if lightning-action is not installed."""
    try:
        import lightning_action  # noqa: F401
    except ImportError:
        msg = (
            "lightning-action is required for LightningActionFeature. "
            "Install with: pip install lightning-action"
        )
        raise ImportError(msg) from None


# --- Model artifact ---


class LightningActionModelBundle(TypedDict):
    model_dir: str  # relative path to checkpoint dir within run_root
    feature_columns: list[str]
    classes: list[int]
    class_names: list[str]
    config: dict[str, object]
    version: str


class LightningActionModelArtifact(JoblibArtifact[LightningActionModelBundle]):
    """Fitted lightning-action model bundle."""

    feature: str = "lightning-action"
    pattern: str = "lightning_action_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Feature class ---


@final
@register_feature
class LightningActionFeature:
    """Supervised temporal action segmentation via lightning-action.

    Trains a temporal neural network classifier (DilatedTCN, RNN, or
    TemporalMLP head + linear classifier) on labeled templates and
    predicts per-frame action probabilities.

    Params:
        model: Pre-fitted LightningActionModelArtifact to load (skip
            training). Default: LightningActionModelArtifact().
        head: Temporal encoder architecture — "dtcn" (dilated temporal
            convolution), "rnn" (LSTM/GRU), or "temporalmlp".
            Default: "dtcn".
        num_hid_units: Hidden units in the temporal encoder.
            Default: 64.
        num_layers: Number of encoder layers. Default: 2.
        num_lags: Lag/kernel size for temporal context. Default: 4.
        activation: Activation function. Default: "lrelu".
        dropout_rate: Dropout rate. Default: 0.1.
        sequence_length: Training sequence length (frames per chunk).
            Default: 500.
        num_epochs: Number of training epochs. Default: 200.
        batch_size: Training batch size. Default: 32.
        learning_rate: Optimizer learning rate. Default: 1e-3.
        weight_decay: Optimizer weight decay. Default: 0.0.
        optimizer: Optimizer type. Default: "Adam".
        weight_classes: If True, weight loss by inverse class frequency.
            Default: True.
        device: Compute device — "cpu" or "gpu". Default: "cpu".
        random_state: Random seed. Default: 42.
        decision_threshold: Probability threshold(s) for positive
            prediction. A float applies to all classes; a dict maps
            class -> threshold. None uses argmax. Default: None.
        default_class: Class label assigned when no class exceeds the
            decision threshold (required).
    """

    name = "lightning-action"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    ModelArtifact = LightningActionModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(GlobalModelParams[LightningActionModelArtifact]):
        model: LightningActionModelArtifact | None = Field(
            default_factory=LightningActionModelArtifact
        )
        # --- Network architecture ---
        head: Literal["temporalmlp", "rnn", "dtcn"] = "dtcn"
        num_hid_units: int = Field(default=64, ge=1)
        num_layers: int = Field(default=2, ge=1)
        num_lags: int = Field(default=4, ge=1)
        activation: str = "lrelu"
        dropout_rate: float = Field(default=0.1, ge=0, le=1)
        # --- Training ---
        sequence_length: int = Field(default=500, ge=10)
        num_epochs: int = Field(default=200, ge=1)
        batch_size: int = Field(default=32, ge=1)
        learning_rate: float = Field(default=1e-3, gt=0)
        weight_decay: float = Field(default=0.0, ge=0)
        optimizer: Literal["Adam", "AdamW"] = "Adam"
        weight_classes: bool = True
        device: str = "cpu"
        random_state: int = 42
        # --- Inference ---
        decision_threshold: float | Mapping[int, float] | None = None
        default_class: int

    def __init__(
        self,
        inputs: LightningActionFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        _check_lightning_action()
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._classes: list[int] | None = None
        self._class_names: list[str] | None = None
        self._templates: pd.DataFrame | None = None
        self._model_dir: Path | None = None
        self._la_model: object | None = None  # cached lightning_action.api.model.Model
        self._config: dict[str, object] | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._feature_columns = None
        self._classes = None
        self._class_names = None
        self._templates = None
        self._model_dir = None
        self._la_model = None
        self._config = None

        # Branch 1: cached model in run_root
        cached_path = run_root / "lightning_action_model.joblib"
        if cached_path.exists():
            bundle: LightningActionModelBundle = (
                LightningActionModelArtifact().from_path(cached_path)
            )
            self._load_bundle(bundle, run_root)
            return True

        # Branch 2: pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._load_bundle(bundle, artifact_paths["model"].parent)
            return True

        # Branch 3: templates to fit
        if self.params.templates is not None and "templates" in artifact_paths:
            self._templates = self.params.templates.from_path(
                artifact_paths["templates"]
            )
            self._feature_columns = feature_columns(self._templates)
            return False

        return False

    def fit(self, inputs: InputStream) -> None:
        from lightning_action.api.model import Model

        if self._templates is None or self._feature_columns is None:
            msg = "[lightning-action] No templates loaded — check load_state."
            raise RuntimeError(msg)

        templates = self._templates
        ensure_columns(templates, ["label", "split"])
        feat_cols = self._feature_columns

        # Determine classes
        self._classes = sorted(int(c) for c in templates["label"].unique())
        self._class_names = [str(c) for c in self._classes]
        n_classes = len(self._classes)
        n_features = len(feat_cols)

        print(
            f"[lightning-action] Training: {n_features} features, "
            f"{n_classes} classes, head={self.params.head}, "
            f"epochs={self.params.num_epochs}",
            file=sys.stderr,
        )

        with tempfile.TemporaryDirectory(prefix="la_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            data_dir = tmpdir / "data"
            output_dir = tmpdir / "output"

            # Write temp CSVs in lightning-action layout
            self._write_training_csvs(templates, feat_cols, data_dir)

            # Build config
            config = self._build_config(n_features, n_classes, str(data_dir))

            # Train
            model = Model.from_config(config)
            model.train(output_dir=str(output_dir), post_inference=False)

            # Persist trained model to a temp location (save_state copies to run_root)
            self._model_dir = Path(tempfile.mkdtemp(prefix="la_model_"))
            shutil.copytree(
                output_dir, self._model_dir / "model", dirs_exist_ok=True
            )
            self._la_model = Model.from_dir(str(self._model_dir / "model"))
            self._config = config

        print("[lightning-action] Training complete.", file=sys.stderr)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        from lightning_action.api.model import Model

        if self._feature_columns is None or self._classes is None:
            msg = (
                "[lightning-action] Model not fitted "
                "— call fit() or load_state() first."
            )
            raise RuntimeError(msg)

        ensure_columns(df, self._feature_columns)

        # Load model if not cached
        if self._la_model is None and self._model_dir is not None:
            model_path = self._model_dir / "model"
            if not model_path.exists():
                # model_dir IS the model directory (loaded from run_root)
                model_path = self._model_dir / "lightning_action_model"
            self._la_model = Model.from_dir(str(model_path))

        if self._la_model is None:
            msg = "[lightning-action] No model available for prediction."
            raise RuntimeError(msg)

        feat_cols = self._feature_columns
        feat_matrix = df[feat_cols].to_numpy(dtype=np.float64)
        n = len(df)

        with tempfile.TemporaryDirectory(prefix="la_apply_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            features_dir = tmpdir / "features"
            pred_dir = tmpdir / "predictions"
            features_dir.mkdir()
            pred_dir.mkdir()

            # Write single-sequence CSV
            pd.DataFrame(feat_matrix, columns=feat_cols).to_csv(
                features_dir / "seq.csv", index=False
            )

            # Run prediction
            self._la_model.predict(
                data_path=str(tmpdir),
                input_dir="features",
                output_dir=str(pred_dir),
            )

            # Read prediction CSV
            pred_files = sorted(pred_dir.glob("*.csv"))
            if not pred_files:
                msg = "[lightning-action] No prediction output found."
                raise RuntimeError(msg)
            pred_df = pd.read_csv(pred_files[0])
            probs = pred_df.to_numpy(dtype=np.float64)

        # Trim or pad to match input length
        if len(probs) > n:
            probs = probs[:n]
        elif len(probs) < n:
            pad = np.full((n - len(probs), probs.shape[1]), np.nan)
            probs = np.vstack([probs, pad])

        # Build output DataFrame (matches XGBoost format)
        meta_cols = [
            c
            for c in [C.frame_col, C.time_col, C.id_col, C.group_col, C.seq_col]
            if c in df.columns
        ]
        result = df[meta_cols].copy()

        for i, cls in enumerate(self._classes):
            result[f"prob_{cls}"] = probs[:, i]

        # Decision threshold logic (same as XGBoost)
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

        # Handle NaN rows (from padding)
        nan_mask = np.isnan(probs).any(axis=1)
        predicted_labels[nan_mask] = self.params.default_class

        result["predicted_label"] = predicted_labels
        return result

    def save_state(self, run_root: Path) -> None:
        if (
            self._model_dir is None
            or self._feature_columns is None
            or self._classes is None
        ):
            return
        run_root.mkdir(parents=True, exist_ok=True)

        # Copy model directory into run_root
        dest = run_root / "lightning_action_model"
        src = self._model_dir / "model"
        if not src.exists():
            src = self._model_dir / "lightning_action_model"
        if src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

        # Save metadata bundle
        bundle: LightningActionModelBundle = {
            "model_dir": "lightning_action_model",
            "feature_columns": self._feature_columns,
            "classes": self._classes,
            "class_names": self._class_names or [],
            "config": self._config or {},
            "version": self.version,
        }
        joblib.dump(bundle, run_root / "lightning_action_model.joblib")

        # Clean up temp model dir (if it's separate from run_root)
        if self._model_dir != run_root and self._model_dir.exists():
            shutil.rmtree(self._model_dir, ignore_errors=True)
        self._model_dir = run_root
        self._la_model = None  # force reload from new location on next apply

    # --- Helpers ---

    def _load_bundle(
        self, bundle: LightningActionModelBundle, base_dir: Path
    ) -> None:
        self._feature_columns = bundle["feature_columns"]
        self._classes = bundle["classes"]
        self._class_names = bundle["class_names"]
        self._config = bundle.get("config")  # pyright: ignore[reportAssignmentType]
        self._model_dir = base_dir
        self._la_model = None  # lazy-load on first apply()

    def _write_training_csvs(
        self,
        templates: pd.DataFrame,
        feat_cols: list[str],
        data_dir: Path,
    ) -> None:
        """Write mosaic templates as CSVs in lightning-action layout."""
        features_dir = data_dir / "features"
        labels_dir = data_dir / "labels"
        features_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        assert self._classes is not None
        assert self._class_names is not None

        for split_name, split_df in templates.groupby("split", sort=False):
            split_df = split_df.reset_index(drop=True)

            # Write features CSV
            split_df[feat_cols].to_csv(
                features_dir / f"{split_name}.csv", index=False
            )

            # Write labels CSV (one-hot encoded)
            label_vals = split_df["label"].values
            one_hot = np.zeros(
                (len(label_vals), len(self._classes)), dtype=int
            )
            for i, cls in enumerate(self._classes):
                one_hot[:, i] = (label_vals == cls).astype(int)
            label_df = pd.DataFrame(one_hot, columns=self._class_names)
            label_df.to_csv(labels_dir / f"{split_name}.csv", index=False)

    def _build_config(
        self, n_features: int, n_classes: int, data_path: str
    ) -> dict[str, object]:
        """Translate mosaic Params into a lightning-action config dict."""
        p = self.params
        return {
            "data": {
                "data_path": data_path,
                "input_dir": "features",
                "weight_classes": p.weight_classes,
            },
            "model": {
                "input_size": n_features,
                "output_size": n_classes,
                "head": p.head,
                "num_hid_units": p.num_hid_units,
                "num_layers": p.num_layers,
                "num_lags": p.num_lags,
                "activation": p.activation,
                "dropout_rate": p.dropout_rate,
                "sequence_length": p.sequence_length,
                "seed": p.random_state,
            },
            "training": {
                "batch_size": p.batch_size,
                "num_epochs": p.num_epochs,
                "device": p.device,
                "sequence_length": p.sequence_length,
                "seed": p.random_state,
                "train_probability": 0.9,
                "val_probability": 0.1,
                "optimizer": {
                    "type": p.optimizer,
                    "lr": p.learning_rate,
                    "wd": p.weight_decay,
                },
            },
        }
