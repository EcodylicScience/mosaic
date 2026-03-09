"""
ModelPredictFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import final

import pandas as pd

from mosaic.core.dataset import _model_run_root, register_feature

from .params import Inputs, OutputType, Params, Result


@final
@register_feature
class ModelPredictFeature:
    """
    Generic wrapper that loads a trained model run and applies it over per-sequence feature tables.
    """

    name = "model-predict"
    version = "0.1"
    parallelizable = True
    output_type: OutputType = None

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        """Model-predict parameters.

        Attributes:
            model_class: Fully-qualified Python class path for the model.
            model_params: Kwargs dict for the model constructor.
            model_run_id: Run ID of the trained model to load.
            model_name: Override for model storage name.
            output_feature_name: Override for the output feature name.
        """

        model_class: str | None = None
        model_params: dict[str, object] | None = None
        model_run_id: str | None = None
        model_name: str | None = None
        output_feature_name: str | None = None

    def __init__(
        self,
        inputs: ModelPredictFeature.Inputs,
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ds = None
        self._model = None
        self._model_name: str | None = None
        self._model_run_id: str | None = None
        self.storage_feature_name = self.params.output_feature_name or self.name
        self.storage_use_input_suffix = True
        self._scope_filter: dict[str, object] = {}

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    def bind_dataset(self, ds):
        self._ds = ds
        model_class_path = self.params.model_class
        if not model_class_path:
            raise ValueError("ModelPredictFeature params must include 'model_class'.")
        module_path, class_name = model_class_path.rsplit(".", 1)
        ModelCls = getattr(importlib.import_module(module_path), class_name)
        model_kwargs = self.params.model_params
        self._model = ModelCls(model_kwargs) if model_kwargs else ModelCls()
        if hasattr(self._model, "bind_dataset"):
            self._model.bind_dataset(ds)
        run_id = str(self.params.model_run_id or "").strip()
        if not run_id:
            raise ValueError("ModelPredictFeature params must include 'model_run_id'.")
        storage_model_name = self.params.model_name or getattr(
            self._model, "storage_model_name", getattr(self._model, "name", None)
        )
        if not storage_model_name:
            raise ValueError("Model must define 'name' or params['model_name'].")
        run_root = _model_run_root(ds, storage_model_name, run_id)
        if not run_root.exists():
            raise FileNotFoundError(f"Model artifacts not found: {run_root}")
        if not hasattr(self._model, "load_trained_model"):
            raise RuntimeError(
                f"Model '{model_class_path}' lacks load_trained_model()."
            )
        self._model.load_trained_model(run_root)
        self._model_name = storage_model_name
        self._model_run_id = run_id
        output_name = self.params.output_feature_name or f"{storage_model_name}-pred"
        self.storage_feature_name = output_name
        self.storage_use_input_suffix = True

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def save_model(self, path: Path) -> None:
        raise NotImplementedError

    def load_model(self, path: Path) -> None:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError(
                "ModelPredictFeature has no model loaded; call bind_dataset first."
            )
        if df is None or df.empty:
            return pd.DataFrame()
        sequence = (
            str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
        )
        group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else ""
        meta = {
            "sequence": sequence,
            "group": group,
            "model_run_id": self._model_run_id,
            "model_name": self._model_name,
        }
        if not hasattr(self._model, "predict_sequence"):
            raise RuntimeError("Model does not implement predict_sequence().")
        result = self._model.predict_sequence(df, meta)
        if result is None:
            return pd.DataFrame()
        if isinstance(result, dict):
            result = pd.DataFrame(result)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        if "sequence" not in result.columns:
            result["sequence"] = sequence
        if "group" not in result.columns:
            result["group"] = group
        if self._model_run_id and "model_run_id" not in result.columns:
            result["model_run_id"] = self._model_run_id
        return result
