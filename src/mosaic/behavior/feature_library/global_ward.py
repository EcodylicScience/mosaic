"""
GlobalWardClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from pydantic import Field

from scipy.cluster.hierarchy import linkage as _sch_linkage

from mosaic.core.dataset import register_feature, _resolve_inputs
from ._param_bases import FeatureParams, ArtifactSpec, NpzLoadSpec
from .helpers import _collect_sequence_blocks, _load_artifact_matrix


@register_feature
class GlobalWardClustering:
    """
    Ward hierarchical clustering on a global feature artifact (e.g. global t-SNE templates).
    """

    name    = "global-ward"
    version = "0.1"
    output_type = "global"

    class Params(FeatureParams):
        """Global Ward clustering parameters.

        Attributes:
            artifact: Feature artifact specification.
            method: Linkage method. Default "ward".
            inputset: Alternative loading via named inputset. Default None.
            inputs: Alternative loading via explicit input list. Default None.
        """

        artifact: ArtifactSpec = Field(default_factory=lambda: ArtifactSpec(
            feature="global-tsne",
            pattern="global_templates_features.npz",
            load=NpzLoadSpec(key="templates"),
        ))
        method: str = "ward"
        inputset: str | None = None
        inputs: list[dict[str, object]] | None = None

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = self.Params.from_overrides(params)

        self._ds: object = None
        self._Z: np.ndarray | None = None
        self._X_shape: tuple[int, int] | None = None
        self._marker_written = False
        self._artifact_inputs_meta: dict[str, object] = {}
        self._scope_filter: dict[str, object] | None = None

    # --- framework API ---

    def bind_dataset(self, ds: object) -> None:
        self._ds = ds

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return True

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        X = self._load_artifact_matrix()
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(f"[global-ward] Need a 2D matrix with >=2 samples; got shape={X.shape}")
        method = self.params.method.lower()
        if method != "ward":
            raise ValueError(f"[global-ward] Only 'ward' is supported here, got '{method}'.")

        self._Z = _sch_linkage(X, method=method)
        self._X_shape = tuple(X.shape)

    def partial_fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Emit a single 1-row marker the first time; subsequent calls return empty DF."""
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        ns, nf = (self._X_shape or (np.nan, np.nan))
        return pd.DataFrame([{
            "linkage_method": self.params.method,
            "n_samples": int(ns) if ns == ns else -1,
            "n_features": int(nf) if nf == nf else -1,
            "model_file": "model.joblib",
        }])

    def save_model(self, path: Path) -> None:
        """Persist the linkage and minimal provenance to model.joblib."""
        if self._Z is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "linkage_matrix": self._Z,
            "n_samples": None if self._X_shape is None else int(self._X_shape[0]),
            "n_features": None if self._X_shape is None else int(self._X_shape[1]),
            "version": self.version,
            "params": self.params.model_dump(),
        }, path)

        np.savez_compressed(path.with_suffix(".npz"),
                            linkage_matrix=self._Z,
                            method=self.params.method,
                            n_samples=(None if self._X_shape is None else int(self._X_shape[0])),
                            n_features=(None if self._X_shape is None else int(self._X_shape[1])))

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._Z = bundle.get("linkage_matrix")
        ns = bundle.get("n_samples")
        nf = bundle.get("n_features")
        self._X_shape = (ns, nf) if ns is not None and nf is not None else None
        saved = bundle.get("params", {})
        if isinstance(saved, dict):
            self.params = self.Params.model_validate(saved)

    def _load_artifact_matrix(self) -> np.ndarray:
        """Resolve the artifact, glob the pattern, and load to a single (N,D) matrix."""
        if self._ds is None:
            raise RuntimeError("[global-ward] Feature not bound to a Dataset; call via dataset.run_feature(...)")

        inputset_name = self.params.inputset
        explicit_inputs = self.params.inputs
        if inputset_name or explicit_inputs:
            specs, meta = _resolve_inputs(
                self._ds,
                explicit_inputs,
                inputset_name,
                explicit_override=(explicit_inputs is not None),
            )
            scope = self._scope_filter or {}
            scope_filter = None
            safe_sequences = set(scope.get("safe_sequences") or [])
            if safe_sequences:
                scope_filter = {"safe_sequences": safe_sequences}
            blocks = _collect_sequence_blocks(
                self._ds,
                specs,
                pair_filter_spec=meta.get("pair_filter"),
                scope_filter=scope_filter,
            )
            if not blocks:
                raise RuntimeError("[global-ward] Inputset produced no usable matrices.")
            X = np.vstack(list(blocks.values()))
            if X.ndim != 2:
                raise ValueError(f"[global-ward] Loaded array must be 2D; got shape={X.shape}")
            self._artifact_inputs_meta = meta
            return X.astype(np.float64, copy=False)

        # Simple artifact branch: delegate to shared helper
        # ArtifactSpec supports dict-like access via DictModel, so the helper works
        X = _load_artifact_matrix(self._ds, self.params.artifact)
        return X.astype(np.float64, copy=False)
