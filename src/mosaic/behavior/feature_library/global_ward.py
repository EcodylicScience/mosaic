"""
GlobalWardClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from scipy.cluster.hierarchy import linkage as _sch_linkage

from mosaic.core.dataset import register_feature

from .helpers import StreamingFeatureHelper, _load_artifact_matrix
from .global_tsne import GlobalTSNE
from .params import (
    ArtifactSpec,
    InputRequire,
    Inputs,
    JoblibLoadSpec,
    LoadSpec,
    NNResult,
    NpzLoadSpec,
    OutputType,
    Params,
    Result,
)


@final
@register_feature
class GlobalWardClustering:
    """
    Ward hierarchical clustering on a global feature artifact (e.g. global t-SNE templates).
    """

    name = "global-ward"
    version = "0.2"
    output_type: OutputType = "global"
    parallelizable = False

    class ModelArtifact(ArtifactSpec):
        """Ward linkage model (model.joblib)."""

        feature: str = "global-ward"
        pattern: str = "model.joblib"
        load: LoadSpec = Field(default_factory=JoblibLoadSpec)

    class LinkageArtifact(ArtifactSpec):
        """Linkage matrix backup (model.npz)."""

        feature: str = "global-ward"
        pattern: str = "model.npz"
        load: LoadSpec = Field(
            default_factory=lambda: NpzLoadSpec(key="linkage_matrix")
        )

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "empty"

    class Params(Params):
        """Global Ward clustering parameters.

        Attributes:
            templates: Templates artifact to cluster.
            method: Linkage method. Default "ward".
            pair_filter: Nearest-neighbor pair filter. Default None.
        """

        templates: GlobalTSNE.TemplatesArtifact = Field(
            default_factory=GlobalTSNE.TemplatesArtifact
        )
        method: str = "ward"
        pair_filter: NNResult | None = None

    def __init__(
        self,
        inputs: GlobalWardClustering.Inputs = Inputs(()),
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True

        self._ds: object = None
        self._Z: np.ndarray | None = None
        self._X_shape: tuple[int, int] | None = None
        self._marker_written = False
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
        # NOTE: time/frame scope filters (filter_start_frame, etc.) are not
        # applied when loading own data. run_feature() raises RuntimeError
        # if these filters are set. Future work: apply them during loading.
        return True

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        X = self._load_artifact_matrix()
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(
                f"[global-ward] Need a 2D matrix with >=2 samples; got shape={X.shape}"
            )
        method = self.params.method.lower()
        if method != "ward":
            raise ValueError(
                f"[global-ward] Only 'ward' is supported here, got '{method}'."
            )

        self._Z = _sch_linkage(X, method=method)
        self._X_shape = tuple(X.shape)

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Emit a single 1-row marker the first time; subsequent calls return empty DF."""
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        ns, nf = self._X_shape or (np.nan, np.nan)
        return pd.DataFrame(
            [
                {
                    "linkage_method": self.params.method,
                    "n_samples": int(ns) if ns == ns else -1,
                    "n_features": int(nf) if nf == nf else -1,
                    "model_file": "model.joblib",
                }
            ]
        )

    def save_model(self, path: Path) -> None:
        """Persist the linkage and minimal provenance to model.joblib."""
        if self._Z is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "linkage_matrix": self._Z,
                "n_samples": None if self._X_shape is None else int(self._X_shape[0]),
                "n_features": None if self._X_shape is None else int(self._X_shape[1]),
                "version": self.version,
                "params": self.params.model_dump(),
            },
            path,
        )

        np.savez_compressed(
            path.with_suffix(".npz"),
            linkage_matrix=self._Z,
            method=self.params.method,
            n_samples=(None if self._X_shape is None else int(self._X_shape[0])),
            n_features=(None if self._X_shape is None else int(self._X_shape[1])),
        )

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
            raise RuntimeError(
                "[global-ward] Feature not bound to a Dataset; call via dataset.run_feature(...)"
            )

        feature_inputs = self.inputs.feature_inputs
        if feature_inputs:
            # Stacked-features mode: load per-frame features via Results
            helper = StreamingFeatureHelper(self._ds, "global-ward")
            if self.params.pair_filter:
                helper.set_pair_filter(self.params.pair_filter)
            scope = self._scope_filter or {}
            scope_filter = None
            safe_sequences = set(scope.get("safe_sequences") or [])
            if safe_sequences:
                scope_filter = {"safe_sequences": safe_sequences}
            manifest = helper.build_manifest_from_results(
                feature_inputs, scope_filter=scope_filter
            )
            if not manifest:
                raise RuntimeError(
                    "[global-ward] Result inputs produced no usable matrices."
                )
            blocks = {}
            for key, entries in manifest.items():
                X_key, _ = helper.load_key_data(entries, extract_frames=False, key=key)
                if X_key is not None and X_key.size > 0:
                    blocks[key] = X_key
            if not blocks:
                raise RuntimeError(
                    "[global-ward] Result inputs produced no usable matrices."
                )
            X = np.vstack(list(blocks.values()))
            if X.ndim != 2:
                raise ValueError(
                    f"[global-ward] Loaded array must be 2D; got shape={X.shape}"
                )
            return X.astype(np.float64, copy=False)

        # Artifact-only mode: delegate to shared helper
        X = _load_artifact_matrix(self._ds, self.params.templates)
        return X.astype(np.float64, copy=False)
