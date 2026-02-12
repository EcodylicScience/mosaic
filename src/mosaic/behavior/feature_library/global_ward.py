"""
GlobalWardClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import joblib

from scipy.cluster.hierarchy import linkage as _sch_linkage

from mosaic.core.dataset import register_feature, _resolve_inputs
from .helpers import _collect_sequence_blocks, _load_artifact_matrix


@register_feature
class GlobalWardClustering:
    """
    Ward hierarchical clustering on a global feature artifact (e.g. global t-SNE templates).

    Params
    ------
    artifact : dict (required)
        {
          "feature": "global-tsne",            # feature that produced the artifact
          "run_id": None,                      # None => latest finished run
          "pattern": "global_templates_features.npz",
          "load": {"kind": "npz", "key": "templates", "transpose": False}
        }
    method : str = "ward"
        Linkage method (Ward requires Euclidean distances).
    """

    name    = "global-ward"
    version = "0.1"
    output_type = "global"

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "artifact": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            "method": "ward",
        }
        if params:
            # shallow-merge
            self.params.update({k: v for k, v in params.items() if k != "artifact"})
            if "artifact" in params:
                a = dict(self.params["artifact"])
                a.update(params["artifact"])
                # nested load merge
                if "load" in params["artifact"]:
                    ld = dict(a.get("load", {}))
                    ld.update(params["artifact"]["load"])
                    a["load"] = ld
                self.params["artifact"] = a

        self._ds = None               # bound Dataset
        self._Z  = None               # linkage matrix (np.ndarray)
        self._X_shape = None          # (n_samples, n_features)
        self._marker_written = False  # ensure only one parquet marker row gets written
        self._artifact_inputs_meta: Dict[str, Any] = {}
        self._scope_filter: Optional[dict] = None

    # ---------- framework API ----------
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return True  # Skip run_feature pre-loading; we load from artifacts

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        # Ignore X_iter; we load from the declared artifact to avoid accidental wrong inputs
        X = self._load_artifact_matrix()
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(f"[global-ward] Need a 2D matrix with >=2 samples; got shape={X.shape}")
        method = str(self.params.get("method", "ward")).lower()
        if method != "ward":
            # You could allow other methods, but Ward is the intended one
            raise ValueError(f"[global-ward] Only 'ward' is supported here, got '{method}'.")

        # SciPy linkage expects samples as rows
        self._Z = _sch_linkage(X, method=method)
        self._X_shape = tuple(X.shape)

    def partial_fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        We don't produce per-(group,sequence) outputs. We emit a single 1-row marker
        the first time transform is called so the run is indexed; subsequent calls return
        an empty DF (so nothing else is written).
        """
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        ns, nf = (self._X_shape or (np.nan, np.nan))
        return pd.DataFrame([{
            "linkage_method": str(self.params.get("method", "ward")),
            "n_samples": int(ns) if ns == ns else -1,     # handle NaN
            "n_features": int(nf) if nf == nf else -1,
            "model_file": "model.joblib",
        }])

    def save_model(self, path: Path) -> None:
        """
        Persist the linkage and minimal provenance to model.joblib.
        Also write a human-usable .npz copy (optional).
        """
        if self._Z is None:
            # nothing fitted; skip
            return
        # Ensure parent exists and path is file path (run_feature passes a file path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "linkage_matrix": self._Z,
            "method": str(self.params.get("method", "ward")),
            "n_samples": None if self._X_shape is None else int(self._X_shape[0]),
            "n_features": None if self._X_shape is None else int(self._X_shape[1]),
            "version": self.version,
            "params": self.params,
        }, path)

        # Optional: also store as npz right next to model (convenient for quick numpy loads)
        np.savez_compressed(path.with_suffix(".npz"),
                            linkage_matrix=self._Z,
                            method=str(self.params.get("method", "ward")),
                            n_samples=(None if self._X_shape is None else int(self._X_shape[0])),
                            n_features=(None if self._X_shape is None else int(self._X_shape[1])))

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._Z = bundle.get("linkage_matrix", None)
        self._X_shape = (bundle.get("n_samples", None), bundle.get("n_features", None))

    def _load_artifact_matrix(self) -> np.ndarray:
        """
        Resolve the artifact (feature/run_id), glob the pattern, and load to a single (N,D) matrix.
        """
        if self._ds is None:
            raise RuntimeError("[global-ward] Feature not bound to a Dataset; call via dataset.run_feature(...)")

        art = self.params.get("artifact", {})
        inputset_name = art.get("inputset")
        explicit_inputs = art.get("inputs") if ("inputs" in art) else None
        if inputset_name or explicit_inputs:
            specs, meta = _resolve_inputs(
                self._ds,
                explicit_inputs,
                inputset_name,
                explicit_override=("inputs" in art),
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
        X = _load_artifact_matrix(self._ds, art)
        return X.astype(np.float64, copy=False)
