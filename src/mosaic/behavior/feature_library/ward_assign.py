"""
WardAssignClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import fcluster

from mosaic.core.dataset import register_feature, _resolve_inputs, _dataset_base_dir
from .helpers import (
    StreamingFeatureHelper,
    _parse_scope_filter, _build_sequence_lookup, _resolve_sequence_identity,
    _get_feature_run_root, _load_joblib_artifact, _load_artifact_matrix,
    _build_index_row,
)
from mosaic.core.helpers import to_safe_name


@register_feature
class WardAssignClustering:
    """
    Assign Ward clusters (cut at n_clusters) to full-frame feature streams by stacking
    multiple prior features (e.g., pair-wavelet social + ego) and reusing the scaler
    from GlobalTSNE.

    Params
    ------
    ward_model : dict
        { "feature": "global-ward__from__global-tsne",
          "run_id": None,
          "pattern": "model.joblib" }
    artifact : dict
        Same structure as GlobalWardClustering.artifact (used to reload the template matrix).
    scaler : optional dict
        Same contract as GlobalKMeans.assign.scaler (joblib w/ StandardScaler).
    inputs : list[dict]
        Feature specs to concatenate per sequence. All inputs are loaded from disk
        (typically resolved via an inputset) and aligned per sequence before assignment.
        To retain real frame indices in the outputs, set `load.frame_column` (defaults
        to "frame" when present) for at least one input.
    n_clusters : int
        Desired Ward cut.
    recalc : bool
        If True, force recomputation even if outputs already exist (pass overwrite=True to
        dataset.run_feature when rerunning). Defaults to False.
    """

    name = "ward-assign"
    version = "0.1"
    parallelizable = True
    output_type = "global"
    skip_transform_phase = True

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "ward_model": {
                "feature": "global-ward__from__global-tsne",
                "run_id": None,
                "pattern": "model.joblib",
            },
            "artifact": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            "scaler": None,
            "inputs": [],
            "inputset": None,
            "n_clusters": 20,
            "recalc": False,
        }
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    d = dict(self.params[k])
                    d.update(v)
                    self.params[k] = d
                else:
                    self.params[k] = v

        ward_feat_name = self.params["ward_model"].get("feature", "global-ward")
        self.storage_feature_name = f"ward-assign__from__{ward_feat_name}"
        self.storage_use_input_suffix = False
        self.skip_existing_outputs = False

        self._ds = None
        self._Z = None
        self._templates = None
        self._cluster_ids = None
        self._assign_nn = None
        self._scaler = None
        self._inputs = []
        self._processed_sequences: set[str] = set()  # Track which sequences were written
        self._run_root: Optional[Path] = None  # Set by dataset.run_feature before fit()
        self._allowed_safe_sequences: Optional[set[str]] = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._scope_filter: Optional[dict] = None
        self._inputs_overridden = bool(params and "inputs" in params)
        self._inputs_meta: Dict[str, Any] = {}
        self._sequence_lookup_cache: Optional[dict[str, tuple[str, str]]] = None
        self._additional_index_rows: list[dict] = []

    def bind_dataset(self, ds):
        self._ds = ds

    def set_run_root(self, path: Path) -> None:
        """Set the output directory for immediate file writes during fit()."""
        self._run_root = path

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}
        self._allowed_safe_sequences, self._pair_map = _parse_scope_filter(scope)
        self._sequence_lookup_cache = None

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def loads_own_data(self): return True  # Skip run_feature pre-loading; we load from artifacts
    def partial_fit(self, X): raise NotImplementedError

    # ---------- helpers ----------
    def _get_sequence_lookup(self) -> dict[str, tuple[str, str]]:
        if self._sequence_lookup_cache is not None:
            return self._sequence_lookup_cache
        self._sequence_lookup_cache = _build_sequence_lookup(
            self._ds, self._allowed_safe_sequences
        )
        return self._sequence_lookup_cache

    def _write_sequence_outputs(
        self,
        safe_seq: str,
        labels: np.ndarray,
        frames: Optional[np.ndarray],
        id1_vals: Optional[np.ndarray] = None,
        id2_vals: Optional[np.ndarray] = None,
        entity_level: str = "global",
    ) -> None:
        if self._run_root is None:
            print(f"[ward-assign] WARN: _run_root not set, labels for {safe_seq} not written", file=sys.stderr)
            return
        labels_arr = np.asarray(labels, dtype=np.int32).ravel()
        frame_arr = None if frames is None else np.asarray(frames, dtype=np.int64).ravel()
        if frame_arr is None or len(frame_arr) != len(labels_arr):
            frame_arr = np.arange(len(labels_arr), dtype=np.int64)
        if id1_vals is None or len(id1_vals) != len(labels_arr):
            id1_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)
        if id2_vals is None or len(id2_vals) != len(labels_arr):
            id2_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)

        # Legacy npz artifact (for backward compatibility with viz/analysis code).
        npz_path = self._run_root / f"global_ward_labels_seq={safe_seq}.npz"
        np.savez_compressed(
            npz_path,
            labels=labels_arr,
            frames=frame_arr,
            id1=id1_vals,
            id2=id2_vals,
            entity_level=np.array([str(entity_level or "global")], dtype=object),
        )

        # Standard per-sequence parquet artifact for index-based loading.
        group, sequence = _resolve_sequence_identity(
            safe_seq, self._pair_map, self._get_sequence_lookup()
        )
        safe_group = to_safe_name(group) if group else ""
        out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        out_path = self._run_root / out_name
        df_out = pd.DataFrame({
            "frame": frame_arr.astype(np.int64, copy=False),
            "cluster": labels_arr,
            "id1": pd.array(id1_vals, dtype="Int64"),
            "id2": pd.array(id2_vals, dtype="Int64"),
            "entity_level": np.full(len(labels_arr), str(entity_level or "global"), dtype=object),
            "sequence": sequence,
            "group": group,
        })
        df_out.to_parquet(out_path, index=False)
        self._additional_index_rows.append(
            _build_index_row(safe_seq, group, sequence, out_path, int(len(df_out)),
                             dataset_root=_dataset_base_dir(self._ds) if self._ds else None)
        )
        self._processed_sequences.add(safe_seq)

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        import gc
        import pyarrow as pa
        if self._ds is None:
            raise RuntimeError("[ward-assign] Dataset not bound. Use dataset.run_feature(...).")
        self._processed_sequences = set()
        self._additional_index_rows = []

        inputs = self.params.get("inputs") or []
        inputset_name = self.params.get("inputset")
        explicit_inputs = inputs if (self._inputs_overridden or not inputset_name) else None
        self._inputs, self._inputs_meta = _resolve_inputs(
            self._ds,
            explicit_inputs,
            inputset_name,
            explicit_override=self._inputs_overridden,
        )

        # Load Ward linkage
        ward_spec = self.params["ward_model"]
        _, ward_root = _get_feature_run_root(self._ds, ward_spec["feature"], ward_spec.get("run_id"))
        pattern = ward_spec.get("pattern", "model.joblib")
        files = sorted(ward_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No Ward model '{pattern}' in {ward_root}")
        bundle = joblib.load(files[0])
        self._Z = bundle.get("linkage_matrix")
        if self._Z is None:
            raise ValueError("[ward-assign] Ward model missing linkage_matrix.")

        # Reload artifact matrix (templates) to derive centroids
        self._templates = _load_artifact_matrix(self._ds, self.params.get("artifact", {}))
        n_clusters = int(self.params.get("n_clusters", 20))
        labels_templates = fcluster(self._Z, n_clusters, criterion="maxclust")
        uniq = np.unique(labels_templates)
        centroids = []
        for cid in uniq:
            mask = labels_templates == cid
            if not mask.any():
                continue
            centroids.append(self._templates[mask].mean(axis=0))
        centroids = np.vstack(centroids)
        self._cluster_ids = uniq.astype(int)
        self._assign_nn = NearestNeighbors(n_neighbors=1).fit(centroids)

        # Free templates and linkage matrix - no longer needed after centroid computation
        del self._templates, self._Z, labels_templates, centroids
        self._templates = None
        self._Z = None
        gc.collect()

        # optional scaler
        scaler_spec = self.params.get("scaler")
        self._scaler = _load_joblib_artifact(self._ds, scaler_spec) if scaler_spec else None

        # Use StreamingFeatureHelper for manifest building and data loading
        helper = StreamingFeatureHelper(self._ds, "ward-assign")
        if self._inputs_meta.get("pair_filter"):
            helper.set_pair_filter(self._inputs_meta["pair_filter"])
        scope_filter = {"safe_sequences": self._allowed_safe_sequences} if self._allowed_safe_sequences else None
        manifest = helper.build_manifest(self._inputs, scope_filter=scope_filter)
        if not manifest:
            raise RuntimeError("[ward-assign] No usable inputs found for assignment.")

        # Process sequences one at a time using direct loading
        # (avoids generator pattern which holds extra reference to data)
        keys = list(manifest.keys())
        n_keys = len(keys)
        for i, safe_seq in enumerate(keys):
            X_full, frames, id1_vals, id2_vals, entity_level = helper.load_key_data_with_identity(
                manifest[safe_seq],
                extract_frames=True,
                key=safe_seq,
            )
            if X_full is None:
                continue

            if self._scaler is not None:
                if not hasattr(self._scaler, "transform"):
                    raise ValueError("[ward-assign] scaler object missing transform().")
                X_use = self._scaler.transform(X_full)
                del X_full  # free raw data immediately
            else:
                X_use = X_full
                # Note: X_use IS X_full here, so don't delete

            idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
            labels = self._cluster_ids[idxs.ravel()]
            self._write_sequence_outputs(
                safe_seq,
                labels,
                frames,
                id1_vals=id1_vals,
                id2_vals=id2_vals,
                entity_level=entity_level,
            )

            # Free memory after each sequence
            del X_use, idxs, labels
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % 10 == 0 or i == n_keys - 1:
                print(f"[ward-assign] Processed {i + 1}/{n_keys} sequences", file=sys.stderr)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Outputs are generated during fit(); transform is intentionally skipped.
        return pd.DataFrame(index=[])

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_df.to_parquet(marker_path, index=False)
        self._additional_index_rows.append({
            "group": "",
            "sequence": marker_seq,
            "group_safe": "",
            "sequence_safe": safe_marker_seq,
            "abs_path": self._ds._relative_to_root(marker_path) if self._ds else str(marker_path.resolve()),
            "n_rows": int(len(marker_df)),
        })

        # Labels are already written to disk during fit(); just save model params
        joblib.dump({
            "params": self.params,
            "n_clusters": int(self.params.get("n_clusters", 20)),
            "processed_sequences": list(self._processed_sequences),
        }, path)

    def get_additional_index_rows(self) -> list[dict]:
        return list(self._additional_index_rows)


#### VISUALIZATION
