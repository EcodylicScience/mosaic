"""
GlobalKMeansClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans

from mosaic.core.dataset import register_feature, _resolve_inputs, _dataset_base_dir
from .helpers import (
    StreamingFeatureHelper,
    _parse_scope_filter, _build_sequence_lookup, _resolve_sequence_identity,
    _get_feature_run_root, _load_joblib_artifact, _build_index_row,
)
from mosaic.core.helpers import to_safe_name


@register_feature
class GlobalKMeansClustering:
    """
    Global K-Means that fits **only** on an artifact matrix resolved from the dataset's
    feature index (no frame-stream fitting). This avoids fit/transform shape mismatch.

    params:
      k: int
      random_state: int
      n_init: int|"auto"
      max_iter: int
      artifact: {
        feature: "global-tsne" | "pair-egocentric" | ...   # required
        run_id: str | None                                 # None => latest
        pattern: str                                       # files inside run root
        load: {                                            # loader spec
          kind: "npz"|"parquet",
          key: str               # for npz
          transpose: bool,       # optional
          columns: [str]|None,   # for parquet; if None -> numeric_only
          drop_columns: [str]|None,
          numeric_only: bool,    # default True for parquet
        }
      }

    Optional assign block:
      assign: {
        scaler: {
          feature: <prior feature name>,      # e.g. "global-tsne"
          run_id: str | None,                # None => latest
          pattern: str,                      # e.g. "global_opentsne_embedding.joblib"
          key: str | None,                   # key to extract from joblib, or None for full object
        },
        inputs: [                           # multi-input spec, same as GlobalTSNE.inputs
          {
            feature: <prior feature name>,
            run_id: str | None,
            pattern: str,
            load: {kind, key, transpose, columns, ...}
          },
          ...
        ]
      }
    If assign is specified, after fitting KMeans, clusters are assigned to full-frame features
    using the provided scaler and multi-inputs. Assigned labels per sequence are saved as
    global_kmeans_labels_seq=<safe_seq>.npz in the run folder.

    Notes:
      • The `assign.scaler` entry is optional. If omitted, assignment is performed in the raw feature space.
        In that case, the concatenated input dimensionality must exactly match the dimensionality used to fit K-Means.

    Fit-time artifacts saved in run folder:
      - kmeans.joblib
      - artifact_meta.json (fit provenance + columns + feature_dim)
      - cluster_centers.npy
      - artifact_labels.npz (labels over the artifact matrix, if computed)
      - cluster_sizes.csv (label,count) if labels computed
      - (if assign) global_kmeans_labels_seq=<safe_seq>.npz (per key)
    """

    name: str = "global-kmeans"
    version: str = "0.3"
    output_type: str = "global"
    skip_transform_phase: bool = True

    def __init__(self, params: dict | None = None):
        self.params = {
            "k": 100,
            "random_state": 42,
            "n_init": "auto",
            "max_iter": 300,
            "artifact": {
                "feature": None,
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            # optional: compute labels on the artifact used for fitting
            "label_artifact_points": True,
            # optional: assign block for assigning clusters to full-frame features
            "assign": None,
        }
        if params:
            self.params.update(params)

        self._ds = None
        self._kmeans = None
        self._fit_dim = None              # int
        self._fit_columns = None          # list[str] when parquet
        self._fit_artifact_info = None    # dict saved to JSON
        self._artifact_labels = None      # np.ndarray[int] (optional)
        self._assign_labels = {}          # dict[safe_seq] = labels (int32)
        self._assign_frames: Dict[str, np.ndarray] = {}
        self._assign_id1: Dict[str, np.ndarray] = {}
        self._assign_id2: Dict[str, np.ndarray] = {}
        self._assign_entity_level: Dict[str, str] = {}
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._sequence_lookup_cache: Optional[dict[str, tuple[str, str]]] = None
        self._additional_index_rows: list[dict] = []
        assign_config = self.params.get("assign") or {}
        self._assign_inputs_override = "inputs" in assign_config
        self._assign_inputs_meta: Dict[str, Any] = {}
        self._allowed_safe_sequences: Optional[set[str]] = None

    def set_scope_constraints(self, scope: Optional[dict]) -> None:
        """
        Capture dataset-level sequence filters so assignment can respect them.
        """
        self._allowed_safe_sequences, self._pair_map = _parse_scope_filter(scope)
        self._sequence_lookup_cache = None

    def _get_sequence_lookup(self) -> dict[str, tuple[str, str]]:
        if self._sequence_lookup_cache is not None:
            return self._sequence_lookup_cache
        self._sequence_lookup_cache = _build_sequence_lookup(
            self._ds, self._allowed_safe_sequences
        )
        return self._sequence_lookup_cache

    # Dataset binding
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def loads_own_data(self) -> bool: return True  # Skip run_feature pre-loading; we load from artifacts
    def bind_dataset(self, ds): self._ds = ds

    # ----------------- Fit helpers -----------------
    def _load_npz_matrix(self, files: list[Path], key: str, transpose: bool) -> tuple[np.ndarray, dict]:
        mats = []
        for p in files:
            npz = np.load(p, allow_pickle=True)
            if key not in npz.files:
                # skip silently; some *.npz may not contain the requested key
                continue
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            A = A.T if transpose else A
            mats.append(A)
        if not mats:
            raise FileNotFoundError(f"No NPZ containing key '{key}' among: {[p.name for p in files]}")
        X = np.vstack(mats)
        meta = {"loader_kind": "npz", "key": key, "transpose": bool(transpose)}
        return X, meta

    def _load_parquet_matrix(self, files: list[Path], spec: dict) -> tuple[np.ndarray, dict, list[str]]:
        cols = spec.get("columns")
        drop_cols = set(spec.get("drop_columns", []))
        numeric_only = bool(spec.get("numeric_only", True))

        def load_df(p: Path) -> pd.DataFrame:
            df = pd.read_parquet(p)
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")
            if cols is not None:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return pd.DataFrame()
                df = df[use]
            else:
                if numeric_only:
                    df = df.select_dtypes(include=["number"])
                    # Drop metadata columns that are numeric but not features
                    for mc in ("frame", "time", "id", "id1", "id2", "id_a", "id_b", "id_A", "id_B"):
                        if mc in df.columns:
                            df = df.drop(columns=[mc])
                else:
                    df = df.apply(pd.to_numeric, errors="coerce")
            return df

        dfs = []
        first_cols = None
        for p in files:
            df = load_df(p)
            if df.empty:
                continue
            if first_cols is None:
                first_cols = df.columns.tolist()
            else:
                # align to first set of columns
                df = df.reindex(columns=first_cols)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No Parquet files with usable numeric columns among: {[p.name for p in files]}")

        D = pd.concat(dfs, ignore_index=True)
        A = D.to_numpy(dtype=np.float32, copy=False)
        meta = {
            "loader_kind": "parquet",
            "columns": first_cols,
            "drop_columns": list(drop_cols) if drop_cols else [],
            "numeric_only": numeric_only,
        }
        return A, meta, first_cols

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.get("artifact") or {}
        feature = art.get("feature")
        if not feature:
            raise ValueError("GlobalKMeansClustering: params['artifact']['feature'] is required.")
        resolved_run_id, run_root = _get_feature_run_root(self._ds, feature, art.get("run_id"))
        pattern = art.get("pattern", "*.npz")
        loader  = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {run_root}")

        kind = str(loader.get("kind", "npz")).lower()
        self._fit_columns = None
        if kind == "npz":
            key = loader.get("key")
            if not key:
                raise ValueError("artifact.load.kind='npz' requires 'key'")
            X, meta = self._load_npz_matrix(files, key, bool(loader.get("transpose", False)))
        elif kind == "parquet":
            X, meta, cols = self._load_parquet_matrix(files, loader)
            self._fit_columns = cols
        else:
            raise ValueError(f"Unknown artifact load.kind='{kind}'")

        # for provenance
        self._fit_artifact_info = {
            "feature": feature,
            "run_id": resolved_run_id,
            "run_root": str(run_root),
            "pattern": pattern,
            "loader": meta,
        }
        return X

    # ----------------- Fit / Transform / Save -----------------
    def fit(self, _X_iter_unused: Iterable[pd.DataFrame]) -> None:
        self._assign_labels = {}
        self._assign_frames = {}
        self._assign_id1 = {}
        self._assign_id2 = {}
        self._assign_entity_level = {}
        self._additional_index_rows = []

        # Fit purely from artifact
        X = self._load_artifact_matrix()
        self._fit_dim = int(X.shape[1])

        if X.shape[0] < int(self.params["k"]):
            raise ValueError(f"Not enough samples to fit KMeans: n={X.shape[0]} < k={self.params['k']}")

        self._kmeans = KMeans(
            n_clusters=int(self.params["k"]),
            n_init=self.params.get("n_init", "auto"),
            random_state=self.params.get("random_state", 42),
            max_iter=int(self.params.get("max_iter", 300)),
        ).fit(X)

        # Optional: label the artifact points used for training (nice to have)
        if bool(self.params.get("label_artifact_points", True)):
            self._artifact_labels = self._kmeans.predict(X)

        # Optional: assign clusters to full-frame features using assign block
        assign = self.params.get("assign")
        if assign:
            import gc
            import pyarrow as pa

            # Optional scaler: if provided, standardize before prediction; else predict in raw space.
            scaler = None
            if "scaler" in assign and assign["scaler"]:
                scaler = _load_joblib_artifact(self._ds, assign["scaler"])

            assign_inputset = assign.get("inputset")
            explicit_assign_inputs = assign.get("inputs") if (self._assign_inputs_override or not assign_inputset) else None
            resolved_assign_inputs, assign_inputs_meta = _resolve_inputs(
                self._ds,
                explicit_assign_inputs,
                assign_inputset,
                explicit_override=self._assign_inputs_override,
            )
            self._assign_inputs_meta = assign_inputs_meta

            # Use StreamingFeatureHelper for manifest building and data loading
            helper = StreamingFeatureHelper(self._ds, "global-kmeans")
            if assign_inputs_meta.get("pair_filter"):
                helper.set_pair_filter(assign_inputs_meta["pair_filter"])
            scope_filter = {"safe_sequences": self._allowed_safe_sequences} if self._allowed_safe_sequences else None
            manifest = helper.build_manifest(resolved_assign_inputs, scope_filter=scope_filter)

            # Process each sequence one at a time using direct loading
            # (avoids generator pattern which holds extra reference to data)
            keys = list(manifest.keys())
            n_keys = len(keys)
            for i, key in enumerate(keys):
                X_full, frames, id1_vals, id2_vals, entity_level = helper.load_key_data_with_identity(
                    manifest[key],
                    extract_frames=True,
                    key=key,
                )
                if X_full is None:
                    continue
                D_total = X_full.shape[1]

                if scaler is not None:
                    # Validate scaler dimensionality
                    if not hasattr(scaler, "n_features_in_"):
                        raise ValueError("Scaler object must have n_features_in_ (e.g. sklearn StandardScaler)")
                    if scaler.n_features_in_ != D_total:
                        raise ValueError(
                            f"Scaler expects n_features_in_={getattr(scaler, 'n_features_in_', None)}, "
                            f"got {D_total} columns for key={key}"
                        )
                    X_use = scaler.transform(X_full)
                    del X_full  # free raw data immediately
                else:
                    # No scaler: require that assign inputs match the KMeans fit dimensionality.
                    if self._fit_dim is None:
                        raise RuntimeError("GlobalKMeansClustering internal error: fit_dim is None before assignment.")
                    if D_total != self._fit_dim:
                        raise ValueError(
                            f"Assign-without-scaler requires feature dim {self._fit_dim}, "
                            f"but got {D_total} for key={key}. Provide a scaler or align inputs."
                        )
                    X_use = X_full
                    # Note: X_use IS X_full here, so don't delete

                labels = self._kmeans.predict(X_use)
                self._assign_labels[key] = labels.astype(np.int32)
                self._assign_frames[key] = None if frames is None else np.asarray(frames, dtype=np.int64).ravel()
                if id1_vals is not None:
                    self._assign_id1[key] = np.asarray(id1_vals, dtype=np.float64).ravel()
                if id2_vals is not None:
                    self._assign_id2[key] = np.asarray(id2_vals, dtype=np.float64).ravel()
                self._assign_entity_level[key] = str(entity_level or "global")

                # Free memory after each sequence
                del X_use, labels
                gc.collect()
                pa.default_memory_pool().release_unused()

                if (i + 1) % 10 == 0 or i == n_keys - 1:
                    print(f"[global-kmeans] Processed {i + 1}/{n_keys} sequences", file=sys.stderr)

            if self._fit_artifact_info is None:
                self._fit_artifact_info = {}
            self._fit_artifact_info["assign_inputs_meta"] = assign_inputs_meta

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Assign clusters to an input DataFrame **only if** its columns can be aligned to the
        fitted feature space:
          - If we fitted on parquet with explicit/numeric columns, require those columns.
          - Otherwise, require numeric matrix with matching dimensionality.
        If alignment fails, return an **empty** DataFrame (no error).
        """
        if self._kmeans is None or self._fit_dim is None:
            raise RuntimeError("GlobalKMeansClustering not fitted yet.")

        # Case 1: we know the exact columns from parquet fit
        if self._fit_columns:
            missing = [c for c in self._fit_columns if c not in X.columns]
            if missing:
                return pd.DataFrame(columns=["frame", "cluster"])  # silently skip
            A = X[self._fit_columns].to_numpy(dtype=np.float32, copy=False)
        else:
            # Case 2: generic numeric-only fallback requiring same dimensionality
            num = X.select_dtypes(include=["number"])
            # Drop metadata columns that are numeric but not features
            for mc in ("frame", "time", "id", "id1", "id2", "id_a", "id_b", "id_A", "id_B"):
                if mc in num.columns:
                    num = num.drop(columns=[mc])
            if num.shape[1] != self._fit_dim:
                return pd.DataFrame(columns=["frame", "cluster"])  # silently skip
            A = num.to_numpy(dtype=np.float32, copy=False)

        mask = np.isfinite(A).all(axis=1)
        labels = np.full(A.shape[0], -1, dtype=np.int32)
        if mask.any():
            labels[mask] = self._kmeans.predict(A[mask])

        out = pd.DataFrame({
            "frame": X["frame"].astype(int, errors="ignore") if "frame" in X.columns else np.arange(len(X), dtype=int),
            "cluster": labels,
        })
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        self._additional_index_rows = []

        # 1) Save the fitted model
        joblib.dump({
            "kmeans": self._kmeans,
            "k": int(self.params["k"]),
            "random_state": self.params.get("random_state", 42),
            "n_init": self.params.get("n_init", "auto"),
            "max_iter": int(self.params.get("max_iter", 300)),
            "fit_dim": int(self._fit_dim or 0),
            "fit_columns": self._fit_columns,
            "artifact_info": self._fit_artifact_info,
            "version": self.version,
        }, path)

        # 2) Save human-usable artifacts
        centers = np.asarray(self._kmeans.cluster_centers_, dtype=np.float32)
        np.save(run_root / "cluster_centers.npy", centers)

        # template/artifact labels + counts (if computed)
        if self._artifact_labels is not None:
            np.savez_compressed(run_root / "artifact_labels.npz", labels=self._artifact_labels)
            # counts for quick inspection
            uniq, cnt = np.unique(self._artifact_labels, return_counts=True)
            pd.DataFrame({"cluster": uniq.astype(int), "count": cnt.astype(int)}) \
              .to_csv(run_root / "cluster_sizes.csv", index=False)

        # 3) Save assigned labels per key if assign block was used
        for safe_seq, labels in self._assign_labels.items():
            labels_arr = np.asarray(labels, dtype=np.int32).ravel()
            frames = self._assign_frames.get(safe_seq)
            if frames is not None and len(frames) != len(labels_arr):
                frames = None
            if frames is None:
                frames = np.arange(len(labels_arr), dtype=np.int64)
            id1_vals = self._assign_id1.get(safe_seq)
            id2_vals = self._assign_id2.get(safe_seq)
            if id1_vals is None or len(id1_vals) != len(labels_arr):
                id1_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)
            if id2_vals is None or len(id2_vals) != len(labels_arr):
                id2_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)
            entity_level = self._assign_entity_level.get(safe_seq, "global")
            group, sequence = _resolve_sequence_identity(
                safe_seq, self._pair_map, self._get_sequence_lookup()
            )
            safe_group = to_safe_name(group) if group else ""
            out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
            out_path = run_root / out_name
            df_out = pd.DataFrame({
                "frame": frames.astype(np.int64, copy=False),
                "cluster": labels_arr,
                "id1": pd.array(id1_vals, dtype="Int64"),
                "id2": pd.array(id2_vals, dtype="Int64"),
                "entity_level": np.full(len(labels_arr), entity_level, dtype=object),
                "sequence": sequence,
                "group": group,
            })
            df_out.to_parquet(out_path, index=False)
            self._additional_index_rows.append(
                _build_index_row(safe_seq, group, sequence, out_path, int(len(df_out)),
                                 dataset_root=_dataset_base_dir(self._ds) if self._ds else None)
            )
            fname = f"global_kmeans_labels_seq={safe_seq}.npz"
            np.savez_compressed(
                run_root / fname,
                labels=labels_arr,
                frames=frames,
                id1=id1_vals,
                id2=id2_vals,
                entity_level=np.array([entity_level], dtype=object),
            )

        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_df.to_parquet(marker_path, index=False)
        self._additional_index_rows.append(
            _build_index_row(safe_marker_seq, "", marker_seq, marker_path, int(len(marker_df)),
                             dataset_root=_dataset_base_dir(self._ds) if self._ds else None)
        )

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._kmeans = bundle["kmeans"]
        self._fit_dim = int(bundle.get("fit_dim") or 0)
        self._fit_columns = bundle.get("fit_columns")
        self._fit_artifact_info = bundle.get("artifact_info", {})

    def get_additional_index_rows(self) -> list[dict]:
        return list(self._additional_index_rows)
