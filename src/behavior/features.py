# features.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, Tuple, List
from collections import defaultdict
import hashlib, json, gc, importlib
import numpy as np
import pandas as pd
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import re, sys, fnmatch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openTSNE import TSNEEmbedding, affinity, initialization
from scipy.cluster.hierarchy import linkage as _sch_linkage
from scipy.cluster.hierarchy import fcluster
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
from .helpers import to_safe_name

INHERIT_REGEX = object()

try:
    import pywt
    _PYWT_OK = True
except Exception:
    _PYWT_OK = False

# import the registry + protocol from your dataset module
try:
    from .dataset import (
        register_feature,
        _feature_run_root,
        _feature_index_path,
        _latest_feature_run_root,
        save_inputset,
        _resolve_inputs,
        _model_run_root,
    )
except Exception:
    def register_feature(cls): return cls
    def _latest_feature_run_root(ds, name): raise RuntimeError("Bind dataset first")
    def _feature_run_root(ds, name, run_id): raise RuntimeError("Bind dataset first")
    def _feature_index_path(ds, name): raise RuntimeError("Bind dataset first")
    def save_inputset(*args, **kwargs): raise RuntimeError("Bind dataset first")
    def _resolve_inputs(*args, **kwargs): raise RuntimeError("Bind dataset first")
    def _model_run_root(ds, name, run_id): raise RuntimeError("Bind dataset first")


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _load_array_from_spec(path: Path, load_spec: dict) -> Optional[np.ndarray]:
    kind = str(load_spec.get("kind", "parquet")).lower()
    transpose = bool(load_spec.get("transpose", False))
    if kind == "npz":
        key = load_spec.get("key")
        if not key:
            raise ValueError("load.kind='npz' requires 'key'")
        npz = np.load(path, allow_pickle=True)
        if key not in npz.files:
            return None
        A = np.asarray(npz[key])
        if A.ndim == 1:
            A = A[None, :]
    elif kind == "parquet":
        df = pd.read_parquet(path)
        drop_cols = load_spec.get("drop_columns")
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        cols = load_spec.get("columns")
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        elif load_spec.get("numeric_only", True):
            df = df.select_dtypes(include=[np.number])
        else:
            df = df.apply(pd.to_numeric, errors="coerce")
        A = df.to_numpy(dtype=np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported load.kind='{kind}'")

    if A.size == 0:
        return None
    if transpose:
        A = A.T
    if A.ndim == 1:
        A = A[None, :]
    return A.astype(np.float32, copy=False)


def _collect_sequence_blocks(ds, specs: list[dict]) -> dict[str, np.ndarray]:
    """
    Load per-sequence stacked matrices for a given list of input specs.
    """
    per_seq: dict[str, list[np.ndarray]] = defaultdict(list)
    for spec in specs:
        feat_name = spec["feature"]
        run_id = spec.get("run_id")
        if run_id is None:
            run_id, run_root = _latest_feature_run_root(ds, feat_name)
        else:
            run_root = _feature_run_root(ds, feat_name, run_id)
        pattern = spec.get("pattern", "*.parquet")
        load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            continue
        seq_map = _build_path_sequence_map(ds, feat_name, run_id)
        for fp in files:
            arr = _load_array_from_spec(fp, load_spec)
            if arr is None:
                continue
            safe_seq = seq_map.get(fp.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(fp.stem)
            per_seq[safe_seq].append(arr)

    blocks: dict[str, np.ndarray] = {}
    for safe_seq, mats in per_seq.items():
        mats = [m for m in mats if m.size]
        if not mats:
            continue
        T_min = min(m.shape[0] for m in mats)
        if T_min <= 0:
            continue
        mats = [m[:T_min] for m in mats]
        blocks[safe_seq] = np.hstack(mats)
    return blocks

def _build_path_sequence_map(ds, feature_name: str, run_id: str | None) -> dict[Path, str]:
    """
    Returns {abs_path -> sequence_safe} for a given feature/run_id using the dataset's feature index.
    """
    mapping: dict[Path, str] = {}
    if ds is None or not feature_name or run_id is None:
        return mapping

    try:
        idx_path = _feature_index_path(ds, feature_name)
    except Exception:
        return mapping
    if not idx_path.exists():
        return mapping

    try:
        df = pd.read_csv(idx_path)
    except Exception:
        return mapping

    run_id_str = str(run_id)
    df = df[df["run_id"].astype(str) == run_id_str]
    if df.empty:
        return mapping

    if "sequence_safe" not in df.columns:
        df["sequence_safe"] = df["sequence"].fillna("").apply(
            lambda v: to_safe_name(str(v)) if str(v).strip() else ""
        )

    for _, row in df.iterrows():
        abs_raw = row.get("abs_path")
        if not isinstance(abs_raw, str) or not abs_raw:
            continue
        try:
            abs_path = Path(abs_raw).resolve()
        except Exception:
            abs_path = Path(abs_raw)
        seq_val = (
            row.get("sequence_safe")
            or row.get("sequence")
            or row.get("group_safe")
            or row.get("group")
            or ""
        )
        seq_val = str(seq_val).strip()
        if not seq_val:
            seq_val = to_safe_name(Path(abs_raw).stem)
        mapping[abs_path] = seq_val
    return mapping

@register_feature
class TemporalStackingFeature:
    """
    Build temporal context windows over an inputset of per-sequence feature files by stacking
    Gaussian-smoothed frames and optional pooled statistics, then saving the result as a new
    feature under features/temporal-stack__from__<inputset>/run_id.
    """

    name = "temporal-stack"
    version = "0.1"
    parallelizable = True

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "inputset": None,
            "inputs": [],
            "half": 60,
            "skip": 5,
            "use_temporal_stack": True,
            "sigma_stack": 30.0,
            "add_pool": True,
            "pool_stats": ("mean",),
            "sigma_pool": 30.0,
            "fps": 30.0,
            "win_sec": 0.5,
            "group_col": "group",
            "sequence_col": "sequence",
            "write_chunk_size": 1000,
            "stack_chunk_size": 1000,
        }
        self.params = dict(defaults)
        if params:
            for k, v in params.items():
                self.params[k] = v
        pool_stats = self.params.get("pool_stats") or []
        if isinstance(pool_stats, str):
            pool_stats = [pool_stats]
        self.params["pool_stats"] = [str(stat).lower() for stat in pool_stats]

        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

        self._ds = None
        self._inputs: list[dict] = []
        self._inputs_meta: dict = {}
        self._resolved_inputs: list[dict] = []
        self._input_cache_ready = False
        self._scope_filter: Optional[dict] = None
        self._allowed_safe_sequences: Optional[set[str]] = None

    def bind_dataset(self, ds):
        self._ds = ds
        explicit_inputs = self.params.get("inputs") or []
        inputset_name = self.params.get("inputset")
        if not explicit_inputs and not inputset_name:
            raise ValueError("temporal-stack: provide params['inputset'] or explicit params['inputs'].")
        explicit_override = bool(explicit_inputs)
        self._inputs, self._inputs_meta = _resolve_inputs(
            ds,
            explicit_inputs,
            inputset_name,
            explicit_override=explicit_override,
        )
        self._resolved_inputs = []
        self._input_cache_ready = False

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}
        safe_sequences = scope.get("safe_sequences") if scope else None
        if safe_sequences:
            self._allowed_safe_sequences = {str(s) for s in safe_sequences}
        else:
            self._allowed_safe_sequences = None

    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def fit(self, X: Iterable[pd.DataFrame]): pass
    def save_model(self, path: Path) -> None: raise NotImplementedError("stateless feature")

    # ------------- Core logic -------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._ds is None:
            raise RuntimeError("temporal-stack: dataset not bound.")
        self._ensure_inputs_ready()

        seq_col = self.params.get("sequence_col", "sequence")
        group_col = self.params.get("group_col", "group")
        sequence = str(df[seq_col].iloc[0]) if seq_col in df.columns and not df.empty else None
        group = str(df[group_col].iloc[0]) if group_col in df.columns and not df.empty else None
        if not sequence:
            raise ValueError("temporal-stack: unable to infer sequence from dataframe; ensure 'sequence' column exists.")

        safe_seq = to_safe_name(sequence)
        # If scope_filter provides canonical mapping, prefer it
        if self._scope_filter:
            pair_map = self._scope_filter.get("pair_safe_map") or {}
            safe_seq = pair_map.get((group, sequence), safe_seq)
        if self._allowed_safe_sequences and safe_seq not in self._allowed_safe_sequences:
            raise ValueError(f"temporal-stack: sequence '{sequence}' not present in resolved inputset scope.")

        base_matrix, base_names = self._load_sequence_matrix(safe_seq)
        if base_matrix is None or base_matrix.size == 0:
            raise ValueError(f"temporal-stack: missing inputs for sequence '{sequence}'.")

        chunk_iter, stacked_names, total_rows = self._chunked_temporal_features(base_matrix, base_names)
        payload = {
            "parquet_chunk_iter": chunk_iter,
            "columns": stacked_names,
            "sequence": sequence,
            "group": group,
            "total_rows": total_rows,
        }
        return payload

    # ------------- Helpers -------------
    def _ensure_inputs_ready(self) -> None:
        if self._input_cache_ready:
            return
        resolved = []
        allowed = self._allowed_safe_sequences
        for spec in self._inputs:
            feat_name = spec.get("feature")
            if not feat_name:
                continue
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, _ = _latest_feature_run_root(self._ds, feat_name)
            else:
                run_id = str(run_id)
            mapping = self._build_sequence_mapping(feat_name, run_id, allowed)
            if not mapping:
                continue
            resolved.append({
                "feature": feat_name,
                "run_id": run_id,
                "load": spec.get("load") or {"kind": "parquet", "numeric_only": True},
                "name": spec.get("name") or feat_name,
                "mapping": mapping,
            })
        if not resolved:
            raise RuntimeError("temporal-stack: no overlapping inputs found for the requested scope.")
        self._resolved_inputs = resolved
        self._input_cache_ready = True

    def _build_sequence_mapping(self, feature_name: str, run_id: str, allowed: Optional[set[str]]) -> dict[str, Path]:
        idx_path = _feature_index_path(self._ds, feature_name)
        if not idx_path.exists():
            raise FileNotFoundError(f"temporal-stack: missing index for feature '{feature_name}' -> {idx_path}")
        df = pd.read_csv(idx_path)
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            raise ValueError(f"temporal-stack: feature '{feature_name}' run_id='{run_id}' has no rows.")
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].fillna("").apply(lambda v: to_safe_name(v) if v else "")
        df = df[df["sequence_safe"].astype(str).str.strip() != ""]
        if allowed:
            df = df[df["sequence_safe"].isin(allowed)]
        mapping: dict[str, Path] = {}
        for _, row in df.iterrows():
            seq_safe = str(row["sequence_safe"])
            abs_path = row.get("abs_path")
            if not seq_safe or not isinstance(abs_path, str) or not abs_path:
                continue
            remapped = self._ds.remap_path(abs_path) if hasattr(self._ds, "remap_path") else Path(abs_path)
            mapping[seq_safe] = remapped
        return mapping

    def _load_sequence_matrix(self, safe_seq: str) -> tuple[Optional[np.ndarray], list[str]]:
        mats: list[np.ndarray] = []
        col_names: list[str] = []
        for entry in self._resolved_inputs:
            path = entry["mapping"].get(safe_seq)
            if not path or not path.exists():
                return None, []
            arr = _load_array_from_spec(path, entry["load"])
            if arr is None or arr.size == 0:
                return None, []
            mats.append(arr)
            prefix = entry["name"]
            col_names.extend([f"{prefix}__f{idx:04d}" for idx in range(arr.shape[1])])
        if not mats:
            return None, []
        min_len = min(m.shape[0] for m in mats)
        if min_len == 0:
            return None, []
        mats = [m[:min_len] for m in mats]
        base = np.hstack(mats).astype(np.float32, copy=False)
        return base, col_names

    def _chunked_temporal_features(self, base: np.ndarray, base_names: list[str]) -> tuple[Iterable[tuple[int, np.ndarray]], list[str], int]:
        total_rows = base.shape[0]
        stack_chunk_size = max(1, int(self.params.get("stack_chunk_size", 20000)))
        use_stack = self.params.get("use_temporal_stack", True)
        add_pool = self.params.get("add_pool", True)

        # Precompute stacking metadata
        half = max(0, int(self.params.get("half", 0)))
        step = max(1, int(self.params.get("skip", 1)))
        sigma_stack = max(0.0, float(self.params.get("sigma_stack", 0.0)))
        offsets = list(range(-half, half + 1, step))
        stack_names = []
        if use_stack:
            for off in offsets:
                stack_names.extend([f"{name}__t{off:+03d}" for name in base_names])
        else:
            stack_names = list(base_names)

        # Precompute smoothing arrays
        smoothed = base
        if sigma_stack > 0 and use_stack:
            smoothed = gaussian_filter1d(base, sigma=sigma_stack, axis=0, mode="nearest")
        padded = None
        if use_stack and half > 0:
            padded = np.pad(smoothed, ((half, half), (0, 0)), mode="edge")
        idx = np.arange(smoothed.shape[0]) + (half if use_stack else 0)

        # Precompute pooled stats
        pooled_names = []
        pooled_arrays: list[np.ndarray] = []
        if add_pool:
            arrays, pooled_names = self._pooled_stats_arrays(base, base_names)
            if arrays:
                pooled_arrays = [arr.astype(np.float32, copy=False) for arr in arrays]

        all_names = list(stack_names)
        if pooled_names:
            all_names.extend(pooled_names)

        def chunk_iterator():
            nonlocal base, smoothed, padded, pooled_arrays
            try:
                for start in range(0, total_rows, stack_chunk_size):
                    end = min(start + stack_chunk_size, total_rows)
                    parts = []
                    if use_stack:
                        chunk_stack = []
                        for off in offsets:
                            if half > 0 and padded is not None:
                                chunk_stack.append(padded[idx[start:end] + off])
                            else:
                                chunk_stack.append(smoothed[start:end])
                        parts.append(np.hstack(chunk_stack))
                    else:
                        parts.append(base[start:end])

                    if pooled_arrays:
                        pool_parts = [arr[start:end] for arr in pooled_arrays]
                        if pool_parts:
                            parts.append(np.hstack(pool_parts))

                    combined = np.hstack(parts).astype(np.float32, copy=False)
                    yield start, combined
            finally:
                try:
                    del base
                except Exception:
                    pass
                if use_stack and sigma_stack > 0:
                    try:
                        del smoothed
                    except Exception:
                        pass
                if padded is not None:
                    try:
                        del padded
                    except Exception:
                        pass
                if pooled_arrays:
                    try:
                        del pooled_arrays
                    except Exception:
                        pass
                gc.collect()

        return chunk_iterator(), all_names, total_rows

    def _temporal_stack(self, base: np.ndarray, base_names: list[str]) -> tuple[np.ndarray, list[str]]:
        half = max(0, int(self.params.get("half", 0)))
        step = max(1, int(self.params.get("skip", 1)))
        sigma = max(0.0, float(self.params.get("sigma_stack", 0.0)))
        if sigma > 0:
            smoothed = gaussian_filter1d(base, sigma=sigma, axis=0, mode="nearest")
        else:
            smoothed = base
        if half == 0:
            return smoothed.astype(np.float32, copy=False), [f"{name}__t+0" for name in base_names]

        padded = np.pad(smoothed, ((half, half), (0, 0)), mode="edge")
        idx = np.arange(smoothed.shape[0]) + half
        offsets = list(range(-half, half + 1, step))
        stacks = []
        names = []
        for off in offsets:
            stacks.append(padded[idx + off])
            names.extend([f"{name}__t{off:+03d}" for name in base_names])
        stacked = np.hstack(stacks).astype(np.float32, copy=False)
        return stacked, names

    def _pooled_stats_arrays(self, base: np.ndarray, base_names: list[str]) -> tuple[list[np.ndarray], list[str]]:
        stats = self.params.get("pool_stats") or []
        if not stats:
            return [], []
        sigma = max(0.0, float(self.params.get("sigma_pool", 0.0)))
        if sigma <= 0:
            sigma = max(1.0, float(self.params.get("win_sec", 0.5)) * float(self.params.get("fps", 30.0)) / 6.0)
        win_frames = max(1, int(round(float(self.params.get("win_sec", 0.5)) * float(self.params.get("fps", 30.0)))))
        truncate = max(1.0, win_frames / (2.0 * sigma)) if sigma > 0 else 4.0
        mean_vals = gaussian_filter1d(base, sigma=sigma, axis=0, mode="nearest", truncate=truncate)
        outputs = []
        names = []
        if "mean" in stats:
            outputs.append(mean_vals)
            names.extend([f"{name}__pool_mean" for name in base_names])
        if "std" in stats or "variance" in stats:
            second = gaussian_filter1d(base ** 2, sigma=sigma, axis=0, mode="nearest", truncate=truncate)
            var = np.clip(second - mean_vals ** 2, 0.0, None)
            if "variance" in stats:
                outputs.append(var)
                names.extend([f"{name}__pool_var" for name in base_names])
            if "std" in stats:
                outputs.append(np.sqrt(var))
                names.extend([f"{name}__pool_std" for name in base_names])
        return outputs, names

    def _pooled_stats(self, base: np.ndarray, base_names: list[str]) -> tuple[Optional[np.ndarray], list[str]]:
        arrays, names = self._pooled_stats_arrays(base, base_names)
        if not arrays:
            return None, []
        pooled = np.hstack([arr.astype(np.float32, copy=False) for arr in arrays])
        return pooled, names


@register_feature
class ModelPredictFeature:
    """
    Generic wrapper that loads a trained model run and applies it over per-sequence feature tables.
    """

    name = "model-predict"
    version = "0.1"
    parallelizable = True

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "model_class": None,
            "model_params": None,
            "model_run_id": None,
            "model_name": None,
            "output_feature_name": None,
        }
        self.params = dict(defaults)
        if params:
            self.params.update(params)
        self._ds = None
        self._model = None
        self._model_name: Optional[str] = None
        self._model_run_id: Optional[str] = None
        self.storage_feature_name = self.params.get("output_feature_name") or self.name
        self.storage_use_input_suffix = True
        self._input_signature: Optional[dict] = None

    def bind_dataset(self, ds):
        self._ds = ds
        model_class_path = self.params.get("model_class")
        if not model_class_path:
            raise ValueError("ModelPredictFeature params must include 'model_class'.")
        module_path, class_name = model_class_path.rsplit(".", 1)
        ModelCls = getattr(importlib.import_module(module_path), class_name)
        model_kwargs = self.params.get("model_params")
        self._model = ModelCls(model_kwargs) if model_kwargs else ModelCls()
        if hasattr(self._model, "bind_dataset"):
            self._model.bind_dataset(ds)
        run_id = str(self.params.get("model_run_id") or "").strip()
        if not run_id:
            raise ValueError("ModelPredictFeature params must include 'model_run_id'.")
        storage_model_name = self.params.get("model_name") or getattr(
            self._model, "storage_model_name", getattr(self._model, "name", None)
        )
        if not storage_model_name:
            raise ValueError("Model must define 'name' or params['model_name'].")
        run_root = _model_run_root(ds, storage_model_name, run_id)
        if not run_root.exists():
            raise FileNotFoundError(f"Model artifacts not found: {run_root}")
        if not hasattr(self._model, "load_trained_model"):
            raise RuntimeError(f"Model '{model_class_path}' lacks load_trained_model().")
        self._model.load_trained_model(run_root)
        self._model_name = storage_model_name
        self._model_run_id = run_id
        output_name = self.params.get("output_feature_name") or f"{storage_model_name}-pred"
        self.storage_feature_name = output_name
        self.storage_use_input_suffix = True
        sig_fn = getattr(self._model, "get_prediction_input_signature", None)
        if callable(sig_fn):
            self._input_signature = sig_fn()

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X: Iterable[pd.DataFrame]) -> None:
        return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("ModelPredictFeature has no model loaded; call bind_dataset first.")
        if df is None or df.empty:
            return pd.DataFrame()
        sequence = str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
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

    def default_input_signature(self) -> Optional[dict]:
        """
        Returns the input specification (input_kind, feature/inputset, resolved run_ids)
        captured when the model was trained, if available.
        """
        if self._input_signature is None:
            return None
        return json.loads(json.dumps(self._input_signature))

# === Orientation-aware features ===# === Orientation-aware features ===


@register_feature
class BodyScaleFeature:
    """
    Per-frame body scale: median intra-animal pose distance.

    Outputs per sequence parquet with columns: frame, id, scale, sequence, group.
    Intended to be averaged later (per sequence or dataset) to derive a single
    normalization constant for downstream orientation features.
    """

    name = "body-scale"
    version = "0.1"
    parallelizable = True

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self._ds = None

    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def fit(self, X: Iterable[pd.DataFrame]):
        return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "frame" not in df.columns or "id" not in df.columns:
            return pd.DataFrame()
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else ""
        sequence = str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
        rows = []
        for frame_val, g in df.groupby("frame", sort=True):
            for id_val, sub in g.groupby("id"):
                pts = []
                for x_col, y_col in pose_pairs:
                    x = sub.iloc[0].get(x_col)
                    y = sub.iloc[0].get(y_col)
                    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                        continue
                    pts.append((float(x), float(y)))
                if len(pts) < 2:
                    continue
                arr = np.asarray(pts, dtype=float)
                dists = np.sqrt(((arr[:, None, :] - arr[None, :, :]) ** 2).sum(axis=2))
                dists = dists[np.triu_indices_from(dists, k=1)]
                if dists.size == 0:
                    continue
                med = float(np.median(dists))
                rows.append({
                    "frame": int(frame_val),
                    "id": id_val,
                    "scale": med,
                    "sequence": sequence,
                    "group": group,
                })
        return pd.DataFrame(rows)


@register_feature
class OrientationRelativeFeature:
    """
    Orientation-aware relative features between animal pairs, order-agnostic to pose points.

    For each frame and ordered pair (id_a -> id_b):
      - Express B in A's body frame (using heading angle and global scale).
      - Emit signed centroid deltas, heading difference, quantiles over B's points
        in A's frame, and nearest-k distances.
    """

    name = "orientation-rel"
    version = "0.1"
    parallelizable = True

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "scale_feature": "body-scale",
            "scale_run_id": None,
            "nearest_k": 3,
            "quantiles": [0.25, 0.5, 0.75],
        }
        self.params = {**defaults, **(params or {})}
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self._ds = None
        self._scale_lookup: dict[str, float] = {}

    def bind_dataset(self, ds):
        self._ds = ds
        self._load_scales()

    def _load_scales(self):
        self._scale_lookup = {}
        if self._ds is None:
            return
        feat = self.params.get("scale_feature", "body-scale")
        run_id = self.params.get("scale_run_id")
        if run_id is None:
            try:
                run_id, _ = _latest_feature_run_root(self._ds, feat)
            except Exception:
                return
        idx_path = _feature_index_path(self._ds, feat)
        if not idx_path.exists():
            return
        df_idx = pd.read_csv(idx_path)
        df_idx = df_idx[df_idx["run_id"].astype(str) == str(run_id)]
        if df_idx.empty:
            return
        for _, row in df_idx.iterrows():
            seq_safe = row.get("sequence_safe") or to_safe_name(row.get("sequence", ""))
            abs_path = row.get("abs_path")
            if not abs_path:
                continue
            try:
                p = Path(abs_path)
                if hasattr(self._ds, "remap_path"):
                    p = self._ds.remap_path(p)
                df_scale = pd.read_parquet(p)
                if "scale" not in df_scale.columns:
                    continue
                mean_scale = float(df_scale["scale"].dropna().mean())
                if np.isfinite(mean_scale) and mean_scale > 0:
                    self._scale_lookup[seq_safe] = mean_scale
            except Exception:
                continue

    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def fit(self, X: Iterable[pd.DataFrame]):
        return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "frame" not in df.columns or "id" not in df.columns or "angle" not in df.columns:
            return pd.DataFrame()
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else ""
        sequence = str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
        seq_safe = to_safe_name(sequence)
        global_scale = self._scale_lookup.get(seq_safe, None)
        quantiles = self.params.get("quantiles", [0.25, 0.5, 0.75])
        nearest_k = int(self.params.get("nearest_k", 3))
        rows = []

        grouped = df.groupby(["frame", "id"], sort=True)
        pose_cache: dict[tuple[int, Any], dict] = {}
        for (frame_val, id_val), sub in grouped:
            pts = []
            for x_col, y_col in pose_pairs:
                x = sub.iloc[0].get(x_col)
                y = sub.iloc[0].get(y_col)
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    continue
                pts.append((float(x), float(y)))
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=float)
            centroid = arr.mean(axis=0)
            angle = float(sub.iloc[0].get("angle", 0.0))
            pose_cache[(int(frame_val), id_val)] = {
                "pts": arr,
                "centroid": centroid,
                "angle": angle,
            }

        frames = sorted({f for f, _ in pose_cache.keys()})
        for f in frames:
            ids_here = [i for (frame_val, i) in pose_cache.keys() if frame_val == f]
            if len(ids_here) < 2:
                continue
            for id_a in ids_here:
                for id_b in ids_here:
                    if id_a == id_b:
                        continue
                    A = pose_cache.get((f, id_a))
                    B = pose_cache.get((f, id_b))
                    if not A or not B:
                        continue
                    pts_B = B["pts"]
                    centroid_B = B["centroid"]
                    angle_A = A["angle"]
                    angle_B = B["angle"]
                    centroid_A = A["centroid"]
                    scale = global_scale
                    if scale is None or not np.isfinite(scale) or scale <= 0:
                        scale = self._local_scale(A["pts"])
                    if scale is None or scale <= 0:
                        continue
                    delta = pts_B - centroid_A
                    rot = self._rotation(-angle_A)
                    rel = (rot @ delta.T).T / scale
                    rel_centroid = (rot @ (centroid_B - centroid_A)) / scale
                    dtheta = self._wrap_angle(angle_B - angle_A)

                    dists = np.sqrt((rel ** 2).sum(axis=1))
                    if dists.size == 0:
                        continue
                    q_vals = np.quantile(dists, quantiles).tolist() if quantiles else []
                    dmin = float(np.min(dists))
                    dmax = float(np.max(dists))
                    dmed = float(np.median(dists))
                    d_near = np.sort(dists)[:max(0, nearest_k)].tolist()
                    d_near += [np.nan] * (max(0, nearest_k) - len(d_near))

                    x_vals = rel[:, 0]
                    y_vals = rel[:, 1]
                    feats = {
                        "frame": int(f),
                        "id_a": id_a,
                        "id_b": id_b,
                        "sequence": sequence,
                        "group": group,
                        "dx": float(rel_centroid[0]),
                        "dy": float(rel_centroid[1]),
                        "dtheta": float(dtheta),
                        "dist_min": dmin,
                        "dist_median": dmed,
                        "dist_max": dmax,
                    }
                    for idx, qv in enumerate(q_vals):
                        feats[f"dist_q{int(quantiles[idx]*100):02d}"] = float(qv)
                    feats["x_min"] = float(np.nanmin(x_vals))
                    feats["x_median"] = float(np.nanmedian(x_vals))
                    feats["x_max"] = float(np.nanmax(x_vals))
                    feats["y_min"] = float(np.nanmin(y_vals))
                    feats["y_median"] = float(np.nanmedian(y_vals))
                    feats["y_max"] = float(np.nanmax(y_vals))
                    for i, val in enumerate(d_near):
                        feats[f"near_{i}"] = float(val) if np.isfinite(val) else np.nan
                    rows.append(feats)

        return pd.DataFrame(rows)

    def _rotation(self, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _wrap_angle(self, a: float) -> float:
        a = (a + np.pi) % (2 * np.pi) - np.pi
        return a

    def _local_scale(self, pts: np.ndarray) -> Optional[float]:
        if pts is None or pts.shape[0] < 2:
            return None
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        d = d[np.triu_indices_from(d, k=1)]
        if d.size == 0:
            return None
        val = float(np.median(d))
        return val if np.isfinite(val) and val > 0 else None

    def _pose_to_points(self, row_vals: np.ndarray) -> np.ndarray:
        N = int(self.params["pose_n"])
        xs = row_vals[:N]; ys = row_vals[N:]
        return np.stack([xs, ys], axis=1)  # (N,2)

    def _intra_lower_tri(self, pts: np.ndarray) -> np.ndarray:
        dif = pts[self._tri_i] - pts[self._tri_j]
        return np.sqrt((dif ** 2).sum(axis=1))  # (n_intra,)

    def _inter_all(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        dif = A[:, None, :] - B[None, :, :]     # (N,N,2)
        d = np.sqrt((dif ** 2).sum(axis=2))     # (N,N)
        return d.ravel()                        # (N*N,)

    def _build_pair_feat(self, rowA: np.ndarray, rowB: np.ndarray) -> np.ndarray:
        parts = []
        A = self._pose_to_points(rowA)
        B = self._pose_to_points(rowB)
        if self.params["include_intra_A"]:
            parts.append(self._intra_lower_tri(A))
        if self.params["include_intra_B"]:
            parts.append(self._intra_lower_tri(B))
        if self.params["include_inter"]:
            parts.append(self._inter_all(A, B))
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)

    def _feature_batches(self, df: pd.DataFrame, for_fit: bool) -> Iterable[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
        """
        Yield (X_batch, meta_frames, persp_array) where:
          - X_batch shape (B, F)
          - meta_frames: dict with possible 'frame' and 'time' arrays (aligned with B)
          - persp_array: (B,) of 0/1 (A→B or B→A) if duplicate_perspective=True, else all zeros.
        """
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)

        df_small, pairs = self._prep_pairs(df)
        bs = int(self.params["batch_size"])
        dup = bool(self.params["duplicate_perspective"])

        # build an iterator over all aligned A/B rows per sequence
        for seq, idA, idB in pairs:
            gseq = df_small[df_small[self.params["seq_col"]] == seq]
            A = gseq[gseq[self.params["id_col"]] == idA][[order_col] + pose_cols].copy()
            B = gseq[gseq[self.params["id_col"]] == idB][[order_col] + pose_cols].copy()
            A = A.sort_values(order_col); B = B.sort_values(order_col)
            # inner-join on the order column (frame/time)
            AB = A.merge(B, on=order_col, suffixes=("_A", "_B"))
            if AB.empty:
                continue

            # slice into batches
            n = len(AB)
            for i in range(0, n, bs):
                j = min(i + bs, n)
                chunk = AB.iloc[i:j]
                # build features for A->B
                XA = chunk[[c + "_A" for c in pose_cols]].to_numpy(dtype=float)
                XB = chunk[[c + "_B" for c in pose_cols]].to_numpy(dtype=float)
                feats = [self._build_pair_feat(a, b) for a, b in zip(XA, XB)]
                X = np.vstack(feats).astype(np.float32, copy=False)

                persp = np.zeros(X.shape[0], dtype=np.int8)
                frames_meta: Dict[str, np.ndarray] = {}
                if "frame" in df.columns:
                    frames_meta["frame"] = chunk[order_col].to_numpy()
                if "time" in df.columns and order_col != "time":
                    # optional time passthrough if present in df; we cannot join time unless it's the order key
                    pass

                if dup:
                    # add B->A echoes
                    feats2 = [self._build_pair_feat(b, a) for a, b in zip(XA, XB)]
                    X2 = np.vstack(feats2).astype(np.float32, copy=False)
                    X = np.vstack([X, X2])
                    persp = np.concatenate([persp, np.ones(X2.shape[0], dtype=np.int8)], axis=0)
                    if "frame" in frames_meta:
                        frames_meta["frame"] = np.concatenate([frames_meta["frame"], frames_meta["frame"]], axis=0)

                # first batch determines feat_len sanity
                if self._feat_len is not None and X.shape[1] != self._feat_len:
                    raise ValueError(f"Feature length mismatch: got {X.shape[1]}, expected {self._feat_len}")

                yield X, frames_meta, persp

@register_feature
class PairEgocentricFeatures:
    """
    'pair-egocentric' — per-sequence egocentric + kinematic features for dyads.
    Produces a row-wise DataFrame with columns:
      - frame (if available) or time passthrough (only if it's the order col)
      - perspective: 0 for A→B, 1 for B→A
      - feature columns (e.g., A_speed, AB_dx_egoA, ...)
      - (optionally) group/sequence if present in df, for convenience

    This feature is *stateless* (no fitting). It infers per-sequence dyads by taking
    the first two IDs present in each sequence, cleans/interpolates pose per animal,
    inner-joins by the chosen order column, and computes A→B and B→A features.
    """

    name    = "pair-egocentric"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        # pose / columns
        pose_n=7,
        x_prefix="poseX", y_prefix="poseY",   # TRex-ish
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),

        # required anatomical indices (must be provided by user if different)
        neck_idx=None,           # REQUIRED (int) unless your skeleton matches defaults
        tail_base_idx=None,      # REQUIRED (int) unless your skeleton matches defaults
        center_mode="mean",      # "mean" or an int landmark index

        # sampling / smoothing
        fps_default=30.0,
        smooth_win=0,            # 0 disables box smoothing before differencing

        # cleaning / interpolation per animal
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        # Enforce required indices if user didn't pass them
        if self.params["neck_idx"] is None:
            # sensible default matching your earlier snippet
            self.params["neck_idx"] = 3
        if self.params["tail_base_idx"] is None:
            self.params["tail_base_idx"] = 6

        self._tri_ready = False  # not used, but kept for symmetry with other feature

    # ------------- Feature protocol -------------
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def finalize_fit(self) -> None: pass
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return
    def partial_fit(self, df: pd.DataFrame) -> None: return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)
        p = self.params

        need = [p["id_col"], p["seq_col"], order_col] + pose_cols
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[pair-egocentric] Missing cols: {missing}")

        # Clean per-animal, per-sequence (future-proof re: pandas groupby.apply)
        df_small = df[need].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        group_cols = [p["seq_col"], p["id_col"]]

        def wrapped_func(g):
            result = self._clean_one_animal(g, pose_cols, order_col)
            # reattach group key(s)
            if isinstance(g.name, tuple):
                for col, val in zip(group_cols, g.name):
                    result[col] = val
            else:
                result[group_cols[0]] = g.name
            return result

        df_small = (
            df_small
            .groupby(group_cols, group_keys=False)
            .apply(wrapped_func, include_groups=False)
        )

        # Build dyads (first two IDs per sequence)
        pairs = []
        for seq, gseq in df_small.groupby(p["seq_col"]):
            ids = sorted(gseq[p["id_col"]].unique())
            if len(ids) >= 2:
                pairs.append((seq, ids[0], ids[1]))

        if not pairs:
            raise ValueError("[pair-egocentric] No sequence with at least two IDs found.")

        out_frames: List[pd.DataFrame] = []
        for seq, idA, idB in pairs:
            gseq = df_small[df_small[p["seq_col"]] == seq]
            A = gseq[gseq[p["id_col"]] == idA][[order_col] + pose_cols].copy()
            B = gseq[gseq[p["id_col"]] == idB][[order_col] + pose_cols].copy()
            if A.empty or B.empty:
                continue

            A = A.sort_values(order_col).rename(columns={order_col: "frame"})
            B = B.sort_values(order_col).rename(columns={order_col: "frame"})
            j = A.merge(B, on="frame", suffixes=("_A", "_B"))
            if j.empty:
                continue

            # fps heuristic: prefer df['fps'] if present and constant; else default
            fps = float(p["fps_default"])
            if "fps" in df.columns:
                try:
                    c = df["fps"].dropna().unique()
                    if len(c) == 1:
                        fps = float(c[0])
                except Exception:
                    pass

            frames, AtoB, BtoA, names = self._build_ego_block_for_joined(j, fps, pose_cols)

            # produce row-wise DataFrames
            dfA = pd.DataFrame(AtoB.T, columns=names)
            dfA["frame"] = frames
            dfA["perspective"] = 0

            dfB = pd.DataFrame(BtoA.T, columns=names)
            dfB["frame"] = frames
            dfB["perspective"] = 1

            # optional pass-through for convenience (constant per call)
            for col in (p["seq_col"], p["group_col"]):
                if col in df.columns:
                    dfA[col] = df[col].iloc[0]
                    dfB[col] = df[col].iloc[0]

            out_frames.extend([dfA, dfB])

        if not out_frames:
            return pd.DataFrame(columns=["perspective", "frame"])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["perspective", "frame"]).reset_index(drop=True)
        return out

    # ------------- Internals -------------
    def _column_names(self) -> Tuple[List[str], List[str]]:
        N = int(self.params["pose_n"])
        xs = [f"{self.params['x_prefix']}{i}" for i in range(N)]
        ys = [f"{self.params['y_prefix']}{i}" for i in range(N)]
        return xs, ys

    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _clean_one_animal(self, g: pd.DataFrame, pose_cols: List[str], order_col: str) -> pd.DataFrame:
        p = self.params
        g = g.sort_values(order_col).copy()
        g = g.set_index(order_col)
        g[pose_cols] = g[pose_cols].replace([np.inf, -np.inf], np.nan)
        g[pose_cols] = g[pose_cols].interpolate(
            method="linear", limit=int(p["linear_interp_limit"]), limit_direction="both"
        )
        g[pose_cols] = g[pose_cols].ffill(limit=int(p["edge_fill_limit"]))
        g[pose_cols] = g[pose_cols].bfill(limit=int(p["edge_fill_limit"]))
        miss_frac = g[pose_cols].isna().mean(axis=1)
        g = g.loc[miss_frac <= float(p["max_missing_fraction"])].copy()
        if g[pose_cols].isna().any().any():
            med = g[pose_cols].median()
            g[pose_cols] = g[pose_cols].fillna(med)
        g = g.reset_index()
        return g

    # --- math helpers ---
    def _smooth_1d(self, x: np.ndarray, win: int) -> np.ndarray:
        if win is None or win <= 1:
            return x
        pad = win // 2
        xp = np.pad(x, pad_width=pad, mode="reflect")
        ker = np.ones(win, dtype=float) / float(win)
        return np.convolve(xp, ker, mode="valid")

    def _safe_unit(self, vx: np.ndarray, vy: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        n = np.sqrt(vx * vx + vy * vy) + eps
        return vx / n, vy / n

    def _angle(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        return np.arctan2(vy, vx)

    def _unwrap_diff(self, theta: np.ndarray, fps: float) -> np.ndarray:
        d = np.gradient(np.unwrap(theta), edge_order=1)
        return d * float(fps)

    def _center_from_points(self, xs: np.ndarray, ys: np.ndarray, mode: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(mode, (int, np.integer)):
            return xs[:, int(mode)], ys[:, int(mode)]
        return xs.mean(axis=1), ys.mean(axis=1)

    def _build_ego_block_for_joined(self, j: pd.DataFrame, fps: float, pose_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        N = int(self.params["pose_n"])
        neck = int(self.params["neck_idx"])
        tail = int(self.params["tail_base_idx"])
        win  = int(self.params["smooth_win"])
        mode = self.params["center_mode"]

        XA = j[[f"{self.params['x_prefix']}{k}_A" for k in range(N)]].to_numpy()
        YA = j[[f"{self.params['y_prefix']}{k}_A" for k in range(N)]].to_numpy()
        XB = j[[f"{self.params['x_prefix']}{k}_B" for k in range(N)]].to_numpy()
        YB = j[[f"{self.params['y_prefix']}{k}_B" for k in range(N)]].to_numpy()
        frames = j["frame"].to_numpy().astype(int)

        # optional smoothing
        if win and win > 1:
            XA = np.vstack([self._smooth_1d(XA[:, k], win) for k in range(N)]).T
            YA = np.vstack([self._smooth_1d(YA[:, k], win) for k in range(N)]).T
            XB = np.vstack([self._smooth_1d(XB[:, k], win) for k in range(N)]).T
            YB = np.vstack([self._smooth_1d(YB[:, k], win) for k in range(N)]).T

        # centers
        cxA, cyA = self._center_from_points(XA, YA, mode)
        cxB, cyB = self._center_from_points(XB, YB, mode)

        # headings (neck - tail) and units
        hxA, hyA = XA[:, neck] - XA[:, tail], YA[:, neck] - YA[:, tail]
        hxB, hyB = XB[:, neck] - XB[:, tail], YB[:, neck] - YB[:, tail]
        uhxA, uhyA = self._safe_unit(hxA, hyA)
        uhxB, uhyB = self._safe_unit(hxB, hyB)
        # left-hand orthogonal
        uoxA, uoyA = -uhyA, uhxA
        uoxB, uoyB = -uhyB, uhxB

        # velocities of centers (per second)
        vAx = np.gradient(cxA) * float(fps)
        vAy = np.gradient(cyA) * float(fps)
        vBx = np.gradient(cxB) * float(fps)
        vBy = np.gradient(cyB) * float(fps)
        speedA = np.sqrt(vAx*vAx + vAy*vAy)
        speedB = np.sqrt(vBx*vBx + vBy*vBy)

        # heading angles + angular speed
        thA = self._angle(uhxA, uhyA)
        thB = self._angle(uhxB, uhyB)
        angspeedA = self._unwrap_diff(thA, fps)
        angspeedB = self._unwrap_diff(thB, fps)

        # ego projections of velocity
        vA_para = vAx * uhxA + vAy * uhyA
        vA_perp = vAx * uoxA + vAy * uoyA
        vB_para = vBx * uhxB + vBy * uhyB
        vB_perp = vBx * uoxB + vBy * uoyB

        # displacement A→B in world + A-centric ego coords of B
        dx = cxB - cxA
        dy = cyB - cyA
        distAB = np.sqrt(dx*dx + dy*dy)

        dxA = dx * uhxA + dy * uhyA
        dyA = dx * uoxA + dy * uoyA

        # B-centric ego coords of A
        dxB = (-dx) * uhxB + (-dy) * uhyB
        dyB = (-dx) * uoxB + (-dy) * uoyB

        # relative heading B wrt A
        dth = np.unwrap(thB) - np.unwrap(thA)
        rel_cos = np.cos(dth)
        rel_sin = np.sin(dth)

        names = [
            "A_speed", "A_v_para", "A_v_perp", "A_ang_speed",
            "A_heading_cos", "A_heading_sin",
            "AB_dist", "AB_dx_egoA", "AB_dy_egoA",
            "rel_heading_cos", "rel_heading_sin",
            "B_speed", "B_v_para", "B_v_perp", "B_ang_speed",
        ]

        AtoB = np.vstack([
            speedA, vA_para, vA_perp, angspeedA,
            np.cos(thA), np.sin(thA),
            distAB, dxA, dyA,
            rel_cos, rel_sin,
            speedB, vB_para, vB_perp, angspeedB,
        ]).astype(np.float32)

        # For B→A, swap roles but keep same semantic ordering (B is 'self')
        BtoA = np.vstack([
            speedB, vB_para, vB_perp, angspeedB,
            np.cos(thB), np.sin(thB),
            distAB, dxB, dyB,
            np.cos(-dth), np.sin(-dth),
            speedA, vA_para, vA_perp, angspeedA,
        ]).astype(np.float32)

        return frames, AtoB, BtoA, names
    


@register_feature
class PairWavelet:
    """
    'pair-wavelet' — CWT spectrograms on PairPoseDistancePCA outputs.
    Expects input df to contain columns:
        - 'perspective' (0 = A→B, 1 = B→A)
        - 'frame' (preferred) or 'time' (if used as order column)
        - PC0..PC{k-1} (k = number of PCA components)
    Returns a DataFrame with columns:
        - frame (or time if that was the order col)
        - perspective
        - W_c{comp}_f{fi}  (log-power, clamped, for each component×frequency)
      and (optionally) passthrough group/sequence if present in df.

    Notes:
      • Stateless (no fitting).
      • FPS is inferred from constant df['fps'] if present; else fps_default.
      • Frequencies are dyadically spaced in [f_min, f_max].
    """

    name = "pair-wavelet"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        # sampling
        fps_default=30.0,

        # wavelet band and resolution
        f_min=0.2,
        f_max=5.0,
        n_freq=25,

        # wavelet family string (PyWavelets)
        wavelet="cmor1.5-1.0",

        # log power clamp
        log_floor=-3.0,

        # naming / passthrough
        pc_prefix="PC",                 # columns like PC0, PC1, ...
        order_pref=("frame", "time"),   # which column to use as the time base
        seq_col="sequence",
        group_col="group",
        cols=None,  # explicit list of columns to transform; if None, fallback to PC prefix or auto-detect numeric columns
    )
    def _select_input_columns(self, df: pd.DataFrame) -> List[str]:
        # 1) explicit columns override
        cols_param = self.params.get("cols", None)
        if cols_param:
            cols = [c for c in cols_param if c in df.columns]
            if not cols:
                raise ValueError("[pair-wavelet] None of the requested 'cols' are present in df.")
            return cols
        # 2) PC-prefixed columns
        pc_cols = self._pc_columns(df, self.params["pc_prefix"])
        if pc_cols:
            return pc_cols
        # 3) Auto-detect: all numeric columns except known meta
        meta_like = {self.params.get("seq_col", "sequence"),
                     self.params.get("group_col", "group"),
                     "frame", "time", "perspective", "id", "fps"}
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_like]
        if not num_cols:
            raise ValueError("[pair-wavelet] Could not auto-detect numeric feature columns.")
        return num_cols

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _PYWT_OK:
            raise ImportError(
                "PyWavelets (pywt) not available. Install with `pip install PyWavelets`."
            )
        self.params = _merge_params(params, self._defaults)
        # pre-build frequency vector & scales for speed; will recompute if params change
        self._cache_key = None
        self._frequencies = None
        self._scales = None
        self._central_f = None

    # ---- feature protocol ----
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def finalize_fit(self) -> None: pass
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return
    def partial_fit(self, df: pd.DataFrame) -> None: return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = self._order_col(df)
        fps = self._infer_fps(df, p["fps_default"])
        in_cols = self._select_input_columns(df)
        if "perspective" not in df.columns:
            raise ValueError("[pair-wavelet] Missing 'perspective' column.")

        # prepare wavelet frequencies/scales
        self._prepare_band(fps)

        # compute per perspective block (keeps ordering stable)
        out_blocks: List[pd.DataFrame] = []
        for persp, g in df.groupby("perspective"):
            g = g.sort_values(order_col)
            Z = g[in_cols].to_numpy(dtype=float)  # shape (T, k)
            T, k = Z.shape

            # compute power spectrogram (k components × n_freq × T)
            power = np.empty((k, len(self._frequencies), T), dtype=np.float32)
            # each component independently
            for comp in range(k):
                coeffs, _ = pywt.cwt(
                    Z[:, comp],
                    self._scales,
                    self._wavelet_obj(),
                    sampling_period=1.0 / float(fps),
                )
                power[comp] = (np.abs(coeffs) ** 2).astype(np.float32)

            # log + clamp
            eps = np.finfo(np.float32).tiny
            log_power = np.log(power + eps)
            log_power = np.maximum(log_power, float(p["log_floor"]))

            # flatten to (T, k*n_freq)
            flat = log_power.reshape(k * len(self._frequencies), T).T  # (T, F_flat)

            # column names: W_{in_cols[comp]}_f{fi}
            colnames = [
                f"W_{in_cols[comp]}_f{fi}"
                for comp in range(k)
                for fi in range(len(self._frequencies))
            ]
            block = pd.DataFrame(flat, columns=colnames)
            block[order_col] = g[order_col].to_numpy()
            block["perspective"] = int(persp)

            # optional passthrough
            for col in (p["seq_col"], p["group_col"]):
                if col in df.columns:
                    block[col] = df[col].iloc[0]

            out_blocks.append(block)

        if not out_blocks:
            return pd.DataFrame(columns=[order_col, "perspective"])

        out = pd.concat(out_blocks, ignore_index=True)
        out = out.sort_values(["perspective", order_col]).reset_index(drop=True)

        # Attach JSON-serializable metadata only (so parquet writers won't error)
        try:
            out.attrs["frequencies_hz"] = self._frequencies.tolist() if self._frequencies is not None else []
            out.attrs["scales"] = self._scales.tolist() if self._scales is not None else []
            out.attrs["wavelet"] = str(self.params.get("wavelet", ""))
            out.attrs["fps"] = float(fps)
            out.attrs["pc_cols"] = [c for c in in_cols if c.startswith(self.params.get("pc_prefix","PC"))]
            out.attrs["input_columns"] = list(map(str, in_cols))
        except Exception:
            # As a safety net, drop attrs if anything is not serializable
            out.attrs.clear()
        return out

    # ---- internals ----
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column in df.")

    def _infer_fps(self, df: pd.DataFrame, default: float) -> float:
        if "fps" in df.columns:
            vals = pd.Series(df["fps"]).dropna().unique()
            if len(vals) == 1:
                try:
                    return float(vals[0])
                except Exception:
                    pass
        return float(default)

    def _pc_columns(self, df: pd.DataFrame, prefix: str) -> List[str]:
        # accept PC0, PC1, ... contiguous from 0 until missing
        pc_cols = []
        i = 0
        while True:
            col = f"{prefix}{i}"
            if col in df.columns:
                pc_cols.append(col)
                i += 1
            else:
                break
        return pc_cols

    def _prepare_band(self, fps: float) -> None:
        key = (self.params["wavelet"], float(self.params["f_min"]),
               float(self.params["f_max"]), int(self.params["n_freq"]), float(fps))
        if self._cache_key == key and self._frequencies is not None:
            return
        f_min = float(self.params["f_min"])
        f_max = float(self.params["f_max"])
        n_freq = int(self.params["n_freq"])
        # dyadic spacing
        freqs = 2.0 ** np.linspace(np.log2(f_min), np.log2(f_max), n_freq)
        w = self._wavelet_obj()
        central_f = pywt.central_frequency(w)
        scales = float(fps) / (freqs * central_f)
        self._frequencies = freqs.astype(np.float32)
        self._scales = scales.astype(np.float32)
        self._central_f = float(central_f)
        self._cache_key = key

    def _wavelet_obj(self):
        return pywt.ContinuousWavelet(self.params["wavelet"])    
    

@register_feature
class GlobalTSNE: 
    """
    Global t-SNE over multiple prior features discovered from the dataset's feature indexes.
    - inputs: a list of dicts describing which feature outputs to include.
      Each input supports:
        {
          "feature": "pair-wavelet",  # prior feature name
          "run_id":  None,                         # optional, pick latest if None
          "pattern": "wavelet_social_seq=*persp=*.npz",  # files to glob inside the run folder
          "load": { "kind": "npz", "key": "spectrogram", "transpose": True }  # loader spec
        }
    The loaded arrays are treated as (T x D); arrays from all inputs with the same sequence
    are length-aligned (min T) and concatenated horizontally.

    Heavy work happens in fit(); transform() returns a tiny stub DF.
    """
    name: str = "global-tsne"
    version: str = "0.2"
    params: dict

    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            # Multi-input spec (good defaults for your current pipeline):
            inputs=[
                {
                    "feature": "pair-wavelet",
                    "run_id": None,  # pick latest
                    "pattern": "wavelet_social_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
                {
                    "feature": "pair-egocentric-wavelet",
                    "run_id": None,
                    "pattern": "wavelet_ego_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
            ],
            inputset=None,
            map_existing_inputs=False,
            reuse_embedding=None,
            artifact=None,
            scaler=None,
            random_state=42,
            r_scaler=200_000,          # cap for standardizer fit sample
            total_templates=2000,      # farthest-first target
            pre_quota_per_key=50,      # pre-sample per key
            perplexity=50,
            n_jobs=8,
            partial_k=25,
            partial_iters=100,
            partial_lr=1.0,
            map_chunk=20_000,
        )
        self.params = {**defaults, **(params or {})}
        self._inputs_overridden = bool(params and "inputs" in params)
        self._rng = np.random.default_rng(self.params["random_state"])
        self._scaler: Optional[StandardScaler] = None
        self._embedding: Optional[TSNEEmbedding] = None
        self._artifacts: Dict[str, Any] = {}
        self._ds = None  # set by bind_dataset
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        self._inputs_meta: Dict[str, Any] = {}
        self._resolved_inputs: List[dict] = []
        self._scope_filter: Optional[dict] = None

    # dataset hook
    def bind_dataset(self, ds) -> None:
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ---- Feature protocol ----
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def partial_fit(self, X: pd.DataFrame) -> None: raise NotImplementedError

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("GlobalTSNE requires dataset binding. Ensure Dataset.run_feature calls bind_dataset().")

        # 0) Gather inputs from feature indexes
        inputset_name = self.params.get("inputset")
        explicit_inputs = self.params["inputs"] if (self._inputs_overridden or not inputset_name) else None
        inputs, inputs_meta = _resolve_inputs(
            self._ds, explicit_inputs, inputset_name,
            explicit_override=self._inputs_overridden
        )
        self._resolved_inputs = inputs
        self._inputs_meta = inputs_meta
        # key is safe_seq (derived from dataset index or filename)
        per_key_parts: Dict[str, List[np.ndarray]] = {}

        allowed_safe = None
        if self._scope_filter:
            allowed_safe = set(self._scope_filter.get("safe_sequences") or [])

        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(self._ds, feat_name)
            else:
                run_root = _feature_run_root(self._ds, feat_name, run_id)

            pattern = spec.get("pattern", "*.npz")
            loader  = spec.get("load", {"kind": "npz", "key": "spectrogram", "transpose": True})

            files = sorted(run_root.glob(pattern))
            if not files:
                print(f"[global-tsne] WARN: no files for {feat_name} ({run_id}) with pattern {pattern}", file=sys.stderr)
                continue

            seq_map = self._feature_seq_map(feat_name, run_id)

            def parse_key(path: Path) -> str:
                stem = path.stem
                m_seq = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
                if m_seq:
                    safe_seq = m_seq.group(1)
                else:
                    safe_seq = seq_map.get(path.resolve())
                if not safe_seq:
                    safe_seq = to_safe_name(stem)
                return str(safe_seq)

            for pth in files:
                key = parse_key(pth)
                if allowed_safe is not None and key not in allowed_safe:
                    continue
                arr = self._load_matrix(pth, loader)
                if arr is None or arr.size == 0:
                    continue
                if key not in per_key_parts:
                    per_key_parts[key] = []
                per_key_parts[key].append(arr)

        if not per_key_parts:
            raise RuntimeError("GlobalTSNE found no usable inputs from the specified prior features.")

        # 1) Concatenate inputs horizontally per key (align on min T)
        features_per_key: Dict[str, np.ndarray] = {}
        n_total = 0
        for key, mats in per_key_parts.items():
            if not mats:
                continue
            T_min = min(m.shape[0] for m in mats)
            mats = [m[:T_min] for m in mats]
            X = np.hstack(mats)  # (T_min, D_total)
            features_per_key[key] = X
            n_total += X.shape[0]

        keys = list(features_per_key.keys())
        if not keys:
            raise RuntimeError("No combined feature frames after alignment.")

        if bool(self.params.get("map_existing_inputs")):
            self._prepare_reuse_artifacts()
            mapped = self._map_sequences(features_per_key)
            self._artifacts["mapped_coords"] = mapped
            self._artifacts["keys"] = keys
            self._artifacts["inputs_meta"] = inputs_meta
            return

        # 2) Global standardizer
        r_scaler = int(self.params["r_scaler"])
        pools = []
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            take = min(X.shape[0], max(1000, int(0.05 * X.shape[0])))
            idx = self._rng.choice(X.shape[0], size=take, replace=False)
            pools.append(X[idx])
        Xsamp = np.vstack(pools)
        if Xsamp.shape[0] > r_scaler:
            idx = self._rng.choice(Xsamp.shape[0], size=r_scaler, replace=False)
            Xsamp = Xsamp[idx]
        scaler = StandardScaler().fit(Xsamp)
        self._scaler = scaler

        # 3) Template selection (farthest-first over pre-sample)
        T_target = int(self.params["total_templates"])
        quota = max(int(self.params["pre_quota_per_key"]), T_target // max(1, len(keys)))
        pre = []
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            take = min(X.shape[0], quota * 3)
            idx = self._rng.choice(X.shape[0], size=take, replace=False)
            pre.append(scaler.transform(X[idx]))
        X_pre = np.vstack(pre)

        sel = [int(self._rng.integers(0, X_pre.shape[0]))]
        d2  = np.sum((X_pre - X_pre[sel[0]])**2, axis=1)
        while len(sel) < min(T_target, X_pre.shape[0]):
            i = int(np.argmax(d2))
            sel.append(i)
            d2 = np.minimum(d2, np.sum((X_pre - X_pre[i])**2, axis=1))
        templates = X_pre[np.array(sel)]

        # 4) Fit openTSNE on templates
        aff = affinity.PerplexityBasedNN(
            templates,
            perplexity=int(self.params["perplexity"]),
            metric="euclidean",
            method="annoy",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        init = initialization.pca(templates, random_state=int(self.params["random_state"]))
        emb = TSNEEmbedding(
            init, aff,
            negative_gradient_method="fft",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        emb.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True, verbose=False)
        emb.optimize(n_iter=750, momentum=0.8, inplace=True, verbose=False)
        self._embedding = emb

        # 5) Map all frames in chunks
        mapped = self._map_sequences(features_per_key)

        # hold artifacts for save_model
        self._artifacts["keys"] = keys
        self._artifacts["templates"] = templates
        self._artifacts["template_indices"] = np.array(sel)
        self._artifacts["mapped_coords"] = mapped
        self._artifacts["inputs_meta"] = inputs_meta

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Just a stub so Dataset.run_feature can index a parquet per (group,sequence)
        return pd.DataFrame({"global_tsne_done": [True]})

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # Save templates
        templates = self._artifacts.get("templates")
        if templates is not None:
            np.savez_compressed(run_root / "global_templates_features.npz", templates=templates)

        # Save tsne coords of templates + selection indices
        if self._embedding is not None:
            Y_templates = np.asarray(self._embedding)
            np.savez_compressed(run_root / "global_tsne_templates.npz",
                                Y=Y_templates, sel=self._artifacts.get("template_indices", np.array([], int)))

        # Save embedding + scaler
        joblib.dump(
            {
                "embedding": self._embedding,
                "scaler": self._scaler,
                "params": self.params,
            },
            run_root / "global_opentsne_embedding.joblib"
        )

        # Save per-key coords
        mapped = self._artifacts.get("mapped_coords", {})
        for safe_seq, Y in mapped.items():
            out_name = f"global_tsne_coords_seq={safe_seq}.npz"
            np.savez_compressed(run_root / out_name, Y=Y)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._embedding = bundle.get("embedding", None)
        self._scaler = bundle.get("scaler", None)

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _resolve_feature_run_root(self, feature_name: str, run_id: str | None) -> tuple[str, Path]:
        if self._ds is None:
            raise RuntimeError("GlobalTSNE requires dataset binding before resolving feature runs.")
        if run_id is None:
            run_id, run_root = _latest_feature_run_root(self._ds, feature_name)
        else:
            run_root = _feature_run_root(self._ds, feature_name, run_id)
        return str(run_id), run_root

    def _load_embedding_from_spec(self, spec: dict) -> tuple[Any, Any]:
        feature = spec.get("feature")
        if not feature:
            raise ValueError("reuse_embedding spec requires 'feature'.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"reuse_embedding: no files matching '{pattern}' in {run_root}")
        bundle = joblib.load(files[0])
        embedding = bundle.get("embedding")
        scaler = bundle.get("scaler")
        if embedding is None:
            raise ValueError(f"reuse_embedding bundle missing 'embedding' object: {files[0]}")
        self._artifacts.setdefault("reuse_sources", {})["embedding"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return embedding, scaler

    def _load_scaler_from_spec(self, spec: dict):
        feature = spec.get("feature")
        if not feature:
            raise ValueError("scaler spec requires 'feature'.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"scaler spec: no files matching '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        scaler = obj if key is None else obj[key]
        self._artifacts.setdefault("reuse_sources", {})["scaler"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return scaler

    def _load_templates_from_spec(self, spec: dict) -> Optional[np.ndarray]:
        feature = spec.get("feature")
        if not feature:
            raise ValueError("artifact spec requires 'feature'.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_templates_features.npz")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"artifact spec: no files matching '{pattern}' in {run_root}")
        load_spec = spec.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        arr = _load_array_from_spec(files[0], load_spec)
        if arr is None:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        self._artifacts.setdefault("reuse_sources", {})["templates"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return arr

    def _prepare_reuse_artifacts(self) -> None:
        emb_spec = self.params.get("reuse_embedding")
        if not emb_spec:
            raise ValueError("map_existing_inputs=True requires params['reuse_embedding'].")
        embedding, scaler = self._load_embedding_from_spec(emb_spec)
        self._embedding = embedding
        self._scaler = scaler
        scaler_override = self.params.get("scaler")
        if scaler_override:
            self._scaler = self._load_scaler_from_spec(scaler_override)
        if self._scaler is None:
            raise ValueError("Reusable embedding bundle missing scaler; provide params['scaler'].")
        art_spec = self.params.get("artifact")
        if art_spec:
            templates = self._load_templates_from_spec(art_spec)
            if templates is not None:
                self._artifacts["templates"] = templates

    def _map_sequences(self, features_per_key: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._embedding is None or self._scaler is None:
            raise RuntimeError("GlobalTSNE mapping requires both embedding and scaler.")
        embedding = self._embedding
        scaler = self._scaler
        CHUNK = int(self.params["map_chunk"])
        Kp = int(self.params["partial_k"])
        Pp = int(self.params["perplexity"])
        It = int(self.params["partial_iters"])
        Lr = float(self.params["partial_lr"])

        def map_chunk_block(X_chunk_std: np.ndarray) -> np.ndarray:
            part = embedding.prepare_partial(X_chunk_std, initialization="median", k=Kp, perplexity=Pp)
            part = part.optimize(
                n_iter=It,
                learning_rate=Lr,
                exaggeration=2.0,
                momentum=0.0,
                inplace=False,
                verbose=False,
            )
            return np.asarray(part)

        mapped: dict[str, np.ndarray] = {}
        for key, X in features_per_key.items():
            if X.shape[0] == 0:
                continue
            Xs = scaler.transform(X)
            blocks = [map_chunk_block(Xs[i:i + CHUNK]) for i in range(0, Xs.shape[0], CHUNK)]
            mapped[key] = np.vstack(blocks) if blocks else np.empty((0, 2), dtype=np.float32)
        return mapped

    # ------- loaders -------
    def _load_matrix(self, pth: Path, spec: dict) -> np.ndarray | None:
        """
        Load a 2D matrix from a file according to a 'load' spec:
        - kind: 'npz' or 'parquet'
        - key:  for npz (required), ignored for parquet
        - columns: optional list of columns for parquet
        - drop_columns: optional list of columns to drop before numeric-only selection
        - numeric_only: default True for parquet; selects only numeric dtypes
        - transpose: bool; if True, returns A.T
        """
        kind = str(spec.get("kind", "parquet")).lower()
        transpose = bool(spec.get("transpose", False))

        if kind == "npz":
            key = spec.get("key", None)
            if not key:
                raise ValueError("load.kind='npz' requires 'key' in load spec")
            npz = np.load(pth, allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"Key '{key}' not found in {pth.name}; available: {list(npz.files)}")
            A = np.asarray(npz[key])
            if A.ndim == 1:  # ensure 2D
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            return A.T if transpose else A

        elif kind == "parquet":
            import pandas as pd
            df = pd.read_parquet(pth)

            # Optional: drop explicitly unwanted columns first
            drop_cols = set(spec.get("drop_columns", []))
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")

            cols = spec.get("columns", None)
            if cols is not None:
                # Use exactly these columns
                df_num = df[cols]
            else:
                # Default: keep numeric columns only
                if spec.get("numeric_only", True):
                    df_num = df.select_dtypes(include=["number"])
                else:
                    # Try to coerce everything except the obvious non-numeric
                    df_num = df.apply(pd.to_numeric, errors="coerce")

            A = df_num.to_numpy(dtype=np.float32, copy=False)
            if A.ndim == 1:
                A = A[None, :]
            return A.T if transpose else A

        else:
            raise ValueError(f"Unknown load.kind='{kind}'")
        

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
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        assign_config = self.params.get("assign") or {}
        self._assign_inputs_override = "inputs" in assign_config
        self._assign_inputs_meta: Dict[str, Any] = {}
        self._scope_constraints: Optional[dict[str, set[str]]] = None

    def _load_one_general(self, path: Path, spec: dict) -> np.ndarray:
        arr = _load_array_from_spec(path, spec)
        if arr is None:
            return np.empty((0, 0), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"_load_one_general: Loaded array is not 2D: {arr.shape}")
        return arr

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key_from_path(self, path: Path, seq_map: dict[Path, str]) -> str:
        """
        Extract safe_seq from file path using naming convention or dataset index fallback.
        """
        stem = path.stem
        m_seq = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
        if m_seq:
            safe_seq = m_seq.group(1)
        else:
            safe_seq = seq_map.get(path.resolve())
        if not safe_seq:
            safe_seq = to_safe_name(stem)
        return str(safe_seq)

    def _collect_inputs_per_key(self, inputs: list[dict]) -> dict[str, list[np.ndarray]]:
        """
        For each input spec, glob pattern, load each file as matrix, group by sequence.
        Returns dict[safe_seq] -> list[np.ndarray]
        """
        per_key_parts = {}
        scope_active = bool(self._scope_constraints)
        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            resolved_run_id, run_root = self._resolve_feature_run_root(feat_name, run_id)
            pattern = spec.get("pattern", "*.npz")
            load_spec = spec.get("load", {"kind": "npz", "key": "spectrogram", "transpose": True})
            files = sorted(run_root.glob(pattern))
            seq_map = self._feature_seq_map(feat_name, resolved_run_id)
            meta_map = self._feature_path_metadata(feat_name, resolved_run_id) if scope_active else {}
            for pth in files:
                try:
                    abs_path = pth.resolve()
                except Exception:
                    abs_path = pth
                meta = meta_map.get(abs_path, {})
                group_val = str(meta.get("group", "") or "")
                seq_val = str(meta.get("sequence", "") or "")
                safe_seq = str(meta.get("sequence_safe", "") or "")
                if not safe_seq:
                    safe_seq = self._extract_key_from_path(pth, seq_map)
                safe_group = to_safe_name(group_val) if group_val else self._extract_safe_group_from_path(pth)
                if not self._is_scope_allowed(group_val, seq_val, safe_seq, safe_group):
                    continue
                key = safe_seq
                arr = self._load_one_general(pth, load_spec)
                if arr is None or arr.size == 0:
                    continue
                if key not in per_key_parts:
                    per_key_parts[key] = []
                per_key_parts[key].append(arr)
        return per_key_parts

    def set_scope_constraints(self, scope: Optional[dict]) -> None:
        """
        Capture dataset-level group/sequence filters so assignment can respect them.
        """
        if not scope:
            self._scope_constraints = None
            return

        def _to_norm_set(values: Any) -> Optional[set[str]]:
            if not values:
                return None
            out = {str(v) for v in values if isinstance(v, (str, int)) or v}
            out = {v for v in out if v}
            return out or None

        groups = _to_norm_set(scope.get("groups"))
        safe_groups = _to_norm_set(scope.get("safe_groups"))
        if groups and not safe_groups:
            safe_groups = {to_safe_name(g) for g in groups}

        sequences = _to_norm_set(scope.get("sequences"))
        safe_sequences = _to_norm_set(scope.get("safe_sequences"))
        if sequences and not safe_sequences:
            safe_sequences = {to_safe_name(s) for s in sequences}

        scoped = {
            k: v for k, v in {
                "groups": groups,
                "safe_groups": safe_groups,
                "sequences": sequences,
                "safe_sequences": safe_sequences,
            }.items() if v
        }
        self._scope_constraints = scoped or None

    def _feature_path_metadata(self, feature_name: str, run_id: str) -> dict[Path, dict[str, str]]:
        """
        Returns {abs_path -> {group, sequence, sequence_safe}} for a prior feature run.
        """
        mapping: dict[Path, dict[str, str]] = {}
        if self._ds is None:
            return mapping
        try:
            idx_path = _feature_index_path(self._ds, feature_name)
        except Exception:
            return mapping
        if not idx_path.exists():
            return mapping
        try:
            df = pd.read_csv(idx_path)
        except Exception:
            return mapping
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            return mapping
        df["group"] = df["group"].fillna("").astype(str)
        df["sequence"] = df["sequence"].fillna("").astype(str)
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].apply(lambda v: to_safe_name(v) if v else "")

        for _, row in df.iterrows():
            abs_raw = row.get("abs_path")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            try:
                abs_path = Path(abs_raw).resolve()
            except Exception:
                abs_path = Path(abs_raw)
            mapping[abs_path] = {
                "group": str(row.get("group", "") or ""),
                "sequence": str(row.get("sequence", "") or ""),
                "sequence_safe": str(row.get("sequence_safe", "") or ""),
            }
        return mapping

    def _extract_safe_group_from_path(self, path: Path) -> str:
        stem = path.stem
        if "__" in stem:
            return stem.split("__", 1)[0]
        return ""

    def _is_scope_allowed(self,
                          group: str,
                          sequence: str,
                          safe_sequence: str,
                          safe_group: Optional[str]) -> bool:
        if not self._scope_constraints:
            return True
        scope = self._scope_constraints
        group = group or ""
        sequence = sequence or ""
        safe_sequence = safe_sequence or ""
        safe_group = safe_group or ""
        if scope.get("groups") and group not in scope["groups"]:
            return False
        if scope.get("safe_groups") and safe_group not in scope["safe_groups"]:
            return False
        if scope.get("sequences") and sequence not in scope["sequences"]:
            return False
        if scope.get("safe_sequences") and safe_sequence not in scope["safe_sequences"]:
            return False
        return True

    def _load_scaler(self, spec: dict):
        """
        Loads a scaler (e.g. sklearn StandardScaler) from a joblib file.
        spec: {
          feature: str,
          run_id: str|None,
          pattern: str (default: "global_opentsne_embedding.joblib"),
          key: str|None,
        }
        Returns: scaler object, or obj[key] if key is not None.
        """
        feat_name = spec["feature"]
        run_id = spec.get("run_id")
        _, run_root = self._resolve_feature_run_root(feat_name, run_id)
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"_load_scaler: No files matching '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        return obj if key is None else obj[key]

    # Dataset binding
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def bind_dataset(self, ds): self._ds = ds

    # ----------------- Fit helpers -----------------
    def _resolve_feature_run_root(self, feature_name: str, run_id: str | None) -> tuple[str, Path]:
        froot = self._ds.get_root("features") / feature_name
        idx_path = froot / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feature_name}' at {idx_path}")
        idx = pd.read_csv(idx_path)
        if run_id is None:
            # pick latest finished if present; else started_at
            if "finished_at" in idx.columns:
                idx = idx.sort_values("finished_at", ascending=False, na_position="last")
            elif "started_at" in idx.columns:
                idx = idx.sort_values("started_at", ascending=False, na_position="last")
            if idx.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(idx.iloc[0]["run_id"])
        resolved = str(run_id)
        run_root = froot / resolved
        if not run_root.exists():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        return resolved, run_root

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
                df = df.select_dtypes(include=["number"]) if numeric_only else df.apply(pd.to_numeric, errors="coerce")
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
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, art.get("run_id"))
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
            # Optional scaler: if provided, standardize before prediction; else predict in raw space.
            scaler = None
            if "scaler" in assign and assign["scaler"]:
                scaler = self._load_scaler(assign["scaler"])

            assign_inputset = assign.get("inputset")
            explicit_assign_inputs = assign.get("inputs") if (self._assign_inputs_override or not assign_inputset) else None
            resolved_assign_inputs, assign_inputs_meta = _resolve_inputs(
                self._ds,
                explicit_assign_inputs,
                assign_inputset,
                explicit_override=self._assign_inputs_override,
            )
            self._assign_inputs_meta = assign_inputs_meta

            # Collect per-key inputs (may be empty)
            per_key_parts = self._collect_inputs_per_key(resolved_assign_inputs)
            for key, mats in per_key_parts.items():
                if not mats:
                    continue
                T_min = min(m.shape[0] for m in mats)
                mats_trim = [m[:T_min] for m in mats]
                X_full = np.hstack(mats_trim)
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

                labels = self._kmeans.predict(X_use)
                self._assign_labels[key] = labels.astype(np.int32)
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
            fname = f"global_kmeans_labels_seq={safe_seq}.npz"
            np.savez_compressed(run_root / fname, labels=labels)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._kmeans = bundle["kmeans"]
        self._fit_dim = int(bundle.get("fit_dim") or 0)
        self._fit_columns = bundle.get("fit_columns")
        self._fit_artifact_info = bundle.get("artifact_info", {})


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

    # ---------- framework API ----------
    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

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
            blocks = _collect_sequence_blocks(self._ds, specs)
            if not blocks:
                raise RuntimeError("[global-ward] Inputset produced no usable matrices.")
            X = np.vstack(list(blocks.values()))
            if X.ndim != 2:
                raise ValueError(f"[global-ward] Loaded array must be 2D; got shape={X.shape}")
            self._artifact_inputs_meta = meta
            return X.astype(np.float64, copy=False)

        feat_name = art.get("feature", None)
        run_id    = art.get("run_id", None)
        pattern   = art.get("pattern", None)
        load_spec = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})

        if not feat_name or not pattern:
            raise ValueError("[global-ward] 'artifact.feature' and 'artifact.pattern' are required in params.")

        idx_path = self._ds.get_root("features") / feat_name / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"[global-ward] No index for feature '{feat_name}'. Expected: {idx_path}")

        df_idx = pd.read_csv(idx_path)
        if run_id is None:
            if "finished_at" in df_idx.columns:
                cand = df_idx[df_idx["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df_idx
                base = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                        ascending=False, kind="stable")
            else:
                base = df_idx.sort_values(by=["started_at"], ascending=False, kind="stable")
            if base.empty:
                raise ValueError(f"[global-ward] No runs found for feature '{feat_name}'.")
            run_id = str(base.iloc[0]["run_id"])

        run_root = self._ds.get_root("features") / feat_name / run_id
        if not run_root.exists():
            raise FileNotFoundError(f"[global-ward] run root not found: {run_root}")

        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[global-ward] No files matching '{pattern}' in {run_root}")

        mats = []
        for fp in files:
            kind = (load_spec.get("kind") or "npz").lower()
            transpose = bool(load_spec.get("transpose", False))
            if kind == "npz":
                key = load_spec.get("key", None)
                if key is None:
                    raise ValueError("[global-ward] 'load.key' required for npz")
                npz = np.load(fp, allow_pickle=True)
                if key not in npz.files:
                    raise KeyError(f"[global-ward] Key '{key}' not found in {fp.name}")
                A = np.asarray(npz[key])
            elif kind == "parquet":
                df = pd.read_parquet(fp)
                cols = load_spec.get("columns", None)
                if cols:
                    A = df[cols].to_numpy(dtype=np.float32)
                else:
                    if load_spec.get("numeric_only", True):
                        A = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                    else:
                        A = df.to_numpy(dtype=np.float32)
            else:
                raise ValueError(f"[global-ward] Unsupported load.kind='{kind}'")

            if A.ndim == 1:
                A = A[None, :]
            if transpose:
                A = A.T
            mats.append(A.astype(np.float64, copy=False))

        if not mats:
            raise RuntimeError(f"[global-ward] No matrices loaded from {run_root} with pattern {pattern}")

        X = mats[0] if len(mats) == 1 else np.vstack(mats)
        if X.ndim != 2:
            raise ValueError(f"[global-ward] Loaded array must be 2D; got {X.shape} from {files[0].name}")
        return X


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
        self.skip_existing_outputs = True

        self._ds = None
        self._Z = None
        self._templates = None
        self._cluster_ids = None
        self._assign_nn = None
        self._scaler = None
        self._inputs = []
        self._sequence_label_store: dict[str, dict[str, Any]] = {}
        self._allowed_safe_sequences: Optional[set[str]] = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._scope_filter: Optional[dict] = None
        self._inputs_overridden = bool(params and "inputs" in params)
        self._inputs_meta: Dict[str, Any] = {}

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}
        safe_sequences = scope.get("safe_sequences") if scope else None
        if safe_sequences:
            self._allowed_safe_sequences = {str(s) for s in safe_sequences}
        else:
            self._allowed_safe_sequences = None
        pair_safe_map = scope.get("pair_safe_map") if scope else None
        if pair_safe_map:
            inv = {}
            for pair, safe in pair_safe_map.items():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                inv[str(safe)] = (str(pair[0]), str(pair[1]))
            self._pair_map = inv
        else:
            self._pair_map = {}

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def partial_fit(self, X): raise NotImplementedError

    # ---------- helpers ----------
    def _resolve_feature_run_root(self, feature_name: str, run_id: Optional[str]) -> tuple[str, Path]:
        froot = self._ds.get_root("features") / feature_name
        idx_path = froot / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feature_name}' at {idx_path}")
        df = pd.read_csv(idx_path)
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                      ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            if df.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(df.iloc[0]["run_id"])
        resolved = str(run_id)
        run_root = froot / resolved
        if not run_root.exists():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        return resolved, run_root

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.get("artifact", {})
        feature = art.get("feature")
        if not feature:
            raise ValueError("[ward-assign] artifact.feature required.")
        run_id, run_root = self._resolve_feature_run_root(feature, art.get("run_id"))
        pattern = art.get("pattern", "*.npz")
        loader = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No files matching '{pattern}' in {run_root}")
        kind = str(loader.get("kind", "npz")).lower()
        if kind == "npz":
            key = loader.get("key")
            npz = np.load(files[0], allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"[ward-assign] Key '{key}' missing in {files[0].name}")
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            return A.astype(np.float32, copy=False)
        elif kind == "parquet":
            df = pd.read_parquet(files[0])
            if loader.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            return df.to_numpy(dtype=np.float32, copy=False)
        else:
            raise ValueError(f"[ward-assign] Unsupported artifact load.kind='{kind}'")

    def _load_scaler(self, spec: Optional[dict]):
        if not spec:
            return None
        feat_name = spec.get("feature")
        if not feat_name:
            raise ValueError("[ward-assign] scaler.feature required.")
        _, run_root = self._resolve_feature_run_root(feat_name, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] Could not locate scaler '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        return obj if key is None else obj[key]

    def _feature_path_metadata(self, feature_name: str, run_id: str) -> dict[Path, dict[str, str]]:
        mapping: dict[Path, dict[str, str]] = {}
        if self._ds is None:
            return mapping
        idx_path = _feature_index_path(self._ds, feature_name)
        if not idx_path.exists():
            return mapping
        try:
            df = pd.read_csv(idx_path)
        except Exception:
            return mapping
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            return mapping
        df["group"] = df["group"].fillna("").astype(str)
        df["sequence"] = df["sequence"].fillna("").astype(str)
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].apply(lambda v: to_safe_name(v) if v else "")

        for _, row in df.iterrows():
            abs_raw = row.get("abs_path")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            try:
                abs_path = Path(abs_raw).resolve()
            except Exception:
                abs_path = Path(abs_raw)
            mapping[abs_path] = {
                "group": str(row.get("group", "") or ""),
                "sequence": str(row.get("sequence", "") or ""),
                "sequence_safe": str(row.get("sequence_safe", "") or ""),
            }
        return mapping

    def _load_one_general(self, path: Path, spec: dict) -> tuple[np.ndarray, Optional[np.ndarray]]:
        frame_vals: Optional[np.ndarray] = None
        kind = str(spec.get("kind", "parquet")).lower()
        if kind == "npz":
            key = spec.get("key")
            npz = np.load(path, allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"[ward-assign] Key '{key}' missing in {path.name}")
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            return A.astype(np.float32, copy=False), None
        elif kind == "parquet":
            df = pd.read_parquet(path)
            frame_col = spec.get("frame_column")
            if frame_col is None:
                frame_col = "frame"
            if frame_col and frame_col in df.columns:
                try:
                    frame_vals = df[frame_col].to_numpy(dtype=np.int64, copy=False)
                except Exception:
                    frame_vals = df[frame_col].to_numpy(copy=False)
                drop_frame = bool(spec.get("drop_frame_column", False))
                if drop_frame and not (spec.get("columns") and frame_col in spec.get("columns", [])):
                    df = df.drop(columns=[frame_col])
            drop_cols = spec.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            cols = spec.get("columns")
            if cols:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return np.empty((0, 0), dtype=np.float32)
                df = df[use]
            elif spec.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if spec.get("transpose"):
                arr = arr.T
            return arr, frame_vals
        else:
            raise ValueError(f"[ward-assign] Unknown load.kind='{kind}'")

    def _infer_sequence(self, df: pd.DataFrame) -> str:
        if "sequence" in df.columns and not df["sequence"].empty:
            val = str(df["sequence"].iloc[0])
            if val:
                return val
        if "sequence_safe" in df.columns and not df["sequence_safe"].empty:
            safe = str(df["sequence_safe"].iloc[0])
            if safe:
                pair = self._pair_map.get(safe)
                return pair[1] if pair else safe
        raise ValueError("[ward-assign] Input DataFrame missing 'sequence' column.")

    def _collect_sequence_inputs(self) -> dict[str, dict[str, Any]]:
        if not self._inputs:
            return {}
        n_inputs = len(self._inputs)
        per_seq: dict[str, dict[str, Any]] = {}
        allowed_safe = set(self._allowed_safe_sequences or [])
        enforce_scope = bool(allowed_safe)

        for idx, spec in enumerate(self._inputs):
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            resolved_run_id, run_root = self._resolve_feature_run_root(feat_name, run_id)
            pattern = spec.get("pattern", "*.parquet")
            files = sorted(run_root.glob(pattern))
            if not files:
                print(f"[ward-assign] WARN: no files for {feat_name} ({resolved_run_id}) pattern={pattern}", file=sys.stderr)
                continue
            seq_map = _build_path_sequence_map(self._ds, feat_name, resolved_run_id)
            meta_map = self._feature_path_metadata(feat_name, resolved_run_id)
            load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
            for pth in files:
                try:
                    abs_path = pth.resolve()
                except Exception:
                    abs_path = pth
                safe_seq = seq_map.get(abs_path)
                if not safe_seq:
                    safe_seq = to_safe_name(pth.stem)
                safe_seq = str(safe_seq)
                if enforce_scope and safe_seq not in allowed_safe:
                    continue
                arr, frame_vals = self._load_one_general(pth, load_spec)
                if arr is None or arr.size == 0:
                    continue
                entry = per_seq.setdefault(safe_seq, {
                    "mats": [None] * n_inputs,
                    "frames": None,
                    "sequence": None,
                    "group": None,
                })
                entry["mats"][idx] = arr
                meta = meta_map.get(abs_path, {})
                if entry["sequence"] is None:
                    entry["sequence"] = meta.get("sequence") or self._pair_map.get(safe_seq, ("", safe_seq))[1]
                if entry["group"] is None:
                    entry["group"] = meta.get("group") or self._pair_map.get(safe_seq, ("", ""))[0]
                if entry["frames"] is None and frame_vals is not None:
                    entry["frames"] = frame_vals

        clean = {}
        for safe_seq, payload in per_seq.items():
            mats = payload["mats"]
            if any((m is None or m.size == 0) for m in mats):
                print(f"[ward-assign] WARN: missing inputs for sequence '{safe_seq}', skipping.", file=sys.stderr)
                continue
            clean[safe_seq] = payload
        return clean

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("[ward-assign] Dataset not bound. Use dataset.run_feature(...).")

        inputs = self.params.get("inputs") or []
        inputset_name = self.params.get("inputset")
        explicit_inputs = inputs if (self._inputs_overridden or not inputset_name) else None
        self._inputs, self._inputs_meta = _resolve_inputs(
            self._ds,
            explicit_inputs,
            inputset_name,
            explicit_override=self._inputs_overridden,
        )

        self._sequence_label_store.clear()

        # Load Ward linkage
        ward_spec = self.params["ward_model"]
        _, ward_root = self._resolve_feature_run_root(ward_spec["feature"], ward_spec.get("run_id"))
        pattern = ward_spec.get("pattern", "model.joblib")
        files = sorted(ward_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No Ward model '{pattern}' in {ward_root}")
        bundle = joblib.load(files[0])
        self._Z = bundle.get("linkage_matrix")
        if self._Z is None:
            raise ValueError("[ward-assign] Ward model missing linkage_matrix.")

        # Reload artifact matrix (templates) to derive centroids
        self._templates = self._load_artifact_matrix()
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

        # optional scaler
        self._scaler = self._load_scaler(self.params.get("scaler"))

        seq_payloads = self._collect_sequence_inputs()
        if not seq_payloads:
            raise RuntimeError("[ward-assign] No usable inputs found for assignment.")

        for safe_seq, payload in seq_payloads.items():
            mats = payload["mats"]
            lengths = [m.shape[0] for m in mats if m is not None]
            if not lengths:
                continue
            T_min = min(lengths)
            mats_trim = [m[:T_min] for m in mats]
            X_full = np.hstack(mats_trim)
            if self._scaler is not None:
                if not hasattr(self._scaler, "transform"):
                    raise ValueError("[ward-assign] scaler object missing transform().")
                X_use = self._scaler.transform(X_full)
            else:
                X_use = X_full
            idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
            labels = self._cluster_ids[idxs.ravel()]
            frames = payload.get("frames")
            if frames is None or len(frames) < T_min:
                frames = np.arange(T_min, dtype=int)
            else:
                frames = frames[:T_min]
            seq_name = payload.get("sequence") or self._pair_map.get(safe_seq, ("", safe_seq))[1]
            self._sequence_label_store[safe_seq] = {
                "sequence": seq_name,
                "frames": frames.astype(int, copy=False),
                "labels": labels.astype(int, copy=False),
            }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._sequence_label_store:
            return pd.DataFrame(index=[])
        seq = self._infer_sequence(df)
        safe_seq = to_safe_name(seq)
        bundle = self._sequence_label_store.get(safe_seq)
        if not bundle:
            return pd.DataFrame(index=[])
        out = pd.DataFrame({
            "frame": bundle["frames"],
            "cluster": bundle["labels"],
            "sequence": seq,
        })
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        joblib.dump({
            "params": self.params,
            "n_clusters": int(self.params.get("n_clusters", 20)),
        }, path)
        for safe_seq, bundle in self._sequence_label_store.items():
            labels = bundle.get("labels")
            if labels is None:
                continue
            np.savez_compressed(run_root / f"global_ward_labels_seq={safe_seq}.npz", labels=labels)


#### VISUALIZATION
def plot_tsne_withcolors(ax,tsne_result,quantity,title,corrskip=1,plotskip=1,colortype='scalar',qmin=0.001,qmax=0.999,alphaval=0.3,s=4,coloroffset=0,cmapname='cool',setxylimquantile=False):
    colordata = quantity.copy()
    if len(colordata)>len(tsne_result):
        colordata = colordata[::corrskip]
    if len(colordata.shape)>1:
        colordata = colordata[:,0]
    if colortype=='scalar':
        cmap=plt.get_cmap(cmapname)  # or 'cool'
        q0,q1 = np.quantile(colordata,[qmin,qmax])
        colordata = colordata-q0
        colordata = colordata/(q1-q0)
        colordata[colordata<0] = 0
        colordata[colordata>1] = 1
        colordata *= 0.99
        colors = cmap(colordata)
    else:
#         cmap=plt.get_cmap('Set1')
        colors = snscolors[colordata.astype(int)+coloroffset]
    tp = tsne_result
    # [ax.scatter([-100],[-100],alpha=1,s=10,color=cmap(i*0.99/np.max(groupvalues)),label='group '+str(i)) for i in np.arange(max(groupvalues)+1)]  # legend hack
    scatterplot = ax.scatter(tp[::plotskip,0],tp[::plotskip,1],s=s,alpha=alphaval,color=colors[::plotskip],rasterized=True)
    if setxylimquantile:
        ax.set_xlim(np.quantile(tp[:,0],[qmin,qmax]))
        ax.set_ylim(np.quantile(tp[:,1],[qmin,qmax]))
    else:
        ax.set_xlim(np.quantile(tp[:,0],[0,1]))
        ax.set_ylim(np.quantile(tp[:,1],[0,1]))
    ax.set_title(title,fontsize=16)       
    return scatterplot, colordata    

@register_feature
class VizGlobalColored:
    """
    Generic wrapper to visualize any global embedding (t-SNE, PCA, UMAP, etc.)
    colored by arbitrary labels.

    Params (defaults target the global t-SNE + k-means workflow):
      coords: {feature, run_id, pattern, load:{kind:"npz", key:"Y"}}
      labels: optional spec matching coords (same structure as coords) OR
              {"source": "labels", "kind": "<label_kind>", "load": {...}}
              to load from labels/<kind>/index.csv (e.g., CalMS21 ground truth)
      coord_key_regex: regex applied to coord filenames (default extracts the sequence name)
      label_key_regex: regex for label filenames (defaults to coord regex)
      label_missing_value: sentinel for missing labels (default -1 -> gray)
      label_order: optional list of label values to fix the color ordering
      label_name_map: optional dict mapping label ids -> display names
      plot_max: max points to scatter
      palette: seaborn palette name or list
      title: plot title
      point_size / point_alpha: scatter style tweaks
      debug_save_arrays: if True, save Y_all/L_all (aligned, pre-subsample) as debug_viz_arrays.npz
    Output:
      - PNG in run_root plus a single marker parquet row for indexing.
    """

    name = "viz-global-colored"
    version = "0.1"

    def __init__(self, params=None):
        defaults = {
            "coords": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_tsne_coords_seq=*.npz",
                "load": {"kind": "npz", "key": "Y", "transpose": False},
            },
            "labels": None,
            "coord_key_regex": r"seq=(.+?)(?:_persp=.*)?$",
            "label_key_regex": INHERIT_REGEX,
            "label_missing_value": -1,
            "label_order": None,
            "label_name_map": None,
            "plot_max": 300_000,
            "palette": "tab20",
            "title": "Global embedding colored scatter",
            "point_size": 2.0,
            "point_alpha": 0.35,
            "debug_save_arrays": False,
        }
        self.params = dict(defaults)
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    merged = dict(self.params[k])
                    merged.update(v)
                    self.params[k] = merged
                else:
                    self.params[k] = v
        self._ds = None
        self._figs = []
        self._marker_written = False
        self._summary = {}
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        self._scope_constraints: Optional[dict] = None
        self._debug_arrays: Optional[dict] = None

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_constraints(self, scope: Optional[dict]) -> None:
        self._scope_constraints = scope or {}

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def partial_fit(self, X): raise NotImplementedError
    def finalize_fit(self): pass

    def _load_artifacts_glob(self, spec):
        ds = self._ds
        idx = ds.get_root("features") / spec["feature"] / "index.csv"
        df = pd.read_csv(idx)
        run_id = spec.get("run_id")
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(by=["finished_at" if len(cand) else "started_at"], ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            run_id = str(df.iloc[0]["run_id"])
        run_root = ds.get_root("features") / spec["feature"] / run_id
        files = sorted(run_root.glob(spec["pattern"]))
        return run_id, run_root, files

    def _load_one(self, path, load_spec):
        kind = load_spec.get("kind", "npz").lower()
        if kind == "npz":
            npz = np.load(path, allow_pickle=True)
            A = np.asarray(npz[load_spec["key"]])
            if load_spec.get("transpose"):
                A = A.T
            return A
        if kind == "parquet":
            df = pd.read_parquet(path)
            drop_cols = load_spec.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            cols = load_spec.get("columns")
            if cols:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return np.empty((0, len(cols)), dtype=np.float32)
                df = df[use]
            elif load_spec.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                # coerce everything numeric-style to avoid object columns
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if load_spec.get("transpose"):
                arr = arr.T
            return arr
        if kind == "joblib":
            obj = joblib.load(path)
            key = load_spec.get("key")
            return obj if key is None else obj[key]
        raise ValueError(f"Unsupported load.kind='{kind}'")

    def _load_labels_from_index(self, spec: dict) -> list[tuple[str, Path]]:
        """
        Load (sequence_safe, path) pairs from labels/<kind>/index.csv.
        Supports optional glob-style filtering via spec['pattern'] on filename.
        """
        if self._ds is None:
            raise RuntimeError("VizGlobalColored requires dataset binding to load labels.")
        kind = spec.get("kind")
        if not kind:
            raise ValueError("labels spec with source='labels' requires 'kind'.")
        labels_root = self._ds.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Labels index not found: {idx_path}")
        df = pd.read_csv(idx_path)
        pattern = spec.get("pattern")
        entries: list[tuple[str, Path]] = []
        for _, row in df.iterrows():
            abs_raw = row.get("abs_path", "")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            pth = Path(abs_raw)
            if pattern and not fnmatch.fnmatch(pth.name, pattern):
                continue
            seq_safe = str(row.get("sequence_safe") or row.get("sequence") or "").strip()
            if not seq_safe:
                seq_safe = to_safe_name(pth.stem)
            entries.append((seq_safe, pth))
        return entries

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key(self, path: Path, regex: Optional[str], seq_map: dict[Path, str]):
        stem = path.stem
        m = re.search(regex, stem) if regex else None
        if not m:
            safe_seq = seq_map.get(path.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(stem)
            return str(safe_seq)
        if m.lastindex is None:
            return m.group(0)
        if m.lastindex == 1:
            return m.group(1)
        return tuple(m.groups())

    def _prepare_color_map(self, labels):
        unique_vals = list(pd.unique(pd.Series(labels, dtype="object")))
        missing = self.params.get("label_missing_value", -1)
        order = self.params.get("label_order")
        if order:
            order_list = [o for o in order if o in unique_vals]
            unique_vals = order_list + [u for u in unique_vals if u not in order_list]

        palette = sns.color_palette(self.params.get("palette", "tab20"),
                                    max(1, len([u for u in unique_vals if u != missing])))
        color_map = {}
        idx = 0
        for val in unique_vals:
            if val == missing:
                continue
            color_map[val] = palette[idx % len(palette)]
            idx += 1
        if missing in unique_vals:
            color_map[missing] = (0.7, 0.7, 0.7)
        return color_map

    def _label_display(self, val):
        name_map = self.params.get("label_name_map") or {}
        if isinstance(name_map, dict):
            if val in name_map:
                return str(name_map[val])
            try:
                ival = int(val)
                if ival in name_map:
                    return str(name_map[ival])
            except Exception:
                pass
        return str(val)

    def fit(self, X_iter):
        coord_spec = self.params["coords"]
        coord_run_id, _, coord_files = self._load_artifacts_glob(coord_spec)
        if not coord_files:
            raise FileNotFoundError("[viz-global-colored] No coordinate files found.")

        scope = self._scope_constraints or {}
        allowed_safe = set(scope.get("safe_sequences") or [])
        allowed_sequences = set(scope.get("sequences") or [])
        allowed_any = set()
        if allowed_sequences:
            allowed_any.update(allowed_sequences)
            allowed_any.update(to_safe_name(s) for s in allowed_sequences)
        allowed_any.update(allowed_safe)

        key_regex = self.params.get("coord_key_regex")
        coord_seq_map = self._feature_seq_map(coord_spec["feature"], coord_run_id)
        Y_list, key_list, n_list = [], [], []
        for f in coord_files:
            key = self._extract_key(f, key_regex, coord_seq_map)
            key_safe = to_safe_name(key)
            if allowed_any and key not in allowed_any and key_safe not in allowed_any:
                continue
            arr = self._load_one(f, coord_spec["load"])
            if arr.ndim != 2:
                arr = np.atleast_2d(arr)
            Y_list.append(arr)
            key_list.append(key)
            n_list.append(arr.shape[0])
        if not Y_list:
            raise RuntimeError("No coordinate arrays loaded.")

        labels_spec = self.params.get("labels")
        missing_value = self.params.get("label_missing_value", -1)
        if labels_spec:
            label_regex_param = self.params.get("label_key_regex", INHERIT_REGEX)
            label_regex = key_regex if label_regex_param == INHERIT_REGEX else label_regex_param
            label_source = str(labels_spec.get("source", "feature")).lower()
            label_load_spec = labels_spec.get("load") or {"kind": "npz", "key": "labels"}
            lab_map: dict[Any, np.ndarray] = {}
            if label_source == "labels":
                entries = self._load_labels_from_index(labels_spec)
                if allowed_any:
                    entries = [(s, p) for s, p in entries if s in allowed_any]
                seq_map = {pth.resolve(): safe_seq for safe_seq, pth in entries}
                for safe_seq, pth in entries:
                    lab_key = self._extract_key(pth, label_regex, seq_map)
                    lab_map[lab_key] = np.asarray(self._load_one(pth, label_load_spec))
            else:
                label_run_id, _, label_files = self._load_artifacts_glob(labels_spec)
                label_seq_map = self._feature_seq_map(labels_spec["feature"], label_run_id)
                for lf in label_files:
                    lab_key = self._extract_key(lf, label_regex, label_seq_map)
                    lab_map[lab_key] = np.asarray(self._load_one(lf, label_load_spec))
            L_parts = []
            for idx, (key, n) in enumerate(zip(key_list, n_list)):
                arr = lab_map.get(key)
                if arr is None:
                    arr = lab_map.get(to_safe_name(str(key)))
                if arr is None:
                    print(f"[viz-global-colored] WARN: missing labels for key={key}; assigning {missing_value}",
                          file=sys.stderr)
                    arr = np.full(n, missing_value, dtype=int)
                else:
                    arr = np.asarray(arr).ravel()
                    if arr.shape[0] < n:
                        if n % arr.shape[0] == 0:
                            # common case: coord stream duplicated per-id/perspective; repeat labels accordingly
                            factor = n // arr.shape[0]
                            if factor == 2:
                                print(f"[viz-global-colored] INFO: labels shorter than coords for key={key} "
                                      f"({arr.shape[0]} vs {n}); repeating labels x{factor} "
                                      "(likely per-id/perspective duplication in inputs).",
                                      file=sys.stderr)
                            arr = np.tile(arr, factor)
                        else:
                            # lengths differ but not a clean multiple; trim coords to label length
                            print(f"[viz-global-colored] INFO: trimming coords for key={key} "
                                  f"from {n} to {arr.shape[0]} (labels length) due to mismatch.",
                                  file=sys.stderr)
                            Y_list[idx] = Y_list[idx][:arr.shape[0]]
                            n_list[idx] = arr.shape[0]
                            n = arr.shape[0]
                    if arr.shape[0] != n:
                        nmin = min(arr.shape[0], n)
                        if nmin <= 0:
                            continue
                        if nmin < n:
                            # trim coords to match available labels
                            print(f"[viz-global-colored] INFO: trimming coords for key={key} "
                                  f"from {n} to {nmin} to match labels {arr.shape[0]}",
                                  file=sys.stderr)
                            Y_list[idx] = Y_list[idx][:nmin]
                            n_list[idx] = nmin
                        arr = arr[:nmin]
                L_parts.append(arr)
            if not L_parts:
                raise RuntimeError("No labels aligned with coordinates.")
            Y_all = np.vstack(Y_list)
            L_all = np.concatenate(L_parts)
        else:
            L_parts = [np.zeros(n, dtype=int) for n in n_list]
            Y_all = np.vstack(Y_list)
            L_all = np.concatenate(L_parts)

        if bool(self.params.get("debug_save_arrays")):
            self._debug_arrays = {
                "Y_all": Y_all.copy(),
                "L_all": L_all.copy(),
                "key_list": list(key_list),
                "n_list": list(n_list),
            }

        plot_max = int(self.params.get("plot_max", 300_000))
        if Y_all.shape[0] > plot_max:
            rng = np.random.default_rng(42)
            sel = rng.choice(Y_all.shape[0], size=plot_max, replace=False)
            Y_plot = Y_all[sel]
            L_plot = L_all[sel]
        else:
            Y_plot = Y_all
            L_plot = L_all

        color_map = self._prepare_color_map(L_all if labels_spec else L_plot)
        colors = np.array([color_map.get(val, (0.7, 0.7, 0.7)) for val in L_plot])

        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.scatter(
            Y_plot[:, 0],
            Y_plot[:, 1],
            c=colors,
            s=float(self.params.get("point_size", 2.0)),
            alpha=float(self.params.get("point_alpha", 0.35)),
            linewidths=0,
        )
        ax.set_title(self.params.get("title", "Global embedding colored scatter"))
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")

        unique_vals = pd.unique(pd.Series(L_plot, dtype="object"))
        if len(unique_vals) <= 15:
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                           markerfacecolor=color_map.get(val, (0.7, 0.7, 0.7)),
                           markeredgecolor="none", label=self._label_display(val))
                for val in unique_vals
            ]
            if handles:
                ax.legend(handles=handles, title="label", loc="best", fontsize=8)

        out_name = "global_colored.png"
        self._figs = [(out_name, fig)]
        self._summary = {
            "points": int(Y_all.shape[0]),
            "plotted": int(Y_plot.shape[0]),
            "labels_present": bool(labels_spec),
        }

    def transform(self, X):
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        return pd.DataFrame([{
            "outputs": ",".join(fname for fname, _ in self._figs),
            "labels_present": bool(self.params.get("labels")),
        }])

    def save_model(self, path: Path):
        run_root = path.parent
        for fname, fig in self._figs:
            fig.savefig(run_root / fname, dpi=150, bbox_inches="tight")
        if bool(self.params.get("debug_save_arrays")) and self._debug_arrays:
            try:
                np.savez_compressed(run_root / "debug_viz_arrays.npz", **self._debug_arrays)
            except Exception as exc:
                print(f"[viz-global-colored] WARN: failed to save debug arrays: {exc}", file=sys.stderr)
        joblib.dump(
            {"params": self.params, "summary": self._summary,
             "files": [fname for fname, _ in self._figs]},
            run_root / "viz.joblib"
        )

VizGlobalTSNEColored = VizGlobalColored
