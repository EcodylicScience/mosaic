"""
TemporalStackingFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
import gc

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

from mosaic.core.dataset import register_feature
from .helpers import _load_array_from_spec
from mosaic.core.dataset import _latest_feature_run_root, _feature_index_path, _resolve_inputs, _feature_run_root
from mosaic.core.helpers import to_safe_name


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
    output_type = None  # Custom chunked output

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
        self._pair_filter_spec = self._inputs_meta.get("pair_filter")
        self._nn_lookup_cache: dict[str, dict] = {}
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
    def wants_full_inputset_data(self) -> bool: return False
    def fit(self, X: Iterable[pd.DataFrame]): pass
    def save_model(self, path: Path) -> None: raise NotImplementedError("stateless feature")

    def _get_or_build_nn_lookup(self, safe_seq: str) -> dict:
        """Build and cache the nearest-neighbor lookup for a sequence."""
        if safe_seq in self._nn_lookup_cache:
            return self._nn_lookup_cache[safe_seq]
        from .helpers import _build_nn_lookup
        lookup = _build_nn_lookup(self._ds, safe_seq, self._pair_filter_spec)
        self._nn_lookup_cache[safe_seq] = lookup
        return lookup

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

        base_matrix, base_names, frame_indices, pair_ids = self._load_sequence_matrix(safe_seq)
        if base_matrix is None or base_matrix.size == 0:
            raise ValueError(f"temporal-stack: missing inputs for sequence '{sequence}'.")

        if pair_ids is not None and len(np.unique(pair_ids, axis=0)) > 1:
            # Multi-pair: process each pair independently to avoid mixing temporal context
            return self._transform_multi_pair(
                base_matrix, base_names, frame_indices, pair_ids,
                sequence, group,
            )

        chunk_iter, stacked_names, total_rows = self._chunked_temporal_features(base_matrix, base_names)
        payload = {
            "parquet_chunk_iter": chunk_iter,
            "columns": stacked_names,
            "sequence": sequence,
            "group": group,
            "total_rows": total_rows,
            "frame_indices": frame_indices,
        }
        if pair_ids is not None:
            # Single pair — store as scalar
            payload["pair_ids"] = (int(pair_ids[0, 0]), int(pair_ids[0, 1]))
        return payload

    def _transform_multi_pair(
        self,
        base_matrix: np.ndarray,
        base_names: list[str],
        frame_indices: Optional[np.ndarray],
        pair_ids: np.ndarray,
        sequence: str,
        group: str,
    ) -> dict:
        """Process each unique pair independently, then concatenate results."""
        unique_pairs = np.unique(pair_ids, axis=0)
        all_stacked: list[np.ndarray] = []
        all_frames: list[np.ndarray] = []
        all_pair_ids: list[np.ndarray] = []
        stacked_names: Optional[list[str]] = None

        for pid in unique_pairs:
            mask = (pair_ids[:, 0] == pid[0]) & (pair_ids[:, 1] == pid[1])
            sub_matrix = base_matrix[mask]
            sub_frames = frame_indices[mask] if frame_indices is not None else None

            chunk_iter, names, total_rows = self._chunked_temporal_features(sub_matrix, base_names)
            if stacked_names is None:
                stacked_names = names

            # Collect all chunks for this pair
            chunks = []
            for _start, chunk_data in chunk_iter:
                chunks.append(chunk_data)
            if not chunks:
                continue
            pair_stacked = np.vstack(chunks)
            all_stacked.append(pair_stacked)
            if sub_frames is not None:
                all_frames.append(sub_frames[:pair_stacked.shape[0]])
            all_pair_ids.append(np.broadcast_to(pid[None, :], (pair_stacked.shape[0], 2)).copy())

        if not all_stacked:
            raise ValueError("temporal-stack: no data produced for any pair.")

        combined_data = np.vstack(all_stacked)
        combined_frames = np.concatenate(all_frames) if all_frames else None
        combined_pair_ids = np.vstack(all_pair_ids)

        return {
            "data": combined_data,
            "columns": stacked_names,
            "sequence": sequence,
            "group": group,
            "frame_indices": combined_frames,
            "pair_ids_per_row": combined_pair_ids,
        }

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

    def _load_sequence_matrix(self, safe_seq: str) -> tuple[Optional[np.ndarray], list[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and concatenate feature matrices for a sequence.

        Returns (base_matrix, col_names, frame_indices, pair_ids)
        where pair_ids is a (N, 2) int array of per-row (id1, id2) or None.

        When combining pair-aware inputs (with id1/id2 columns) with non-pair
        inputs, non-pair rows are replicated via a (frame, perspective) join to
        match the pair-aware row count.
        """
        # Phase 1: read all inputs and detect pair structure
        # Build NN pair filter if configured
        df_filter = None
        if self._pair_filter_spec:
            from .helpers import _nn_pair_mask
            nn_lookup = self._get_or_build_nn_lookup(safe_seq)
            if nn_lookup:
                def df_filter(df: pd.DataFrame) -> pd.DataFrame:
                    mask = _nn_pair_mask(df, nn_lookup)
                    return df.loc[mask].reset_index(drop=True)

        loaded: list[dict] = []
        for entry in self._resolved_inputs:
            path = entry["mapping"].get(safe_seq)
            if not path or not path.exists():
                return None, [], None, None
            df_full = pd.read_parquet(path)
            if df_filter is not None:
                df_full = df_filter(df_full)
                if df_full.empty:
                    return None, [], None, None
            has_pairs = "id1" in df_full.columns and "id2" in df_full.columns
            arr, _ = _load_array_from_spec(path, entry["load"], df=df_full)
            if arr is None or arr.size == 0:
                return None, [], None, None
            loaded.append({
                "arr": arr, "entry": entry,
                "has_pairs": has_pairs, "df": df_full,
            })
        if not loaded:
            return None, [], None, None

        any_pairs = any(e["has_pairs"] for e in loaded)
        all_pairs = all(e["has_pairs"] for e in loaded)

        # ---- Homogeneous case: all pair-aware or all non-pair ----
        if not any_pairs or all_pairs:
            mats = [e["arr"] for e in loaded]
            min_len = min(m.shape[0] for m in mats)
            if min_len == 0:
                return None, [], None, None
            mats = [m[:min_len] for m in mats]
            df0 = loaded[0]["df"]
            frame_indices = (
                df0["frame"].to_numpy(dtype=np.int32)[:min_len]
                if "frame" in df0.columns else None
            )
            pair_ids = None
            if any_pairs:
                pair_ids = np.column_stack([
                    df0["id1"].to_numpy(dtype=np.int32)[:min_len],
                    df0["id2"].to_numpy(dtype=np.int32)[:min_len],
                ])
            col_names: list[str] = []
            for e in loaded:
                prefix = e["entry"]["name"]
                col_names.extend(
                    [f"{prefix}__f{idx:04d}" for idx in range(e["arr"].shape[1])]
                )
            base = np.hstack(mats).astype(np.float32, copy=False)
            return base, col_names, frame_indices, pair_ids

        # ---- Mixed case: pair-aware master + non-pair inputs ----
        master_idx = next(i for i, e in enumerate(loaded) if e["has_pairs"])
        master_df = loaded[master_idx]["df"]
        master_arr = loaded[master_idx]["arr"]
        n_master = master_arr.shape[0]

        master_frames = (
            master_df["frame"].to_numpy(dtype=np.int32)
            if "frame" in master_df.columns else None
        )
        pair_ids = np.column_stack([
            master_df["id1"].to_numpy(dtype=np.int32),
            master_df["id2"].to_numpy(dtype=np.int32),
        ])
        master_persp = (
            master_df["perspective"].to_numpy()
            if "perspective" in master_df.columns else None
        )

        mats: list[np.ndarray] = []
        col_names: list[str] = []
        for i, e in enumerate(loaded):
            arr = e["arr"]
            prefix = e["entry"]["name"]
            col_names.extend(
                [f"{prefix}__f{idx:04d}" for idx in range(arr.shape[1])]
            )
            if e["has_pairs"]:
                # Same pair structure — align with master by row
                if arr.shape[0] >= n_master:
                    mats.append(arr[:n_master])
                else:
                    padded = np.zeros((n_master, arr.shape[1]), dtype=np.float32)
                    padded[: arr.shape[0]] = arr
                    mats.append(padded)
            else:
                # Non-pair: replicate rows via (frame, perspective) join
                df_np = e["df"]
                expanded = np.zeros((n_master, arr.shape[1]), dtype=np.float32)
                use_persp = (
                    master_persp is not None
                    and "perspective" in df_np.columns
                )
                if master_frames is not None and "frame" in df_np.columns:
                    np_frames = df_np["frame"].to_numpy()
                    if use_persp:
                        np_persp = df_np["perspective"].to_numpy()
                        lookup: dict[tuple, int] = {}
                        for ri in range(len(np_frames)):
                            lookup[(int(np_frames[ri]), np_persp[ri])] = ri
                        for j in range(n_master):
                            key = (int(master_frames[j]), master_persp[j])
                            si = lookup.get(key)
                            if si is not None and si < arr.shape[0]:
                                expanded[j] = arr[si]
                    else:
                        lookup_f: dict[int, int] = {}
                        for ri, f in enumerate(np_frames):
                            lookup_f[int(f)] = ri
                        for j in range(n_master):
                            si = lookup_f.get(int(master_frames[j]))
                            if si is not None and si < arr.shape[0]:
                                expanded[j] = arr[si]
                mats.append(expanded)

        base = np.hstack(mats).astype(np.float32, copy=False)
        return base, col_names, master_frames, pair_ids

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
