"""
TemporalStackingFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter1d

from mosaic.core.helpers import to_safe_name
from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.index import feature_index_path, latest_feature_run_root

from .helpers import _load_array_from_spec
from .spec import (
    COLUMNS,
    ArtifactSpec,
    Inputs,
    NNResult,
    OutputType,
    Params,
    ParquetLoadSpec,
    Result,
    register_feature,
)


@final
@register_feature
class TemporalStackingFeature:
    """
    Build temporal context windows over per-sequence feature files by stacking
    Gaussian-smoothed frames and optional pooled statistics, then saving the result as a new
    feature under features/temporal-stack__from__<inputs>/run_id.
    """

    name = "temporal-stack"
    version = "0.2"
    parallelizable = True
    output_type: OutputType = None

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        """Temporal-stacking parameters.

        Attributes:
            half: Half-window size in frames. Default 60.
            skip: Step between stacked frames. Default 5.
            use_temporal_stack: Compute temporal stack. Default True.
            sigma_stack: Gaussian sigma for temporal stack. Default 30.0.
            add_pool: Add pooled statistics. Default True.
            pool_stats: Statistics to pool. Default ("mean",).
            sigma_pool: Gaussian sigma for pooling. Default 30.0.
            fps: Frames per second. Default 30.0.
            win_sec: Window size in seconds. Default 0.5.
            write_chunk_size: Rows per output chunk. Default 1000.
            stack_chunk_size: Rows per stacking chunk. Default 1000.
        """

        half: int = Field(default=60, ge=0)
        skip: int = Field(default=5, ge=1)
        use_temporal_stack: bool = True
        sigma_stack: float = 30.0
        add_pool: bool = True
        pool_stats: tuple[str, ...] = ("mean",)
        sigma_pool: float = 30.0
        fps: float = Field(default=30.0, gt=0)
        win_sec: float = Field(default=0.5, gt=0)
        write_chunk_size: int = Field(default=1000, ge=1)
        stack_chunk_size: int = Field(default=1000, ge=1)
        pair_filter: NNResult | None = None

        @model_validator(mode="before")
        @classmethod
        def _normalize_pool_stats(cls, data: dict[str, object]) -> dict[str, object]:
            if isinstance(data, dict):
                ps = data.get("pool_stats")
                if isinstance(ps, str):
                    data["pool_stats"] = (ps.lower(),)
                elif ps is not None:
                    data["pool_stats"] = tuple(str(s).lower() for s in ps)
            return data

    def __init__(
        self,
        inputs: TemporalStackingFeature.Inputs,
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

        self._ds = None
        self._inputs: list[ArtifactSpec] = []
        self._resolved_inputs: list[tuple[ArtifactSpec, dict[str, Path]]] = []
        self._input_cache_ready = False
        self._scope: Scope = Scope()

    def bind_dataset(self, ds):
        self._ds = ds
        self._inputs = [
            ArtifactSpec(
                feature=r.feature,
                run_id=r.run_id,
                load=ParquetLoadSpec(numeric_only=True),
            )
            for r in self.inputs.feature_inputs
        ]
        self._nn_lookup_cache: dict[str, dict] = {}
        self._resolved_inputs = []
        self._input_cache_ready = False

    def set_scope(self, scope: Scope) -> None:
        self._scope = scope

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def wants_full_input_data(self) -> bool:
        # NOTE: returning False means yield_input_data uses metadata_only
        # mode, which skips time/frame filtering. run_feature() raises if
        # these filters are set. Future work: apply them in _load_sequence_matrix.
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]):
        pass

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def save_model(self, path: Path) -> None:
        raise NotImplementedError("stateless feature")

    def load_model(self, path: Path) -> None:
        raise NotImplementedError

    def _get_or_build_nn_lookup(self, entry_key: str) -> dict:
        """Build and cache the nearest-neighbor lookup for a sequence."""
        if entry_key in self._nn_lookup_cache:
            return self._nn_lookup_cache[entry_key]
        from .helpers import _build_nn_lookup

        lookup = _build_nn_lookup(self._ds, entry_key, self.params.pair_filter)
        self._nn_lookup_cache[entry_key] = lookup
        return lookup

    # ------------- Core logic -------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._ds is None:
            raise RuntimeError("temporal-stack: dataset not bound.")
        self._ensure_inputs_ready()

        seq_col = COLUMNS.seq_col
        group_col = COLUMNS.group_col
        sequence = (
            str(df[seq_col].iloc[0]) if seq_col in df.columns and not df.empty else None
        )
        group = (
            str(df[group_col].iloc[0])
            if group_col in df.columns and not df.empty
            else None
        )
        if not sequence:
            raise ValueError(
                "temporal-stack: unable to infer sequence from dataframe; ensure 'sequence' column exists."
            )

        entry_key = to_safe_name(sequence)
        allowed_entry_keys = self._scope.entry_keys or None
        if (
            allowed_entry_keys
            and entry_key not in allowed_entry_keys
        ):
            raise ValueError(
                f"temporal-stack: sequence '{sequence}' not present in resolved input scope."
            )

        base_matrix, base_names, frame_indices, pair_ids = self._load_sequence_matrix(
            entry_key
        )
        if base_matrix is None or base_matrix.size == 0:
            raise ValueError(
                f"temporal-stack: missing inputs for sequence '{sequence}'."
            )

        if pair_ids is not None and len(np.unique(pair_ids, axis=0)) > 1:
            # Multi-pair: process each pair independently to avoid mixing temporal context
            return self._transform_multi_pair(
                base_matrix,
                base_names,
                frame_indices,
                pair_ids,
                sequence,
                group,
            )

        chunk_iter, stacked_names, total_rows = self._chunked_temporal_features(
            base_matrix, base_names
        )
        payload = {
            "parquet_chunk_iter": chunk_iter,
            "columns": stacked_names,
            "sequence": sequence,
            "group": group,
            "total_rows": total_rows,
            "frame_indices": frame_indices,
        }
        if pair_ids is not None:
            # Single pair -- store as scalar
            payload["pair_ids"] = (int(pair_ids[0, 0]), int(pair_ids[0, 1]))
        return payload

    def _transform_multi_pair(
        self,
        base_matrix: np.ndarray,
        base_names: list[str],
        frame_indices: np.ndarray | None,
        pair_ids: np.ndarray,
        sequence: str,
        group: str,
    ) -> dict:
        """Process each unique pair independently, then concatenate results."""
        unique_pairs = np.unique(pair_ids, axis=0)
        all_stacked: list[np.ndarray] = []
        all_frames: list[np.ndarray] = []
        all_pair_ids: list[np.ndarray] = []
        stacked_names: list[str] | None = None

        for pid in unique_pairs:
            mask = (pair_ids[:, 0] == pid[0]) & (pair_ids[:, 1] == pid[1])
            sub_matrix = base_matrix[mask]
            sub_frames = frame_indices[mask] if frame_indices is not None else None

            chunk_iter, names, total_rows = self._chunked_temporal_features(
                sub_matrix, base_names
            )
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
                all_frames.append(sub_frames[: pair_stacked.shape[0]])
            all_pair_ids.append(
                np.broadcast_to(pid[None, :], (pair_stacked.shape[0], 2)).copy()
            )

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
        resolved: list[tuple[ArtifactSpec, dict[str, Path]]] = []
        allowed = self._scope.entry_keys or None
        for spec in self._inputs:
            feat_name = spec.feature
            if not feat_name:
                continue
            run_id = spec.run_id
            if run_id is None:
                run_id, _ = latest_feature_run_root(self._ds, feat_name)
            else:
                run_id = str(run_id)
            mapping = self._build_sequence_mapping(feat_name, run_id, allowed)
            if not mapping:
                continue
            resolved.append((spec, mapping))
        if not resolved:
            raise RuntimeError(
                "temporal-stack: no overlapping inputs found for the requested scope."
            )
        self._resolved_inputs = resolved
        self._input_cache_ready = True

    def _build_sequence_mapping(
        self, feature_name: str, run_id: str, allowed: set[str] | None
    ) -> dict[str, Path]:
        idx_path = feature_index_path(self._ds, feature_name)
        if not idx_path.exists():
            raise FileNotFoundError(
                f"temporal-stack: missing index for feature '{feature_name}' -> {idx_path}"
            )
        df = pd.read_csv(idx_path)
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            raise ValueError(
                f"temporal-stack: feature '{feature_name}' run_id='{run_id}' has no rows."
            )
        df["_seq_safe"] = df["sequence"].fillna("").apply(lambda v: to_safe_name(v))
        df = df[df["_seq_safe"] != ""]
        if allowed:
            df = df[df["_seq_safe"].isin(allowed)]
        mapping: dict[str, Path] = {}
        for _, row in df.iterrows():
            seq_safe = str(row["_seq_safe"])
            abs_path = row.get("abs_path")
            if not seq_safe or not isinstance(abs_path, str) or not abs_path:
                continue
            remapped = self._ds.resolve_path(abs_path)
            mapping[seq_safe] = remapped
        return mapping

    def _load_sequence_matrix(
        self, entry_key: str
    ) -> tuple[np.ndarray | None, list[str], np.ndarray | None, np.ndarray | None]:
        """
        Load and concatenate feature matrices for a sequence.

        Returns (base_matrix, col_names, frame_indices, pair_ids)
        where pair_ids is a (N, 2) int array of per-row (id1, id2) or None.

        When combining pair-aware inputs (with id1/id2 columns) with non-pair
        inputs, non-pair rows are replicated via a (frame, perspective) join to
        match the pair-aware row count.
        """
        # Build NN pair filter if configured
        df_filter = None
        if self.params.pair_filter:
            from .helpers import _nn_pair_mask

            nn_lookup = self._get_or_build_nn_lookup(entry_key)
            if nn_lookup:

                def df_filter(df: pd.DataFrame) -> pd.DataFrame:
                    mask = _nn_pair_mask(df, nn_lookup)
                    return df.loc[mask].reset_index(drop=True)

        loaded: list[dict] = []
        for artifact, mapping in self._resolved_inputs:
            path = mapping.get(entry_key)
            if not path or not path.exists():
                return None, [], None, None
            df_full = pd.read_parquet(path)
            if df_filter is not None:
                df_full = df_filter(df_full)
                if df_full.empty:
                    return None, [], None, None
            has_pairs = "id1" in df_full.columns and "id2" in df_full.columns
            arr, _ = _load_array_from_spec(path, artifact.load, df=df_full)
            if arr is None or arr.size == 0:
                return None, [], None, None
            loaded.append(
                {
                    "arr": arr,
                    "name": artifact.feature,
                    "has_pairs": has_pairs,
                    "df": df_full,
                }
            )
        if not loaded:
            return None, [], None, None

        any_pairs = any(e["has_pairs"] for e in loaded)
        all_pairs = all(e["has_pairs"] for e in loaded)

        # Homogeneous case: all pair-aware or all non-pair
        if not any_pairs or all_pairs:
            mats = [e["arr"] for e in loaded]
            min_len = min(m.shape[0] for m in mats)
            if min_len == 0:
                return None, [], None, None
            mats = [m[:min_len] for m in mats]
            df0 = loaded[0]["df"]
            frame_indices = (
                df0["frame"].to_numpy(dtype=np.int32)[:min_len]
                if "frame" in df0.columns
                else None
            )
            pair_ids = None
            if any_pairs:
                pair_ids = np.column_stack(
                    [
                        df0["id1"].to_numpy(dtype=np.int32)[:min_len],
                        df0["id2"].to_numpy(dtype=np.int32)[:min_len],
                    ]
                )
            col_names: list[str] = []
            for e in loaded:
                prefix = e["name"]
                col_names.extend(
                    [f"{prefix}__f{idx:04d}" for idx in range(e["arr"].shape[1])]
                )
            base = np.hstack(mats).astype(np.float32, copy=False)
            return base, col_names, frame_indices, pair_ids

        # Mixed case: pair-aware master + non-pair inputs
        master_idx = next(i for i, e in enumerate(loaded) if e["has_pairs"])
        master_df = loaded[master_idx]["df"]
        master_arr = loaded[master_idx]["arr"]
        n_master = master_arr.shape[0]

        master_frames = (
            master_df["frame"].to_numpy(dtype=np.int32)
            if "frame" in master_df.columns
            else None
        )
        pair_ids = np.column_stack(
            [
                master_df["id1"].to_numpy(dtype=np.int32),
                master_df["id2"].to_numpy(dtype=np.int32),
            ]
        )
        master_persp = (
            master_df["perspective"].to_numpy()
            if "perspective" in master_df.columns
            else None
        )

        mats: list[np.ndarray] = []
        col_names: list[str] = []
        for i, e in enumerate(loaded):
            arr = e["arr"]
            prefix = e["entry"]["name"]
            col_names.extend([f"{prefix}__f{idx:04d}" for idx in range(arr.shape[1])])
            if e["has_pairs"]:
                # Same pair structure -- align with master by row
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
                use_persp = master_persp is not None and "perspective" in df_np.columns
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

    def _chunked_temporal_features(
        self, base: np.ndarray, base_names: list[str]
    ) -> tuple[Iterable[tuple[int, np.ndarray]], list[str], int]:
        total_rows = base.shape[0]
        stack_chunk_size = self.params.stack_chunk_size
        use_stack = self.params.use_temporal_stack
        add_pool = self.params.add_pool

        # Precompute stacking metadata
        half = self.params.half
        step = self.params.skip
        sigma_stack = self.params.sigma_stack
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
            smoothed = gaussian_filter1d(
                base, sigma=sigma_stack, axis=0, mode="nearest"
            )
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

    # Pre-chunking helpers (_temporal_stack, _pooled_stats) removed;
    # see _chunked_temporal_features

    def _pooled_stats_arrays(
        self, base: np.ndarray, base_names: list[str]
    ) -> tuple[list[np.ndarray], list[str]]:
        stats = self.params.pool_stats
        if not stats:
            return [], []
        sigma = self.params.sigma_pool
        if sigma <= 0:
            sigma = max(1.0, self.params.win_sec * self.params.fps / 6.0)
        win_frames = max(1, int(round(self.params.win_sec * self.params.fps)))
        truncate = max(1.0, win_frames / (2.0 * sigma)) if sigma > 0 else 4.0
        mean_vals = gaussian_filter1d(
            base, sigma=sigma, axis=0, mode="nearest", truncate=truncate
        )
        outputs = []
        names = []
        if "mean" in stats:
            outputs.append(mean_vals)
            names.extend([f"{name}__pool_mean" for name in base_names])
        if "std" in stats or "variance" in stats:
            second = gaussian_filter1d(
                base**2, sigma=sigma, axis=0, mode="nearest", truncate=truncate
            )
            var = np.clip(second - mean_vals**2, 0.0, None)
            if "variance" in stats:
                outputs.append(var)
                names.extend([f"{name}__pool_var" for name in base_names])
            if "std" in stats:
                outputs.append(np.sqrt(var))
                names.extend([f"{name}__pool_std" for name in base_names])
        return outputs, names
