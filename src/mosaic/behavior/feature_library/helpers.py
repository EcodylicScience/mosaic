"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

import gc
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

from .params import ArtifactSpec, JoblibLoadSpec, NpzLoadSpec, ParquetLoadSpec

if TYPE_CHECKING:
    from .params import DictModel, LoadSpec, Result

import numpy as np
import pandas as pd

from mosaic.core.helpers import to_safe_name


def apply_exclude_cols(
    df: pd.DataFrame,
    exclude_cols: list[str] | None,
) -> pd.DataFrame:
    """Drop rows where any *exclude_cols* column is truthy.

    Silently skips column names not present in *df*.
    Returns *df* unchanged when *exclude_cols* is empty/None.
    """
    if not exclude_cols:
        return df
    present = [c for c in exclude_cols if c in df.columns]
    if not present:
        return df
    return df[~df[present].any(axis=1)]


def _pose_column_pairs(columns: Iterable[str]) -> list[tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs from column names."""
    pose_pairs = []
    xs = [c for c in columns if c.startswith("poseX")]
    for x_col in sorted(xs):
        idx = x_col[5:]
        y_col = f"poseY{idx}"
        if y_col in columns:
            pose_pairs.append((x_col, y_col))
    return pose_pairs


def _load_array_from_spec(
    path: Path,
    load_spec: LoadSpec,
    extract_frame_col: str | None = None,
    df_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    df: pd.DataFrame | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load a numpy array from a file according to a load specification.

    Parameters
    ----------
    path : Path
        Path to the file to load
    load_spec : LoadSpec
        Typed load specification (NpzLoadSpec, ParquetLoadSpec, or JoblibLoadSpec)
    extract_frame_col : str, optional
        If provided, extract this column as frame indices before feature loading
    df_filter : callable, optional
        If provided, applied to the raw parquet DataFrame **before** metadata
        columns are dropped.  Signature: ``(pd.DataFrame) -> pd.DataFrame``.
        Ignored for non-parquet files.
    df : pd.DataFrame, optional
        Pre-loaded DataFrame to use instead of reading from *path*.  When
        provided, ``df_filter`` should already have been applied by the caller.

    Returns
    -------
    tuple[np.ndarray or None, np.ndarray or None]
        (features array as float32, frame indices or None)
    """
    import pyarrow as pa

    frames: np.ndarray | None = None

    if isinstance(load_spec, NpzLoadSpec):
        npz = np.load(path, allow_pickle=True)
        if load_spec.key not in npz.files:
            return None, None
        A = np.asarray(npz[load_spec.key])
        if A.ndim == 1:
            A = A[None, :]
        if load_spec.transpose:
            A = A.T

    elif isinstance(load_spec, ParquetLoadSpec):
        if df is not None:
            df = df.copy()
        else:
            df = pd.read_parquet(path)

        if df_filter is not None:
            df = df_filter(df)
            if df.empty:
                return None, None

        frame_col = extract_frame_col or load_spec.frame_column
        if frame_col and frame_col in df.columns:
            try:
                frames = df[frame_col].to_numpy(dtype=np.int64, copy=True)
            except Exception:
                frames = df[frame_col].to_numpy(copy=True)

        if load_spec.drop_columns:
            df = df.drop(
                columns=[c for c in load_spec.drop_columns if c in df.columns],
                errors="ignore",
            )
        if load_spec.columns:
            df = df[[c for c in load_spec.columns if c in df.columns]]
        elif load_spec.numeric_only:
            df = df.select_dtypes(include=[np.number])
            for mc in (
                "frame",
                "time",
                "id",
                "id1",
                "id2",
                "id_a",
                "id_b",
                "id_A",
                "id_B",
            ):
                if mc in df.columns:
                    df = df.drop(columns=[mc])
        else:
            df = df.apply(pd.to_numeric, errors="coerce")
        A = df.to_numpy(dtype=np.float32, copy=True)
        del df
        pa.default_memory_pool().release_unused()

        if load_spec.transpose:
            A = A.T

    else:
        raise ValueError(f"Unsupported load spec type: {type(load_spec).__name__}")

    if A.size == 0:
        return None, frames
    if A.ndim == 1:
        A = A[None, :]
    return A.astype(np.float32, copy=False), frames



def _normalize_identity_columns(
    df: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None, str]:
    """
    Extract canonical identity columns from a frame-aligned DataFrame.

    Returns
    -------
    tuple
        (id1_series_or_None, id2_series_or_None, entity_level)
        where entity_level is one of {"global", "individual", "pair"}.
    """
    if "id1" in df.columns and "id2" in df.columns:
        id1 = pd.to_numeric(df["id1"], errors="coerce")
        id2 = pd.to_numeric(df["id2"], errors="coerce")
        if id1.notna().any() and id2.notna().any():
            return id1, id2, "pair"
        if id1.notna().any():
            return id1, pd.Series(np.nan, index=df.index), "individual"
        return None, None, "global"

    if "id" in df.columns:
        id1 = pd.to_numeric(df["id"], errors="coerce")
        if id1.notna().any():
            return id1, pd.Series(np.nan, index=df.index), "individual"
        return None, None, "global"

    # Backward-compatible aliases for older pair outputs
    for a_col, b_col in (("id_a", "id_b"), ("id_A", "id_B")):
        if a_col in df.columns and b_col in df.columns:
            id1 = pd.to_numeric(df[a_col], errors="coerce")
            id2 = pd.to_numeric(df[b_col], errors="coerce")
            if id1.notna().any() and id2.notna().any():
                return id1, id2, "pair"
            if id1.notna().any():
                return id1, pd.Series(np.nan, index=df.index), "individual"
            return None, None, "global"

    return None, None, "global"


def _build_path_sequence_map(
    ds, feature_name: str, run_id: str | None
) -> dict[Path, str]:
    """
    Build mapping from absolute file paths to sequence_safe names.

    Uses the dataset's feature index to create a lookup table.

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    feature_name : str
        Name of the feature
    run_id : str or None
        Run ID to filter by

    Returns
    -------
    dict[Path, str]
        Mapping from absolute path to sequence_safe name
    """
    # Import here to avoid circular import
    from mosaic.core.dataset import _feature_index_path

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
        df["sequence_safe"] = (
            df["sequence"]
            .fillna("")
            .apply(lambda v: to_safe_name(str(v)) if str(v).strip() else "")
        )

    for _, row in df.iterrows():
        abs_raw = row.get("abs_path")
        if not isinstance(abs_raw, str) or not abs_raw:
            continue
        try:
            # Use dataset's resolve_path to handle relative paths correctly
            # (relative paths are stored relative to dataset root, not CWD)
            abs_path = (
                ds.resolve_path(abs_raw)
                if hasattr(ds, "resolve_path")
                else Path(abs_raw).resolve()
            )
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


# -----------------------------------------------------------------------------
# Nearest-neighbor pair filtering utilities
# -----------------------------------------------------------------------------


def _build_nn_lookup(
    ds,
    sequence_safe: str,
    pair_filter_spec: DictModel | dict,
) -> dict[tuple[int, int], int]:
    """
    Build a per-frame nearest-neighbor lookup for a given sequence.

    Returns ``{(frame, individual_id): nn_id}`` from the nearest-neighbor
    feature parquet.  Returns an empty dict when the NN feature has not been
    computed for this sequence (fail-open).
    """
    from mosaic.core.dataset import (
        _feature_index_path,
        _feature_run_root,
        _latest_feature_run_root,
    )

    feat_name = pair_filter_spec.get("feature", "nearest-neighbor")
    run_id = pair_filter_spec.get("run_id")

    try:
        if run_id is None:
            run_id, _ = _latest_feature_run_root(ds, feat_name)
        else:
            _feature_run_root(ds, feat_name, str(run_id))
    except (ValueError, FileNotFoundError):
        return {}

    idx_path = _feature_index_path(ds, feat_name)
    if not idx_path.exists():
        return {}

    try:
        df_idx = pd.read_csv(idx_path)
    except Exception:
        return {}

    df_idx = df_idx[df_idx["run_id"].astype(str) == str(run_id)]
    if df_idx.empty:
        return {}

    if "sequence_safe" not in df_idx.columns:
        df_idx = df_idx.copy()
        df_idx["sequence_safe"] = (
            df_idx["sequence"]
            .fillna("")
            .apply(lambda v: to_safe_name(str(v)) if str(v).strip() else "")
        )

    match = df_idx[df_idx["sequence_safe"] == sequence_safe]
    if match.empty:
        return {}

    abs_path_str = match.iloc[0].get("abs_path", "")
    nn_path = (
        ds.resolve_path(abs_path_str)
        if hasattr(ds, "resolve_path")
        else Path(abs_path_str)
    )
    if not nn_path.exists():
        return {}

    try:
        df_nn = pd.read_parquet(nn_path)
    except Exception:
        return {}

    frame_col = (
        "frame"
        if "frame" in df_nn.columns
        else ("time" if "time" in df_nn.columns else None)
    )
    if frame_col is None or "id" not in df_nn.columns or "nn_id" not in df_nn.columns:
        return {}

    frames = df_nn[frame_col].to_numpy()
    ids = df_nn["id"].to_numpy()
    nn_ids = df_nn["nn_id"].to_numpy()

    lookup: dict[tuple[int, int], int] = {}
    for f, ind, nn in zip(frames, ids, nn_ids):
        if not np.isnan(nn):
            lookup[(int(f), int(ind))] = int(nn)

    return lookup


def _nn_pair_mask(
    df: pd.DataFrame,
    nn_lookup: dict[tuple[int, int], int],
) -> np.ndarray:
    """
    Return a boolean mask for rows in a pair-feature DataFrame where at
    least one individual in the pair considers the other its nearest neighbor.

    For pair (id1, id2) at frame F the row is kept when:
      - ``nn_lookup[(F, id1)] == id2``  (id1's NN is id2), **or**
      - ``nn_lookup[(F, id2)] == id1``  (id2's NN is id1).

    Returns all-True when the DataFrame lacks the required columns or
    when *nn_lookup* is empty (fail-open).
    """
    frame_col = (
        "frame" if "frame" in df.columns else ("time" if "time" in df.columns else None)
    )
    if (
        not nn_lookup
        or frame_col is None
        or "id1" not in df.columns
        or "id2" not in df.columns
    ):
        return np.ones(len(df), dtype=bool)

    frames = df[frame_col].to_numpy(dtype=int)
    id1s = df["id1"].to_numpy(dtype=int)
    id2s = df["id2"].to_numpy(dtype=int)

    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        f = int(frames[i])
        a = int(id1s[i])
        b = int(id2s[i])
        if nn_lookup.get((f, a)) == b or nn_lookup.get((f, b)) == a:
            mask[i] = True

    return mask


# -----------------------------------------------------------------------------
# StreamingFeatureHelper - Unified streaming processor for global features
# -----------------------------------------------------------------------------


class StreamingFeatureHelper:
    """
    Unified streaming processor for global features.

    Provides common functionality for:
    - Building file manifests without loading data
    - Streaming data loading with automatic memory cleanup
    - Progress logging
    - Scope filtering

    Usage:
        helper = StreamingFeatureHelper(ds, "my-feature")
        manifest = helper.build_manifest(inputs, scope_filter)
        for key, X, frames in helper.iter_sequences(manifest, extract_frames=True):
            # process X, frames
            pass
    """

    def __init__(self, ds, feature_name: str):
        """
        Initialize the streaming helper.

        Parameters
        ----------
        ds : Dataset
            Dataset instance
        feature_name : str
            Name of the feature (for logging)
        """
        self.ds = ds
        self.feature_name = feature_name
        self._seq_path_cache: dict[tuple[str, str], dict[Path, str]] = {}
        self._pair_filter_spec: dict | None = None
        self._nn_lookup_cache: dict[str, dict] = {}

    def set_pair_filter(self, pair_filter_spec: DictModel | dict | None) -> None:
        """Configure pair filtering (e.g. nearest-neighbor) for subsequent loads."""
        self._pair_filter_spec = pair_filter_spec

    def _get_nn_lookup(self, key: str) -> dict:
        """Build and cache the NN lookup for a sequence key."""
        if key in self._nn_lookup_cache:
            return self._nn_lookup_cache[key]
        lookup = (
            _build_nn_lookup(self.ds, key, self._pair_filter_spec)
            if self._pair_filter_spec
            else {}
        )
        self._nn_lookup_cache[key] = lookup
        return lookup

    def _make_df_filter(
        self, key: str
    ) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
        """Return a DataFrame filter callable for the given sequence, or None."""
        if self._pair_filter_spec is None:
            return None
        nn_lookup = self._get_nn_lookup(key)
        if not nn_lookup:
            return None

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            mask = _nn_pair_mask(df, nn_lookup)
            return df.loc[mask].reset_index(drop=True)

        return _filter

    def build_manifest_from_results(
        self,
        results: tuple[Result, ...],
        scope_filter: dict | None = None,
    ) -> dict[str, list[tuple[Path, LoadSpec]]]:
        """Build manifest from Result objects (per-frame parquet outputs)."""
        specs = [
            ArtifactSpec(feature=r.feature, run_id=r.run_id, load=ParquetLoadSpec())
            for r in results
        ]
        manifest = self.build_manifest(specs, scope_filter=scope_filter)
        if not manifest:
            labels = ", ".join(f"{r.feature} ({r.run_id})" for r in results)
            raise RuntimeError(
                f"[{self.feature_name}] build_manifest_from_results: "
                f"no per-sequence parquet files found for inputs: {labels}"
            )
        return manifest

    def build_manifest(
        self,
        inputs: list[ArtifactSpec],
        scope_filter: dict | None = None,
    ) -> dict[str, list[tuple[Path, LoadSpec]]]:
        """
        Build manifest of file paths per sequence WITHOUT loading data.

        Parameters
        ----------
        inputs : list[ArtifactSpec]
            List of typed artifact specifications.
        scope_filter : dict, optional
            Scope constraints with keys:
            - safe_sequences: set of allowed sequence names

        Returns
        -------
        dict[str, list[tuple[Path, LoadSpec]]]
            Mapping from sequence key to list of (path, load_spec) tuples
        """
        from mosaic.core.dataset import _feature_run_root, _latest_feature_run_root

        manifest: dict[str, list[tuple[Path, LoadSpec]]] = {}

        # Parse scope filter
        allowed_safe = None
        if scope_filter:
            safe_seqs = scope_filter.get("safe_sequences")
            if safe_seqs:
                allowed_safe = {str(s) for s in safe_seqs}

        for spec in inputs:
            feat_name = spec.feature
            run_id = spec.run_id
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(self.ds, feat_name)
            else:
                run_root = _feature_run_root(self.ds, feat_name, run_id)

            pattern = spec.pattern
            load_spec = spec.load
            files = sorted(run_root.glob(pattern))

            if not files:
                print(
                    f"[{self.feature_name}] WARN: no files for {feat_name} ({run_id}) pattern={pattern}",
                    file=sys.stderr,
                )
                continue

            seq_map = self._get_seq_map(feat_name, run_id)

            for pth in files:
                key = self._extract_key(pth, seq_map)
                if key is None:
                    continue
                if allowed_safe is not None and key not in allowed_safe:
                    continue
                if key not in manifest:
                    manifest[key] = []
                manifest[key].append((pth, load_spec))

        return manifest

    def iter_sequences(
        self,
        manifest: dict[str, list[tuple[Path, LoadSpec]]],
        extract_frames: bool = False,
        progress_interval: int = 10,
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray | None]]:
        """
        Iterate through manifest, yielding (key, X_combined, frames) one at a time.

        Automatically handles memory cleanup after each sequence.

        Parameters
        ----------
        manifest : dict[str, list[tuple[Path, dict]]]
            Manifest from build_manifest()
        extract_frames : bool, default False
            Whether to extract frame column from parquet files
        progress_interval : int, default 10
            Log progress every N sequences

        Yields
        ------
        tuple[str, np.ndarray, np.ndarray | None]
            (sequence_key, feature_matrix, frame_indices or None)
        """
        import pyarrow as pa

        n_keys = len(manifest)
        for i, (key, file_specs) in enumerate(manifest.items()):
            X, frames = self.load_key_data(
                file_specs, extract_frames=extract_frames, key=key
            )
            if X is None:
                continue

            yield key, X, frames

            # Memory cleanup after yielding
            del X
            if frames is not None:
                del frames
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % progress_interval == 0 or i == n_keys - 1:
                print(
                    f"[{self.feature_name}] Processed {i + 1}/{n_keys} sequences",
                    file=sys.stderr,
                )

    def load_key_data(
        self,
        file_specs: list[tuple[Path, LoadSpec]],
        extract_frames: bool = False,
        key: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Load and concatenate data for a single key.

        Parameters
        ----------
        file_specs : list[tuple[Path, dict]]
            List of (path, load_spec) tuples
        extract_frames : bool, default False
            Whether to extract frame column
        key : str, optional
            Sequence key (needed for pair filtering via NN lookup)

        Returns
        -------
        tuple[np.ndarray or None, np.ndarray or None]
            (combined_features, frame_indices or None)
        """
        import pyarrow as pa

        df_filter = self._make_df_filter(key) if key else None

        mats = []
        frames = None

        for pth, load_spec in file_specs:
            frame_col = "frame" if extract_frames else None
            arr, frame_vals = _load_array_from_spec(
                pth, load_spec, extract_frame_col=frame_col, df_filter=df_filter
            )
            if arr is None or arr.size == 0:
                continue
            mats.append(arr)
            if frames is None and frame_vals is not None:
                frames = frame_vals

        if not mats:
            return None, None

        T_min = min(m.shape[0] for m in mats)
        mats_trim = [m[:T_min] for m in mats]
        X_full = np.hstack(mats_trim)

        del mats, mats_trim
        gc.collect()
        pa.default_memory_pool().release_unused()

        if frames is not None and len(frames) >= T_min:
            frames = frames[:T_min]
        elif frames is None or len(frames) < T_min:
            frames = np.arange(T_min, dtype=np.int64) if extract_frames else None

        return X_full, frames

    def _load_identity_from_spec(
        self,
        path: Path,
        load_spec: LoadSpec,
        df_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """
        Load identity metadata from a source file.

        Returns
        -------
        tuple
            (id1_values_or_None, id2_values_or_None, entity_level)
        """
        if not isinstance(load_spec, ParquetLoadSpec):
            return None, None, "global"

        df = pd.read_parquet(path)
        if df_filter is not None:
            df = df_filter(df)
            if df.empty:
                return None, None, "global"

        id1, id2, level = _normalize_identity_columns(df)
        if id1 is None:
            return None, None, "global"
        id1_vals = id1.to_numpy(dtype=np.float64, copy=True)
        id2_vals = (
            id2.to_numpy(dtype=np.float64, copy=True)
            if id2 is not None
            else np.full(len(df), np.nan, dtype=np.float64)
        )
        return id1_vals, id2_vals, level

    def load_key_data_with_identity(
        self,
        file_specs: list[tuple[Path, LoadSpec]],
        extract_frames: bool = False,
        key: str | None = None,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        str,
    ]:
        """
        Load and concatenate data for a key, preserving identity metadata.

        Returns
        -------
        tuple
            (combined_features, frame_indices, id1_values, id2_values, entity_level)
        """
        import pyarrow as pa

        df_filter = self._make_df_filter(key) if key else None
        mats = []
        frames = None
        id1_vals: np.ndarray | None = None
        id2_vals: np.ndarray | None = None
        entity_level = "global"

        for pth, load_spec in file_specs:
            frame_col = "frame" if extract_frames else None
            arr, frame_vals = _load_array_from_spec(
                pth, load_spec, extract_frame_col=frame_col, df_filter=df_filter
            )
            if arr is None or arr.size == 0:
                continue

            if id1_vals is None:
                cur_id1, cur_id2, cur_level = self._load_identity_from_spec(
                    pth, load_spec, df_filter=df_filter
                )
                if cur_id1 is not None:
                    id1_vals = cur_id1
                    id2_vals = cur_id2
                    entity_level = cur_level

            mats.append(arr)
            if frames is None and frame_vals is not None:
                frames = frame_vals

        if not mats:
            return None, None, None, None, "global"

        T_min = min(m.shape[0] for m in mats)
        mats_trim = [m[:T_min] for m in mats]
        X_full = np.hstack(mats_trim)

        del mats, mats_trim
        gc.collect()
        pa.default_memory_pool().release_unused()

        if frames is not None and len(frames) >= T_min:
            frames = frames[:T_min]
        elif frames is None or len(frames) < T_min:
            frames = np.arange(T_min, dtype=np.int64) if extract_frames else None

        if id1_vals is not None:
            id1_vals = id1_vals[:T_min]
            if id2_vals is None:
                id2_vals = np.full(T_min, np.nan, dtype=np.float64)
            else:
                id2_vals = id2_vals[:T_min]
        else:
            id2_vals = None

        return X_full, frames, id1_vals, id2_vals, entity_level

    def _get_seq_map(self, feature_name: str, run_id: str) -> dict[Path, str]:
        """Get cached sequence path mapping."""
        cache_key = (feature_name, str(run_id))
        if cache_key in self._seq_path_cache:
            return self._seq_path_cache[cache_key]
        mapping = _build_path_sequence_map(self.ds, feature_name, run_id)
        self._seq_path_cache[cache_key] = mapping
        return mapping

    def _extract_key(self, path: Path, seq_map: dict[Path, str]) -> str | None:
        """Extract sequence key from file path, or None if not a sequence file."""
        stem = path.stem
        # Try to extract from filename pattern: seq=<key>
        m = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
        if m:
            return m.group(1)
        # Fallback to sequence map
        key = seq_map.get(path.resolve())
        if key:
            return str(key)
        return None


# -----------------------------------------------------------------------------
# Shared helpers for global feature patterns
# -----------------------------------------------------------------------------


def _parse_scope_filter(
    scope: dict | None,
) -> tuple[set[str] | None, dict[str, tuple[str, str]]]:
    """
    Parse a scope filter dict into structured components.

    Parameters
    ----------
    scope : dict or None
        Scope filter with optional keys:
        - safe_sequences: iterable of allowed safe sequence names
        - pair_safe_map: {(group, sequence) -> safe_name} mapping

    Returns
    -------
    tuple
        (allowed_safe_sequences or None, pair_map: {safe_name -> (group, sequence)})
    """
    if not scope:
        return None, {}
    safe_sequences = scope.get("safe_sequences")
    allowed = {str(s) for s in safe_sequences} if safe_sequences else None
    pair_safe_map = scope.get("pair_safe_map")
    pair_map: dict[str, tuple[str, str]] = {}
    if pair_safe_map:
        for pair, safe in pair_safe_map.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pair_map[str(safe)] = (str(pair[0]), str(pair[1]))
    return allowed, pair_map


def _build_sequence_lookup(
    ds,
    allowed_safe_sequences: set[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Build a {safe_seq -> (group, sequence)} lookup from the tracks index.

    Parameters
    ----------
    ds : Dataset
        Dataset instance (must have get_root("tracks"))
    allowed_safe_sequences : set[str] or None
        If provided, only include sequences in this set

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping from safe sequence name to (group, sequence) tuple
    """
    lookup: dict[str, tuple[str, str]] = {}
    if ds is None:
        return lookup
    idx_path = ds.get_root("tracks") / "index.csv"
    if not idx_path.exists():
        return lookup
    try:
        df = pd.read_csv(idx_path)
    except Exception:
        return lookup
    if df.empty:
        return lookup
    if "group" in df.columns:
        df["group"] = df["group"].fillna("").astype(str)
    else:
        df["group"] = ""
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].fillna("").astype(str)
    else:
        df["sequence"] = ""
    if "sequence_safe" in df.columns:
        df["sequence_safe"] = df["sequence_safe"].fillna("").astype(str)
    else:
        df["sequence_safe"] = df["sequence"].apply(
            lambda v: to_safe_name(v) if v else ""
        )
    if allowed_safe_sequences:
        df = df[df["sequence_safe"].isin({str(s) for s in allowed_safe_sequences})]
    for _, row in df.iterrows():
        safe_seq = str(row.get("sequence_safe", "")).strip()
        if not safe_seq or safe_seq in lookup:
            continue
        lookup[safe_seq] = (str(row.get("group", "")), str(row.get("sequence", "")))
    return lookup


def _resolve_sequence_identity(
    safe_seq: str,
    pair_map: dict[str, tuple[str, str]],
    sequence_lookup: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """
    Map a safe sequence name to (group, sequence).

    Checks pair_map first, then sequence_lookup, falls back to ("", safe_seq).
    """
    if safe_seq in pair_map:
        return pair_map[safe_seq]
    if safe_seq in sequence_lookup:
        return sequence_lookup[safe_seq]
    return "", safe_seq


def _get_feature_run_root(
    ds,
    feature_name: str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """
    Resolve (run_id, run_root_path) for a feature.

    If run_id is None, picks the latest finished run.
    """
    from mosaic.core.dataset import _feature_run_root, _latest_feature_run_root

    if run_id is None:
        run_id, run_root = _latest_feature_run_root(ds, feature_name)
    else:
        run_root = _feature_run_root(ds, feature_name, str(run_id))
    return str(run_id), run_root


def _load_joblib_artifact(ds, artifact: ArtifactSpec) -> object:
    """
    Load a joblib artifact from a feature run root.

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    artifact : ArtifactSpec
        Typed artifact specification with feature, run_id, pattern, and load spec.

    Returns
    -------
    object
        The loaded object, or obj[key] if load spec has a key
    """
    import joblib

    if not isinstance(artifact.load, JoblibLoadSpec):
        raise ValueError(
            f"_load_joblib_artifact requires JoblibLoadSpec, "
            f"got {type(artifact.load).__name__}"
        )
    _, run_root = _get_feature_run_root(ds, artifact.feature, artifact.run_id)
    files = sorted(run_root.glob(artifact.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{artifact.pattern}' in {run_root}"
        )
    obj = joblib.load(files[0])
    return obj if artifact.load.key is None else obj[artifact.load.key]


def _load_artifact_matrix(ds, artifact: ArtifactSpec) -> np.ndarray:
    """
    Load a numeric matrix from a feature artifact (npz or parquet).

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    artifact : ArtifactSpec
        Typed artifact specification with feature, run_id, pattern, and load spec.

    Returns
    -------
    np.ndarray
        Loaded matrix as float32
    """
    _, run_root = _get_feature_run_root(ds, artifact.feature, artifact.run_id)
    files = sorted(run_root.glob(artifact.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{artifact.pattern}' in {run_root}"
        )

    loader = artifact.load
    kind = loader.kind
    transpose = bool(getattr(loader, "transpose", False))

    if kind == "npz":
        key = getattr(loader, "key", None)
        if not key:
            raise ValueError("artifact.load.kind='npz' requires 'key'")
        mats = []
        for fp in files:
            npz = np.load(fp, allow_pickle=True)
            if key not in npz.files:
                continue
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            if transpose:
                A = A.T
            mats.append(A.astype(np.float32, copy=False))
        if not mats:
            raise FileNotFoundError(
                f"No NPZ containing key '{key}' among: {[f.name for f in files]}"
            )
        return np.vstack(mats) if len(mats) > 1 else mats[0]
    elif kind == "parquet":
        dfs = []
        cols = getattr(loader, "columns", None)
        numeric_only = getattr(loader, "numeric_only", True)
        for fp in files:
            df = pd.read_parquet(fp)
            if cols:
                df = df[[c for c in cols if c in df.columns]]
            elif numeric_only:
                df = df.select_dtypes(include=[np.number])
                for mc in (
                    "frame",
                    "time",
                    "id",
                    "id1",
                    "id2",
                    "id_a",
                    "id_b",
                    "id_A",
                    "id_B",
                ):
                    if mc in df.columns:
                        df = df.drop(columns=[mc])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            if not df.empty:
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(
                f"No parquet files with usable numeric columns among: "
                f"{[f.name for f in files]}"
            )
        D = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        A = D.to_numpy(dtype=np.float32, copy=False)
        if transpose:
            A = A.T
        return A
    else:
        raise ValueError(f"Unsupported artifact load.kind='{kind}'")


def _build_index_row(
    safe_seq: str,
    group: str,
    sequence: str,
    output_path: Path,
    n_rows: int | None = None,
    dataset_root: Path | None = None,
) -> dict:
    """
    Build a standard index row dict for get_additional_index_rows().

    When *dataset_root* is provided, ``abs_path`` is stored relative to it
    (portable).  Otherwise falls back to the resolved absolute path.
    """
    safe_group = to_safe_name(group) if group else ""
    resolved = output_path.resolve()
    if dataset_root is not None:
        try:
            path_str = str(resolved.relative_to(dataset_root))
        except ValueError:
            path_str = str(resolved)
    else:
        path_str = str(resolved)
    return {
        "group": group,
        "sequence": sequence,
        "group_safe": safe_group,
        "sequence_safe": safe_seq,
        "abs_path": path_str,
        "n_rows": n_rows,
    }
