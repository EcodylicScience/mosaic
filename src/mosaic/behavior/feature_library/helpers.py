"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

from .spec import ArtifactSpec, JoblibLoadSpec, NpzLoadSpec, ParquetLoadSpec

if TYPE_CHECKING:
    from .spec import DictModel, LoadSpec, Result

import numpy as np
import pandas as pd

from mosaic.core.helpers import to_safe_name
from mosaic.core.pipeline._utils import Scope


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


@dataclass(frozen=True, slots=True)
class KeyData:
    """Result of loading and merging data for a single manifest key.

    Attributes
    ----------
    features : np.ndarray
        (N, D) feature matrix, float32.
    frames : np.ndarray
        (N,) frame indices, int64.
    id1 : np.ndarray | None
        (N,) first identity column, float64. None when global.
    id2 : np.ndarray | None
        (N,) second identity column, float64. None when global.
    entity_level : str
        One of "global", "individual", "pair".
    """

    features: np.ndarray
    frames: np.ndarray
    id1: np.ndarray | None
    id2: np.ndarray | None
    entity_level: str


def _load_parquet_dataframe(
    path: Path,
    load_spec: LoadSpec,
    df_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Load a parquet file as a full DataFrame, applying filter if given.

    Returns None for non-parquet specs or if the result is empty after
    filtering.
    """
    if not isinstance(load_spec, ParquetLoadSpec):
        return None
    df = pd.read_parquet(path)
    if df_filter is not None:
        df = df_filter(df)
    if df.empty:
        return None
    return df


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
    from mosaic.core.pipeline.index import (
        feature_index_path,
        feature_run_root,
        latest_feature_run_root,
    )

    feat_name = pair_filter_spec.get("feature", "nearest-neighbor")
    run_id = pair_filter_spec.get("run_id")

    try:
        if run_id is None:
            run_id, _ = latest_feature_run_root(ds, feat_name)
        else:
            feature_run_root(ds, feat_name, str(run_id))
    except (ValueError, FileNotFoundError):
        return {}

    idx_path = feature_index_path(ds, feat_name)
    if not idx_path.exists():
        return {}

    try:
        df_idx = pd.read_csv(idx_path)
    except Exception:
        return {}

    df_idx = df_idx[df_idx["run_id"].astype(str) == str(run_id)]
    if df_idx.empty:
        return {}

    match = df_idx[
        df_idx["sequence"].fillna("").apply(lambda v: to_safe_name(str(v)))
        == sequence_safe
    ]
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
        manifest = helper.build_manifest(inputs, scope)
        for key, kd in helper.iter_sequences(manifest):
            # process kd.features, kd.frames
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
        scope: Scope | None = None,
    ) -> dict[str, list[tuple[Path, LoadSpec]]]:
        """Build manifest from Result objects (per-frame parquet outputs)."""
        specs = [
            ArtifactSpec(feature=r.feature, run_id=r.run_id, load=ParquetLoadSpec())
            for r in results
        ]
        manifest = self.build_manifest(specs, scope=scope)
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
        scope: Scope | None = None,
    ) -> dict[str, list[tuple[Path, LoadSpec]]]:
        """
        Build manifest of file paths per sequence WITHOUT loading data.

        Parameters
        ----------
        inputs : list[ArtifactSpec]
            List of typed artifact specifications.
        scope : Scope, optional
            Resolved scope with entries to filter by.

        Returns
        -------
        dict[str, list[tuple[Path, LoadSpec]]]
            Mapping from sequence key to list of (path, load_spec) tuples
        """
        from mosaic.core.pipeline.index import feature_run_root, latest_feature_run_root

        manifest: dict[str, list[tuple[Path, LoadSpec]]] = {}

        allowed_keys = scope.entry_keys if scope and scope.entries else None

        for spec in inputs:
            feat_name = spec.feature
            run_id = spec.run_id
            if run_id is None:
                run_id, run_root = latest_feature_run_root(self.ds, feat_name)
            else:
                run_root = feature_run_root(self.ds, feat_name, run_id)

            pattern = spec.pattern
            load_spec = spec.load
            files = sorted(run_root.glob(pattern))

            if not files:
                print(
                    f"[{self.feature_name}] WARN: no files for {feat_name} ({run_id}) pattern={pattern}",
                    file=sys.stderr,
                )
                continue

            for pth in files:
                key = self._extract_key(pth)
                if allowed_keys is not None and key not in allowed_keys:
                    continue
                if key not in manifest:
                    manifest[key] = []
                manifest[key].append((pth, load_spec))

        return manifest

    def iter_sequences(
        self,
        manifest: dict[str, list[tuple[Path, LoadSpec]]],
        progress_interval: int = 10,
    ) -> Iterator[tuple[str, KeyData]]:
        """Iterate through manifest, yielding (key, KeyData) one at a time."""
        import pyarrow as pa

        n_keys = len(manifest)
        for i, (key, file_specs) in enumerate(manifest.items()):
            kd = self.load_key_data(file_specs, key=key)
            if kd is None:
                continue

            yield key, kd

            del kd
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % progress_interval == 0 or i == n_keys - 1:
                print(
                    f"[{self.feature_name}] Processed {i + 1}/{n_keys} sequences",
                    file=sys.stderr,
                )

    _ALIGN_COLS = ("frame", "time", "id", "id1", "id2")

    def load_key_data(
        self,
        file_specs: list[tuple[Path, LoadSpec]],
        key: str | None = None,
    ) -> KeyData | None:
        """Load and merge data for a single manifest key.

        For parquet inputs, merges on shared frame/identity columns
        (inner join) so that misaligned inputs are correctly aligned.
        For non-parquet inputs (npz), falls back to min-trim + hstack.

        Returns None when no usable data is found or when parquet inputs
        have no overlapping frames.
        """
        df_filter = self._make_df_filter(key) if key else None

        parquet_dfs: list[pd.DataFrame] = []
        array_mats: list[np.ndarray] = []
        array_frames: np.ndarray | None = None

        for path, load_spec in file_specs:
            df = _load_parquet_dataframe(path, load_spec, df_filter=df_filter)
            if df is not None:
                parquet_dfs.append(df)
            else:
                arr, frame_vals = _load_array_from_spec(
                    path, load_spec, extract_frame_col="frame", df_filter=df_filter
                )
                if arr is not None and arr.size > 0:
                    array_mats.append(arr)
                    if array_frames is None and frame_vals is not None:
                        array_frames = frame_vals

        if not parquet_dfs and not array_mats:
            return None

        if parquet_dfs:
            merged = self._merge_parquet_inputs(parquet_dfs)
            if merged is None or merged.empty:
                return None
            return self._keydata_from_merged(merged)

        return self._keydata_from_arrays(array_mats, array_frames)

    def _merge_parquet_inputs(
        self, dfs: list[pd.DataFrame]
    ) -> pd.DataFrame | None:
        """Merge multiple parquet DataFrames on shared alignment columns."""
        if len(dfs) == 1:
            return dfs[0]

        merged = dfs[0]
        for i, df_next in enumerate(dfs[1:], 1):
            on_cols = [
                c for c in self._ALIGN_COLS
                if c in merged.columns and c in df_next.columns
            ]
            if not on_cols:
                on_cols = [
                    c for c in ("frame", "time")
                    if c in merged.columns and c in df_next.columns
                ]
            if not on_cols:
                # No shared alignment columns -- fall back to row-aligned concat
                min_len = min(len(merged), len(df_next))
                right = df_next.iloc[:min_len].reset_index(drop=True)
                # Suffix duplicate columns so they survive concat
                right_rename = {
                    c: f"{c}__{i}"
                    for c in right.columns if c in merged.columns
                }
                if right_rename:
                    right = right.rename(columns=right_rename)
                merged = pd.concat(
                    [merged.iloc[:min_len].reset_index(drop=True), right],
                    axis=1,
                )
                continue
            # Suffix duplicate feature columns so they survive the merge
            # (old hstack kept all columns regardless of name overlap)
            rename_map = {
                c: f"{c}__{i}"
                for c in df_next.columns
                if c not in on_cols and c in merged.columns
            }
            if rename_map:
                df_next = df_next.rename(columns=rename_map)
            # Drop non-feature meta columns that already exist in merged
            meta_dupes = [
                c for c in df_next.columns
                if c not in on_cols
                and c in merged.columns
                and c not in rename_map.values()
            ]
            if meta_dupes:
                df_next = df_next.drop(columns=meta_dupes)
            merged = merged.merge(df_next, how="inner", on=on_cols)

        if merged.empty:
            return None
        return merged

    def _keydata_from_merged(self, df: pd.DataFrame) -> KeyData:
        """Extract KeyData fields from a merged DataFrame."""
        id1_series, id2_series, entity_level = _normalize_identity_columns(df)

        if "frame" in df.columns:
            frames = df["frame"].to_numpy(dtype=np.int64, copy=True)
        elif "time" in df.columns:
            frames = df["time"].to_numpy(dtype=np.int64, copy=True)
        else:
            frames = np.arange(len(df), dtype=np.int64)

        meta_cols = {
            "frame", "time", "id", "id1", "id2",
            "id_a", "id_b", "id_A", "id_B",
            "group", "sequence", "entity_level",
        }
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        feat_cols = [
            c for c in df.columns
            if c not in meta_cols and c in numeric_cols
        ]

        features = df[feat_cols].to_numpy(dtype=np.float32, copy=True)

        id1 = (
            id1_series.to_numpy(dtype=np.float64, copy=True)
            if id1_series is not None else None
        )
        id2 = (
            id2_series.to_numpy(dtype=np.float64, copy=True)
            if id2_series is not None else None
        )

        return KeyData(
            features=features,
            frames=frames,
            id1=id1,
            id2=id2,
            entity_level=entity_level,
        )

    @staticmethod
    def _keydata_from_arrays(
        mats: list[np.ndarray],
        frames: np.ndarray | None,
    ) -> KeyData:
        """Build KeyData from raw numpy arrays (npz fallback)."""
        import pyarrow as pa

        T_min = min(m.shape[0] for m in mats)
        trimmed = [m[:T_min] for m in mats]
        features = np.hstack(trimmed)

        del mats, trimmed
        gc.collect()
        pa.default_memory_pool().release_unused()

        if frames is not None and len(frames) >= T_min:
            frames = frames[:T_min]
        else:
            frames = np.arange(T_min, dtype=np.int64)

        return KeyData(
            features=features,
            frames=frames,
            id1=None,
            id2=None,
            entity_level="global",
        )

    def _extract_key(self, path: Path) -> str:
        """Extract entry key from file path (the bare stem)."""
        return path.stem


# -----------------------------------------------------------------------------
# Shared helpers for global feature patterns
# -----------------------------------------------------------------------------


def _resolve_sequence_identity(
    entry_key: str,
    entry_map: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Map an entry key to (group, sequence).

    Looks up in entry_map, falls back to ("", entry_key).
    """
    if entry_key in entry_map:
        return entry_map[entry_key]
    return "", entry_key


def _get_feature_run_root(
    ds,
    feature_name: str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """
    Resolve (run_id, run_root_path) for a feature.

    If run_id is None, picks the latest finished run.
    """
    from mosaic.core.pipeline.index import feature_run_root, latest_feature_run_root

    if run_id is None:
        run_id, run_root = latest_feature_run_root(ds, feature_name)
    else:
        run_root = feature_run_root(ds, feature_name, str(run_id))
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
        raise FileNotFoundError(f"No files matching '{artifact.pattern}' in {run_root}")
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
        raise FileNotFoundError(f"No files matching '{artifact.pattern}' in {run_root}")

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


@dataclass(frozen=True, slots=True)
class PartialIndexRow:
    """Partial index row from global features (group/sequence/path/n_rows).

    Produced by _build_index_row(), consumed by _append_external_row()
    in run.py which fills in the remaining FeatureIndexRow fields.
    """

    group: str
    sequence: str
    abs_path: str
    n_rows: int = 0


def _build_index_row(
    group: str,
    sequence: str,
    output_path: Path,
    n_rows: int = 0,
) -> PartialIndexRow:
    """Build a partial index row for get_additional_index_rows()."""
    return PartialIndexRow(
        group=group,
        sequence=sequence,
        abs_path=str(output_path.resolve()),
        n_rows=n_rows,
    )
