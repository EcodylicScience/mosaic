"""Data loading, artifact resolution, and key-data helpers.

Moved from ``mosaic.behavior.feature_library.helpers`` to live closer to the
pipeline infrastructure they depend on.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from ._loaders import JoblibLoadSpec, LoadSpec, ParquetLoadSpec
from .index import (
    feature_index,
    feature_index_path,
    feature_run_root,
    latest_feature_run_root,
)
from .types import ArtifactSpec, NNResult

if TYPE_CHECKING:
    from ..dataset import Dataset

# TODO, also see below: This shares logic, and hardcodes columns defined in feature_library.params.COLUMNS
# so COLUMNS needs to move to core or pipeline, and we'd need to derive the sets dynamically from it
# should use the new Columns.meta_set() | {"id1", "id2"} - id1 and id2 should actually also move to Columns for consistency.
_ALIGN_COLS = frozenset({"frame", "time", "id", "id1", "id2"})

# TODO: this shares logic with feature_library.params.PoseConfig
# so pose config, columns etc should also move to pipeline/core
def pose_column_pairs(columns: Iterable[str]) -> list[tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs from column names."""
    pose_pairs: list[tuple[str, str]] = []
    xs = [c for c in columns if c.startswith("poseX")]
    for x_col in sorted(xs):
        idx = x_col[5:]
        y_col = f"poseY{idx}"
        if y_col in columns:
            pose_pairs.append((x_col, y_col))
    return pose_pairs


# is this really a valid approach? We may rather want to raise if not numeric but exists
# also, features decide whether they are "pair", "individual" or "global" internally, why do we need this here?
# just to write the __global__ marker row to the index?
def normalize_identity_columns(
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


def load_parquet_dataframe(
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


def _merge_parquet_inputs(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame | None:
    """Merge parquet DataFrames on shared alignment columns.

    Accepts an iterator -- DataFrames are consumed one at a time and
    merged incrementally, so only two are in memory at once.
    """
    it = iter(dfs)
    merged = next(it, None)
    if merged is None:
        return None

    for i, df_next in enumerate(it, 1):
        on_cols = _ALIGN_COLS & set(merged.columns) & set(df_next.columns)
        if not on_cols:
            msg = (
                f"Cannot merge input {i}: no shared alignment columns "
                f"({_ALIGN_COLS}) between merged ({list(merged.columns)}) "
                f"and input ({list(df_next.columns)})"
            )
            raise ValueError(msg)

        # Suffix duplicate feature columns so they survive the merge
        rename_map = {
            c: f"{c}__{i}"
            for c in df_next.columns
            if c not in on_cols and c in merged.columns
        }
        if rename_map:
            df_next = df_next.rename(columns=rename_map)
        merged = merged.merge(df_next, how="inner", on=list(on_cols))

    if merged.empty:
        return None
    return merged


def load_entry_data(
    file_specs: list[tuple[Path, LoadSpec]],
    filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
) -> pd.DataFrame | None:
    """Load and merge data for a single manifest entry.

    All manifest entries are parquet. Multiple inputs are merged via
    inner join on shared alignment columns. DataFrames are loaded
    lazily and merged incrementally to minimize peak memory.
    """

    def _load_dfs() -> Iterator[pd.DataFrame]:
        for path, load_spec in file_specs:
            df = load_parquet_dataframe(path, load_spec)
            if df is not None:
                yield df

    merged = _merge_parquet_inputs(_load_dfs())
    if merged is None or merged.empty:
        return None
    for fn in filters:
        merged = fn(merged)
        if merged.empty:
            return None
    return merged


def build_nn_lookup(
    ds: Dataset,
    group: str,
    sequence: str,
    pair_filter_spec: NNResult,
) -> dict[tuple[int, int], int]:
    """
    Build a per-frame nearest-neighbor lookup for a given sequence.

    Returns ``{(frame, individual_id): nn_id}`` from the nearest-neighbor
    feature parquet.  Returns an empty dict when the NN feature has not been
    computed for this sequence (fail-open).
    """

    feature_name = pair_filter_spec.feature
    idx = feature_index(feature_index_path(ds, feature_name))

    run_id = pair_filter_spec.run_id
    if run_id is None:
        run_id = idx.latest_run_id()

    idx_df = idx.read(run_id=run_id, filter_ext=".parquet", entries=[(group, sequence)])
    if idx_df.empty:
        return {}

    nn_path = idx_df.iloc[0]["abs_path"]
    df_nn = pd.read_parquet(nn_path)

    # below is probably too defensive, NNResult (as a per sequence result) has a known parquet schema
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


def nn_pair_mask(
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


def resolve_sequence_identity(
    entry_key: str,
    entry_map: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Map an entry key to (group, sequence).

    Looks up in entry_map, falls back to ("", entry_key).
    """
    if entry_key in entry_map:
        return entry_map[entry_key]
    return "", entry_key


def get_feature_run_root(
    ds: Dataset,
    feature_name: str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """
    Resolve (run_id, run_root_path) for a feature.

    If run_id is None, picks the latest finished run.
    """
    if run_id is None:
        run_id, run_root = latest_feature_run_root(ds, feature_name)
    else:
        run_root = feature_run_root(ds, feature_name, run_id)
    return str(run_id), run_root


def load_joblib_artifact(ds: Dataset, artifact: ArtifactSpec) -> object:
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
    if not isinstance(artifact.load, JoblibLoadSpec):
        raise ValueError(
            f"load_joblib_artifact requires JoblibLoadSpec, "
            f"got {type(artifact.load).__name__}"
        )
    _, run_root = get_feature_run_root(ds, artifact.feature, artifact.run_id)
    files = sorted(run_root.glob(artifact.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{artifact.pattern}' in {run_root}")
    return artifact.from_path(files[0])
