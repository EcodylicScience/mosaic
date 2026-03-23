"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import InterpolationConfig

__all__ = [
    "clean_animal_track",
    "ego_rotate",
    "ensure_columns",
    "feature_columns",
    "load_result_for",
    "nn_lookup_for",
    "smooth_1d",
    "unwrap_diff",
    "wrap_angle",
]


# Extra non-feature columns that may appear in result DataFrames alongside
# the standard COLUMNS metadata (id, sequence, group, frame, time).
_EXTRA_META = {"id1", "id2", "entity_level", "perspective", "fps"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of numeric feature column names in *df*.

    Excludes standard metadata columns (COLUMNS.meta_set()) and known
    non-feature columns (id1, id2, entity_level, perspective, fps).
    """
    from mosaic.core.pipeline.types import COLUMNS as C

    exclude = C.meta_set() | (_EXTRA_META & set(df.columns))
    return sorted(set(df.select_dtypes(include="number").columns) - exclude)


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing from *df*."""
    missing = set(required) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def nn_lookup_for(
    nn_index: pd.DataFrame | None, df: pd.DataFrame
) -> dict[tuple[int, int], int] | None:
    """Build NN lookup for the sequence in *df* from a dependency index.

    Returns a ``{(frame, id): nn_id}`` dict, or ``None`` if *nn_index* is not
    provided.  Used by features that apply a nearest-neighbor pair filter in
    their ``apply()`` method.
    """
    from mosaic.core.pipeline.types import COLUMNS as C

    if nn_index is None:
        return None
    group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
    sequence = str(df[C.seq_col].iloc[0])
    df_nn = load_result_for(nn_index, group, sequence)
    frames = df_nn[C.frame_col].to_numpy()
    ids = df_nn[C.id_col].to_numpy()
    nn_ids = df_nn["nn_id"].to_numpy()
    lookup: dict[tuple[int, int], int] = {}
    for f, ind, nn in zip(frames, ids, nn_ids):
        if not np.isnan(nn):
            lookup[(int(f), int(ind))] = int(nn)
    return lookup


def load_result_for(index: pd.DataFrame, group: str, sequence: str) -> pd.DataFrame:
    """Look up and read the upstream result parquet for a (group, sequence) pair.

    Filters *index* (a dependency index DataFrame with ``group``, ``sequence``,
    and ``abs_path`` columns) to exactly one row, then reads and returns the
    parquet at that path.

    Raises ``FileNotFoundError`` if no match is found, and ``ValueError`` if
    more than one row matches (ambiguous upstream data).
    """
    match = index[(index["group"] == group) & (index["sequence"] == sequence)]
    if match.empty:
        msg = f"No upstream result for group={group!r}, sequence={sequence!r}"
        raise FileNotFoundError(msg)
    if len(match) > 1:
        msg = (
            f"Ambiguous upstream result: {len(match)} rows match "
            f"group={group!r}, sequence={sequence!r}"
        )
        raise ValueError(msg)
    path = Path(str(match.iloc[0]["abs_path"]))
    return pd.read_parquet(path)


# --- Shared helpers for per-sequence features ---


def clean_animal_track(
    g: pd.DataFrame,
    data_cols: list[str],
    order_col: str,
    config: InterpolationConfig,
) -> pd.DataFrame:
    """Sort, interpolate, fill, and drop rows with excessive missing data."""
    g = g.sort_values(order_col).copy()
    g = g.set_index(order_col)
    g[data_cols] = g[data_cols].replace([np.inf, -np.inf], np.nan)
    g[data_cols] = g[data_cols].interpolate(
        method="linear",
        limit=config.linear_interp_limit,
        limit_direction="both",
    )
    g[data_cols] = g[data_cols].ffill(limit=config.edge_fill_limit)
    g[data_cols] = g[data_cols].bfill(limit=config.edge_fill_limit)
    miss_frac = g[data_cols].isna().mean(axis=1)
    g = g.loc[miss_frac <= config.max_missing_fraction].copy()
    if g[data_cols].isna().any().any():
        med = g[data_cols].median()
        g[data_cols] = g[data_cols].fillna(med)
    g = g.reset_index()
    return g


def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Moving average with reflected padding."""
    if win is None or win <= 1:
        return x
    pad = win // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    ker = np.ones(win, dtype=float) / float(win)
    return np.convolve(xp, ker, mode="valid")


def unwrap_diff(theta: np.ndarray, fps: float) -> np.ndarray:
    """Compute angular velocity from angle array."""
    d = np.gradient(np.unwrap(theta), edge_order=1)
    return d * float(fps)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def ego_rotate(
    dx: np.ndarray, dy: np.ndarray, heading: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world-frame deltas into ego frame (heading aligned with +x)."""
    ct = np.cos(heading)
    st = np.sin(heading)
    dx_ego = dx * ct + dy * st
    dy_ego = -dx * st + dy * ct
    return dx_ego, dy_ego
