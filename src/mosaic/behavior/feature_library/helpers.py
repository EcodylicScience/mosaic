"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import InterpolationConfig

__all__ = [
    "apply_exclude_cols",
    "clean_animal_track",
    "clean_tracks_grouped",
    "ego_rotate",
    "ensure_columns",
    "feature_columns",
    "smooth_1d",
    "unwrap_diff",
    "wrap_angle",
]


# Extra non-feature columns that may appear in result DataFrames alongside
# the standard COLUMNS metadata (id, sequence, group, frame, time).
_EXTRA_META = {"id1", "id2", "entity_level", "perspective", "fps", "label", "split"}


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


def clean_tracks_grouped(
    df: pd.DataFrame,
    group_cols: list[str],
    data_cols: list[str],
    order_col: str,
    config: InterpolationConfig,
) -> pd.DataFrame:
    """Clean tracks per group, preserving group columns in the result.

    Pandas 3.0 excludes group columns from ``groupby().apply()`` results.
    This wrapper uses ``group_keys=True`` and resets the index to restore them.
    """
    return (
        df.groupby(group_cols, group_keys=True)
        .apply(
            lambda g: clean_animal_track(g, data_cols, order_col, config),
            include_groups=False,
        )
        .reset_index(level=group_cols)
        .reset_index(drop=True)
    )


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
