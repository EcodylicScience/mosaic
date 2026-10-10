"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mosaic.core.individuals import iter_individuals
from mosaic.core.pipeline.types import COLUMNS, META_COLS, NON_PAYLOAD_COLS

from .types import InterpolationConfig

if TYPE_CHECKING:
    import numpy.typing as npt
    from pandas._typing import Scalar

__all__ = [
    "apply_exclude_cols",
    "clean_animal_track",
    "clean_tracks_grouped",
    "ego_rotate",
    "ensure_columns",
    "entry_identity",
    "farthest_first",
    "feature_columns",
    "finite_rows",
    "label_individuals",
    "meta_columns",
    "smooth_1d",
    "unwrap_diff",
    "wrap_angle",
]


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

    Excludes the identity columns (``META_COLS``) and the non-identity bookkeeping
    ones (``BOOKKEEPING_COLS``), which is ``NON_PAYLOAD_COLS``: the numeric
    payload that a ``Payload`` requirement names. Together with
    :func:`meta_columns` this partitions *df* by construction, which is the point:
    a feature that asks both questions cannot answer them with two different sets.
    """
    return sorted(set(df.select_dtypes(include="number").columns) - NON_PAYLOAD_COLS)


def finite_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the rows of *matrix* that are finite in every column.

    A global model reads a row as one vector, and a clustering or embedding
    library refuses a vector holding NaN or infinity. Its fit and its apply both
    read only the rows this mask keeps.
    """
    return np.isfinite(matrix).all(axis=1)


def _finite_squared_distances(pool: np.ndarray, row: int) -> np.ndarray:
    """The squared distance of each row of *pool* from *row*, over the columns finite in both."""
    difference = pool - pool[row]
    return np.where(np.isfinite(difference), difference**2, 0.0).sum(axis=1)


def farthest_first(
    pool: np.ndarray, n_select: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Select up to *n_select* distinct rows of *pool* by greedy farthest-point sampling.

    *rng* draws the first row from the rows with a finite value. Each next row
    is the unselected one farthest from the rows selected so far, by the
    squared distance over the columns finite in both rows. A non-finite value
    counts as missing, and a row containing one is compared on its other
    columns. A row with no finite value is at distance 0 from every row. It is
    selected only when every unselected row is at distance 0.

    Returns:
        The selected rows and their indices, for subsetting parallel arrays
        such as the entry labels.
    """
    n_rows = int(pool.shape[0])
    with_finite_value = np.flatnonzero(np.isfinite(pool).any(axis=1))
    candidates = with_finite_value if with_finite_value.size else np.arange(n_rows)
    first = int(candidates[rng.integers(0, candidates.size)])
    selected: list[int] = [first]
    distances = _finite_squared_distances(pool, first)
    distances[first] = -np.inf
    while len(selected) < min(n_select, n_rows):
        farthest = int(np.argmax(distances))
        selected.append(farthest)
        distances = np.minimum(distances, _finite_squared_distances(pool, farthest))
        distances[farthest] = -np.inf
    indices = np.array(selected)
    return pool[indices], indices


def label_individuals(
    df: pd.DataFrame,
    label: Callable[[Scalar | None, pd.DataFrame], npt.ArrayLike],
    column: str = "syllable",
) -> pd.DataFrame:
    """Label each individual's rows of *df*, as :func:`iter_individuals` yields them.

    Args:
        df: The rows to label. Left unchanged.
        label: Called with each individual's id and rows. It returns one label
            per row.
        column: The column of the labels.

    Returns:
        The frame and the label of each individual's rows, and the id when *df*
        has an id column. The individuals come in the order of
        :func:`iter_individuals`, each in frame order.
    """
    labeled: list[pd.DataFrame] = []
    for individual, rows in iter_individuals(df):
        result = pd.DataFrame(
            {
                COLUMNS.frame_col: rows[COLUMNS.frame_col].to_numpy(),
                column: label(individual, rows),
            }
        )
        if individual is not None:
            result[COLUMNS.id_col] = individual
        labeled.append(result)
    return pd.concat(labeled, ignore_index=True)


def entry_identity(df: pd.DataFrame) -> tuple[str, str]:
    """Return the group and the sequence of *df*'s first row, as text.

    A table without the column, or without rows, has ``""`` for it.
    """
    group = ""
    sequence = ""
    if not df.empty:
        if COLUMNS.group_col in df.columns:
            group = str(df[COLUMNS.group_col].iloc[0])
        if COLUMNS.seq_col in df.columns:
            sequence = str(df[COLUMNS.seq_col].iloc[0])
    return group, sequence


def meta_columns(df: pd.DataFrame) -> list[str]:
    """Return the sorted identity columns present in *df*.

    What a feature that rebuilds its output carries through unchanged. The
    complement of :func:`feature_columns` over the identity half, read from the
    pipeline's own ``META_COLS`` rather than spelled per feature -- five features
    spelled it separately and three of the spellings disagreed.
    """
    return sorted(META_COLS & set(df.columns))


# What to say when a missing column has a known producer. A tracks table records
# what the tracker measured; a heading is derived from keypoints, so no tracker
# reports one and no table carries it until a feature puts it there. Without this
# the error named the column and stopped, which is unhelpful precisely when the
# reason is "this moved out of the converters" rather than "your data is broken".
_PRODUCER_HINTS: dict[str, str] = {
    COLUMNS.orientation_col: (
        "no tracker reports a heading -- it is derived from keypoints. Run the "
        "'heading' feature first (choosing method='two_point' with front_idx/"
        "rear_idx, or method='pca'), or point orientation_col= at a column this "
        "table already has"
    ),
    "SPEED": (
        "speed is derived from position, not measured. Run the 'speed-angvel' "
        "feature first"
    ),
}


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing from *df*.

    Names the feature that produces a missing column when there is one, so a
    table that simply has not had a step run against it says so, rather than
    reading as a malformed table.
    """
    missing = sorted(set(required) - set(df.columns))
    if not missing:
        return
    lines = [f"Missing required columns: {missing}"]
    lines.extend(
        f"  {column}: {_PRODUCER_HINTS[column]}."
        for column in missing
        if column in _PRODUCER_HINTS
    )
    raise ValueError("\n".join(lines))


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
