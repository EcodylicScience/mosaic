"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mosaic.core.individuals import iter_individuals
from mosaic.core.pipeline.memory_watch import host_memory_bytes
from mosaic.core.pipeline.types import COLUMNS, META_COLS, NON_PAYLOAD_COLS

from .types import InterpolationConfig

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from pandas._typing import Scalar

__all__ = [
    "apply_exclude_cols",
    "block_rows",
    "check_memory_headroom",
    "chosen_payload_columns",
    "clean_animal_track",
    "clean_tracks_grouped",
    "consecutive_runs",
    "cv_block_assignment",
    "cv_block_bounds",
    "cv_fold_assignment",
    "ego_rotate",
    "ensure_columns",
    "entry_identity",
    "farthest_first",
    "feature_columns",
    "finite_keypoints",
    "finite_rows",
    "iter_series",
    "iter_series_positions",
    "label_individuals",
    "label_series",
    "meta_columns",
    "one_row_per_frame_and_id",
    "out_of_fold_choice",
    "pairwise_median_distance",
    "refuse_repeated_series_rows",
    "row_max",
    "row_median",
    "row_quantile",
    "scored_entry",
    "series_columns",
    "smooth_1d",
    "sort_nan_last",
    "train_entries_of_templates",
    "train_labeled_frames_of_templates",
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


def chosen_payload_columns(
    df: pd.DataFrame, columns: Sequence[str] | None, feature: str
) -> list[str]:
    """Return the payload columns of *df* that a step reads.

    Args:
        df: The step's merged input.
        columns: The names that the step's ``columns`` param lists. ``None``
            reads every payload column, as :func:`feature_columns` returns them.
        feature: The step's feature name, which the error names.

    Returns:
        *columns* in their order, or every payload column when it is ``None``.

    Raises:
        ValueError: A name is an identity or bookkeeping column, or is not a
            numeric column of *df*. The message lists the payload columns of
            *df*.
    """
    payload = feature_columns(df)
    if columns is None:
        return payload
    not_payload = [name for name in columns if name in NON_PAYLOAD_COLS]
    not_found = [
        name for name in columns if name not in NON_PAYLOAD_COLS and name not in payload
    ]
    if not_payload or not_found:
        reasons: list[str] = []
        if not_payload:
            reasons.append(f"Identity or bookkeeping columns: {not_payload}.")
        if not_found:
            reasons.append(f"Not a numeric column of the input: {not_found}.")
        listed = " ".join(reasons)
        msg = (
            f"[{feature}] columns may name only payload columns. {listed} "
            f"The input's payload columns: {payload}."
        )
        raise ValueError(msg)
    return list(columns)


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


# The columns of a pair row, in the order that a series is keyed and sorted by them.
_PAIR_ORDER = ("id1", "id2", "perspective")


def series_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns that key the series of *df*, in the order of their key.

    A series is the rows of one individual (the ``id`` column), of one ordered
    pair (``id1``, ``id2`` and ``perspective``, those of them *df* has), or the
    whole table when it has neither. A table with an ``id`` is read by
    individual whatever else it carries.
    """
    if COLUMNS.id_col in df.columns:
        return [COLUMNS.id_col]
    return [column for column in _PAIR_ORDER if column in df.columns]


def iter_series(
    df: pd.DataFrame, order_col: str = COLUMNS.frame_col
) -> Iterator[tuple[tuple[Scalar, ...], pd.DataFrame]]:
    """Yield the rows of each series in *df*, ordered by *order_col*.

    A model that reads rows in order, as an autoregressive one does, reads one
    animal's rows at a time. A pair table holds two rows per frame, one per
    perspective, so reading it as a single series would interleave the animals.

    Args:
        df: The rows to split. Left unchanged.
        order_col: The column that orders each series' rows.

    Yields:
        ``(key, rows)`` per series, the rows sorted by *order_col* and indexed
        from 0. The key holds the values of :func:`series_columns`, and is
        empty for a table without identity. A table with an ``id`` splits as
        :func:`~mosaic.core.individuals.iter_individuals` splits it, the ids in
        the order they first appear. A pair table's series come sorted by
        ``(id1, id2, perspective)``, whatever the order of its rows. A row
        whose key is missing belongs to no series and is not yielded.
    """
    columns = series_columns(df)
    if not columns or columns == [COLUMNS.id_col]:
        for individual, rows in iter_individuals(df, order_col):
            yield (() if individual is None else (individual,)), rows
        return
    for _, rows in df.groupby(columns, sort=True):
        rows = rows.sort_values(order_col, kind="stable").reset_index(drop=True)
        yield tuple(rows[name].iloc[0] for name in columns), rows


def iter_series_positions(
    df: pd.DataFrame, order_col: str = COLUMNS.frame_col
) -> Iterator[npt.NDArray[np.intp]]:
    """Yield the row positions of each series in *df*, ordered by *order_col*.

    The positional counterpart of :func:`iter_series`, for a caller that writes a
    result back to the rows it came from. The positions index *df* as it is, so
    ``df.iloc[positions]`` is a series in time order and an array filled at
    ``positions`` lines up with the input.

    Args:
        df: The rows to split. Left unchanged.
        order_col: The column that orders each series' rows.

    Yields:
        The positions of one series, sorted by *order_col* and, among equal
        values, by position. A table with neither ``id`` nor pair columns is
        one series. The series come sorted by their key, and a row whose key is
        missing is in none.
    """
    keys = series_columns(df)
    if keys:
        groups = [
            np.asarray(rows, dtype=np.intp)
            for rows in df.groupby(keys, sort=True).indices.values()
        ]
    else:
        groups = [np.arange(len(df), dtype=np.intp)]
    order = df[order_col].to_numpy()
    for rows in groups:
        yield rows[np.argsort(order[rows], kind="stable")]


def label_series(
    df: pd.DataFrame,
    label: Callable[[tuple[Scalar, ...], pd.DataFrame], npt.ArrayLike | None],
    column: str = "syllable",
) -> pd.DataFrame:
    """Label each series' rows of *df*, as :func:`iter_series` yields them.

    Args:
        df: The rows to label. Left unchanged.
        label: Called with each series' key and rows. It returns one label per
            row, or ``None`` to leave the series out of the result.
        column: The column of the labels.

    Returns:
        The frame and the label of each series' rows, the columns of
        :func:`series_columns`, so that the labels join back to the rows they
        describe, and the ``group`` and ``sequence`` of *df* when it has them, as
        the output of any other feature names its entry. The series come in the
        order of :func:`iter_series`, each in frame order. When no series is
        labeled the result has no rows, the frame and key columns keep the dtypes
        they have in *df*, and the label column is ``int64``, so that
        concatenating it with another result keeps their dtypes.
    """
    keys = [
        *series_columns(df),
        *(c for c in (COLUMNS.group_col, COLUMNS.seq_col) if c in df.columns),
    ]
    labeled: list[pd.DataFrame] = []
    for key, rows in iter_series(df):
        labels = label(key, rows)
        if labels is None:
            continue
        result = pd.DataFrame(
            {COLUMNS.frame_col: rows[COLUMNS.frame_col].to_numpy(), column: labels}
        )
        for name in keys:
            result[name] = rows[name].to_numpy()
        labeled.append(result)
    if not labeled:
        empty = df.iloc[:0][[COLUMNS.frame_col, *keys]].reset_index(drop=True)
        empty.insert(1, column, np.empty(0, dtype=np.int64))
        return empty
    return pd.concat(labeled, ignore_index=True)


def refuse_repeated_series_rows(
    df: pd.DataFrame, who: str, order_col: str = COLUMNS.frame_col
) -> None:
    """Refuse *df* when a series has two rows of one *order_col* value.

    The series are those of :func:`iter_series`. The two perspectives of a pair
    at one frame are two series, and not a repeat.

    Args:
        df: The rows to check.
        who: The feature that the refusal names.
        order_col: The column that orders each series' rows.

    Raises:
        ValueError: Naming *who*, the first repeated *order_col* value and its
            series.
    """
    keys = series_columns(df)
    read = df.dropna(subset=keys) if keys else df
    repeated = np.flatnonzero(read.duplicated([order_col, *keys]).to_numpy())
    if repeated.size == 0:
        return
    first = int(repeated[0])
    where = f"{order_col} {read[order_col].iloc[first]}"
    if keys:
        series = ", ".join(f"{name} {read[name].iloc[first]}" for name in keys)
        where += f" of {series}"
    msg = f"{who}: {where} appears in more than one row"
    raise ValueError(msg)


def consecutive_runs(
    frames: npt.ArrayLike, keep: npt.ArrayLike
) -> list[tuple[int, int]]:
    """Return the runs of consecutive frames that *keep* marks, as row positions.

    A model that reads a window of neighboring frames reads one animal's rows at
    a time, and only where each frame follows the one before it. A run ends at a
    gap in *frames* and at a row that *keep* leaves out.

    Args:
        frames: The frame number of each row of one series, in increasing order.
        keep: One flag per row. A row that is ``False`` belongs to no run.

    Returns:
        The ``(start, stop)`` positions of each run, in order, half-open as a
        slice reads them.
    """
    frame = np.asarray(frames)
    kept = np.asarray(keep, dtype=bool)
    if frame.size == 0:
        return []
    follows = np.zeros(frame.size, dtype=bool)
    follows[1:] = kept[1:] & kept[:-1] & (np.diff(frame) == 1)
    starts = np.flatnonzero(kept & ~follows)
    ends = np.zeros(frame.size, dtype=bool)
    ends[:-1] = ~follows[1:]
    ends[-1] = True
    stops = np.flatnonzero(kept & ends) + 1
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


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


def refuse_unlabeled_templates(templates: pd.DataFrame, feature: str) -> None:
    """Raise when a labeled templates table holds a row with a negative label.

    ``-1`` marks an unlabeled row (``mosaic.core.helpers.UNLABELED``), which a
    classifier would otherwise learn as a class of its own. extract-labeled-templates
    0.4 never samples one, so such a row comes from an older run or a table built by
    hand.

    Args:
        templates: The table, with a ``label`` column.
        feature: The consuming feature's name, for the message.
    """
    labels = templates["label"].to_numpy()
    n_unlabeled = int((labels < 0).sum())
    if n_unlabeled:
        msg = (
            f"[{feature}] {n_unlabeled} template rows have a negative label, which "
            f"marks a row with no label; rerun extract-labeled-templates (0.4 or "
            f"later), which samples labeled rows only"
        )
        raise ValueError(msg)


def train_entries_of_templates(templates_path: Path) -> list[tuple[str, str]] | None:
    """Return the train entries of the run that wrote *templates_path*.

    The provenance lists every entry the run saw, one that drew no template row
    included, which is why a fit that deals the entries reads it and not the rows.

    Args:
        templates_path: The ``templates.parquet`` of an ``extract-labeled-templates`` run.

    Returns:
        The ``(group, sequence)`` of each entry the run assigned to ``train``, or
        ``None`` when the run left no ``template_provenance.parquet`` beside the
        templates.
    """
    path = templates_path.parent / "template_provenance.parquet"
    if not path.exists():
        return None
    provenance = pd.read_parquet(path)
    train = provenance[provenance["split"] == "train"]
    return [(str(g), str(s)) for g, s in zip(train["group"], train["sequence"])]


def train_labeled_frames_of_templates(
    templates_path: Path,
) -> dict[tuple[str, str], tuple[int, int]] | None:
    """Return the labeled frames of each train entry of the run that wrote *templates_path*.

    A classifier that cross-validates by block cuts these frames into blocks, the
    same frames that ``sequence-classifier`` reads from its own labeled input.

    Args:
        templates_path: The ``templates.parquet`` of an ``extract-labeled-templates`` run.

    Returns:
        The ``(first, stop)`` of each train entry that has a labeled frame: the first
        frame of any of its series that a label covers, and one past the last. An
        entry with no label has none. ``None`` when the run left no provenance
        beside the templates, or a provenance from before ``extract-labeled-templates``
        0.7, which does not record them.
    """
    path = templates_path.parent / "template_provenance.parquet"
    if not path.exists():
        return None
    provenance = pd.read_parquet(path)
    if not {"labeled_start", "labeled_stop"} <= set(provenance.columns):
        return None
    train = provenance[
        (provenance["split"] == "train") & provenance["labeled_start"].notna()
    ]
    return {
        (str(g), str(s)): (int(first), int(stop))
        for g, s, first, stop in zip(
            train["group"],
            train["sequence"],
            train["labeled_start"],
            train["labeled_stop"],
            strict=True,
        )
    }


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


# --- Row-wise statistics over an individual's keypoints ---
#
# A feature that measures each (frame, id) by its keypoints computes every row at
# once on ``(n_rows, n_keypoints)`` arrays. A keypoint that is missing is NaN in
# both coordinates, so the statistics below read a sorted row, NaN last, and the
# count of entries before the NaN.

# Floats held in one block of rows, so that the temporaries of a block (a few arrays
# of that size) stay in the tens of megabytes however long the entry is.
_BLOCK_CELLS = 1 << 20


def block_rows(width: int) -> int:
    """Return how many rows of *width* floats to compute at once."""
    return max(1, _BLOCK_CELLS // max(width, 1))


def one_row_per_frame_and_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return the first row of each ``(frame, id)`` of *df*, ordered by frame, then id.

    A feature that measures an individual in a frame reads one row of it. The
    tracks guarantee one, and a repeat is ignored rather than averaged. A row
    with no frame or no id belongs to no individual and is left out.
    """
    keys = df[[COLUMNS.frame_col, COLUMNS.id_col]]
    first = ~keys.duplicated() & keys.notna().all(axis=1)
    return df[first].sort_values([COLUMNS.frame_col, COLUMNS.id_col], kind="stable")


def finite_keypoints(
    rows: pd.DataFrame, pose_pairs: Sequence[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y of every keypoint of *rows*, NaN in both where either is not finite.

    Args:
        rows: A table with the pose columns.
        pose_pairs: The ``(x_column, y_column)`` pairs, in keypoint order, as
            ``pose_column_pairs`` returns them.

    Returns:
        Two ``(len(rows), len(pose_pairs))`` float64 arrays. A keypoint with one
        coordinate missing is missing as a whole.
    """
    x = rows[[x_col for x_col, _ in pose_pairs]].to_numpy(dtype=float)
    y = rows[[y_col for _, y_col in pose_pairs]].to_numpy(dtype=float)
    missing = ~(np.isfinite(x) & np.isfinite(y))
    return np.where(missing, np.nan, x), np.where(missing, np.nan, y)


def sort_nan_last(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort each row of *values* ascending, NaN last, and count the entries that are not NaN."""
    ordered = np.sort(values, axis=1)
    return ordered, np.count_nonzero(~np.isnan(ordered), axis=1)


def _row_entry(ordered: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.take_along_axis(ordered, index[:, None], axis=1)[:, 0]


def row_max(ordered: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Return the largest of the first *count* entries of each sorted row, NaN for none."""
    return _row_entry(ordered, np.maximum(count - 1, 0))


def row_median(ordered: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Return the median of the first *count* entries of each sorted row, NaN for none."""
    last = np.maximum(count - 1, 0)
    middle = (_row_entry(ordered, last // 2) + _row_entry(ordered, (last + 1) // 2)) / 2
    return np.where(count > 0, middle, np.nan)


def row_quantile(ordered: np.ndarray, count: np.ndarray, q: float) -> np.ndarray:
    """Return the *q* quantile of the first *count* entries of each sorted row, NaN for none.

    Interpolates linearly between neighbours, as ``np.quantile`` does.

    Raises:
        ValueError: *q* is outside ``[0, 1]``.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantiles must be in the range [0, 1]")
    last = np.maximum(count - 1, 0)
    position = last * q
    below = np.floor(position).astype(np.intp)
    low = _row_entry(ordered, below)
    high = _row_entry(ordered, np.minimum(below + 1, last))
    gap = position - below
    step = high - low
    value = np.where(gap >= 0.5, high - step * (1 - gap), low + step * gap)
    return np.where(count > 0, value, np.nan)


def pairwise_median_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return, for each row, the median distance between its keypoints.

    Args:
        x: ``(n_rows, n_keypoints)`` x coordinates, NaN where a keypoint is
            missing, as :func:`finite_keypoints` returns them.
        y: The same for y.

    Returns:
        The ``(n_rows,)`` medians over the keypoints present, NaN for a row with
        fewer than two.
    """
    n_rows, n_keypoints = x.shape
    median = np.full(n_rows, np.nan)
    first, second = np.triu_indices(n_keypoints, k=1)
    step = block_rows(first.size)
    for lo in range(0, n_rows if first.size else 0, step):
        dx = x[lo : lo + step, first] - x[lo : lo + step, second]
        dy = y[lo : lo + step, first] - y[lo : lo + step, second]
        ordered, count = sort_nan_last(np.sqrt(dx * dx + dy * dy))
        median[lo : lo + step] = row_median(ordered, count)
    return median


# --- Memory preflight for a fit that holds its training set ---


def check_memory_headroom(
    estimated_bytes: int, feature: str, what: str, advice: str
) -> None:
    """Raise when *estimated_bytes* would leave less than a fifth of the available memory.

    A fit that holds its whole training set, and copies of it, fails late and
    takes the host with it. This asks first. Where the host reports no memory,
    as without ``psutil``, it passes.

    Args:
        estimated_bytes: The peak the work adds to what the process holds.
        feature: The feature that refuses, which the message names.
        what: What the estimate is of, such as ``"fitting 6 models on 143,000
            rows"``.
        advice: What the user can change, in a sentence that ends the message.

    Raises:
        MemoryError: The estimate exceeds 80% of the available memory.
    """
    _, available = host_memory_bytes()
    if available <= 0 or estimated_bytes <= available * 0.8:
        return
    msg = (
        f"[{feature}] {what} would take about {estimated_bytes / 1024**3:.1f} GB "
        f"at its peak, and only {available / 1024**3:.1f} GB of memory is "
        f"available. {advice}"
    )
    raise MemoryError(msg)


# --- Cross-validation by sequence ---
#
# A classifier that holds each training sequence out of the model that scores it deals
# the sequences into folds once, at its fit, and chooses a fold's model for each entry it
# scores. The two halves live here so that every classifier deals and chooses alike.


def _cv_entry_key(seed: int, group: str, sequence: str) -> str:
    """Return the sort key of an entry: the sha256 of the JSON text ``[seed, group, sequence]``.

    JSON keeps a group and a sequence apart (``("a", "bc")`` and ``("ab", "c")``
    differ, and a blank group stays a blank string), and unlike ``hash()`` it is
    the same in every process.
    """
    text = json.dumps([seed, group, sequence], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cv_fold_assignment(
    entries: Iterable[tuple[str, str]], n_folds: int, seed: int = 0
) -> dict[tuple[str, str], int]:
    """Deal entries into *n_folds* folds, balanced in their count.

    The entries are ordered by the sha256 of ``[seed, group, sequence]`` and dealt
    out in turn, so entry ``rank`` goes to fold ``rank % n_folds``. Folds differ by
    at most one entry, and the result depends on the entries and the seed alone:
    not on the order they arrive in, the process, or the rows each drew.

    Args:
        entries: The ``(group, sequence)`` of every entry to deal. A repeated
            entry is dealt once.
        n_folds: The number of folds, at least 2 and at most the number of
            entries.
        seed: Changes the folds and nothing else.

    Returns:
        The fold, from 0, of each entry.

    Raises:
        ValueError: *n_folds* is below 2 or above the number of entries.
    """
    unique = sorted(set(entries))
    if n_folds < 2:
        msg = f"cross-validation needs at least 2 folds, not {n_folds}"
        raise ValueError(msg)
    if n_folds > len(unique):
        msg = (
            f"cannot deal {len(unique)} train entries into {n_folds} folds. "
            f"Lower the number of folds to at most {len(unique)}"
        )
        raise ValueError(msg)
    ranked = sorted(unique, key=lambda entry: _cv_entry_key(seed, *entry))
    return {entry: rank % n_folds for rank, entry in enumerate(ranked)}


def cv_block_bounds(first: int, stop: int, n_blocks: int) -> list[tuple[int, int, int]]:
    """Cut the frames ``first`` up to ``stop`` into *n_blocks* contiguous, near-equal blocks.

    Block ``i`` of ``n_blocks`` spans ``first + i * span // n_blocks`` up to
    ``first + (i + 1) * span // n_blocks``, with ``span = stop - first``, so the
    blocks tile the range exactly and differ in length by at most one frame. A range
    shorter than *n_blocks* has fewer non-empty blocks: an empty one is left out, and
    the others keep their index.

    Args:
        first: The first frame of the range.
        stop: One past the last frame of the range.
        n_blocks: The number of blocks, at least 1.

    Returns:
        ``(index, start, stop)`` of each non-empty block, in order, half-open in
        the frame number.
    """
    span = stop - first
    edges = [first + (i * span) // n_blocks for i in range(n_blocks)] + [stop]
    return [
        (index, edges[index], edges[index + 1])
        for index in range(n_blocks)
        if edges[index + 1] > edges[index]
    ]


def _cv_block_key(seed: int, group: str, sequence: str, block: int) -> str:
    """Return the sort key of a block: the sha256 of the JSON text ``[seed, group, sequence, block]``."""
    text = json.dumps([seed, group, sequence, block], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cv_block_assignment(
    blocks: Iterable[tuple[str, str, int]], n_folds: int, seed: int = 0
) -> dict[tuple[str, str, int], int]:
    """Deal blocks into *n_folds* folds, balanced in their count.

    The block counterpart of :func:`cv_fold_assignment`: the blocks are ordered by the
    sha256 of ``[seed, group, sequence, block]`` and dealt out in turn, so block
    ``rank`` goes to fold ``rank % n_folds``. A dataset with one training sequence
    therefore still has *n_folds* folds, and the result depends on the blocks and the
    seed alone, not on the order they arrive in or the process.

    Args:
        blocks: The ``(group, sequence, block index)`` of every block to deal. A
            repeated block is dealt once.
        n_folds: The number of folds, at least 2 and at most the number of blocks.
        seed: Changes the folds and nothing else.

    Returns:
        The fold, from 0, of each block.

    Raises:
        ValueError: *n_folds* is below 2 or above the number of blocks.
    """
    unique = sorted(set(blocks))
    if n_folds < 2:
        msg = f"cross-validation needs at least 2 folds, not {n_folds}"
        raise ValueError(msg)
    if n_folds > len(unique):
        msg = (
            f"cannot deal {len(unique)} blocks into {n_folds} folds. Lower the "
            f"number of folds to at most {len(unique)}, or cut each sequence "
            "into more blocks"
        )
        raise ValueError(msg)
    ranked = sorted(unique, key=lambda block: _cv_block_key(seed, *block))
    return {block: rank % n_folds for rank, block in enumerate(ranked)}


def scored_entry(df: pd.DataFrame, feature: str) -> tuple[str, str]:
    """Return the ``(group, sequence)`` of the one entry that *df* holds, for cross-validation.

    The entry is read from the ``group`` and ``sequence`` columns by
    :func:`entry_identity`, the function that names an entry in the templates. A
    table without those columns would read as the entry ``("", "")``, which is in
    no fold, and would be scored by the full model that was fitted on its own
    sequence. It is refused instead.

    Args:
        df: One entry's table.
        feature: The feature that scores, which the message names.

    Raises:
        ValueError: *df* lacks ``group`` or ``sequence``, or holds more than one entry.
    """
    missing = [
        column for column in (COLUMNS.group_col, COLUMNS.seq_col) if column not in df
    ]
    if missing:
        msg = (
            f"[{feature}] cross-validation names the fold of an entry by its "
            f"group and sequence, and the input lacks {missing}"
        )
        raise ValueError(msg)
    if len(df[[COLUMNS.group_col, COLUMNS.seq_col]].drop_duplicates()) > 1:
        msg = (
            f"[{feature}] cross-validation scores one entry at a time, and the "
            "input holds more than one entry"
        )
        raise ValueError(msg)
    return entry_identity(df)


def out_of_fold_choice(
    df: pd.DataFrame, fold_of: Mapping[tuple[str, str], int], feature: str
) -> int | None:
    """Return the fold whose model scores the entry *df* holds, ``None`` for the full model.

    An entry in a fold is scored by the model that was fitted without it. Any other
    entry, such as one of a test group, was in no fold's training rows and no
    model's, so the full model scores it. The entry is named by :func:`scored_entry`.

    Args:
        df: One entry's table.
        fold_of: The fold of each ``(group, sequence)`` that was dealt.
        feature: The feature that scores, which the message names.

    Raises:
        ValueError: *df* lacks ``group`` or ``sequence``, or holds more than one entry.
    """
    return fold_of.get(scored_entry(df, feature))
