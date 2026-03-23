from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Helpers ---


def _read_tracks_index(ds: Dataset) -> pd.DataFrame:
    """Read tracks/index.csv with keep_default_na=False."""
    idx_path = ds.get_root("tracks") / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError("tracks/index.csv not found; run conversion first.")
    return pd.read_csv(idx_path, keep_default_na=False)


def _filter_index(
    df_idx: pd.DataFrame,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Filter index rows by group, sequence, and/or allowed (group, sequence) pairs.

    TODO: Move to IndexCSV once tracks/index.csv is migrated to IndexCSV.
    """
    mask = pd.Series(True, index=df_idx.index)
    if groups is not None:
        mask &= df_idx["group"].isin(set(groups))
    if sequences is not None:
        mask &= df_idx["sequence"].isin(set(sequences))
    if allowed_pairs is not None:
        pair_mask = pd.Series(
            [
                (row["group"], row["sequence"]) in allowed_pairs
                for _, row in df_idx.iterrows()
            ],
            index=df_idx.index,
        )
        mask &= pair_mask
    return df_idx[mask]


# --- Sequence iteration ---


def yield_sequences(
    ds: Dataset,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """
    Yield (group, sequence, df) for standardized tracks present in tracks/index.csv,
    filtered by groups and/or sequences if provided.
    """
    df_idx = _filter_index(_read_tracks_index(ds), groups, sequences, allowed_pairs)

    for _, row in df_idx.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        p = ds.resolve_path(row["abs_path"])
        if not p.exists():
            raise FileNotFoundError(f"Stale tracks index: ({g},{s}) -> {p}")
        yield g, s, pd.read_parquet(p)


def yield_sequences_with_overlap(
    ds: Dataset,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
    overlap_frames: int = 0,
) -> Iterator[tuple[str, str, pd.DataFrame, int, int]]:
    """
    Yield (group, sequence, df, df_core_start, df_core_end) with optional overlap from adjacent sequences.

    For continuous datasets, this loads frames from neighboring segments to handle
    edge effects in rolling windows, smoothing, etc.

    Parameters
    ----------
    ds : Dataset
        The dataset instance
    groups : Iterable[str], optional
        Filter to specific groups
    sequences : Iterable[str], optional
        Filter to specific sequences
    allowed_pairs : set[tuple[str, str]], optional
        Filter to specific (group, sequence) pairs
    overlap_frames : int, default 0
        Number of frames to load from adjacent segments.
        If > 0, loads `overlap_frames` from the end of the previous segment
        and from the start of the next segment.

    Yields
    ------
    tuple[str, str, pd.DataFrame, int, int]
        (group, sequence, df_with_overlap, core_start_idx, core_end_idx)

        - df_with_overlap: DataFrame containing the main sequence plus overlap
        - core_start_idx: Index where the main sequence starts (after prefix overlap)
        - core_end_idx: Index where the main sequence ends (before suffix overlap)

        The caller can use these indices to trim output back to the original segment.

    Examples
    --------
    >>> for g, s, df, start, end in yield_sequences_with_overlap(ds, overlap_frames=300):
    ...     # df contains: [prev_300_frames] + [main_sequence] + [next_300_frames]
    ...     # Compute features on full df for continuity
    ...     features = compute_rolling_average(df)
    ...     # Trim to original segment for output
    ...     features_trimmed = features.iloc[start:end]
    """
    if overlap_frames <= 0:
        for g, s, df in yield_sequences(ds, groups, sequences, allowed_pairs):
            yield g, s, df, 0, len(df)
        return

    # Single index read for path lookup, adjacency, and filtering
    df_idx = _read_tracks_index(ds)

    # Build path lookup and sorted sequence list per group (for adjacency)
    path_lookup: dict[tuple[str, str], str] = {}
    seqs_by_group: dict[str, list[tuple[str, str]]] = {}
    for _, row in df_idx.sort_values(["group", "sequence"]).iterrows():
        pair = (row["group"], row["sequence"])
        path_lookup[pair] = row["abs_path"]
        seqs_by_group.setdefault(pair[0], []).append(pair)

    def load_parquet(group: str, seq: str) -> pd.DataFrame:
        abs_path = path_lookup.get((group, seq))
        if abs_path is None:
            raise KeyError(f"({group},{seq}) not in tracks index")
        p = ds.resolve_path(abs_path)
        if not p.exists():
            raise FileNotFoundError(f"Stale tracks index: ({group},{seq}) -> {p}")
        return pd.read_parquet(p)

    # Filter to requested scope
    df_filtered = _filter_index(df_idx, groups, sequences, allowed_pairs)

    for _, row in df_filtered.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        df_main = load_parquet(g, s)

        parts = []
        prefix_len = 0

        # Find adjacent sequences within the same group
        group_seqs = seqs_by_group.get(g, [])
        try:
            idx = next(
                i for i, (gg, ss) in enumerate(group_seqs) if gg == g and ss == s
            )
        except StopIteration:
            idx = -1

        # Load prefix overlap (last N frames of previous segment)
        if idx > 0:
            prev_g, prev_s = group_seqs[idx - 1]
            df_prev = load_parquet(prev_g, prev_s)
            if df_prev is not None and len(df_prev) > 0:
                n_take = min(overlap_frames, len(df_prev))
                parts.append(df_prev.iloc[-n_take:])
                prefix_len = n_take

        core_start = prefix_len
        parts.append(df_main)
        core_end = core_start + len(df_main)

        # Load suffix overlap (first N frames of next segment)
        if 0 <= idx < len(group_seqs) - 1:
            next_g, next_s = group_seqs[idx + 1]
            df_next = load_parquet(next_g, next_s)
            if df_next is not None and len(df_next) > 0:
                n_take = min(overlap_frames, len(df_next))
                parts.append(df_next.iloc[:n_take])

        if len(parts) == 1:
            df_combined = parts[0]
        else:
            df_combined = pd.concat(parts, ignore_index=True)

        yield g, s, df_combined, core_start, core_end
