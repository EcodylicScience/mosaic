from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._utils import ResolvedInput, Scope
from .index import (
    feature_index,
    feature_index_path,
    latest_feature_run_root,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

    from .types import InputsLike


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


# --- Feature data iteration ---


def yield_feature_data(
    ds: Dataset,
    feature_name: str,
    run_id: str | None = None,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """
    Yield (group, sequence, df) from a prior feature's saved outputs.
    If run_id is None, pick the most recent finished run_id for that feature (by finished_at).
    """
    idx_path = feature_index_path(ds, feature_name)
    idx = feature_index(idx_path)
    df_idx = idx.read()

    if run_id is None:
        run_id = idx.latest_run_id()

    df_idx = df_idx[df_idx["run_id"] == run_id]
    if df_idx.empty:
        raise ValueError(
            f"No entries for feature '{feature_name}' with run_id '{run_id}'."
        )

    df_idx = _filter_index(df_idx, groups, sequences, allowed_pairs)

    for _, row in df_idx.iterrows():
        g, s = row["group"], row["sequence"]
        p = Path(row["abs_path"])
        if p.suffix != ".parquet":
            msg = f"Expected .parquet in feature index, got: {p}"
            raise ValueError(msg)
        df = pd.read_parquet(p)
        # Skip marker tiny tables (<= 1 row or < 2 numeric cols)
        if len(df) <= 1:
            continue
        if df.select_dtypes(include=[np.number]).shape[1] < 2:
            continue
        yield g, s, df


# --- Entry resolution ---


def resolve_tracks_entries(
    ds: Dataset,
    groups_set: set[str] | None,
    seq_set: set[str] | None,
) -> set[tuple[str, str]]:
    """Resolve (group, sequence) entries from tracks/index.csv."""
    df_idx = _filter_index(_read_tracks_index(ds), groups_set, seq_set)
    return set(zip(df_idx["group"], df_idx["sequence"]))


def resolve_feature_entries(
    ds: Dataset,
    feat_name: str,
    run_id: str | None,
    groups_set: set[str] | None,
    seq_set: set[str] | None,
    columns: list[str] | None = None,
) -> tuple[set[tuple[str, str]], ResolvedInput]:
    """Resolve entries and resolved input for a feature run."""
    if run_id is None:
        try:
            run_id, _ = latest_feature_run_root(ds, feat_name)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to resolve latest run for feature '{feat_name}': {exc}"
            ) from exc
    idx_path = feature_index_path(ds, feat_name)
    df_idx = feature_index(idx_path).read()
    df_idx = df_idx[df_idx["run_id"] == run_id]
    # Drop global marker rows (written by run.py for global-only features)
    df_idx = df_idx[df_idx["sequence"] != "__global__"]

    df_idx = _filter_index(df_idx, groups_set, seq_set)

    entries = set(zip(df_idx["group"], df_idx["sequence"]))
    path_map: dict[tuple[str, str], Path] = {}
    for _, row in df_idx.iterrows():
        p = Path(row["abs_path"])
        if p.suffix != ".parquet":
            msg = f"Expected .parquet in feature index, got: {p}"
            raise ValueError(msg)
        path_map[(row["group"], row["sequence"])] = p

    resolved = ResolvedInput(
        kind="feature",
        feature=feat_name,
        run_id=run_id,
        path_map=path_map,
        columns=columns,
    )
    return entries, resolved


# --- Input resolution ---


def _resolve_input_specs(
    ds: Dataset,
    specs: list[tuple[str, str | None, str | None, list[str] | None]],
    groups_set: set[str] | None,
    seq_set: set[str] | None,
    label: str = "",
) -> tuple[set[tuple[str, str]], list[ResolvedInput]]:
    """Resolve a list of (kind, feature, run_id, columns) specs into entries and resolved inputs."""
    per_input_entries: list[set[tuple[str, str]]] = []
    resolved_inputs: list[ResolvedInput] = []

    for kind, feat_name, run_id, columns in specs:
        if kind == "tracks":
            entries = resolve_tracks_entries(ds, groups_set, seq_set)
            if not entries and label:
                print(
                    f"[{label}] WARN: tracks spec has no data matching the requested scope.",
                    file=sys.stderr,
                )
            per_input_entries.append(entries)
            resolved_inputs.append(ResolvedInput(kind="tracks", columns=columns))
            continue

        if not feat_name:
            continue

        entries, resolved = resolve_feature_entries(
            ds,
            feat_name,
            run_id,
            groups_set,
            seq_set,
            columns=columns,
        )

        if not entries and label:
            print(
                f"[{label}] WARN: feature '{feat_name}' run '{resolved.run_id}' has no data matching the requested scope.",
                file=sys.stderr,
            )
        per_input_entries.append(entries)
        resolved_inputs.append(resolved)

    if not per_input_entries:
        raise ValueError(
            f"No usable inputs resolved{' for ' + label if label else ''}."
        )

    allowed_entries = set.intersection(*per_input_entries)
    if not allowed_entries:
        raise ValueError(
            f"No overlapping sequences{' for ' + label if label else ''} in the requested scope."
        )

    return allowed_entries, resolved_inputs


def resolve_input_scope(
    ds: Dataset,
    inputs: InputsLike,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
) -> tuple[Scope, list[ResolvedInput]]:
    """Resolve an Inputs collection into a Scope and resolved inputs.

    Returns (scope, resolved_inputs) -- scope holds the filtered entries,
    resolved_inputs holds the data source references for iteration.
    """
    groups_set = set(groups) if groups is not None else None
    seq_set = set(sequences) if sequences is not None else None

    specs: list[tuple[str, str | None, str | None, list[str] | None]] = []
    for item in inputs.root:
        if item == "tracks":
            specs.append(("tracks", None, None, None))
        else:
            specs.append(("feature", item.feature, item.run_id, None))

    allowed_entries, resolved_inputs = _resolve_input_specs(
        ds,
        specs,
        groups_set,
        seq_set,
    )
    return Scope(entries=allowed_entries), resolved_inputs


def yield_input_data(
    ds: Dataset,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    *,
    entries: set[tuple[str, str]],
    resolved_inputs: list[ResolvedInput],
    metadata_only: bool = False,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    if not entries:
        return

    # Lightweight path: yield only routing columns (group, sequence, frame)
    # for features that load their own data and don't need the full merge.
    if metadata_only:
        for g, s, df_tracks in yield_sequences(
            ds, groups, sequences, allowed_pairs=entries
        ):
            routing_cols = [
                c
                for c in ("frame", "time", "group", "sequence", "id")
                if c in df_tracks.columns
            ]
            yield g, s, df_tracks[routing_cols]
        return

    track_specs = [ri for ri in resolved_inputs if ri.kind == "tracks"]
    feat_specs = [ri for ri in resolved_inputs if ri.kind == "feature"]

    for g, s, df_tracks in yield_sequences(
        ds, groups, sequences, allowed_pairs=entries
    ):
        df = df_tracks
        # Apply track column filter if provided
        if track_specs:
            cols = track_specs[-1].columns
            if cols:
                keep = [c for c in cols if c in df.columns]
                df = df[keep]
        # Merge in feature specs if any
        for spec in feat_specs:
            if not spec.path_map:
                continue
            pth = spec.path_map.get((g, s))
            if pth is None:
                continue
            try:
                if spec.columns:
                    merge_keys = {
                        "frame",
                        "time",
                        "id",
                        "group",
                        "sequence",
                        "id1",
                        "id2",
                    }
                    read_cols = list(set(spec.columns) | merge_keys)
                    try:
                        import pyarrow.parquet as pq

                        available = set(pq.read_schema(pth).names)
                        read_cols = [c for c in read_cols if c in available]
                    except Exception:
                        read_cols = None  # fall back to reading all columns
                    df_feat = pd.read_parquet(pth, columns=read_cols)
                else:
                    df_feat = pd.read_parquet(pth)
            except Exception:
                continue
            if df_feat.empty:
                continue
            # Merge on shared meta columns
            on_cols = [
                c
                for c in ("frame", "time", "id", "group", "sequence")
                if c in df.columns and c in df_feat.columns
            ]
            if not on_cols:
                on_cols = [
                    c
                    for c in ("frame", "time")
                    if c in df.columns and c in df_feat.columns
                ]
            if not on_cols:
                raise ValueError(
                    f"Cannot merge tracks and feature '{spec.feature}' for ({g},{s}): "
                    f"no shared columns (tracks: {list(df.columns)}, "
                    f"feature: {list(df_feat.columns)})"
                )
            else:
                df = df.merge(df_feat, how="left", on=on_cols)

        yield g, s, df
