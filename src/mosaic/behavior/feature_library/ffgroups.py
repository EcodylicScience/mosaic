from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from numba import njit  # pyright: ignore[reportUnknownVariableType]
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    TrackInput,
    resolve_order_col,
)

from .registry import register_feature

# --- Numba-accelerated union-find for connected components ---


@njit
def _union_find_root(parent: np.ndarray, i: int) -> int:
    """Find root with path compression."""
    if parent[i] != i:
        parent[i] = _union_find_root(parent, parent[i])
    return parent[i]


@njit
def _union_find_union(parent: np.ndarray, rank: np.ndarray, x: int, y: int) -> None:
    """Union by rank."""
    rx, ry = _union_find_root(parent, x), _union_find_root(parent, y)
    if rx != ry:
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1


@njit
def _connected_components_numba(adj_matrix: np.ndarray, n: int) -> np.ndarray:
    """Find connected components using union-find."""
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                _union_find_union(parent, rank, i, j)

    # Get component labels
    components = np.empty(n, dtype=np.int32)
    for i in range(n):
        components[i] = _union_find_root(parent, i)

    # Relabel to sequential 0, 1, 2...
    unique_roots = np.unique(components)
    label_map = np.zeros(n, dtype=np.int32)
    for new_label in range(len(unique_roots)):
        label_map[unique_roots[new_label]] = new_label

    for i in range(n):
        components[i] = label_map[components[i]]

    return components


@njit
def _calculate_gmembership_numba(
    pwdf: np.ndarray, nparticles: int, numsteps: int, threshold: float
) -> np.ndarray:
    """Vectorized group membership using Numba."""
    groupmembership = np.zeros((numsteps, nparticles), dtype=np.int32)

    for step in range(numsteps):
        # Build adjacency matrix from flattened pairwise distances
        adj = np.zeros((nparticles, nparticles), dtype=np.float32)
        for i in range(nparticles):
            for j in range(nparticles - 1):
                # pwdf shape is (T, N, N-1) - distances to other particles
                val = pwdf[step, i, j]
                # Map j back to actual particle index (skip diagonal)
                actual_j = j if j < i else j + 1
                if not np.isnan(val) and val < threshold:
                    adj[i, actual_j] = 1.0

        groupmembership[step] = _connected_components_numba(adj, nparticles)

    return groupmembership


# --- Event detection helpers ---


def _sorted_members(ids: pd.Series) -> tuple[int, ...]:
    """Aggregate group members into a sorted tuple of int IDs."""
    return tuple(sorted(ids.astype(int)))


def _find_events(
    df: pd.DataFrame,
    minimal_length: int,
    frame_col: str,
    id_col: str,
) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
    """Find contiguous events where stable subgroups persist."""
    unique_frames = np.unique(df[frame_col].astype(int))
    if len(unique_frames) == 0:
        return []

    # Split into runs of consecutive frames (gap > 1 starts a new run)
    splits = np.flatnonzero(np.diff(unique_frames) > 1) + 1

    finalized: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    for run in np.split(unique_frames, splits):
        start, end = int(run[0]), int(run[-1])
        if end - start < minimal_length:
            continue

        data = df.loc[(df[frame_col] >= start) & (df[frame_col] <= end)]

        # For each (frame, group_membership), compute sorted tuple of member IDs
        members = (
            data.groupby([frame_col, "group_membership"])[id_col]
            .agg(_sorted_members)
            .reset_index(name="members")
        )

        # For each unique member-set, find contiguous frame runs
        for _, grp in members.groupby("members"):
            member_key: tuple[int, ...] = grp["members"].iloc[0]
            member_frames = np.sort(grp[frame_col].to_numpy(dtype=int))
            member_splits = np.flatnonzero(np.diff(member_frames) > 1) + 1
            for member_run in np.split(member_frames, member_splits):
                duration = int(member_run[-1]) - int(member_run[0])
                if duration >= minimal_length:
                    finalized.append(
                        (member_key, (int(member_run[0]), int(member_run[-1])))
                    )

    return finalized


def _get_events_info(
    df: pd.DataFrame,
    threshold_ev_duration: int = 1,
    *,
    frame_col: str = C.frame_col,
    id_col: str = C.id_col,
) -> pd.DataFrame:
    """Label rows with event IDs for detected fission-fusion events.

    Returns a DataFrame containing only the rows that belong to an event,
    with an added ``"event"`` column holding the integer event label.
    """
    events = _find_events(
        df, minimal_length=threshold_ev_duration, frame_col=frame_col, id_col=id_col
    )
    if not events:
        return pd.DataFrame(columns=list(df.columns) + ["event"])

    frames = df[frame_col].to_numpy()
    ids = df[id_col].to_numpy()
    event_labels = np.full(len(df), -1, dtype=int)

    for event_id, (member_ids, (start, end)) in enumerate(events):
        id_set = set(member_ids)
        mask = (frames >= start) & (frames <= end) & np.isin(ids, list(id_set))
        event_labels[mask] = event_id

    in_event = event_labels >= 0
    result = df.loc[in_event].copy()
    result.insert(loc=1, column="event", value=event_labels[in_event])
    return result.reset_index(drop=True)


# --- Feature class ---


@final
@register_feature
class FFGroups:
    """
    Per-sequence fission-fusion grouping metrics.

    Inputs: raw tracks (columns: x, y, id, frame/time, group, sequence).
    Outputs per (frame, id):
      - group_membership (component label)
      - group_size (size of that component)
      - event (event id from dp.get_events_info, -1 if not in an event)

    Params:
        distance_cutoff: Pairwise distance threshold below which two
            animals are considered in the same group. Default: 50.0.
        window_size: Sliding-window size (frames) for smoothing the
            pairwise distance matrix before thresholding. Default: 5.
        min_event_duration: Minimum number of contiguous frames for a
            stable subgroup to be registered as an event. Default: 1.
    """

    name = "ffgroups"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        distance_cutoff: float = Field(default=50.0, gt=0)
        window_size: int = Field(default=5, ge=1)
        min_event_duration: int = Field(default=1, ge=1)

    def __init__(
        self,
        inputs: FFGroups.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        p = self.params
        id_col = C.id_col
        x_col, y_col = C.x_col, C.y_col
        frame_col = C.frame_col
        time_col = C.time_col
        group_col, seq_col = C.group_col, C.seq_col
        distance_cutoff = p.distance_cutoff
        win = max(1, p.window_size)
        if np.mod(win, 2) == 0:
            raise ValueError("window_size must be an odd integer")

        min_event = p.min_event_duration

        # Basic ordering and bookkeeping
        try:
            df = df.sort_values(resolve_order_col(df)).reset_index(drop=True)
        except ValueError:
            pass
        group_val = str(df[group_col].iloc[0]) if group_col in df.columns else ""
        seq_val = str(df[seq_col].iloc[0]) if seq_col in df.columns else ""

        frames = np.asarray(sorted(df[frame_col].dropna().unique()), dtype=int)
        ids = np.asarray(sorted(df[id_col].dropna().unique()))
        if frames.size == 0 or ids.size == 0:
            return pd.DataFrame()

        id_to_idx = {v: i for i, v in enumerate(ids)}
        T, N = len(frames), len(ids)
        frame_to_pos = {f: i for i, f in enumerate(frames)}

        # --- Vectorized pairwise distance tensor (T, N, N) ---
        # Clean data once instead of per-frame groupby
        df_clean = df[[frame_col, id_col, x_col, y_col]].copy()
        df_clean[[x_col, y_col]] = df_clean[[x_col, y_col]].replace(
            [np.inf, -np.inf], np.nan
        )
        df_clean = df_clean.dropna(subset=[id_col, x_col, y_col])
        df_clean = df_clean.drop_duplicates(subset=[frame_col, id_col], keep="last")

        # Build (T, N, 2) position tensor via direct index assignment
        fidx = df_clean[frame_col].map(frame_to_pos)
        iidx = df_clean[id_col].map(id_to_idx)
        valid = fidx.notna() & iidx.notna()
        fidx = fidx[valid].to_numpy(dtype=int)
        iidx = iidx[valid].to_numpy(dtype=int)
        x_vals = df_clean.loc[valid, x_col].to_numpy(dtype=np.float32)
        y_vals = df_clean.loc[valid, y_col].to_numpy(dtype=np.float32)

        pos = np.full((T, N, 2), np.nan, dtype=np.float32)
        pos[fidx, iidx, 0] = x_vals
        pos[fidx, iidx, 1] = y_vals

        # Pairwise Euclidean distances via broadcasting
        # diff shape: (T, N, N, 2); NaN positions propagate naturally
        diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
        pairwise = np.linalg.norm(diff, axis=-1).astype(np.float32)
        del diff

        # Self-distance -> NaN
        diag_idx = np.arange(N)
        pairwise[:, diag_idx, diag_idx] = np.nan

        # Smooth along time axis with centered window
        if win > 1 and T > 0:
            pad = win // 2
            pad_block = np.full((pad, N, N), np.nan, dtype=np.float32)
            padded = np.concatenate([pad_block, pairwise, pad_block], axis=0)
            if padded.shape[0] >= win:
                # windows shape: (T, N, N, win) when sliding along time axis only
                windows = sliding_window_view(padded, window_shape=win, axis=0)
                # Chunked nanmean to reduce peak memory usage
                chunk_size = 1000
                pairwise_smooth = np.empty((T, N, N), dtype=np.float32)
                for i in range(0, T, chunk_size):
                    end = min(i + chunk_size, T)
                    pairwise_smooth[i:end] = np.nanmean(windows[i:end], axis=-1)
                del windows
            else:
                pairwise_smooth = pairwise
            del padded, pad_block
        else:
            pairwise_smooth = pairwise

        # Prepare input for dp.calculate_gmembership: (T, N, N-1)
        mask = ~np.eye(N, dtype=bool)
        pwdf = pairwise_smooth[:, mask].reshape(T, N, N - 1)
        groupmembership = _calculate_gmembership_numba(pwdf, N, T, distance_cutoff)

        # Group sizes per frame/id
        gm = groupmembership.astype(np.int64, copy=False)
        if gm.size:
            max_label = int(gm.max(initial=0))
            counts = np.zeros((T, max_label + 1), dtype=np.int32)
            np.add.at(counts, (np.arange(T)[:, None], gm), 1)
            group_sizes = counts[np.arange(T)[:, None], gm]
        else:
            group_sizes = np.zeros_like(gm, dtype=np.int32)

        # Build base output
        out = pd.DataFrame(
            {
                frame_col: np.repeat(frames, N),
                id_col: np.tile(ids, T),
                "group_membership": groupmembership.reshape(-1),
                "group_size": group_sizes.reshape(-1),
            }
        )

        if time_col and time_col in df.columns:
            time_map = df.groupby(frame_col)[time_col].first()
            out[time_col] = out[frame_col].map(time_map)
        if seq_col in df.columns:
            out[seq_col] = seq_val
        else:
            out[seq_col] = seq_val
        if group_col in df.columns:
            out[group_col] = group_val
        else:
            out[group_col] = group_val

        # Event detection
        df_events = _get_events_info(
            out, min_event, frame_col=frame_col, id_col=id_col
        )[[frame_col, id_col, "event"]]
        out = out.merge(df_events, how="left", on=[frame_col, id_col])
        out["event"] = out["event"].fillna(-1).astype(int)
        return out
