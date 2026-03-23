"""Regression tests: vectorized FFGroups.apply vs reference implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist

from mosaic.behavior.feature_library.ffgroups import (
    FFGroups,
    _calculate_gmembership_numba,
    _get_events_info,
)
from mosaic.behavior.feature_library.spec import COLUMNS as C


def _make_track_data(
    n_frames: int = 200,
    n_ids: int = 8,
    *,
    missing_fraction: float = 0.0,
    inf_fraction: float = 0.0,
    duplicate_fraction: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic tracking data."""
    rng = np.random.RandomState(seed)

    frames = np.repeat(np.arange(n_frames), n_ids)
    ids = np.tile(np.arange(n_ids), n_frames)
    x = rng.uniform(0, 500, size=len(frames)).astype(np.float32)
    y = rng.uniform(0, 500, size=len(frames)).astype(np.float32)

    df = pd.DataFrame(
        {
            "frame": frames,
            "id": ids,
            "X": x,
            "Y": y,
            "group": "test_group",
            "sequence": "test_seq",
        }
    )

    n_total = len(df)

    # Introduce missing values (drop some rows)
    if missing_fraction > 0:
        drop_mask = rng.random(n_total) < missing_fraction
        df = df[~drop_mask].reset_index(drop=True)

    # Introduce inf values
    if inf_fraction > 0:
        n_inf = int(len(df) * inf_fraction)
        inf_idx = rng.choice(len(df), size=n_inf, replace=False)
        df.loc[inf_idx[:n_inf // 2], "X"] = np.inf
        df.loc[inf_idx[n_inf // 2:], "Y"] = -np.inf

    # Introduce duplicate (frame, id) rows
    if duplicate_fraction > 0:
        n_dup = int(len(df) * duplicate_fraction)
        dup_idx = rng.choice(len(df), size=n_dup, replace=False)
        dups = df.iloc[dup_idx].copy()
        # Slightly different coordinates for duplicates
        dups["X"] = dups["X"] + rng.uniform(-1, 1, size=len(dups)).astype(np.float32)
        dups["Y"] = dups["Y"] + rng.uniform(-1, 1, size=len(dups)).astype(np.float32)
        df = pd.concat([df, dups], ignore_index=True)

    return df


# --- Reference implementation (original per-frame groupby, kept for regression) ---


def _apply_reference(feature: FFGroups, df: pd.DataFrame) -> pd.DataFrame:
    """Original per-frame groupby implementation, kept for regression testing."""
    if df.empty:
        return pd.DataFrame()

    p = feature.params
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

    order_cols = [c for c in (frame_col, time_col) if c in df.columns]
    if order_cols:
        df = df.sort_values(order_cols).reset_index(drop=True)
    group_val = str(df[group_col].iloc[0]) if group_col in df.columns else ""
    seq_val = str(df[seq_col].iloc[0]) if seq_col in df.columns else ""

    frames = np.asarray(sorted(df[frame_col].dropna().unique()), dtype=int)
    ids = np.asarray(sorted(df[id_col].dropna().unique()))
    if frames.size == 0 or ids.size == 0:
        return pd.DataFrame()

    id_to_idx = {v: i for i, v in enumerate(ids)}
    T, N = len(frames), len(ids)

    pairwise = np.full((T, N, N), np.nan, dtype=np.float32)
    frame_to_pos = {f: i for i, f in enumerate(frames)}
    all_ids_set = set(ids.tolist())

    for f, sub in df.groupby(frame_col):
        if f not in frame_to_pos:
            continue
        t_idx = frame_to_pos[f]
        coords = sub[[id_col, x_col, y_col]].replace([np.inf, -np.inf], np.nan)
        coords = coords.dropna(subset=[id_col, x_col, y_col])
        if coords.empty:
            continue
        coords = coords.drop_duplicates(subset=[id_col], keep="last").set_index(
            id_col
        )
        ids_present = coords.index.to_numpy()
        xy = coords[[x_col, y_col]].to_numpy(dtype=np.float32, copy=False)

        if len(ids_present) == N and set(ids_present) == all_ids_set:
            order = np.argsort(
                [id_to_idx.get(i, N + idx) for idx, i in enumerate(ids_present)]
            )
            xy_full = xy[order]
            dmat = cdist(xy_full, xy_full).astype(np.float32)
            np.fill_diagonal(dmat, np.nan)
            pairwise[t_idx] = dmat
        else:
            dmat = cdist(xy, xy).astype(np.float32)
            np.fill_diagonal(dmat, np.nan)

            global_idx = []
            local_idx = []
            for local_i, id_i in enumerate(ids_present):
                gi = id_to_idx.get(id_i)
                if gi is None:
                    continue
                global_idx.append(gi)
                local_idx.append(local_i)
            if not global_idx:
                continue
            global_idx = np.asarray(global_idx, dtype=int)
            local_idx = np.asarray(local_idx, dtype=int)
            pairwise[t_idx] = np.nan
            pairwise[t_idx][np.ix_(global_idx, global_idx)] = dmat[
                np.ix_(local_idx, local_idx)
            ]

    if win > 1 and T > 0:
        pad = win // 2
        pad_block = np.full((pad, N, N), np.nan, dtype=np.float32)
        padded = np.concatenate([pad_block, pairwise, pad_block], axis=0)
        if padded.shape[0] >= win:
            windows = sliding_window_view(padded, window_shape=win, axis=0)
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

    mask = ~np.eye(N, dtype=bool)
    pwdf = pairwise_smooth[:, mask].reshape(T, N, N - 1)
    groupmembership = _calculate_gmembership_numba(pwdf, N, T, distance_cutoff)

    gm = groupmembership.astype(np.int64, copy=False)
    if gm.size:
        max_label = int(gm.max(initial=0))
        counts = np.zeros((T, max_label + 1), dtype=np.int32)
        np.add.at(counts, (np.arange(T)[:, None], gm), 1)
        group_sizes = counts[np.arange(T)[:, None], gm]
    else:
        group_sizes = np.zeros_like(gm, dtype=np.int32)

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

    try:
        df_events = _get_events_info(
            out, min_event, frame_col=frame_col, id_col=id_col
        )[[frame_col, id_col, "event"]]
        out = out.merge(df_events, how="left", on=[frame_col, id_col])
    except Exception:
        out["event"] = np.nan

    out["event"] = out["event"].fillna(-1).astype(int)
    return out


@pytest.fixture
def feature():
    return FFGroups(
        params={"distance_cutoff": 50, "window_size": 5, "min_event_duration": 3}
    )


class TestFFGroupsVectorizedEquivalence:
    """Verify vectorized apply matches the original reference implementation."""

    def test_clean_data(self, feature):
        """All IDs present in every frame, no missing/inf/duplicates."""
        df = _make_track_data(n_frames=100, n_ids=8)
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_missing_individuals(self, feature):
        """Some (frame, id) pairs missing."""
        df = _make_track_data(n_frames=100, n_ids=8, missing_fraction=0.1)
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_inf_values(self, feature):
        """Some coordinates are +/- inf."""
        df = _make_track_data(n_frames=100, n_ids=8, inf_fraction=0.05)
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_duplicate_rows(self, feature):
        """Duplicate (frame, id) pairs with different coordinates."""
        df = _make_track_data(n_frames=100, n_ids=8, duplicate_fraction=0.05)
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_combined_issues(self, feature):
        """Missing + inf + duplicates combined."""
        df = _make_track_data(
            n_frames=200,
            n_ids=8,
            missing_fraction=0.1,
            inf_fraction=0.03,
            duplicate_fraction=0.03,
        )
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_window_size_1(self):
        """No smoothing (window_size=1)."""
        feat = FFGroups(
            params={"distance_cutoff": 50, "window_size": 1, "min_event_duration": 1}
        )
        df = _make_track_data(n_frames=50, n_ids=4, missing_fraction=0.05)
        result = feat.apply(df)
        expected = _apply_reference(feat, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_two_ids(self):
        """Minimal number of individuals."""
        feat = FFGroups(
            params={"distance_cutoff": 100, "window_size": 3, "min_event_duration": 1}
        )
        df = _make_track_data(n_frames=50, n_ids=2, missing_fraction=0.1)
        result = feat.apply(df)
        expected = _apply_reference(feat, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_single_frame(self, feature):
        """Edge case: only one frame."""
        df = _make_track_data(n_frames=1, n_ids=4)
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self, feature):
        """Edge case: empty input."""
        df = pd.DataFrame()
        result = feature.apply(df)
        expected = _apply_reference(feature, df)
        pd.testing.assert_frame_equal(result, expected)
