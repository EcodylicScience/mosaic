from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from itertools import groupby, count

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist

import networkx as nx
from behavior.dataset import register_feature


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


@register_feature
class FFGroups:
    """
    Per-sequence fission–fusion grouping metrics.

    Inputs: raw tracks (columns: x, y, id, frame/time, group, sequence).
    Outputs per (frame, id):
      - group_membership (component label)
      - group_size (size of that component)
      - event (event id from dp.get_events_info, -1 if not in an event)
    """

    name = "ffgroups"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        id_col="id",
        x_col="x",
        y_col="y",
        frame_col="frame",
        time_col="time",
        group_col="group",
        seq_col="sequence",
        distance_cutoff=50.0,   # distance threshold for same-group assignment
        window_size=5,          # smoothing window (frames); uses nanmean
        min_event_duration=1,   # frames
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

    # ---- Dataset hooks ----
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ---- Fit/transform contract ----
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return None
    def save_model(self, path: Path) -> None: return None
    def load_model(self, path: Path) -> None: return None

    # ---- Core logic ----
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        id_col = p["id_col"]
        x_col, y_col = p["x_col"], p["y_col"]
        frame_col = p["frame_col"]
        time_col = p.get("time_col")
        group_col, seq_col = p["group_col"], p["seq_col"]
        distance_cutoff = float(p["distance_cutoff"])
        win = max(1, int(p["window_size"]))
        if np.mod(win,2)==0:
            raise ValueError('window_size must be an odd integer')

        min_event = int(p["min_event_duration"])

        # Basic ordering and bookkeeping
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

        # Build pairwise distance tensor (T, N, N) with nan for missing individuals
        pairwise = np.full((T, N, N), np.nan, dtype=np.float32)
        frame_to_pos = {f: i for i, f in enumerate(frames)}
        all_ids_set = set(ids.tolist())

        for f, sub in df.groupby(frame_col):
            if f not in frame_to_pos:
                continue
            t_idx = frame_to_pos[f]
            coords = sub[[id_col, x_col, y_col]].dropna(subset=[id_col, x_col, y_col])
            if coords.empty:
                continue
            coords = coords.drop_duplicates(subset=[id_col], keep="last").set_index(id_col)
            ids_present = coords.index.to_numpy()
            xy = coords[[x_col, y_col]].to_numpy(dtype=np.float32, copy=False)

            if len(ids_present) == N and set(ids_present) == all_ids_set:
                # Fast path: all IDs present; reorder positions to global id order and assign directly
                order = np.argsort([id_to_idx.get(i, N + idx) for idx, i in enumerate(ids_present)])
                xy_full = xy[order]
                dmat = cdist(xy_full, xy_full).astype(np.float32)
                np.fill_diagonal(dmat, np.nan)
                pairwise[t_idx] = dmat
            else:
                # Distance matrix for present ids only
                dmat = cdist(xy, xy).astype(np.float32)
                np.fill_diagonal(dmat, np.nan)

                # Map into full (N x N) slots
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
                pairwise[t_idx] = np.nan  # reset row in case of partial fill
                pairwise[t_idx][np.ix_(global_idx, global_idx)] = dmat[np.ix_(local_idx, local_idx)]

        # Smooth along time axis with centered window
        if win > 1 and T > 0:
            pad = win // 2
            pad_block = np.full((pad, N, N), np.nan, dtype=np.float32)
            padded = np.concatenate([pad_block, pairwise, pad_block], axis=0)
            if padded.shape[0] >= win:
                # windows shape: (T, N, N, win) when sliding along time axis only
                windows = sliding_window_view(padded, window_shape=win, axis=0)
                pairwise_smooth = np.nanmean(windows, axis=-1)
            else:
                pairwise_smooth = pairwise
        else:
            pairwise_smooth = pairwise

        # Prepare input for dp.calculate_gmembership: (T, N, N-1)
        mask = ~np.eye(N, dtype=bool)
        pwdf = pairwise_smooth[:, mask].reshape(T, N, N - 1)
        groupmembership = _calculate_gmembership(pwdf, nparticles=N, numsteps=T, threshold=distance_cutoff)

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
        out = pd.DataFrame({
            frame_col: np.repeat(frames, N),
            id_col: np.tile(ids, T),
            "group_membership": groupmembership.reshape(-1),
            "group_size": group_sizes.reshape(-1),
        })

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

        # Event detection (dp.get_events_info expects 'FishID')
        event_input = out.rename(columns={id_col: "FishID"})
        try:
            df_events, _ = _get_events_info(event_input, threshold_ev_duration=min_event)
            df_events = df_events[["frame", "FishID", "event"]]
            out = out.merge(df_events.rename(columns={"FishID": id_col, "event": "event"}),
                            how="left", on=[frame_col, id_col])
        except Exception:
            # If event detection fails, still return membership/size
            out["event"] = np.nan

        out["event"] = out["event"].fillna(-1).astype(int)
        return out


# ===== Embedded helpers (from data_processing.py) =====

def _getcomponents(Aij_single: np.ndarray, numfish: int):
    G = nx.Graph(Aij_single)
    connected = list(nx.connected_components(G))
    componentsizes = [len(c) for c in connected]
    components = np.zeros(numfish, dtype=int) - 1
    for i in range(len(connected)):
        components[list(connected[i])] = i
    largestgroupsize = np.max(componentsizes) if componentsizes else 0
    numcomponents = len(componentsizes)
    return components, largestgroupsize, numcomponents


def _calculate_gmembership(pwdf, nparticles, numsteps, threshold):
    mask = np.tile(True, (nparticles, nparticles))
    mask[range(nparticles), range(nparticles)] = False

    groupmembership = np.zeros((numsteps, nparticles))
    for idx, step in enumerate(range(numsteps)):
        Aij = np.zeros((nparticles, nparticles))
        Aij[mask] = np.reshape(np.heaviside(threshold - pwdf[step], 0), -1)
        Aij[np.isnan(Aij)] = 0
        groupmembership[step] = _getcomponents(Aij, numfish=nparticles)[0]
    return groupmembership


def _find_events(df, minimal_length):
    finalized = []
    all_data = []
    for _, g in groupby(np.unique(df.frame.astype(int)), key=lambda n, c=count(): n - next(c)):
        l = list(g)
        all_data.append((l[0], l[-1], l))
    for i, (start, end, l) in enumerate(all_data):
        if end - start >= minimal_length:
            data = df.loc[(df.frame >= start) & (df.frame <= end)]
            clusters = {}

            for frame in l:
                found = {c: False for c in clusters}
                sub = data.loc[data.frame == frame]
                for clusterid in sub.group_membership.unique():
                    ids = tuple(set(sub.FishID[sub.group_membership == clusterid].values.astype(int)))
                    if ids in clusters:
                        clusters[ids]["end"] = frame
                    else:
                        clusters[ids] = {"start": frame, "end": frame}
                    found[ids] = True

                for c in list(found):
                    if not found[c]:
                        if clusters[c]["end"] - clusters[c]["start"] >= minimal_length:
                            finalized.append((c, (clusters[c]["start"], clusters[c]["end"])))
                        del clusters[c]
    return finalized


def _get_events_info(df, threshold_ev_duration=1):
    clusters_ = _find_events(df, minimal_length=threshold_ev_duration)
    concat = []
    for event, (ids, (start, end)) in enumerate(clusters_):
        mask0 = (df["frame"] >= start) & (df["frame"] <= end)
        mask1 = np.zeros_like(mask0, dtype=bool)
        for ID in ids:
            mask1[mask0] = np.logical_or(mask1[mask0], df["FishID"][mask0] == ID)
        mask = np.logical_and(mask0, mask1)
        df_events = df[mask]
        df_events.insert(loc=1, column="event", value=event)
        concat.append(df_events)
    if not concat:
        return pd.DataFrame(columns=list(df.columns) + ["event"]), clusters_
    df_with_events = pd.concat(concat, axis=0).reset_index(drop=True)
    return df_with_events, clusters_
