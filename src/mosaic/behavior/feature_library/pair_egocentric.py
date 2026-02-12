"""
PairEgocentricFeatures feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from itertools import combinations
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature
from .helpers import _merge_params


@register_feature
class PairEgocentricFeatures:
    """
    'pair-egocentric' — per-sequence egocentric + kinematic features for dyads.
    Produces a row-wise DataFrame with columns:
      - frame (if available) or time passthrough (only if it's the order col)
      - perspective: 0 for A→B, 1 for B→A
      - id1, id2: pair identifiers
      - feature columns (e.g., A_speed, AB_dx_egoA, ...)
      - (optionally) group/sequence if present in df, for convenience

    This feature is *stateless* (no fitting). It computes features for all C(n,2)
    pairs per sequence, cleans/interpolates pose per animal, inner-joins by the
    chosen order column, and computes A→B and B→A features for each pair.
    """

    name    = "pair-egocentric"
    version = "0.1"
    parallelizable = True
    output_type = "per_frame"

    _defaults = dict(
        # pose / columns
        pose_n=7,
        pose_indices=None,  # list of pose point indices to use, or None for all 0..pose_n-1
        x_prefix="poseX", y_prefix="poseY",   # TRex-ish
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),

        # required anatomical indices (must be provided by user if different)
        neck_idx=None,           # REQUIRED (int) unless your skeleton matches defaults
        tail_base_idx=None,      # REQUIRED (int) unless your skeleton matches defaults
        center_mode="mean",      # "mean" or an int landmark index

        # sampling / smoothing
        fps_default=30.0,
        smooth_win=0,            # 0 disables box smoothing before differencing

        # cleaning / interpolation per animal
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        # Enforce required indices if user didn't pass them
        if self.params["neck_idx"] is None:
            # sensible default matching your earlier snippet
            self.params["neck_idx"] = 3
        if self.params["tail_base_idx"] is None:
            self.params["tail_base_idx"] = 6

        self._tri_ready = False  # not used, but kept for symmetry with other feature

    # ------------- Feature protocol -------------
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def finalize_fit(self) -> None: pass
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return
    def partial_fit(self, df: pd.DataFrame) -> None: return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)
        p = self.params

        need = [p["id_col"], p["seq_col"], order_col] + pose_cols
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[pair-egocentric] Missing cols: {missing}")

        # Clean per-animal, per-sequence (future-proof re: pandas groupby.apply)
        df_small = df[need].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        group_cols = [p["seq_col"], p["id_col"]]

        def wrapped_func(g):
            result = self._clean_one_animal(g, pose_cols, order_col)
            # reattach group key(s)
            if isinstance(g.name, tuple):
                for col, val in zip(group_cols, g.name):
                    result[col] = val
            else:
                result[group_cols[0]] = g.name
            return result

        df_small = (
            df_small
            .groupby(group_cols, group_keys=False)
            .apply(wrapped_func, include_groups=False)
        )

        # Build dyads (all C(n,2) pairs per sequence)
        pairs = []
        for seq, gseq in df_small.groupby(p["seq_col"]):
            ids = sorted(gseq[p["id_col"]].unique())
            if len(ids) >= 2:
                for idA, idB in combinations(ids, 2):
                    pairs.append((seq, idA, idB))

        if not pairs:
            raise ValueError("[pair-egocentric] No sequence with at least two IDs found.")

        out_frames: List[pd.DataFrame] = []
        for seq, idA, idB in pairs:
            gseq = df_small[df_small[p["seq_col"]] == seq]
            A = gseq[gseq[p["id_col"]] == idA][[order_col] + pose_cols].copy()
            B = gseq[gseq[p["id_col"]] == idB][[order_col] + pose_cols].copy()
            if A.empty or B.empty:
                continue

            A = A.sort_values(order_col).rename(columns={order_col: "frame"})
            B = B.sort_values(order_col).rename(columns={order_col: "frame"})
            j = A.merge(B, on="frame", suffixes=("_A", "_B"))
            if j.empty:
                continue

            # fps heuristic: prefer df['fps'] if present and constant; else default
            fps = float(p["fps_default"])
            if "fps" in df.columns:
                try:
                    c = df["fps"].dropna().unique()
                    if len(c) == 1:
                        fps = float(c[0])
                except Exception:
                    pass

            frames, AtoB, BtoA, names = self._build_ego_block_for_joined(j, fps, pose_cols)

            # produce row-wise DataFrames
            dfA = pd.DataFrame(AtoB.T, columns=names)
            dfA["frame"] = frames
            dfA["perspective"] = 0
            dfA["id1"] = idA
            dfA["id2"] = idB

            dfB = pd.DataFrame(BtoA.T, columns=names)
            dfB["frame"] = frames
            dfB["perspective"] = 1
            dfB["id1"] = idB
            dfB["id2"] = idA

            # optional pass-through for convenience (constant per call)
            for col in (p["seq_col"], p["group_col"]):
                if col in df.columns:
                    dfA[col] = df[col].iloc[0]
                    dfB[col] = df[col].iloc[0]

            out_frames.extend([dfA, dfB])

        if not out_frames:
            return pd.DataFrame(columns=["perspective", "frame", "id1", "id2"])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["frame", "id1", "id2"]).reset_index(drop=True)
        return out

    # ------------- Internals -------------
    def _get_pose_indices(self) -> List[int]:
        """Return the list of pose point indices to use."""
        indices = self.params.get("pose_indices")
        if indices is None:
            return list(range(int(self.params["pose_n"])))
        return list(indices)

    def _effective_pose_n(self) -> int:
        """Return the number of pose points being used."""
        return len(self._get_pose_indices())

    def _map_anatomical_idx(self, param_name: str) -> int:
        """Map an anatomical index (neck_idx, tail_base_idx) to position in filtered array.

        If pose_indices is set, the param value is treated as an absolute pose point index
        and mapped to its position in pose_indices. If pose_indices is None, returns the
        value directly as an index into the full array.
        """
        val = int(self.params[param_name])
        indices = self.params.get("pose_indices")
        if indices is None:
            return val
        try:
            return list(indices).index(val)
        except ValueError:
            raise ValueError(
                f"[pair-egocentric] {param_name}={val} is not in pose_indices={indices}. "
                f"When using pose_indices, {param_name} must be one of the selected indices."
            )

    def _column_names(self) -> Tuple[List[str], List[str]]:
        indices = self._get_pose_indices()
        xs = [f"{self.params['x_prefix']}{i}" for i in indices]
        ys = [f"{self.params['y_prefix']}{i}" for i in indices]
        return xs, ys

    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _clean_one_animal(self, g: pd.DataFrame, pose_cols: List[str], order_col: str) -> pd.DataFrame:
        p = self.params
        g = g.sort_values(order_col).copy()
        g = g.set_index(order_col)
        g[pose_cols] = g[pose_cols].replace([np.inf, -np.inf], np.nan)
        g[pose_cols] = g[pose_cols].interpolate(
            method="linear", limit=int(p["linear_interp_limit"]), limit_direction="both"
        )
        g[pose_cols] = g[pose_cols].ffill(limit=int(p["edge_fill_limit"]))
        g[pose_cols] = g[pose_cols].bfill(limit=int(p["edge_fill_limit"]))
        miss_frac = g[pose_cols].isna().mean(axis=1)
        g = g.loc[miss_frac <= float(p["max_missing_fraction"])].copy()
        if g[pose_cols].isna().any().any():
            med = g[pose_cols].median()
            g[pose_cols] = g[pose_cols].fillna(med)
        g = g.reset_index()
        return g

    # --- math helpers ---
    def _smooth_1d(self, x: np.ndarray, win: int) -> np.ndarray:
        if win is None or win <= 1:
            return x
        pad = win // 2
        xp = np.pad(x, pad_width=pad, mode="reflect")
        ker = np.ones(win, dtype=float) / float(win)
        return np.convolve(xp, ker, mode="valid")

    def _safe_unit(self, vx: np.ndarray, vy: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        n = np.sqrt(vx * vx + vy * vy) + eps
        return vx / n, vy / n

    def _angle(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        return np.arctan2(vy, vx)

    def _unwrap_diff(self, theta: np.ndarray, fps: float) -> np.ndarray:
        d = np.gradient(np.unwrap(theta), edge_order=1)
        return d * float(fps)

    def _center_from_points(self, xs: np.ndarray, ys: np.ndarray, mode: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(mode, (int, np.integer)):
            return xs[:, int(mode)], ys[:, int(mode)]
        return xs.mean(axis=1), ys.mean(axis=1)

    def _build_ego_block_for_joined(self, j: pd.DataFrame, fps: float, pose_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        indices = self._get_pose_indices()
        N = len(indices)
        neck = self._map_anatomical_idx("neck_idx")
        tail = self._map_anatomical_idx("tail_base_idx")
        win  = int(self.params["smooth_win"])
        mode = self.params["center_mode"]

        XA = j[[f"{self.params['x_prefix']}{k}_A" for k in indices]].to_numpy()
        YA = j[[f"{self.params['y_prefix']}{k}_A" for k in indices]].to_numpy()
        XB = j[[f"{self.params['x_prefix']}{k}_B" for k in indices]].to_numpy()
        YB = j[[f"{self.params['y_prefix']}{k}_B" for k in indices]].to_numpy()
        frames = j["frame"].to_numpy().astype(int)

        # optional smoothing
        if win and win > 1:
            XA = np.vstack([self._smooth_1d(XA[:, k], win) for k in range(N)]).T
            YA = np.vstack([self._smooth_1d(YA[:, k], win) for k in range(N)]).T
            XB = np.vstack([self._smooth_1d(XB[:, k], win) for k in range(N)]).T
            YB = np.vstack([self._smooth_1d(YB[:, k], win) for k in range(N)]).T

        # centers
        cxA, cyA = self._center_from_points(XA, YA, mode)
        cxB, cyB = self._center_from_points(XB, YB, mode)

        # headings (neck - tail) and units
        hxA, hyA = XA[:, neck] - XA[:, tail], YA[:, neck] - YA[:, tail]
        hxB, hyB = XB[:, neck] - XB[:, tail], YB[:, neck] - YB[:, tail]
        uhxA, uhyA = self._safe_unit(hxA, hyA)
        uhxB, uhyB = self._safe_unit(hxB, hyB)
        # left-hand orthogonal
        uoxA, uoyA = -uhyA, uhxA
        uoxB, uoyB = -uhyB, uhxB

        # velocities of centers (per second)
        vAx = np.gradient(cxA) * float(fps)
        vAy = np.gradient(cyA) * float(fps)
        vBx = np.gradient(cxB) * float(fps)
        vBy = np.gradient(cyB) * float(fps)
        speedA = np.sqrt(vAx*vAx + vAy*vAy)
        speedB = np.sqrt(vBx*vBx + vBy*vBy)

        # heading angles + angular speed
        thA = self._angle(uhxA, uhyA)
        thB = self._angle(uhxB, uhyB)
        angspeedA = self._unwrap_diff(thA, fps)
        angspeedB = self._unwrap_diff(thB, fps)

        # ego projections of velocity
        vA_para = vAx * uhxA + vAy * uhyA
        vA_perp = vAx * uoxA + vAy * uoyA
        vB_para = vBx * uhxB + vBy * uhyB
        vB_perp = vBx * uoxB + vBy * uoyB

        # displacement A→B in world + A-centric ego coords of B
        dx = cxB - cxA
        dy = cyB - cyA
        distAB = np.sqrt(dx*dx + dy*dy)

        dxA = dx * uhxA + dy * uhyA
        dyA = dx * uoxA + dy * uoyA

        # B-centric ego coords of A
        dxB = (-dx) * uhxB + (-dy) * uhyB
        dyB = (-dx) * uoxB + (-dy) * uoyB

        # relative heading B wrt A
        dth = np.unwrap(thB) - np.unwrap(thA)
        rel_cos = np.cos(dth)
        rel_sin = np.sin(dth)

        names = [
            "A_speed", "A_v_para", "A_v_perp", "A_ang_speed",
            "A_heading_cos", "A_heading_sin",
            "AB_dist", "AB_dx_egoA", "AB_dy_egoA",
            "rel_heading_cos", "rel_heading_sin",
            "B_speed", "B_v_para", "B_v_perp", "B_ang_speed",
        ]

        AtoB = np.vstack([
            speedA, vA_para, vA_perp, angspeedA,
            np.cos(thA), np.sin(thA),
            distAB, dxA, dyA,
            rel_cos, rel_sin,
            speedB, vB_para, vB_perp, angspeedB,
        ]).astype(np.float32)

        # For B→A, swap roles but keep same semantic ordering (B is 'self')
        BtoA = np.vstack([
            speedB, vB_para, vB_perp, angspeedB,
            np.cos(thB), np.sin(thB),
            distAB, dxB, dyB,
            np.cos(-dth), np.sin(-dth),
            speedA, vA_para, vA_perp, angspeedA,
        ]).astype(np.float32)

        return frames, AtoB, BtoA, names
