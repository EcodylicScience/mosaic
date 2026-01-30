"""
PairPositionFeatures - egocentric dyadic features using only (x, y, angle).

Drop-in replacement for PairEgocentricFeatures when pose keypoints are not
available. Uses the ANGLE column directly for heading instead of computing
from neck→tail vector.

Output columns match PairEgocentricFeatures exactly, enabling use with
downstream features like PairWavelet.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

from behavior.dataset import register_feature
from .helpers import _merge_params


@register_feature
class PairPositionFeatures:
    """
    'pair-position' — per-sequence egocentric + kinematic features for all pairs.

    Unlike PairEgocentricFeatures which requires full pose keypoints, this feature
    works with minimal input: just (x, y, angle) per animal.

    For N animals per sequence, computes features for all N*(N-1)/2 unique pairs,
    each with two perspectives (A→B and B→A).

    Output columns (per row):
      - frame: frame number
      - perspective: 0 for A→B, 1 for B→A
      - id_A, id_B: IDs of the two animals in this pair
      - A_speed, A_v_para, A_v_perp, A_ang_speed: focal kinematics
      - A_heading_cos, A_heading_sin: focal heading
      - AB_dist: inter-animal distance
      - AB_dx_egoA, AB_dy_egoA: partner position in focal's egocentric frame
      - rel_heading_cos, rel_heading_sin: relative heading
      - B_speed, B_v_para, B_v_perp, B_ang_speed: partner kinematics
      - (optionally) group, sequence for convenience
    """

    name = "pair-position"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        # column names
        x_col="X",
        y_col="Y",
        angle_col="ANGLE",  # heading in radians
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),

        # sampling
        fps_default=30.0,
        smooth_win=0,  # 0 disables smoothing

        # cleaning / interpolation
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)

    # ------------- Feature protocol -------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def finalize_fit(self) -> None:
        pass

    def fit(self, X_iter) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = self._order_col(df)

        # Required columns
        need = [p["id_col"], p["seq_col"], order_col, p["x_col"], p["y_col"], p["angle_col"]]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[pair-position] Missing columns: {missing}")

        # Select and clean data
        cols = need.copy()
        df_small = df[cols].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        # Clean per-animal, per-sequence
        group_cols = [p["seq_col"], p["id_col"]]
        data_cols = [p["x_col"], p["y_col"], p["angle_col"]]

        def clean_animal(g):
            result = self._clean_one_animal(g, data_cols, order_col)
            if isinstance(g.name, tuple):
                for col, val in zip(group_cols, g.name):
                    result[col] = val
            else:
                result[group_cols[0]] = g.name
            return result

        df_small = (
            df_small
            .groupby(group_cols, group_keys=False)
            .apply(clean_animal, include_groups=False)
        )

        # Build all pairs for each sequence
        out_frames: List[pd.DataFrame] = []

        for seq, gseq in df_small.groupby(p["seq_col"]):
            ids = sorted(gseq[p["id_col"]].unique())
            if len(ids) < 2:
                continue

            # All unique pairs
            for idA, idB in combinations(ids, 2):
                pair_df = self._compute_pair_features(gseq, idA, idB, order_col, df)
                if pair_df is not None and not pair_df.empty:
                    out_frames.append(pair_df)

        if not out_frames:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "frame", "perspective", "id_A", "id_B",
                "A_speed", "A_v_para", "A_v_perp", "A_ang_speed",
                "A_heading_cos", "A_heading_sin",
                "AB_dist", "AB_dx_egoA", "AB_dy_egoA",
                "rel_heading_cos", "rel_heading_sin",
                "B_speed", "B_v_para", "B_v_perp", "B_ang_speed",
            ])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["id_A", "id_B", "perspective", "frame"]).reset_index(drop=True)
        return out

    def _compute_pair_features(
        self, gseq: pd.DataFrame, idA: int, idB: int, order_col: str, orig_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Compute features for a single pair (A, B) with both perspectives."""
        p = self.params

        # Extract data for each animal
        A = gseq[gseq[p["id_col"]] == idA][[order_col, p["x_col"], p["y_col"], p["angle_col"]]].copy()
        B = gseq[gseq[p["id_col"]] == idB][[order_col, p["x_col"], p["y_col"], p["angle_col"]]].copy()

        if A.empty or B.empty:
            return None

        # Sort and rename for merge
        A = A.sort_values(order_col).rename(columns={order_col: "frame"})
        B = B.sort_values(order_col).rename(columns={order_col: "frame"})

        # Inner join on frame
        j = A.merge(B, on="frame", suffixes=("_A", "_B"))
        if j.empty:
            return None

        # Get fps
        fps = float(p["fps_default"])
        if "fps" in orig_df.columns:
            try:
                c = orig_df["fps"].dropna().unique()
                if len(c) == 1:
                    fps = float(c[0])
            except Exception:
                pass

        # Build features
        frames, AtoB, BtoA, names = self._build_features(j, fps)

        # Create DataFrames for both perspectives
        dfA = pd.DataFrame(AtoB.T, columns=names)
        dfA["frame"] = frames
        dfA["perspective"] = 0
        dfA["id_A"] = idA
        dfA["id_B"] = idB

        dfB = pd.DataFrame(BtoA.T, columns=names)
        dfB["frame"] = frames
        dfB["perspective"] = 1
        dfB["id_A"] = idB  # Swap: B is now the focal
        dfB["id_B"] = idA

        # Pass through group/sequence
        for col in (p["seq_col"], p["group_col"]):
            if col in orig_df.columns:
                val = orig_df[col].iloc[0]
                dfA[col] = val
                dfB[col] = val

        return pd.concat([dfA, dfB], ignore_index=True)

    def _build_features(
        self, j: pd.DataFrame, fps: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Build egocentric features from joined pair data."""
        p = self.params
        win = int(p["smooth_win"])

        # Extract positions and angles
        cxA = j[f"{p['x_col']}_A"].to_numpy()
        cyA = j[f"{p['y_col']}_A"].to_numpy()
        cxB = j[f"{p['x_col']}_B"].to_numpy()
        cyB = j[f"{p['y_col']}_B"].to_numpy()
        thA = j[f"{p['angle_col']}_A"].to_numpy()
        thB = j[f"{p['angle_col']}_B"].to_numpy()
        frames = j["frame"].to_numpy().astype(int)

        # Optional smoothing
        if win and win > 1:
            cxA = self._smooth_1d(cxA, win)
            cyA = self._smooth_1d(cyA, win)
            cxB = self._smooth_1d(cxB, win)
            cyB = self._smooth_1d(cyB, win)
            thA = self._smooth_1d(thA, win)
            thB = self._smooth_1d(thB, win)

        # Unit heading vectors from angle
        uhxA, uhyA = np.cos(thA), np.sin(thA)
        uhxB, uhyB = np.cos(thB), np.sin(thB)

        # Orthogonal vectors (left-hand perpendicular)
        uoxA, uoyA = -uhyA, uhxA
        uoxB, uoyB = -uhyB, uhxB

        # Velocities (per second)
        vAx = np.gradient(cxA) * fps
        vAy = np.gradient(cyA) * fps
        vBx = np.gradient(cxB) * fps
        vBy = np.gradient(cyB) * fps

        speedA = np.sqrt(vAx * vAx + vAy * vAy)
        speedB = np.sqrt(vBx * vBx + vBy * vBy)

        # Angular speed
        angspeedA = self._unwrap_diff(thA, fps)
        angspeedB = self._unwrap_diff(thB, fps)

        # Ego projections of velocity
        vA_para = vAx * uhxA + vAy * uhyA
        vA_perp = vAx * uoxA + vAy * uoyA
        vB_para = vBx * uhxB + vBy * uhyB
        vB_perp = vBx * uoxB + vBy * uoyB

        # Inter-animal displacement
        dx = cxB - cxA
        dy = cyB - cyA
        distAB = np.sqrt(dx * dx + dy * dy)

        # A-centric egocentric coords of B
        dxA = dx * uhxA + dy * uhyA
        dyA = dx * uoxA + dy * uoyA

        # B-centric egocentric coords of A
        dxB = (-dx) * uhxB + (-dy) * uhyB
        dyB = (-dx) * uoxB + (-dy) * uoyB

        # Relative heading
        dth = np.unwrap(thB) - np.unwrap(thA)
        rel_cos = np.cos(dth)
        rel_sin = np.sin(dth)

        # Feature names (matching PairEgocentricFeatures)
        names = [
            "A_speed", "A_v_para", "A_v_perp", "A_ang_speed",
            "A_heading_cos", "A_heading_sin",
            "AB_dist", "AB_dx_egoA", "AB_dy_egoA",
            "rel_heading_cos", "rel_heading_sin",
            "B_speed", "B_v_para", "B_v_perp", "B_ang_speed",
        ]

        # A→B perspective
        AtoB = np.vstack([
            speedA, vA_para, vA_perp, angspeedA,
            np.cos(thA), np.sin(thA),
            distAB, dxA, dyA,
            rel_cos, rel_sin,
            speedB, vB_para, vB_perp, angspeedB,
        ]).astype(np.float32)

        # B→A perspective (swap roles)
        BtoA = np.vstack([
            speedB, vB_para, vB_perp, angspeedB,
            np.cos(thB), np.sin(thB),
            distAB, dxB, dyB,
            np.cos(-dth), np.sin(-dth),
            speedA, vA_para, vA_perp, angspeedA,
        ]).astype(np.float32)

        return frames, AtoB, BtoA, names

    # ------------- Helpers -------------
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _clean_one_animal(
        self, g: pd.DataFrame, data_cols: List[str], order_col: str
    ) -> pd.DataFrame:
        """Clean and interpolate data for one animal."""
        p = self.params
        g = g.sort_values(order_col).copy()
        g = g.set_index(order_col)

        # Replace inf with nan
        g[data_cols] = g[data_cols].replace([np.inf, -np.inf], np.nan)

        # Interpolate
        g[data_cols] = g[data_cols].interpolate(
            method="linear",
            limit=int(p["linear_interp_limit"]),
            limit_direction="both"
        )

        # Edge fill
        g[data_cols] = g[data_cols].ffill(limit=int(p["edge_fill_limit"]))
        g[data_cols] = g[data_cols].bfill(limit=int(p["edge_fill_limit"]))

        # Drop rows with too much missing data
        miss_frac = g[data_cols].isna().mean(axis=1)
        g = g.loc[miss_frac <= float(p["max_missing_fraction"])].copy()

        # Fill remaining with median
        if g[data_cols].isna().any().any():
            med = g[data_cols].median()
            g[data_cols] = g[data_cols].fillna(med)

        g = g.reset_index()
        return g

    def _smooth_1d(self, x: np.ndarray, win: int) -> np.ndarray:
        if win is None or win <= 1:
            return x
        pad = win // 2
        xp = np.pad(x, pad_width=pad, mode="reflect")
        ker = np.ones(win, dtype=float) / float(win)
        return np.convolve(xp, ker, mode="valid")

    def _unwrap_diff(self, theta: np.ndarray, fps: float) -> np.ndarray:
        """Compute angular velocity from angle array."""
        d = np.gradient(np.unwrap(theta), edge_order=1)
        return d * float(fps)
