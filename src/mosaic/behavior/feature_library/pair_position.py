"""
PairPositionFeatures - egocentric dyadic features using only (x, y, angle).

Drop-in replacement for PairEgocentricFeatures when pose keypoints are not
available. Uses the ANGLE column directly for heading instead of computing
from neck->tail vector.

Output columns match PairEgocentricFeatures exactly, enabling use with
downstream features like PairWavelet.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
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

from .helpers import clean_tracks_grouped, ensure_columns, smooth_1d, unwrap_diff
from .registry import register_feature
from .types import InterpolationConfig, SamplingConfig


@final
@register_feature
class PairPositionFeatures:
    """
    'pair-position' -- per-sequence egocentric + kinematic features for all pairs.

    Unlike PairEgocentricFeatures which requires full pose keypoints, this feature
    works with minimal input: just (x, y, angle) per animal.

    For N animals per sequence, computes features for all N*(N-1)/2 unique pairs,
    each with two perspectives (A->B and B->A).

    Output columns (per row):
      - frame: frame number
      - perspective: 0 for A->B, 1 for B->A
      - id1, id2: IDs of the two animals in this pair
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
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
        sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    def __init__(
        self,
        inputs: PairPositionFeatures.Inputs = Inputs(("tracks",)),
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
        order_col = resolve_order_col(df)

        required = [C.id_col, C.seq_col, C.x_col, C.y_col, C.orientation_col]
        ensure_columns(df, required)

        df_small = df[[order_col] + required]

        # Clean per-animal, per-sequence
        group_cols = [C.seq_col, C.id_col]
        data_cols = [C.x_col, C.y_col, C.orientation_col]

        df_small = clean_tracks_grouped(
            df_small, group_cols, data_cols, order_col, self.params.interpolation
        )

        # Build all pairs for each sequence
        out_frames: list[pd.DataFrame] = []

        for _, gseq in df_small.groupby(C.seq_col):
            ids = sorted(gseq[C.id_col].unique())
            if len(ids) < 2:
                continue

            # All unique pairs
            for id_a, id_b in combinations(ids, 2):
                pair_df = self._compute_pair_features(gseq, id_a, id_b, order_col, df)
                if pair_df is not None and not pair_df.empty:
                    out_frames.append(pair_df)

        if not out_frames:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    C.frame_col,
                    "perspective",
                    "id1",
                    "id2",
                    "A_speed",
                    "A_v_para",
                    "A_v_perp",
                    "A_ang_speed",
                    "A_heading_cos",
                    "A_heading_sin",
                    "AB_dist",
                    "AB_dx_egoA",
                    "AB_dy_egoA",
                    "rel_heading_cos",
                    "rel_heading_sin",
                    "B_speed",
                    "B_v_para",
                    "B_v_perp",
                    "B_ang_speed",
                ]
            )

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["id1", "id2", "perspective", C.frame_col]).reset_index(
            drop=True
        )
        return out

    def _compute_pair_features(
        self,
        gseq: pd.DataFrame,
        id_a: int,
        id_b: int,
        order_col: str,
        orig_df: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Compute features for a single pair (A, B) with both perspectives."""
        p = self.params

        # Extract data for each animal
        df_a = gseq[gseq[C.id_col] == id_a][
            [order_col, C.x_col, C.y_col, C.orientation_col]
        ]
        df_b = gseq[gseq[C.id_col] == id_b][
            [order_col, C.x_col, C.y_col, C.orientation_col]
        ]

        if df_a.empty or df_b.empty:
            return None

        # Sort and rename for merge
        df_a = df_a.sort_values(order_col).rename(columns={order_col: C.frame_col})
        df_b = df_b.sort_values(order_col).rename(columns={order_col: C.frame_col})

        # Inner join on frame
        j = df_a.merge(df_b, on=C.frame_col, suffixes=("_A", "_B"))
        if j.empty:
            return None

        # Get fps
        fps = float(p.sampling.fps_default)
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
        dfA[C.frame_col] = frames
        dfA["perspective"] = 0
        dfA["id1"] = id_a
        dfA["id2"] = id_b

        dfB = pd.DataFrame(BtoA.T, columns=names)
        dfB[C.frame_col] = frames
        dfB["perspective"] = 1
        dfB["id1"] = id_b  # Swap: B is now the focal
        dfB["id2"] = id_a

        # Pass through group/sequence
        for col in (C.seq_col, C.group_col):
            if col in orig_df.columns:
                val = orig_df[col].iloc[0]
                dfA[col] = val
                dfB[col] = val

        return pd.concat([dfA, dfB], ignore_index=True)

    def _build_features(
        self, j: pd.DataFrame, fps: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build egocentric features from joined pair data."""
        p = self.params
        win = p.sampling.smooth_win

        # Extract positions and angles
        cxA = j[f"{C.x_col}_A"].to_numpy()
        cyA = j[f"{C.y_col}_A"].to_numpy()
        cxB = j[f"{C.x_col}_B"].to_numpy()
        cyB = j[f"{C.y_col}_B"].to_numpy()
        thA = j[f"{C.orientation_col}_A"].to_numpy()
        thB = j[f"{C.orientation_col}_B"].to_numpy()
        frames = j[C.frame_col].to_numpy()

        # Optional smoothing
        if win and win > 1:
            cxA = smooth_1d(cxA, win)
            cyA = smooth_1d(cyA, win)
            cxB = smooth_1d(cxB, win)
            cyB = smooth_1d(cyB, win)
            thA = smooth_1d(thA, win)
            thB = smooth_1d(thB, win)

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
        angspeedA = unwrap_diff(thA, fps)
        angspeedB = unwrap_diff(thB, fps)

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
            "A_speed",
            "A_v_para",
            "A_v_perp",
            "A_ang_speed",
            "A_heading_cos",
            "A_heading_sin",
            "AB_dist",
            "AB_dx_egoA",
            "AB_dy_egoA",
            "rel_heading_cos",
            "rel_heading_sin",
            "B_speed",
            "B_v_para",
            "B_v_perp",
            "B_ang_speed",
        ]

        # A->B perspective
        AtoB = np.vstack(
            [
                speedA,
                vA_para,
                vA_perp,
                angspeedA,
                np.cos(thA),
                np.sin(thA),
                distAB,
                dxA,
                dyA,
                rel_cos,
                rel_sin,
                speedB,
                vB_para,
                vB_perp,
                angspeedB,
            ]
        ).astype(np.float32)

        # B->A perspective (swap roles)
        BtoA = np.vstack(
            [
                speedB,
                vB_para,
                vB_perp,
                angspeedB,
                np.cos(thB),
                np.sin(thB),
                distAB,
                dxB,
                dyB,
                np.cos(-dth),
                np.sin(-dth),
                speedA,
                vA_para,
                vA_perp,
                angspeedA,
            ]
        ).astype(np.float32)

        return frames, AtoB, BtoA, names
