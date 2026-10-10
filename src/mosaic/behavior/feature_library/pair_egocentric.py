"""
PairEgocentricFeatures feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Annotated, ClassVar, Final, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Keypoints,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    POSE_CONFIG_DESCRIPTION,
    PoseConfig,
    TrackInputs,
    resolve_order_col,
)
from mosaic.core.params import Declared, Params

from ._sampling_descriptions import SAMPLING_WITH_SMOOTHING_DESCRIPTION
from .helpers import clean_tracks_grouped, ensure_columns, smooth_1d, unwrap_diff
from .registry import register_feature
from .types import (
    INTERPOLATION_CONFIG_DESCRIPTION,
    InterpolationConfig,
    SamplingConfig,
)

_NECK_IDX_DESCRIPTION = (
    "Index of the neck keypoint in the pose array. Paired with "
    "tail_base_idx to compute heading. The direction runs from the "
    "tail-base to the neck, the direction the animal faces. When "
    "pose.pose_indices is set, this index must be one of the selected "
    "indices."
)

_TAIL_BASE_IDX_DESCRIPTION = (
    "Index of the tail-base keypoint in the pose array. Paired with "
    "neck_idx to compute heading. The direction runs from the "
    "tail-base to the neck, the direction the animal faces. When "
    "pose.pose_indices is set, this index must be one of the selected "
    "indices."
)

_CENTER_MODE_DESCRIPTION = (
    "How the animal's center point is computed from its pose "
    "keypoints. Every accepted value averages all keypoints. "
    "Selecting a single keypoint by index needs an integer, which "
    "this field does not accept."
)

_CENTER_MODE_UNWIRED = (
    "every accepted value averages all keypoints -- the per-keypoint "
    "branch needs an int this field refuses"
)

EGOCENTRIC_PAIR_COLUMNS: Final[tuple[str, ...]] = (
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
)
"""The measurements of one pair row, in output order.

``pair-position`` writes the same columns from a centroid and a heading, and a
step reading the output of one reads the output of the other.
"""

PAIR_EGOCENTRIC_PROVIDES: Final = Provides(
    adds=(
        Column(C.frame_col),
        Column("perspective"),
        Column("id1"),
        Column("id2"),
        Column(C.seq_col),
        *(Column(name) for name in EGOCENTRIC_PAIR_COLUMNS),
    ),
    conditional=kept(C.group_col),
)
"""The output of ``pair-egocentric`` and ``pair-position``.

Each renames the order column to ``frame``, whether its values are frames or times.
Each reads ``sequence`` and copies it.
"""


@final
@register_feature
class PairEgocentricFeatures:
    """Egocentric and kinematic features of every pair, from pose keypoints.

    Produces a row-wise DataFrame with columns:
      - frame: the order column's values, frames or else times
      - perspective: 0 for A->B, 1 for B->A
      - id1, id2: pair identifiers
      - feature columns (e.g., A_speed, AB_dx_egoA, ...)
      - sequence, and group when the input has it

    This feature is *stateless* (no fitting). It computes features for all C(n,2)
    pairs per sequence, cleans/interpolates pose per animal, inner-joins by the
    chosen order column, and computes A->B and B->A features for each pair.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_egocentric.PairEgocentricFeatures.Params`.
    """

    category = "per-frame"
    name = "pair-egocentric"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.seq_col),
        ORDER_COLUMN,
        Keypoints(),
    )
    provides: ClassVar[Provides] = PAIR_EGOCENTRIC_PROVIDES

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        interpolation: Annotated[
            InterpolationConfig, Declared(INTERPOLATION_CONFIG_DESCRIPTION)
        ] = Field(default_factory=InterpolationConfig)
        sampling: Annotated[
            SamplingConfig, Declared(SAMPLING_WITH_SMOOTHING_DESCRIPTION)
        ] = Field(default_factory=SamplingConfig)
        pose: Annotated[PoseConfig, Declared(POSE_CONFIG_DESCRIPTION)] = Field(
            default_factory=PoseConfig
        )
        neck_idx: Annotated[int, Declared(_NECK_IDX_DESCRIPTION)] = 3
        tail_base_idx: Annotated[int, Declared(_TAIL_BASE_IDX_DESCRIPTION)] = 6
        center_mode: Annotated[
            str, Declared(_CENTER_MODE_DESCRIPTION, unwired=_CENTER_MODE_UNWIRED)
        ] = "mean"

    def __init__(
        self,
        inputs: PairEgocentricFeatures.Inputs = Inputs(("tracks",)),
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
        p = self.params
        x_cols, y_cols = p.pose.keypoint_columns()
        pose_cols = x_cols + y_cols
        [neck] = p.pose.resolve_keypoints(
            [p.neck_idx], who="[pair-egocentric] neck_idx"
        )
        [tail] = p.pose.resolve_keypoints(
            [p.tail_base_idx], who="[pair-egocentric] tail_base_idx"
        )
        order_col = resolve_order_col(df)

        required = [C.id_col, C.seq_col] + pose_cols
        ensure_columns(df, required)

        df_small = df[[order_col] + required]

        group_cols = [C.seq_col, C.id_col]

        df_small = clean_tracks_grouped(
            df_small, group_cols, pose_cols, order_col, p.interpolation
        )

        # Build dyads (all C(n,2) pairs per sequence)
        pairs: list[tuple[str, int, int]] = []
        for seq, gseq in df_small.groupby(C.seq_col):
            ids: list[int] = sorted(gseq[C.id_col].unique())
            if len(ids) >= 2:
                for id_a, id_b in combinations(ids, 2):
                    pairs.append((str(seq), int(id_a), int(id_b)))

        if not pairs:
            raise ValueError(
                "[pair-egocentric] No sequence with at least two IDs found."
            )

        out_frames: list[pd.DataFrame] = []
        for seq_key, id_a, id_b in pairs:
            gseq = df_small[df_small[C.seq_col] == seq_key]
            df_a = gseq[gseq[C.id_col] == id_a][[order_col] + pose_cols]
            df_b = gseq[gseq[C.id_col] == id_b][[order_col] + pose_cols]
            if df_a.empty or df_b.empty:
                continue

            df_a = df_a.sort_values(order_col).rename(columns={order_col: C.frame_col})
            df_b = df_b.sort_values(order_col).rename(columns={order_col: C.frame_col})
            j = df_a.merge(df_b, on=C.frame_col, suffixes=("_A", "_B"))
            if j.empty:
                continue

            # fps heuristic: prefer df['fps'] if present and constant; else default
            fps = p.sampling.fps_default
            if "fps" in df.columns:
                try:
                    c = df["fps"].dropna().unique()
                    if len(c) == 1:
                        fps = float(c[0])
                except Exception:
                    pass

            frames, AtoB, BtoA, names = self._build_ego_block_for_joined(
                j, fps, x_cols, y_cols, neck, tail
            )

            # produce row-wise DataFrames
            dfA = pd.DataFrame(AtoB.T, columns=names)
            dfA[C.frame_col] = frames
            dfA["perspective"] = 0
            dfA["id1"] = id_a
            dfA["id2"] = id_b

            dfB = pd.DataFrame(BtoA.T, columns=names)
            dfB[C.frame_col] = frames
            dfB["perspective"] = 1
            dfB["id1"] = id_b
            dfB["id2"] = id_a

            # optional pass-through for convenience (constant per call)
            for col in (C.seq_col, C.group_col):
                if col in df.columns:
                    dfA[col] = df[col].iloc[0]
                    dfB[col] = df[col].iloc[0]

            out_frames.extend([dfA, dfB])

        if not out_frames:
            return pd.DataFrame(columns=["perspective", C.frame_col, "id1", "id2"])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values([C.frame_col, "id1", "id2"]).reset_index(drop=True)
        return out

    # --- Internals ---
    def _safe_unit(
        self, vx: np.ndarray, vy: np.ndarray, eps: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray]:
        n = np.sqrt(vx * vx + vy * vy) + eps
        return vx / n, vy / n

    def _angle(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        return np.arctan2(vy, vx)

    def _center_from_points(
        self, xs: np.ndarray, ys: np.ndarray, mode: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(mode, (int, np.integer)):
            return xs[:, int(mode)], ys[:, int(mode)]
        return xs.mean(axis=1), ys.mean(axis=1)

    def _build_ego_block_for_joined(
        self,
        j: pd.DataFrame,
        fps: float,
        x_cols: list[str],
        y_cols: list[str],
        neck: int,
        tail: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Measure one pair from its joined rows.

        *neck* and *tail* are positions in *x_cols* and *y_cols*, as
        :meth:`PoseConfig.resolve_keypoints` returns them.
        """
        n_pts = len(x_cols)
        win = self.params.sampling.smooth_win
        mode = self.params.center_mode

        xa = j[[f"{column}_A" for column in x_cols]].to_numpy()
        ya = j[[f"{column}_A" for column in y_cols]].to_numpy()
        xb = j[[f"{column}_B" for column in x_cols]].to_numpy()
        yb = j[[f"{column}_B" for column in y_cols]].to_numpy()
        frames = j[C.frame_col].to_numpy()

        # optional smoothing
        if win and win > 1:
            xa = np.vstack([smooth_1d(xa[:, k], win) for k in range(n_pts)]).T
            ya = np.vstack([smooth_1d(ya[:, k], win) for k in range(n_pts)]).T
            xb = np.vstack([smooth_1d(xb[:, k], win) for k in range(n_pts)]).T
            yb = np.vstack([smooth_1d(yb[:, k], win) for k in range(n_pts)]).T

        # centers
        cxA, cyA = self._center_from_points(xa, ya, mode)
        cxB, cyB = self._center_from_points(xb, yb, mode)

        # headings (neck - tail) and units
        hxA, hyA = xa[:, neck] - xa[:, tail], ya[:, neck] - ya[:, tail]
        hxB, hyB = xb[:, neck] - xb[:, tail], yb[:, neck] - yb[:, tail]
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
        speedA = np.sqrt(vAx * vAx + vAy * vAy)
        speedB = np.sqrt(vBx * vBx + vBy * vBy)

        # heading angles + angular speed
        thA = self._angle(uhxA, uhyA)
        thB = self._angle(uhxB, uhyB)
        angspeedA = unwrap_diff(thA, fps)
        angspeedB = unwrap_diff(thB, fps)

        # ego projections of velocity
        vA_para = vAx * uhxA + vAy * uhyA
        vA_perp = vAx * uoxA + vAy * uoyA
        vB_para = vBx * uhxB + vBy * uhyB
        vB_perp = vBx * uoxB + vBy * uoyB

        # displacement A->B in world + A-centric ego coords of B
        dx = cxB - cxA
        dy = cyB - cyA
        distAB = np.sqrt(dx * dx + dy * dy)

        dxA = dx * uhxA + dy * uhyA
        dyA = dx * uoxA + dy * uoyA

        # B-centric ego coords of A
        dxB = (-dx) * uhxB + (-dy) * uhyB
        dyB = (-dx) * uoxB + (-dy) * uoyB

        # relative heading B wrt A
        dth = np.unwrap(thB) - np.unwrap(thA)
        rel_cos = np.cos(dth)
        rel_sin = np.sin(dth)

        names: list[str] = list(EGOCENTRIC_PAIR_COLUMNS)

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

        # For B->A, swap roles but keep same semantic ordering (B is 'self')
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
