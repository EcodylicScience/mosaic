"""
PairFacing -- per-frame directional facing metric for all ordered pairs.

Ported from Valerie's BeesInADish ``Dish.compute_social_facing``
(Apis/BeesInADish/object_trex.py). For every ordered (focal, target) pair
of individuals in a sequence, computes:

  - body_angle_deg:  focal's body-axis angle (head -> abdomen by default), deg
  - bearing_deg:     direction from focal head to target head, deg
  - angle_diff_deg:  |wrap180(bearing - body_angle)|, deg
  - distance:        head-to-head euclidean distance (cm if cm_per_pixel set,
                     else px)
  - is_facing:       (angle_diff_deg < angle_thresh_deg) AND
                     (distance < dist_thresh)

Unlike PairInteractionFilter (which checks *mutual* mouth-to-mouth orientation
for trophallaxis detection), PairFacing is **directional** -- one row per
ordered (focal, target) pair per frame. It is the building block for the
AttentionTarget feature.

Body-axis convention
--------------------
With the default ``body_axis_from="head_to_abdomen"`` the body_angle is the
vector from the focal's head keypoint to its abdomen keypoint. This matches
Valerie's original code byte-for-byte. Note that under this convention
``is_facing`` triggers when the abdomen (not the head) is pointing toward
the target. Set ``body_axis_from="abdomen_to_head"`` to flip the sign so
that ``is_facing`` triggers on head-toward-target (the conventional reading).
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd

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

from .helpers import ensure_columns
from .registry import register_feature


def _wrap180_deg(angle: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to [-180, 180]. Mirrors Valerie's wrap180."""
    return (angle + 180.0) % 360.0 - 180.0


@final
@register_feature
class PairFacing:
    """
    Per-frame directional facing metric for all ordered pairs of individuals.

    Output columns (one row per ordered (focal, target) per frame):
      - frame
      - focal_id, target_id
      - body_angle_deg
      - bearing_deg
      - angle_diff_deg
      - distance        (cm if cm_per_pixel set, else px)
      - is_facing       (bool)

    Params
    ------
    angle_thresh_deg : float
        Max angle_diff_deg for is_facing. Default 45.0 (Valerie's value).
    dist_thresh : float
        Max distance for is_facing. Default 3.0 (cm if cm_per_pixel set,
        else px).
    cm_per_pixel : float | None
        Scale factor. When set, distance is reported in cm and dist_thresh
        is interpreted as cm. Default None (distance stays in px).
    pose_head_index : int
        Pose keypoint index for the head. Default 3 (TRex bee layout:
        0=L-antenna, 1=R-antenna, 2=proboscis, 3=head, 4=thorax,
        5=abdomen tip).
    pose_abdomen_index : int
        Pose keypoint index for the abdomen tip. Default 5.
    body_axis_from : Literal["head_to_abdomen", "abdomen_to_head"]
        Direction convention for the body axis. Default "head_to_abdomen"
        reproduces Valerie's compute_social_facing exactly. Use
        "abdomen_to_head" to flip so is_facing means "head pointed at
        target".
    x_prefix, y_prefix : str
        Pose column prefixes (default "poseX", "poseY", matching PoseConfig).
    """

    name = "pair-facing"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        angle_thresh_deg: float = 45.0
        dist_thresh: float = 3.0
        cm_per_pixel: float | None = None
        pose_head_index: int = 3
        pose_abdomen_index: int = 5
        body_axis_from: str = "head_to_abdomen"
        x_prefix: str = "poseX"
        y_prefix: str = "poseY"

    def __init__(
        self,
        inputs: PairFacing.Inputs = Inputs(("tracks",)),
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
            return self._empty_output()

        p = self.params
        if p.body_axis_from not in {"head_to_abdomen", "abdomen_to_head"}:
            raise ValueError(
                f"body_axis_from must be 'head_to_abdomen' or "
                f"'abdomen_to_head', got {p.body_axis_from!r}"
            )

        order_col = resolve_order_col(df)
        head_x = f"{p.x_prefix}{p.pose_head_index}"
        head_y = f"{p.y_prefix}{p.pose_head_index}"
        abd_x = f"{p.x_prefix}{p.pose_abdomen_index}"
        abd_y = f"{p.y_prefix}{p.pose_abdomen_index}"
        ensure_columns(df, [C.id_col, head_x, head_y, abd_x, abd_y])

        all_rows: list[pd.DataFrame] = []

        iter_groups = (
            df.groupby(C.seq_col) if C.seq_col in df.columns else [(None, df)]
        )

        for _, gseq in iter_groups:
            ids = sorted(gseq[C.id_col].unique())
            if len(ids) < 2:
                continue

            per_id: dict[object, pd.DataFrame] = {}
            for animal_id in ids:
                sub = (
                    gseq[gseq[C.id_col] == animal_id][
                        [order_col, head_x, head_y, abd_x, abd_y]
                    ]
                    .sort_values(order_col)
                    .rename(columns={order_col: C.frame_col})
                )
                per_id[animal_id] = sub

            for focal_id, target_id in permutations(ids, 2):
                pair_df = self._compute_directed_pair(
                    per_id[focal_id], per_id[target_id],
                    head_x, head_y, abd_x, abd_y,
                )
                if pair_df is None or pair_df.empty:
                    continue
                pair_df["focal_id"] = focal_id
                pair_df["target_id"] = target_id
                all_rows.append(pair_df)

        if not all_rows:
            return self._empty_output()

        out = pd.concat(all_rows, ignore_index=True)

        for col in (C.group_col, C.seq_col):
            if col in df.columns:
                out[col] = df[col].iloc[0]

        cols = [
            C.frame_col, "focal_id", "target_id",
            "body_angle_deg", "bearing_deg", "angle_diff_deg",
            "distance", "is_facing",
        ]
        extra = [c for c in out.columns if c not in cols]
        return out[cols + extra]

    def _compute_directed_pair(
        self,
        df_f: pd.DataFrame,
        df_t: pd.DataFrame,
        head_x: str, head_y: str, abd_x: str, abd_y: str,
    ) -> pd.DataFrame | None:
        merged = df_f.merge(
            df_t[[C.frame_col, head_x, head_y]],
            on=C.frame_col,
            suffixes=("_f", "_t"),
        )
        if merged.empty:
            return None

        p = self.params

        fx = merged[f"{head_x}_f"].to_numpy(dtype=float)
        fy = merged[f"{head_y}_f"].to_numpy(dtype=float)
        ax = merged[abd_x].to_numpy(dtype=float)
        ay = merged[abd_y].to_numpy(dtype=float)
        tx = merged[f"{head_x}_t"].to_numpy(dtype=float)
        ty = merged[f"{head_y}_t"].to_numpy(dtype=float)

        if p.body_axis_from == "head_to_abdomen":
            body_angle = np.degrees(np.arctan2(ay - fy, ax - fx))
        else:
            body_angle = np.degrees(np.arctan2(fy - ay, fx - ax))

        bearing = np.degrees(np.arctan2(ty - fy, tx - fx))
        angle_diff = np.abs(_wrap180_deg(bearing - body_angle))

        distance_px = np.hypot(tx - fx, ty - fy)
        if p.cm_per_pixel is not None:
            distance = distance_px * float(p.cm_per_pixel)
        else:
            distance = distance_px

        valid = np.isfinite(angle_diff) & np.isfinite(distance)
        is_facing = (
            (angle_diff < p.angle_thresh_deg)
            & (distance < p.dist_thresh)
            & valid
        )

        return pd.DataFrame({
            C.frame_col: merged[C.frame_col].to_numpy(),
            "body_angle_deg": body_angle,
            "bearing_deg": bearing,
            "angle_diff_deg": angle_diff,
            "distance": distance,
            "is_facing": is_facing,
        })

    def _empty_output(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                C.frame_col, "focal_id", "target_id",
                "body_angle_deg", "bearing_deg", "angle_diff_deg",
                "distance", "is_facing",
            ]
        )
