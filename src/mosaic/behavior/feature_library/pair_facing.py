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
See ``PairFacing.Params.body_axis_from`` for the convention and its known
values.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Annotated, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    TrackInputs,
    resolve_order_col,
)
from mosaic.core.params import Declared, Params

from .helpers import ensure_columns
from .registry import register_feature


def _wrap180_deg(angle: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to [-180, 180]. Mirrors Valerie's wrap180."""
    return (angle + 180.0) % 360.0 - 180.0


_ANGLE_THRESH_DEG_DESCRIPTION = "The maximum angle_diff_deg for is_facing to be true."

_DIST_THRESH_DESCRIPTION = (
    "The maximum distance for is_facing to be true, in the unit distance "
    "is reported in: centimeters when cm_per_pixel is set, pixels "
    "otherwise."
)

_CM_PER_PIXEL_DESCRIPTION = (
    "The scale factor applied to the head-to-head distance. Unset, "
    "distance and dist_thresh are read directly from the tracks with no "
    "scaling applied."
)

_POSE_HEAD_INDEX_DESCRIPTION = (
    "The pose keypoint index for the head. The default of 3 assumes the "
    "TRex bee layout described in the class docstring."
)

_POSE_ABDOMEN_INDEX_DESCRIPTION = (
    "The pose keypoint index for the abdomen tip. The default of 5 "
    "assumes the TRex bee layout described in the class docstring."
)

_BODY_AXIS_FROM_DESCRIPTION = (
    "The direction convention for the body axis. Known values are "
    "head_to_abdomen and abdomen_to_head. head_to_abdomen reproduces the "
    "original BeesInADish computation, under which is_facing triggers "
    "when the abdomen points toward the target. abdomen_to_head flips the "
    "sign so is_facing triggers when the head points toward the target."
)

_X_PREFIX_DESCRIPTION = "The column name prefix for pose X coordinates."

_Y_PREFIX_DESCRIPTION = "The column name prefix for pose Y coordinates."


@final
@register_feature
class PairFacing:
    """
    Per-frame directional facing metric for all ordered pairs of individuals.

    Output columns (one row per ordered pair per frame, keyed by
    ``(frame, id1, id2, perspective)``):
      - frame
      - id1 (the focal), id2 (the target), perspective
      - body_angle_deg
      - bearing_deg
      - angle_diff_deg
      - distance        (cm if cm_per_pixel set, else px)
      - is_facing       (bool)

    Assumes the TRex bee keypoint layout: 0=L-antenna, 1=R-antenna,
    2=proboscis, 3=head, 4=thorax, 5=abdomen tip. pose_head_index and
    pose_abdomen_index default to 3 and 5 under that layout.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_facing.PairFacing.Params`.
    """

    category = "per-frame"
    name = "pair-facing"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        angle_thresh_deg: Annotated[
            float, Declared(_ANGLE_THRESH_DEG_DESCRIPTION, unit="deg")
        ] = 45.0
        dist_thresh: Annotated[float, Declared(_DIST_THRESH_DESCRIPTION)] = 3.0
        cm_per_pixel: Annotated[
            float | None, Declared(_CM_PER_PIXEL_DESCRIPTION, unit="cm/px")
        ] = None
        pose_head_index: Annotated[int, Declared(_POSE_HEAD_INDEX_DESCRIPTION)] = 3
        pose_abdomen_index: Annotated[
            int, Declared(_POSE_ABDOMEN_INDEX_DESCRIPTION)
        ] = 5
        body_axis_from: Annotated[
            str,
            Field(examples=["head_to_abdomen", "abdomen_to_head"]),
            Declared(_BODY_AXIS_FROM_DESCRIPTION),
        ] = "head_to_abdomen"
        x_prefix: Annotated[str, Declared(_X_PREFIX_DESCRIPTION)] = "poseX"
        y_prefix: Annotated[str, Declared(_Y_PREFIX_DESCRIPTION)] = "poseY"

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

        iter_groups = df.groupby(C.seq_col) if C.seq_col in df.columns else [(None, df)]

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
                    per_id[focal_id],
                    per_id[target_id],
                    head_x,
                    head_y,
                    abd_x,
                    abd_y,
                )
                if pair_df is None or pair_df.empty:
                    continue
                pair_df["id1"] = focal_id
                pair_df["id2"] = target_id
                # The focal-first ordering is perspective 0; its mirror is 1.
                pair_df["perspective"] = 0 if focal_id < target_id else 1
                all_rows.append(pair_df)

        if not all_rows:
            return self._empty_output()

        out = pd.concat(all_rows, ignore_index=True)

        for col in (C.group_col, C.seq_col):
            if col in df.columns:
                out[col] = df[col].iloc[0]

        cols = [
            C.frame_col,
            "id1",
            "id2",
            "perspective",
            "body_angle_deg",
            "bearing_deg",
            "angle_diff_deg",
            "distance",
            "is_facing",
        ]
        extra = [c for c in out.columns if c not in cols]
        return out[cols + extra]

    def _compute_directed_pair(
        self,
        df_f: pd.DataFrame,
        df_t: pd.DataFrame,
        head_x: str,
        head_y: str,
        abd_x: str,
        abd_y: str,
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
            (angle_diff < p.angle_thresh_deg) & (distance < p.dist_thresh) & valid
        )

        return pd.DataFrame(
            {
                C.frame_col: merged[C.frame_col].to_numpy(),
                "body_angle_deg": body_angle,
                "bearing_deg": bearing,
                "angle_diff_deg": angle_diff,
                "distance": distance,
                "is_facing": is_facing,
            }
        )

    def _empty_output(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                C.frame_col,
                "id1",
                "id2",
                "perspective",
                "body_angle_deg",
                "bearing_deg",
                "angle_diff_deg",
                "distance",
                "is_facing",
            ]
        )
