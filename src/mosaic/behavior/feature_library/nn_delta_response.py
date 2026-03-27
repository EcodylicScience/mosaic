from __future__ import annotations

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
    Result,
    TrackInput,
    resolve_order_col,
)

from .helpers import ego_rotate, wrap_angle
from .registry import register_feature
from .types import SamplingConfig


@final
@register_feature
class NearestNeighborDelta:
    """
    Per-sequence feature that measures how a focal fish changes position/heading/speed over
    the next `diff_numframes` frames relative to its nearest neighbor at the current frame.

    Expected inputs (via tracks or an Inputs() that merges tracks + nearest-neighbor feature):
      - position/heading/speed columns for the focal (`x`, `y`, `ANGLE`, `speed_col`)
      - nearest-neighbor id column (`nn_id_col`, default: 'nn_id')
      - neighbor offsets in ego frame (`nn_delta_x_ego` / `nn_delta_y_ego`); if missing, world
        offsets (`nn_delta_x` / `nn_delta_y`) are rotated using the focal heading.

    Outputs per focal row (filtered to frames with a valid future sample `diff_numframes` ahead):
      frame, id, group, sequence, nn_id, neighbor_x/y (ego), neighbor_focal (if available),
      dx, dy, dt, dangle (wrapped; optionally scaled by fps), dspeed, plus passthrough columns
      like group_size/event/Focal_fish when present.

    Params:
        sampling: Frame rate and smoothing settings. Default: SamplingConfig().
        speed_col: Column name for speed. Default: "SPEED#wcentroid".
        nn_id_col: Column name for the nearest-neighbor ID.
            Default: "nn_id".
        nn_dx_ego_col: Column for neighbor delta-x in ego frame.
            Default: "nn_delta_x_ego".
        nn_dy_ego_col: Column for neighbor delta-y in ego frame.
            Default: "nn_delta_y_ego".
        nn_dx_world_col: Fallback column for neighbor delta-x in world
            frame (used when ego columns are absent).
            Default: "nn_delta_x".
        nn_dy_world_col: Fallback column for neighbor delta-y in world
            frame. Default: "nn_delta_y".
        focal_col: Column name for the focal-animal flag.
            Default: "Focal_fish".
        diff_numframes: Number of frames ahead to compute the future
            response delta. Default: 4.
        wrap_angle: If True, wrap heading differences to [-pi, pi].
            Default: True.
        divide_dangle_by_frames: If True, divide the heading change by
            diff_numframes. Default: True.
        scale_dangle_by_fps: If True, multiply dangle by fps to convert
            to radians/sec. Default: True.
        tag_cols: Additional columns to pass through to the output.
            Default: [].
    """

    name = "nn-delta-response"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        sampling: SamplingConfig = Field(default_factory=SamplingConfig)
        speed_col: str = "SPEED#wcentroid"
        nn_id_col: str = "nn_id"
        nn_dx_ego_col: str = "nn_delta_x_ego"
        nn_dy_ego_col: str = "nn_delta_y_ego"
        nn_dx_world_col: str = "nn_delta_x"
        nn_dy_world_col: str = "nn_delta_y"
        focal_col: str = "Focal_fish"
        diff_numframes: int = Field(default=4, ge=1)
        wrap_angle: bool = True
        divide_dangle_by_frames: bool = True
        scale_dangle_by_fps: bool = True
        tag_cols: list[str] = Field(default_factory=list)

    def __init__(
        self,
        inputs: NearestNeighborDelta.Inputs = Inputs(("tracks",)),
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

        # Resolve required columns with a few fallbacks
        speed_col = (
            p.speed_col
            if p.speed_col in df.columns
            else ("speed" if "speed" in df.columns else None)
        )
        nn_id_col = (
            p.nn_id_col
            if p.nn_id_col in df.columns
            else ("nn_fishID" if "nn_fishID" in df.columns else None)
        )
        if (
            speed_col is None
            or nn_id_col is None
            or C.frame_col not in df
            or C.id_col not in df
        ):
            return pd.DataFrame()

        try:
            order_col = resolve_order_col(df)
        except ValueError:
            return pd.DataFrame()
        df = df.sort_values([order_col, C.id_col]).reset_index(drop=True)

        diff_n = p.diff_numframes
        fps = p.sampling.fps_default

        # Optional neighbor focal lookup (frame + id -> focal flag)
        focal_lookup = None
        if p.focal_col in df.columns:
            focal_lookup = df[[C.frame_col, C.id_col, p.focal_col]].rename(
                columns={C.id_col: "_nid", p.focal_col: "neighbor_focal"}
            )

        outputs: list[pd.DataFrame] = []
        for focal_id, g in df.groupby(C.id_col, sort=False):
            g = g.sort_values(order_col)
            future = g[
                [C.x_col, C.y_col, C.orientation_col, speed_col, C.frame_col]
            ].shift(-diff_n)
            delta = (
                future
                - g[[C.x_col, C.y_col, C.orientation_col, speed_col, C.frame_col]]
            )

            valid_mask = delta[C.frame_col].notna() & (delta[C.frame_col] == diff_n)
            if not valid_mask.any():
                continue

            rows = g.loc[valid_mask].copy()
            rows["dx"] = delta.loc[valid_mask, C.x_col].to_numpy()
            rows["dy"] = delta.loc[valid_mask, C.y_col].to_numpy()
            rows["dt"] = delta.loc[valid_mask, C.frame_col].to_numpy()
            dangle = delta.loc[valid_mask, C.orientation_col].to_numpy()
            if p.wrap_angle:
                dangle = wrap_angle(dangle)
            if p.divide_dangle_by_frames:
                dangle = dangle / diff_n
            if p.scale_dangle_by_fps:
                dangle = dangle * fps
            rows["dangle"] = dangle
            rows["dspeed"] = delta.loc[valid_mask, speed_col].to_numpy()

            # Neighbor position in ego frame: prefer existing ego offsets, else rotate world offsets
            if p.nn_dx_ego_col in g.columns and p.nn_dy_ego_col in g.columns:
                rows["neighbor_x"] = g.loc[valid_mask, p.nn_dx_ego_col].to_numpy()
                rows["neighbor_y"] = g.loc[valid_mask, p.nn_dy_ego_col].to_numpy()
            elif p.nn_dx_world_col in g.columns and p.nn_dy_world_col in g.columns:
                dx_world = g.loc[valid_mask, p.nn_dx_world_col].to_numpy()
                dy_world = g.loc[valid_mask, p.nn_dy_world_col].to_numpy()
                heading = g.loc[valid_mask, C.orientation_col].to_numpy()
                nx, ny = ego_rotate(dx_world, dy_world, heading)
                rows["neighbor_x"] = nx
                rows["neighbor_y"] = ny
            else:
                rows["neighbor_x"] = np.nan
                rows["neighbor_y"] = np.nan

            # Neighbor focal flag (if available)
            if focal_lookup is not None:
                neighbor_meta = rows[[C.frame_col, nn_id_col]].rename(
                    columns={nn_id_col: "_nid"}
                )
                rows["neighbor_focal"] = neighbor_meta.merge(
                    focal_lookup, on=[C.frame_col, "_nid"], how="left"
                )["neighbor_focal"].to_numpy()

            rows["nn_id"] = g.loc[valid_mask, nn_id_col].to_numpy()
            rows[C.id_col] = focal_id
            if C.group_col in g.columns:
                rows[C.group_col] = g.loc[valid_mask, C.group_col].to_numpy()
            if C.seq_col in g.columns:
                rows[C.seq_col] = g.loc[valid_mask, C.seq_col].to_numpy()
            if C.time_col in g.columns:
                rows[C.time_col] = g.loc[valid_mask, C.time_col].to_numpy()
            for passthrough in ("group_size", "event", p.focal_col):
                if passthrough in g.columns and passthrough not in rows.columns:
                    rows[passthrough] = g.loc[valid_mask, passthrough].to_numpy()

            # Tag columns: passthrough + neighbor lookup
            for tag_col in p.tag_cols:
                if tag_col in g.columns and tag_col not in rows.columns:
                    rows[tag_col] = g.loc[valid_mask, tag_col].to_numpy()
                if tag_col in df.columns:
                    tag_lookup = df[[C.frame_col, C.id_col, tag_col]].rename(
                        columns={C.id_col: "_nid", tag_col: f"neighbor_{tag_col}"}
                    )
                    neighbor_meta = rows[[C.frame_col, nn_id_col]].rename(
                        columns={nn_id_col: "_nid"}
                    )
                    rows[f"neighbor_{tag_col}"] = neighbor_meta.merge(
                        tag_lookup, on=[C.frame_col, "_nid"], how="left"
                    )[f"neighbor_{tag_col}"].to_numpy()

            outputs.append(rows)

        if not outputs:
            return pd.DataFrame()

        out_df = pd.concat(outputs, ignore_index=True)
        col_order = [
            c
            for c in (
                C.frame_col,
                C.time_col,
                C.group_col,
                C.seq_col,
                C.id_col,
                "nn_id",
            )
            if c in out_df.columns
        ]
        col_order += [
            c
            for c in ("neighbor_x", "neighbor_y", "neighbor_focal")
            if c in out_df.columns
        ]
        col_order += [
            c for c in ("dx", "dy", "dt", "dangle", "dspeed") if c in out_df.columns
        ]
        for c in ("group_size", "event", p.focal_col):
            if c in out_df.columns and c not in col_order:
                col_order.append(c)
        for tag_col in p.tag_cols:
            for c in (tag_col, f"neighbor_{tag_col}"):
                if c in out_df.columns and c not in col_order:
                    col_order.append(c)
        col_order += [c for c in out_df.columns if c not in col_order]
        return out_df[col_order]
