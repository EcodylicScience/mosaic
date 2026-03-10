from __future__ import annotations

from pathlib import Path
from typing import Iterable, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.dataset import register_feature

from .params import (
    COLUMNS,
    Inputs,
    OutputType,
    Params,
    Result,
    SamplingConfig,
    TrackInput,
    resolve_order_col,
)


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _ego_rotate(
    dx: np.ndarray, dy: np.ndarray, heading: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate world-frame deltas into the ego frame of the focal (heading aligned with +x).
    """
    ct = np.cos(heading)
    st = np.sin(heading)
    dx_ego = dx * ct + dy * st
    dy_ego = -dx * st + dy * ct
    return dx_ego, dy_ego


@final
@register_feature
class NearestNeighborDelta:
    """
    Per-sequence feature that measures how a focal fish changes position/heading/speed over
    the next `diff_numframes` frames relative to its nearest neighbor at the current frame.

    Expected inputs (via tracks or an inputset that merges tracks + nearest-neighbor feature):
      - position/heading/speed columns for the focal (`x`, `y`, `ANGLE`, `speed_col`)
      - nearest-neighbor id column (`nn_id_col`, default: 'nn_id')
      - neighbor offsets in ego frame (`nn_delta_x_ego` / `nn_delta_y_ego`); if missing, world
        offsets (`nn_delta_x` / `nn_delta_y`) are rotated using the focal heading.

    Outputs per focal row (filtered to frames with a valid future sample `diff_numframes` ahead):
      frame, id, group, sequence, nn_id, neighbor_x/y (ego), neighbor_focal (if available),
      dx, dy, dt, dangle (wrapped; optionally scaled by fps), dspeed, plus passthrough columns
      like group_size/event/Focal_fish when present.
    """

    name = "nn-delta-response"
    version = "0.1"
    parallelizable = True
    output_type: OutputType = "per_frame"

    class Inputs(Inputs[TrackInput | Result]):
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

    def __init__(
        self,
        inputs: NearestNeighborDelta.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False
        self._ds = None
        self._scope_filter: dict[str, object] | None = None

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path: Path) -> None:
        return

    def load_model(self, path: Path) -> None:
        return

    # ----------------------- Core logic --------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params

        # Resolve required columns with a few fallbacks
        id_col = COLUMNS.id_col
        frame_col = COLUMNS.frame_col
        angle_col = COLUMNS.orientation_col
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
            or frame_col not in df
            or id_col not in df
        ):
            # Missing critical inputs; nothing to compute.
            return pd.DataFrame()

        # Order for reproducibility
        try:
            order_col = resolve_order_col(df)
        except ValueError:
            return pd.DataFrame()
        df = df.sort_values([order_col, id_col]).reset_index(drop=True)

        diff_n = p.diff_numframes
        wrap_angles = p.wrap_angle
        divide_by_frames = p.divide_dangle_by_frames
        scale_by_fps = p.scale_dangle_by_fps
        fps = p.sampling.fps_default

        # Optional neighbor focal lookup (frame + id -> focal flag)
        focal_lookup = None
        if p.focal_col in df.columns:
            focal_lookup = df[[frame_col, id_col, p.focal_col]].rename(
                columns={id_col: "_nid", p.focal_col: "neighbor_focal"}
            )

        outputs = []
        for focal_id, g in df.groupby(id_col, sort=False):
            g = g.sort_values(order_col)
            # Future samples diff_numframes ahead
            future = g[
                [COLUMNS.x_col, COLUMNS.y_col, angle_col, speed_col, frame_col]
            ].shift(-diff_n)
            delta = (
                future
                - g[[COLUMNS.x_col, COLUMNS.y_col, angle_col, speed_col, frame_col]]
            )

            valid_mask = delta[frame_col].notna() & (delta[frame_col] == diff_n)
            if not valid_mask.any():
                continue

            # Only keep columns needed for output (meta + source for computation)
            keep_cols = [frame_col, id_col]
            for c in (COLUMNS.time_col, COLUMNS.group_col, COLUMNS.seq_col,
                      COLUMNS.x_col, COLUMNS.y_col, angle_col, speed_col, nn_id_col):
                if c in g.columns and c not in keep_cols:
                    keep_cols.append(c)
            for c in (p.nn_dx_ego_col, p.nn_dy_ego_col,
                      p.nn_dx_world_col, p.nn_dy_world_col):
                if c in g.columns and c not in keep_cols:
                    keep_cols.append(c)
            for c in ("group_size", "event", p.focal_col):
                if c in g.columns and c not in keep_cols:
                    keep_cols.append(c)
            rows = g.loc[valid_mask, keep_cols].copy()
            rows["dx"] = delta.loc[valid_mask, COLUMNS.x_col].to_numpy()
            rows["dy"] = delta.loc[valid_mask, COLUMNS.y_col].to_numpy()
            rows["dt"] = delta.loc[valid_mask, frame_col].to_numpy()
            dangle = delta.loc[valid_mask, angle_col].to_numpy()
            if wrap_angles:
                dangle = _wrap_angle(dangle)
            if divide_by_frames:
                dangle = dangle / diff_n
            if scale_by_fps:
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
                heading = g.loc[valid_mask, angle_col].to_numpy()
                nx, ny = _ego_rotate(dx_world, dy_world, heading)
                rows["neighbor_x"] = nx
                rows["neighbor_y"] = ny
            else:
                rows["neighbor_x"] = np.nan
                rows["neighbor_y"] = np.nan

            # Neighbor focal flag (if available)
            if focal_lookup is not None:
                neighbor_meta = rows[[frame_col, nn_id_col]].rename(
                    columns={nn_id_col: "_nid"}
                )
                rows["neighbor_focal"] = neighbor_meta.merge(
                    focal_lookup, on=[frame_col, "_nid"], how="left"
                )["neighbor_focal"].to_numpy()

            # Pass through meta columns
            rows["nn_id"] = g.loc[valid_mask, nn_id_col].to_numpy()
            rows[id_col] = focal_id
            if COLUMNS.group_col in g.columns:
                rows[COLUMNS.group_col] = g.loc[
                    valid_mask, COLUMNS.group_col
                ].to_numpy()
            if COLUMNS.seq_col in g.columns:
                rows[COLUMNS.seq_col] = g.loc[valid_mask, COLUMNS.seq_col].to_numpy()
            if COLUMNS.time_col in g.columns:
                rows[COLUMNS.time_col] = g.loc[valid_mask, COLUMNS.time_col].to_numpy()
            for passthrough in ("group_size", "event", p.focal_col):
                if passthrough in g.columns and passthrough not in rows.columns:
                    rows[passthrough] = g.loc[valid_mask, passthrough].to_numpy()

            outputs.append(rows)

        if not outputs:
            return pd.DataFrame()

        out_df = pd.concat(outputs, ignore_index=True)
        # Ensure stable column order: meta first, then deltas and neighbor info
        col_order = [
            c
            for c in (
                frame_col,
                COLUMNS.time_col,
                COLUMNS.group_col,
                COLUMNS.seq_col,
                id_col,
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
        return out_df[col_order]
