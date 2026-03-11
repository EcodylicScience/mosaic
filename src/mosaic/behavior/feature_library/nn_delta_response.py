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
        speed_col: str = "speed"
        nn_id_col: str = "nn_id"
        nn_dx_ego_col: str = "nn_delta_x_ego"
        nn_dy_ego_col: str = "nn_delta_y_ego"
        nn_dx_world_col: str = "nn_delta_x"
        nn_dy_world_col: str = "nn_delta_y"
        focal_col: str = "Focal_fish"
        tag_cols: list[str] = Field(default_factory=list)
        diff_numframes: int = Field(default=4, ge=1)
        wrap_angle: bool = True
        divide_dangle_by_frames: bool = True
        scale_dangle_by_fps: bool = True
        divide_dspeed_by_frames: bool = True
        scale_dspeed_by_fps: bool = True

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
        x_col = COLUMNS.x_col
        y_col = COLUMNS.y_col
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
            return pd.DataFrame()

        try:
            order_col = resolve_order_col(df)
        except ValueError:
            return pd.DataFrame()

        # Sort by (id, order) so groupby().shift() operates within each fish
        df = df.sort_values([id_col, order_col]).reset_index(drop=True)

        diff_n = p.diff_numframes
        fps = p.sampling.fps

        # --- Vectorized shift and delta across ALL fish at once ---
        delta_cols = [x_col, y_col, angle_col, speed_col, frame_col]
        future = df.groupby(id_col, sort=False)[delta_cols].shift(-diff_n)
        delta = future - df[delta_cols]

        valid_mask = delta[frame_col].notna() & (delta[frame_col] == diff_n)
        if not valid_mask.any():
            return pd.DataFrame()

        # --- Build keep_cols list (same logic, applied once) ---
        keep_cols = [frame_col, id_col]
        for c in (COLUMNS.time_col, COLUMNS.group_col, COLUMNS.seq_col,
                  x_col, y_col, angle_col, speed_col, nn_id_col):
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)
        for c in (p.nn_dx_ego_col, p.nn_dy_ego_col,
                  p.nn_dx_world_col, p.nn_dy_world_col):
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)
        for c in ("group_size", "event", p.focal_col):
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)
        for c in p.tag_cols:
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)

        # --- Single slice for all valid rows ---
        out = df.loc[valid_mask, keep_cols].copy()
        out["dx"] = delta.loc[valid_mask, x_col].to_numpy()
        out["dy"] = delta.loc[valid_mask, y_col].to_numpy()
        out["dt"] = delta.loc[valid_mask, frame_col].to_numpy()

        dangle = delta.loc[valid_mask, angle_col].to_numpy()
        if p.wrap_angle:
            dangle = _wrap_angle(dangle)
        if p.divide_dangle_by_frames:
            dangle = dangle / diff_n
        if p.scale_dangle_by_fps:
            dangle = dangle * fps
        out["dangle"] = dangle

        dspeed = delta.loc[valid_mask, speed_col].to_numpy()
        if p.divide_dspeed_by_frames:
            dspeed = dspeed / diff_n
        if p.scale_dspeed_by_fps:
            dspeed = dspeed * fps
        out["dspeed"] = dspeed

        # --- Neighbor position in ego frame (vectorized) ---
        if p.nn_dx_ego_col in df.columns and p.nn_dy_ego_col in df.columns:
            out["neighbor_x"] = df.loc[valid_mask, p.nn_dx_ego_col].to_numpy()
            out["neighbor_y"] = df.loc[valid_mask, p.nn_dy_ego_col].to_numpy()
        elif p.nn_dx_world_col in df.columns and p.nn_dy_world_col in df.columns:
            dx_world = df.loc[valid_mask, p.nn_dx_world_col].to_numpy()
            dy_world = df.loc[valid_mask, p.nn_dy_world_col].to_numpy()
            heading = df.loc[valid_mask, angle_col].to_numpy()
            nx, ny = _ego_rotate(dx_world, dy_world, heading)
            out["neighbor_x"] = nx
            out["neighbor_y"] = ny
        else:
            out["neighbor_x"] = np.nan
            out["neighbor_y"] = np.nan

        # --- Rename nn_id_col to output name ---
        out["nn_id"] = out[nn_id_col].to_numpy()

        # --- Neighbor lookups: single indexed lookup instead of per-fish merges ---
        if p.focal_col in df.columns:
            focal_idx = df.set_index([frame_col, id_col])[p.focal_col]
            # Drop duplicates in the index (keep first) to avoid reindex errors
            focal_idx = focal_idx[~focal_idx.index.duplicated(keep="first")]
            keys = pd.MultiIndex.from_arrays(
                [out[frame_col].to_numpy(), out[nn_id_col].to_numpy()]
            )
            out["neighbor_focal"] = focal_idx.reindex(keys).values

        for tag_col in p.tag_cols:
            if tag_col in df.columns:
                tag_idx = df.set_index([frame_col, id_col])[tag_col]
                tag_idx = tag_idx[~tag_idx.index.duplicated(keep="first")]
                keys = pd.MultiIndex.from_arrays(
                    [out[frame_col].to_numpy(), out[nn_id_col].to_numpy()]
                )
                out[f"neighbor_{tag_col}"] = tag_idx.reindex(keys).values

        # --- Passthrough columns already in out from keep_cols ---

        # --- Stable column order ---
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
            if c in out.columns
        ]
        col_order += [
            c
            for c in ("neighbor_x", "neighbor_y", "neighbor_focal")
            if c in out.columns
        ]
        col_order += [
            c for c in ("dx", "dy", "dt", "dangle", "dspeed") if c in out.columns
        ]
        for c in ("group_size", "event", p.focal_col):
            if c in out.columns and c not in col_order:
                col_order.append(c)
        for c in p.tag_cols:
            if c in out.columns and c not in col_order:
                col_order.append(c)
            neighbor_c = f"neighbor_{c}"
            if neighbor_c in out.columns and neighbor_c not in col_order:
                col_order.append(neighbor_c)
        return out[col_order]
