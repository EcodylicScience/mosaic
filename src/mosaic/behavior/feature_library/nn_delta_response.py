from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
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


def _row_lags(g: pd.DataFrame, lag_col: str | None, default_lag: int) -> np.ndarray:
    """Per-row lag in frames for one individual's rows.

    ``default_lag`` everywhere unless ``lag_col`` names a column, in which case
    each finite value >= 1 wins (rounded to whole frames, since the lag indexes
    rows). Non-finite or sub-frame entries fall back rather than raising: a lag
    derived from a per-fish median speed is undefined for a fish with no valid
    speed, and that is a reason to use the nominal delay, not to drop the fish.
    """
    if lag_col is None:
        return np.full(len(g), default_lag, dtype=np.int64)
    raw = g[lag_col].to_numpy(dtype=float)
    lags = np.rint(raw)
    usable = np.isfinite(lags) & (lags >= 1)
    return np.where(usable, lags, default_lag).astype(np.int64)


def _windowed_delta(
    vals: np.ndarray, lags: np.ndarray, frame_idx: int, *, forward: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Difference across a per-row window, plus which rows have a valid one.

    ``vals`` is one individual's rows, already sorted, as ``(n, k)`` floats;
    ``lags`` is the per-row window in frames. Returns ``(delta, ok)`` where
    ``delta[i]`` is ``vals[i + lag] - vals[i]`` looking forward, or
    ``vals[i] - vals[i - lag]`` looking back.

    The gather is *positional* and then checked against the frame column, which
    is what makes a gap in the track fail: rows are contiguous in position but
    not necessarily in frame number, and a window that silently straddles a gap
    would report a displacement over the wrong elapsed time. With a uniform lag
    this reproduces ``shift(-lag)`` exactly, including its treatment of the
    ``lag`` rows at the boundary, which have no partner and are never valid.
    """
    n = len(vals)
    pos = np.arange(n)
    target = pos + lags if forward else pos - lags
    ok = (target >= 0) & (target < n)
    gathered = vals[np.where(ok, target, 0)]
    delta = gathered - vals if forward else vals - gathered
    # The frame column must advance by exactly the lag, or the window spans a gap.
    return delta, ok & (delta[:, frame_idx] == lags)


@final
@register_feature
class NearestNeighborDelta:
    """
    Per-sequence feature that measures how a focal individual changes position/heading/speed over
    the next `diff_numframes` frames relative to its nearest neighbor at the current frame.

    Expected inputs (via tracks or an Inputs() that merges tracks + nearest-neighbor feature):
      - position/heading/speed columns for the focal (`x`, `y`, `ANGLE`, `speed_col`)
      - nearest-neighbor id column (`nn_id_col`, default: 'nn_id')
      - neighbor offsets in ego frame (`nn_delta_x_ego` / `nn_delta_y_ego`); if missing, world
        offsets (`nn_delta_x` / `nn_delta_y`) are rotated using the focal heading.

    Outputs per focal row (filtered to frames with a valid future sample `diff_numframes` ahead):
      frame, id, group, sequence, nn_id, neighbor_x/y (ego), neighbor_focal (if available),
      dx, dy, dt, dangle (wrapped; optionally scaled by fps), dspeed, plus passthrough columns
      like group_size/event/Focal_fish when present. With `emit_backward`, also dx_back/dy_back
      (the same window run backwards, NaN where there is no valid past); with
      `diff_numframes_col`, also dt_frames (the per-row lag actually used).

    Params:
        sampling: Frame rate and smoothing settings. Default: SamplingConfig().
        speed_col: Column name for speed. Default: "SPEED".
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
        diff_numframes_col: Column holding a *per-row* lag in frames, used
            instead of `diff_numframes` wherever it is finite and >= 1. This
            is what expresses a speed-adjusted delay (`tau_i = tau_ref *
            S_ref / S_group`), which a single scalar cannot, because one
            feature run carries one parameter set across every entry. The
            column is built upstream, where the per-condition or per-fish
            mapping lives. Default: None (uniform lag, the fast path).
        emit_backward: If True, also emit the *backward* window
            (`dx_back`, `dy_back` = position at t minus position at t-tau).
            Rows whose backward window is invalid get NaN rather than being
            dropped, so a forward-only analysis keeps every row it had.
            Default: False.
        wrap_angle: If True, wrap heading differences to [-pi, pi].
            Default: True.
        divide_dangle_by_frames: If True, divide the heading change by
            diff_numframes. Default: True.
        scale_dangle_by_fps: If True, multiply dangle by fps to convert
            to radians/sec. Default: True.
        tag_cols: Additional columns to pass through to the output.
            Default: [].
    """

    category = "per-frame"
    name = "nn-delta-response"
    version = "0.3"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        sampling: SamplingConfig = Field(default_factory=SamplingConfig)
        speed_col: str = "SPEED"
        nn_id_col: str = "nn_id"
        nn_dx_ego_col: str = "nn_delta_x_ego"
        nn_dy_ego_col: str = "nn_delta_y_ego"
        nn_dx_world_col: str = "nn_delta_x"
        nn_dy_world_col: str = "nn_delta_y"
        focal_col: str = "Focal_fish"
        diff_numframes: int = Field(default=4, ge=1)
        diff_numframes_col: str | None = None
        emit_backward: bool = False
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
        missing: list[str] = []
        if speed_col is None:
            missing.append(
                f"a speed column ({p.speed_col!r}, or 'speed'). Tracks tables carry "
                "no speed of their own -- it is derived, so it belongs to a "
                "feature: run 'speed-angvel' first, or point speed_col= at a "
                "column this table actually has"
            )
        if nn_id_col is None:
            missing.append(f"a neighbor id column ({p.nn_id_col!r}, or 'nn_fishID')")
        if C.frame_col not in df:
            missing.append(repr(C.frame_col))
        if C.id_col not in df:
            missing.append(repr(C.id_col))
        if missing:
            raise ValueError(
                f"{self.name} cannot run on this table: it needs "
                + "; ".join(missing)
                + ". This used to return an empty frame instead, so a run over a "
                "table missing one of these reported success having computed "
                "nothing."
            )

        order_col = resolve_order_col(df)
        df = df.sort_values([order_col, C.id_col]).reset_index(drop=True)

        diff_n = p.diff_numframes
        fps = p.sampling.fps_default

        # Optional neighbor focal lookup (frame + id -> focal flag)
        focal_lookup = None
        if p.focal_col in df.columns:
            focal_lookup = df[[C.frame_col, C.id_col, p.focal_col]].rename(
                columns={C.id_col: "_nid", p.focal_col: "neighbor_focal"}
            )

        lag_col = (
            p.diff_numframes_col
            if p.diff_numframes_col and p.diff_numframes_col in df.columns
            else None
        )
        delta_cols = [C.x_col, C.y_col, C.orientation_col, speed_col, C.frame_col]
        i_x, i_y, i_ang, i_spd, i_frm = range(5)

        outputs: list[pd.DataFrame] = []
        for focal_id, g in df.groupby(C.id_col, sort=False):
            g = g.sort_values(order_col)
            vals = g[delta_cols].to_numpy(dtype=float)
            lags = _row_lags(g, lag_col, diff_n)

            fwd, ok_fwd = _windowed_delta(vals, lags, i_frm, forward=True)
            valid_mask = pd.Series(ok_fwd, index=g.index)
            if not ok_fwd.any():
                continue

            rows = g.loc[valid_mask].copy()
            sel_lags = lags[ok_fwd]
            rows["dx"] = fwd[ok_fwd, i_x]
            rows["dy"] = fwd[ok_fwd, i_y]
            rows["dt"] = fwd[ok_fwd, i_frm]
            dangle = fwd[ok_fwd, i_ang]
            if p.wrap_angle:
                dangle = wrap_angle(dangle)
            if p.divide_dangle_by_frames:
                dangle = dangle / sel_lags
            if p.scale_dangle_by_fps:
                dangle = dangle * fps
            rows["dangle"] = dangle
            rows["dspeed"] = fwd[ok_fwd, i_spd]
            if lag_col is not None:
                rows["dt_frames"] = sel_lags

            # Backward window: the same gather run the other way. Model B's target
            # is |r(t+tau) - r(t)| - |r(t) - r(t-tau)|, so it needs both arms; rows
            # with no valid past (the first tau of a track, or across a frame gap)
            # get NaN so they drop out of Model B alone, not out of Model A too.
            if p.emit_backward:
                back, ok_back = _windowed_delta(vals, lags, i_frm, forward=False)
                back = np.where(ok_back[:, None], back, np.nan)
                rows["dx_back"] = back[ok_fwd, i_x]
                rows["dy_back"] = back[ok_fwd, i_y]

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
            c
            for c in (
                "dx",
                "dy",
                "dt",
                "dangle",
                "dspeed",
                "dx_back",
                "dy_back",
                "dt_frames",
            )
            if c in out_df.columns
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
