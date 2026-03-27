from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from mosaic.core.pipeline.types import (
    COLUMNS,
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    TrackInput,
    resolve_order_col,
)

from .registry import register_feature


def _pose_column_pairs(columns) -> list[tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs from column names."""
    pose_pairs = []
    xs = [c for c in columns if c.startswith("poseX")]
    for x_col in sorted(xs):
        idx = x_col[5:]
        y_col = f"poseY{idx}"
        if y_col in columns:
            pose_pairs.append((x_col, y_col))
    return pose_pairs


def _savgol_with_nan(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay filter, handling NaN/inf values."""
    bad = ~np.isfinite(arr)
    filled = arr.copy()
    filled[bad] = 0.0
    smoothed = savgol_filter(filled, window_length=window, polyorder=polyorder)
    smoothed[bad] = np.nan
    return smoothed


@final
@register_feature
class TrajectorySmooth:
    """
    Per-sequence feature that smooths and interpolates trajectory positions.

    Pipeline (per individual):
      1. Bad-frame detection: flag frames with speed > speed_threshold,
         expand flagged region by expand_frames in each direction.
      2. Interpolation: set positions to NaN at bad frames, linearly
         interpolate, forward/backward fill edges. Controlled separately
         for centroid (interpolate_centroid) and pose (interpolate_pose).
      3. Savgol smoothing: apply savgol_filter to centroid X/Y and all
         pose columns (always, regardless of interpolation flags).

    Output is the full track DataFrame with smoothed positions replacing
    originals, plus a ``bad_frame`` boolean column. Downstream features
    can consume this via ``Inputs((Result(feature="trajectory-smooth"),))``.

    Params:
        speed_threshold: Speed above which a frame is flagged as bad.
            When ``fps`` is set, interpreted as units/sec (e.g. 40 cm/s);
            otherwise units/frame. Default: None (no bad-frame detection).
        fps: Frames per second. When provided, ``speed_threshold`` is
            converted from units/sec to units/frame internally.
            Default: None.
        interpolate_centroid: If True, replace bad-frame centroid positions
            with linear interpolation. Default: True.
        interpolate_pose: If True, replace bad-frame pose keypoint positions
            with linear interpolation. Default: False.
        expand_frames: Number of frames to expand the bad-frame region in
            each direction. Default: 2.
        savgol_window: Window length for Savitzky-Golay smoothing. Must be
            odd and >= savgol_polyorder + 1. None disables smoothing.
            Default: None.
        savgol_polyorder: Polynomial order for Savitzky-Golay filter.
            Default: 2.
    """

    name = "trajectory-smooth"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        speed_threshold: float | None = None
        fps: float | None = None
        interpolate_centroid: bool = True
        interpolate_pose: bool = False
        expand_frames: int = 2
        savgol_window: int | None = None
        savgol_polyorder: int = 2

    def __init__(
        self,
        inputs: TrajectorySmooth.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State (stateless feature) ---

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

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        df = df.copy()
        df = df.sort_values(order_col).reset_index(drop=True)
        id_col = COLUMNS.id_col

        if id_col not in df.columns:
            raise ValueError(f"Missing id column '{id_col}'.")

        pose_pairs = _pose_column_pairs(df.columns)

        out_parts = []
        for _, sub in df.groupby(id_col, sort=False):
            out_parts.append(
                self._process_one_id(sub.copy(), p, order_col, pose_pairs)
            )

        if not out_parts:
            return pd.DataFrame()
        return pd.concat(out_parts, axis=0, ignore_index=True)

    # --- Internal helpers ---

    def _process_one_id(
        self,
        sub: pd.DataFrame,
        p: TrajectorySmooth.Params,
        order_col: str,
        pose_pairs: list[tuple[str, str]],
    ) -> pd.DataFrame:
        sub = sub.sort_values(order_col).reset_index(drop=True)
        x_col, y_col = COLUMNS.x_col, COLUMNS.y_col
        n = len(sub)

        # --- Step 1: Bad-frame detection ---
        if p.speed_threshold is not None and x_col in sub.columns and y_col in sub.columns:
            # Convert threshold to units/frame when fps is given
            threshold_per_frame = (
                p.speed_threshold / p.fps if p.fps is not None
                else p.speed_threshold
            )
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)
            dx = np.diff(x, prepend=np.nan)
            dy = np.diff(y, prepend=np.nan)
            speed = np.hypot(dx, dy)
            bad_mask = speed > threshold_per_frame
            # Expand bad region
            if p.expand_frames > 0 and np.any(bad_mask):
                bad_indices = np.where(bad_mask)[0]
                expanded = set()
                for idx in bad_indices:
                    for offset in range(-p.expand_frames, p.expand_frames + 1):
                        ei = idx + offset
                        if 0 <= ei < n:
                            expanded.add(ei)
                bad_mask = np.zeros(n, dtype=bool)
                bad_mask[list(expanded)] = True
            sub["bad_frame"] = bad_mask
        else:
            sub["bad_frame"] = False

        bad = sub["bad_frame"].to_numpy(dtype=bool)

        # --- Step 2: Interpolation (centroid) ---
        if p.interpolate_centroid and x_col in sub.columns and y_col in sub.columns:
            sub.loc[bad, x_col] = np.nan
            sub.loc[bad, y_col] = np.nan
            sub[x_col] = sub[x_col].interpolate(method="linear")
            sub[y_col] = sub[y_col].interpolate(method="linear")
            # Fill edges
            sub[x_col] = sub[x_col].ffill().bfill()
            sub[y_col] = sub[y_col].ffill().bfill()

        # --- Step 3: Interpolation (pose) ---
        if p.interpolate_pose and pose_pairs:
            for px_col, py_col in pose_pairs:
                sub.loc[bad, px_col] = np.nan
                sub.loc[bad, py_col] = np.nan
                sub[px_col] = sub[px_col].interpolate(method="linear")
                sub[py_col] = sub[py_col].interpolate(method="linear")
                sub[px_col] = sub[px_col].ffill().bfill()
                sub[py_col] = sub[py_col].ffill().bfill()

        # --- Step 4: Savgol smoothing ---
        if p.savgol_window is not None and n >= p.savgol_window:
            window = p.savgol_window
            polyorder = p.savgol_polyorder

            # Centroid
            if x_col in sub.columns and y_col in sub.columns:
                sub[x_col] = _savgol_with_nan(
                    sub[x_col].to_numpy(dtype=float), window, polyorder
                )
                sub[y_col] = _savgol_with_nan(
                    sub[y_col].to_numpy(dtype=float), window, polyorder
                )

            # Pose (always, regardless of interpolate_pose)
            for px_col, py_col in pose_pairs:
                sub[px_col] = _savgol_with_nan(
                    sub[px_col].to_numpy(dtype=float), window, polyorder
                )
                sub[py_col] = _savgol_with_nan(
                    sub[py_col].to_numpy(dtype=float), window, polyorder
                )

        return sub
