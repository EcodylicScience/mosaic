from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.signal import savgol_filter

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

from .helpers import ensure_columns
from .registry import register_feature


def _diff_with_step(arr: np.ndarray, step: int) -> np.ndarray:
    """Forward difference with a step; pads leading values with NaN."""
    out = np.full_like(arr, np.nan, dtype=float)
    if step < 1 or arr.size <= step:
        return out
    out[step:] = arr[step:] - arr[:-step]
    return out


def _angular_diff(arr: np.ndarray, step: int) -> np.ndarray:
    """Angle difference wrapped to [-pi, pi]."""
    raw = _diff_with_step(arr, step)
    raw = (raw + np.pi) % (2 * np.pi) - np.pi
    return raw


def _dt(
    step: int,
    time_arr: np.ndarray | None,
    n: int,
    frame_arr: np.ndarray | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Time delta array for a given step size.

    Priority:
    1. ``frame_arr`` / ``fps`` — most robust for constant-fps data, immune to
       irregular real timestamps.
    2. ``time_arr`` — real timestamps, used only when frame is unavailable.
    3. Constant ``float(step)`` — last resort when neither is available.
    """
    if frame_arr is not None and fps is not None:
        dt = _diff_with_step(frame_arr, step) / fps
    elif time_arr is not None:
        dt = _diff_with_step(time_arr, step)
    else:
        dt = np.full(n, float(step))
        dt[:step] = np.nan
    dt[dt == 0] = np.nan
    return dt


def _compute_speed(
    x: np.ndarray,
    y: np.ndarray,
    step: int,
    time_arr: np.ndarray | None,
    frame_arr: np.ndarray | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Translational speed: displacement / dt."""
    dx = _diff_with_step(x, step)
    dy = _diff_with_step(y, step)
    dist = np.sqrt(dx**2 + dy**2)
    return dist / _dt(step, time_arr, len(x), frame_arr=frame_arr, fps=fps)


def _compute_angvel(
    angle: np.ndarray,
    step: int,
    time_arr: np.ndarray | None,
    frame_arr: np.ndarray | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Angular velocity: wrapped angle difference / dt."""
    dtheta = _angular_diff(angle, step)
    return dtheta / _dt(step, time_arr, len(angle), frame_arr=frame_arr, fps=fps)


@final
@register_feature
class SpeedAngvel:
    """
    Per-sequence feature computing translational speed and angular velocity.

    Outputs (per frame):
      - speed: displacement magnitude between consecutive frames divided by dt
      - angvel: wrapped heading difference (rad) divided by dt
      - speed_step / angvel_step: same, but using a configurable step_size
        (omitted if step_size is None)
      - speed_smooth: Savitzky-Golay smoothed speed (polyorder=1), only present
        when smooth_window is set in Params

    Time-delta (dt) computation:
      Speed and angular velocity require dividing by a time interval.  The
      source for dt is chosen by priority:

      1. **frame + fps** (recommended for constant-fps video): when ``fps``
         is set in Params, dt is computed as ``frame_diff / fps``.  This is
         immune to irregular real timestamps that some trackers embed in the
         ``time`` column (e.g. TRex uses wall-clock timestamps that may
         jitter by several milliseconds per frame).  It also correctly
         handles frame gaps from dropped/bad frames.
      2. **time column**: if ``fps`` is not set but a ``time`` column exists,
         dt is computed from consecutive time differences.
      3. **array index**: last resort when neither frame+fps nor time is
         available — assumes each row is one step apart.

      For most video-based tracking data, setting ``fps`` is strongly
      recommended to avoid speed artifacts from timestamp jitter.
    """

    name = "speed-angvel"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        step_size: int | None = Field(default=None, ge=1)
        smooth_window: int | None = None
        fps: float | None = Field(
            default=None,
            description="Frames per second. When set, dt is derived from the "
            "frame column (frame_diff / fps) instead of the time column. "
            "This is more robust for constant-fps data where the time column "
            "may contain irregular real timestamps.",
        )

    def __init__(
        self,
        inputs: SpeedAngvel.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State ---

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
        if df.empty:
            return pd.DataFrame()

        order_col = resolve_order_col(df)
        ensure_columns(df, [C.id_col, C.x_col, C.y_col])
        df = df.sort_values([C.id_col, order_col]).reset_index(drop=True)

        out_parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(C.id_col, sort=False):
            out_parts.append(self._compute_one_id(sub, order_col))

        if not out_parts:
            return pd.DataFrame()
        return pd.concat(out_parts, axis=0, ignore_index=True)

    @staticmethod
    def _smooth_speed(speed: np.ndarray, window: int) -> np.ndarray:
        """Apply Savitzky-Golay filter (polyorder=1) to speed, handling NaN/inf."""
        bad = ~np.isfinite(speed)
        filled = speed.copy()
        filled[bad] = 0.0
        smoothed = savgol_filter(filled, window_length=window, polyorder=1)
        smoothed[bad] = np.nan
        return smoothed

    def _compute_one_id(self, sub: pd.DataFrame, order_col: str) -> pd.DataFrame:
        # Already sorted by (id, order_col) in apply()
        x = sub[C.x_col].to_numpy(dtype=float)
        y = sub[C.y_col].to_numpy(dtype=float)
        angle = (
            sub[C.orientation_col].to_numpy(dtype=float)
            if C.orientation_col in sub.columns
            else None
        )
        time_arr = (
            sub[C.time_col].to_numpy(dtype=float) if C.time_col in sub.columns else None
        )
        frame_arr = (
            sub[C.frame_col].to_numpy(dtype=float)
            if C.frame_col in sub.columns
            else None
        )
        fps = self.params.fps

        # Common kwargs for _compute_speed / _compute_angvel
        dt_kw = dict(time_arr=time_arr, frame_arr=frame_arr, fps=fps)

        out = pd.DataFrame(
            {"speed": _compute_speed(x, y, step=1, **dt_kw)},
            index=sub.index,
        )
        if angle is not None:
            out["angvel"] = _compute_angvel(angle, step=1, **dt_kw)

        if self.params.smooth_window is not None:
            out["speed_smooth"] = self._smooth_speed(
                out["speed"].to_numpy(), self.params.smooth_window
            )

        step_size = self.params.step_size
        if step_size is not None:
            out["speed_step"] = _compute_speed(x, y, step=step_size, **dt_kw)
            if angle is not None:
                out["angvel_step"] = _compute_angvel(
                    angle, step=step_size, **dt_kw
                )

        meta = C.meta_set() & set(sub.columns)
        return out.join(sub[sorted(meta)])
