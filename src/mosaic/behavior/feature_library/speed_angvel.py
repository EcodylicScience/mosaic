from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, final

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.signal import savgol_filter

from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.individuals import iter_individuals
from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    Column,
    Conditional,
    Provides,
    Requirement,
)
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
from mosaic.core.timebase import step_seconds

from .helpers import ensure_columns
from .registry import register_feature

_STEP_SIZE_DESCRIPTION = (
    "The additional frame step for speed_step and angvel_step, computed "
    "beside the one-frame difference. Unset, neither column is emitted."
)

_SMOOTH_WINDOW_DESCRIPTION = (
    "The window Savitzky-Golay smoothing (polyorder 1) averages speed "
    "over to produce speed_smooth. Unset, speed_smooth is not emitted. "
    "Smoothing raises when the window is 0 or 1. An individual with fewer "
    "rows than the window is smoothed with the largest odd window that "
    "fits, and one with fewer than 3 rows keeps its unsmoothed speed."
)

_MIN_SMOOTH_WINDOW = 3
"""The shortest window that a first-order Savitzky-Golay filter smooths over."""


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


def _compute_speed(
    x: np.ndarray,
    y: np.ndarray,
    step: int,
    time_arr: np.ndarray,
) -> np.ndarray:
    """Translational speed: displacement over the seconds the rows span."""
    dx = _diff_with_step(x, step)
    dy = _diff_with_step(y, step)
    dist = np.sqrt(dx**2 + dy**2)
    return dist / step_seconds(time_arr, lag=step)


def _compute_angvel(
    angle: np.ndarray,
    step: int,
    time_arr: np.ndarray,
) -> np.ndarray:
    """Angular velocity: wrapped angle difference over the seconds the rows span."""
    dtheta = _angular_diff(angle, step)
    return dtheta / step_seconds(time_arr, lag=step)


@final
@register_feature
class SpeedAngvel:
    """
    Per-sequence feature computing translational speed and angular velocity.

    Outputs (per frame):
      - speed: displacement magnitude between consecutive frames divided by dt
      - vx / vy: the signed components of that same displacement over dt
      - angvel: wrapped heading difference (rad) divided by dt
      - speed_step / angvel_step: same, but using a configurable step_size
        (omitted if step_size is None)
      - speed_smooth: Savitzky-Golay smoothed speed (polyorder=1), only present
        when smooth_window is set in Params

    Time-delta (dt) computation:
      Speed and angular velocity divide by the seconds between the rows, read
      from the ``time`` column. A step of several rows divides by the seconds
      those rows span. The feature takes no frame rate: a frame count says how
      long a step lasted only on a table that dropped no frame, and the
      ``time`` column says it on every table. A row whose step is not positive
      has NaN speed and angular velocity.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.speed_angvel.SpeedAngvel.Params`.
    """

    category = "per-frame"
    name = "speed-angvel"
    version = "0.3"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.x_col),
        Column(C.y_col),
        Column(C.time_col),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(Column("speed"), Column("vx"), Column("vy")),
        conditional=(
            Conditional(Column("angvel"), when_present=Column(C.orientation_col)),
            Conditional(Column("speed_smooth"), when_param="smooth_window"),
            Conditional(Column("speed_step"), when_param="step_size"),
            Conditional(
                Column("angvel_step"),
                when_param="step_size",
                when_present=Column(C.orientation_col),
            ),
            *META_KEPT,
        ),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        step_size: Annotated[
            int | None, Declared(_STEP_SIZE_DESCRIPTION, unit="frames")
        ] = Field(default=None, ge=1)
        smooth_window: Annotated[
            int | None, Declared(_SMOOTH_WINDOW_DESCRIPTION, unit="frames")
        ] = None

    def __init__(
        self,
        inputs: SpeedAngvel.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State ---

    @staticmethod
    def temporal_reach(params: SpeedAngvel.Params) -> int:
        """Frames on each side that one row reads.

        A difference reaches back ``step_size`` frames (one without it). The
        Savitzky-Golay window of ``speed_smooth`` is centered, so it reads
        ``smooth_window // 2`` frames of speed on each side, and each speed reads
        one frame further back. Near a sequence's ends the filter fits its first
        or last window instead, which reaches further and is not counted. Rows
        are taken to be consecutive frames.
        """
        reach = params.step_size or 1
        if params.smooth_window is not None:
            reach = max(reach, params.smooth_window // 2 + 1)
        return reach

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

        ensure_columns(df, [C.time_col])
        order_col = resolve_order_col(df)
        # Sorting by id first yields the individuals in id order.
        out_parts = [
            self._compute_one_id(rows, order_col)
            for _, rows in iter_individuals(
                df.sort_values(C.id_col, kind="stable"), order_col
            )
        ]

        if not out_parts:
            return pd.DataFrame()
        return pd.concat(out_parts, axis=0, ignore_index=True)

    @staticmethod
    def _smooth_speed(speed: np.ndarray, window: int) -> np.ndarray:
        """Apply Savitzky-Golay filter (polyorder=1) to speed, handling NaN/inf."""
        rows = len(speed)
        if window > rows:
            window = rows - (rows + 1) % 2
            if window < _MIN_SMOOTH_WINDOW:
                return speed.copy()
        bad = ~np.isfinite(speed)
        filled = speed.copy()
        filled[bad] = 0.0
        smoothed = savgol_filter(filled, window_length=window, polyorder=1)
        smoothed[bad] = np.nan
        return smoothed

    def _compute_one_id(self, sub: pd.DataFrame, order_col: str) -> pd.DataFrame:
        """Measure one individual's rows, sorted by *order_col*."""
        x = sub[C.x_col].to_numpy(dtype=float)
        y = sub[C.y_col].to_numpy(dtype=float)
        angle = (
            sub[C.orientation_col].to_numpy(dtype=float)
            if C.orientation_col in sub.columns
            else None
        )
        time_arr = sub[C.time_col].to_numpy(dtype=float)

        # Velocity components alongside the magnitude. They are the same
        # differencing the speed already does, and the converters used to write
        # them into the track table as VX/VY -- deriving them there made them
        # look like tracker output. Emitting them here costs one division each
        # and means nothing has to difference positions a second time.
        dt = step_seconds(time_arr)
        out = pd.DataFrame(
            {
                "speed": _compute_speed(x, y, 1, time_arr),
                "vx": _diff_with_step(x, 1) / dt,
                "vy": _diff_with_step(y, 1) / dt,
            },
            index=sub.index,
        )
        if angle is not None:
            out["angvel"] = _compute_angvel(angle, 1, time_arr)

        if self.params.smooth_window is not None:
            out["speed_smooth"] = self._smooth_speed(
                out["speed"].to_numpy(), self.params.smooth_window
            )

        step_size = self.params.step_size
        if step_size is not None:
            out["speed_step"] = _compute_speed(x, y, step_size, time_arr)
            if angle is not None:
                out["angvel_step"] = _compute_angvel(angle, step_size, time_arr)

        meta = C.meta_set() & set(sub.columns)
        return out.join(sub[sorted(meta)])
