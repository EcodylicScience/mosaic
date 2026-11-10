from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter

from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.individuals import iter_individuals
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import (
    EmitsLevel,
    COLUMNS,
    DependencyLookup,
    InputStream,
    TrackInputs,
    resolve_order_col,
)
from mosaic.core.params import Declared, Params
from mosaic.core.pose_columns import pose_column_pairs

from .registry import register_feature


@dataclass(frozen=True)
class _FillMarks:
    """Per-row marks of the centroid positions that interpolation writes."""

    interpolated: npt.NDArray[np.bool_]
    interp_run_frames: npt.NDArray[np.int64]
    unbridgeable: npt.NDArray[np.bool_]


def _fill_runs(fill: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """The maximal runs of True in *fill*, each as ``(first, last)`` inclusive."""
    edges = np.diff(np.concatenate(([0], fill.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _fill_marks(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    bad: npt.NDArray[np.bool_],
    threshold_per_frame: float | None,
) -> _FillMarks:
    """Mark the rows that interpolation fills, and the runs it cannot bridge.

    A row is filled when its position is missing or *bad*. A run of filled rows
    with an observed row on both sides is unbridgeable when the straight line
    between those two rows moves faster than *threshold_per_frame*: their
    distance over the run's length plus one frame. A run at a sequence edge is
    filled from one side and is never unbridgeable. Rows count as consecutive
    frames.
    """
    n = len(x)
    fill = ~np.isfinite(x) | ~np.isfinite(y) | bad
    run_frames = np.zeros(n, dtype=np.int64)
    unbridgeable = np.zeros(n, dtype=bool)
    for first, last in _fill_runs(fill):
        run_frames[first : last + 1] = last - first + 1
        if threshold_per_frame is None or first == 0 or last == n - 1:
            continue
        before, after = first - 1, last + 1
        distance = float(np.hypot(x[after] - x[before], y[after] - y[before]))
        if distance / (after - before) > threshold_per_frame:
            unbridgeable[first : last + 1] = True
    return _FillMarks(
        interpolated=fill, interp_run_frames=run_frames, unbridgeable=unbridgeable
    )


def _savgol_with_nan(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay filter, handling NaN/inf values."""
    bad = ~np.isfinite(arr)
    filled = arr.copy()
    filled[bad] = 0.0
    smoothed = savgol_filter(filled, window_length=window, polyorder=polyorder)
    smoothed[bad] = np.nan
    return smoothed


_SPEED_THRESHOLD_DESCRIPTION = (
    "Unset, no frame is flagged bad. Set, a frame is flagged when its "
    "displacement from the previous frame exceeds this value, per frame "
    "or, when fps is set, per second. Uses the same unit as the X and Y "
    "position columns."
)

_FPS_DESCRIPTION = (
    "Unset, speed_threshold is read as a per-frame value. Set, converts "
    "speed_threshold from a per-second rate to per-frame."
)

_INTERPOLATE_CENTROID_DESCRIPTION = (
    "Replace a bad frame's centroid position with linear interpolation "
    "between the nearest good frames, extending from the nearest good "
    "value at a sequence edge. Every remaining gap in each column is "
    "filled the same way."
)

_INTERPOLATE_POSE_DESCRIPTION = (
    "Replace a bad frame's pose keypoint positions with linear "
    "interpolation between the nearest good frames, extending from the "
    "nearest good value at a sequence edge. Every remaining gap in each "
    "column is filled the same way."
)

_EXPAND_FRAMES_DESCRIPTION = (
    "How far on each side of a speed-flagged row the bad flag extends."
)

_SAVGOL_WINDOW_DESCRIPTION = (
    "Unset, no Savitzky-Golay smoothing is applied. Set, smooths centroid "
    "and pose positions regardless of the interpolation settings, for an "
    "id with at least this many rows. Must exceed savgol_polyorder. An "
    "odd value keeps the filter window centered."
)

_SAVGOL_POLYORDER_DESCRIPTION = "The polynomial order of the Savitzky-Golay filter."

_LONG_RUN_FRAMES_DESCRIPTION = (
    "A filled centroid run longer than this is marked long_interp."
)

_MARK_COLUMNS: tuple[str, ...] = (
    "interpolated",
    "interp_run_frames",
    "unbridgeable",
    "long_interp",
)
"""The interpolation marks of every output row."""


@final
@register_feature
class TrajectorySmooth:
    """
    Per-sequence feature that smooths and interpolates trajectory positions.

    Pipeline (per individual):
      1. Bad-frame detection: flag frames with speed > speed_threshold,
         expand flagged region by expand_frames in each direction.
      2. Bridge check: a run of rows to fill (a missing centroid or a bad
         frame) whose straight-line bridge between the observed rows on
         either side exceeds speed_threshold is unbridgeable. The speed gate
         alone never compares the rows on either side of a missing run, and
         fills a persistent jump between two individuals with a line.
      3. Interpolation: set positions to NaN at bad frames, linearly
         interpolate, forward/backward fill edges. Controlled separately
         for centroid (interpolate_centroid) and pose (interpolate_pose).
         Unbridgeable runs are filled too, so features that need
         continuous positions still run; a reduction to per-individual
         metrics excludes them through its ``exclude_cols``.
      4. Savgol smoothing: apply savgol_filter to centroid X/Y and all
         pose columns (always, regardless of interpolation flags).

    Output is the full track DataFrame with smoothed positions replacing
    originals, plus these columns:

      - ``bad_frame``: the frame failed the speed gate, with expansion.
      - ``interpolated``: the centroid position was written by
        interpolation or an edge fill.
      - ``interp_run_frames``: the length of the filled run the row lies
        in, 0 for an observed row.
      - ``unbridgeable``: the row lies in an unbridgeable run. A run at a
        sequence edge is filled from one side and is never unbridgeable.
      - ``long_interp``: ``interpolated`` in a run longer than
        ``long_run_frames``.

    Without ``interpolate_centroid`` the centroid is not filled, the marks
    are False and 0, and positions stay NaN. Without ``speed_threshold`` no
    run is unbridgeable.

    Downstream features consume the ``Result`` the run returned, or its
    storage name:
    ``Inputs((Result(feature="trajectory-smooth__from__tracks"),))``.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.trajectory_smooth.TrajectorySmooth.Params`.
    """

    category = "per-frame"
    name = "trajectory-smooth"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    requires: ClassVar[tuple[Requirement, ...]] = (Column(COLUMNS.id_col), ORDER_COLUMN)
    provides: ClassVar[Provides] = Provides(
        passthrough=True,
        adds=(Column("bad_frame"), *(Column(name) for name in _MARK_COLUMNS)),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        speed_threshold: Annotated[
            float | None, Declared(_SPEED_THRESHOLD_DESCRIPTION)
        ] = None
        fps: Annotated[float | None, Declared(_FPS_DESCRIPTION, unit="fps")] = None
        interpolate_centroid: Annotated[
            bool, Declared(_INTERPOLATE_CENTROID_DESCRIPTION)
        ] = True
        interpolate_pose: Annotated[bool, Declared(_INTERPOLATE_POSE_DESCRIPTION)] = (
            False
        )
        expand_frames: Annotated[
            int, Declared(_EXPAND_FRAMES_DESCRIPTION, unit="frames")
        ] = 2
        savgol_window: Annotated[
            int | None, Declared(_SAVGOL_WINDOW_DESCRIPTION, unit="frames")
        ] = None
        savgol_polyorder: Annotated[int, Declared(_SAVGOL_POLYORDER_DESCRIPTION)] = 2
        long_run_frames: Annotated[
            int, Declared(_LONG_RUN_FRAMES_DESCRIPTION, unit="frames")
        ] = 30

    def __init__(
        self,
        inputs: TrajectorySmooth.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State (stateless feature) ---

    @staticmethod
    def temporal_reach(params: TrajectorySmooth.Params) -> int | None:
        """Frames on each side that one row reads, ``None`` under interpolation.

        A bad or missing frame is filled from the nearest good frames, however far
        away they are, so no param bounds an interpolating run. Without
        interpolation a row reads the Savitzky-Golay window (``savgol_window // 2``
        frames, centered) and, for ``bad_frame``, a displacement flag expanded by
        ``expand_frames`` (the speed of a frame reads one further back). Near a
        sequence's ends the filter fits its first or last window, which reaches
        further and is not counted. Rows are taken to be consecutive frames.
        """
        if params.interpolate_centroid or params.interpolate_pose:
            return None
        reach = 0
        if params.speed_threshold is not None:
            reach = params.expand_frames + 1
        if params.savgol_window is not None:
            reach = max(reach, params.savgol_window // 2)
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
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        # Sorting first orders the individuals by their first row in time.
        df = df.sort_values(order_col).reset_index(drop=True)
        pose_pairs = pose_column_pairs(df.columns)

        out_parts = [
            self._process_one_id(rows, p, order_col, pose_pairs)
            for _, rows in iter_individuals(df, order_col)
        ]

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
        """Clean and smooth one individual's rows, sorted by *order_col* and indexed from 0."""
        x_col, y_col = COLUMNS.x_col, COLUMNS.y_col
        n = len(sub)

        # Standardize inf to NaN so interpolation and smoothing handle gaps correctly
        for col in sub.select_dtypes(include=[np.floating]).columns:
            sub[col] = sub[col].replace([np.inf, -np.inf], np.nan)

        has_centroid = x_col in sub.columns and y_col in sub.columns
        # The threshold in position units per frame
        threshold_per_frame: float | None = None
        if p.speed_threshold is not None:
            threshold_per_frame = (
                p.speed_threshold / p.fps if p.fps is not None else p.speed_threshold
            )

        # --- Step 1: Bad-frame detection ---
        if threshold_per_frame is not None and has_centroid:
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

        # --- Step 2: Bridge check and interpolation marks ---
        if p.interpolate_centroid and has_centroid:
            marks = _fill_marks(
                sub[x_col].to_numpy(dtype=float),
                sub[y_col].to_numpy(dtype=float),
                bad,
                threshold_per_frame,
            )
            sub["interpolated"] = marks.interpolated
            sub["interp_run_frames"] = marks.interp_run_frames
            sub["unbridgeable"] = marks.unbridgeable
            sub["long_interp"] = marks.interpolated & (
                marks.interp_run_frames > p.long_run_frames
            )
        else:
            sub["interpolated"] = False
            sub["interp_run_frames"] = np.zeros(n, dtype=np.int64)
            sub["unbridgeable"] = False
            sub["long_interp"] = False

        # --- Step 3: Interpolation (centroid) ---
        if p.interpolate_centroid and has_centroid:
            sub.loc[bad, x_col] = np.nan
            sub.loc[bad, y_col] = np.nan
            sub[x_col] = sub[x_col].interpolate(method="linear")
            sub[y_col] = sub[y_col].interpolate(method="linear")
            # Fill edges
            sub[x_col] = sub[x_col].ffill().bfill()
            sub[y_col] = sub[y_col].ffill().bfill()

        # --- Step 4: Interpolation (pose) ---
        if p.interpolate_pose and pose_pairs:
            for px_col, py_col in pose_pairs:
                sub.loc[bad, px_col] = np.nan
                sub.loc[bad, py_col] = np.nan
                sub[px_col] = sub[px_col].interpolate(method="linear")
                sub[py_col] = sub[py_col].interpolate(method="linear")
                sub[px_col] = sub[px_col].ffill().bfill()
                sub[py_col] = sub[py_col].ffill().bfill()

        # --- Step 5: Savgol smoothing ---
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
