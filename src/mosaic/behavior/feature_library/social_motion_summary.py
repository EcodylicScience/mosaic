from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    Params,
    TrackInputs,
    resolve_order_col,
)

from .helpers import ensure_columns
from .registry import register_feature


def _safe_ratio(num: float, den: float) -> float:
    """``num / den`` guarded against a zero / non-finite denominator."""
    if not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num / den)


def _iqr(vals: np.ndarray) -> float:
    """Interquartile range (75th - 25th percentile), NaN-safe."""
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return float("nan")
    q75, q25 = np.nanpercentile(vals, [75.0, 25.0])
    return float(q75 - q25)


def _dispersion(vals: np.ndarray) -> tuple[float, float]:
    """Return (CV = std/mean, robust-CV = IQR/median) for a value series.

    The robust form is more stable on noisy tracking data where a few large
    outliers inflate the mean/std.
    """
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return float("nan"), float("nan")
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    med = float(np.nanmedian(vals))
    cv = _safe_ratio(std, mean)
    rcv = _safe_ratio(_iqr(vals), med)
    return cv, rcv


def _magnitude_stats(prefix: str, mag: np.ndarray) -> dict[str, float]:
    """Mean/median central tendency plus both dispersion forms for ``|mag|``.

    ``mag`` is already a magnitude array (e.g. ``|accel|``). Emits
    ``{prefix}_mean``, ``{prefix}_med``, ``{prefix}_cv`` (std/mean) and
    ``{prefix}_rcv`` (IQR/median).
    """
    if mag.size == 0 or not np.any(np.isfinite(mag)):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_med": float("nan"),
            f"{prefix}_cv": float("nan"),
            f"{prefix}_rcv": float("nan"),
        }
    mean = float(np.nanmean(mag))
    med = float(np.nanmedian(mag))
    std = float(np.nanstd(mag))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_med": med,
        f"{prefix}_cv": _safe_ratio(std, mean),
        f"{prefix}_rcv": _safe_ratio(_iqr(mag), med),
    }


def _derivative(values: np.ndarray, frames: np.ndarray, fps: float) -> np.ndarray:
    """Discrete time-derivative ``d(values)/dt`` with ``dt = d(frame) / fps``.

    Returns an array of length ``len(values) - 1`` aligned to the intervals
    between consecutive samples. Intervals with a non-positive frame gap are
    set to NaN so gaps in the track do not create spurious spikes.
    """
    if values.size < 2:
        return np.empty(0, dtype=float)
    dv = np.diff(values)
    dframe = np.diff(frames)
    dt = dframe / fps
    dt = np.where(dt > 0, dt, np.nan)
    return dv / dt


@final
@register_feature
class SocialMotionSummary:
    """Per-fish summary of social-interaction and locomotor-style metrics.

    A ``summary`` feature (one output row per ``id`` per sequence) built to
    provide mechanism / interaction metrics that are **not** mechanically tied
    to how often an individual is isolated -- unlike isolation-event duration.
    It consumes already-computed per-frame features and reduces them per fish.

    Consumes (merged on ``frame``/``id`` by the pipeline):
      - ``nearest-neighbor``: ``nn_id``, ``nn_delta_angle``,
        ``nn_delta_x_ego``, ``nn_delta_y_ego``
      - ``speed-angvel``: ``speed``
      - ``ffgroups`` (optional): ``group_membership``, ``group_size``

    Social metrics (over frames with a valid nearest neighbor):
      - ``nn_align``:        mean cos(nn_delta_angle) -- local heading alignment
      - ``frac_nn_ahead``:   fraction of social time the neighbor is ahead
      - ``nn_bearing_x/y``:  mean unit-vector bearing of the neighbor (ego frame)
      - ``speed_match_nn``:    mean |own speed - nearest-neighbor speed|
        (needs only ``nn_id`` + ``speed`` -- independent of group definitions)
      - ``speed_match_group``: mean |own speed - group-mean speed| (needs
        ``group_membership``)

    Motion metrics (over all frames):
      - ``speed_cv`` / ``speed_rcv``: speed dispersion (std/mean and IQR/median)
      - ``accel_{mean,med,cv,rcv}``: acceleration magnitude |d speed / dt|
      - ``jerk_{mean,med,cv,rcv}``:  jerk magnitude |d accel / dt|
      - ``kick_rate`` / ``burst_coast_ratio`` (only if ``compute_burst_coast``)

    Params:
        fps: Frames per second, used to convert frame steps to seconds when
            differentiating speed. Default 30.0.
        speed_col: Column holding per-frame speed. Default "speed".
        social_min_group_size: Minimum ``group_size`` for a frame to count as
            "social" (when ``group_size`` is available). Default 2.
        compute_burst_coast: If True, also emit a simple burst-and-coast
            gait summary (``kick_rate``, ``burst_coast_ratio``). Default False.
    """

    category = "summary"
    name = "social-motion-summary"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        fps: float = 30.0
        speed_col: str = "speed"
        social_min_group_size: int = 2
        compute_burst_coast: bool = False

    def __init__(
        self,
        inputs: SocialMotionSummary.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State protocol (stateless per-sequence feature) ---

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

    def _neighbor_speed(self, df: pd.DataFrame, order_col: str) -> np.ndarray:
        """Speed of each row's nearest neighbor, via a self-join on (frame, nn_id)."""
        speed_col = self.params.speed_col
        lut = df[[order_col, C.id_col, speed_col]].copy()
        lut[C.id_col] = lut[C.id_col].astype(float)
        lut = lut.rename(columns={C.id_col: "_nn_key", speed_col: "_nbr_speed"})
        left = pd.DataFrame(
            {
                order_col: df[order_col].to_numpy(),
                "_nn_key": df["nn_id"].astype(float).to_numpy(),
            }
        )
        merged = left.merge(lut, on=[order_col, "_nn_key"], how="left")
        return merged["_nbr_speed"].to_numpy(dtype=float)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        ensure_columns(df, [C.id_col, order_col, p.speed_col])
        df = df.sort_values([C.id_col, order_col]).reset_index(drop=True)

        has_gm = "group_membership" in df.columns
        has_gs = "group_size" in df.columns
        has_nn = "nn_id" in df.columns
        has_angle = "nn_delta_angle" in df.columns
        has_ego = "nn_delta_x_ego" in df.columns and "nn_delta_y_ego" in df.columns

        neighbor_speed = (
            self._neighbor_speed(df, order_col) if has_nn else np.full(len(df), np.nan)
        )
        if has_gm:
            group_mean = df.groupby([order_col, "group_membership"])[
                p.speed_col
            ].transform("mean")
            dev_group = df[p.speed_col].to_numpy(dtype=float) - group_mean.to_numpy(
                dtype=float
            )
        else:
            dev_group = np.full(len(df), np.nan)

        rows: list[dict[str, object]] = []
        for fid, g in df.groupby(C.id_col, sort=True):
            idx = g.index.to_numpy()
            speed = g[p.speed_col].to_numpy(dtype=float)
            frames = g[order_col].to_numpy(dtype=float)

            # --- social mask: frames with a valid nearest neighbor ---
            if has_nn:
                social = np.isfinite(g["nn_id"].to_numpy(dtype=float))
            else:
                social = np.zeros(len(g), dtype=bool)
            if has_gs:
                social = social & (
                    g["group_size"].to_numpy(dtype=float) >= p.social_min_group_size
                )

            row: dict[str, object] = {C.id_col: fid}

            # --- NN heading alignment ---
            if has_angle:
                ang = g["nn_delta_angle"].to_numpy(dtype=float)[social]
                row["nn_align"] = (
                    float(np.nanmean(np.cos(ang))) if ang.size else float("nan")
                )
            else:
                row["nn_align"] = float("nan")

            # --- Neighbor bearing preference ---
            if has_ego:
                dxe = g["nn_delta_x_ego"].to_numpy(dtype=float)[social]
                dye = g["nn_delta_y_ego"].to_numpy(dtype=float)[social]
                if dxe.size:
                    row["frac_nn_ahead"] = float(np.nanmean((dxe > 0).astype(float)))
                    norm = np.sqrt(dxe**2 + dye**2)
                    norm = np.where(norm > 0, norm, np.nan)
                    row["nn_bearing_x"] = float(np.nanmean(dxe / norm))
                    row["nn_bearing_y"] = float(np.nanmean(dye / norm))
                else:
                    row["frac_nn_ahead"] = float("nan")
                    row["nn_bearing_x"] = float("nan")
                    row["nn_bearing_y"] = float("nan")
            else:
                row["frac_nn_ahead"] = float("nan")
                row["nn_bearing_x"] = float("nan")
                row["nn_bearing_y"] = float("nan")

            # --- Speed matching to nearest neighbor (group-free) ---
            if has_nn:
                own = speed[social]
                nbr = neighbor_speed[idx][social]
                diff = np.abs(own - nbr)
                row["speed_match_nn"] = (
                    float(np.nanmean(diff))
                    if diff.size and np.any(np.isfinite(diff))
                    else float("nan")
                )
            else:
                row["speed_match_nn"] = float("nan")

            # --- Speed matching to group mean (needs group membership) ---
            if has_gm:
                dg = np.abs(dev_group[idx][social])
                row["speed_match_group"] = (
                    float(np.nanmean(dg))
                    if dg.size and np.any(np.isfinite(dg))
                    else float("nan")
                )
            else:
                row["speed_match_group"] = float("nan")

            # --- Motion / locomotor style (all frames) ---
            cv, rcv = _dispersion(speed)
            row["speed_cv"] = cv
            row["speed_rcv"] = rcv

            accel = _derivative(speed, frames, p.fps)
            row.update(_magnitude_stats("accel", np.abs(accel)))

            # jerk = d(accel)/dt over the intervals between accel samples
            if accel.size >= 2:
                jerk = _derivative(accel, frames[1:], p.fps)
            else:
                jerk = np.empty(0, dtype=float)
            row.update(_magnitude_stats("jerk", np.abs(jerk)))

            if p.compute_burst_coast:
                row.update(self._burst_coast(speed, accel, frames, p.fps))

            rows.append(row)

        out = pd.DataFrame(rows)

        # Attach sequence / group metadata (constant within a sequence)
        for meta_col in (C.seq_col, C.group_col):
            if meta_col in df.columns and meta_col not in out.columns:
                out[meta_col] = df[meta_col].iloc[0]

        return out.reset_index(drop=True)

    @staticmethod
    def _burst_coast(
        speed: np.ndarray, accel: np.ndarray, frames: np.ndarray, fps: float
    ) -> dict[str, float]:
        """Minimal burst-and-coast gait summary.

        ``kick_rate`` = number of acceleration peaks (a burst onset, where
        acceleration crosses from positive to non-positive) per second.
        ``burst_coast_ratio`` = fraction of intervals with positive acceleration.
        """
        if accel.size == 0 or not np.any(np.isfinite(accel)):
            return {"kick_rate": float("nan"), "burst_coast_ratio": float("nan")}
        pos = accel > 0
        # peaks: a positive-accel sample immediately followed by non-positive
        peaks = int(np.sum(pos[:-1] & ~pos[1:])) if accel.size >= 2 else 0
        duration_s = float((frames[-1] - frames[0]) / fps) if frames.size >= 2 else 0.0
        kick_rate = _safe_ratio(float(peaks), duration_s)
        burst_coast_ratio = float(np.mean(pos))
        return {"kick_rate": kick_rate, "burst_coast_ratio": burst_coast_ratio}
