from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
import numpy as np
import pandas as pd

from behavior.dataset import register_feature


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


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


@register_feature
class SpeedAngvel:
    """
    Per-sequence feature computing translational speed and angular velocity.

    Outputs (per frame):
      - speed: displacement magnitude between consecutive frames divided by dt
      - angvel: wrapped heading difference (rad) divided by dt
      - speed_step / angvel_step: same, but using a configurable step_size
        (omitted if step_size is None)
    """

    name = "speed-angvel"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        id_col="id",
        seq_col="sequence",
        group_col="group",
        time_col="time",
        order_pref=("frame", "time"),
        x_col="x",
        y_col="y",
        angle_col="ANGLE",
        step_size=None,  # if None, skip _step outputs
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
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
        order_col = self._order_col(df)
        df = df.sort_values(order_col).reset_index(drop=True)
        id_col = p["id_col"]

        if id_col not in df.columns:
            raise ValueError(f"Missing id column '{id_col}' for per-id speed/angvel.")

        out_parts = []
        for _, sub in df.groupby(id_col, sort=False):
            out_parts.append(self._compute_one_id(sub, p, order_col))

        if not out_parts:
            return pd.DataFrame()
        return pd.concat(out_parts, axis=0, ignore_index=True)

    # ------------------ Internal helpers ------------------------
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _dt(self, step: int, time_arr: Optional[np.ndarray], n: int) -> np.ndarray:
        if step < 1:
            raise ValueError("step must be >= 1")
        if time_arr is not None:
            dt = _diff_with_step(time_arr, step)
        else:
            dt = np.full(n, float(step))
            dt[:step] = np.nan
        dt[dt == 0] = np.nan
        return dt

    def _compute_speed(self, x: np.ndarray, y: np.ndarray, step: int, time_arr: Optional[np.ndarray]) -> np.ndarray:
        dx = _diff_with_step(x, step)
        dy = _diff_with_step(y, step)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dt = self._dt(step, time_arr, len(x))
        return dist / dt

    def _compute_angvel(self, angle: np.ndarray, step: int, time_arr: Optional[np.ndarray]) -> np.ndarray:
        dtheta = _angular_diff(angle, step)
        dt = self._dt(step, time_arr, len(angle))
        return dtheta / dt

    def _compute_one_id(self, sub: pd.DataFrame, p: dict, order_col: str) -> pd.DataFrame:
        sub = sub.sort_values(order_col).reset_index(drop=True)
        x = sub[p["x_col"]].to_numpy(dtype=float)
        y = sub[p["y_col"]].to_numpy(dtype=float)
        angle = sub[p["angle_col"]].to_numpy(dtype=float) if p["angle_col"] in sub.columns else None
        time_arr = sub[p["time_col"]].to_numpy(dtype=float) if p["time_col"] in sub.columns else None

        out = pd.DataFrame({
            "speed": self._compute_speed(x, y, step=1, time_arr=time_arr),
        })
        if angle is not None:
            out["angvel"] = self._compute_angvel(angle, step=1, time_arr=time_arr)

        step_size = p.get("step_size")
        if step_size:
            step_size = int(step_size)
            out["speed_step"] = self._compute_speed(x, y, step=step_size, time_arr=time_arr)
            if angle is not None:
                out["angvel_step"] = self._compute_angvel(angle, step=step_size, time_arr=time_arr)

        # Attach meta columns from this sub-id
        for c in ("frame", "time", p["seq_col"], p["group_col"], p["id_col"]):
            if c in sub.columns:
                out[c] = sub[c].values
        return out
