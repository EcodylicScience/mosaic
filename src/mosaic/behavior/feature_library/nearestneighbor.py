from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _ego_rotate(dx: np.ndarray, dy: np.ndarray, heading: np.ndarray) -> tuple:
    """
    Rotate world-frame deltas into the ego frame of the focal
    (heading aligned with +x). heading is in radians.
    """
    ct = np.cos(heading)
    st = np.sin(heading)
    dx_ego = dx * ct + dy * st
    dy_ego = -dx * st + dy * ct
    return dx_ego, dy_ego


@register_feature
class NearestNeighbor:
    """
    Per-sequence feature computing nearest-neighbor identity and relative kinematics.

    Outputs per frame (one row per individual):
      - nn_id: id of nearest neighbor (NaN if none)
      - nn_delta_x / nn_delta_y: neighbor position minus focal, world frame
      - nn_dist: Euclidean distance to nearest neighbor
      - nn_delta_angle: neighbor heading minus focal, wrapped to [-pi, pi]
      - nn_delta_x_ego / nn_delta_y_ego: neighbor offset in focal ego frame
    """

    name = "nearest-neighbor"
    version = "0.1"
    parallelizable = True
    output_type = "per_frame"

    _defaults = dict(
        id_col="id",
        seq_col="sequence",
        group_col="group",
        x_col="X",
        y_col="Y",
        angle_col="ANGLE",
        order_pref=("frame", "time"),
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

        x = df[p["x_col"]].to_numpy(dtype=float)
        y = df[p["y_col"]].to_numpy(dtype=float)
        ids = df[p["id_col"]].to_numpy()
        angles = df[p["angle_col"]].to_numpy(dtype=float) if p["angle_col"] in df.columns else None

        n = len(df)
        nn_id = np.full(n, np.nan, dtype=float)
        nn_dx = np.full(n, np.nan, dtype=float)
        nn_dy = np.full(n, np.nan, dtype=float)
        nn_dist = np.full(n, np.nan, dtype=float)
        nn_dangle = np.full(n, np.nan, dtype=float) if angles is not None else None
        nn_dx_ego = np.full(n, np.nan, dtype=float)
        nn_dy_ego = np.full(n, np.nan, dtype=float)

        # Work frame by frame to avoid mixing across time
        if "frame" in df.columns:
            grouper = df.groupby("frame", sort=False)
        elif "time" in df.columns:
            grouper = df.groupby("time", sort=False)
        else:
            raise ValueError("Need either 'frame' or 'time' column to group rows per timestep.")

        for _, g in grouper:
            idx = g.index.to_numpy()
            if len(idx) < 2:
                continue
            gx = g[p["x_col"]].to_numpy(dtype=float)
            gy = g[p["y_col"]].to_numpy(dtype=float)
            gids = g[p["id_col"]].to_numpy()
            gang = g[p["angle_col"]].to_numpy(dtype=float) if angles is not None else None

            dx_matrix = gx[np.newaxis, :] - gx[:, np.newaxis]
            dy_matrix = gy[np.newaxis, :] - gy[:, np.newaxis]
            dist_matrix = np.sqrt(dx_matrix ** 2 + dy_matrix ** 2)
            np.fill_diagonal(dist_matrix, np.inf)

            nn_idx = np.argmin(dist_matrix, axis=1)
            nn_id[idx] = gids[nn_idx]
            nn_dx[idx] = gx[nn_idx] - gx
            nn_dy[idx] = gy[nn_idx] - gy
            nn_dist[idx] = dist_matrix[np.arange(len(idx)), nn_idx]

            if nn_dangle is not None:
                nn_dangle[idx] = _wrap_angle(gang[nn_idx] - gang)

            dx_ego, dy_ego = _ego_rotate(nn_dx[idx], nn_dy[idx], gang if angles is not None else np.zeros(len(idx)))
            nn_dx_ego[idx] = dx_ego
            nn_dy_ego[idx] = dy_ego

        out = pd.DataFrame({
            "nn_id": nn_id,
            "nn_delta_x": nn_dx,
            "nn_delta_y": nn_dy,
            "nn_dist": nn_dist,
            "nn_delta_x_ego": nn_dx_ego,
            "nn_delta_y_ego": nn_dy_ego,
        })
        if nn_dangle is not None:
            out["nn_delta_angle"] = nn_dangle

        # Attach meta columns
        for c in ("frame", "time", p["seq_col"], p["group_col"], p["id_col"]):
            if c in df.columns:
                out[c] = df[c].values

        return out

    # ------------------ Internal helpers ------------------------
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")
