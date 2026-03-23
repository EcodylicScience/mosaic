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
    Inputs,
    InputStream,
    Params,
    TrackInput,
    resolve_order_col,
)

from .helpers import ego_rotate, ensure_columns, wrap_angle
from .registry import register_feature


@final
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
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: NearestNeighbor.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State protocol ---

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
        df = df.sort_values(order_col).reset_index(drop=True)

        angles = (
            df[C.orientation_col].to_numpy(dtype=float)
            if C.orientation_col in df.columns
            else None
        )

        n = len(df)
        nn_id = np.full(n, np.nan, dtype=float)
        nn_dx = np.full(n, np.nan, dtype=float)
        nn_dy = np.full(n, np.nan, dtype=float)
        nn_dist = np.full(n, np.nan, dtype=float)
        nn_dangle = np.full(n, np.nan, dtype=float) if angles is not None else None
        nn_dx_ego = np.full(n, np.nan, dtype=float)
        nn_dy_ego = np.full(n, np.nan, dtype=float)

        for _, g in df.groupby(order_col, sort=False):
            idx = g.index.to_numpy()
            if len(idx) < 2:
                continue
            gx = g[C.x_col].to_numpy(dtype=float)
            gy = g[C.y_col].to_numpy(dtype=float)
            gids = g[C.id_col].to_numpy()
            gang = (
                g[C.orientation_col].to_numpy(dtype=float)
                if angles is not None
                else None
            )

            dx_matrix = gx[np.newaxis, :] - gx[:, np.newaxis]
            dy_matrix = gy[np.newaxis, :] - gy[:, np.newaxis]
            dist_matrix = np.sqrt(dx_matrix**2 + dy_matrix**2)
            np.fill_diagonal(dist_matrix, np.inf)

            nn_idx = np.argmin(dist_matrix, axis=1)
            nn_id[idx] = gids[nn_idx]
            nn_dx[idx] = gx[nn_idx] - gx
            nn_dy[idx] = gy[nn_idx] - gy
            nn_dist[idx] = dist_matrix[np.arange(len(idx)), nn_idx]

            if nn_dangle is not None:
                nn_dangle[idx] = wrap_angle(gang[nn_idx] - gang)

            dx_ego, dy_ego = ego_rotate(
                nn_dx[idx],
                nn_dy[idx],
                gang if angles is not None else np.zeros(len(idx)),
            )
            nn_dx_ego[idx] = dx_ego
            nn_dy_ego[idx] = dy_ego

        out = pd.DataFrame(
            {
                "nn_id": nn_id,
                "nn_delta_x": nn_dx,
                "nn_delta_y": nn_dy,
                "nn_dist": nn_dist,
                "nn_delta_x_ego": nn_dx_ego,
                "nn_delta_y_ego": nn_dy_ego,
            },
            index=df.index,
        )
        if nn_dangle is not None:
            out["nn_delta_angle"] = nn_dangle

        meta = C.meta_set() & set(df.columns)
        return out.join(df[sorted(meta)])
