from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import numpy as np
import pandas as pd

from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    ORDER_COLUMN,
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
from mosaic.core.params import Params

from .helpers import ego_rotate, wrap_angle
from .registry import register_feature


@final
@register_feature
class NearestNeighbor:
    """
    Per-sequence feature computing nearest-neighbor identity and relative kinematics.

    Outputs per frame (one row per individual):
      - nn_id: id of nearest neighbor (NaN if none). An individual whose
        position is missing in a frame is nobody's neighbor there, and has none.
      - nn_delta_x / nn_delta_y: neighbor position minus focal, world frame
      - nn_dist: Euclidean distance to nearest neighbor
      - nn_delta_angle: neighbor heading minus focal, wrapped to [-pi, pi]
      - nn_ego_unrotated: True when no heading column was available, so the
        `_ego` offsets above are world-frame rather than egocentric
      - nn_delta_x_ego / nn_delta_y_ego: neighbor offset in focal ego frame
    """

    category = "per-frame"
    name = "nearest-neighbor"
    # 0.3: an individual missing from a frame is nobody's neighbor. Its NaN
    # distance used to win argmin, which blanked every row of that frame.
    version = "0.3"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.x_col),
        Column(C.y_col),
        ORDER_COLUMN,
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column("nn_id"),
            Column("nn_delta_x"),
            Column("nn_delta_y"),
            Column("nn_dist"),
            Column("nn_delta_x_ego"),
            Column("nn_delta_y_ego"),
            Column("nn_ego_unrotated"),
        ),
        conditional=(
            Conditional(
                Column("nn_delta_angle"), when_present=Column(C.orientation_col)
            ),
            *META_KEPT,
        ),
    )

    class Inputs(TrackInputs):
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
            # An individual without a position is never the nearest. argmin
            # returns the first NaN it meets, so a missing one would otherwise
            # be every row's neighbor, at a NaN distance.
            dist_matrix[np.isnan(dist_matrix)] = np.inf

            nn_idx = np.argmin(dist_matrix, axis=1)
            nearest = dist_matrix[np.arange(len(idx)), nn_idx]
            # A row whose own position is missing, or whose every other
            # individual is, has no neighbor and keeps NaN.
            found = np.isfinite(nearest)
            rows, nn_idx = idx[found], nn_idx[found]
            nn_id[rows] = gids[nn_idx]
            nn_dx[rows] = gx[nn_idx] - gx[found]
            nn_dy[rows] = gy[nn_idx] - gy[found]
            nn_dist[rows] = nearest[found]

            if nn_dangle is not None and gang is not None:
                nn_dangle[rows] = wrap_angle(gang[nn_idx] - gang[found])

            dx_ego, dy_ego = ego_rotate(
                nn_dx[rows],
                nn_dy[rows],
                gang[found] if gang is not None else np.zeros(len(rows)),
            )
            nn_dx_ego[rows] = dx_ego
            nn_dy_ego[rows] = dy_ego

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

        # Without a heading there is nothing to rotate into, so the ``_ego``
        # offsets above are the *world* offsets under an egocentric name -- which
        # reads as a perfectly good egocentric measurement and is not one. The
        # fallback stays (a neighbour's distance and bearing in world coordinates
        # is still worth having), but it stops being invisible: a reader can now
        # tell an unrotated frame from a genuinely zero heading, which is the
        # distinction the numbers alone do not carry.
        out["nn_ego_unrotated"] = angles is None

        meta = C.meta_set() & set(df.columns)
        return out.join(df[sorted(meta)])
