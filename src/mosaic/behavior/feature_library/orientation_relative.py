"""
OrientationRelativeFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.spatial.distance import pdist

from .helpers import (
    _pose_column_pairs,  # pyright: ignore[reportPrivateUsage]
    ensure_columns,
    load_result_for,
    wrap_angle,
)
from .spec import COLUMNS as C
from .spec import BodyScaleResult, Inputs, Params, TrackInput, register_feature


@final
@register_feature
class OrientationRelativeFeature:
    """
    Orientation-aware relative features between animal pairs, order-agnostic to pose points.

    For each frame and ordered pair (id_a -> id_b):
      - Express B in A's body frame (using heading angle and global scale).
      - Emit signed centroid deltas, heading difference, quantiles over B's points
        in A's frame, and nearest-k distances.
    """

    name = "orientation-rel"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        """Orientation-relative feature parameters.

        Attributes:
            scale: Body-scale artifact for normalization.
            nearest_k: Number of nearest pose-point distances to emit.
                Default 3.
            quantiles: Distance distribution quantiles to compute.
                Default [0.25, 0.5, 0.75].
        """

        scale: BodyScaleResult = Field(default_factory=BodyScaleResult)
        nearest_k: int = Field(default=3, ge=1)
        quantiles: list[float] = Field(default=[0.25, 0.5, 0.75])

    def __init__(
        self,
        inputs: OrientationRelativeFeature.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._scale_index: pd.DataFrame | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        self._scale_index = None
        if dependency_indices and "scale" in dependency_indices:
            self._scale_index = dependency_indices["scale"]
        return True

    def _scale_for(self, group: str, sequence: str) -> float | None:
        """Look up mean body scale for the given sequence from the dependency index."""
        if self._scale_index is None:
            return None

        df_scale = load_result_for(self._scale_index, group, sequence)
        mean_scale = float(df_scale["scale"].dropna().mean())
        if np.isfinite(mean_scale) and mean_scale > 0:
            return mean_scale
        return None

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        ensure_columns(df, [C.frame_col, C.id_col, C.orientation_col])
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
        sequence = str(df[C.seq_col].iloc[0]) if C.seq_col in df.columns else ""
        global_scale = self._scale_for(group, sequence)
        quantiles = self.params.quantiles
        nearest_k = self.params.nearest_k
        rows: list[dict[str, object]] = []

        x_cols = [x for x, _ in pose_pairs]
        y_cols = [y for _, y in pose_pairs]

        grouped = df.groupby([C.frame_col, C.id_col], sort=True)
        pose_cache: dict[tuple[int, object], dict] = {}
        for (frame_val, id_val), sub in grouped:
            row = sub.iloc[0]
            xs = row[x_cols].to_numpy(dtype=float)
            ys = row[y_cols].to_numpy(dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() < 2:
                continue
            arr = np.column_stack((xs[valid], ys[valid]))
            centroid = arr.mean(axis=0)
            angle = float(row.get(C.orientation_col, 0.0))
            pose_cache[(int(frame_val), id_val)] = {
                "pts": arr,
                "centroid": centroid,
                "angle": angle,
            }

        frames = sorted({f for f, _ in pose_cache})
        for f in frames:
            ids_here = [i for (fv, i) in pose_cache if fv == f]
            if len(ids_here) < 2:
                continue
            for id_a in ids_here:
                for id_b in ids_here:
                    if id_a == id_b:
                        continue
                    cache_a = pose_cache[(f, id_a)]
                    cache_b = pose_cache[(f, id_b)]
                    pts_b = cache_b["pts"]
                    centroid_a = cache_a["centroid"]
                    centroid_b = cache_b["centroid"]
                    angle_a = cache_a["angle"]
                    angle_b = cache_b["angle"]
                    scale = global_scale
                    if scale is None or not np.isfinite(scale) or scale <= 0:
                        scale = _local_scale(cache_a["pts"])
                    if scale is None or scale <= 0:
                        continue
                    delta = pts_b - centroid_a
                    rot = _rotation(-angle_a)
                    rel = (rot @ delta.T).T / scale
                    rel_centroid = (rot @ (centroid_b - centroid_a)) / scale
                    dtheta = wrap_angle(np.array(angle_b - angle_a))

                    dists = np.sqrt((rel**2).sum(axis=1))
                    q_vals = np.quantile(dists, quantiles).tolist() if quantiles else []
                    d_near = np.sort(dists)[:nearest_k].tolist()
                    d_near += [np.nan] * (nearest_k - len(d_near))

                    x_vals = rel[:, 0]
                    y_vals = rel[:, 1]
                    feats: dict[str, object] = {
                        C.frame_col: int(f),
                        "id_a": id_a,
                        "id_b": id_b,
                        C.seq_col: sequence,
                        C.group_col: group,
                        "dx": float(rel_centroid[0]),
                        "dy": float(rel_centroid[1]),
                        "dtheta": float(dtheta),
                        "dist_min": float(np.min(dists)),
                        "dist_median": float(np.median(dists)),
                        "dist_max": float(np.max(dists)),
                    }
                    for idx, qv in enumerate(q_vals):
                        feats[f"dist_q{int(quantiles[idx] * 100):02d}"] = float(qv)
                    feats["x_min"] = float(np.nanmin(x_vals))
                    feats["x_median"] = float(np.nanmedian(x_vals))
                    feats["x_max"] = float(np.nanmax(x_vals))
                    feats["y_min"] = float(np.nanmin(y_vals))
                    feats["y_median"] = float(np.nanmedian(y_vals))
                    feats["y_max"] = float(np.nanmax(y_vals))
                    for i, val in enumerate(d_near):
                        feats[f"near_{i}"] = float(val) if np.isfinite(val) else np.nan
                    rows.append(feats)

        return pd.DataFrame(rows)


def _rotation(angle: float) -> np.ndarray:
    """2D rotation matrix."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def _local_scale(pts: np.ndarray) -> float | None:
    """Median pairwise distance as fallback body scale."""
    if pts.shape[0] < 2:
        return None
    dists = pdist(pts)
    val = float(np.median(dists))
    return val if np.isfinite(val) and val > 0 else None
