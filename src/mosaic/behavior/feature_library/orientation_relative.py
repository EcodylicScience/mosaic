"""
OrientationRelativeFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.dataset import (
    _feature_index_path,
    _latest_feature_run_root,
    register_feature,
)
from mosaic.core.helpers import to_safe_name

from .helpers import _pose_column_pairs
from .params import Inputs, OutputType, Params, TrackInput


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
    output_type: OutputType = "per_frame"

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        """Orientation-relative feature parameters.

        Attributes:
            scale_feature: Body-scale feature to load for normalization.
                Default "body-scale".
            scale_run_id: Specific run ID for the scale feature.
                None picks the latest finished run.
            nearest_k: Number of nearest pose-point distances to emit.
                Default 3.
            quantiles: Distance distribution quantiles to compute.
                Default [0.25, 0.5, 0.75].
        """

        scale_feature: str = "body-scale"
        scale_run_id: str | None = None
        nearest_k: int = Field(default=3, ge=1)
        quantiles: list[float] = Field(default=[0.25, 0.5, 0.75])

    def __init__(
        self,
        inputs: OrientationRelativeFeature.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self._ds = None
        self._scale_lookup: dict[str, float] = {}
        self._scope_filter: dict[str, object] = {}

    def bind_dataset(self, ds):
        self._ds = ds
        self._load_scales()

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    def _load_scales(self):
        self._scale_lookup = {}
        if self._ds is None:
            return
        feat = self.params.scale_feature
        run_id = self.params.scale_run_id
        if run_id is None:
            try:
                run_id, _ = _latest_feature_run_root(self._ds, feat)
            except Exception:
                return
        idx_path = _feature_index_path(self._ds, feat)
        if not idx_path.exists():
            return
        df_idx = pd.read_csv(idx_path)
        df_idx = df_idx[df_idx["run_id"].astype(str) == str(run_id)]
        if df_idx.empty:
            return
        for _, row in df_idx.iterrows():
            seq_safe = row.get("sequence_safe") or to_safe_name(row.get("sequence", ""))
            abs_path = row.get("abs_path")
            if not abs_path:
                continue
            try:
                p = self._ds.resolve_path(abs_path)
                df_scale = pd.read_parquet(p)
                if "scale" not in df_scale.columns:
                    continue
                mean_scale = float(df_scale["scale"].dropna().mean())
                if np.isfinite(mean_scale) and mean_scale > 0:
                    self._scale_lookup[seq_safe] = mean_scale
            except Exception:
                continue

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]):
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def save_model(self, path: Path) -> None:
        raise NotImplementedError

    def load_model(self, path: Path) -> None:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if (
            "frame" not in df.columns
            or "id" not in df.columns
            or "angle" not in df.columns
        ):
            return pd.DataFrame()
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else ""
        sequence = (
            str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
        )
        seq_safe = to_safe_name(sequence)
        global_scale = self._scale_lookup.get(seq_safe, None)
        quantiles = self.params.quantiles
        nearest_k = self.params.nearest_k
        rows = []

        grouped = df.groupby(["frame", "id"], sort=True)
        pose_cache: dict[tuple[int, object], dict] = {}
        for (frame_val, id_val), sub in grouped:
            pts = []
            for x_col, y_col in pose_pairs:
                x = sub.iloc[0].get(x_col)
                y = sub.iloc[0].get(y_col)
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    continue
                pts.append((float(x), float(y)))
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=float)
            centroid = arr.mean(axis=0)
            angle = float(sub.iloc[0].get("angle", 0.0))
            pose_cache[(int(frame_val), id_val)] = {
                "pts": arr,
                "centroid": centroid,
                "angle": angle,
            }

        frames = sorted({f for f, _ in pose_cache.keys()})
        for f in frames:
            ids_here = [i for (frame_val, i) in pose_cache.keys() if frame_val == f]
            if len(ids_here) < 2:
                continue
            for id_a in ids_here:
                for id_b in ids_here:
                    if id_a == id_b:
                        continue
                    A = pose_cache.get((f, id_a))
                    B = pose_cache.get((f, id_b))
                    if not A or not B:
                        continue
                    pts_B = B["pts"]
                    centroid_B = B["centroid"]
                    angle_A = A["angle"]
                    angle_B = B["angle"]
                    centroid_A = A["centroid"]
                    scale = global_scale
                    if scale is None or not np.isfinite(scale) or scale <= 0:
                        scale = self._local_scale(A["pts"])
                    if scale is None or scale <= 0:
                        continue
                    delta = pts_B - centroid_A
                    rot = self._rotation(-angle_A)
                    rel = (rot @ delta.T).T / scale
                    rel_centroid = (rot @ (centroid_B - centroid_A)) / scale
                    dtheta = self._wrap_angle(angle_B - angle_A)

                    dists = np.sqrt((rel**2).sum(axis=1))
                    if dists.size == 0:
                        continue
                    q_vals = np.quantile(dists, quantiles).tolist() if quantiles else []
                    dmin = float(np.min(dists))
                    dmax = float(np.max(dists))
                    dmed = float(np.median(dists))
                    d_near = np.sort(dists)[: max(0, nearest_k)].tolist()
                    d_near += [np.nan] * (max(0, nearest_k) - len(d_near))

                    x_vals = rel[:, 0]
                    y_vals = rel[:, 1]
                    feats = {
                        "frame": int(f),
                        "id_a": id_a,
                        "id_b": id_b,
                        "sequence": sequence,
                        "group": group,
                        "dx": float(rel_centroid[0]),
                        "dy": float(rel_centroid[1]),
                        "dtheta": float(dtheta),
                        "dist_min": dmin,
                        "dist_median": dmed,
                        "dist_max": dmax,
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

    def _rotation(self, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _wrap_angle(self, a: float) -> float:
        a = (a + np.pi) % (2 * np.pi) - np.pi
        return a

    def _local_scale(self, pts: np.ndarray) -> float | None:
        if pts is None or pts.shape[0] < 2:
            return None
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        d = d[np.triu_indices_from(d, k=1)]
        if d.size == 0:
            return None
        val = float(np.median(d))
        return val if np.isfinite(val) and val > 0 else None

    # Pair-distance helpers removed; see pairposedistancepca.py
