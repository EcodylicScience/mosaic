"""
OrientationRelativeFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature
from mosaic.core.dataset import _latest_feature_run_root, _feature_index_path, _feature_run_root
from mosaic.core.helpers import to_safe_name


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
    output_type = "per_frame"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "scale_feature": "body-scale",
            "scale_run_id": None,
            "nearest_k": 3,
            "quantiles": [0.25, 0.5, 0.75],
        }
        self.params = {**defaults, **(params or {})}
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self._ds = None
        self._scale_lookup: dict[str, float] = {}

    def bind_dataset(self, ds):
        self._ds = ds
        self._load_scales()

    def _load_scales(self):
        self._scale_lookup = {}
        if self._ds is None:
            return
        feat = self.params.get("scale_feature", "body-scale")
        run_id = self.params.get("scale_run_id")
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
                p = Path(abs_path)
                if hasattr(self._ds, "remap_path"):
                    p = self._ds.remap_path(p)
                df_scale = pd.read_parquet(p)
                if "scale" not in df_scale.columns:
                    continue
                mean_scale = float(df_scale["scale"].dropna().mean())
                if np.isfinite(mean_scale) and mean_scale > 0:
                    self._scale_lookup[seq_safe] = mean_scale
            except Exception:
                continue

    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def fit(self, X: Iterable[pd.DataFrame]):
        return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "frame" not in df.columns or "id" not in df.columns or "angle" not in df.columns:
            return pd.DataFrame()
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else ""
        sequence = str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
        seq_safe = to_safe_name(sequence)
        global_scale = self._scale_lookup.get(seq_safe, None)
        quantiles = self.params.get("quantiles", [0.25, 0.5, 0.75])
        nearest_k = int(self.params.get("nearest_k", 3))
        rows = []

        grouped = df.groupby(["frame", "id"], sort=True)
        pose_cache: dict[tuple[int, Any], dict] = {}
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

                    dists = np.sqrt((rel ** 2).sum(axis=1))
                    if dists.size == 0:
                        continue
                    q_vals = np.quantile(dists, quantiles).tolist() if quantiles else []
                    dmin = float(np.min(dists))
                    dmax = float(np.max(dists))
                    dmed = float(np.median(dists))
                    d_near = np.sort(dists)[:max(0, nearest_k)].tolist()
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
                        feats[f"dist_q{int(quantiles[idx]*100):02d}"] = float(qv)
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

    def _local_scale(self, pts: np.ndarray) -> Optional[float]:
        if pts is None or pts.shape[0] < 2:
            return None
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        d = d[np.triu_indices_from(d, k=1)]
        if d.size == 0:
            return None
        val = float(np.median(d))
        return val if np.isfinite(val) and val > 0 else None

    def _pose_to_points(self, row_vals: np.ndarray) -> np.ndarray:
        N = int(self.params["pose_n"])
        xs = row_vals[:N]; ys = row_vals[N:]
        return np.stack([xs, ys], axis=1)  # (N,2)

    def _intra_lower_tri(self, pts: np.ndarray) -> np.ndarray:
        dif = pts[self._tri_i] - pts[self._tri_j]
        return np.sqrt((dif ** 2).sum(axis=1))  # (n_intra,)

    def _inter_all(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        dif = A[:, None, :] - B[None, :, :]     # (N,N,2)
        d = np.sqrt((dif ** 2).sum(axis=2))     # (N,N)
        return d.ravel()                        # (N*N,)

    def _build_pair_feat(self, rowA: np.ndarray, rowB: np.ndarray) -> np.ndarray:
        parts = []
        A = self._pose_to_points(rowA)
        B = self._pose_to_points(rowB)
        if self.params["include_intra_A"]:
            parts.append(self._intra_lower_tri(A))
        if self.params["include_intra_B"]:
            parts.append(self._intra_lower_tri(B))
        if self.params["include_inter"]:
            parts.append(self._inter_all(A, B))
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)

    def _feature_batches(self, df: pd.DataFrame, for_fit: bool) -> Iterable[Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
        """
        Yield (X_batch, meta_frames, persp_array) where:
          - X_batch shape (B, F)
          - meta_frames: dict with possible 'frame' and 'time' arrays (aligned with B)
          - persp_array: (B,) of 0/1 (A→B or B→A) if duplicate_perspective=True, else all zeros.
        """
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)

        df_small, pairs = self._prep_pairs(df)
        bs = int(self.params["batch_size"])
        dup = bool(self.params["duplicate_perspective"])

        # build an iterator over all aligned A/B rows per sequence
        for seq, idA, idB in pairs:
            gseq = df_small[df_small[self.params["seq_col"]] == seq]
            A = gseq[gseq[self.params["id_col"]] == idA][[order_col] + pose_cols].copy()
            B = gseq[gseq[self.params["id_col"]] == idB][[order_col] + pose_cols].copy()
            A = A.sort_values(order_col); B = B.sort_values(order_col)
            # inner-join on the order column (frame/time)
            AB = A.merge(B, on=order_col, suffixes=("_A", "_B"))
            if AB.empty:
                continue

            # slice into batches
            n = len(AB)
            for i in range(0, n, bs):
                j = min(i + bs, n)
                chunk = AB.iloc[i:j]
                # build features for A->B
                XA = chunk[[c + "_A" for c in pose_cols]].to_numpy(dtype=float)
                XB = chunk[[c + "_B" for c in pose_cols]].to_numpy(dtype=float)
                feats = [self._build_pair_feat(a, b) for a, b in zip(XA, XB)]
                X = np.vstack(feats).astype(np.float32, copy=False)

                persp = np.zeros(X.shape[0], dtype=np.int8)
                frames_meta: Dict[str, np.ndarray] = {}
                if "frame" in df.columns:
                    frames_meta["frame"] = chunk[order_col].to_numpy()
                if "time" in df.columns and order_col != "time":
                    # optional time passthrough if present in df; we cannot join time unless it's the order key
                    pass

                if dup:
                    # add B->A echoes
                    feats2 = [self._build_pair_feat(b, a) for a, b in zip(XA, XB)]
                    X2 = np.vstack(feats2).astype(np.float32, copy=False)
                    X = np.vstack([X, X2])
                    persp = np.concatenate([persp, np.ones(X2.shape[0], dtype=np.int8)], axis=0)
                    if "frame" in frames_meta:
                        frames_meta["frame"] = np.concatenate([frames_meta["frame"], frames_meta["frame"]], axis=0)

                # first batch determines feat_len sanity
                if self._feat_len is not None and X.shape[1] != self._feat_len:
                    raise ValueError(f"Feature length mismatch: got {X.shape[1]}, expected {self._feat_len}")

                yield X, frames_meta, persp
