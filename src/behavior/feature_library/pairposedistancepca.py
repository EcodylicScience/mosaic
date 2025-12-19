from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from behavior.dataset import register_feature
from behavior.features import _merge_params
from behavior.helpers import to_safe_name


@register_feature
class PairPoseDistancePCA:
    """
    'pair-posedistance-pca' — builds per-frame pairwise pose-distance features and
    fits an IncrementalPCA globally; outputs PC scores per sequence (and perspective).
    """

    name = "pair-posedistance-pca"
    version = "0.1"

    _defaults = dict(
        pose_n=7,
        x_prefix="poseX",
        y_prefix="poseY",
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),
        include_intra_A=True,
        include_intra_B=True,
        include_inter=True,
        duplicate_perspective=True,
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,
        n_components=6,
        batch_size=5000,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ipca: Optional[IncrementalPCA] = IncrementalPCA(
            n_components=self.params["n_components"],
            batch_size=self.params["batch_size"],
        )
        self._fitted = False
        self._tri_i: Optional[np.ndarray] = None
        self._tri_j: Optional[np.ndarray] = None
        self._feat_len: Optional[int] = None

    # ---------- Feature protocol ----------
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return True
    def finalize_fit(self) -> None: pass

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        for df in X_iter:
            self.partial_fit(df)

    def partial_fit(self, df: pd.DataFrame) -> None:
        for Xb, _, _ in self._feature_batches(df, for_fit=True):
            if Xb.size == 0:
                continue
            self._ipca.partial_fit(Xb)
            self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("pair-posedistance-pca: not fitted yet; run fit/partial_fit first.")

        pcs: List[pd.DataFrame] = []
        for Xb, meta_frames, meta_persp in self._feature_batches(df, for_fit=False):
            if Xb.size == 0:
                continue
            Zb = self._ipca.transform(Xb)
            out = pd.DataFrame(Zb, columns=[f"PC{i}" for i in range(Zb.shape[1])])
            if "frame" in meta_frames:
                out["frame"] = meta_frames["frame"]
            if "time" in meta_frames:
                out["time"] = meta_frames["time"]
            out["perspective"] = meta_persp
            for col in (self.params["seq_col"], self.params["group_col"]):
                if col in df.columns:
                    out[col] = df[col].iloc[0]
            pcs.append(out)

        if not pcs:
            return pd.DataFrame(columns=["perspective"] + [f"PC{i}" for i in range(self.params["n_components"])])

        out_df = pd.concat(pcs, ignore_index=True)
        if "frame" in out_df.columns:
            out_df = out_df.sort_values(["perspective", "frame"]).reset_index(drop=True)
        elif "time" in out_df.columns:
            out_df = out_df.sort_values(["perspective", "time"]).reset_index(drop=True)
        return out_df

    def save_model(self, path: Path) -> None:
        if not self._fitted:
            raise NotImplementedError("Model not fitted; nothing to save.")
        payload = dict(
            ipca=self._ipca,
            params=self.params,
            tri_i=self._tri_i,
            tri_j=self._tri_j,
            feat_len=self._feat_len,
        )
        joblib.dump(payload, path)

    def load_model(self, path: Path) -> None:
        obj = joblib.load(path)
        self._ipca = obj["ipca"]
        self.params = _merge_params(obj.get("params", {}), self._defaults)
        self._tri_i = obj.get("tri_i", None)
        self._tri_j = obj.get("tri_j", None)
        self._feat_len = obj.get("feat_len", None)
        self._fitted = True

    # ---------- Internals ----------
    def _column_names(self) -> Tuple[List[str], List[str]]:
        N = int(self.params["pose_n"])
        xs = [f"{self.params['x_prefix']}{i}" for i in range(N)]
        ys = [f"{self.params['y_prefix']}{i}" for i in range(N)]
        return xs, ys

    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column to order rows.")

    def _clean_one_animal(self, g: pd.DataFrame, pose_cols: List[str], order_col: str) -> pd.DataFrame:
        p = self.params
        g = g.sort_values(order_col).copy()
        g = g.set_index(order_col)
        g[pose_cols] = g[pose_cols].replace([np.inf, -np.inf], np.nan)
        g[pose_cols] = g[pose_cols].interpolate(
            method="linear", limit=int(p["linear_interp_limit"]), limit_direction="both"
        )
        g[pose_cols] = g[pose_cols].ffill(limit=int(p["edge_fill_limit"]))
        g[pose_cols] = g[pose_cols].bfill(limit=int(p["edge_fill_limit"]))
        miss_frac = g[pose_cols].isna().mean(axis=1)
        g = g.loc[miss_frac <= float(p["max_missing_fraction"])].copy()
        if g[pose_cols].isna().any().any():
            med = g[pose_cols].median()
            g[pose_cols] = g[pose_cols].fillna(med)
        g = g.reset_index()
        return g

    def _prep_pairs(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[Any, Any, Any]]]:
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)

        need = [self.params["id_col"], self.params["seq_col"], order_col] + pose_cols
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[pair-posedistance-pca] Missing cols: {missing}")

        df_small = df[need].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        group_cols = [self.params["seq_col"], self.params["id_col"]]

        def wrapped_func(g):
            result = self._clean_one_animal(g, pose_cols, order_col)
            if isinstance(g.name, tuple):
                for col, val in zip(group_cols, g.name):
                    result[col] = val
            else:
                result[group_cols[0]] = g.name
            return result

        df_small = (
            df_small
            .groupby(group_cols, group_keys=False)
            .apply(wrapped_func, include_groups=False)
        )

        pairs: List[Tuple[Any, Any, Any]] = []
        for seq, gseq in df_small.groupby(self.params["seq_col"]):
            ids = sorted(gseq[self.params["id_col"]].unique())
            if len(ids) < 2:
                continue
            idA, idB = ids[:2]
            pairs.append((seq, idA, idB))

        if not pairs:
            raise ValueError("[pair-posedistance-pca] No sequence with at least two IDs found.")

        if self._tri_i is None or self._tri_j is None or self._feat_len is None:
            N = int(self.params["pose_n"])
            tri_i, tri_j = np.tril_indices(N, k=-1)
            n_intra = len(tri_i)
            n_cross = N * N
            feat_len = 0
            if self.params["include_intra_A"]:
                feat_len += n_intra
            if self.params["include_intra_B"]:
                feat_len += n_intra
            if self.params["include_inter"]:
                feat_len += n_cross
            self._tri_i, self._tri_j, self._feat_len = tri_i, tri_j, feat_len
        return df_small, pairs

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
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)

        df_small, pairs = self._prep_pairs(df)
        bs = int(self.params["batch_size"])
        dup = bool(self.params["duplicate_perspective"])

        for seq, idA, idB in pairs:
            gseq = df_small[df_small[self.params["seq_col"]] == seq]
            A = gseq[gseq[self.params["id_col"]] == idA][[order_col] + pose_cols].copy()
            B = gseq[gseq[self.params["id_col"]] == idB][[order_col] + pose_cols].copy()
            A = A.sort_values(order_col); B = B.sort_values(order_col)
            AB = A.merge(B, on=order_col, suffixes=("_A", "_B"))
            if AB.empty:
                continue

            n = len(AB)
            for i in range(0, n, bs):
                j = min(i + bs, n)
                chunk = AB.iloc[i:j]
                XA = chunk[[c + "_A" for c in pose_cols]].to_numpy(dtype=float)
                XB = chunk[[c + "_B" for c in pose_cols]].to_numpy(dtype=float)
                feats = [self._build_pair_feat(a, b) for a, b in zip(XA, XB)]
                X = np.vstack(feats).astype(np.float32, copy=False)

                persp = np.zeros(X.shape[0], dtype=np.int8)
                frames_meta: Dict[str, np.ndarray] = {}
                if "frame" in df.columns:
                    frames_meta["frame"] = chunk[order_col].to_numpy()

                if dup:
                    feats2 = [self._build_pair_feat(b, a) for a, b in zip(XA, XB)]
                    X2 = np.vstack(feats2).astype(np.float32, copy=False)
                    X = np.vstack([X, X2])
                    persp = np.concatenate([persp, np.ones(X2.shape[0], dtype=np.int8)], axis=0)
                    if "frame" in frames_meta:
                        frames_meta["frame"] = np.concatenate([frames_meta["frame"], frames_meta["frame"]], axis=0)

                if self._feat_len is not None and X.shape[1] != self._feat_len:
                    raise ValueError(f"Feature length mismatch: got {X.shape[1]}, expected {self._feat_len}")

                yield X, frames_meta, persp

    def _pose_to_points(self, row_vals: np.ndarray) -> np.ndarray:
        N = int(self.params["pose_n"])
        xs = row_vals[:N]; ys = row_vals[N:]
        return np.stack([xs, ys], axis=1)

    def _intra_lower_tri(self, pts: np.ndarray) -> np.ndarray:
        dif = pts[self._tri_i] - pts[self._tri_j]
        return np.sqrt((dif ** 2).sum(axis=1))

    def _inter_all(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        dif = A[:, None, :] - B[None, :, :]
        d = np.sqrt((dif ** 2).sum(axis=2))
        return d.ravel()
