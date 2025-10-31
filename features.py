# features.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import IncrementalPCA
try:
    import pywt
    _PYWT_OK = True
except Exception:
    _PYWT_OK = False

# import the registry + protocol from your dataset module
# (adjust the import path if dataset.py lives in a package)
from dataset import register_feature

def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out

@register_feature
class PairPoseDistancePCA:
    """
    'pair-posedistance-pca' — builds per-frame pairwise pose-distance features and
    fits an IncrementalPCA globally; outputs PC scores per sequence (and perspective).
    
    Output of transform(df) is a DataFrame with:
      - frame (if present) and/or time (if present)
      - perspective: 0 for A→B, 1 for B→A (if duplicate_perspective=True)
      - PC0..PC{k-1}
      - (optionally) group/sequence if present in df, for convenience

    Model state (IPCA, mean_, components_, indices) is persisted via save_model().
    """

    # registry-facing metadata
    name    = "pair-posedistance-pca"
    version = "0.1"

    # ---------- Defaults ----------
    _defaults = dict(
        # pose / columns
        pose_n=7,                               # number of pose points per animal
        x_prefix="poseX", y_prefix="poseY",     # TRex-ish column naming
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),           # priority to order frames

        # feature config
        include_intra_A=True,
        include_intra_B=True,
        include_inter=True,
        duplicate_perspective=True,

        # cleaning / interpolation per animal
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,

        # IPCA
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
        # will be set after first feature-shape discovery
        self._tri_i: Optional[np.ndarray] = None
        self._tri_j: Optional[np.ndarray] = None
        self._feat_len: Optional[int] = None

    # ------------- Feature protocol -------------
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return True
    def finalize_fit(self) -> None: pass

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        for df in X_iter:
            self.partial_fit(df)

    def partial_fit(self, df: pd.DataFrame) -> None:
        # stream feature batches
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
            Zb = self._ipca.transform(Xb)  # (B, k)
            out = pd.DataFrame(Zb, columns=[f"PC{i}" for i in range(Zb.shape[1])])

            # bring back frame/time if we have it
            if "frame" in meta_frames:
                out["frame"] = meta_frames["frame"]
            if "time" in meta_frames:
                out["time"] = meta_frames["time"]
            out["perspective"] = meta_persp  # 0 or 1
            # optional pass-through if present and constant in df
            for col in (self.params["seq_col"], self.params["group_col"]):
                if col in df.columns:
                    out[col] = df[col].iloc[0]
            pcs.append(out)

        if not pcs:
            return pd.DataFrame(columns=["perspective"] + [f"PC{i}" for i in range(self.params["n_components"])])

        out_df = pd.concat(pcs, ignore_index=True)
        # sort if frame present
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

    # ------------- Internals -------------
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
        # interpolate, then edge-fill
        g[pose_cols] = g[pose_cols].replace([np.inf, -np.inf], np.nan)
        g[pose_cols] = g[pose_cols].interpolate(
            method="linear", limit=int(p["linear_interp_limit"]), limit_direction="both"
        )
        g[pose_cols] = g[pose_cols].ffill(limit=int(p["edge_fill_limit"]))
        g[pose_cols] = g[pose_cols].bfill(limit=int(p["edge_fill_limit"]))
        # drop frames with too much missing (row-wise)
        miss_frac = g[pose_cols].isna().mean(axis=1)
        g = g.loc[miss_frac <= float(p["max_missing_fraction"])].copy()
        # last fill (median) if needed
        if g[pose_cols].isna().any().any():
            med = g[pose_cols].median()
            g[pose_cols] = g[pose_cols].fillna(med)
        g = g.reset_index()
        return g

    def _prep_pairs(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[Any, Any, Any]]]:
        """
        Return cleaned df_small with only needed cols and the pairs index:
        [(sequence, idA, idB), ...] choosing the first two IDs per sequence.
        """
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)

        need = [self.params["id_col"], self.params["seq_col"], order_col] + pose_cols
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[pair-posedistance-pca] Missing cols: {missing}")

        # sanitize types & clean per-animal
        df_small = df[need].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        # future-proof grouping with include_groups=True functionality
        group_cols = [self.params["seq_col"], self.params["id_col"]]

        def wrapped_func(g):
            # `g.name` holds the current group key(s)
            result = self._clean_one_animal(g, pose_cols, order_col)

            # Reattach group key(s) as columns (since they’re no longer in `g`)
            if isinstance(g.name, tuple):
                for col, val in zip(group_cols, g.name):
                    result[col] = val
            else:
                result[group_cols[0]] = g.name

            return result

        df_small = (
            df_small
            .groupby(group_cols, group_keys=False)
            .apply(wrapped_func, include_groups=False)  # explicitly future-proof
        )        

        # build (seq -> first two ids)
        pairs: List[Tuple[Any, Any, Any]] = []
        for seq, gseq in df_small.groupby(self.params["seq_col"]):
            ids = sorted(gseq[self.params["id_col"]].unique())
            if len(ids) < 2:
                continue
            idA, idB = ids[:2]
            pairs.append((seq, idA, idB))

        if not pairs:
            raise ValueError("[pair-posedistance-pca] No sequence with at least two IDs found.")

        # cache lower-tri indices and feature length once
        if self._tri_i is None or self._tri_j is None or self._feat_len is None:
            N = int(self.params["pose_n"])
            tri_i, tri_j = np.tril_indices(N, k=-1)
            n_intra = len(tri_i)
            n_cross = N * N
            feat_len = 0
            if self.params["include_intra_A"]: feat_len += n_intra
            if self.params["include_intra_B"]: feat_len += n_intra
            if self.params["include_inter"]:   feat_len += n_cross
            self._tri_i, self._tri_j, self._feat_len = tri_i, tri_j, feat_len
        return df_small, pairs

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

@register_feature
class PairEgocentricFeatures:
    """
    'pair-egocentric' — per-sequence egocentric + kinematic features for dyads.
    Produces a row-wise DataFrame with columns:
      - frame (if available) or time passthrough (only if it's the order col)
      - perspective: 0 for A→B, 1 for B→A
      - feature columns (e.g., A_speed, AB_dx_egoA, ...)
      - (optionally) group/sequence if present in df, for convenience

    This feature is *stateless* (no fitting). It infers per-sequence dyads by taking
    the first two IDs present in each sequence, cleans/interpolates pose per animal,
    inner-joins by the chosen order column, and computes A→B and B→A features.
    """

    name    = "pair-egocentric"
    version = "0.1"

    _defaults = dict(
        # pose / columns
        pose_n=7,
        x_prefix="poseX", y_prefix="poseY",   # TRex-ish
        id_col="id",
        seq_col="sequence",
        group_col="group",
        order_pref=("frame", "time"),

        # required anatomical indices (must be provided by user if different)
        neck_idx=None,           # REQUIRED (int) unless your skeleton matches defaults
        tail_base_idx=None,      # REQUIRED (int) unless your skeleton matches defaults
        center_mode="mean",      # "mean" or an int landmark index

        # sampling / smoothing
        fps_default=30.0,
        smooth_win=0,            # 0 disables box smoothing before differencing

        # cleaning / interpolation per animal
        linear_interp_limit=10,
        edge_fill_limit=3,
        max_missing_fraction=0.10,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        # Enforce required indices if user didn't pass them
        if self.params["neck_idx"] is None:
            # sensible default matching your earlier snippet
            self.params["neck_idx"] = 3
        if self.params["tail_base_idx"] is None:
            self.params["tail_base_idx"] = 6

        self._tri_ready = False  # not used, but kept for symmetry with other feature

    # ------------- Feature protocol -------------
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def finalize_fit(self) -> None: pass
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return
    def partial_fit(self, df: pd.DataFrame) -> None: return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x_cols, y_cols = self._column_names()
        pose_cols = x_cols + y_cols
        order_col = self._order_col(df)
        p = self.params

        need = [p["id_col"], p["seq_col"], order_col] + pose_cols
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"[pair-egocentric] Missing cols: {missing}")

        # Clean per-animal, per-sequence (future-proof re: pandas groupby.apply)
        df_small = df[need].copy()
        if order_col == "frame":
            df_small[order_col] = df_small[order_col].astype(int, errors="ignore")

        group_cols = [p["seq_col"], p["id_col"]]

        def wrapped_func(g):
            result = self._clean_one_animal(g, pose_cols, order_col)
            # reattach group key(s)
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

        # Build dyads (first two IDs per sequence)
        pairs = []
        for seq, gseq in df_small.groupby(p["seq_col"]):
            ids = sorted(gseq[p["id_col"]].unique())
            if len(ids) >= 2:
                pairs.append((seq, ids[0], ids[1]))

        if not pairs:
            raise ValueError("[pair-egocentric] No sequence with at least two IDs found.")

        out_frames: List[pd.DataFrame] = []
        for seq, idA, idB in pairs:
            gseq = df_small[df_small[p["seq_col"]] == seq]
            A = gseq[gseq[p["id_col"]] == idA][[order_col] + pose_cols].copy()
            B = gseq[gseq[p["id_col"]] == idB][[order_col] + pose_cols].copy()
            if A.empty or B.empty:
                continue

            A = A.sort_values(order_col).rename(columns={order_col: "frame"})
            B = B.sort_values(order_col).rename(columns={order_col: "frame"})
            j = A.merge(B, on="frame", suffixes=("_A", "_B"))
            if j.empty:
                continue

            # fps heuristic: prefer df['fps'] if present and constant; else default
            fps = float(p["fps_default"])
            if "fps" in df.columns:
                try:
                    c = df["fps"].dropna().unique()
                    if len(c) == 1:
                        fps = float(c[0])
                except Exception:
                    pass

            frames, AtoB, BtoA, names = self._build_ego_block_for_joined(j, fps, pose_cols)

            # produce row-wise DataFrames
            dfA = pd.DataFrame(AtoB.T, columns=names)
            dfA["frame"] = frames
            dfA["perspective"] = 0

            dfB = pd.DataFrame(BtoA.T, columns=names)
            dfB["frame"] = frames
            dfB["perspective"] = 1

            # optional pass-through for convenience (constant per call)
            for col in (p["seq_col"], p["group_col"]):
                if col in df.columns:
                    dfA[col] = df[col].iloc[0]
                    dfB[col] = df[col].iloc[0]

            out_frames.extend([dfA, dfB])

        if not out_frames:
            return pd.DataFrame(columns=["perspective", "frame"])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["perspective", "frame"]).reset_index(drop=True)
        return out

    # ------------- Internals -------------
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

    # --- math helpers ---
    def _smooth_1d(self, x: np.ndarray, win: int) -> np.ndarray:
        if win is None or win <= 1:
            return x
        pad = win // 2
        xp = np.pad(x, pad_width=pad, mode="reflect")
        ker = np.ones(win, dtype=float) / float(win)
        return np.convolve(xp, ker, mode="valid")

    def _safe_unit(self, vx: np.ndarray, vy: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        n = np.sqrt(vx * vx + vy * vy) + eps
        return vx / n, vy / n

    def _angle(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        return np.arctan2(vy, vx)

    def _unwrap_diff(self, theta: np.ndarray, fps: float) -> np.ndarray:
        d = np.gradient(np.unwrap(theta), edge_order=1)
        return d * float(fps)

    def _center_from_points(self, xs: np.ndarray, ys: np.ndarray, mode: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(mode, (int, np.integer)):
            return xs[:, int(mode)], ys[:, int(mode)]
        return xs.mean(axis=1), ys.mean(axis=1)

    def _build_ego_block_for_joined(self, j: pd.DataFrame, fps: float, pose_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        N = int(self.params["pose_n"])
        neck = int(self.params["neck_idx"])
        tail = int(self.params["tail_base_idx"])
        win  = int(self.params["smooth_win"])
        mode = self.params["center_mode"]

        XA = j[[f"{self.params['x_prefix']}{k}_A" for k in range(N)]].to_numpy()
        YA = j[[f"{self.params['y_prefix']}{k}_A" for k in range(N)]].to_numpy()
        XB = j[[f"{self.params['x_prefix']}{k}_B" for k in range(N)]].to_numpy()
        YB = j[[f"{self.params['y_prefix']}{k}_B" for k in range(N)]].to_numpy()
        frames = j["frame"].to_numpy().astype(int)

        # optional smoothing
        if win and win > 1:
            XA = np.vstack([self._smooth_1d(XA[:, k], win) for k in range(N)]).T
            YA = np.vstack([self._smooth_1d(YA[:, k], win) for k in range(N)]).T
            XB = np.vstack([self._smooth_1d(XB[:, k], win) for k in range(N)]).T
            YB = np.vstack([self._smooth_1d(YB[:, k], win) for k in range(N)]).T

        # centers
        cxA, cyA = self._center_from_points(XA, YA, mode)
        cxB, cyB = self._center_from_points(XB, YB, mode)

        # headings (neck - tail) and units
        hxA, hyA = XA[:, neck] - XA[:, tail], YA[:, neck] - YA[:, tail]
        hxB, hyB = XB[:, neck] - XB[:, tail], YB[:, neck] - YB[:, tail]
        uhxA, uhyA = self._safe_unit(hxA, hyA)
        uhxB, uhyB = self._safe_unit(hxB, hyB)
        # left-hand orthogonal
        uoxA, uoyA = -uhyA, uhxA
        uoxB, uoyB = -uhyB, uhxB

        # velocities of centers (per second)
        vAx = np.gradient(cxA) * float(fps)
        vAy = np.gradient(cyA) * float(fps)
        vBx = np.gradient(cxB) * float(fps)
        vBy = np.gradient(cyB) * float(fps)
        speedA = np.sqrt(vAx*vAx + vAy*vAy)
        speedB = np.sqrt(vBx*vBx + vBy*vBy)

        # heading angles + angular speed
        thA = self._angle(uhxA, uhyA)
        thB = self._angle(uhxB, uhyB)
        angspeedA = self._unwrap_diff(thA, fps)
        angspeedB = self._unwrap_diff(thB, fps)

        # ego projections of velocity
        vA_para = vAx * uhxA + vAy * uhyA
        vA_perp = vAx * uoxA + vAy * uoyA
        vB_para = vBx * uhxB + vBy * uhyB
        vB_perp = vBx * uoxB + vBy * uoyB

        # displacement A→B in world + A-centric ego coords of B
        dx = cxB - cxA
        dy = cyB - cyA
        distAB = np.sqrt(dx*dx + dy*dy)

        dxA = dx * uhxA + dy * uhyA
        dyA = dx * uoxA + dy * uoyA

        # B-centric ego coords of A
        dxB = (-dx) * uhxB + (-dy) * uhyB
        dyB = (-dx) * uoxB + (-dy) * uoyB

        # relative heading B wrt A
        dth = np.unwrap(thB) - np.unwrap(thA)
        rel_cos = np.cos(dth)
        rel_sin = np.sin(dth)

        names = [
            "A_speed", "A_v_para", "A_v_perp", "A_ang_speed",
            "A_heading_cos", "A_heading_sin",
            "AB_dist", "AB_dx_egoA", "AB_dy_egoA",
            "rel_heading_cos", "rel_heading_sin",
            "B_speed", "B_v_para", "B_v_perp", "B_ang_speed",
        ]

        AtoB = np.vstack([
            speedA, vA_para, vA_perp, angspeedA,
            np.cos(thA), np.sin(thA),
            distAB, dxA, dyA,
            rel_cos, rel_sin,
            speedB, vB_para, vB_perp, angspeedB,
        ]).astype(np.float32)

        # For B→A, swap roles but keep same semantic ordering (B is 'self')
        BtoA = np.vstack([
            speedB, vB_para, vB_perp, angspeedB,
            np.cos(thB), np.sin(thB),
            distAB, dxB, dyB,
            np.cos(-dth), np.sin(-dth),
            speedA, vA_para, vA_perp, angspeedA,
        ]).astype(np.float32)

        return frames, AtoB, BtoA, names
    


@register_feature
class PairPoseDistanceWavelet:
    """
    'pair-posedistance-wavelet' — CWT spectrograms on PairPoseDistancePCA outputs.
    Expects input df to contain columns:
        - 'perspective' (0 = A→B, 1 = B→A)
        - 'frame' (preferred) or 'time' (if used as order column)
        - PC0..PC{k-1} (k = number of PCA components)
    Returns a DataFrame with columns:
        - frame (or time if that was the order col)
        - perspective
        - W_c{comp}_f{fi}  (log-power, clamped, for each component×frequency)
      and (optionally) passthrough group/sequence if present in df.

    Notes:
      • Stateless (no fitting).
      • FPS is inferred from constant df['fps'] if present; else fps_default.
      • Frequencies are dyadically spaced in [f_min, f_max].
    """

    name = "pair-posedistance-wavelet"
    version = "0.1"

    _defaults = dict(
        # sampling
        fps_default=30.0,

        # wavelet band and resolution
        f_min=0.2,
        f_max=5.0,
        n_freq=25,

        # wavelet family string (PyWavelets)
        wavelet="cmor1.5-1.0",

        # log power clamp
        log_floor=-3.0,

        # naming / passthrough
        pc_prefix="PC",                 # columns like PC0, PC1, ...
        order_pref=("frame", "time"),   # which column to use as the time base
        seq_col="sequence",
        group_col="group",
        cols=None,  # explicit list of columns to transform; if None, fallback to PC prefix or auto-detect numeric columns
    )
    def _select_input_columns(self, df: pd.DataFrame) -> List[str]:
        # 1) explicit columns override
        cols_param = self.params.get("cols", None)
        if cols_param:
            cols = [c for c in cols_param if c in df.columns]
            if not cols:
                raise ValueError("[pair-posedistance-wavelet] None of the requested 'cols' are present in df.")
            return cols
        # 2) PC-prefixed columns
        pc_cols = self._pc_columns(df, self.params["pc_prefix"])
        if pc_cols:
            return pc_cols
        # 3) Auto-detect: all numeric columns except known meta
        meta_like = {self.params.get("seq_col", "sequence"),
                     self.params.get("group_col", "group"),
                     "frame", "time", "perspective", "id", "fps"}
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_like]
        if not num_cols:
            raise ValueError("[pair-posedistance-wavelet] Could not auto-detect numeric feature columns.")
        return num_cols

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _PYWT_OK:
            raise ImportError(
                "PyWavelets (pywt) not available. Install with `pip install PyWavelets`."
            )
        self.params = _merge_params(params, self._defaults)
        # pre-build frequency vector & scales for speed; will recompute if params change
        self._cache_key = None
        self._frequencies = None
        self._scales = None
        self._central_f = None

    # ---- feature protocol ----
    def needs_fit(self) -> bool: return False
    def supports_partial_fit(self) -> bool: return False
    def finalize_fit(self) -> None: pass
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: return
    def partial_fit(self, df: pd.DataFrame) -> None: return

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = self._order_col(df)
        fps = self._infer_fps(df, p["fps_default"])
        in_cols = self._select_input_columns(df)
        if "perspective" not in df.columns:
            raise ValueError("[pair-posedistance-wavelet] Missing 'perspective' column.")

        # prepare wavelet frequencies/scales
        self._prepare_band(fps)

        # compute per perspective block (keeps ordering stable)
        out_blocks: List[pd.DataFrame] = []
        for persp, g in df.groupby("perspective"):
            g = g.sort_values(order_col)
            Z = g[in_cols].to_numpy(dtype=float)  # shape (T, k)
            T, k = Z.shape

            # compute power spectrogram (k components × n_freq × T)
            power = np.empty((k, len(self._frequencies), T), dtype=np.float32)
            # each component independently
            for comp in range(k):
                coeffs, _ = pywt.cwt(
                    Z[:, comp],
                    self._scales,
                    self._wavelet_obj(),
                    sampling_period=1.0 / float(fps),
                )
                power[comp] = (np.abs(coeffs) ** 2).astype(np.float32)

            # log + clamp
            eps = np.finfo(np.float32).tiny
            log_power = np.log(power + eps)
            log_power = np.maximum(log_power, float(p["log_floor"]))

            # flatten to (T, k*n_freq)
            flat = log_power.reshape(k * len(self._frequencies), T).T  # (T, F_flat)

            # column names: W_{in_cols[comp]}_f{fi}
            colnames = [
                f"W_{in_cols[comp]}_f{fi}"
                for comp in range(k)
                for fi in range(len(self._frequencies))
            ]
            block = pd.DataFrame(flat, columns=colnames)
            block[order_col] = g[order_col].to_numpy()
            block["perspective"] = int(persp)

            # optional passthrough
            for col in (p["seq_col"], p["group_col"]):
                if col in df.columns:
                    block[col] = df[col].iloc[0]

            out_blocks.append(block)

        if not out_blocks:
            return pd.DataFrame(columns=[order_col, "perspective"])

        out = pd.concat(out_blocks, ignore_index=True)
        out = out.sort_values(["perspective", order_col]).reset_index(drop=True)

        # Attach JSON-serializable metadata only (so parquet writers won't error)
        try:
            out.attrs["frequencies_hz"] = self._frequencies.tolist() if self._frequencies is not None else []
            out.attrs["scales"] = self._scales.tolist() if self._scales is not None else []
            out.attrs["wavelet"] = str(self.params.get("wavelet", ""))
            out.attrs["fps"] = float(fps)
            out.attrs["pc_cols"] = [c for c in in_cols if c.startswith(self.params.get("pc_prefix","PC"))]
            out.attrs["input_columns"] = list(map(str, in_cols))
        except Exception:
            # As a safety net, drop attrs if anything is not serializable
            out.attrs.clear()
        return out

    # ---- internals ----
    def _order_col(self, df: pd.DataFrame) -> str:
        for c in self.params["order_pref"]:
            if c in df.columns:
                return c
        raise ValueError("Need either 'frame' or 'time' column in df.")

    def _infer_fps(self, df: pd.DataFrame, default: float) -> float:
        if "fps" in df.columns:
            vals = pd.Series(df["fps"]).dropna().unique()
            if len(vals) == 1:
                try:
                    return float(vals[0])
                except Exception:
                    pass
        return float(default)

    def _pc_columns(self, df: pd.DataFrame, prefix: str) -> List[str]:
        # accept PC0, PC1, ... contiguous from 0 until missing
        pc_cols = []
        i = 0
        while True:
            col = f"{prefix}{i}"
            if col in df.columns:
                pc_cols.append(col)
                i += 1
            else:
                break
        return pc_cols

    def _prepare_band(self, fps: float) -> None:
        key = (self.params["wavelet"], float(self.params["f_min"]),
               float(self.params["f_max"]), int(self.params["n_freq"]), float(fps))
        if self._cache_key == key and self._frequencies is not None:
            return
        f_min = float(self.params["f_min"])
        f_max = float(self.params["f_max"])
        n_freq = int(self.params["n_freq"])
        # dyadic spacing
        freqs = 2.0 ** np.linspace(np.log2(f_min), np.log2(f_max), n_freq)
        w = self._wavelet_obj()
        central_f = pywt.central_frequency(w)
        scales = float(fps) / (freqs * central_f)
        self._frequencies = freqs.astype(np.float32)
        self._scales = scales.astype(np.float32)
        self._central_f = float(central_f)
        self._cache_key = key

    def _wavelet_obj(self):
        return pywt.ContinuousWavelet(self.params["wavelet"])    