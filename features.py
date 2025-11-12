# features.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, Tuple, List
from collections import defaultdict
import hashlib, json
import numpy as np
import pandas as pd
import joblib
import re, sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openTSNE import TSNEEmbedding, affinity, initialization
from scipy.cluster.hierarchy import linkage as _sch_linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.neighbors import NearestNeighbors
from helpers import to_safe_name


try:
    import pywt
    _PYWT_OK = True
except Exception:
    _PYWT_OK = False

# import the registry + protocol from your dataset module
from dataset import register_feature
# features.py



INHERIT_REGEX = "<<inherit>>"

try:
    # import helpers from dataset.py
    from dataset import register_feature, _feature_run_root, _feature_index_path, \
                        _latest_feature_run_root
except Exception:
    def register_feature(cls): return cls
    def _latest_feature_run_root(ds, name): raise RuntimeError("Bind dataset first")
    def _feature_run_root(ds, name, run_id): raise RuntimeError("Bind dataset first")
    def _feature_index_path(ds, name): raise RuntimeError("Bind dataset first")

def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


INPUTSET_DIRNAME = "inputsets"


def _dataset_base_dir(ds) -> Path:
    """
    Resolve the directory that holds dataset-level config (sibling to dataset manifest).
    """
    base = getattr(ds, "manifest_path", None)
    if base is not None:
        base = Path(base)
        base = base.parent if base.is_file() else base
    else:
        # Fall back to features root parent
        base = Path(ds.get_root("features")).parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def _inputset_dir(ds) -> Path:
    base = _dataset_base_dir(ds)
    path = base / INPUTSET_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _inputset_path(ds, name: str) -> Path:
    safe = to_safe_name(name)
    if not safe:
        raise ValueError("Inputset name must contain alphanumeric characters.")
    return _inputset_dir(ds) / f"{safe}.json"


def save_inputset(ds, name: str, inputs: list[dict], description: Optional[str] = None,
                  overwrite: bool = False) -> Path:
    """
    Persist an inputset JSON under <dataset_root>/inputsets/<name>.json.
    """
    path = _inputset_path(ds, name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Inputset '{name}' already exists: {path}")
    payload = {
        "name": name,
        "description": description or "",
        "inputs": inputs or [],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _fingerprint_inputs(inputs: list[dict]) -> str:
    serialized = json.dumps(inputs or [], sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _load_inputset(ds, name: str) -> tuple[list[dict], dict]:
    path = _inputset_path(ds, name)
    if not path.exists():
        raise FileNotFoundError(f"Inputset '{name}' not found at {path}")
    data = json.loads(path.read_text())
    inputs = data.get("inputs") or []
    fingerprint = _fingerprint_inputs(inputs)
    meta = {
        "inputset": name,
        "inputs_fingerprint": fingerprint,
        "inputs_source": "inputset",
        "inputset_path": str(path),
        "description": data.get("description", ""),
    }
    return inputs, meta


def _resolve_inputs(ds, explicit_inputs: Optional[list[dict]], inputset_name: Optional[str],
                    explicit_override: bool = False) -> tuple[list[dict], dict]:
    """
    Determine which inputs to use based on explicit params vs. named inputset.
    If inputset_name is provided, it overrides defaults unless explicit_inputs was
    explicitly supplied by the caller (explicit_override=True).
    """
    if inputset_name:
        inputs, meta = _load_inputset(ds, inputset_name)
        if explicit_inputs and explicit_override:
            inputs = explicit_inputs
            meta = {
                "inputset": inputset_name,
                "inputs_fingerprint": _fingerprint_inputs(inputs),
                "inputs_source": "explicit",
            }
    else:
        inputs = explicit_inputs or []
        meta = {
            "inputset": None,
            "inputs_fingerprint": _fingerprint_inputs(inputs),
            "inputs_source": "explicit" if explicit_inputs else "default",
        }

    if not inputs:
        raise ValueError("No inputs resolved; provide params['inputs'] or params['inputset'].")
    return inputs, meta


def _load_array_from_spec(path: Path, load_spec: dict) -> Optional[np.ndarray]:
    kind = str(load_spec.get("kind", "parquet")).lower()
    transpose = bool(load_spec.get("transpose", False))
    if kind == "npz":
        key = load_spec.get("key")
        if not key:
            raise ValueError("load.kind='npz' requires 'key'")
        npz = np.load(path, allow_pickle=True)
        if key not in npz.files:
            return None
        A = np.asarray(npz[key])
        if A.ndim == 1:
            A = A[None, :]
    elif kind == "parquet":
        df = pd.read_parquet(path)
        drop_cols = load_spec.get("drop_columns")
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        cols = load_spec.get("columns")
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        elif load_spec.get("numeric_only", True):
            df = df.select_dtypes(include=[np.number])
        else:
            df = df.apply(pd.to_numeric, errors="coerce")
        A = df.to_numpy(dtype=np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported load.kind='{kind}'")

    if A.size == 0:
        return None
    if transpose:
        A = A.T
    if A.ndim == 1:
        A = A[None, :]
    return A.astype(np.float32, copy=False)


def _collect_sequence_blocks(ds, specs: list[dict]) -> dict[str, np.ndarray]:
    """
    Load per-sequence stacked matrices for a given list of input specs.
    """
    per_seq: dict[str, list[np.ndarray]] = defaultdict(list)
    for spec in specs:
        feat_name = spec["feature"]
        run_id = spec.get("run_id")
        if run_id is None:
            run_id, run_root = _latest_feature_run_root(ds, feat_name)
        else:
            run_root = _feature_run_root(ds, feat_name, run_id)
        pattern = spec.get("pattern", "*.parquet")
        load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            continue
        seq_map = _build_path_sequence_map(ds, feat_name, run_id)
        for fp in files:
            arr = _load_array_from_spec(fp, load_spec)
            if arr is None:
                continue
            safe_seq = seq_map.get(fp.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(fp.stem)
            per_seq[safe_seq].append(arr)

    blocks: dict[str, np.ndarray] = {}
    for safe_seq, mats in per_seq.items():
        mats = [m for m in mats if m.size]
        if not mats:
            continue
        T_min = min(m.shape[0] for m in mats)
        if T_min <= 0:
            continue
        mats = [m[:T_min] for m in mats]
        blocks[safe_seq] = np.hstack(mats)
    return blocks

def _build_path_sequence_map(ds, feature_name: str, run_id: str | None) -> dict[Path, str]:
    """
    Returns {abs_path -> sequence_safe} for a given feature/run_id using the dataset's feature index.
    """
    mapping: dict[Path, str] = {}
    if ds is None or not feature_name or run_id is None:
        return mapping

    try:
        idx_path = _feature_index_path(ds, feature_name)
    except Exception:
        return mapping
    if not idx_path.exists():
        return mapping

    try:
        df = pd.read_csv(idx_path)
    except Exception:
        return mapping

    run_id_str = str(run_id)
    df = df[df["run_id"].astype(str) == run_id_str]
    if df.empty:
        return mapping

    if "sequence_safe" not in df.columns:
        df["sequence_safe"] = df["sequence"].fillna("").apply(
            lambda v: to_safe_name(str(v)) if str(v).strip() else ""
        )

    for _, row in df.iterrows():
        abs_raw = row.get("abs_path")
        if not isinstance(abs_raw, str) or not abs_raw:
            continue
        try:
            abs_path = Path(abs_raw).resolve()
        except Exception:
            abs_path = Path(abs_raw)
        seq_val = (
            row.get("sequence_safe")
            or row.get("sequence")
            or row.get("group_safe")
            or row.get("group")
            or ""
        )
        seq_val = str(seq_val).strip()
        if not seq_val:
            seq_val = to_safe_name(Path(abs_raw).stem)
        mapping[abs_path] = seq_val
    return mapping

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
class PairWavelet:
    """
    'pair-wavelet' — CWT spectrograms on PairPoseDistancePCA outputs.
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

    name = "pair-wavelet"
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
                raise ValueError("[pair-wavelet] None of the requested 'cols' are present in df.")
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
            raise ValueError("[pair-wavelet] Could not auto-detect numeric feature columns.")
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
            raise ValueError("[pair-wavelet] Missing 'perspective' column.")

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
    

@register_feature
class GlobalTSNE: 
    """
    Global t-SNE over multiple prior features discovered from the dataset's feature indexes.
    - inputs: a list of dicts describing which feature outputs to include.
      Each input supports:
        {
          "feature": "pair-wavelet",  # prior feature name
          "run_id":  None,                         # optional, pick latest if None
          "pattern": "wavelet_social_seq=*persp=*.npz",  # files to glob inside the run folder
          "load": { "kind": "npz", "key": "spectrogram", "transpose": True }  # loader spec
        }
    The loaded arrays are treated as (T x D); arrays from all inputs with the same sequence
    are length-aligned (min T) and concatenated horizontally.

    Heavy work happens in fit(); transform() returns a tiny stub DF.
    """
    name: str = "global-tsne"
    version: str = "0.2"
    params: dict

    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            # Multi-input spec (good defaults for your current pipeline):
            inputs=[
                {
                    "feature": "pair-wavelet",
                    "run_id": None,  # pick latest
                    "pattern": "wavelet_social_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
                {
                    "feature": "pair-egocentric-wavelet",
                    "run_id": None,
                    "pattern": "wavelet_ego_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
            ],
            inputset=None,
            random_state=42,
            r_scaler=200_000,          # cap for standardizer fit sample
            total_templates=2000,      # farthest-first target
            pre_quota_per_key=50,      # pre-sample per key
            perplexity=50,
            n_jobs=8,
            partial_k=25,
            partial_iters=100,
            partial_lr=1.0,
            map_chunk=20_000,
        )
        self.params = {**defaults, **(params or {})}
        self._inputs_overridden = bool(params and "inputs" in params)
        self._rng = np.random.default_rng(self.params["random_state"])
        self._scaler: Optional[StandardScaler] = None
        self._embedding: Optional[TSNEEmbedding] = None
        self._artifacts: Dict[str, Any] = {}
        self._ds = None  # set by bind_dataset
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        self._inputs_meta: Dict[str, Any] = {}
        self._resolved_inputs: List[dict] = []

    # dataset hook
    def bind_dataset(self, ds) -> None:
        self._ds = ds

    # ---- Feature protocol ----
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def partial_fit(self, X: pd.DataFrame) -> None: raise NotImplementedError

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("GlobalTSNE requires dataset binding. Ensure Dataset.run_feature calls bind_dataset().")

        # 0) Gather inputs from feature indexes
        inputset_name = self.params.get("inputset")
        explicit_inputs = self.params["inputs"] if (self._inputs_overridden or not inputset_name) else None
        inputs, inputs_meta = _resolve_inputs(
            self._ds, explicit_inputs, inputset_name,
            explicit_override=self._inputs_overridden
        )
        self._resolved_inputs = inputs
        self._inputs_meta = inputs_meta
        # key is safe_seq (derived from dataset index or filename)
        per_key_parts: Dict[str, List[np.ndarray]] = {}

        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(self._ds, feat_name)
            else:
                run_root = _feature_run_root(self._ds, feat_name, run_id)

            pattern = spec.get("pattern", "*.npz")
            loader  = spec.get("load", {"kind": "npz", "key": "spectrogram", "transpose": True})

            files = sorted(run_root.glob(pattern))
            if not files:
                print(f"[global-tsne] WARN: no files for {feat_name} ({run_id}) with pattern {pattern}", file=sys.stderr)
                continue

            seq_map = self._feature_seq_map(feat_name, run_id)

            def parse_key(path: Path) -> str:
                stem = path.stem
                m_seq = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
                if m_seq:
                    safe_seq = m_seq.group(1)
                else:
                    safe_seq = seq_map.get(path.resolve())
                if not safe_seq:
                    safe_seq = to_safe_name(stem)
                return str(safe_seq)

            for pth in files:
                key = parse_key(pth)
                arr = self._load_matrix(pth, loader)
                if arr is None or arr.size == 0:
                    continue
                if key not in per_key_parts:
                    per_key_parts[key] = []
                per_key_parts[key].append(arr)

        if not per_key_parts:
            raise RuntimeError("GlobalTSNE found no usable inputs from the specified prior features.")

        # 1) Concatenate inputs horizontally per key (align on min T)
        features_per_key: Dict[str, np.ndarray] = {}
        n_total = 0
        for key, mats in per_key_parts.items():
            if not mats:
                continue
            T_min = min(m.shape[0] for m in mats)
            mats = [m[:T_min] for m in mats]
            X = np.hstack(mats)  # (T_min, D_total)
            features_per_key[key] = X
            n_total += X.shape[0]

        keys = list(features_per_key.keys())
        if not keys:
            raise RuntimeError("No combined feature frames after alignment.")

        # 2) Global standardizer
        r_scaler = int(self.params["r_scaler"])
        pools = []
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            take = min(X.shape[0], max(1000, int(0.05 * X.shape[0])))
            idx = self._rng.choice(X.shape[0], size=take, replace=False)
            pools.append(X[idx])
        Xsamp = np.vstack(pools)
        if Xsamp.shape[0] > r_scaler:
            idx = self._rng.choice(Xsamp.shape[0], size=r_scaler, replace=False)
            Xsamp = Xsamp[idx]
        scaler = StandardScaler().fit(Xsamp)
        self._scaler = scaler

        # 3) Template selection (farthest-first over pre-sample)
        T_target = int(self.params["total_templates"])
        quota = max(int(self.params["pre_quota_per_key"]), T_target // max(1, len(keys)))
        pre = []
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            take = min(X.shape[0], quota * 3)
            idx = self._rng.choice(X.shape[0], size=take, replace=False)
            pre.append(scaler.transform(X[idx]))
        X_pre = np.vstack(pre)

        sel = [int(self._rng.integers(0, X_pre.shape[0]))]
        d2  = np.sum((X_pre - X_pre[sel[0]])**2, axis=1)
        while len(sel) < min(T_target, X_pre.shape[0]):
            i = int(np.argmax(d2))
            sel.append(i)
            d2 = np.minimum(d2, np.sum((X_pre - X_pre[i])**2, axis=1))
        templates = X_pre[np.array(sel)]

        # 4) Fit openTSNE on templates
        aff = affinity.PerplexityBasedNN(
            templates,
            perplexity=int(self.params["perplexity"]),
            metric="euclidean",
            method="annoy",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        init = initialization.pca(templates, random_state=int(self.params["random_state"]))
        emb = TSNEEmbedding(
            init, aff,
            negative_gradient_method="fft",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        emb.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True, verbose=False)
        emb.optimize(n_iter=750, momentum=0.8, inplace=True, verbose=False)
        self._embedding = emb

        # 5) Map all frames in chunks
        CHUNK = int(self.params["map_chunk"])
        Kp, Pp, It, Lr = int(self.params["partial_k"]), int(self.params["perplexity"]), int(self.params["partial_iters"]), float(self.params["partial_lr"])

        def map_chunk(embedding: TSNEEmbedding, X_chunk_std: np.ndarray) -> np.ndarray:
            part = embedding.prepare_partial(X_chunk_std, initialization="median", k=Kp, perplexity=Pp)
            part = part.optimize(n_iter=It, learning_rate=Lr, exaggeration=2.0,
                                 momentum=0.0, inplace=False, verbose=False)
            return np.asarray(part)

        mapped = {}
        for key in keys:
            X = features_per_key[key]
            if X.shape[0] == 0: continue
            Xs = scaler.transform(X)
            blocks = [map_chunk(emb, Xs[i:i+CHUNK]) for i in range(0, Xs.shape[0], CHUNK)]
            mapped[key] = np.vstack(blocks)

        # hold artifacts for save_model
        self._artifacts["keys"] = keys
        self._artifacts["templates"] = templates
        self._artifacts["template_indices"] = np.array(sel)
        self._artifacts["mapped_coords"] = mapped
        self._artifacts["inputs_meta"] = inputs_meta

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Just a stub so Dataset.run_feature can index a parquet per (group,sequence)
        return pd.DataFrame({"global_tsne_done": [True]})

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # Save templates
        templates = self._artifacts.get("templates")
        if templates is not None:
            np.savez_compressed(run_root / "global_templates_features.npz", templates=templates)

        # Save tsne coords of templates + selection indices
        if self._embedding is not None:
            Y_templates = np.asarray(self._embedding)
            np.savez_compressed(run_root / "global_tsne_templates.npz",
                                Y=Y_templates, sel=self._artifacts.get("template_indices", np.array([], int)))

        # Save embedding + scaler
        joblib.dump(
            {
                "embedding": self._embedding,
                "scaler": self._scaler,
                "params": self.params,
            },
            run_root / "global_opentsne_embedding.joblib"
        )

        # Save per-key coords
        mapped = self._artifacts.get("mapped_coords", {})
        for safe_seq, Y in mapped.items():
            out_name = f"global_tsne_coords_seq={safe_seq}.npz"
            np.savez_compressed(run_root / out_name, Y=Y)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._embedding = bundle.get("embedding", None)
        self._scaler = bundle.get("scaler", None)

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    # ------- loaders -------
    def _load_matrix(self, pth: Path, spec: dict) -> np.ndarray | None:
        """
        Load a 2D matrix from a file according to a 'load' spec:
        - kind: 'npz' or 'parquet'
        - key:  for npz (required), ignored for parquet
        - columns: optional list of columns for parquet
        - drop_columns: optional list of columns to drop before numeric-only selection
        - numeric_only: default True for parquet; selects only numeric dtypes
        - transpose: bool; if True, returns A.T
        """
        kind = str(spec.get("kind", "parquet")).lower()
        transpose = bool(spec.get("transpose", False))

        if kind == "npz":
            key = spec.get("key", None)
            if not key:
                raise ValueError("load.kind='npz' requires 'key' in load spec")
            npz = np.load(pth, allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"Key '{key}' not found in {pth.name}; available: {list(npz.files)}")
            A = np.asarray(npz[key])
            if A.ndim == 1:  # ensure 2D
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            return A.T if transpose else A

        elif kind == "parquet":
            import pandas as pd
            df = pd.read_parquet(pth)

            # Optional: drop explicitly unwanted columns first
            drop_cols = set(spec.get("drop_columns", []))
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")

            cols = spec.get("columns", None)
            if cols is not None:
                # Use exactly these columns
                df_num = df[cols]
            else:
                # Default: keep numeric columns only
                if spec.get("numeric_only", True):
                    df_num = df.select_dtypes(include=["number"])
                else:
                    # Try to coerce everything except the obvious non-numeric
                    df_num = df.apply(pd.to_numeric, errors="coerce")

            A = df_num.to_numpy(dtype=np.float32, copy=False)
            if A.ndim == 1:
                A = A[None, :]
            return A.T if transpose else A

        else:
            raise ValueError(f"Unknown load.kind='{kind}'")
        

@register_feature
class GlobalKMeansClustering:
    """
    Global K-Means that fits **only** on an artifact matrix resolved from the dataset's
    feature index (no frame-stream fitting). This avoids fit/transform shape mismatch.

    params:
      k: int
      random_state: int
      n_init: int|"auto"
      max_iter: int
      artifact: {
        feature: "global-tsne" | "pair-egocentric" | ...   # required
        run_id: str | None                                 # None => latest
        pattern: str                                       # files inside run root
        load: {                                            # loader spec
          kind: "npz"|"parquet",
          key: str               # for npz
          transpose: bool,       # optional
          columns: [str]|None,   # for parquet; if None -> numeric_only
          drop_columns: [str]|None,
          numeric_only: bool,    # default True for parquet
        }
      }

    Optional assign block:
      assign: {
        scaler: {
          feature: <prior feature name>,      # e.g. "global-tsne"
          run_id: str | None,                # None => latest
          pattern: str,                      # e.g. "global_opentsne_embedding.joblib"
          key: str | None,                   # key to extract from joblib, or None for full object
        },
        inputs: [                           # multi-input spec, same as GlobalTSNE.inputs
          {
            feature: <prior feature name>,
            run_id: str | None,
            pattern: str,
            load: {kind, key, transpose, columns, ...}
          },
          ...
        ]
      }
    If assign is specified, after fitting KMeans, clusters are assigned to full-frame features
    using the provided scaler and multi-inputs. Assigned labels per sequence are saved as
    global_kmeans_labels_seq=<safe_seq>.npz in the run folder.

    Notes:
      • The `assign.scaler` entry is optional. If omitted, assignment is performed in the raw feature space.
        In that case, the concatenated input dimensionality must exactly match the dimensionality used to fit K-Means.

    Fit-time artifacts saved in run folder:
      - kmeans.joblib
      - artifact_meta.json (fit provenance + columns + feature_dim)
      - cluster_centers.npy
      - artifact_labels.npz (labels over the artifact matrix, if computed)
      - cluster_sizes.csv (label,count) if labels computed
      - (if assign) global_kmeans_labels_seq=<safe_seq>.npz (per key)
    """

    name: str = "global-kmeans"
    version: str = "0.3"

    def __init__(self, params: dict | None = None):
        self.params = {
            "k": 100,
            "random_state": 42,
            "n_init": "auto",
            "max_iter": 300,
            "artifact": {
                "feature": None,
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            # optional: compute labels on the artifact used for fitting
            "label_artifact_points": True,
            # optional: assign block for assigning clusters to full-frame features
            "assign": None,
        }
        if params:
            self.params.update(params)

        self._ds = None
        self._kmeans = None
        self._fit_dim = None              # int
        self._fit_columns = None          # list[str] when parquet
        self._fit_artifact_info = None    # dict saved to JSON
        self._artifact_labels = None      # np.ndarray[int] (optional)
        self._assign_labels = {}          # dict[safe_seq] = labels (int32)
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}
        assign_config = self.params.get("assign") or {}
        self._assign_inputs_override = "inputs" in assign_config
        self._assign_inputs_meta: Dict[str, Any] = {}

    def _load_one_general(self, path: Path, spec: dict) -> np.ndarray:
        arr = _load_array_from_spec(path, spec)
        if arr is None:
            return np.empty((0, 0), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"_load_one_general: Loaded array is not 2D: {arr.shape}")
        return arr

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key_from_path(self, path: Path, seq_map: dict[Path, str]) -> str:
        """
        Extract safe_seq from file path using naming convention or dataset index fallback.
        """
        stem = path.stem
        m_seq = re.search(r"seq=(.+?)(?:_persp=.*)?$", stem)
        if m_seq:
            safe_seq = m_seq.group(1)
        else:
            safe_seq = seq_map.get(path.resolve())
        if not safe_seq:
            safe_seq = to_safe_name(stem)
        return str(safe_seq)

    def _collect_inputs_per_key(self, inputs: list[dict]) -> dict[str, list[np.ndarray]]:
        """
        For each input spec, glob pattern, load each file as matrix, group by sequence.
        Returns dict[safe_seq] -> list[np.ndarray]
        """
        per_key_parts = {}
        for spec in inputs:
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            resolved_run_id, run_root = self._resolve_feature_run_root(feat_name, run_id)
            pattern = spec.get("pattern", "*.npz")
            load_spec = spec.get("load", {"kind": "npz", "key": "spectrogram", "transpose": True})
            files = sorted(run_root.glob(pattern))
            seq_map = self._feature_seq_map(feat_name, resolved_run_id)
            for pth in files:
                key = self._extract_key_from_path(pth, seq_map)
                arr = self._load_one_general(pth, load_spec)
                if arr is None or arr.size == 0:
                    continue
                if key not in per_key_parts:
                    per_key_parts[key] = []
                per_key_parts[key].append(arr)
        return per_key_parts

    def _load_scaler(self, spec: dict):
        """
        Loads a scaler (e.g. sklearn StandardScaler) from a joblib file.
        spec: {
          feature: str,
          run_id: str|None,
          pattern: str (default: "global_opentsne_embedding.joblib"),
          key: str|None,
        }
        Returns: scaler object, or obj[key] if key is not None.
        """
        feat_name = spec["feature"]
        run_id = spec.get("run_id")
        _, run_root = self._resolve_feature_run_root(feat_name, run_id)
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"_load_scaler: No files matching '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        return obj if key is None else obj[key]

    # Dataset binding
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def bind_dataset(self, ds): self._ds = ds

    # ----------------- Fit helpers -----------------
    def _resolve_feature_run_root(self, feature_name: str, run_id: str | None) -> tuple[str, Path]:
        froot = self._ds.get_root("features") / feature_name
        idx_path = froot / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feature_name}' at {idx_path}")
        idx = pd.read_csv(idx_path)
        if run_id is None:
            # pick latest finished if present; else started_at
            if "finished_at" in idx.columns:
                idx = idx.sort_values("finished_at", ascending=False, na_position="last")
            elif "started_at" in idx.columns:
                idx = idx.sort_values("started_at", ascending=False, na_position="last")
            if idx.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(idx.iloc[0]["run_id"])
        resolved = str(run_id)
        run_root = froot / resolved
        if not run_root.exists():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        return resolved, run_root

    def _load_npz_matrix(self, files: list[Path], key: str, transpose: bool) -> tuple[np.ndarray, dict]:
        mats = []
        for p in files:
            npz = np.load(p, allow_pickle=True)
            if key not in npz.files:
                # skip silently; some *.npz may not contain the requested key
                continue
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            A = A.astype(np.float32, copy=False)
            A = A.T if transpose else A
            mats.append(A)
        if not mats:
            raise FileNotFoundError(f"No NPZ containing key '{key}' among: {[p.name for p in files]}")
        X = np.vstack(mats)
        meta = {"loader_kind": "npz", "key": key, "transpose": bool(transpose)}
        return X, meta

    def _load_parquet_matrix(self, files: list[Path], spec: dict) -> tuple[np.ndarray, dict, list[str]]:
        cols = spec.get("columns")
        drop_cols = set(spec.get("drop_columns", []))
        numeric_only = bool(spec.get("numeric_only", True))

        def load_df(p: Path) -> pd.DataFrame:
            df = pd.read_parquet(p)
            if drop_cols:
                df = df.drop(columns=list(drop_cols), errors="ignore")
            if cols is not None:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return pd.DataFrame()
                df = df[use]
            else:
                df = df.select_dtypes(include=["number"]) if numeric_only else df.apply(pd.to_numeric, errors="coerce")
            return df

        dfs = []
        first_cols = None
        for p in files:
            df = load_df(p)
            if df.empty:
                continue
            if first_cols is None:
                first_cols = df.columns.tolist()
            else:
                # align to first set of columns
                df = df.reindex(columns=first_cols)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No Parquet files with usable numeric columns among: {[p.name for p in files]}")

        D = pd.concat(dfs, ignore_index=True)
        A = D.to_numpy(dtype=np.float32, copy=False)
        meta = {
            "loader_kind": "parquet",
            "columns": first_cols,
            "drop_columns": list(drop_cols) if drop_cols else [],
            "numeric_only": numeric_only,
        }
        return A, meta, first_cols

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.get("artifact") or {}
        feature = art.get("feature")
        if not feature:
            raise ValueError("GlobalKMeansClustering: params['artifact']['feature'] is required.")
        resolved_run_id, run_root = self._resolve_feature_run_root(feature, art.get("run_id"))
        pattern = art.get("pattern", "*.npz")
        loader  = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {run_root}")

        kind = str(loader.get("kind", "npz")).lower()
        self._fit_columns = None
        if kind == "npz":
            key = loader.get("key")
            if not key:
                raise ValueError("artifact.load.kind='npz' requires 'key'")
            X, meta = self._load_npz_matrix(files, key, bool(loader.get("transpose", False)))
        elif kind == "parquet":
            X, meta, cols = self._load_parquet_matrix(files, loader)
            self._fit_columns = cols
        else:
            raise ValueError(f"Unknown artifact load.kind='{kind}'")

        # for provenance
        self._fit_artifact_info = {
            "feature": feature,
            "run_id": resolved_run_id,
            "run_root": str(run_root),
            "pattern": pattern,
            "loader": meta,
        }
        return X

    # ----------------- Fit / Transform / Save -----------------
    def fit(self, _X_iter_unused: Iterable[pd.DataFrame]) -> None:
        # Fit purely from artifact
        X = self._load_artifact_matrix()
        self._fit_dim = int(X.shape[1])

        if X.shape[0] < int(self.params["k"]):
            raise ValueError(f"Not enough samples to fit KMeans: n={X.shape[0]} < k={self.params['k']}")

        self._kmeans = KMeans(
            n_clusters=int(self.params["k"]),
            n_init=self.params.get("n_init", "auto"),
            random_state=self.params.get("random_state", 42),
            max_iter=int(self.params.get("max_iter", 300)),
        ).fit(X)

        # Optional: label the artifact points used for training (nice to have)
        if bool(self.params.get("label_artifact_points", True)):
            self._artifact_labels = self._kmeans.predict(X)

        # Optional: assign clusters to full-frame features using assign block
        assign = self.params.get("assign")
        if assign:
            # Optional scaler: if provided, standardize before prediction; else predict in raw space.
            scaler = None
            if "scaler" in assign and assign["scaler"]:
                scaler = self._load_scaler(assign["scaler"])

            assign_inputset = assign.get("inputset")
            explicit_assign_inputs = assign.get("inputs") if (self._assign_inputs_override or not assign_inputset) else None
            resolved_assign_inputs, assign_inputs_meta = _resolve_inputs(
                self._ds,
                explicit_assign_inputs,
                assign_inputset,
                explicit_override=self._assign_inputs_override,
            )
            self._assign_inputs_meta = assign_inputs_meta

            # Collect per-key inputs (may be empty)
            per_key_parts = self._collect_inputs_per_key(resolved_assign_inputs)
            for key, mats in per_key_parts.items():
                if not mats:
                    continue
                T_min = min(m.shape[0] for m in mats)
                mats_trim = [m[:T_min] for m in mats]
                X_full = np.hstack(mats_trim)
                D_total = X_full.shape[1]

                if scaler is not None:
                    # Validate scaler dimensionality
                    if not hasattr(scaler, "n_features_in_"):
                        raise ValueError("Scaler object must have n_features_in_ (e.g. sklearn StandardScaler)")
                    if scaler.n_features_in_ != D_total:
                        raise ValueError(
                            f"Scaler expects n_features_in_={getattr(scaler, 'n_features_in_', None)}, "
                            f"got {D_total} columns for key={key}"
                        )
                    X_use = scaler.transform(X_full)
                else:
                    # No scaler: require that assign inputs match the KMeans fit dimensionality.
                    if self._fit_dim is None:
                        raise RuntimeError("GlobalKMeansClustering internal error: fit_dim is None before assignment.")
                    if D_total != self._fit_dim:
                        raise ValueError(
                            f"Assign-without-scaler requires feature dim {self._fit_dim}, "
                            f"but got {D_total} for key={key}. Provide a scaler or align inputs."
                        )
                    X_use = X_full

                labels = self._kmeans.predict(X_use)
                self._assign_labels[key] = labels.astype(np.int32)
            if self._fit_artifact_info is None:
                self._fit_artifact_info = {}
            self._fit_artifact_info["assign_inputs_meta"] = assign_inputs_meta

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Assign clusters to an input DataFrame **only if** its columns can be aligned to the
        fitted feature space:
          - If we fitted on parquet with explicit/numeric columns, require those columns.
          - Otherwise, require numeric matrix with matching dimensionality.
        If alignment fails, return an **empty** DataFrame (no error).
        """
        if self._kmeans is None or self._fit_dim is None:
            raise RuntimeError("GlobalKMeansClustering not fitted yet.")

        # Case 1: we know the exact columns from parquet fit
        if self._fit_columns:
            missing = [c for c in self._fit_columns if c not in X.columns]
            if missing:
                return pd.DataFrame(columns=["frame", "cluster"])  # silently skip
            A = X[self._fit_columns].to_numpy(dtype=np.float32, copy=False)
        else:
            # Case 2: generic numeric-only fallback requiring same dimensionality
            num = X.select_dtypes(include=["number"])
            if num.shape[1] != self._fit_dim:
                return pd.DataFrame(columns=["frame", "cluster"])  # silently skip
            A = num.to_numpy(dtype=np.float32, copy=False)

        mask = np.isfinite(A).all(axis=1)
        labels = np.full(A.shape[0], -1, dtype=np.int32)
        if mask.any():
            labels[mask] = self._kmeans.predict(A[mask])

        out = pd.DataFrame({
            "frame": X["frame"].astype(int, errors="ignore") if "frame" in X.columns else np.arange(len(X), dtype=int),
            "cluster": labels,
        })
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # 1) Save the fitted model
        joblib.dump({
            "kmeans": self._kmeans,
            "k": int(self.params["k"]),
            "random_state": self.params.get("random_state", 42),
            "n_init": self.params.get("n_init", "auto"),
            "max_iter": int(self.params.get("max_iter", 300)),
            "fit_dim": int(self._fit_dim or 0),
            "fit_columns": self._fit_columns,
            "artifact_info": self._fit_artifact_info,
            "version": self.version,
        }, path)

        # 2) Save human-usable artifacts
        centers = np.asarray(self._kmeans.cluster_centers_, dtype=np.float32)
        np.save(run_root / "cluster_centers.npy", centers)

        # template/artifact labels + counts (if computed)
        if self._artifact_labels is not None:
            np.savez_compressed(run_root / "artifact_labels.npz", labels=self._artifact_labels)
            # counts for quick inspection
            uniq, cnt = np.unique(self._artifact_labels, return_counts=True)
            pd.DataFrame({"cluster": uniq.astype(int), "count": cnt.astype(int)}) \
              .to_csv(run_root / "cluster_sizes.csv", index=False)

        # 3) Save assigned labels per key if assign block was used
        for safe_seq, labels in self._assign_labels.items():
            fname = f"global_kmeans_labels_seq={safe_seq}.npz"
            np.savez_compressed(run_root / fname, labels=labels)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._kmeans = bundle["kmeans"]
        self._fit_dim = int(bundle.get("fit_dim") or 0)
        self._fit_columns = bundle.get("fit_columns")
        self._fit_artifact_info = bundle.get("artifact_info", {})


@register_feature
class GlobalWardClustering:
    """
    Ward hierarchical clustering on a global feature artifact (e.g. global t-SNE templates).

    Params
    ------
    artifact : dict (required)
        {
          "feature": "global-tsne",            # feature that produced the artifact
          "run_id": None,                      # None => latest finished run
          "pattern": "global_templates_features.npz",
          "load": {"kind": "npz", "key": "templates", "transpose": False}
        }
    method : str = "ward"
        Linkage method (Ward requires Euclidean distances).
    """

    name    = "global-ward"
    version = "0.1"

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "artifact": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            "method": "ward",
        }
        if params:
            # shallow-merge
            self.params.update({k: v for k, v in params.items() if k != "artifact"})
            if "artifact" in params:
                a = dict(self.params["artifact"])
                a.update(params["artifact"])
                # nested load merge
                if "load" in params["artifact"]:
                    ld = dict(a.get("load", {}))
                    ld.update(params["artifact"]["load"])
                    a["load"] = ld
                self.params["artifact"] = a

        self._ds = None               # bound Dataset
        self._Z  = None               # linkage matrix (np.ndarray)
        self._X_shape = None          # (n_samples, n_features)
        self._marker_written = False  # ensure only one parquet marker row gets written
        self._artifact_inputs_meta: Dict[str, Any] = {}

    # ---------- framework API ----------
    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        # Ignore X_iter; we load from the declared artifact to avoid accidental wrong inputs
        X = self._load_artifact_matrix()
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(f"[global-ward] Need a 2D matrix with >=2 samples; got shape={X.shape}")
        method = str(self.params.get("method", "ward")).lower()
        if method != "ward":
            # You could allow other methods, but Ward is the intended one
            raise ValueError(f"[global-ward] Only 'ward' is supported here, got '{method}'.")

        # SciPy linkage expects samples as rows
        self._Z = _sch_linkage(X, method=method)
        self._X_shape = tuple(X.shape)

    def partial_fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        We don't produce per-(group,sequence) outputs. We emit a single 1-row marker
        the first time transform is called so the run is indexed; subsequent calls return
        an empty DF (so nothing else is written).
        """
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        ns, nf = (self._X_shape or (np.nan, np.nan))
        return pd.DataFrame([{
            "linkage_method": str(self.params.get("method", "ward")),
            "n_samples": int(ns) if ns == ns else -1,     # handle NaN
            "n_features": int(nf) if nf == nf else -1,
            "model_file": "model.joblib",
        }])

    def save_model(self, path: Path) -> None:
        """
        Persist the linkage and minimal provenance to model.joblib.
        Also write a human-usable .npz copy (optional).
        """
        if self._Z is None:
            # nothing fitted; skip
            return
        # Ensure parent exists and path is file path (run_feature passes a file path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "linkage_matrix": self._Z,
            "method": str(self.params.get("method", "ward")),
            "n_samples": None if self._X_shape is None else int(self._X_shape[0]),
            "n_features": None if self._X_shape is None else int(self._X_shape[1]),
            "version": self.version,
            "params": self.params,
        }, path)

        # Optional: also store as npz right next to model (convenient for quick numpy loads)
        np.savez_compressed(path.with_suffix(".npz"),
                            linkage_matrix=self._Z,
                            method=str(self.params.get("method", "ward")),
                            n_samples=(None if self._X_shape is None else int(self._X_shape[0])),
                            n_features=(None if self._X_shape is None else int(self._X_shape[1])))

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._Z = bundle.get("linkage_matrix", None)
        self._X_shape = (bundle.get("n_samples", None), bundle.get("n_features", None))

    def _load_artifact_matrix(self) -> np.ndarray:
        """
        Resolve the artifact (feature/run_id), glob the pattern, and load to a single (N,D) matrix.
        """
        if self._ds is None:
            raise RuntimeError("[global-ward] Feature not bound to a Dataset; call via dataset.run_feature(...)")

        art = self.params.get("artifact", {})
        inputset_name = art.get("inputset")
        explicit_inputs = art.get("inputs") if ("inputs" in art) else None
        if inputset_name or explicit_inputs:
            specs, meta = _resolve_inputs(
                self._ds,
                explicit_inputs,
                inputset_name,
                explicit_override=("inputs" in art),
            )
            blocks = _collect_sequence_blocks(self._ds, specs)
            if not blocks:
                raise RuntimeError("[global-ward] Inputset produced no usable matrices.")
            X = np.vstack(list(blocks.values()))
            if X.ndim != 2:
                raise ValueError(f"[global-ward] Loaded array must be 2D; got shape={X.shape}")
            self._artifact_inputs_meta = meta
            return X.astype(np.float64, copy=False)

        feat_name = art.get("feature", None)
        run_id    = art.get("run_id", None)
        pattern   = art.get("pattern", None)
        load_spec = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})

        if not feat_name or not pattern:
            raise ValueError("[global-ward] 'artifact.feature' and 'artifact.pattern' are required in params.")

        idx_path = self._ds.get_root("features") / feat_name / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"[global-ward] No index for feature '{feat_name}'. Expected: {idx_path}")

        df_idx = pd.read_csv(idx_path)
        if run_id is None:
            if "finished_at" in df_idx.columns:
                cand = df_idx[df_idx["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df_idx
                base = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                        ascending=False, kind="stable")
            else:
                base = df_idx.sort_values(by=["started_at"], ascending=False, kind="stable")
            if base.empty:
                raise ValueError(f"[global-ward] No runs found for feature '{feat_name}'.")
            run_id = str(base.iloc[0]["run_id"])

        run_root = self._ds.get_root("features") / feat_name / run_id
        if not run_root.exists():
            raise FileNotFoundError(f"[global-ward] run root not found: {run_root}")

        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[global-ward] No files matching '{pattern}' in {run_root}")

        mats = []
        for fp in files:
            kind = (load_spec.get("kind") or "npz").lower()
            transpose = bool(load_spec.get("transpose", False))
            if kind == "npz":
                key = load_spec.get("key", None)
                if key is None:
                    raise ValueError("[global-ward] 'load.key' required for npz")
                npz = np.load(fp, allow_pickle=True)
                if key not in npz.files:
                    raise KeyError(f"[global-ward] Key '{key}' not found in {fp.name}")
                A = np.asarray(npz[key])
            elif kind == "parquet":
                df = pd.read_parquet(fp)
                cols = load_spec.get("columns", None)
                if cols:
                    A = df[cols].to_numpy(dtype=np.float32)
                else:
                    if load_spec.get("numeric_only", True):
                        A = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                    else:
                        A = df.to_numpy(dtype=np.float32)
            else:
                raise ValueError(f"[global-ward] Unsupported load.kind='{kind}'")

            if A.ndim == 1:
                A = A[None, :]
            if transpose:
                A = A.T
            mats.append(A.astype(np.float64, copy=False))

        if not mats:
            raise RuntimeError(f"[global-ward] No matrices loaded from {run_root} with pattern {pattern}")

        X = mats[0] if len(mats) == 1 else np.vstack(mats)
        if X.ndim != 2:
            raise ValueError(f"[global-ward] Loaded array must be 2D; got {X.shape} from {files[0].name}")
        return X


@register_feature
class WardAssignClustering:
    """
    Assign Ward clusters (cut at n_clusters) to full-frame feature streams by stacking
    multiple prior features (e.g., pair-wavelet social + ego) and reusing the scaler
    from GlobalTSNE.

    Params
    ------
    ward_model : dict
        { "feature": "global-ward__from__global-tsne",
          "run_id": None,
          "pattern": "model.joblib" }
    artifact : dict
        Same structure as GlobalWardClustering.artifact (used to reload the template matrix).
    scaler : optional dict
        Same contract as GlobalKMeans.assign.scaler (joblib w/ StandardScaler).
    inputs : list[dict]
        Feature specs to concatenate per sequence. The FIRST entry must match the feature
        passed to dataset.run_feature(... input_feature=...) so that its DataFrames stream
        through transform(); remaining entries are loaded from disk per sequence.
    n_clusters : int
        Desired Ward cut.
    recalc : bool
        If True, force recomputation even if outputs already exist (pass overwrite=True to
        dataset.run_feature when rerunning). Defaults to False.
    """

    name = "ward-assign"
    version = "0.1"

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "ward_model": {
                "feature": "global-ward__from__global-tsne",
                "run_id": None,
                "pattern": "model.joblib",
            },
            "artifact": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates", "transpose": False},
            },
            "scaler": None,
            "inputs": [],
            "inputset": None,
            "n_clusters": 20,
            "recalc": False,
        }
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    d = dict(self.params[k])
                    d.update(v)
                    self.params[k] = d
                else:
                    self.params[k] = v

        ward_feat_name = self.params["ward_model"].get("feature", "global-ward")
        self.storage_feature_name = f"ward-assign__from__{ward_feat_name}"
        self.storage_use_input_suffix = False
        self.skip_existing_outputs = True

        self._ds = None
        self._Z = None
        self._templates = None
        self._cluster_ids = None
        self._assign_nn = None
        self._scaler = None
        self._inputs = []
        self._input_run_ids: list[str] = []
        self._input_seq_paths: list[dict[str, Path]] = []
        self._sequence_label_store: dict[str, np.ndarray] = {}
        self._written_sequences: set[str] = set()
        self._inputs_overridden = bool(params and "inputs" in params)
        self._inputs_meta: Dict[str, Any] = {}

    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def partial_fit(self, X): raise NotImplementedError

    # ---------- helpers ----------
    def _resolve_feature_run_root(self, feature_name: str, run_id: Optional[str]) -> tuple[str, Path]:
        froot = self._ds.get_root("features") / feature_name
        idx_path = froot / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feature_name}' at {idx_path}")
        df = pd.read_csv(idx_path)
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                      ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            if df.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(df.iloc[0]["run_id"])
        resolved = str(run_id)
        run_root = froot / resolved
        if not run_root.exists():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        return resolved, run_root

    def _load_artifact_matrix(self) -> np.ndarray:
        art = self.params.get("artifact", {})
        feature = art.get("feature")
        if not feature:
            raise ValueError("[ward-assign] artifact.feature required.")
        run_id, run_root = self._resolve_feature_run_root(feature, art.get("run_id"))
        pattern = art.get("pattern", "*.npz")
        loader = art.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No files matching '{pattern}' in {run_root}")
        kind = str(loader.get("kind", "npz")).lower()
        if kind == "npz":
            key = loader.get("key")
            npz = np.load(files[0], allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"[ward-assign] Key '{key}' missing in {files[0].name}")
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            return A.astype(np.float32, copy=False)
        elif kind == "parquet":
            df = pd.read_parquet(files[0])
            if loader.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            return df.to_numpy(dtype=np.float32, copy=False)
        else:
            raise ValueError(f"[ward-assign] Unsupported artifact load.kind='{kind}'")

    def _load_scaler(self, spec: Optional[dict]):
        if not spec:
            return None
        feat_name = spec.get("feature")
        if not feat_name:
            raise ValueError("[ward-assign] scaler.feature required.")
        _, run_root = self._resolve_feature_run_root(feat_name, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] Could not locate scaler '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        return obj if key is None else obj[key]

    def _load_one_general(self, path: Path, spec: dict) -> np.ndarray:
        kind = str(spec.get("kind", "parquet")).lower()
        if kind == "npz":
            key = spec.get("key")
            npz = np.load(path, allow_pickle=True)
            if key not in npz.files:
                raise KeyError(f"[ward-assign] Key '{key}' missing in {path.name}")
            A = np.asarray(npz[key])
            if A.ndim == 1:
                A = A[None, :]
            return A.astype(np.float32, copy=False)
        elif kind == "parquet":
            df = pd.read_parquet(path)
            drop_cols = spec.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            cols = spec.get("columns")
            if cols:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return np.empty((0, 0), dtype=np.float32)
                df = df[use]
            elif spec.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if spec.get("transpose"):
                arr = arr.T
            return arr
        else:
            raise ValueError(f"[ward-assign] Unknown load.kind='{kind}'")

    def _df_to_matrix(self, df: pd.DataFrame, spec: dict) -> np.ndarray:
        drop_cols = spec.get("drop_columns")
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        cols = spec.get("columns")
        if cols:
            df_num = df[cols]
        else:
            if spec.get("numeric_only", True):
                df_num = df.select_dtypes(include=[np.number])
            else:
                df_num = df.apply(pd.to_numeric, errors="coerce")
        arr = df_num.to_numpy(dtype=np.float32, copy=False)
        if spec.get("transpose"):
            arr = arr.T
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _infer_sequence(self, df: pd.DataFrame) -> str:
        if "sequence" in df.columns and not df["sequence"].empty:
            val = str(df["sequence"].iloc[0])
            if val:
                return val
        if "group" in df.columns and "sequence" in df.columns:
            return str(df["sequence"].iloc[0])
        raise ValueError("[ward-assign] Input DataFrame missing 'sequence' column.")

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("[ward-assign] Dataset not bound. Use dataset.run_feature(...).")

        inputs = self.params.get("inputs") or []
        inputset_name = self.params.get("inputset")
        explicit_inputs = inputs if (self._inputs_overridden or not inputset_name) else None
        self._inputs, self._inputs_meta = _resolve_inputs(
            self._ds,
            explicit_inputs,
            inputset_name,
            explicit_override=self._inputs_overridden,
        )

        self._written_sequences.clear()

        # Load Ward linkage
        ward_spec = self.params["ward_model"]
        _, ward_root = self._resolve_feature_run_root(ward_spec["feature"], ward_spec.get("run_id"))
        pattern = ward_spec.get("pattern", "model.joblib")
        files = sorted(ward_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"[ward-assign] No Ward model '{pattern}' in {ward_root}")
        bundle = joblib.load(files[0])
        self._Z = bundle.get("linkage_matrix")
        if self._Z is None:
            raise ValueError("[ward-assign] Ward model missing linkage_matrix.")

        # Reload artifact matrix (templates) to derive centroids
        self._templates = self._load_artifact_matrix()
        n_clusters = int(self.params.get("n_clusters", 20))
        labels_templates = fcluster(self._Z, n_clusters, criterion="maxclust")
        uniq = np.unique(labels_templates)
        centroids = []
        for cid in uniq:
            mask = labels_templates == cid
            if not mask.any():
                continue
            centroids.append(self._templates[mask].mean(axis=0))
        centroids = np.vstack(centroids)
        self._cluster_ids = uniq.astype(int)
        self._assign_nn = NearestNeighbors(n_neighbors=1).fit(centroids)

        # optional scaler
        self._scaler = self._load_scaler(self.params.get("scaler"))

        # Prebuild sequence -> path maps for inputs (for loading additional specs)
        self._input_run_ids = []
        self._input_seq_paths = []
        for spec in self._inputs:
            run_id, _ = self._resolve_feature_run_root(spec["feature"], spec.get("run_id"))
            seq_map = _build_path_sequence_map(self._ds, spec["feature"], run_id)
            inv = {}
            for path, seq in seq_map.items():
                inv[str(seq)] = path
            self._input_run_ids.append(run_id)
            self._input_seq_paths.append(inv)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._inputs:
            return pd.DataFrame(index=[])
        seq = self._infer_sequence(df)
        safe_seq = to_safe_name(seq)
        if safe_seq in self._written_sequences:
            return pd.DataFrame(index=[])
        mats = []

        for idx, spec in enumerate(self._inputs):
            if idx == 0:
                mats.append(self._df_to_matrix(df, spec.get("load", {})))
            else:
                seq_map = self._input_seq_paths[idx]
                path = seq_map.get(safe_seq)
                if not path:
                    print(f"[ward-assign] WARN: missing feature '{spec['feature']}' for sequence {seq}", file=sys.stderr)
                    return pd.DataFrame(index=[])
                mats.append(self._load_one_general(Path(path), spec.get("load", {})))

        if not mats:
            return pd.DataFrame(index=[])

        lengths = [m.shape[0] for m in mats if m.size]
        if not lengths:
            return pd.DataFrame(index=[])
        T_min = min(lengths)
        mats = [m[:T_min] for m in mats]

        X_full = np.hstack(mats)
        if self._scaler is not None:
            if not hasattr(self._scaler, "transform"):
                raise ValueError("[ward-assign] scaler object missing transform().")
            X_use = self._scaler.transform(X_full)
        else:
            X_use = X_full

        if self._assign_nn is None or self._cluster_ids is None:
            raise RuntimeError("[ward-assign] Model not fitted; call dataset.run_feature first.")

        idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
        labels = self._cluster_ids[idxs.ravel()]

        if "frame" in df.columns:
            frame_vals = df["frame"].to_numpy()[:T_min]
        else:
            frame_vals = np.arange(T_min, dtype=int)

        out = pd.DataFrame({
            "frame": frame_vals.astype(int, copy=False),
            "cluster": labels.astype(int, copy=False),
            "sequence": seq,
        })
        self._sequence_label_store[safe_seq] = labels.astype(int, copy=False)
        self._written_sequences.add(safe_seq)
        return out

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        joblib.dump({
            "params": self.params,
            "n_clusters": int(self.params.get("n_clusters", 20)),
        }, path)
        for safe_seq, labels in self._sequence_label_store.items():
            np.savez_compressed(run_root / f"global_ward_labels_seq={safe_seq}.npz", labels=labels)


#### VISUALIZATION
def plot_tsne_withcolors(ax,tsne_result,quantity,title,corrskip=1,plotskip=1,colortype='scalar',qmin=0.001,qmax=0.999,alphaval=0.3,s=4,coloroffset=0,cmapname='cool',setxylimquantile=False):
    colordata = quantity.copy()
    if len(colordata)>len(tsne_result):
        colordata = colordata[::corrskip]
    if len(colordata.shape)>1:
        colordata = colordata[:,0]
    if colortype=='scalar':
        cmap=plt.get_cmap(cmapname)  # or 'cool'
        q0,q1 = np.quantile(colordata,[qmin,qmax])
        colordata = colordata-q0
        colordata = colordata/(q1-q0)
        colordata[colordata<0] = 0
        colordata[colordata>1] = 1
        colordata *= 0.99
        colors = cmap(colordata)
    else:
#         cmap=plt.get_cmap('Set1')
        colors = snscolors[colordata.astype(int)+coloroffset]
    tp = tsne_result
    # [ax.scatter([-100],[-100],alpha=1,s=10,color=cmap(i*0.99/np.max(groupvalues)),label='group '+str(i)) for i in np.arange(max(groupvalues)+1)]  # legend hack
    scatterplot = ax.scatter(tp[::plotskip,0],tp[::plotskip,1],s=s,alpha=alphaval,color=colors[::plotskip],rasterized=True)
    if setxylimquantile:
        ax.set_xlim(np.quantile(tp[:,0],[qmin,qmax]))
        ax.set_ylim(np.quantile(tp[:,1],[qmin,qmax]))
    else:
        ax.set_xlim(np.quantile(tp[:,0],[0,1]))
        ax.set_ylim(np.quantile(tp[:,1],[0,1]))
    ax.set_title(title,fontsize=16)       
    return scatterplot, colordata    

@register_feature
class VizGlobalColored:
    """
    Generic wrapper to visualize any global embedding (t-SNE, PCA, UMAP, etc.)
    colored by arbitrary labels.

    Params (defaults target the global t-SNE + k-means workflow):
      coords: {feature, run_id, pattern, load:{kind:"npz", key:"Y"}}
      labels: optional spec matching coords (same structure as coords)
      coord_key_regex: regex applied to coord filenames (default extracts the sequence name)
      label_key_regex: regex for label filenames (defaults to coord regex)
      label_missing_value: sentinel for missing labels (default -1 -> gray)
      plot_max: max points to scatter
      palette: seaborn palette name or list
      title: plot title
      point_size / point_alpha: scatter style tweaks
    Output:
      - PNG in run_root plus a single marker parquet row for indexing.
    """

    name = "viz-global-colored"
    version = "0.1"

    def __init__(self, params=None):
        defaults = {
            "coords": {
                "feature": "global-tsne",
                "run_id": None,
                "pattern": "global_tsne_coords_seq=*.npz",
                "load": {"kind": "npz", "key": "Y", "transpose": False},
            },
            "labels": None,
            "coord_key_regex": r"seq=(.+?)(?:_persp=.*)?$",
            "label_key_regex": INHERIT_REGEX,
            "label_missing_value": -1,
            "plot_max": 300_000,
            "palette": "tab20",
            "title": "Global embedding colored scatter",
            "point_size": 2.0,
            "point_alpha": 0.35,
        }
        self.params = dict(defaults)
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    merged = dict(self.params[k])
                    merged.update(v)
                    self.params[k] = merged
                else:
                    self.params[k] = v
        self._ds = None
        self._figs = []
        self._marker_written = False
        self._summary = {}
        self._seq_path_cache: Dict[Tuple[str, str], Dict[Path, str]] = {}

    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self): return True
    def supports_partial_fit(self): return False
    def partial_fit(self, X): raise NotImplementedError
    def finalize_fit(self): pass

    def _load_artifacts_glob(self, spec):
        ds = self._ds
        idx = ds.get_root("features") / spec["feature"] / "index.csv"
        df = pd.read_csv(idx)
        run_id = spec.get("run_id")
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(by=["finished_at" if len(cand) else "started_at"], ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            run_id = str(df.iloc[0]["run_id"])
        run_root = ds.get_root("features") / spec["feature"] / run_id
        files = sorted(run_root.glob(spec["pattern"]))
        return run_id, run_root, files

    def _load_one(self, path, load_spec):
        kind = load_spec.get("kind", "npz").lower()
        if kind == "npz":
            npz = np.load(path, allow_pickle=True)
            A = np.asarray(npz[load_spec["key"]])
            if load_spec.get("transpose"):
                A = A.T
            return A
        if kind == "parquet":
            df = pd.read_parquet(path)
            drop_cols = load_spec.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            cols = load_spec.get("columns")
            if cols:
                use = [c for c in cols if c in df.columns]
                if not use:
                    return np.empty((0, len(cols)), dtype=np.float32)
                df = df[use]
            elif load_spec.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                # coerce everything numeric-style to avoid object columns
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if load_spec.get("transpose"):
                arr = arr.T
            return arr
        if kind == "joblib":
            obj = joblib.load(path)
            key = load_spec.get("key")
            return obj if key is None else obj[key]
        raise ValueError(f"Unsupported load.kind='{kind}'")

    def _feature_seq_map(self, feature_name: str, run_id: str | None) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key(self, path: Path, regex: Optional[str], seq_map: dict[Path, str]):
        stem = path.stem
        m = re.search(regex, stem) if regex else None
        if not m:
            safe_seq = seq_map.get(path.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(stem)
            return str(safe_seq)
        if m.lastindex is None:
            return m.group(0)
        if m.lastindex == 1:
            return m.group(1)
        return tuple(m.groups())

    def _prepare_color_map(self, labels):
        unique_vals = list(pd.unique(pd.Series(labels, dtype="object")))
        missing = self.params.get("label_missing_value", -1)
        palette = sns.color_palette(self.params.get("palette", "tab20"),
                                    max(1, len([u for u in unique_vals if u != missing])))
        color_map = {}
        idx = 0
        for val in unique_vals:
            if val == missing:
                continue
            color_map[val] = palette[idx % len(palette)]
            idx += 1
        if missing in unique_vals:
            color_map[missing] = (0.7, 0.7, 0.7)
        return color_map

    def fit(self, X_iter):
        coord_spec = self.params["coords"]
        coord_run_id, _, coord_files = self._load_artifacts_glob(coord_spec)
        if not coord_files:
            raise FileNotFoundError("[viz-global-colored] No coordinate files found.")

        key_regex = self.params.get("coord_key_regex")
        coord_seq_map = self._feature_seq_map(coord_spec["feature"], coord_run_id)
        Y_list, key_list, n_list = [], [], []
        for f in coord_files:
            key = self._extract_key(f, key_regex, coord_seq_map)
            arr = self._load_one(f, coord_spec["load"])
            if arr.ndim != 2:
                arr = np.atleast_2d(arr)
            Y_list.append(arr)
            key_list.append(key)
            n_list.append(arr.shape[0])
        if not Y_list:
            raise RuntimeError("No coordinate arrays loaded.")
        Y_all = np.vstack(Y_list)

        labels_spec = self.params.get("labels")
        missing_value = self.params.get("label_missing_value", -1)
        if labels_spec:
            label_run_id, _, label_files = self._load_artifacts_glob(labels_spec)
            label_regex_param = self.params.get("label_key_regex", INHERIT_REGEX)
            if label_regex_param == INHERIT_REGEX:
                label_regex = key_regex
            else:
                label_regex = label_regex_param
            label_seq_map = self._feature_seq_map(labels_spec["feature"], label_run_id)
            lab_map = {}
            for lf in label_files:
                lab_key = self._extract_key(lf, label_regex, label_seq_map)
                lab_map[lab_key] = np.asarray(self._load_one(lf, labels_spec["load"]))
            L_parts = []
            for key, n in zip(key_list, n_list):
                arr = lab_map.get(key)
                if arr is None:
                    print(f"[viz-global-colored] WARN: missing labels for key={key}; assigning {missing_value}",
                          file=sys.stderr)
                    arr = np.full(n, missing_value, dtype=int)
                else:
                    arr = np.asarray(arr).ravel()
                    if arr.shape[0] != n:
                        nmin = min(arr.shape[0], n)
                        trimmed = arr[:nmin]
                        if nmin < n:
                            pad = np.full(n - nmin, missing_value, dtype=trimmed.dtype)
                            arr = np.concatenate([trimmed, pad])
                        else:
                            arr = trimmed
                L_parts.append(arr)
            L_all = np.concatenate(L_parts)
        else:
            L_parts = [np.zeros(n, dtype=int) for n in n_list]
            L_all = np.concatenate(L_parts)

        plot_max = int(self.params.get("plot_max", 300_000))
        if Y_all.shape[0] > plot_max:
            rng = np.random.default_rng(42)
            sel = rng.choice(Y_all.shape[0], size=plot_max, replace=False)
            Y_plot = Y_all[sel]
            L_plot = L_all[sel]
        else:
            Y_plot = Y_all
            L_plot = L_all

        color_map = self._prepare_color_map(L_all if labels_spec else L_plot)
        colors = np.array([color_map.get(val, (0.7, 0.7, 0.7)) for val in L_plot])

        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.scatter(
            Y_plot[:, 0],
            Y_plot[:, 1],
            c=colors,
            s=float(self.params.get("point_size", 2.0)),
            alpha=float(self.params.get("point_alpha", 0.35)),
            linewidths=0,
        )
        ax.set_title(self.params.get("title", "Global embedding colored scatter"))
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")

        unique_vals = pd.unique(pd.Series(L_plot, dtype="object"))
        if len(unique_vals) <= 15:
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                           markerfacecolor=color_map.get(val, (0.7, 0.7, 0.7)),
                           markeredgecolor="none", label=str(val))
                for val in unique_vals
            ]
            if handles:
                ax.legend(handles=handles, title="label", loc="best", fontsize=8)

        out_name = "global_colored.png"
        self._figs = [(out_name, fig)]
        self._summary = {
            "points": int(Y_all.shape[0]),
            "plotted": int(Y_plot.shape[0]),
            "labels_present": bool(labels_spec),
        }

    def transform(self, X):
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        return pd.DataFrame([{
            "outputs": ",".join(fname for fname, _ in self._figs),
            "labels_present": bool(self.params.get("labels")),
        }])

    def save_model(self, path: Path):
        run_root = path.parent
        for fname, fig in self._figs:
            fig.savefig(run_root / fname, dpi=150, bbox_inches="tight")
        joblib.dump(
            {"params": self.params, "summary": self._summary,
             "files": [fname for fname, _ in self._figs]},
            run_root / "viz.joblib"
        )

VizGlobalTSNEColored = VizGlobalColored
