# dataset_task1.py
from __future__ import annotations
from pathlib import Path
import os, re, numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from time import perf_counter
import concurrent.futures as cf

from util import to_raw_seq, load_task1_labels_from_many, _load_concat_wavelet, temporal_stack_features

PAIR_PAT = re.compile(r"wavelet_social_seq=(.*)_persp=(\d+)\.npz")

import numpy as np
import pandas as pd
from pathlib import Path
import json



def load_calms21(path):
    """
    Load a single CalMS21 file: either .npy (dict) produced by the converter,
    or the original .json (same nested structure but with lists).
    Returns a nested dict: group -> seq_id -> dict(...)
    """
    p = Path(path)
    if p.suffix == ".npy":
        data = np.load(p, allow_pickle=True).item()
    elif p.suffix == ".json":
        with open(p, "r") as f:
            data = json.load(f)
        # leave lists; we’ll np.array() per-sequence below
    else:
        raise ValueError("Path must be a .npy or .json file")
    return data

def angle_from_two_points(neck_xy, tail_xy):
    # heading from tail -> neck, angle w.r.t +x (radians)
    v = neck_xy - tail_xy
    return np.arctan2(v[:,1], v[:,0])

def angle_from_pca(XY): 
    """
    PCA-based heading (fallback). XY: (T, 7, 2) landmarks for one animal.
    We use the first principal component per frame; sign is arbitrary.
    """
    T = XY.shape[0]
    ang = np.zeros(T, dtype=float)
    for t in range(T):
        pts = XY[t]  # (7,2)
        mu = pts.mean(axis=0)
        c = pts - mu
        cov = c.T @ c
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]  # 2-vector
        ang[t] = np.arctan2(v[1], v[0])
    return ang

def to_trex_df(one_seq_dict, groupname, seq_id):
    """
    Convert a single sequence dict to TRex-like long DataFrame (rows = frames x animals).
    """
    # ---- CONFIG: set these if you know the landmark mapping for CalMS21 ----
    # If you know which landmark indices define heading (tail->neck), set them:
    NECK_IDX = None  # e.g., 3
    TAIL_IDX = None  # e.g., 5
    # Pick features: either 'features' present or 'keypoints'
    use_features = ("features" in one_seq_dict)
    if use_features:
        # not used in TRex columns; just keep for later if you wish
        features = np.asarray(one_seq_dict["features"])  # (T, 60)
    keypoints = np.asarray(one_seq_dict["keypoints"])    # (T, 2, 2, 7)
    scores    = np.asarray(one_seq_dict["scores"])       # (T, 2, 7)
    ann       = np.asarray(one_seq_dict["annotations"]) if "annotations" in one_seq_dict else None
    meta      = one_seq_dict.get("metadata", {})
    fps       = float(meta.get("fps", meta.get("frame_rate", 30.0)))

    T = keypoints.shape[0]
    n_anim = keypoints.shape[1]
    n_lm   = keypoints.shape[3]
    assert n_anim == 2, f"Expected 2 animals, found {n_anim}"
    assert n_lm == 7,   f"Expected 7 keypoints, found {n_lm}"

    rows = []
    for a in range(n_anim):
        # Extract XY for this animal: (T, 7, 2)
        X = keypoints[:, a, 0, :]  # (T, 7)
        Y = keypoints[:, a, 1, :]  # (T, 7)
        XY = np.stack([X, Y], axis=-1)  # (T, 7, 2)

        # Centroid over landmarks
        cx = X.mean(axis=1)  # (T,)
        cy = Y.mean(axis=1)

        # Vel/acc (finite diff)
        VX = np.gradient(cx) * fps
        VY = np.gradient(cy) * fps
        SPEED = np.hypot(VX, VY)
        AX = np.gradient(VX) * fps
        AY = np.gradient(VY) * fps

        # Angle
        if (NECK_IDX is not None) and (TAIL_IDX is not None):
            neck = XY[:, NECK_IDX, :]  # (T,2)
            tail = XY[:, TAIL_IDX, :]
            ANGLE = angle_from_two_points(neck, tail)
        else:
            ANGLE = angle_from_pca(XY)

        # Build a per-frame DataFrame
        data = {
            "frame": np.arange(T, dtype=int),
            "time":  np.arange(T, dtype=float) / fps,
            "id":    np.full(T, a, dtype=int),
            "X#wcentroid": cx,
            "Y#wcentroid": cy,
            "VX": VX, "VY": VY,
            "SPEED": SPEED, "AX": AX, "AY": AY,
            "ANGLE": ANGLE,
            "group": np.full(T, groupname),
            "sequence": np.full(T, seq_id),
        }

        # Pose columns
        for k in range(n_lm):
            data[f"poseX{k}"] = X[:, k]
            data[f"poseY{k}"] = Y[:, k]

        # Optional: label per frame if present
        if ann is not None:
            # Task 1 typically: (T,) ints; Task 2/3 could be multi-annotator/binary stacks.
            lbl = ann
            if lbl.ndim > 1:
                # try to reduce sensibly (majority or first column)
                lbl = ann[:, 0]
            data["label"] = lbl.astype(int)

        rows.append(pd.DataFrame(data))

    out = pd.concat(rows, ignore_index=True)
    # Add placeholders for TRex fields you don’t have:
    out["missing"] = False
    out["visual_identification_p"] = 1.0
    out["timestamp"] = out["time"]
    # Optional placeholders commonly in your schema:
    for col in ["X","Y","SPEED#pcentroid","SPEED#wcentroid","midline_x","midline_y",
                "midline_length","midline_segment_length","normalized_midline",
                "ANGULAR_V#centroid","ANGULAR_A#centroid","BORDER_DISTANCE#pcentroid",
                "MIDLINE_OFFSET","num_pixels","detection_p"]:
        if col not in out.columns:
            out[col] = np.nan

    return out

def calms21_to_trex_df(path):
    """
    Load a CalMS21 .npy/.json and return a single concatenated TRex-like DataFrame
    for all groups and sequences inside.
    """
    nested = load_calms21(path)
    dfs = []
    for groupname, group in nested.items():
        for seq_id, seq in group.items():
            # ensure arrays
            seq = {
                k: (np.array(v) if isinstance(v, list) else v)
                for k,v in seq.items()
            }
            dfs.append(to_trex_df(seq, groupname, seq_id))
    return pd.concat(dfs, ignore_index=True)

def discover_pairs_strict(base: Path):
    """Return sorted list of pair-wavelet files (strict modern naming)."""
    files = sorted((base).glob("wavelet_social_seq=*persp=*.npz"))
    if not files:
        raise FileNotFoundError("No pair wavelet files found.")
    return files

def _one_block(pair_fpath: Path, base: Path, calms_npys, half, skip, add_pool, pool_stats, sigma_stack, sigma_pool):
    m = PAIR_PAT.match(pair_fpath.name)
    if not m: return None
    safe_seq, persp = m.group(1), int(m.group(2))
    raw_seq = to_raw_seq(safe_seq)

    X = load_concat_wavelet(base, safe_seq, persp)              # (T, Fpair+Fego)
    gt = load_task1_labels_from_many(calms_npys, raw_seq)       # (T,)
    n  = min(X.shape[0], gt.shape[0])
    if n <= 0: return None

    Xw, centers = temporal_stack_features(
        X[:n], half_window=half, skip=skip, add_pool=add_pool, pool_stats=pool_stats,
        apply_to=("scale","pool"), sigma_stack=sigma_stack, sigma_pool=sigma_pool
    )
    if Xw.shape[0] == 0: return None
    yb = gt[centers].astype(int, copy=False)
    gb = np.full(len(centers), raw_seq, dtype=object)
    return (Xw, yb, gb)

def _sizes_and_offsets(blocks):
    sizes = np.fromiter((b.shape[0] for b in blocks), count=len(blocks), dtype=np.int64)
    if sizes.size == 0: return sizes, sizes
    offs = np.empty_like(sizes); offs[0] = 0
    if sizes.size > 1: np.cumsum(sizes[:-1], out=offs[1:])
    return sizes, offs

def _parallel_copy(dstX, dsty, blocksX, blocksY, offsets, max_workers=None):
    def _copy_one(off, Xb, yb):
        n = Xb.shape[0]
        dstX[off:off+n] = Xb
        dsty[off:off+n] = yb
    if len(blocksX) == 0: return
    with cf.ThreadPoolExecutor(max_workers=max_workers or (os.cpu_count() or 4)) as ex:
        futures = [ex.submit(_copy_one, int(off), Xb, yb) for Xb, yb, off in zip(blocksX, blocksY, offsets)]
        for fut in cf.as_completed(futures): fut.result()

def _stack_split(blocks, split_map):
    """Split by sequence prefix ('task1/train' vs 'task1/test') using raw_seq."""
    trX, trY, trG = [], [], []
    teX, teY, teG = [], [], []
    for Xb, yb, gb in blocks:
        raw_seq = gb[0]
        # map raw_seq -> 'train' or 'test'
        part = split_map.get(raw_seq, None)
        if part == "train":
            trX.append(np.ascontiguousarray(Xb, dtype=np.float32)); trY.append(yb.astype(np.int32)); trG.append(gb)
        elif part == "test":
            teX.append(np.ascontiguousarray(Xb, dtype=np.float32)); teY.append(yb.astype(np.int32)); teG.append(gb)
        # else: silently drop (shouldn't happen if labels came from those files)
    if not trX or not teX:
        raise RuntimeError(f"Empty split: train={len(trX)} blocks, test={len(teX)} blocks.")
    return trX, trY, trG, teX, teY, teG

def _build_split_map_from_npys(calms_train_npy, calms_test_npy):
    """
    Read both npy dicts and build a map raw_seq -> 'train'/'test'
    so we *exactly* match the challenge split.
    """
    split = {}
    dtr = np.load(calms_train_npy, allow_pickle=True).item()
    for _, seqs in dtr.items():
        for k in seqs.keys(): split[k] = "train"
    dte = np.load(calms_test_npy, allow_pickle=True).item()
    for _, seqs in dte.items():
        for k in seqs.keys(): split[k] = "test"
    return split

def make_task1_dataset(
    base: Path,
    calms_train_npy: Path,
    calms_test_npy: Path,
    *,
    half=16, skip=2, add_pool=True, pool_stats=("mean","std"),
    sigma_stack=None, sigma_pool=None,
    n_jobs=8
):
    """
    Returns:
      X_train, y_train, X_test, y_test, scaler (StandardScaler), groups_train, groups_test
    """
    split_map = _build_split_map_from_npys(calms_train_npy, calms_test_npy)

    pair_files = discover_pairs_strict(base)
    blocks = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_block)(pf, base, [calms_train_npy, calms_test_npy],
                            half, skip, add_pool, pool_stats, sigma_stack, sigma_pool)
        for pf in pair_files
    )
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        raise RuntimeError("No usable data produced after loading/stacking.")

    # split by official split_map (not a random split)
    trX, trY, trG, teX, teY, teG = _stack_split(blocks, split_map)

    # stack arrays
    D = trX[0].shape[1]
    tr_sizes, tr_offs = _sizes_and_offsets(trX)
    te_sizes, te_offs = _sizes_and_offsets(teX)
    X_train_raw = np.empty((int(tr_sizes.sum()), D), dtype=np.float32)
    y_train     = np.empty((int(tr_sizes.sum()),), dtype=np.int32)
    X_test_raw  = np.empty((int(te_sizes.sum()), D), dtype=np.float32)
    y_test      = np.empty((int(te_sizes.sum()),), dtype=np.int32)

    t0 = perf_counter()
    _parallel_copy(X_train_raw, y_train, trX, trY, tr_offs)
    _parallel_copy(X_test_raw,  y_test,  teX, teY, te_offs)
    t1 = perf_counter()
    print(f"[stack] train={X_train_raw.shape}, test={X_test_raw.shape} in {t1 - t0:.2f}s")

    groups_train = np.concatenate(trG)
    groups_test  = np.concatenate(teG)

    # global scaler on train
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32, copy=False)
    X_test  = scaler.transform(X_test_raw ).astype(np.float32, copy=False)

    return X_train, y_train, X_test, y_test, scaler, groups_train, groups_test


# -------------------------------
# 0) Load the official split dicts
# -------------------------------
def _load_task1_split(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    d = np.load(path, allow_pickle=True).item()
    if not isinstance(d, dict):
        raise ValueError(f"Unexpected format in {path} (expected dict)")
    return d

# Flatten dict-of-groups → {raw_seq_key: record}
def _flatten_split_dict(dsplit: dict) -> dict:
    flat = {}
    for group, seqs in dsplit.items():
        if not isinstance(seqs, dict):
            continue
        flat.update(seqs)
    return flat

# Extract the label vector (T,) for a raw_seq from one split dict
def _labels_from_split(flat: dict, raw_seq: str) -> np.ndarray | None:
    rec = flat.get(raw_seq)
    if rec is None:
        return None
    # The common key for Task 1 annotations is "annotations"
    anns = rec.get("annotations", None)
    if anns is None:
        return None
    arr = np.asarray(anns)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        arr = arr[:, 0]  # first annotator if stacked
    return arr.astype(int, copy=False)

def _process_one_block_task1(BASE, task1_train_dict, task1_test_dict, train_keys, test_keys, safe_seq: str, persp: int,
                             HALF, SKIP, ADD_POOL, POOL_STATS, SIGMA_STACK, SIGMA_POOL,
                             use_temporal_stack=True):
    raw_seq = to_raw_seq(safe_seq)

    # Decide split based on official lists
    split = None
    if raw_seq in train_keys and raw_seq in test_keys:
        # Very unlikely; prefer train and warn
        print(f"[WARN] {raw_seq} present in both splits; using TRAIN.")
        split = "train"
    elif raw_seq in train_keys:
        split = "train"
    elif raw_seq in test_keys:
        split = "test"
    else:
        # Not part of Task 1 splits -> skip
        return None

    # Load features (pair + ego) strict
    X = _load_concat_wavelet(BASE, safe_seq, persp)  # (T, Fpair+Fego)

    # Labels from the corresponding split only
    if split == "train":
        y_full = _labels_from_split(task1_train_dict, raw_seq)
    else:
        y_full = _labels_from_split(task1_test_dict, raw_seq)

    if y_full is None:
        # No labels for this sequence in the declared split -> skip
        return None

    n = min(X.shape[0], y_full.shape[0])
    if n <= 0:
        return None

    if use_temporal_stack:
        Xw, centers = temporal_stack_features(
            X[:n].astype(np.float32, copy=False),
            half_window=HALF,
            skip=SKIP,
            add_pool=ADD_POOL,
            pool_stats=POOL_STATS,
            apply_to=("scale", "pool"),
            sigma_stack=SIGMA_STACK,
            sigma_pool=SIGMA_POOL,
        )
        if Xw.shape[0] == 0:
            return None
        yb = y_full[centers].astype(int, copy=False)
        gb = np.full(len(centers), raw_seq, dtype=object)
        return split, Xw.astype(np.float32, copy=False), yb, gb
    else:
        Xw = sliding_stats_block(X[:n], WIN).astype(np.float32, copy=False)
        yb = y_full[:n].astype(int, copy=False)
        gb = np.full(n, raw_seq, dtype=object)
        return split, Xw, yb, gb

def _chunk_bounds(n_rows, chunk):
    k = max(1, (n_rows + chunk - 1) // chunk)
    for i in range(k):
        a = i * chunk
        b = min(n_rows, (i + 1) * chunk)
        if b > a:
            yield a, b

def _sum_and_sumsq(X, a, b):
    Xi = X[a:b].astype(np.float64, copy=False)
    s  = Xi.sum(axis=0)
    ss = np.square(Xi, dtype=np.float64).sum(axis=0)
    n  = Xi.shape[0]
    return n, s, ss