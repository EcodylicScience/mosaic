# util.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from pathlib import Path
import numpy as np
import re
import pandas as pd 
import numpy as np
from pathlib import Path
    



############################################################################################################################################


#### TO DO:  SOME OF THE THESE, ESPCIALLY THE FIRST 3, ARE REDUNDANT, CAN BE CONDENSED
def load_social_wavelet_frames(BASE, safe_seq: str, persp: int) -> np.ndarray:
    p = BASE / f"wavelet_social_seq={safe_seq}_persp={persp}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p.name}")
    arr = np.load(p)["spectrogram"].T   # (n_frames, F_social)
    return arr

# Helpers to find wavelets
def find_social_wavelet(base, safe_seq, persp):
    cands = [
        base / f"wavelet_social_seq={safe_seq}_persp={persp}.npz",
        base / f"wavelet_spectrogram_seq={safe_seq}_persp={persp}.npz",
    ]
    for p in cands:
        if p.exists():
            return p
    return None

def find_ego_wavelet(base, safe_seq, persp):
    p = base / f"wavelet_ego_seq={safe_seq}_persp={persp}.npz"
    return p if p.exists() else None

def load_combined_features(base, safe_seq, persp):
    """Return (X_combined: (T_min, D_total)), or None if neither exists."""
    ps = find_social_wavelet(base, safe_seq, persp)
    pe = find_ego_wavelet(base, safe_seq, persp)

    if ps is None and pe is None:
        return None

    parts = []
    lengths = []

    if ps is not None:
        Xs = np.load(ps)["spectrogram"].T  # (T, D_s)
        parts.append(Xs); lengths.append(Xs.shape[0])
    if pe is not None:
        Xe = np.load(pe)["spectrogram"].T  # (T, D_e)
        parts.append(Xe); lengths.append(Xe.shape[0])

    if not parts:
        return None

    T_min = min(lengths)
    parts = [p[:T_min] for p in parts]
    return np.hstack(parts)  # (T_min, D_total)

    


# ---------- naming ----------
def to_raw_seq(safe_seq: str) -> str:
    # Task 3: task3-<behavior>-<split>-<rest>
    m = re.match(r"^(task3)-([^-]+)-(train|test)-(.*)$", safe_seq)
    if m:
        task, behavior, split, rest = m.groups()
        return f"{task}/{behavior}/{split}/{rest}"

    # Task 1/2: task1-<split>-<rest> or task2-<split>-<rest>
    m = re.match(r"^(task[12])-(train|test)-(.*)$", safe_seq)
    if m:
        task, split, rest = m.groups()
        return f"{task}/{split}/{rest}"

    # Fallback: replace first two '-' with '/' (keeps remaining '-' in tail)
    parts = safe_seq.split("-", 2)
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}/{parts[2]}"
    # Last resort: single replacement
    return safe_seq.replace("-", "/", 1)

# ---------- strict file finders (modern names only) ----------
def find_pair_wavelet(in_dir: Path, safe_seq: str, persp: int) -> Path:
    p = in_dir / f"wavelet_social_seq={safe_seq}_persp={persp}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing pair wavelet: {p.name}")
    return p

def find_ego_wavelet(in_dir: Path, safe_seq: str, persp: int) -> Path:
    p = in_dir / f"wavelet_ego_seq={safe_seq}_persp={persp}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing ego wavelet: {p.name}")
    return p

# ---------- GT loading ----------
def load_task1_labels_from_many(npy_paths, raw_seq_key: str) -> np.ndarray:
    """
    Search a list of .npy dict files for Task 1 GT for this sequence.
    Returns (T,) int array.
    """
    for p in npy_paths:
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True).item()
        for _, seqs in d.items():
            if raw_seq_key in seqs:
                seq_dict = seqs[raw_seq_key]
                ann = np.asarray(seq_dict.get("annotations", None))
                if ann is None or ann.size == 0:
                    continue
                if ann.ndim > 1:
                    ann = ann[:, 0]
                return ann.astype(int)
    raise KeyError(f"GT for '{raw_seq_key}' not found in: {[str(p) for p in npy_paths]}")

def fps_from_npz(npz, default=1.0) -> float:
    return float(npz["fps"]) if "fps" in npz.files else float(default)



### DATA PROCESSING / DECOMPOSITION
def clean_one_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(ORDER_COL).copy()
    g = g.set_index(ORDER_COL)

    g[POSE_COLS] = g[POSE_COLS].interpolate(
        method="linear", limit=LINEAR_INTERP_LIMIT, limit_direction="both"
    )
    g[POSE_COLS] = g[POSE_COLS].ffill(limit=EDGE_FILL_LIMIT)
    g[POSE_COLS] = g[POSE_COLS].bfill(limit=EDGE_FILL_LIMIT)

    miss_frac = g[POSE_COLS].isna().mean(axis=1)
    g = g.loc[miss_frac <= MAX_MISSING_FRACTION].copy()

    if g[POSE_COLS].isna().any().any():
        med = g[POSE_COLS].median()
        g[POSE_COLS] = g[POSE_COLS].fillna(med)

    g = g.reset_index()
    return g


def pose_to_points(row_vals: np.ndarray):
    # row_vals shape: (2N,) = [poseX0..N-1, poseY0..N-1]
    xs = row_vals[:POSE_N]
    ys = row_vals[POSE_N:]
    return np.stack([xs, ys], axis=1)  # (N,2)

def intra_lower_tri_dists(pts: np.ndarray):
    dif = pts[tri_i] - pts[tri_j]
    return np.sqrt((dif**2).sum(axis=1))  # (n_intra,)

def inter_all_dists(ptsA: np.ndarray, ptsB: np.ndarray):
    # (N,2) vs (N,2) -> (N*N,)
    # broadcast pairwise
    dif = ptsA[:, None, :] - ptsB[None, :, :]  # (N, N, 2)
    d = np.sqrt((dif**2).sum(axis=2))          # (N, N)
    return d.ravel()                            # (N*N,)

def build_pair_features(rowA: np.ndarray, rowB: np.ndarray):
    """
    rowA/rowB: (2N,) arrays of [poseX..., poseY...]
    Return concatenated [intraA, intraB, interAB] (as configured).
    """
    ptsA = pose_to_points(rowA)
    ptsB = pose_to_points(rowB)
    parts = []
    if INCLUDE_INTRA_A:
        parts.append(intra_lower_tri_dists(ptsA))
    if INCLUDE_INTRA_B:
        parts.append(intra_lower_tri_dists(ptsB))
    if INCLUDE_INTER:
        parts.append(inter_all_dists(ptsA, ptsB))
    return np.concatenate(parts, axis=0)  # (feat_len,)

# -----------------------------
# 5) Streaming generator over all sequences (batched)
# -----------------------------
def dyad_row_generator(df_all: pd.DataFrame, batch_size: int):
    """
    Yields (X_batch, seqs, frames, perspectives)
    - X_batch: (B, feat_len)
    - seqs: list of sequence names (len=B)
    - frames: np.array of frame indices (len=B)
    - perspectives: np.array of 0 for (A,B) and 1 for (B,A) if duplicated, else 0
    """
    X_buf, seq_buf, frame_buf, persp_buf = [], [], [], []
    for seq, (idA, idB) in pairs_per_seq.items():
        gA = df_all[(df_all[SEQ_COL] == seq) & (df_all[ID_COL] == idA)][[ORDER_COL] + POSE_COLS]
        gB = df_all[(df_all[SEQ_COL] == seq) & (df_all[ID_COL] == idB)][[ORDER_COL] + POSE_COLS]

        # inner join on frame/time to synchronize
        gA = gA.rename(columns={ORDER_COL: "frame"})
        gB = gB.rename(columns={ORDER_COL: "frame"})
        j = gA.merge(gB, on="frame", suffixes=("_A", "_B"))  # (n_sync, ...)

        if j.empty:
            continue

        # Build features for synced rows
        XA = j[[c + "_A" for c in POSE_COLS]].to_numpy()  # (n_sync, 2N)
        XB = j[[c + "_B" for c in POSE_COLS]].to_numpy()  # (n_sync, 2N)
        frames = j["frame"].to_numpy()

        for i in range(XA.shape[0]):
            # perspective A->B
            X_feat = build_pair_features(XA[i], XB[i])
            X_buf.append(X_feat)
            seq_buf.append(seq)
            frame_buf.append(int(frames[i]))
            persp_buf.append(0)

            if DUPLICATE_PERSPECTIVE:
                # perspective B->A (flip order)
                X_feat_flip = build_pair_features(XB[i], XA[i])
                X_buf.append(X_feat_flip)
                seq_buf.append(seq)
                frame_buf.append(int(frames[i]))
                persp_buf.append(1)

            if len(X_buf) >= batch_size:
                yield np.vstack(X_buf), seq_buf, np.array(frame_buf), np.array(persp_buf, dtype=int)
                X_buf, seq_buf, frame_buf, persp_buf = [], [], [], []

    # flush
    if X_buf:
        yield np.vstack(X_buf), seq_buf, np.array(frame_buf), np.array(persp_buf, dtype=int)


### EGOCENTRIC CALCULATION
def _smooth_1d(x, win):
    if win <= 1:
        return x
    # simple centered moving average with reflection pad
    pad = win // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    ker = np.ones(win, dtype=float) / win
    return np.convolve(xp, ker, mode="valid")

def _safe_unit(vx, vy, eps=1e-8):
    n = np.sqrt(vx*vx + vy*vy) + eps
    return vx / n, vy / n

def _angle(vx, vy):
    return np.arctan2(vy, vx)

def _unwrap_diff(theta, fps):
    # angular velocity (rad/s) with unwrap
    d = np.gradient(np.unwrap(theta), edge_order=1)
    return d * fps

def _center_from_points(xs, ys, mode):
    if isinstance(mode, int):
        return xs[:, mode], ys[:, mode]
    # mean of all points
    return xs.mean(axis=1), ys.mean(axis=1)

def _build_ego_block_for_joined(j, fps):
    """
    j: joined dataframe (per sequence) with columns:
       frame, poseXk_A, poseYk_A, poseXk_B, poseYk_B  (k=0..POSE_N-1)
    Returns:
      frames (T,), feats_AtoB (F,T), feats_BtoA (F,T), names (list of F names)
    """
    # --- extract matrices (T,N) for A and B
    T = len(j)
    N = POSE_N
    XA = j[[f"poseX{k}_A" for k in range(N)]].to_numpy()
    YA = j[[f"poseY{k}_A" for k in range(N)]].to_numpy()
    XB = j[[f"poseX{k}_B" for k in range(N)]].to_numpy()
    YB = j[[f"poseY{k}_B" for k in range(N)]].to_numpy()
    frames = j["frame"].to_numpy().astype(int)

    # optional smoothing before diffs (per keypoint)
    if SMOOTH_WIN and SMOOTH_WIN > 1:
        XA = np.vstack([_smooth_1d(XA[:, k], SMOOTH_WIN) for k in range(N)]).T
        YA = np.vstack([_smooth_1d(YA[:, k], SMOOTH_WIN) for k in range(N)]).T
        XB = np.vstack([_smooth_1d(XB[:, k], SMOOTH_WIN) for k in range(N)]).T
        YB = np.vstack([_smooth_1d(YB[:, k], SMOOTH_WIN) for k in range(N)]).T

    # centers
    cxA, cyA = _center_from_points(XA, YA, CENTER_MODE)
    cxB, cyB = _center_from_points(XB, YB, CENTER_MODE)

    # headings (neck - tail)
    hxA, hyA = XA[:, NECK_IDX] - XA[:, TAIL_BASE_IDX], YA[:, NECK_IDX] - YA[:, TAIL_BASE_IDX]
    hxB, hyB = XB[:, NECK_IDX] - XB[:, TAIL_BASE_IDX], YB[:, NECK_IDX] - YB[:, TAIL_BASE_IDX]
    uhxA, uhyA = _safe_unit(hxA, hyA)
    uhxB, uhyB = _safe_unit(hxB, hyB)

    # orthogonal (left-hand) unit
    uoxA, uoyA = -uhyA, uhxA
    uoxB, uoyB = -uhyB, uhxB

    # velocities (center point), m/s in pixel units -> per second with fps
    vAx = np.gradient(cxA) * fps
    vAy = np.gradient(cyA) * fps
    vBx = np.gradient(cxB) * fps
    vBy = np.gradient(cyB) * fps
    speedA = np.sqrt(vAx*vAx + vAy*vAy)
    speedB = np.sqrt(vBx*vBx + vBy*vBy)

    # heading angles and angular speeds
    thA = _angle(uhxA, uhyA)
    thB = _angle(uhxB, uhyB)
    angspeedA = _unwrap_diff(thA, fps)
    angspeedB = _unwrap_diff(thB, fps)

    # ego projections of velocity
    vA_para = vAx * uhxA + vAy * uhyA
    vA_perp = vAx * uoxA + vAy * uoyA
    vB_para = vBx * uhxB + vBy * uhyB
    vB_perp = vBx * uoxB + vBy * uoyB

    # displacement A->B in world
    dx = cxB - cxA
    dy = cyB - cyA
    distAB = np.sqrt(dx*dx + dy*dy)

    # A-centric ego coords of B
    dxA = dx * uhxA + dy * uhyA      # forward (+) if in front of A
    dyA = dx * uoxA + dy * uoyA      # left (+) if to the left of A

    # B-centric ego coords of A (for perspective flip)
    dxB = (-dx) * uhxB + (-dy) * uhyB
    dyB = (-dx) * uoxB + (-dy) * uoyB

    # relative heading (B wrt A): Δθ = θB - θA, encode as sin/cos
    dth = np.unwrap(thB) - np.unwrap(thA)
    rel_cos = np.cos(dth)
    rel_sin = np.sin(dth)

    # Feature stacks (A→B perspective first)
    names = [
        "A_speed", "A_v_para", "A_v_perp", "A_ang_speed",
        "A_heading_cos", "A_heading_sin",
        "AB_dist", "AB_dx_egoA", "AB_dy_egoA",
        "rel_heading_cos", "rel_heading_sin",
        # also include B kinematics (useful even in A→B view)
        "B_speed", "B_v_para", "B_v_perp", "B_ang_speed",
    ]
    AtoB = np.vstack([
        speedA, vA_para, vA_perp, angspeedA,
        np.cos(thA), np.sin(thA),
        distAB, dxA, dyA,
        rel_cos, rel_sin,
        speedB, vB_para, vB_perp, angspeedB,
    ]).astype(np.float32)

    # For B→A, swap roles and reuse the same ordering semantics (B is 'self')
    BtoA = np.vstack([
        speedB, vB_para, vB_perp, angspeedB,
        np.cos(thB), np.sin(thB),
        distAB, dxB, dyB,
        np.cos(-dth), np.sin(-dth),   # rel heading of A wrt B = -(θB-θA)
        speedA, vA_para, vA_perp, angspeedA,
    ]).astype(np.float32)

    return frames, AtoB, BtoA, names


# wavelet and ego
def load_gt_labels_from_many(npy_paths, raw_seq_key: str) -> np.ndarray:
    for p in npy_paths:
        if not p.exists():
            continue
        dd = np.load(p, allow_pickle=True).item()
        if not isinstance(dd, dict):
            continue
        # Flatten one level: {group: {seq_key: rec}}
        for _, seqs in dd.items():
            if not isinstance(seqs, dict):
                continue
            rec = seqs.get(raw_seq_key)

def load_pair_wavelet(base, safe_seq, persp):
    for name in (f"wavelet_social_seq={safe_seq}_persp={persp}.npz",
                 f"wavelet_spectrogram_seq={safe_seq}_persp={persp}.npz"):
        p = base / name
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"Missing pair wavelet for seq={safe_seq}, persp={persp}")

def load_ego_wavelet(base, safe_seq, persp):
    # This assumes you saved egocentric wavelets as wavelet_ego_seq=..._persp=...
    p = base / f"wavelet_ego_seq={safe_seq}_persp={persp}.npz"
    if p.exists():
        return np.load(p)
    # If you used a different filename, adjust here.
    raise FileNotFoundError(f"Missing egocentric wavelet for seq={safe_seq}, persp={persp}")

def concat_pair_ego_features(pair_npz, ego_npz):
    X_pair = np.asarray(pair_npz["spectrogram"]).T  # (T, F_pair)
    X_ego  = np.asarray(ego_npz["spectrogram"]).T   # (T, F_ego)
    # align length
    n = min(X_pair.shape[0], X_ego.shape[0])
    return np.hstack([X_pair[:n], X_ego[:n]]), n

def fps_from_npz(npz, default=1.0):
    return float(npz["fps"]) if "fps" in npz.files else default


# features.py


# ---------- array shape helpers ----------
def ensure_TF(arr: np.ndarray) -> np.ndarray:
    """Ensure (T, F). If (F, T) (rows << cols), transpose."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    r, c = arr.shape
    return arr.T if r <= c else arr

# ---------- low-level loaders ----------
def load_concat_wavelet(in_dir: Path, safe_seq: str, persp: int) -> np.ndarray:
    """
    Strict: load pair + ego wavelets, align by min(T), return float32 (T, F_pair+F_ego).
    """
    pp = find_pair_wavelet(in_dir, safe_seq, persp)
    pe = find_ego_wavelet(in_dir,  safe_seq, persp)
    with np.load(pp) as P: Xp = ensure_TF(np.asarray(P["spectrogram"]))
    with np.load(pe) as E: Xe = ensure_TF(np.asarray(E["spectrogram"]))
    T = min(Xp.shape[0], Xe.shape[0])
    return np.hstack([Xp[:T], Xe[:T]]).astype(np.float32, copy=False)

def load_concat_wavelet_with_meta(in_dir: Path, safe_seq: str, persp: int):
    pp = find_pair_wavelet(in_dir, safe_seq, persp)
    pe = find_ego_wavelet(in_dir,  safe_seq, persp)
    P = np.load(pp); E = np.load(pe)
    Xp = ensure_TF(np.asarray(P["spectrogram"]))
    Xe = ensure_TF(np.asarray(E["spectrogram"]))
    T  = min(Xp.shape[0], Xe.shape[0])
    Xc = np.hstack([Xp[:T], Xe[:T]]).astype(np.float32, copy=False)
    fps = fps_from_npz(P, 1.0)
    meta = {"pair_path": str(pp), "ego_path": str(pe), "pair_shape": Xp.shape,
            "ego_shape": Xe.shape, "concat_shape": Xc.shape, "used_T": T, "fps": float(fps)}
    return Xc, float(fps), meta

# ---------- temporal stacking (with optional Gaussian weights) ----------
def _gaussian_weights(offsets: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        w = np.ones_like(offsets, dtype=np.float32)
    else:
        w = np.exp(-0.5 * (offsets.astype(np.float32) / float(sigma))**2)
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(offsets, dtype=np.float32) / len(offsets)

def _weighted_mean(W: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (W * w[:, None]).sum(axis=0)

def _weighted_std(W: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = _weighted_mean(W, w)
    var = (w[:, None] * (W - mu)**2).sum(axis=0)
    return np.sqrt(np.maximum(var, 0.0))

def temporal_stack_features(
    X: np.ndarray,
    half_window: int = 16,
    skip: int = 1,
    add_pool: bool = True,
    pool_stats=("mean","std"),
    apply_to=("scale","pool"),
    sigma_stack: float | None = None,
    sigma_pool:  float | None = None,
):
    """
    Returns:
      X_stack: (N_kept, F * num_steps [+ pooled])
      centers: (N_kept,) center frame indices
    """
    if apply_to is None: apply_to = tuple()
    if isinstance(apply_to, str): apply_to = (apply_to,)

    T, F = X.shape
    offsets  = np.arange(-half_window, half_window + 1, skip, dtype=int)
    half_eff = offsets[-1]
    start = max(half_window, half_eff)
    end   = min(T - half_window, T - half_eff)
    if end <= start:
        return np.empty((0, F), dtype=X.dtype), np.array([], dtype=int)

    centers = np.arange(start, end, dtype=int)
    num_steps = len(offsets)

    w_stack = _gaussian_weights(offsets, sigma_stack) if ("scale" in apply_to) else None

    X_stack = np.empty((len(centers), F * num_steps), dtype=X.dtype)
    for i, t in enumerate(centers):
        W = X[t + offsets, :]  # (num_steps, F)
        if w_stack is not None: W = W * w_stack[:, None]
        X_stack[i] = W.reshape(-1)

    if add_pool:
        dense_offsets = np.arange(-half_window, half_window + 1, dtype=int)
        w_pool = _gaussian_weights(dense_offsets, sigma_pool) if ("pool" in apply_to) else None
        pooled = []
        for t in centers:
            W = X[t - half_window : t + half_window + 1, :]
            parts = []
            for stat in pool_stats:
                if stat == "mean":
                    parts.append(_weighted_mean(W, w_pool) if w_pool is not None else W.mean(axis=0))
                elif stat == "std":
                    parts.append(_weighted_std(W, w_pool) if w_pool is not None else W.std(axis=0))
                elif stat == "max":
                    parts.append(W.max(axis=0))
                elif stat == "min":
                    parts.append(W.min(axis=0))
            if parts: pooled.append(np.concatenate(parts, axis=0))
        if pooled:
            pooled = np.vstack(pooled)
            X_stack = np.hstack([X_stack, pooled])

    return X_stack.astype(np.float32, copy=False), centers

def expected_temporal_dim(F_base: int, half: int, skip: int, pool_stats: tuple[str, ...]) -> int:
    n_steps = len(range(-half, half+1, skip))
    n_pool  = len(pool_stats) if pool_stats else 0
    return F_base * (n_steps + n_pool)


# ===========================
# Load & concatenate pair + egocentric WAVELET features (STRICT modern naming)
# ==========================

# ------------------------------------------------------------------------------------
# Assumptions:
#   • Pair wavelet:  wavelet_social_seq=<SAFE>_persp=<p>.npz  (required)
#   • Ego  wavelet:  wavelet_ego_seq=<SAFE>_persp=<p>.npz     (required)
#   • Key in both:   "spectrogram"  (2D array; may be (F,T) or (T,F))
#   • We ignore:     social_ego_*, ego_kinematics_*, legacy names
# ------------------------------------------------------------------------------------

# ---------- Minimal helpers ----------
def _ensure_TF(arr: np.ndarray) -> np.ndarray:
    """
    Ensure (T, F). If likely (F, T) (rows << cols), transpose.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    r, c = arr.shape
    return arr.T if r <= c else arr

def _fps_from_npz(npz, default=1.0):
    return float(npz["fps"]) if "fps" in npz.files else default

def _find_pair_wavelet_file(in_dir: Path, safe_seq: str, persp: int) -> Path:
    p = in_dir / f"wavelet_social_seq={safe_seq}_persp={persp}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing pair wavelet: {p.name}")
    return p

def _find_ego_wavelet_file(in_dir: Path, safe_seq: str, persp: int) -> Path:
    p = in_dir / f"wavelet_ego_seq={safe_seq}_persp={persp}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing ego wavelet: {p.name}")
    return p

def _load_concat_wavelet(in_dir: Path, safe_seq: str, persp: int) -> np.ndarray:
    """
    Strict: load pair + ego wavelet spectrograms and hstack (truncate to min T).
    Returns float32 (T, F_pair + F_ego).
    """
    pair_path = _find_pair_wavelet_file(in_dir, safe_seq, persp)
    ego_path  = _find_ego_wavelet_file(in_dir,  safe_seq, persp)

    with np.load(pair_path) as P:
        Xp = _ensure_TF(np.asarray(P["spectrogram"]))
    with np.load(ego_path) as E:
        Xe = _ensure_TF(np.asarray(E["spectrogram"]))

    T = min(Xp.shape[0], Xe.shape[0])
    X = np.hstack([Xp[:T], Xe[:T]]).astype(np.float32, copy=False)
    return X

def load_concat_wavelet_features(in_dir: Path, safe_seq: str, persp: int):
    """
    Returns (X_concat, fps, meta)
      X_concat: (T, F_pair + F_ego) float32
      fps:      float (from pair file if present, else 1.0)
      meta:     dict with paths/shapes/used_T
    """
    pair_path = _find_pair_wavelet_file(in_dir, safe_seq, persp)
    ego_path  = _find_ego_wavelet_file(in_dir,  safe_seq, persp)

    P = np.load(pair_path)
    E = np.load(ego_path)

    Xp = _ensure_TF(np.asarray(P["spectrogram"]))
    Xe = _ensure_TF(np.asarray(E["spectrogram"]))

    T  = min(Xp.shape[0], Xe.shape[0])
    Xc = np.hstack([Xp[:T], Xe[:T]]).astype(np.float32, copy=False)
    fps = _fps_from_npz(P, 1.0)
    meta = {
        "source": "pair+ego_strict",
        "pair_path": str(pair_path),
        "ego_path": str(ego_path),
        "pair_shape": Xp.shape,
        "ego_shape": Xe.shape,
        "concat_shape": Xc.shape,
        "used_T": T,
        "fps": float(fps),
    }
    return Xc, float(fps), meta

# ---------- tiny utility: expected stacked dimension (for sanity prints if you want) ----------
def _expected_temporal_dim(F_base: int, half: int, skip: int, pool_stats: tuple[str, ...]) -> int:
    n_steps = len(range(-half, half+1, skip))
    n_pool  = len(pool_stats) if pool_stats else 0
    return F_base * (n_steps + n_pool)

# ===== Your data-loader (parallel) — strict filenames only =====

# ---------------------------
# Temporal stacking with optional 1D Gaussian kernel
# ---------------------------

def _gaussian_weights(offsets: np.ndarray, sigma: float) -> np.ndarray:
    """
    offsets: 1D array of integer frame offsets relative to the center.
    sigma:   std dev in *frames*.
    Returns normalized weights summing to 1.
    """
    if sigma is None or sigma <= 0:
        # uniform (flat) weights
        w = np.ones_like(offsets, dtype=np.float32)
    else:
        w = np.exp(-0.5 * (offsets.astype(np.float32) / float(sigma))**2)
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(offsets, dtype=np.float32) / len(offsets)

def _weighted_mean(W: np.ndarray, w: np.ndarray) -> np.ndarray:
    # W: (Twin, F); w: (Twin,)
    # returns (F,)
    return (W * w[:, None]).sum(axis=0)

def _weighted_std(W: np.ndarray, w: np.ndarray) -> np.ndarray:
    # numerically stable weighted std
    mu = _weighted_mean(W, w)
    var = (w[:, None] * (W - mu)**2).sum(axis=0)  # weights already normalized to sum 1
    return np.sqrt(np.maximum(var, 0.0))

def temporal_stack_features(
    X: np.ndarray,
    half_window: int = 16,
    skip: int = 1,
    add_pool: bool = True,
    pool_stats=("mean", "std"),
    # --- Gaussian kernel controls ---
    apply_to=("scale", "pool"),     # any of {"scale", "pool"}; use () or None for none
    sigma_stack: float = None,      # Gaussian sigma (frames) for the stacked (skip-sampled) offsets
    sigma_pool: float = None        # Gaussian sigma (frames) for the dense pooling window
):
    """
    X: (T, F) per-frame features (e.g., wavelet over PCA for social pairs)
    half_window: number of frames before/after the center
    skip: take every 'skip' frame within the *stacked* window
    add_pool: append pooled features over the full dense window
    pool_stats: which pooled stats to append ("mean","std","max","min")
    apply_to: tuple indicating where to apply Gaussian weights:
              - "scale": multiply stacked time slices by Gaussian weights before flattening
              - "pool" : use Gaussian *weighted* pooling for mean/std (max/min remain unweighted)
    sigma_stack: std dev (in frames) for the stacked offsets weighting
    sigma_pool:  std dev (in frames) for the dense pooling weighting

    Returns:
      X_stack: (N_kept,  F*(num_steps)  [+ pooled dims])
      keep_idx: (N_kept,) center indices in the original time axis
    """
    if apply_to is None:
        apply_to = tuple()
    if isinstance(apply_to, str):
        apply_to = (apply_to,)

    T, F = X.shape

    # --- Offsets for the *stacked* (skip-sampled) window ---
    offsets = np.arange(-half_window, half_window + 1, skip, dtype=int)   # e.g., [-50,-45,...,0,...,45,50]
    half_eff = offsets[-1]                                                # effective half-span considering skip

    # centers we can keep without crossing sequence boundaries
    start = max(half_window, half_eff)
    end   = min(T - half_window, T - half_eff)
    if end <= start:
        return np.empty((0, F), dtype=X.dtype), np.array([], dtype=int)

    centers = np.arange(start, end, dtype=int)           # center indices (N_kept,)
    num_steps = len(offsets)

    # --- Gaussian weights for the *stacked* offsets (for "scale") ---
    if "scale" in apply_to:
        w_stack = _gaussian_weights(offsets, sigma_stack)   # shape (num_steps,)
    else:
        w_stack = np.ones(num_steps, dtype=np.float32)

    # --- Build stacked matrix, optionally Gaussian-scaled over time ---
    X_stack = np.empty((len(centers), F * num_steps), dtype=X.dtype)
    for i, t in enumerate(centers):
        window = X[t + offsets, :]  # (num_steps, F)
        if "scale" in apply_to:
            window = (window * w_stack[:, None])            # apply temporal weights to each slice
        X_stack[i] = window.reshape(-1)

    # --- Optional pooled features over the *dense* window: [-half_window, +half_window] with step 1 ---
    if add_pool:
        dense_offsets = np.arange(-half_window, half_window + 1, dtype=int)  # every frame in the window
        if "pool" in apply_to:
            w_pool = _gaussian_weights(dense_offsets, sigma_pool)             # (2*half_window+1,)
        else:
            w_pool = None  # means "flat" pooling

        pooled = []
        for t in centers:
            W = X[t - half_window : t + half_window + 1, :]  # (Twin_dense, F)
            stats_parts = []
            for stat in pool_stats:
                if stat == "mean":
                    if w_pool is None:
                        stats_parts.append(W.mean(axis=0))
                    else:
                        stats_parts.append(_weighted_mean(W, w_pool))
                elif stat == "std":
                    if w_pool is None:
                        stats_parts.append(W.std(axis=0))
                    else:
                        stats_parts.append(_weighted_std(W, w_pool))
                elif stat == "max":
                    stats_parts.append(W.max(axis=0))   # max/min are not naturally "weighted"
                elif stat == "min":
                    stats_parts.append(W.min(axis=0))
            if stats_parts:
                stats_vec = np.concatenate(stats_parts, axis=0)
                pooled.append(stats_vec)

        if pooled:  # if we actually added any stats
            pooled = np.vstack(pooled)  # (N_kept, F * len(pool_stats))
            X_stack = np.hstack([X_stack, pooled])

    return X_stack, centers
def align_labels(y, centers):
    """Pick the center-frame label for each stacked sample."""
    return y[centers]


def to_safe_seq(seq_str: str) -> str:
    """Match your saved filenames: replace '/' with '-'."""
    return seq_str.replace("/", "-")

def to_raw_seq(safe_seq: str) -> str:
    """Recover CalMS21 key from safe filename: replace '-' back to '/'."""
    return safe_seq.replace("-", "/")

def load_all_gt(npy_path: Path) -> dict:
    """
    Returns dict: sequence_key -> (T,) int GT array.
    CalMS21 npy is a dict of groups -> dict of sequences -> dict with 'annotations'.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    out = {}
    for group, seqs in data.items():
        for seq_key, seq_dict in seqs.items():
            if "annotations" in seq_dict:
                ann = np.asarray(seq_dict["annotations"])
                if ann.ndim > 1:
                    ann = ann[:, 0]
                out[seq_key] = ann.astype(int)
    return out


def load_kmeans_assigner(BASE):
    # Expect: {"scaler","kmeans","nn"} saved earlier
    pack = joblib.load(BASE / "global_kmeans_assigner.joblib")
    return pack["scaler"], pack["nn"]

def load_ward_assigner(BASE, n_clusters: int):
    Z = joblib.load(BASE / "global_hierarchical_linkage.joblib")["linkage_matrix"]
    templates = np.load(BASE / "global_templates_features.npz")["templates"]  # standardized
    labels_templates = fcluster(Z, n_clusters, criterion="maxclust")
    uniq = np.unique(labels_templates)
    centroids = np.vstack([templates[labels_templates == c].mean(axis=0) for c in uniq])
    scaler = joblib.load(BASE / "global_opentsne_embedding.joblib")["scaler"]
    nn = NearestNeighbors(n_neighbors=1).fit(centroids)
    return scaler, nn

def load_pair_features(safe_seq: str, persp: int) -> np.ndarray:
    npz_path = find_pair_wavelet(safe_seq, persp)
    if npz_path is None:
        raise FileNotFoundError(f"Missing pair wavelet for seq={safe_seq}, persp={persp}.")
    npz = np.load(npz_path)
    return np.asarray(npz["spectrogram"]).T  # (n_frames, n_feats)

def load_individual_features(BASE, uid: int) -> np.ndarray:
    npz = np.load(BASE / f"wavelet_spectrogram_id{uid}.npz")
    return np.asarray(npz["spectrogram"]).T  # (n_frames, 250)


def to_multi_hot(y_mc: np.ndarray, behaviors) -> np.ndarray:
    Y = np.zeros((y_mc.shape[0], len(behaviors)), dtype=np.float32)
    for j, b in enumerate(behaviors):
        Y[:, j] = (y_mc == b).astype(np.float32)
    return Y

def infer_base_feats(feat_total, time_steps, ADD_POOL=True):
    # Try all divisors to find a consistent base_feats
    for base in range(1, 8192):
        pooled_dim = feat_total - base * time_steps
        if pooled_dim < 0: break
        if not ADD_POOL and pooled_dim == 0:
            return base, 0, False
        if ADD_POOL and pooled_dim % base == 0:
            return base, pooled_dim, True
    raise ValueError(f"Cannot infer BASE_FEATS from feat_total={feat_total}, time_steps={time_steps}")