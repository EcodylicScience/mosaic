"""Shared utilities for track format converters."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import math
import json
import numpy as np
import pandas as pd


def load_calms21(path: Path | str):
    """
    Load a single CalMS21 file: either .npy (dict) or the original .json.
    Returns a nested dict: group -> seq_id -> dict(...)
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=True).item()
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported CalMS21 path (expect .npy or .json): {p}")


def angle_from_two_points(neck_xy: np.ndarray, tail_xy: np.ndarray) -> np.ndarray:
    """
    heading from tail -> neck, angle w.r.t +x (radians)
    neck_xy, tail_xy: (T,2)
    """
    v = neck_xy - tail_xy
    return np.arctan2(v[:, 1], v[:, 0])


def angle_from_pca(XY: np.ndarray) -> np.ndarray:
    """
    PCA-based heading (fallback). XY: (T, L, 2) landmarks for one animal.
    Uses first principal component per frame; sign is arbitrary.

    Vectorized: computes 2x2 covariance matrices for all frames at once
    and solves the eigenproblem analytically (closed-form for 2x2 symmetric).
    """
    # XY shape: (T, L, 2)
    # Center each frame
    mu = XY.mean(axis=1, keepdims=True)  # (T, 1, 2)
    c = XY - mu                           # (T, L, 2)

    # Covariance: (T, 2, 2) via einsum  cov[t] = c[t].T @ c[t]
    cov = np.einsum('tli,tlj->tij', c, c)  # (T, 2, 2)

    # For a 2x2 symmetric matrix [[a, b], [b, d]], the larger eigenvalue's
    # eigenvector can be computed analytically.
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    d = cov[:, 1, 1]

    # Eigenvalues: 0.5*(a+d) +/- 0.5*sqrt((a-d)^2 + 4*b^2)
    # We only need the eigenvector of the larger eigenvalue.
    diff = a - d
    disc = np.sqrt(diff * diff + 4.0 * b * b)
    lam_max = 0.5 * ((a + d) + disc)

    # Eigenvector for lam_max: (lam_max - d, b) or (b, lam_max - a)
    # Use (b, lam_max - a) to avoid division; normalize via arctan2
    vx = b
    vy = lam_max - a

    # Fallback: when b ≈ 0, the matrix is diagonal → eigenvector is (1,0) or (0,1)
    diag_mask = np.abs(b) < 1e-12
    vx = np.where(diag_mask, np.where(a >= d, 1.0, 0.0), vx)
    vy = np.where(diag_mask, np.where(a >= d, 0.0, 1.0), vy)

    return np.arctan2(vy, vx)


def norm_hint(x: Optional[Any]) -> Optional[str]:
    """Normalize a group/sequence hint: treat None, NaN, '', 'nan', 'none' as None."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None
        return s
    # Pandas NA/NaT, etc.
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)
