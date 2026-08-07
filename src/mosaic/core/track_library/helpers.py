"""Shared utilities for track format converters."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import json
import numpy as np

from mosaic.core.helpers import text_cell

if TYPE_CHECKING:
    import pandas as pd


def column_names(frame: "pd.DataFrame") -> list[str]:
    """The frame's column labels as plain strings.

    A pandas ``Index`` is keyed by an untyped label type, so iterating it
    directly leaves every name unknown to the type checker and every use of one
    unknown in turn. Materializing the labels once, here, keeps that confined to
    a single expression instead of spreading through every caller's loop.
    """
    return [str(name) for name in frame.columns.tolist()]


def column_array(frame: "pd.DataFrame", name: str) -> np.ndarray:
    """One column as a numpy array, for the same reason as :func:`column_names`.

    numpy's dtypes are typed where pandas' element access is not, so a caller
    that needs to ask what a column *holds* -- rather than merely carry it --
    gets a typed answer by going through this.
    """
    return np.asarray(frame[name])


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
    c = XY - mu  # (T, L, 2)

    # Covariance: (T, 2, 2) via einsum  cov[t] = c[t].T @ c[t]
    cov = np.einsum("tli,tlj->tij", c, c)  # (T, 2, 2)

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


def norm_hint(x: object) -> str | None:
    """A group/sequence hint, absent spellings collapsed to ``None``.

    One delegation rather than a second implementation of what counts as absent:
    a converter reading a hint and the index writer recording what it produced
    have to agree, and two copies of that rule are free to drift. ``None`` here
    rather than ``""`` because the callers spell their fallback as ``or``.
    """
    return text_cell(x) or None
