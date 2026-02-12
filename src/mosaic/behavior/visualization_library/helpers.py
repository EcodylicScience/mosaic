"""Shared visualization utilities.

This module contains helper functions used across visualization modules:
- Geometry utilities (pose extraction, bounding boxes, centroids)
- Color utilities (palettes, color assignment)
- Video I/O utilities (capture, writer, scaling)
- Cropping utilities (safe crop with padding, rotation)
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Any, Optional, Iterable, Dict
import numpy as np
import pandas as pd
import cv2


# =============================================================================
# Parameter Merging (follows feature_library pattern)
# =============================================================================

def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user overrides with defaults. None values in overrides use defaults."""
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


# =============================================================================
# Color Palettes
# =============================================================================

ID_PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (168, 218, 220),
    (240, 128, 128),
    (255, 195, 0),
    (88, 24, 69),
    (0, 109, 119),
    (144, 190, 109),
    (67, 170, 139),
]

LABEL_PALETTE = [
    (220, 20, 60),
    (25, 130, 196),
    (60, 179, 113),
    (255, 140, 0),
    (147, 112, 219),
    (70, 130, 180),
    (255, 182, 193),
    (0, 206, 209),
    (255, 215, 0),
    (244, 164, 96),
]


def _color_for_id(id_val: Any) -> Tuple[int, int, int]:
    """Hash-based color selection for individual IDs."""
    idx = hash(str(id_val)) % len(ID_PALETTE)
    return ID_PALETTE[idx]


def _color_for_label(label_val: Any) -> Tuple[int, int, int]:
    """Hash-based color selection for labels."""
    idx = hash(str(label_val)) % len(LABEL_PALETTE)
    return LABEL_PALETTE[idx]


# =============================================================================
# Geometry Utilities
# =============================================================================

def _pose_column_pairs(columns: Iterable[str]) -> list[Tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs from column names."""
    pose_pairs = []
    xs = [c for c in columns if c.startswith("poseX")]
    for x_col in sorted(xs):
        idx = x_col[5:]
        y_col = f"poseY{idx}"
        if y_col in columns:
            pose_pairs.append((x_col, y_col))
    return pose_pairs


def _extract_pose_points(row: pd.Series, pose_pairs: list[Tuple[str, str]]) -> list[Tuple[float, float]]:
    """Extract list of (x, y) coordinates from a DataFrame row."""
    pts = []
    for x_col, y_col in pose_pairs:
        x = row.get(x_col)
        y = row.get(y_col)
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts.append((float(x), float(y)))
    return pts


def _compute_bbox(points: list[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box from point list."""
    xs = [p[0] for p in points if np.isfinite(p[0])]
    ys = [p[1] for p in points if np.isfinite(p[1])]
    if not xs or not ys:
        return (np.nan, np.nan, np.nan, np.nan)
    return (min(xs), min(ys), max(xs), max(ys))


def _extract_centroid(row: pd.Series, candidates: list[Tuple[str, str]]) -> Optional[Tuple[float, float]]:
    """Find first valid centroid from candidate (x, y) column pairs."""
    for x_col, y_col in candidates:
        x = row.get(x_col)
        y = row.get(y_col)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            return (float(x), float(y))
    return None


# =============================================================================
# Video I/O Utilities
# =============================================================================

def _open_video_capture(path: Path | str) -> Tuple[cv2.VideoCapture, float, Tuple[int, int]]:
    """Initialize OpenCV VideoCapture. Returns (cap, fps, (width, height))."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), (width, height)


def _scaled_size(base_size: Tuple[int, int], downscale: float) -> Tuple[int, int]:
    """Compute resized dimensions with downscale factor."""
    w, h = base_size
    if downscale and downscale > 0 and downscale != 1.0:
        return (max(1, int(w * downscale)), max(1, int(h * downscale)))
    return base_size


def create_video_writer(
    path: Path | str,
    fps: float,
    frame_size: Tuple[int, int],
    fourcc: str = "mp4v"
) -> cv2.VideoWriter:
    """Create a VideoWriter with standard settings.

    Parameters
    ----------
    path : Path or str
        Output file path
    fps : float
        Frames per second
    frame_size : tuple of (width, height)
        Frame dimensions
    fourcc : str
        Four-character codec code (default "mp4v")

    Returns
    -------
    cv2.VideoWriter
        Opened video writer. Caller is responsible for calling .release()
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(path), codec, float(fps), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {path}")
    return writer


# =============================================================================
# Label/Data Lookup Utilities
# =============================================================================

def _lookup_label_series(per_id_map: dict, id_val: Any) -> Optional[pd.Series]:
    """Flexible lookup with fallback: exact match -> str(id_val) -> int(id_val) -> None."""
    if id_val in per_id_map:
        return per_id_map[id_val]
    candidates = []
    if id_val is not None:
        candidates.append(str(id_val))
        try:
            candidates.append(int(id_val))
        except Exception:
            pass
    for cand in candidates:
        if cand in per_id_map:
            return per_id_map[cand]
    for fallback in (None, "", "none"):
        if fallback in per_id_map:
            return per_id_map[fallback]
    return None


def _scalar_from_series(value: Any) -> Any:
    """Extract scalar from pd.Series or return as-is."""
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return value.iloc[-1]
    return value


def _format_label_text(value: Any) -> str:
    """Format value(s) for text overlay."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(_format_label_text(v) for v in value)
    try:
        return f"{value}"
    except Exception:
        return str(value)


# =============================================================================
# Cropping Utilities (NEW for egocentric crop)
# =============================================================================

def safe_crop_with_padding(
    image: np.ndarray,
    center: Tuple[int, int],
    crop_size: Tuple[int, int],
    pad_value: int = 0
) -> np.ndarray:
    """Extract a crop centered at (cx, cy), padding if out of bounds.

    Parameters
    ----------
    image : np.ndarray
        Source image (H, W, C) or (H, W)
    center : tuple of (cx, cy)
        Center point in pixel coordinates
    crop_size : tuple of (width, height)
        Size of output crop
    pad_value : int
        Padding color (0=black, 255=white)

    Returns
    -------
    np.ndarray
        Cropped image of size (height, width, C) or (height, width)
    """
    cx, cy = int(center[0]), int(center[1])
    crop_w, crop_h = crop_size
    h, w = image.shape[:2]

    # Compute source region
    x0 = cx - crop_w // 2
    y0 = cy - crop_h // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    # Compute valid source region (clamped to image bounds)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    # Compute destination region in output crop
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    # Create output with padding
    if image.ndim == 3:
        crop = np.full((crop_h, crop_w, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        crop = np.full((crop_h, crop_w), pad_value, dtype=image.dtype)

    # Copy valid region
    if src_x1 > src_x0 and src_y1 > src_y0:
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return crop


def compute_heading_angle(neck_xy: Tuple[float, float], tail_xy: Tuple[float, float]) -> float:
    """Compute heading angle from anatomical landmarks (neck -> tail direction).

    Parameters
    ----------
    neck_xy : tuple of (x, y)
        Neck/head position
    tail_xy : tuple of (x, y)
        Tail position

    Returns
    -------
    float
        Heading angle in radians. 0 = facing right (+x), pi/2 = facing down (+y)
        The heading points FROM tail TO neck (i.e., where the animal is facing).
    """
    dx = neck_xy[0] - tail_xy[0]
    dy = neck_xy[1] - tail_xy[1]
    return float(np.arctan2(dy, dx))


def infer_angle_degrees(angle_series: pd.Series) -> bool:
    """Infer whether angle values are in degrees or radians.

    Parameters
    ----------
    angle_series : pd.Series
        Series of angle values

    Returns
    -------
    bool
        True if angles appear to be in degrees, False if radians
    """
    # Filter to finite values
    valid = angle_series.dropna()
    valid = valid[np.isfinite(valid)]
    if valid.empty:
        return False  # Default to radians if no data

    abs_max = np.abs(valid).max()
    # If max absolute value > 2*pi, likely degrees
    return abs_max > 2 * np.pi
