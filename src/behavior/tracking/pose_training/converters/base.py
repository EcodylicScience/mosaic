"""Shared utilities for converting pose annotations to YOLO pose format.

YOLO pose label format (per line, all values normalized to [0, 1]):
    <class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <kp0_vis> <kp1_x> <kp1_y> <kp1_vis> ...

Visibility flags: 0 = not labeled, 1 = labeled but occluded, 2 = labeled and visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class KeypointSchema:
    """Defines the keypoint layout for a pose model."""
    names: list[str]
    skeleton: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_keypoints(self) -> int:
        return len(self.names)

    @property
    def kpt_shape(self) -> list[int]:
        """Return [num_keypoints, 3] for YOLO data.yaml."""
        return [self.num_keypoints, 3]


def keypoints_to_bbox(
    kps_xy: np.ndarray,
    img_w: int,
    img_h: int,
    margin: float = 0.1,
) -> tuple[float, float, float, float]:
    """Compute a normalized bounding box from keypoint pixel coordinates.

    Parameters
    ----------
    kps_xy : ndarray, shape (N, 2)
        Keypoint (x, y) pixel coordinates. NaN/inf entries are ignored.
    img_w, img_h : int
        Image dimensions for normalization.
    margin : float
        Fractional margin to add around the tight bbox (relative to bbox size).

    Returns
    -------
    (cx, cy, w, h) : tuple[float, ...]
        Normalized center-x, center-y, width, height in [0, 1].
    """
    valid = np.isfinite(kps_xy).all(axis=1)
    if not valid.any():
        return (0.0, 0.0, 0.0, 0.0)

    pts = kps_xy[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * margin
    pad_y = bh * margin

    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(float(img_w), x_max + pad_x)
    y_max = min(float(img_h), y_max + pad_y)

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return (
        float(np.clip(cx, 0, 1)),
        float(np.clip(cy, 0, 1)),
        float(np.clip(w, 0, 1)),
        float(np.clip(h, 0, 1)),
    )


def normalize_coords(x: float, y: float, img_w: int, img_h: int) -> tuple[float, float]:
    """Normalize pixel coordinates to [0, 1]."""
    return (float(np.clip(x / img_w, 0, 1)), float(np.clip(y / img_h, 0, 1)))


def format_yolo_pose_line(
    class_id: int,
    bbox_cxcywh: tuple[float, float, float, float],
    keypoints_xyv: Sequence[tuple[float, float, int]],
) -> str:
    """Format a single YOLO pose annotation line.

    Parameters
    ----------
    class_id : int
        Object class (typically 0 for single-class pose).
    bbox_cxcywh : tuple
        Normalized (center_x, center_y, width, height).
    keypoints_xyv : sequence of (x, y, visibility)
        Normalized keypoint coordinates and visibility flag per keypoint.

    Returns
    -------
    str
        Formatted YOLO pose label line.
    """
    parts = [str(class_id)]
    parts.extend(f"{v:.6f}" for v in bbox_cxcywh)
    for kx, ky, kv in keypoints_xyv:
        parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(int(kv))])
    return " ".join(parts)


def write_yolo_label(path: str | Path, lines: list[str]) -> None:
    """Write YOLO label lines to a .txt file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
