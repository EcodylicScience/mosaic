"""Shared utilities for converting annotations to YOLO pose / POLO point format.

YOLO pose label format (per line, all values normalized to [0, 1]):
    <class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <kp0_vis> <kp1_x> <kp1_y> <kp1_vis> ...

POLO point label format (per line):
    <class_id> <radius> <x_rel> <y_rel>

Visibility flags (YOLO): 0 = not labeled, 1 = labeled but occluded, 2 = labeled and visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class PointDetectionSchema:
    """Defines the class layout for a point-detection model (e.g. POLO)."""

    names: list[str]
    radii: dict[int, float] = field(default_factory=dict)  # class_id -> radius (px)

    @property
    def num_classes(self) -> int:
        return len(self.names)


@dataclass
class LocalizerSchema:
    """Defines the class layout for a localizer heatmap model."""

    names: list[str]
    thresholds: dict[int, float] = field(default_factory=dict)  # class_id -> threshold

    @property
    def num_classes(self) -> int:
        return len(self.names)


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


def format_polo_label_line(
    class_id: int,
    radius: float,
    x_rel: float,
    y_rel: float,
) -> str:
    """Format a single POLO point-detection annotation line.

    Parameters
    ----------
    class_id : int
        Object class.
    radius : float
        Class-specific radius in pixels.
    x_rel, y_rel : float
        Normalized point coordinates in [0, 1].

    Returns
    -------
    str
        Formatted POLO label line: ``<class_id> <radius> <x_rel> <y_rel>``.
    """
    return f"{class_id} {radius:.1f} {x_rel:.6f} {y_rel:.6f}"


def write_yolo_label(path: str | Path, lines: list[str]) -> None:
    """Write label lines (YOLO pose or POLO point) to a .txt file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
