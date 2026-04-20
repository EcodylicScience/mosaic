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
from typing import Literal, Sequence

import numpy as np

BBoxMethod = Literal["tight", "isotropic", "oriented"]


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


def _aabb_to_norm_cxcywh(
    x_min: float, y_min: float, x_max: float, y_max: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Clip an AABB to the image, then normalize to center-xywh in [0, 1]."""
    x_min = max(0.0, float(x_min))
    y_min = max(0.0, float(y_min))
    x_max = min(float(img_w), float(x_max))
    y_max = min(float(img_h), float(y_max))
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


def keypoints_to_bbox_isotropic(
    kps_xy: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    pad_frac_of_body: float = 0.30,
    min_pad_px: float = 20.0,
    head_idx: int | None = None,
    tail_idx: int | None = None,
) -> tuple[float, float, float, float]:
    """Tight bbox around valid keypoints, then an absolute pixel pad in x and y.

    Unlike ``keypoints_to_bbox`` with ``method='tight'``, the pad is a fixed
    number of pixels (scaled to body length), so it does *not* collapse to
    zero when the keypoints are colinear — which is exactly the degenerate
    case for midline-only keypoint schemas with an axis-aligned subject.

    Parameters
    ----------
    kps_xy : ndarray, shape (N, 2)
        Keypoint pixel coordinates. NaN/inf entries are treated as invalid.
    img_w, img_h : int
        Image dimensions used for clipping and normalization.
    pad_frac_of_body : float
        Pad in pixels as a fraction of the animal's body length.
    min_pad_px : float
        Floor for the absolute pad, for degenerate / overlapping keypoints.
    head_idx, tail_idx : int or None
        If given and both are valid, body length is ``‖head − tail‖``; else
        the diagonal of the tight keypoint bbox is used.

    Returns
    -------
    (cx, cy, w, h) : tuple of float in [0, 1].
    """
    valid = np.isfinite(kps_xy).all(axis=1)
    if not valid.any():
        return (0.0, 0.0, 0.0, 0.0)

    pts = kps_xy[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    def _valid_at(idx: int | None) -> bool:
        return (
            idx is not None
            and 0 <= idx < len(kps_xy)
            and bool(np.isfinite(kps_xy[idx]).all())
        )

    if _valid_at(head_idx) and _valid_at(tail_idx):
        body = float(np.linalg.norm(kps_xy[head_idx] - kps_xy[tail_idx]))
    else:
        body = float(np.hypot(x_max - x_min, y_max - y_min))

    pad = max(float(min_pad_px), float(pad_frac_of_body) * body)
    return _aabb_to_norm_cxcywh(
        x_min - pad, y_min - pad, x_max + pad, y_max + pad, img_w, img_h,
    )


def keypoints_to_bbox_oriented(
    kps_xy: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    head_idx: int,
    tail_idx: int,
    length_pad_frac: float = 0.25,
    side_pad_frac: float = 0.35,
    fallback_kwargs: dict | None = None,
) -> tuple[float, float, float, float]:
    """Oriented body rectangle (head→tail axis) then its AABB enclosing rect.

    Builds a 4-corner rectangle whose long axis runs along ``head − tail``,
    extended beyond head and tail by ``length_pad_frac · L`` at each end and
    padded perpendicularly by ``side_pad_frac · L`` on each side. The final
    bbox is the axis-aligned enclosing rect of those 4 corners, normalized
    to [0, 1].

    Falls back to ``keypoints_to_bbox_isotropic`` (with any
    ``fallback_kwargs``) when head or tail keypoint is invalid, or when the
    head-tail distance is numerically zero.
    """
    N = len(kps_xy)
    head_ok = (0 <= head_idx < N) and bool(np.isfinite(kps_xy[head_idx]).all())
    tail_ok = (0 <= tail_idx < N) and bool(np.isfinite(kps_xy[tail_idx]).all())

    if not (head_ok and tail_ok):
        return keypoints_to_bbox_isotropic(
            kps_xy, img_w, img_h,
            head_idx=head_idx if head_ok else None,
            tail_idx=tail_idx if tail_ok else None,
            **(fallback_kwargs or {}),
        )

    head = np.asarray(kps_xy[head_idx], dtype=float)
    tail = np.asarray(kps_xy[tail_idx], dtype=float)
    axis = head - tail
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return keypoints_to_bbox_isotropic(
            kps_xy, img_w, img_h,
            head_idx=head_idx, tail_idx=tail_idx,
            **(fallback_kwargs or {}),
        )

    u = axis / L
    n = np.array([-u[1], u[0]])
    half_side = side_pad_frac * L
    ext = length_pad_frac * L
    head_ext = head + u * ext
    tail_ext = tail - u * ext
    corners = np.stack([
        head_ext + n * half_side,
        head_ext - n * half_side,
        tail_ext + n * half_side,
        tail_ext - n * half_side,
    ])
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    return _aabb_to_norm_cxcywh(x_min, y_min, x_max, y_max, img_w, img_h)


def keypoints_to_bbox(
    kps_xy: np.ndarray,
    img_w: int,
    img_h: int,
    margin: float = 0.1,
    *,
    method: BBoxMethod = "tight",
    head_idx: int | None = None,
    tail_idx: int | None = None,
    pad_frac_of_body: float = 0.30,
    min_pad_px: float = 20.0,
    length_pad_frac: float = 0.25,
    side_pad_frac: float = 0.35,
) -> tuple[float, float, float, float]:
    """Compute a normalized bounding box from keypoint pixel coordinates.

    Dispatches to one of three methods:

    - ``'tight'`` (default, legacy behavior): min/max of valid keypoints
      padded by ``margin`` as a *fraction of the tight bbox size*. Collapses
      to zero in the thin direction when keypoints are colinear.
    - ``'isotropic'``: min/max of valid keypoints plus an *absolute* pad in
      both x and y (``pad_frac_of_body · body_length`` or ``min_pad_px``).
      See :func:`keypoints_to_bbox_isotropic`.
    - ``'oriented'``: build an oriented rectangle from the head→tail axis
      with configurable length/side pads, then take its AABB. Requires
      ``head_idx`` and ``tail_idx``. See :func:`keypoints_to_bbox_oriented`.

    Parameters
    ----------
    kps_xy : ndarray, shape (N, 2)
        Keypoint (x, y) pixel coordinates. NaN/inf entries are ignored.
    img_w, img_h : int
        Image dimensions for normalization.
    margin : float
        Fractional margin used only when ``method='tight'``.
    method : {'tight', 'isotropic', 'oriented'}
        Bbox-derivation strategy.
    head_idx, tail_idx : int or None
        Keypoint indices of the head and tail. Required for ``'oriented'``;
        optional (but recommended) for ``'isotropic'`` to compute body length.
    pad_frac_of_body, min_pad_px : float
        Parameters for ``'isotropic'`` (also used as fallback for
        ``'oriented'`` when head/tail are missing).
    length_pad_frac, side_pad_frac : float
        Parameters for ``'oriented'``.
    """
    if method == "tight":
        valid = np.isfinite(kps_xy).all(axis=1)
        if not valid.any():
            return (0.0, 0.0, 0.0, 0.0)
        pts = kps_xy[valid]
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bw = x_max - x_min
        bh = y_max - y_min
        return _aabb_to_norm_cxcywh(
            x_min - bw * margin, y_min - bh * margin,
            x_max + bw * margin, y_max + bh * margin,
            img_w, img_h,
        )
    if method == "isotropic":
        return keypoints_to_bbox_isotropic(
            kps_xy, img_w, img_h,
            pad_frac_of_body=pad_frac_of_body,
            min_pad_px=min_pad_px,
            head_idx=head_idx, tail_idx=tail_idx,
        )
    if method == "oriented":
        if head_idx is None or tail_idx is None:
            raise ValueError(
                "method='oriented' requires both head_idx and tail_idx"
            )
        return keypoints_to_bbox_oriented(
            kps_xy, img_w, img_h,
            head_idx=head_idx, tail_idx=tail_idx,
            length_pad_frac=length_pad_frac,
            side_pad_frac=side_pad_frac,
            fallback_kwargs={
                "pad_frac_of_body": pad_frac_of_body,
                "min_pad_px": min_pad_px,
            },
        )
    raise ValueError(f"Unknown bbox method: {method!r}")


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
