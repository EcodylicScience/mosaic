"""Deriving a bounding box from keypoints.

Three strategies, because a tight box around the keypoints is wrong for the
common case. A midline-only schema on an axis-aligned animal puts every point on
one line, so the tight box has zero height and the instance is untrainable --
which is what ``isotropic`` and ``oriented`` exist to avoid.

**These return a normalized centre-xywh tuple, which is YOLO's shape rather than
this package's.** That is inherited, not intended: the geometry is a property of
the keypoints and the normalization is a property of the format being written.
:class:`BboxPolicy` is the seam that separates them, and until an emitter takes
that argument the functions keep the return type their callers expect.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from mosaic.core.pipeline.types import Params

__all__ = [
    "BBoxMethod",
    "BboxPolicy",
    "keypoints_to_bbox",
    "keypoints_to_bbox_isotropic",
    "keypoints_to_bbox_oriented",
]

BBoxMethod = Literal["tight", "isotropic", "oriented"]
"""How a box is derived from the points it must contain.

``tight``
    The axis-aligned hull of the valid keypoints, plus a fractional margin.
    Degenerate whenever the points are colinear.
``isotropic``
    The tight hull, then an absolute pixel pad scaled to body length, so a
    colinear subject still gets a box with area.
``oriented``
    Padded along the head-tail axis and across it separately, which is the
    only one that stays snug on a diagonal animal.
"""


class BboxPolicy(Params):
    """How an emitter should derive a box when the source supplied none.

    A parameter object rather than a pile of keyword arguments, because it has
    to reach a run identifier: two datasets emitted from one annotation set with
    different padding are different training data, and a hash over the emitter's
    params is what says so.

    Its existence is what retired the bespoke rewriter. Recomputing boxes on an
    already-emitted dataset used to mean parsing YOLO rows back into arrays and
    editing four columns, because the padding choice was not expressible at the
    point the dataset was built. Now it is, so re-padding is reading the dataset
    and emitting it again under a different policy -- see
    :func:`mosaic.tracking.pose_training.repad.repad_yolo_pose`.

    **Padding is not a nicety.** A tight box around a midline-only schema on an
    axis-aligned animal has zero height, and an instance with no height cannot
    be trained on. ``isotropic`` pads by a fraction of body length with a pixel
    floor; ``oriented`` pads along and across the head-tail axis separately.

    The defaults reproduce what the converters hardcode today, so declaring a
    policy changes nothing until someone changes the policy.
    """

    method: BBoxMethod = "tight"
    margin: float = 0.1
    pad_frac_of_body: float = 0.30
    min_pad_px: float = 20.0
    length_pad_frac: float = 0.25
    side_pad_frac: float = 0.35
    head_index: int | None = Field(default=None)
    tail_index: int | None = Field(default=None)


def _aabb_to_norm_cxcywh(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_w: int,
    img_h: int,
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
        x_min - pad,
        y_min - pad,
        x_max + pad,
        y_max + pad,
        img_w,
        img_h,
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
            kps_xy,
            img_w,
            img_h,
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
            kps_xy,
            img_w,
            img_h,
            head_idx=head_idx,
            tail_idx=tail_idx,
            **(fallback_kwargs or {}),
        )

    u = axis / L
    n = np.array([-u[1], u[0]])
    half_side = side_pad_frac * L
    ext = length_pad_frac * L
    head_ext = head + u * ext
    tail_ext = tail - u * ext
    corners = np.stack(
        [
            head_ext + n * half_side,
            head_ext - n * half_side,
            tail_ext + n * half_side,
            tail_ext - n * half_side,
        ]
    )
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
            x_min - bw * margin,
            y_min - bh * margin,
            x_max + bw * margin,
            y_max + bh * margin,
            img_w,
            img_h,
        )
    if method == "isotropic":
        return keypoints_to_bbox_isotropic(
            kps_xy,
            img_w,
            img_h,
            pad_frac_of_body=pad_frac_of_body,
            min_pad_px=min_pad_px,
            head_idx=head_idx,
            tail_idx=tail_idx,
        )
    if method == "oriented":
        if head_idx is None or tail_idx is None:
            raise ValueError("method='oriented' requires both head_idx and tail_idx")
        return keypoints_to_bbox_oriented(
            kps_xy,
            img_w,
            img_h,
            head_idx=head_idx,
            tail_idx=tail_idx,
            length_pad_frac=length_pad_frac,
            side_pad_frac=side_pad_frac,
            fallback_kwargs={
                "pad_frac_of_body": pad_frac_of_body,
                "min_pad_px": min_pad_px,
            },
        )
    raise ValueError(f"Unknown bbox method: {method!r}")
