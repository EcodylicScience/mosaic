"""The three ways a box is derived from the keypoints it must contain.

A tight box around a midline-only schema on an axis-aligned animal has zero
height, and an instance with no height cannot be trained on. These pin that,
and pin the two strategies that exist because of it -- which is why padding is
not a nicety: it is what makes a whole class of labelling usable at all.
"""

from __future__ import annotations


import numpy as np
import pytest

from mosaic.core.annotations.bbox import (
    keypoints_to_bbox,
    keypoints_to_bbox_isotropic,
    keypoints_to_bbox_oriented,
)

IMG_W, IMG_H = 1000, 1000
HEAD_IDX, TAIL_IDX = 0, 5


def _horizontal_mouse_kps() -> np.ndarray:
    """6 keypoints colinear along a horizontal line at y=500.

    head x=300 ... tail x=700. This is the degenerate case for the
    tight-margin bbox: y_max − y_min ≈ 0.
    """
    xs = np.linspace(300.0, 700.0, 6)
    ys = np.full(6, 500.0)
    return np.stack([xs, ys], axis=1)


def _diagonal_mouse_kps() -> np.ndarray:
    """6 keypoints on a 45° line from (300, 300) to (700, 700)."""
    xs = np.linspace(300.0, 700.0, 6)
    ys = np.linspace(300.0, 700.0, 6)
    return np.stack([xs, ys], axis=1)


# -----------------------------------------------------------------------------
# core bbox math
# -----------------------------------------------------------------------------


def test_tight_method_collapses_on_horizontal_mouse() -> None:
    """The legacy 'tight' behavior produces near-zero height — the bug we fix."""
    kps = _horizontal_mouse_kps()
    cx, cy, w, h = keypoints_to_bbox(kps, IMG_W, IMG_H, method="tight", margin=0.1)
    assert w > 0.2  # wide along x
    assert h < 0.01  # collapses in y — this is the bug


def test_isotropic_gives_nonzero_height_on_horizontal_mouse() -> None:
    kps = _horizontal_mouse_kps()
    cx, cy, w, h = keypoints_to_bbox(
        kps,
        IMG_W,
        IMG_H,
        method="isotropic",
        head_idx=HEAD_IDX,
        tail_idx=TAIL_IDX,
        pad_frac_of_body=0.3,
        min_pad_px=20.0,
    )
    # body_length = 400 px; pad = max(20, 0.3*400) = 120 px
    # height = 2*120 = 240 px -> normalized 0.24
    assert h == pytest.approx(0.24, abs=0.01)
    assert w == pytest.approx((400 + 2 * 120) / IMG_W, abs=0.01)


def test_oriented_gives_nonzero_height_on_horizontal_mouse() -> None:
    kps = _horizontal_mouse_kps()
    cx, cy, w, h = keypoints_to_bbox(
        kps,
        IMG_W,
        IMG_H,
        method="oriented",
        head_idx=HEAD_IDX,
        tail_idx=TAIL_IDX,
        length_pad_frac=0.25,
        side_pad_frac=0.35,
    )
    # L = 400 px; body oriented along +x so the AABB has:
    #   w = L + 2*(L*0.25) = 1.5*L = 600 px = 0.6 normalized
    #   h = 2*(L*0.35) = 0.7*L = 280 px = 0.28 normalized
    assert w == pytest.approx(0.6, abs=0.01)
    assert h == pytest.approx(0.28, abs=0.01)


def test_oriented_on_diagonal_mouse_has_both_dims() -> None:
    """Diagonal subject: axis-aligned enclosing rect of the oriented rect."""
    kps = _diagonal_mouse_kps()
    cx, cy, w, h = keypoints_to_bbox_oriented(
        kps,
        IMG_W,
        IMG_H,
        head_idx=HEAD_IDX,
        tail_idx=TAIL_IDX,
        length_pad_frac=0.25,
        side_pad_frac=0.35,
    )
    # L = sqrt(2)*400 ≈ 565.7; the rectangle is rotated 45°. Both the
    # length-extension and the side-pad contribute to x and y equally, so
    # w and h should be approximately equal.
    assert w == pytest.approx(h, rel=0.05)
    assert w > 0.5  # clearly larger than a degenerate bbox


def test_oriented_falls_back_to_isotropic_when_head_missing() -> None:
    """If head is NaN (v=0 upstream), oriented should fall back."""
    kps = _horizontal_mouse_kps()
    kps[HEAD_IDX] = [np.nan, np.nan]
    cx, cy, w, h = keypoints_to_bbox_oriented(
        kps,
        IMG_W,
        IMG_H,
        head_idx=HEAD_IDX,
        tail_idx=TAIL_IDX,
        length_pad_frac=0.25,
        side_pad_frac=0.35,
        fallback_kwargs={"pad_frac_of_body": 0.3, "min_pad_px": 20.0},
    )
    # Should behave like isotropic with only the 5 remaining keypoints
    assert h > 0.01  # not collapsed


def test_dispatcher_raises_on_oriented_without_indices() -> None:
    kps = _horizontal_mouse_kps()
    with pytest.raises(ValueError, match="head_idx"):
        keypoints_to_bbox(kps, IMG_W, IMG_H, method="oriented")


def test_isotropic_enforces_min_pad() -> None:
    """Two keypoints on top of each other => body length 0 => use min_pad_px."""
    kps = np.array([[500.0, 500.0]] * 6)
    cx, cy, w, h = keypoints_to_bbox_isotropic(
        kps,
        IMG_W,
        IMG_H,
        head_idx=HEAD_IDX,
        tail_idx=TAIL_IDX,
        pad_frac_of_body=0.3,
        min_pad_px=20.0,
    )
    # pad = 20 px, so bbox is 40x40 = 0.04 x 0.04
    assert w == pytest.approx(0.04, abs=0.005)
    assert h == pytest.approx(0.04, abs=0.005)
