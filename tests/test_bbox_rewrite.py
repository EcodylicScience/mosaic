"""Tests for keypoint-derived bbox methods and dataset rewriting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from mosaic.tracking.pose_training.bbox_rewrite import rewrite_dataset_bboxes
from mosaic.tracking.pose_training.converters.base import (
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
        kps, IMG_W, IMG_H,
        method="isotropic",
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        pad_frac_of_body=0.3, min_pad_px=20.0,
    )
    # body_length = 400 px; pad = max(20, 0.3*400) = 120 px
    # height = 2*120 = 240 px -> normalized 0.24
    assert h == pytest.approx(0.24, abs=0.01)
    assert w == pytest.approx((400 + 2 * 120) / IMG_W, abs=0.01)


def test_oriented_gives_nonzero_height_on_horizontal_mouse() -> None:
    kps = _horizontal_mouse_kps()
    cx, cy, w, h = keypoints_to_bbox(
        kps, IMG_W, IMG_H,
        method="oriented",
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        length_pad_frac=0.25, side_pad_frac=0.35,
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
        kps, IMG_W, IMG_H,
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        length_pad_frac=0.25, side_pad_frac=0.35,
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
        kps, IMG_W, IMG_H,
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        length_pad_frac=0.25, side_pad_frac=0.35,
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
        kps, IMG_W, IMG_H,
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        pad_frac_of_body=0.3, min_pad_px=20.0,
    )
    # pad = 20 px, so bbox is 40x40 = 0.04 x 0.04
    assert w == pytest.approx(0.04, abs=0.005)
    assert h == pytest.approx(0.04, abs=0.005)


# -----------------------------------------------------------------------------
# dataset rewriter
# -----------------------------------------------------------------------------


def _write_tiny_dataset(root: Path, *, num_kpts: int = 6) -> None:
    """Write a minimal train split with one horizontal-mouse image+label."""
    img_dir = root / "train" / "images"
    lbl_dir = root / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img_path = img_dir / "frame000.png"
    Image.new("RGB", (IMG_W, IMG_H), color=(128, 128, 128)).save(img_path)

    # Horizontal mouse label in normalized coords, all v=2
    kps = _horizontal_mouse_kps()
    parts = ["0", "0.5", "0.5", "0.4", "0.001"]  # tight degenerate bbox
    for (x, y) in kps:
        parts += [f"{x / IMG_W:.6f}", f"{y / IMG_H:.6f}", "2"]
    (lbl_dir / "frame000.txt").write_text(" ".join(parts) + "\n")


def test_rewrite_isotropic_fixes_degenerate_bbox(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst_iso"
    _write_tiny_dataset(src)

    summary = rewrite_dataset_bboxes(
        src, dst,
        method="isotropic", num_kpts=6,
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        pad_frac_of_body=0.3, min_pad_px=20.0,
    )

    assert summary["train"]["rows"] == 1
    assert summary["train"]["images_copied"] == 1

    out = (dst / "train" / "labels" / "frame000.txt").read_text().split()
    new_h = float(out[4])
    assert new_h > 0.2  # was ~0.001 before


def test_rewrite_handles_v0_as_invalid(tmp_path: Path) -> None:
    """Keypoints with v=0 should be ignored in bbox computation."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src_img = src / "train" / "images"
    src_lbl = src / "train" / "labels"
    src_img.mkdir(parents=True)
    src_lbl.mkdir(parents=True)
    Image.new("RGB", (IMG_W, IMG_H), color=(0, 0, 0)).save(src_img / "f.png")

    # head kp has v=0, others v=2. Oriented should fall back to isotropic.
    kps = _horizontal_mouse_kps()
    parts = ["0", "0.5", "0.5", "0.4", "0.001"]
    for i, (x, y) in enumerate(kps):
        v = 0 if i == HEAD_IDX else 2
        parts += [f"{x / IMG_W:.6f}", f"{y / IMG_H:.6f}", str(v)]
    (src_lbl / "f.txt").write_text(" ".join(parts) + "\n")

    summary = rewrite_dataset_bboxes(
        src, dst,
        method="oriented", num_kpts=6,
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
        length_pad_frac=0.25, side_pad_frac=0.35,
        pad_frac_of_body=0.3, min_pad_px=20.0,
    )
    assert summary["train"]["rows_fallback_to_isotropic"] == 1


def test_rewrite_preserves_keypoints(tmp_path: Path) -> None:
    """Bbox changes but keypoint columns are byte-identical."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_tiny_dataset(src)

    src_parts = (src / "train" / "labels" / "frame000.txt").read_text().split()

    rewrite_dataset_bboxes(
        src, dst,
        method="oriented", num_kpts=6,
        head_idx=HEAD_IDX, tail_idx=TAIL_IDX,
    )
    dst_parts = (dst / "train" / "labels" / "frame000.txt").read_text().split()

    # All keypoint fields (tokens 5 onwards) should match
    assert dst_parts[5:] == src_parts[5:]
    # Bbox fields should differ
    assert dst_parts[1:5] != src_parts[1:5]
