"""Rewrite the bbox field of an existing YOLO-pose dataset.

Given a YOLO-pose dataset laid out as::

    <src>/<split>/images/<stem>.<ext>
    <src>/<split>/labels/<stem>.txt

where each label row is ``cls cx cy w h kp0x kp0y v0 kp1x kp1y v1 ...`` (all
normalized to [0, 1]) and ``v_i ∈ {0, 1, 2}``, produce a parallel dataset
``<dst>`` with the same keypoints but with the bbox columns recomputed using
one of the methods in :mod:`mosaic.tracking.pose_training.converters.base`.

This is the post-processing step that lets you train a model whose detection
bbox encloses the full animal body even when only midline keypoints were
labeled — without relabeling.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from .converters.base import (
    BBoxMethod,
    keypoints_to_bbox,
)

ImageLinkMode = Literal["copy", "symlink", "auto"]

_SPLITS = ("train", "valid", "test")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _parse_row(line: str, num_kpts: int) -> tuple[int, list[float], np.ndarray] | None:
    """Parse one YOLO-pose label row into (cls, [cx,cy,w,h], kps[N,3])."""
    toks = line.strip().split()
    if not toks:
        return None
    if len(toks) < 5 + num_kpts * 3:
        raise ValueError(
            f"Row has {len(toks)} tokens, expected {5 + num_kpts * 3} "
            f"(cls + 4 bbox + {num_kpts}·3 keypoints): {line!r}"
        )
    cls = int(toks[0])
    cxcywh = [float(v) for v in toks[1:5]]
    kp = np.array(toks[5 : 5 + num_kpts * 3], dtype=float).reshape(num_kpts, 3)
    return cls, cxcywh, kp


def _format_row(cls: int, cxcywh: tuple[float, float, float, float], kp: np.ndarray) -> str:
    """Render a YOLO-pose label row back to string form."""
    parts = [str(cls)] + [f"{v:.6f}" for v in cxcywh]
    for x, y, v in kp:
        parts += [f"{x:.6f}", f"{y:.6f}", f"{int(round(v))}"]
    return " ".join(parts)


def _kp_to_xy_pixels(kp: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalized keypoints with visibility to (N, 2) pixel coords.

    Keypoints with ``v == 0`` (not labeled) are mapped to NaN so that the
    downstream bbox functions (which use ``np.isfinite``) ignore them.
    ``v == 1`` (labeled but occluded) and ``v == 2`` (labeled + visible)
    are both treated as valid.
    """
    xy = np.full((len(kp), 2), np.nan, dtype=float)
    for i, (x, y, v) in enumerate(kp):
        if v >= 1:
            xy[i, 0] = float(x) * img_w
            xy[i, 1] = float(y) * img_h
    return xy


def _link_or_copy_image(src: Path, dst: Path, mode: ImageLinkMode) -> str:
    """Place the image at ``dst`` using the requested mode.

    Returns the effective mode used ("symlink" | "copy"), which may differ
    from the request when ``mode='auto'`` falls back.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    # Resolve src through any intermediate symlinks so we point at the real
    # file. Costs nothing and avoids symlink chains across the dataset.
    real_src = src.resolve()

    if mode == "copy":
        shutil.copy2(real_src, dst)
        return "copy"

    if mode == "symlink":
        os.symlink(real_src, dst)
        return "symlink"

    # auto
    try:
        os.symlink(real_src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(real_src, dst)
        return "copy"


def _find_image(images_dir: Path, stem: str) -> Path | None:
    """Locate an image whose filename stem matches ``stem``."""
    for ext in _IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def rewrite_dataset_bboxes(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    method: BBoxMethod,
    num_kpts: int,
    head_idx: int | None = None,
    tail_idx: int | None = None,
    image_link: ImageLinkMode = "copy",
    splits: tuple[str, ...] | None = None,
    overwrite_existing: bool = True,
    **method_kwargs,
) -> dict:
    """Produce a parallel YOLO-pose dataset with recomputed bboxes.

    Reads every label file under ``<src_dir>/<split>/labels/*.txt``, recomputes
    the bbox using ``method`` (one of ``'tight'``, ``'isotropic'``,
    ``'oriented'``), and writes the result to
    ``<dst_dir>/<split>/labels/<stem>.txt``. Images are placed in the matching
    ``<dst_dir>/<split>/images/`` directory using ``image_link``.

    Keypoints are passed through unchanged. Keypoints with ``v == 0`` are
    treated as invalid (filtered to NaN before bbox derivation); ``v == 1``
    and ``v == 2`` are both treated as valid, matching COCO conventions.

    Parameters
    ----------
    src_dir, dst_dir : path
        Source and destination dataset roots. Writes to
        ``<dst_dir>/<split>/{images,labels}`` for every present split.
    method : {'tight', 'isotropic', 'oriented'}
        Bbox-derivation strategy. See
        :func:`mosaic.tracking.pose_training.converters.base.keypoints_to_bbox`.
    num_kpts : int
        Number of keypoints per instance. Must match the dataset's
        ``kpt_shape``.
    head_idx, tail_idx : int or None
        Keypoint indices used by ``'isotropic'`` (optional) and ``'oriented'``
        (required).
    image_link : {'copy', 'symlink', 'auto'}
        Image placement strategy. ``'copy'`` is the safest default (portable
        across filesystems incl. Dropbox/SMB); ``'symlink'`` saves disk on
        local filesystems; ``'auto'`` tries symlink and falls back to copy.
    splits : tuple of str, optional
        Subset of splits to process. Defaults to all of ``train``, ``valid``,
        ``test`` that exist.
    overwrite_existing : bool
        If True, wipe ``<dst_dir>`` before writing. If False, overwrite files
        as encountered but leave unrelated content intact.
    **method_kwargs
        Forwarded to ``keypoints_to_bbox`` (``pad_frac_of_body``,
        ``min_pad_px``, ``length_pad_frac``, ``side_pad_frac``, ``margin``).

    Returns
    -------
    dict
        Summary counters keyed by split plus a ``'total'`` entry, each with
        ``rows``, ``rows_no_valid_kp``, ``rows_fallback_to_isotropic``,
        ``images_linked``, ``images_copied``.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    if overwrite_existing and dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = tuple(s for s in _SPLITS if (src_dir / s / "labels").exists())

    if method == "oriented" and (head_idx is None or tail_idx is None):
        raise ValueError(
            "method='oriented' requires both head_idx and tail_idx"
        )

    summary: dict = {"total": _empty_split_summary()}

    for split in splits:
        src_img_dir = src_dir / split / "images"
        src_lbl_dir = src_dir / split / "labels"
        dst_img_dir = dst_dir / split / "images"
        dst_lbl_dir = dst_dir / split / "labels"
        if not src_lbl_dir.exists():
            continue
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_sum = _empty_split_summary()
        img_size_cache: dict[str, tuple[int, int]] = {}

        for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
            stem = lbl_path.stem
            img_src = _find_image(src_img_dir, stem)
            if img_src is None:
                # No matching image — skip the row but note it
                split_sum["missing_image"] += 1
                continue

            # Place the image into the new dataset
            img_dst = dst_img_dir / img_src.name
            used = _link_or_copy_image(img_src, img_dst, image_link)
            if used == "symlink":
                split_sum["images_linked"] += 1
            else:
                split_sum["images_copied"] += 1

            # Get image dimensions (cache by stem)
            if stem not in img_size_cache:
                with Image.open(img_src) as im:
                    img_size_cache[stem] = im.size  # (W, H)
            img_w, img_h = img_size_cache[stem]

            # Rewrite rows
            out_lines: list[str] = []
            for line in lbl_path.read_text().splitlines():
                parsed = _parse_row(line, num_kpts)
                if parsed is None:
                    continue
                cls, cxcywh_orig, kp = parsed
                split_sum["rows"] += 1

                # Build pixel-space xy with NaN-for-invalid
                kps_xy = _kp_to_xy_pixels(kp, img_w, img_h)

                if not np.isfinite(kps_xy).all(axis=1).any():
                    # No valid keypoints — keep the original bbox unchanged
                    split_sum["rows_no_valid_kp"] += 1
                    out_lines.append(_format_row(cls, tuple(cxcywh_orig), kp))
                    continue

                # Track oriented->isotropic fallback
                if method == "oriented":
                    head_valid = (
                        head_idx is not None
                        and 0 <= head_idx < num_kpts
                        and bool(np.isfinite(kps_xy[head_idx]).all())
                    )
                    tail_valid = (
                        tail_idx is not None
                        and 0 <= tail_idx < num_kpts
                        and bool(np.isfinite(kps_xy[tail_idx]).all())
                    )
                    if not (head_valid and tail_valid):
                        split_sum["rows_fallback_to_isotropic"] += 1

                new_bbox = keypoints_to_bbox(
                    kps_xy, img_w, img_h,
                    method=method,
                    head_idx=head_idx,
                    tail_idx=tail_idx,
                    **method_kwargs,
                )
                out_lines.append(_format_row(cls, new_bbox, kp))

            (dst_lbl_dir / lbl_path.name).write_text(
                "\n".join(out_lines) + ("\n" if out_lines else "")
            )

        summary[split] = split_sum
        for k, v in split_sum.items():
            summary["total"][k] += v

    return summary


def _empty_split_summary() -> dict:
    return {
        "rows": 0,
        "rows_no_valid_kp": 0,
        "rows_fallback_to_isotropic": 0,
        "images_linked": 0,
        "images_copied": 0,
        "missing_image": 0,
    }
