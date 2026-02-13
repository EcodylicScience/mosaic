"""Convert Lightning Pose CSV output to YOLO pose format.

Lightning Pose CSVs use a DeepLabCut-style multi-header format:
    Row 0: scorer (e.g. "heatmap_mhcrnn_tracker")
    Row 1: bodypart names (e.g. "nose", "left_ear", ...)
    Row 2: coordinate type ("x", "y", "likelihood")
    Row 3+: frame_index, then triplets of (x, y, likelihood) per keypoint

This converter reads pre-extracted frames (from ``behavior.media.extract_frames``)
and produces YOLO pose label files for the frame indices present in the extraction
manifest, avoiding redundant video decoding.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .base import (
    KeypointSchema,
    format_yolo_pose_line,
    keypoints_to_bbox,
    normalize_coords,
    write_yolo_label,
)

# Default 27-keypoint mouse schema from Lightning Pose
MOUSE_LP_27 = KeypointSchema(
    names=[
        "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
        "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
        "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
        "tail1", "tail2", "tail3", "tail4", "tail5", "left_shoulder",
        "left_midside", "left_hip", "right_shoulder", "right_midside",
        "right_hip", "tail_end", "head_midpoint",
    ],
    skeleton=[
        # Head
        (0, 26), (26, 5), (26, 6), (1, 3), (2, 4),
        # Spine
        (0, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        # Tail
        (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 25),
        # Left side
        (7, 19), (19, 20), (20, 21),
        # Right side
        (7, 22), (22, 23), (23, 24),
    ],
)


def _parse_lp_csv(csv_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Parse a Lightning Pose multi-header CSV.

    Returns
    -------
    df : DataFrame
        Columns are (bodypart, coord) MultiIndex.  Index is frame number.
    bodypart_names : list[str]
        Ordered unique bodypart names as they appear in the CSV.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    # The top-level header is the scorer name — drop it to get (bodypart, coord)
    df.columns = df.columns.droplevel(0)
    # Deduplicate bodypart names preserving order
    seen = set()
    bodypart_names = []
    for bp, _ in df.columns:
        if bp not in seen:
            seen.add(bp)
            bodypart_names.append(bp)
    return df, bodypart_names


def _extract_keypoints_for_frame(
    df: pd.DataFrame,
    frame_idx: int,
    selected_bodyparts: list[str],
    img_w: int,
    img_h: int,
    confidence_threshold: float,
    bbox_margin: float,
    class_id: int,
) -> str | None:
    """Extract keypoints for a single frame and return a YOLO pose label line.

    Returns None if the frame is not in the CSV or the bounding box is
    degenerate (all keypoints invalid).
    """
    if frame_idx not in df.index:
        return None

    row = df.loc[frame_idx]
    n_kps = len(selected_bodyparts)
    kps_xy = np.zeros((n_kps, 2), dtype=np.float64)
    kps_conf = np.zeros(n_kps, dtype=np.float64)

    for i, bp in enumerate(selected_bodyparts):
        kps_xy[i, 0] = row[(bp, "x")]
        kps_xy[i, 1] = row[(bp, "y")]
        kps_conf[i] = row[(bp, "likelihood")]

    bbox = keypoints_to_bbox(kps_xy, img_w, img_h, margin=bbox_margin)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None

    kps_xyv = []
    n_visible = 0
    for i in range(n_kps):
        nx, ny = normalize_coords(kps_xy[i, 0], kps_xy[i, 1], img_w, img_h)
        vis = 2 if kps_conf[i] >= confidence_threshold else 0
        n_visible += vis > 0
        kps_xyv.append((nx, ny, vis))

    # Skip frames where no keypoints pass the confidence threshold —
    # the bbox is unreliable when computed from low-confidence coordinates.
    if n_visible == 0:
        return None

    return format_yolo_pose_line(class_id, bbox, kps_xyv)


def convert_lightning_pose(
    csv_path: str | Path,
    extracted_frames: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    img_w: int,
    img_h: int,
    keypoint_indices: Sequence[int] | None = None,
    confidence_threshold: float = 0.5,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    class_id: int = 0,
    bbox_margin: float = 0.1,
    symlink_images: bool = True,
    seed: int = 42,
) -> KeypointSchema:
    """Convert LP CSV to YOLO pose labels using pre-extracted frames.

    Instead of re-decoding the video, this function takes the frame records
    produced by ``behavior.media.extract_frames`` (via the manifest's
    ``files`` list or ``FrameExtractionResult.files``).  For each frame
    that has a matching row in the LP CSV it writes a YOLO pose label and
    symlinks (or copies) the already-extracted image into the YOLO dataset
    tree.

    Parameters
    ----------
    csv_path : path
        Lightning Pose CSV file.
    extracted_frames : list of dict
        Each dict must have ``"frame_index"`` (int) and ``"path"`` (str)
        keys, plus ``"width"`` and ``"height"`` (optional — falls back to
        *img_w* / *img_h*).  This is the ``files`` field from a
        ``FrameExtractionResult`` or ``load_extraction_manifest()``.
    output_dir : path
        Root directory for the YOLO dataset output.
    img_w, img_h : int
        Image dimensions (from ``video_meta`` in the manifest).
    keypoint_indices : sequence of int, optional
        Indices into the CSV's bodypart list to include.  None = all.
    confidence_threshold : float
        Minimum likelihood to mark a keypoint as visible (vis=2).
        Below this threshold the keypoint is marked invisible (vis=0).
    split : (train, valid, test) floats
        Fraction of frames per split.  Must sum to ~1.0.
    class_id : int
        YOLO class ID for the animal (typically 0).
    bbox_margin : float
        Fractional margin around keypoints for bounding box.
    symlink_images : bool
        If True (default), create symlinks to the original extracted frame
        images.  If False, copy them.
    seed : int
        Random seed for train/valid/test assignment.

    Returns
    -------
    KeypointSchema
        The keypoint schema used (reflects any subsetting via *keypoint_indices*).
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    # Parse CSV
    df, all_bodyparts = _parse_lp_csv(csv_path)
    if keypoint_indices is not None:
        selected_bodyparts = [all_bodyparts[i] for i in keypoint_indices]
    else:
        selected_bodyparts = all_bodyparts
        keypoint_indices = list(range(len(all_bodyparts)))

    schema = KeypointSchema(
        names=list(selected_bodyparts),
        skeleton=MOUSE_LP_27.skeleton if keypoint_indices is None else [],
    )

    # Filter to frames present in both the extraction manifest and the CSV
    csv_frame_set = set(df.index.values)
    usable_frames = [
        rec for rec in extracted_frames
        if int(rec["frame_index"]) in csv_frame_set
    ]
    if not usable_frames:
        print(f"[lightning_pose] WARNING: no overlap between extracted frames "
              f"and LP CSV ({csv_path.name}).  0 labels written.")
        return schema

    # Assign frames to splits
    rng = random.Random(seed)
    shuffled = list(usable_frames)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    split_assignment: dict[int, str] = {}
    for rec in shuffled[:n_train]:
        split_assignment[int(rec["frame_index"])] = "train"
    for rec in shuffled[n_train:n_train + n_valid]:
        split_assignment[int(rec["frame_index"])] = "valid"
    for rec in shuffled[n_train + n_valid:]:
        split_assignment[int(rec["frame_index"])] = "test"

    # Create output directories
    for subset in ("train", "valid", "test"):
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    # Process each extracted frame
    written = 0
    skipped = 0
    for rec in usable_frames:
        frame_idx = int(rec["frame_index"])
        src_image = Path(rec["path"])
        w = int(rec.get("width", img_w))
        h = int(rec.get("height", img_h))

        line = _extract_keypoints_for_frame(
            df, frame_idx, selected_bodyparts,
            w, h, confidence_threshold, bbox_margin, class_id,
        )
        if line is None:
            skipped += 1
            continue

        subset = split_assignment.get(frame_idx, "train")
        stem = f"frame_{frame_idx:08d}"

        # Write YOLO label
        write_yolo_label(output_dir / subset / "labels" / f"{stem}.txt", [line])

        # Link or copy the already-extracted image
        dest_image = output_dir / subset / "images" / (stem + src_image.suffix)
        if dest_image.exists() or dest_image.is_symlink():
            dest_image.unlink()
        if symlink_images:
            dest_image.symlink_to(src_image.resolve())
        else:
            shutil.copy2(src_image, dest_image)

        written += 1

    # Remove empty test split if no frames assigned
    test_imgs = output_dir / "test" / "images"
    if test_imgs.exists() and not any(test_imgs.iterdir()):
        shutil.rmtree(output_dir / "test")

    print(f"[lightning_pose] Wrote {written} labels to {output_dir}"
          + (f"  (skipped {skipped} degenerate)" if skipped else ""))
    print(f"  Keypoints: {len(selected_bodyparts)} ({', '.join(selected_bodyparts[:5])}...)")
    print(f"  Splits: train={n_train}, valid={n_valid}, test={n - n_train - n_valid}")

    return schema
