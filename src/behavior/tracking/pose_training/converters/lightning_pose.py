"""Convert Lightning Pose CSV output to YOLO pose format.

Lightning Pose CSVs use a DeepLabCut-style multi-header format:
    Row 0: scorer (e.g. "heatmap_mhcrnn_tracker")
    Row 1: bodypart names (e.g. "nose", "left_ear", ...)
    Row 2: coordinate type ("x", "y", "likelihood")
    Row 3+: frame_index, then triplets of (x, y, likelihood) per keypoint

This converter extracts frames from the corresponding video, writes each
frame as an image, and produces YOLO pose label files alongside them.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Sequence

import cv2
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


def convert_lightning_pose(
    csv_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    *,
    keypoint_indices: Sequence[int] | None = None,
    confidence_threshold: float = 0.5,
    frame_step: int = 1,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    class_id: int = 0,
    bbox_margin: float = 0.1,
    image_ext: str = ".jpg",
    seed: int = 42,
) -> KeypointSchema:
    """Convert a Lightning Pose CSV + video to a YOLO pose dataset.

    Parameters
    ----------
    csv_path : path
        Lightning Pose CSV file.
    video_path : path
        Corresponding video file (used to extract frames and get dimensions).
    output_dir : path
        Root directory for the YOLO dataset output.
    keypoint_indices : sequence of int, optional
        Indices into the CSV's bodypart list to include.  None = all.
    confidence_threshold : float
        Minimum likelihood to mark a keypoint as visible (vis=2).
        Below this threshold: vis=0.
    frame_step : int
        Sample every Nth frame.  1 = every frame.
    split : (train, valid, test) floats
        Fraction of frames per split.  Must sum to ~1.0.
    class_id : int
        YOLO class ID for the animal (typically 0).
    bbox_margin : float
        Fractional margin around keypoints for bounding box.
    image_ext : str
        Image format for saved frames.
    seed : int
        Random seed for train/valid/test assignment.

    Returns
    -------
    KeypointSchema
        The keypoint schema used (reflects any subsetting via keypoint_indices).
    """
    csv_path = Path(csv_path)
    video_path = Path(video_path)
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

    # Open video to get dimensions
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine which frames to process
    csv_frames = df.index.values
    sampled_frames = csv_frames[::frame_step]

    # Assign frames to splits
    rng = random.Random(seed)
    frame_list = list(sampled_frames)
    rng.shuffle(frame_list)
    n = len(frame_list)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    split_assignment = {}
    for f in frame_list[:n_train]:
        split_assignment[f] = "train"
    for f in frame_list[n_train:n_train + n_valid]:
        split_assignment[f] = "valid"
    for f in frame_list[n_train + n_valid:]:
        split_assignment[f] = "test"

    # Create output directories
    for subset in ("train", "valid", "test"):
        (output_dir / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / subset / "labels").mkdir(parents=True, exist_ok=True)

    # Process frames
    written = 0
    for frame_idx in sorted(sampled_frames):
        if frame_idx >= total_video_frames:
            break

        # Read frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Extract keypoints for this frame
        row = df.loc[frame_idx]
        kps_xy = np.zeros((len(selected_bodyparts), 2), dtype=np.float64)
        kps_conf = np.zeros(len(selected_bodyparts), dtype=np.float64)

        for i, bp in enumerate(selected_bodyparts):
            kps_xy[i, 0] = row[(bp, "x")]
            kps_xy[i, 1] = row[(bp, "y")]
            kps_conf[i] = row[(bp, "likelihood")]

        # Compute bounding box
        bbox = keypoints_to_bbox(kps_xy, img_w, img_h, margin=bbox_margin)

        # Skip if bbox is degenerate (all keypoints invalid)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue

        # Format keypoints as normalized (x, y, visibility)
        kps_xyv = []
        for i in range(len(selected_bodyparts)):
            nx, ny = normalize_coords(kps_xy[i, 0], kps_xy[i, 1], img_w, img_h)
            vis = 2 if kps_conf[i] >= confidence_threshold else 0
            kps_xyv.append((nx, ny, vis))

        # Write label
        line = format_yolo_pose_line(class_id, bbox, kps_xyv)
        subset = split_assignment.get(frame_idx, "train")
        stem = f"frame_{int(frame_idx):08d}"

        write_yolo_label(output_dir / subset / "labels" / f"{stem}.txt", [line])

        # Write image
        img_path = output_dir / subset / "images" / f"{stem}{image_ext}"
        cv2.imwrite(str(img_path), frame)
        written += 1

    cap.release()

    # Remove empty test split if no frames assigned
    test_imgs = output_dir / "test" / "images"
    if test_imgs.exists() and not any(test_imgs.iterdir()):
        shutil.rmtree(output_dir / "test")

    print(f"[lightning_pose] Wrote {written} frames to {output_dir}")
    print(f"  Keypoints: {len(selected_bodyparts)} ({', '.join(selected_bodyparts[:5])}...)")
    print(f"  Splits: train={n_train}, valid={n_valid}, test={n - n_train - n_valid}")

    return schema
