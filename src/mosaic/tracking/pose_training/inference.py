"""Test inference for trained YOLO pose models.

This is test/development code for evaluating trained pose models on video.
The production pose inference pipeline is handled by TRex.

Requires: pip install ultralytics
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for inference. "
            "Install it with: pip install ultralytics"
        )


def run_inference(
    model_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    frame_step: int = 1,
    conf_threshold: float = 0.25,
    max_frames: int | None = None,
    device: str = "0",
    save_images: bool = True,
    imgsz: int = 640,
) -> list[Any]:
    """Run pose inference on a video and optionally save annotated frames.

    Parameters
    ----------
    model_path : path
        Path to trained .pt model.
    video_path : path
        Path to input video.
    output_dir : path, optional
        Where to save annotated frames.  None = don't save.
    frame_step : int
        Process every Nth frame.
    conf_threshold : float
        Minimum detection confidence.
    max_frames : int, optional
        Stop after this many frames.
    device : str
        Device for inference.
    save_images : bool
        If True and output_dir set, save annotated frames.
    imgsz : int
        Inference image size.

    Returns
    -------
    list
        List of ultralytics Results objects, one per processed frame.
    """
    YOLO = _require_ultralytics()
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        results = model.predict(
            source=frame,
            device=device,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )
        all_results.extend(results)

        if save_images and output_dir is not None:
            annotated = results[0].plot()
            fname = f"frame_{frame_idx:08d}.jpg"
            cv2.imwrite(str(Path(output_dir) / fname), annotated)

        processed += 1
        frame_idx += 1

        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    print(f"[inference] Processed {processed}/{total_frames} frames from {video_path}")

    return all_results


def visualize_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list[tuple[int, int]] | None = None,
    *,
    confidence: np.ndarray | None = None,
    conf_threshold: float = 0.3,
    point_radius: int = 4,
    point_color: tuple[int, int, int] = (0, 255, 0),
    line_color: tuple[int, int, int] = (255, 255, 0),
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw keypoints and skeleton on a frame.

    Parameters
    ----------
    frame : ndarray, shape (H, W, 3)
        BGR image.
    keypoints : ndarray, shape (N, 2) or (N, 3)
        Keypoint coordinates in pixels.  If shape (N, 3), third column is confidence.
    skeleton : list of (i, j), optional
        Pairs of keypoint indices to connect with lines.
    confidence : ndarray, shape (N,), optional
        Per-keypoint confidence (overrides 3rd column of keypoints if present).
    conf_threshold : float
        Only draw keypoints with confidence above this.

    Returns
    -------
    ndarray
        Annotated frame (copy).
    """
    out = frame.copy()
    n_kps = keypoints.shape[0]

    if confidence is None and keypoints.shape[1] >= 3:
        confidence = keypoints[:, 2]
    if confidence is None:
        confidence = np.ones(n_kps)

    # Draw skeleton lines first (behind points)
    if skeleton:
        for i, j in skeleton:
            if i >= n_kps or j >= n_kps:
                continue
            if confidence[i] < conf_threshold or confidence[j] < conf_threshold:
                continue
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(out, pt1, pt2, line_color, line_thickness)

    # Draw keypoints
    for k in range(n_kps):
        if confidence[k] < conf_threshold:
            continue
        pt = (int(keypoints[k, 0]), int(keypoints[k, 1]))
        cv2.circle(out, pt, point_radius, point_color, -1)

    return out


def inference_to_dataframe(results: list[Any]) -> pd.DataFrame:
    """Convert YOLO inference results to a DataFrame.

    Extracts keypoints from each result frame and produces a DataFrame
    with columns compatible with the trex_v1 schema (poseX{k}, poseY{k}, poseP{k}).

    Parameters
    ----------
    results : list
        List of ultralytics Results objects (one per frame).

    Returns
    -------
    DataFrame
        Columns: frame, id, poseX0, poseY0, poseP0, poseX1, poseY1, poseP1, ...
    """
    rows = []
    for frame_idx, result in enumerate(results):
        if result.keypoints is None:
            continue
        kps_data = result.keypoints.data  # (num_detections, num_keypoints, 3)
        if kps_data is None or len(kps_data) == 0:
            continue

        kps_np = kps_data.cpu().numpy() if hasattr(kps_data, "cpu") else np.asarray(kps_data)

        for det_idx in range(kps_np.shape[0]):
            row: dict[str, Any] = {
                "frame": frame_idx,
                "id": det_idx,
            }
            n_kps = kps_np.shape[1]
            for k in range(n_kps):
                row[f"poseX{k}"] = float(kps_np[det_idx, k, 0])
                row[f"poseY{k}"] = float(kps_np[det_idx, k, 1])
                row[f"poseP{k}"] = float(kps_np[det_idx, k, 2]) if kps_np.shape[2] > 2 else 1.0
            rows.append(row)

    return pd.DataFrame(rows)
