"""Test inference for trained YOLO pose and POLO point-detection models.

This is test/development code for evaluating trained models on video.
The production pose inference pipeline is handled by TRex.

Requires:
    Pose:  pip install ultralytics
    POLO:  pip install git+https://github.com/mooch443/POLO.git
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
    start_frame: int = 0,
    end_frame: int | None = None,
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
    start_frame : int
        Frame index to start inference from (default 0).
    end_frame : int, optional
        Frame index to stop at (exclusive). None = run to end of video.
    frame_step : int
        Process every Nth frame (relative to start_frame).
    conf_threshold : float
        Minimum detection confidence.
    max_frames : int, optional
        Stop after this many processed frames.
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

    raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(raw_count) if 0 < raw_count < 1e12 else None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    frame_idx = start_frame
    processed = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_step != 0:
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
    total_str = str(total_frames) if total_frames is not None else "?"
    print(f"[inference] Processed {processed}/{total_str} frames from {video_path}")

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


# --------------------------------------------------------------------------- #
# POLO point-detection inference
# --------------------------------------------------------------------------- #

def _require_polo():
    """Import YOLO from a POLO fork and verify the 'locate' task is available."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "A POLO-compatible ultralytics fork is required for point "
            "inference.  Install with:\n"
            "  pip install git+https://github.com/mooch443/POLO.git"
        )
    try:
        from ultralytics.nn.tasks import LocalizationModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "Your ultralytics installation does not support the 'locate' task. "
            "Install the POLO fork:\n"
            "  pip install git+https://github.com/mooch443/POLO.git"
        )
    return YOLO


def run_point_inference(
    model_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    conf_threshold: float = 0.25,
    dor: float = 0.8,
    radii: dict[int, float] | None = None,
    max_frames: int | None = None,
    device: str = "0",
    save_images: bool = True,
    imgsz: int = 640,
) -> list[Any]:
    """Run POLO point-detection inference on a video.

    Parameters
    ----------
    model_path : path
        Path to trained POLO ``.pt`` model.
    video_path : path
        Path to input video.
    output_dir : path, optional
        Where to save annotated frames.
    start_frame, end_frame, frame_step : int
        Frame selection parameters.
    conf_threshold : float
        Minimum detection confidence.
    dor : float
        Distance of Reference threshold for post-processing.
    radii : dict, optional
        Override radii. ``{class_id: radius_px}``.
    max_frames : int, optional
        Stop after this many processed frames.
    device : str
        Device for inference.
    save_images : bool
        If True and output_dir set, save annotated frames.
    imgsz : int
        Inference image size.

    Returns
    -------
    list
        List of ultralytics Results objects with ``.locations`` attribute.
    """
    YOLO = _require_polo()
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(raw_count) if 0 < raw_count < 1e12 else None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    predict_kwargs: dict[str, Any] = dict(
        device=device,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    if radii is not None:
        predict_kwargs["radii"] = radii

    all_results = []
    frame_idx = start_frame
    processed = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_step != 0:
            frame_idx += 1
            continue

        results = model.predict(source=frame, **predict_kwargs)
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
    total_str = str(total_frames) if total_frames is not None else "?"
    print(f"[point_inference] Processed {processed}/{total_str} frames from {video_path}")

    return all_results


def locations_to_dataframe(results: list[Any]) -> pd.DataFrame:
    """Convert POLO inference results to a DataFrame.

    Parameters
    ----------
    results : list
        List of ultralytics Results objects with ``.locations`` attribute.

    Returns
    -------
    DataFrame
        Columns: frame, detection_id, x, y, confidence, class_id, class_name.
    """
    rows = []
    for frame_idx, result in enumerate(results):
        locs = getattr(result, "locations", None)
        if locs is None:
            continue
        locs_data = getattr(locs, "data", None)
        if locs_data is None or len(locs_data) == 0:
            continue

        locs_np = locs_data.cpu().numpy() if hasattr(locs_data, "cpu") else np.asarray(locs_data)
        names = getattr(result, "names", {})

        for det_idx in range(locs_np.shape[0]):
            # POLO locations: [x, y, conf, cls] or [x, y, track_id, conf, cls]
            if locs_np.shape[1] >= 5:
                x, y, _track, conf, cls = locs_np[det_idx, :5]
            else:
                x, y, conf, cls = locs_np[det_idx, :4]

            rows.append({
                "frame": frame_idx,
                "detection_id": det_idx,
                "x": float(x),
                "y": float(y),
                "confidence": float(conf),
                "class_id": int(cls),
                "class_name": names.get(int(cls), f"class_{int(cls)}"),
            })

    return pd.DataFrame(rows)
