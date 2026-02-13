"""Heatmap inference and peak detection for the localizer model.

Runs the localizer encoder on full images, detects peaks in the
sigmoid-activated heatmap, applies subpixel refinement, and converts
detections to image-pixel coordinates.

Requires: ``torch >= 2.0``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for localizer inference. "
            "Install with: pip install mosaic-behavior[localizer]"
        )


# --------------------------------------------------------------------------- #
# Single-image detection
# --------------------------------------------------------------------------- #

def detect_locations(
    model: Any,
    image: np.ndarray,
    thresholds: dict[int, float] | float = 0.5,
    *,
    device: str = "cpu",
    min_distance: int = 3,
    refine_window: int = 7,
) -> list[dict]:
    """Detect animal locations in a single image.

    Parameters
    ----------
    model : LocalizerEncoder
        Localizer model (must be in eval mode on the correct device).
    image : ndarray
        BGR or grayscale image.
    thresholds : dict or float
        Per-class detection thresholds ``{class_id: threshold}``, or a
        single threshold applied to all classes.
    device : str
        Device string (used only for input tensor placement).
    min_distance : int
        Minimum distance between peaks in heatmap pixels.
    refine_window : int
        Window size for subpixel center-of-mass refinement.

    Returns
    -------
    list of dict
        Each dict: ``{x, y, confidence, class_id}`` in image pixel coords.
    """
    torch = _require_torch()
    from .localizer_model import LocalizerEncoder

    STRIDE = LocalizerEncoder.STRIDE
    OFFSET = LocalizerEncoder.OFFSET

    # Prepare input
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    inp = gray.astype(np.float32) / 255.0
    inp = torch.from_numpy(inp[np.newaxis, np.newaxis])  # (1, 1, H, W)

    if device != "cpu":
        dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    inp = inp.to(dev)

    # Forward pass
    with torch.no_grad():
        heatmap = model(inp)  # (1, C, H', W')
    heatmap = heatmap[0].cpu().numpy()  # (C, H', W')

    # Normalize thresholds
    num_classes = heatmap.shape[0]
    if isinstance(thresholds, (int, float)):
        thresholds = {i: float(thresholds) for i in range(num_classes)}

    detections: list[dict] = []
    half_w = refine_window // 2

    for class_id in range(num_classes):
        ch = heatmap[class_id]
        thresh = thresholds.get(class_id, 0.5)

        # Peak detection: local maximum filter + threshold
        local_max = maximum_filter(ch, size=2 * min_distance + 1)
        peaks = (ch == local_max) & (ch >= thresh)

        peak_coords = np.argwhere(peaks)  # (N, 2) — [row, col]

        for row, col in peak_coords:
            confidence = float(ch[row, col])

            # Subpixel refinement via center-of-mass in a local window
            h_map, w_map = ch.shape
            r0 = max(0, row - half_w)
            r1 = min(h_map, row + half_w + 1)
            c0 = max(0, col - half_w)
            c1 = min(w_map, col + half_w + 1)

            window = ch[r0:r1, c0:c1]
            if window.sum() > 0:
                rows_idx = np.arange(r0, r1)
                cols_idx = np.arange(c0, c1)
                row_refined = float(np.average(rows_idx, weights=window.sum(axis=1)))
                col_refined = float(np.average(cols_idx, weights=window.sum(axis=0)))
            else:
                row_refined = float(row)
                col_refined = float(col)

            # Convert heatmap coordinates → image pixel coordinates
            x_img = col_refined * STRIDE + OFFSET
            y_img = row_refined * STRIDE + OFFSET

            detections.append({
                "x": x_img,
                "y": y_img,
                "confidence": confidence,
                "class_id": class_id,
            })

    return detections


# --------------------------------------------------------------------------- #
# Video inference
# --------------------------------------------------------------------------- #

def run_localizer_inference(
    model_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    num_classes: int = 4,
    initial_channels: int = 32,
    thresholds: dict[int, float] | float = 0.5,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
    device: str = "0",
    save_images: bool = True,
    min_distance: int = 3,
    refine_window: int = 7,
    point_radius: int = 4,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
) -> list[list[dict]]:
    """Run localizer inference on a video.

    Parameters
    ----------
    model_path : path
        Path to trained ``.pt`` or ``.h5`` model weights.
    video_path : path
        Path to input video.
    output_dir : path, optional
        Where to save annotated frames.
    num_classes : int
        Number of output heatmap channels.
    initial_channels : int
        Base channel width.
    thresholds : dict or float
        Detection thresholds per class.
    start_frame, end_frame, frame_step : int
        Frame selection parameters.
    max_frames : int, optional
        Stop after this many processed frames.
    device : str
        Device for inference.
    save_images : bool
        Save annotated frames to *output_dir*.
    min_distance : int
        Minimum peak distance in heatmap pixels.
    refine_window : int
        Subpixel refinement window size.
    point_radius : int
        Radius of drawn detection points (for visualization).
    class_colors : dict, optional
        ``{class_id: (B, G, R)}`` color mapping for visualization.

    Returns
    -------
    list of list of dict
        Per-frame detection lists.
    """
    torch = _require_torch()
    from .localizer_model import LocalizerEncoder
    from .localizer_weights import load_localizer_weights

    # Load model
    encoder = LocalizerEncoder(num_classes=num_classes, initial_channels=initial_channels)
    load_localizer_weights(encoder, model_path)

    if device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    encoder.to(dev)
    encoder.eval()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(raw_count) if 0 < raw_count < 1e12 else None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default palette
    if class_colors is None:
        palette = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        class_colors = {i: palette[i % len(palette)] for i in range(num_classes)}

    all_results: list[list[dict]] = []
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

        detections = detect_locations(
            encoder, frame, thresholds,
            device=device,
            min_distance=min_distance,
            refine_window=refine_window,
        )
        all_results.append(detections)

        if save_images and output_dir is not None:
            annotated = frame.copy()
            for det in detections:
                pt = (int(round(det["x"])), int(round(det["y"])))
                color = class_colors.get(det["class_id"], (0, 255, 0))
                cv2.circle(annotated, pt, point_radius, color, -1)
            fname = f"frame_{frame_idx:08d}.jpg"
            cv2.imwrite(str(Path(output_dir) / fname), annotated)

        processed += 1
        frame_idx += 1

        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    total_str = str(total_frames) if total_frames is not None else "?"
    print(
        f"[localizer_inference] Processed {processed}/{total_str} "
        f"frames from {video_path}"
    )

    return all_results


# --------------------------------------------------------------------------- #
# DataFrame conversion
# --------------------------------------------------------------------------- #

def localizer_detections_to_dataframe(
    results: list[list[dict]],
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert localizer detection results to a DataFrame.

    Parameters
    ----------
    results : list of list of dict
        Per-frame detection lists from :func:`run_localizer_inference`.
    class_names : list of str, optional
        Human-readable class names.

    Returns
    -------
    DataFrame
        Columns: ``frame, detection_id, x, y, confidence, class_id, class_name``.
    """
    rows = []
    for frame_idx, detections in enumerate(results):
        for det_idx, det in enumerate(detections):
            class_id = det["class_id"]
            rows.append({
                "frame": frame_idx,
                "detection_id": det_idx,
                "x": det["x"],
                "y": det["y"],
                "confidence": det["confidence"],
                "class_id": class_id,
                "class_name": (
                    class_names[class_id]
                    if class_names and class_id < len(class_names)
                    else f"class_{class_id}"
                ),
            })
    return pd.DataFrame(rows)
