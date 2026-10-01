"""Heatmap inference and peak detection for the localizer model.

Runs the localizer encoder on full images, detects peaks in the
sigmoid-activated heatmap, applies subpixel refinement, and converts
detections to image-pixel coordinates.

Requires: ``torch >= 2.0``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import cv2
import numpy as np
import pandas as pd
from mosaic_media import MediaFacts
from scipy.ndimage import maximum_filter

from mosaic.core.media.video_io import entry_paths, read_entry_frames
from mosaic.optional_dependency import require
from mosaic.tracking.external.runner.ultralytics_protocol import (
    POINT_COLUMNS,
    POINT_DTYPES,
)

if TYPE_CHECKING:
    from .localizer_model import LocalizerEncoder


class LocalizerDetection(TypedDict):
    """One location the localizer detected, in image pixel coordinates."""

    x: float
    y: float
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class LocalizerFrame:
    """The locations that the localizer detected in one frame that it read.

    Attributes:
        frame: The frame's index in the video, or across an entry's clips when
            it read several.
        detections: The locations detected in the frame, which may be none.
    """

    frame: int
    detections: tuple[LocalizerDetection, ...]


def _require_torch() -> ModuleType:
    return require("torch", "deep-learning", "localizer inference")


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
) -> list[LocalizerDetection]:
    """Detect animal locations in a single image.

    Parameters
    ----------
    model : LocalizerEncoder
        Localizer model (must be in eval mode).  Automatically moved to
        *device* if not already there.
    image : ndarray
        BGR or grayscale image.
    thresholds : dict or float
        Per-class detection thresholds ``{class_id: threshold}``, or a
        single threshold applied to all classes.
    device : str
        Device string for inference (e.g. ``"0"`` for first GPU, ``"cpu"``).
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
    model = model.to(dev)

    # Forward pass
    with torch.no_grad():
        heatmap = model(inp)  # (1, C, H', W')
    heatmap = heatmap[0].cpu().numpy()  # (C, H', W')

    # Normalize thresholds
    num_classes = heatmap.shape[0]
    if isinstance(thresholds, (int, float)):
        thresholds = {i: float(thresholds) for i in range(num_classes)}

    detections: list[LocalizerDetection] = []
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

            detections.append(
                {
                    "x": x_img,
                    "y": y_img,
                    "confidence": confidence,
                    "class_id": class_id,
                }
            )

    return detections


# --------------------------------------------------------------------------- #
# Video inference
# --------------------------------------------------------------------------- #


def run_localizer_inference(
    model_path: str | Path,
    video_paths: str | Path | Sequence[Path],
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
    facts: Sequence[MediaFacts] | None = None,
) -> list[LocalizerFrame]:
    """Run localizer inference on a video, or on an entry's clips as one video.

    Parameters
    ----------
    model_path : path
        Path to trained ``.pt`` or ``.h5`` model weights.
    video_paths : path or sequence of path
        The input video, or an entry's clips in order. A frame is numbered by its
        index counted across them (:func:`read_entry_frames`).
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
    facts : sequence of MediaFacts, optional
        The probed facts of *video_paths*, parallel to them, as an index row or
        a probe records them. They are gated for analysis before any frame is
        read, and each file is probed for them when they are not given.

    Returns
    -------
    list of LocalizerFrame
        The detections of each frame read, in order, each with the frame's index.
        Under ``start_frame`` or ``frame_step`` that is the index of the frame
        read, not its place among the frames read.
    """
    encoder = _load_encoder(
        model_path,
        num_classes=num_classes,
        initial_channels=initial_channels,
        device=device,
    )

    # Imgstore-aware, and one frame axis over several clips.
    paths = entry_paths(video_paths)
    frames = read_entry_frames(
        paths,
        facts=facts,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        target="analysis",
    )

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default palette
    if class_colors is None:
        palette = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        class_colors = {i: palette[i % len(palette)] for i in range(num_classes)}

    all_results: list[LocalizerFrame] = []
    processed = 0

    try:
        for frame_idx, frame in frames:
            detections = detect_locations(
                encoder,
                frame,
                thresholds,
                device=device,
                min_distance=min_distance,
                refine_window=refine_window,
            )
            all_results.append(LocalizerFrame(frame_idx, tuple(detections)))

            if save_images and output_dir is not None:
                annotated = frame.copy()
                for det in detections:
                    pt = (int(round(det["x"])), int(round(det["y"])))
                    color = class_colors.get(det["class_id"], (0, 255, 0))
                    cv2.circle(annotated, pt, point_radius, color, -1)
                fname = f"frame_{frame_idx:08d}.jpg"
                cv2.imwrite(str(Path(output_dir) / fname), annotated)

            processed += 1

            if max_frames is not None and processed >= max_frames:
                break
    finally:
        frames.close()

    names = ", ".join(path.name for path in paths)
    print(f"[localizer_inference] Processed {processed} frames from {names}")

    return all_results


def _load_encoder(
    model_path: str | Path, *, num_classes: int, initial_channels: int, device: str
) -> LocalizerEncoder:
    """Return the localizer network with *model_path*'s weights, ready to run.

    The network is on *device*, a GPU index or ``"cpu"``, and in eval mode. A GPU
    index falls back to the CPU when CUDA is not available.
    """
    torch = _require_torch()
    from .localizer_model import LocalizerEncoder
    from .localizer_weights import load_localizer_weights

    encoder = LocalizerEncoder(
        num_classes=num_classes, initial_channels=initial_channels
    )
    load_localizer_weights(encoder, model_path)

    if device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    encoder.to(dev)
    encoder.eval()
    return encoder


# --------------------------------------------------------------------------- #
# DataFrame conversion
# --------------------------------------------------------------------------- #


def localizer_detections_to_dataframe(
    results: Sequence[LocalizerFrame],
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert localizer detection results to a DataFrame.

    Parameters
    ----------
    results : sequence of LocalizerFrame
        Per-frame detections from :func:`run_localizer_inference`. Each row's
        ``frame`` is the ``frame`` of its :class:`LocalizerFrame`.
    class_names : list of str, optional
        Human-readable class names.

    Returns
    -------
    DataFrame
        The columns of :data:`POINT_COLUMNS`, typed by :data:`POINT_DTYPES`,
        whether or not any frame contains a detection. Frames without a detection
        give an empty table that still names every column.
    """
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        for det_idx, det in enumerate(result.detections):
            class_id = det["class_id"]
            rows.append(
                {
                    "frame": result.frame,
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
                }
            )
    return pd.DataFrame(rows, columns=list(POINT_COLUMNS)).astype(POINT_DTYPES)
