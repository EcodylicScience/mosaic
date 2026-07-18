"""Test inference for trained YOLO pose and POLO point-detection models.

This is test/development code for evaluating trained models on video.
The production pose inference pipeline is handled by TRex.

Requires:
    Pose:  pip install ultralytics
    POLO:  pip install git+https://github.com/mooch443/POLO.git
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pandas as pd
from mosaic_media import MediaFacts, MediaProbeError
from mosaic_media.io import FFmpegVideoWriter

from mosaic.core.media.video_io import (
    FrameReader,
    get_video_metadata,
    open_frame_reader,
    prefetch_batches,
)


class _InferenceResult(Protocol):
    """The ultralytics ``Results`` surface used while consuming a batch."""

    def plot(self) -> np.ndarray: ...


class _InferenceModel(Protocol):
    """The ultralytics model surface the batched predict loop calls."""

    def predict(
        self, source: list[np.ndarray], **kwargs: object
    ) -> list[_InferenceResult]: ...


class _ProgressBar(Protocol):
    """The tqdm progress-bar surface the batched predict loop advances."""

    def update(self, n: int) -> object: ...


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for inference. "
            "Install it with: pip install ultralytics"
        )


def _run_predict_loop(
    model: _InferenceModel,
    reader: FrameReader,
    predict_kwargs: dict[str, Any],
    *,
    output_dir: str | Path | None,
    save_images: bool,
    max_frames: int | None,
    batch_size: int,
    prefetch: bool,
    pbar: _ProgressBar | None,
) -> tuple[list[Any], int]:
    """Read frames from *reader* in batches, call ``model.predict`` per batch,
    and optionally save annotated frames.

    Shared by every inference entry point (pose and point-detection, single-frame
    and throughput-batched): the reader's decode-time resize/facts, the batch
    size, and the prefetch cadence are supplied by the caller and are the only
    things that differ between them. A single-frame call site passes
    ``batch_size=1, prefetch=False``, which reproduces the same per-frame
    predict/save behavior as reading one frame at a time.
    """
    all_results: list[Any] = []
    processed = 0

    def _consume_batch(indices: np.ndarray, batch_frames: np.ndarray) -> bool:
        """Predict and save one batch; return True once max_frames is reached."""
        nonlocal processed
        if max_frames is not None:
            remaining = max_frames - processed
            if remaining <= 0:
                return True
            if len(indices) > remaining:
                indices = indices[:remaining]
                batch_frames = batch_frames[:remaining]

        frames_list = [batch_frames[i] for i in range(len(indices))]
        results = model.predict(source=frames_list, **predict_kwargs)
        all_results.extend(results)

        if save_images and output_dir is not None:
            for i, result in enumerate(results):
                annotated = result.plot()
                fname = f"frame_{indices[i]:08d}.jpg"
                cv2.imwrite(str(Path(output_dir) / fname), annotated)

        processed += len(indices)
        if pbar is not None:
            pbar.update(len(indices))
        return max_frames is not None and processed >= max_frames

    if prefetch:
        frame_queue: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(
            maxsize=2
        )
        worker = threading.Thread(
            target=prefetch_batches,
            args=(reader, frame_queue, batch_size),
            daemon=True,
        )
        worker.start()

        while True:
            item = frame_queue.get()
            if item is None:
                break
            indices, batch_frames = item
            if _consume_batch(indices, batch_frames):
                break

        worker.join(timeout=5)
    else:
        while True:
            indices, batch_frames = reader.read_batch(batch_size)
            if len(indices) == 0:
                break
            if _consume_batch(indices, batch_frames):
                break

    return all_results, processed


def run_inference_opencv(
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
    """Run pose inference on a video, one frame at a time.

    Decodes sequentially through
    :func:`~mosaic.core.media.video_io.open_frame_reader` (in-process libav,
    not OpenCV) at the video's native resolution -- there is no decode-time
    resize, so returned keypoints stay in the source video's pixel coordinate
    space. Use :func:`run_inference` for higher-throughput batched decoding
    (which resizes to fit ``imgsz`` at decode time, so its returned coordinates
    are in that resized space instead).

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

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    predict_kwargs: dict[str, Any] = dict(
        device=device,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    reader = open_frame_reader(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
    )
    total_str = str(reader.frame_count)

    try:
        all_results, processed = _run_predict_loop(
            model,
            reader,
            predict_kwargs,
            output_dir=output_dir,
            save_images=save_images,
            max_frames=max_frames,
            batch_size=1,
            prefetch=False,
            pbar=None,
        )
    finally:
        reader.close()

    print(f"[inference] Processed {processed}/{total_str} frames from {video_path}")

    return all_results


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
    batch_size: int = 8,
    prefetch: bool = True,
    verbose: bool = True,
    facts: MediaFacts | None = None,
) -> list[Any]:
    """Run pose inference on a video and optionally save annotated frames.

    Reads frames in-process via :func:`~mosaic.core.media.video_io.open_frame_reader`
    with decode-time resize and batched ``model.predict()`` calls for higher GPU
    utilization.

    Parameters
    ----------
    model_path : path
        Path to trained ``.pt`` model.
    video_path : path
        Path to input video.
    output_dir : path, optional
        Where to save annotated frames.
    start_frame, end_frame, frame_step : int
        Frame selection parameters.
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
    batch_size : int
        Number of frames per batch for ``model.predict()``.
    prefetch : bool
        Use a background thread to read frames ahead of inference.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    list
        List of ultralytics Results objects, one per processed frame.
    """
    YOLO = _require_ultralytics()
    model = YOLO(str(model_path))

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    predict_kwargs: dict[str, Any] = dict(
        device=device,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    # Get video metadata for resize computation and progress bar
    meta = get_video_metadata(video_path)
    resize_dims = _compute_resize(meta.width, meta.height, imgsz)

    # Compute expected total frames for progress bar
    eff_end = min(end_frame, meta.frame_count) if end_frame is not None else meta.frame_count
    expected_frames = max(0, len(range(start_frame, eff_end, frame_step)))
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames)

    # Progress bar
    pbar = None
    if verbose:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=expected_frames, desc="Inference", unit="frame")
        except ImportError:
            pass

    try:
        reader = open_frame_reader(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            resize=resize_dims,
            facts=facts,
        )
        try:
            all_results, processed = _run_predict_loop(
                model,
                reader,
                predict_kwargs,
                output_dir=output_dir,
                save_images=save_images,
                max_frames=max_frames,
                batch_size=batch_size,
                prefetch=prefetch,
                pbar=pbar,
            )
        finally:
            reader.close()
    finally:
        if pbar is not None:
            pbar.close()

    total_str = str(meta.frame_count)
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


def run_point_inference_opencv(
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
    """Run POLO point-detection inference on a video, one frame at a time.

    Decodes sequentially through
    :func:`~mosaic.core.media.video_io.open_frame_reader` (in-process libav,
    not OpenCV) at the video's native resolution -- there is no decode-time
    resize, so returned locations stay in the source video's pixel coordinate
    space. Use :func:`run_point_inference` for higher-throughput batched
    decoding (which resizes to fit ``imgsz`` at decode time, so its returned
    coordinates are in that resized space instead).

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

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    predict_kwargs: dict[str, Any] = dict(
        device=device,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    if radii is not None:
        predict_kwargs["radii"] = radii

    reader = open_frame_reader(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
    )
    total_str = str(reader.frame_count)

    try:
        all_results, processed = _run_predict_loop(
            model,
            reader,
            predict_kwargs,
            output_dir=output_dir,
            save_images=save_images,
            max_frames=max_frames,
            batch_size=1,
            prefetch=False,
            pbar=None,
        )
    finally:
        reader.close()

    print(f"[point_inference] Processed {processed}/{total_str} frames from {video_path}")

    return all_results


def _compute_resize(source_w: int, source_h: int, imgsz: int) -> tuple[int, int]:
    """Compute resize dimensions that fit within imgsz x imgsz, preserving aspect ratio."""
    scale = min(imgsz / source_w, imgsz / source_h)
    if scale >= 1.0:
        return source_w, source_h
    w = max(1, round(source_w * scale))
    h = max(1, round(source_h * scale))
    # Ensure even dimensions for ffmpeg compatibility
    w = w if w % 2 == 0 else w + 1
    h = h if h % 2 == 0 else h + 1
    return w, h


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
    batch_size: int = 8,
    prefetch: bool = True,
    verbose: bool = True,
    facts: MediaFacts | None = None,
) -> list[Any]:
    """Run POLO point-detection inference on a video.

    Reads frames in-process via :func:`~mosaic.core.media.video_io.open_frame_reader`
    with decode-time resize and batched ``model.predict()`` calls for higher GPU
    utilization.

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
        Device for inference (``"0"`` for first GPU, ``"cpu"`` for CPU).
    save_images : bool
        If True and output_dir set, save annotated frames.
    imgsz : int
        Inference image size.
    batch_size : int
        Number of frames per batch for ``model.predict()``.
    prefetch : bool
        Use a background thread to read frames ahead of inference.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    list
        List of ultralytics Results objects with ``.locations`` attribute.
    """
    YOLO = _require_polo()
    model = YOLO(str(model_path))

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

    # Get video metadata for resize computation and progress bar
    meta = get_video_metadata(video_path)
    resize_dims = _compute_resize(meta.width, meta.height, imgsz)

    # Compute expected total frames for progress bar
    eff_end = min(end_frame, meta.frame_count) if end_frame is not None else meta.frame_count
    expected_frames = max(0, len(range(start_frame, eff_end, frame_step)))
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames)

    # Progress bar
    pbar = None
    if verbose:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=expected_frames, desc="Inference", unit="frame")
        except ImportError:
            pass

    try:
        reader = open_frame_reader(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            resize=resize_dims,
            facts=facts,
        )
        try:
            all_results, processed = _run_predict_loop(
                model,
                reader,
                predict_kwargs,
                output_dir=output_dir,
                save_images=save_images,
                max_frames=max_frames,
                batch_size=batch_size,
                prefetch=prefetch,
                pbar=pbar,
            )
        finally:
            reader.close()
    finally:
        if pbar is not None:
            pbar.close()

    total_str = str(meta.frame_count)
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


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

_DEFAULT_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 165, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (128, 0, 255),
    8: (255, 128, 0),
    9: (0, 128, 255),
}


def visualize_detections(
    frame: np.ndarray,
    detections: list[dict],
    *,
    conf_threshold: float = 0.3,
    point_radius: int = 5,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
    show_labels: bool = False,
) -> np.ndarray:
    """Draw point detections on a frame.

    Parameters
    ----------
    frame : ndarray, shape (H, W, 3)
        BGR image.
    detections : list of dict
        Each dict has keys ``x``, ``y``, ``confidence``, and optionally
        ``class_id`` and ``class_name``.
    conf_threshold : float
        Only draw detections with confidence above this.
    point_radius : int
        Radius of detection circles.
    class_colors : dict, optional
        ``{class_id: (B, G, R)}``.  Falls back to a built-in palette.
    show_labels : bool
        If True, draw class name or id next to each detection.

    Returns
    -------
    ndarray
        Annotated frame (copy).
    """
    colors = class_colors or _DEFAULT_CLASS_COLORS
    out = frame.copy()

    for det in detections:
        conf = det.get("confidence", 1.0)
        if conf < conf_threshold:
            continue

        x, y = int(det["x"]), int(det["y"])
        cid = det.get("class_id", 0)
        color = colors.get(cid, (0, 255, 0))

        cv2.circle(out, (x, y), point_radius, color, -1)
        cv2.circle(out, (x, y), point_radius, (0, 0, 0), 1)  # black outline

        if show_labels:
            label = det.get("class_name", str(cid))
            cv2.putText(
                out, label, (x + point_radius + 2, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    return out


def _detect_result_type(results: list) -> str:
    """Auto-detect inference result type from the results list."""
    if not results:
        raise ValueError("results list is empty")

    first = results[0]

    # Localizer: list[list[dict]]
    if isinstance(first, list):
        return "localizer"

    # Pose: ultralytics Results with .keypoints
    if hasattr(first, "keypoints") and first.keypoints is not None:
        kps = first.keypoints
        if hasattr(kps, "data") and kps.data is not None and len(kps.data) > 0:
            return "pose"

    # Point: ultralytics Results with .locations
    if hasattr(first, "locations") and first.locations is not None:
        return "point"

    # Fallback: check if it has boxes (generic detection)
    if hasattr(first, "boxes"):
        return "pose"

    raise ValueError(
        "Cannot auto-detect result type. Pass result_type='pose', 'point', or 'localizer'."
    )


def _extract_point_detections(result: Any) -> list[dict]:
    """Extract point detections from a POLO result as a list of dicts."""
    locs = getattr(result, "locations", None)
    if locs is None:
        return []
    locs_data = getattr(locs, "data", None)
    if locs_data is None or len(locs_data) == 0:
        return []

    locs_np = locs_data.cpu().numpy() if hasattr(locs_data, "cpu") else np.asarray(locs_data)
    names = getattr(result, "names", {})

    dets = []
    for i in range(locs_np.shape[0]):
        if locs_np.shape[1] >= 5:
            x, y, _track, conf, cls = locs_np[i, :5]
        else:
            x, y, conf, cls = locs_np[i, :4]
        dets.append({
            "x": float(x),
            "y": float(y),
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": names.get(int(cls), f"class_{int(cls)}"),
        })
    return dets


def visualize_inference(
    video_path: str | Path,
    results: list,
    *,
    result_type: str | None = None,
    rendering: str = "custom",
    output_path: str | Path | None = None,
    show_window: bool = True,
    window_name: str = "Inference",
    start_frame: int = 0,
    frame_step: int = 1,
    # Pose options
    skeleton: list[tuple[int, int]] | None = None,
    conf_threshold: float = 0.3,
    point_radius: int = 4,
    point_color: tuple[int, int, int] = (0, 255, 0),
    line_color: tuple[int, int, int] = (255, 255, 0),
    line_thickness: int = 2,
    # Point/localizer options
    class_colors: dict[int, tuple[int, int, int]] | None = None,
    show_labels: bool = False,
    det_point_radius: int = 5,
    # Writer options
    crf: int = 23,
    # Progress
    verbose: bool = True,
) -> Path | None:
    """Visualize inference results overlaid on the source video.

    Supports three result types:

    - **pose**: YOLO pose results with ``.keypoints`` — draws skeleton and keypoints
    - **point**: POLO point-detection results with ``.locations`` — draws detection dots
    - **localizer**: ``list[list[dict]]`` from localizer — draws detection dots

    Parameters
    ----------
    video_path : path
        Source video file.
    results : list
        Inference results — ultralytics Results objects or
        ``list[list[dict]]`` for localizer.
    result_type : str, optional
        ``"pose"``, ``"point"``, or ``"localizer"``.  Auto-detected if None.
    rendering : str
        ``"custom"`` (default) for manual drawing with full control, or
        ``"ultralytics"`` to use ``result.plot()`` (pose/point only).
    output_path : path, optional
        Save annotated video to this path (MP4).  Uses ffmpeg for fast
        H.264 encoding when available, falls back to OpenCV VideoWriter.
    show_window : bool
        Display video in an OpenCV window with interactive controls.
    window_name : str
        Name for the display window.
    start_frame : int
        Frame index of the first result (must match the inference run).
    frame_step : int
        Frame stepping used during inference (must match).
    skeleton : list of (i, j), optional
        Keypoint index pairs for skeleton lines (pose only).
    conf_threshold : float
        Minimum confidence for drawing keypoints/detections.
    point_radius : int
        Keypoint circle radius (pose).
    point_color : tuple
        BGR color for keypoints (pose).
    line_color : tuple
        BGR color for skeleton lines (pose).
    line_thickness : int
        Skeleton line width (pose).
    class_colors : dict, optional
        ``{class_id: (B, G, R)}`` for detection dots (point/localizer).
    show_labels : bool
        Draw class labels next to detections (point/localizer).
    det_point_radius : int
        Detection dot radius (point/localizer).
    crf : int
        H.264 quality (0–51, lower = better).  Only used with ffmpeg writer.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    Path or None
        Path to saved video file, or None if ``output_path`` was not set.

    Keyboard Controls (when show_window=True)
    -----------------------------------------
    - **q** / **Esc**: Quit
    - **Space**: Pause / resume
    - **d**: Step one frame (while paused)
    - **s**: Save current frame as PNG
    """
    if not results:
        raise ValueError("results list is empty")

    rtype = result_type or _detect_result_type(results)
    if rtype not in ("pose", "point", "localizer"):
        raise ValueError(f"Unknown result_type: {rtype!r}")

    if rendering not in ("custom", "ultralytics"):
        raise ValueError(f"Unknown rendering mode: {rendering!r}")

    if rendering == "ultralytics" and rtype == "localizer":
        raise ValueError("rendering='ultralytics' is not supported for localizer results")

    meta = get_video_metadata(video_path)

    # Compute scale factors for coordinate mapping
    # Ultralytics results store orig_shape = (H, W) of the inference input
    scale_x, scale_y = 1.0, 1.0
    if rendering == "custom" and rtype in ("pose", "point") and hasattr(results[0], "orig_shape"):
        inf_h, inf_w = results[0].orig_shape
        if inf_w != meta.width or inf_h != meta.height:
            scale_x = meta.width / inf_w
            scale_y = meta.height / inf_h

    # Open video: a sequential windowed reader in lockstep with `results`,
    # rather than a per-result seek - `results[i]` always corresponds to
    # frame `start_frame + i * frame_step`, exactly the sequence this reader
    # yields.
    reader = open_frame_reader(video_path, start_frame=start_frame, frame_step=frame_step)
    frame_iterator = iter(reader)

    # Open writer
    writer = None
    out_path = None
    if output_path is not None:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = FFmpegVideoWriter(
                out_path, meta.width, meta.height,
                fps=meta.fps, crf=crf,
            )
        except (RuntimeError, MediaProbeError):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_path), fourcc, meta.fps, (meta.width, meta.height),
            )

    # Progress bar
    pbar = None
    if verbose:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(results), desc="Visualize", unit="frame")
        except ImportError:
            pass

    paused = False
    step_once = False

    try:
        for i, result in enumerate(results):
            target_frame = start_frame + i * frame_step
            try:
                _yielded_index, frame = next(frame_iterator)
            except StopIteration:
                break

            # Annotate
            if rendering == "ultralytics":
                annotated = result.plot()
                # Resize to source resolution if inference was at different size
                ah, aw = annotated.shape[:2]
                if aw != meta.width or ah != meta.height:
                    annotated = cv2.resize(annotated, (meta.width, meta.height))
            elif rtype == "pose":
                annotated = frame.copy()
                kps_attr = getattr(result, "keypoints", None)
                if kps_attr is not None:
                    kps_data = getattr(kps_attr, "data", None)
                    if kps_data is not None and len(kps_data) > 0:
                        kps_np = kps_data.cpu().numpy() if hasattr(kps_data, "cpu") else np.asarray(kps_data)
                        for det_idx in range(kps_np.shape[0]):
                            kpts = kps_np[det_idx].copy()
                            kpts[:, 0] *= scale_x
                            kpts[:, 1] *= scale_y
                            annotated = visualize_keypoints(
                                annotated, kpts, skeleton,
                                conf_threshold=conf_threshold,
                                point_radius=point_radius,
                                point_color=point_color,
                                line_color=line_color,
                                line_thickness=line_thickness,
                            )
            elif rtype == "point":
                dets = _extract_point_detections(result)
                if scale_x != 1.0 or scale_y != 1.0:
                    for d in dets:
                        d["x"] *= scale_x
                        d["y"] *= scale_y
                annotated = visualize_detections(
                    frame, dets,
                    conf_threshold=conf_threshold,
                    point_radius=det_point_radius,
                    class_colors=class_colors,
                    show_labels=show_labels,
                )
            else:  # localizer
                dets = result if isinstance(result, list) else []
                annotated = visualize_detections(
                    frame, dets,
                    conf_threshold=conf_threshold,
                    point_radius=det_point_radius,
                    class_colors=class_colors,
                    show_labels=show_labels,
                )

            # Write
            if writer is not None:
                if isinstance(writer, cv2.VideoWriter):
                    writer.write(annotated)
                else:
                    writer.write(annotated)

            # Display
            if show_window:
                if not paused or step_once or i == 0:
                    cv2.imshow(window_name, annotated)

                delay = 1 if not paused else 50
                key = cv2.waitKey(delay) & 0xFF

                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    paused = not paused
                    step_once = False
                elif key == ord("d"):
                    paused = True
                    step_once = True
                elif key == ord("s"):
                    snap = Path(f"frame_{target_frame}.png")
                    cv2.imwrite(str(snap), annotated)
                    print(f"[visualize] saved frame -> {snap}")
                else:
                    step_once = False

            if pbar is not None:
                pbar.update(1)
    finally:
        reader.close()
        if writer is not None:
            if isinstance(writer, cv2.VideoWriter):
                writer.release()
            else:
                writer.close()
        if show_window:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
        if pbar is not None:
            pbar.close()

    if out_path is not None:
        print(f"[visualize] saved video -> {out_path}")
    return out_path
