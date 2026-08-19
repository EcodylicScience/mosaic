"""Drawing inference results over the video they came from.

What ran the models used to live here too, and no longer does: YOLO pose and POLO
point inference are AGPL-3.0 and now run in environments of their own, reached as
a subprocess from
:mod:`mosaic.tracking.pose_training.ultralytics_infer`. What is left is mosaic's
own OpenCV drawing, which imports nothing from either.

:func:`visualize_inference` reads results objects rather than a predictions table,
so nothing in mosaic produces its ``pose`` and ``point`` inputs any more -- only a
caller holding an Ultralytics of their own can. Its ``localizer`` input, plain
``list[list[dict]]``, is still mosaic's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mosaic_media import MediaProbeError
from mosaic_media.io import FFmpegVideoWriter

from mosaic.core.media.video_io import get_video_metadata, open_frame_reader
from mosaic.user_paths import user_path


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
                out,
                label,
                (x + point_radius + 2, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
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

    # Pose: Ultralytics Results with .keypoints
    if hasattr(first, "keypoints") and first.keypoints is not None:
        kps = first.keypoints
        if hasattr(kps, "data") and kps.data is not None and len(kps.data) > 0:
            return "pose"

    # Point: Ultralytics Results with .locations
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

    locs_np = (
        locs_data.cpu().numpy() if hasattr(locs_data, "cpu") else np.asarray(locs_data)
    )
    names = getattr(result, "names", {})

    dets = []
    for i in range(locs_np.shape[0]):
        if locs_np.shape[1] >= 5:
            x, y, _track, conf, cls = locs_np[i, :5]
        else:
            x, y, conf, cls = locs_np[i, :4]
        dets.append(
            {
                "x": float(x),
                "y": float(y),
                "confidence": float(conf),
                "class_id": int(cls),
                "class_name": names.get(int(cls), f"class_{int(cls)}"),
            }
        )
    return dets


def visualize_inference(
    video_path: str | Path,
    results: list,
    *,
    result_type: str | None = None,
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
        Inference results — Ultralytics Results objects or
        ``list[list[dict]]`` for localizer.
    result_type : str, optional
        ``"pose"``, ``"point"``, or ``"localizer"``.  Auto-detected if None.
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

    meta = get_video_metadata(video_path)

    # Compute scale factors for coordinate mapping
    # Ultralytics results store orig_shape = (H, W) of the inference input
    scale_x, scale_y = 1.0, 1.0
    if rtype in ("pose", "point") and hasattr(results[0], "orig_shape"):
        inf_h, inf_w = results[0].orig_shape
        if inf_w != meta.width or inf_h != meta.height:
            scale_x = meta.width / inf_w
            scale_y = meta.height / inf_h

    # Open video: a sequential windowed reader in lockstep with `results`,
    # rather than a per-result seek - `results[i]` always corresponds to
    # frame `start_frame + i * frame_step`, exactly the sequence this reader
    # yields.
    reader = open_frame_reader(
        video_path, start_frame=start_frame, frame_step=frame_step, target="analysis"
    )
    frame_iterator = iter(reader)

    # Open writer
    writer = None
    out_path = None
    if output_path is not None:
        out_path = user_path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = FFmpegVideoWriter(
                out_path,
                meta.width,
                meta.height,
                fps=meta.fps,
                crf=crf,
            )
        except (RuntimeError, MediaProbeError):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_path),
                fourcc,
                meta.fps,
                (meta.width, meta.height),
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
            if rtype == "pose":
                annotated = frame.copy()
                kps_attr = getattr(result, "keypoints", None)
                if kps_attr is not None:
                    kps_data = getattr(kps_attr, "data", None)
                    if kps_data is not None and len(kps_data) > 0:
                        kps_np = (
                            kps_data.cpu().numpy()
                            if hasattr(kps_data, "cpu")
                            else np.asarray(kps_data)
                        )
                        for det_idx in range(kps_np.shape[0]):
                            kpts = kps_np[det_idx].copy()
                            kpts[:, 0] *= scale_x
                            kpts[:, 1] *= scale_y
                            annotated = visualize_keypoints(
                                annotated,
                                kpts,
                                skeleton,
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
                    frame,
                    dets,
                    conf_threshold=conf_threshold,
                    point_radius=det_point_radius,
                    class_colors=class_colors,
                    show_labels=show_labels,
                )
            else:  # localizer
                dets = result if isinstance(result, list) else []
                annotated = visualize_detections(
                    frame,
                    dets,
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
