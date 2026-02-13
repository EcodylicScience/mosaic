"""Video I/O helpers for media extraction."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata for a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


def _ffprobe_fps(path: Path) -> float:
    """Get fps from ffprobe via r_frame_rate (works for raw H.264 streams)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        if streams:
            rate_str = streams[0].get("r_frame_rate", "")
            if "/" in rate_str:
                num, den = rate_str.split("/", 1)
                den = float(den)
                if den > 0:
                    return float(num) / den
            elif rate_str:
                return float(rate_str)
    except Exception as exc:
        print(f"[WARN] ffprobe fps fallback failed for {path}: {exc}", file=sys.stderr)
    return 0.0


def _count_frames_by_decoding(path: Path) -> int:
    """Count frames by decoding the entire video (reliable for raw streams)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return count


def get_video_metadata(video_path: Path | str) -> VideoMetadata:
    """Read basic metadata from a video file.

    For containerless streams (e.g. raw .h264) where OpenCV cannot determine
    fps or frame count from headers, falls back to ffprobe for fps and
    decode-counting for frame count.
    """
    path = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    # Detect bad metadata (containerless streams return 0 or garbage negatives)
    needs_fallback = frame_count <= 0 or fps <= 0

    if needs_fallback:
        if fps <= 0:
            fps = _ffprobe_fps(path)
        if frame_count <= 0:
            frame_count = _count_frames_by_decoding(path)

    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def normalize_frame_range(
    frame_count: int,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> tuple[int, int]:
    """Clamp and validate the inclusive frame extraction range."""
    if frame_count <= 0:
        raise ValueError("Video has no frames.")

    start = 0 if start_frame is None else int(start_frame)
    end = (frame_count - 1) if end_frame is None else int(end_frame)

    start = max(0, start)
    end = min(frame_count - 1, end)
    if start > end:
        raise ValueError(f"Invalid frame range after clamping: start={start}, end={end}")
    return start, end


def normalize_crop_rect(
    crop: Optional[tuple[int, int, int, int] | dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> Optional[tuple[int, int, int, int]]:
    """Normalize crop input to an in-bounds (x, y, w, h) rectangle."""
    if crop is None:
        return None

    if isinstance(crop, dict):
        try:
            x = int(crop["x"])
            y = int(crop["y"])
            w = int(crop["w"])
            h = int(crop["h"])
        except KeyError as exc:
            raise ValueError("Crop dict must include x, y, w, h keys.") from exc
    else:
        if len(crop) != 4:
            raise ValueError("Crop tuple/list must be length 4: (x, y, w, h).")
        x, y, w, h = [int(v) for v in crop]

    x = max(0, x)
    y = max(0, y)
    w = min(w, max(0, frame_width - x))
    h = min(h, max(0, frame_height - y))
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Crop rectangle is empty/out-of-bounds after clamping: {(x, y, w, h)} "
            f"for frame size {(frame_width, frame_height)}"
        )
    return (x, y, w, h)


def apply_crop(frame: np.ndarray, crop_rect: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    """Apply an optional (x, y, w, h) crop rectangle."""
    if crop_rect is None:
        return frame
    x, y, w, h = crop_rect
    return frame[y:y + h, x:x + w]


def _has_container(cap: cv2.VideoCapture) -> bool:
    """Check if the video has a proper container with seekable metadata.

    Raw elementary streams (e.g. .h264) report garbage or zero for
    CAP_PROP_FRAME_COUNT. This is a reliable, non-destructive indicator.
    """
    raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return 0 < raw_count < 1e12


def extract_candidate_features(
    video_path: Path | str,
    start_frame: int,
    end_frame: int,
    candidate_step: int,
    resize: tuple[int, int],
    grayscale: bool,
    crop_rect: Optional[tuple[int, int, int, int]],
    max_candidates: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode candidate frames and return:
      - candidate frame indices (N,)
      - flattened feature vectors (N, D)
    """
    if candidate_step <= 0:
        raise ValueError("candidate_step must be > 0")
    if resize[0] <= 0 or resize[1] <= 0:
        raise ValueError("resize must be (width>0, height>0)")

    cap = cv2.VideoCapture(str(Path(video_path).expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    indices: list[int] = []
    features: list[np.ndarray] = []
    max_n = None if max_candidates is None else max(1, int(max_candidates))

    try:
        seekable = _has_container(cap)
        # For seekable videos, jump to start_frame; otherwise read sequentially
        if seekable and int(start_frame) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            frame_idx = int(start_frame)
        else:
            # Read sequentially from the beginning, skip frames before start
            frame_idx = 0

        while frame_idx <= int(end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= int(start_frame):
                if (frame_idx - int(start_frame)) % int(candidate_step) == 0:
                    work = apply_crop(frame, crop_rect)
                    work = cv2.resize(work, resize, interpolation=cv2.INTER_AREA)
                    if grayscale:
                        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
                    vec = work.reshape(-1).astype(np.float32, copy=False) / 255.0
                    indices.append(frame_idx)
                    features.append(vec)
                    if max_n is not None and len(indices) >= max_n:
                        break
            frame_idx += 1
    finally:
        cap.release()

    if not indices:
        raise RuntimeError("No candidate frames available in the requested range.")

    return np.asarray(indices, dtype=np.int32), np.vstack(features).astype(np.float32, copy=False)


def save_frames_as_png(
    video_path: Path | str,
    frame_indices: np.ndarray,
    output_dir: Path | str,
    crop_rect: Optional[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """Decode selected frames and save each as PNG.

    For non-seekable streams (e.g. raw .h264), reads sequentially and
    captures target frames as they are reached.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(Path(video_path).expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    target_indices = sorted(set(int(i) for i in frame_indices))
    records: list[dict[str, Any]] = []

    try:
        seekable = _has_container(cap)

        if seekable:
            # Original seek-based approach
            for frame_idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {frame_idx} from {video_path}")
                frame = apply_crop(frame, crop_rect)
                out_name = f"frame_{frame_idx:06d}.png"
                out_path = out_dir / out_name
                ok_write = cv2.imwrite(str(out_path), frame)
                if not ok_write:
                    raise RuntimeError(f"Failed to write PNG frame: {out_path}")
                h, w = frame.shape[:2]
                records.append({
                    "frame_index": frame_idx,
                    "path": out_name,
                    "width": int(w),
                    "height": int(h),
                })
        else:
            # Sequential read for non-seekable streams
            target_set = set(target_indices)
            max_target = target_indices[-1]
            frame_idx = 0
            while frame_idx <= max_target:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in target_set:
                    frame = apply_crop(frame, crop_rect)
                    out_name = f"frame_{frame_idx:06d}.png"
                    out_path = out_dir / out_name
                    ok_write = cv2.imwrite(str(out_path), frame)
                    if not ok_write:
                        raise RuntimeError(f"Failed to write PNG frame: {out_path}")
                    h, w = frame.shape[:2]
                    records.append({
                        "frame_index": frame_idx,
                        "path": out_name,
                        "width": int(w),
                        "height": int(h),
                    })
                frame_idx += 1
    finally:
        cap.release()

    # Restore original order from frame_indices
    record_map = {r["frame_index"]: r for r in records}
    records = [record_map[int(i)] for i in frame_indices if int(i) in record_map]

    return records
