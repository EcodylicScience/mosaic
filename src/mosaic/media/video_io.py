"""Video I/O helpers for media extraction."""

from __future__ import annotations

import bisect
import json
import subprocess
import sys
import warnings
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


# ─── Multi-video support ───


@dataclass(frozen=True)
class VideoSegment:
    """Metadata for one video file within a multi-video sequence."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    start_frame: int  # global frame index where this segment starts
    seekable: bool  # True if container-based (not raw H.264)


class MultiVideoReader:
    """Unified read interface across N ordered video files.

    Provides a single virtual frame index space across N video files.
    Video 0 has frames [0, N0), video 1 has frames [N0, N0+N1), etc.

    For single-video sequences, accepts a single Path and works as a
    thin wrapper with minimal overhead.

    Parameters
    ----------
    video_paths : list[Path] | Path | str
        One or more video file paths, in playback order.
    """

    def __init__(self, video_paths: list[Path] | Path | str):
        # Set _closed early so __del__ is safe if __init__ fails partway
        self._closed: bool = False
        self._current_cap: Optional[cv2.VideoCapture] = None

        if isinstance(video_paths, (str, Path)):
            video_paths = [Path(video_paths)]
        else:
            video_paths = [Path(p) for p in video_paths]
        if not video_paths:
            raise ValueError("At least one video path is required.")

        self._segments: list[VideoSegment] = []
        self._seg_starts: list[int] = []  # for bisect lookup
        self._build_segments(video_paths)

        self._current_seg_idx: int = 0
        self._current_local_frame: int = 0
        self._global_frame: int = 0

    def _build_segments(self, paths: list[Path]) -> None:
        cumulative = 0
        for p in paths:
            meta = get_video_metadata(p)
            # Check seekability without keeping the capture open
            cap = cv2.VideoCapture(str(meta.path))
            seekable = _has_container(cap) if cap.isOpened() else False
            cap.release()
            seg = VideoSegment(
                path=meta.path,
                frame_count=meta.frame_count,
                fps=meta.fps,
                width=meta.width,
                height=meta.height,
                start_frame=cumulative,
                seekable=seekable,
            )
            self._segments.append(seg)
            self._seg_starts.append(cumulative)
            cumulative += meta.frame_count

        # Validate resolution consistency
        dims = {(s.width, s.height) for s in self._segments}
        if len(dims) > 1:
            raise ValueError(
                f"Resolution mismatch across videos in sequence: {dims}. "
                "All videos must have the same resolution."
            )

        # Warn on fps mismatch
        fps_vals = {round(s.fps, 4) for s in self._segments}
        if len(fps_vals) > 1:
            warnings.warn(
                f"FPS mismatch across videos in sequence: {fps_vals}. "
                f"Using first video's fps ({self._segments[0].fps})."
            )

    # ── Properties ──

    @property
    def total_frames(self) -> int:
        return sum(s.frame_count for s in self._segments)

    @property
    def fps(self) -> float:
        return self._segments[0].fps if self._segments else 0.0

    @property
    def width(self) -> int:
        return self._segments[0].width if self._segments else 0

    @property
    def height(self) -> int:
        return self._segments[0].height if self._segments else 0

    @property
    def video_count(self) -> int:
        return len(self._segments)

    @property
    def segments(self) -> list[VideoSegment]:
        return list(self._segments)

    @property
    def frame_position(self) -> int:
        return self._global_frame

    # ── Frame-to-segment mapping ──

    def segment_for_frame(self, global_frame: int) -> tuple[int, int]:
        """Return (segment_index, local_frame) for a global frame index."""
        if global_frame < 0 or global_frame >= self.total_frames:
            raise IndexError(
                f"Global frame {global_frame} out of range [0, {self.total_frames})"
            )
        # bisect_right gives the insertion point; subtract 1 for the segment
        idx = bisect.bisect_right(self._seg_starts, global_frame) - 1
        local = global_frame - self._seg_starts[idx]
        return idx, local

    # ── Open / close helpers ──

    def _open_segment(self, seg_idx: int) -> None:
        if self._current_cap is not None:
            self._current_cap.release()
        seg = self._segments[seg_idx]
        self._current_cap = cv2.VideoCapture(str(seg.path))
        if not self._current_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {seg.path}")
        self._current_seg_idx = seg_idx
        self._current_local_frame = 0

    # ── Seeking ──

    def seek(self, global_frame: int) -> None:
        """Seek to a global frame position.

        For seekable videos, uses CAP_PROP_POS_FRAMES. For non-seekable
        streams (raw H.264), reopens the segment and skips sequentially.
        """
        seg_idx, local_frame = self.segment_for_frame(global_frame)
        seg = self._segments[seg_idx]

        need_reopen = (
            self._current_cap is None
            or seg_idx != self._current_seg_idx
        )

        if need_reopen:
            self._open_segment(seg_idx)

        if local_frame == 0:
            # Already at the start of the segment after open
            pass
        elif seg.seekable:
            self._current_cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
            self._current_local_frame = local_frame
        else:
            # Non-seekable: may need to reopen and skip forward
            if local_frame < self._current_local_frame:
                self._open_segment(seg_idx)
            while self._current_local_frame < local_frame:
                ok, _ = self._current_cap.read()
                if not ok:
                    break
                self._current_local_frame += 1

        self._global_frame = global_frame

    # ── Reading ──

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame, automatically transitioning between videos.

        Returns (success, frame) like cv2.VideoCapture.read().
        """
        if self._closed:
            return False, None
        if self._global_frame >= self.total_frames:
            return False, None

        if self._current_cap is None:
            self._open_segment(self._current_seg_idx)

        ok, frame = self._current_cap.read()
        if not ok:
            # Current segment exhausted — try next
            next_idx = self._current_seg_idx + 1
            if next_idx >= len(self._segments):
                return False, None
            self._open_segment(next_idx)
            ok, frame = self._current_cap.read()
            if not ok:
                return False, None

        self._global_frame += 1
        self._current_local_frame += 1
        return True, frame

    # ── Cleanup ──

    def close(self) -> None:
        if not self._closed:
            if self._current_cap is not None:
                self._current_cap.release()
                self._current_cap = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self.total_frames


# ─── Multi-video extraction helpers ───


def extract_candidate_features_multi(
    reader: MultiVideoReader,
    start_frame: int,
    end_frame: int,
    candidate_step: int,
    resize: tuple[int, int],
    grayscale: bool,
    crop_rect: Optional[tuple[int, int, int, int]],
    max_candidates: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract candidate features across a MultiVideoReader.

    Same as ``extract_candidate_features`` but reads from multiple videos
    via the unified reader. Frame indices are global (virtual).
    """
    if candidate_step <= 0:
        raise ValueError("candidate_step must be > 0")
    if resize[0] <= 0 or resize[1] <= 0:
        raise ValueError("resize must be (width>0, height>0)")

    indices: list[int] = []
    features: list[np.ndarray] = []
    max_n = None if max_candidates is None else max(1, int(max_candidates))

    reader.seek(start_frame)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ok, frame = reader.read()
        if not ok:
            break
        if (frame_idx - start_frame) % candidate_step == 0:
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

    if not indices:
        raise RuntimeError("No candidate frames available in the requested range.")

    return np.asarray(indices, dtype=np.int32), np.vstack(features).astype(np.float32, copy=False)


def save_frames_as_png_multi(
    reader: MultiVideoReader,
    frame_indices: np.ndarray,
    output_dir: Path | str,
    crop_rect: Optional[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """Save selected frames from a MultiVideoReader as PNG files.

    For seekable sequences, seeks to each frame. For non-seekable,
    reads sequentially and captures target frames.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_indices = sorted(set(int(i) for i in frame_indices))
    records: list[dict[str, Any]] = []

    # Check if all segments are seekable
    all_seekable = all(s.seekable for s in reader.segments)

    if all_seekable:
        for frame_idx in target_indices:
            reader.seek(frame_idx)
            ok, frame = reader.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to decode global frame {frame_idx}"
                )
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
        # Sequential read through all segments
        target_set = set(target_indices)
        max_target = target_indices[-1]
        reader.seek(0)
        frame_idx = 0
        while frame_idx <= max_target:
            ok, frame = reader.read()
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

    # Restore original order from frame_indices
    record_map = {r["frame_index"]: r for r in records}
    records = [record_map[int(i)] for i in frame_indices if int(i) in record_map]
    return records
