"""Video I/O helpers for media extraction."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Sequence

import cv2
import numpy as np
from mosaic_media import MediaFacts, MediaProbeError, probe_media
from mosaic_media.io import MultiVideoReader as _PlainMultiReader
from mosaic_media.io import SeekIndex
from mosaic_media.io import VideoReader

# Downscale filter for candidate-feature frames. Read at call time so it stays
# monkeypatchable. INTER_AREA is the area-averaging filter suited to downscaling
# (no aliasing), and using one filter across every extraction path keeps k-means
# candidate features comparable and their selections reproducible.
RESIZE_INTERPOLATION = cv2.INTER_AREA


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata for a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


class SupportsCapture(Protocol):
    """The ``cv2.VideoCapture`` subset that :class:`_ImgStoreMultiReader` relies on.

    :class:`mosaic.core.media.imgstore_io.ImgStoreCapture` satisfies this
    structurally. Every segment :class:`_ImgStoreMultiReader` handles is
    imgstore-backed, so it constructs an ``ImgStoreCapture`` directly.
    """

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    # Param names match cv2.VideoCapture (`propId`) so it satisfies this protocol.
    def set(self, propId: int, value: float) -> bool: ...
    def get(self, propId: int) -> float: ...
    def release(self) -> None: ...


class SupportsSeekRead(Protocol):
    """The lean read/seek surface shared by VideoReader and ImgStoreCapture.

    Narrower than :class:`SupportsCapture`: an absolute-frame ``seek`` plus a
    sequential ``read``, without the ``cv2`` CAP_PROP idiom. Both
    :class:`mosaic_media.io.VideoReader` and
    :class:`mosaic.core.media.imgstore_io.ImgStoreCapture` satisfy this
    structurally.
    """

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def fps(self) -> float: ...
    @property
    def frame_count(self) -> int: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def seek(self, frame_index: int) -> None: ...
    def close(self) -> None: ...


class FrameReader(Protocol):
    """High-throughput sequential reader interface (in-process- or imgstore-backed).

    :class:`mosaic_media.io.VideoReader` and
    :class:`mosaic.core.media.imgstore_io.ImgStoreFrameReader` both satisfy this.
    """

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def fps(self) -> float: ...
    @property
    def frame_count(self) -> int: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def read_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]: ...
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...


# Static check: VideoReader satisfies the lean seek/read surface structurally,
# same as ImgStoreCapture (see the mirrored check in imgstore_io.py).
_: type[SupportsSeekRead] = VideoReader


def facts_to_video_metadata(path: Path, facts: MediaFacts) -> VideoMetadata:
    """Build display-oriented :class:`VideoMetadata` from media facts.

    Swaps coded width/height on a quarter-turn rotation so the returned
    dimensions are display-oriented. The single place this swap lives: both the
    probe path (:func:`get_video_metadata`) and stored-facts callers (frame
    extraction) build metadata through it, so they cannot drift.
    """
    if facts.rotation_degrees % 180 == 90:
        width, height = facts.height, facts.width
    else:
        width, height = facts.width, facts.height
    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        fps=facts.fps,
        frame_count=facts.frame_count,
    )


def get_video_metadata(video_path: Path | str) -> VideoMetadata:
    """Read basic metadata from a video file.

    imgstore directories dispatch to the imgstore metadata reader (no
    cv2/ffprobe). Everything else is probed via ``mosaic_media.probe_media``
    and returns display-oriented width/height (coded dimensions swapped on a
    quarter-turn rotation).
    """
    from .imgstore_io import imgstore_metadata, is_imgstore

    if is_imgstore(video_path):
        # Resolve happens inside imgstore_metadata; a store is a directory,
        # read through the imgstore index rather than mosaic_media's
        # ffprobe-based probe, so dispatch before that probe.
        return imgstore_metadata(video_path)

    path = Path(video_path).expanduser().resolve()
    try:
        facts = probe_media(path)
    except MediaProbeError as exc:
        message = f"Failed to probe video: {path}"
        raise MediaProbeError(message) from exc

    return facts_to_video_metadata(path, facts)


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


def extract_candidate_features(
    video_path: Path | str,
    start_frame: int,
    end_frame: int,
    candidate_step: int,
    resize: tuple[int, int],
    grayscale: bool,
    crop_rect: Optional[tuple[int, int, int, int]],
    max_candidates: Optional[int] = None,
    facts: MediaFacts | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode candidate frames and return:
      - candidate frame indices (N,)
      - flattened feature vectors (N, D)

    When *facts* is supplied (a dataset-scoped caller holding the media index
    row) the plain-video reader injects them instead of re-probing; a bare-path
    caller passes ``None`` and the file is probed.
    """
    from .imgstore_io import is_imgstore

    if candidate_step <= 0:
        raise ValueError("candidate_step must be > 0")
    if resize[0] <= 0 or resize[1] <= 0:
        raise ValueError("resize must be (width>0, height>0)")

    resolved = Path(video_path).expanduser().resolve()
    indices: list[int] = []
    features: list[np.ndarray] = []
    max_n = None if max_candidates is None else max(1, int(max_candidates))

    def _accumulate(frame_idx: int, frame: np.ndarray) -> bool:
        """Crop/resize/flatten one candidate frame; return True once max_n is hit."""
        work = apply_crop(frame, crop_rect)
        work = cv2.resize(work, resize, interpolation=RESIZE_INTERPOLATION)
        if grayscale:
            work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        vec = work.reshape(-1).astype(np.float32, copy=False) / 255.0
        indices.append(frame_idx)
        features.append(vec)
        return max_n is not None and len(indices) >= max_n

    if is_imgstore(resolved):
        # imgstore has no cv2/av-decodable container; route through the
        # imgstore-native windowed reader instead of VideoReader. It has no
        # decode-time resize/grayscale option, so crop/resize/grayscale are
        # always applied afterward here (mirrors the plain-video path below).
        reader = open_frame_reader(
            resolved,
            start_frame=int(start_frame),
            end_frame=int(end_frame) + 1,
            frame_step=int(candidate_step),
        )
        try:
            for frame_idx, frame in reader:
                if _accumulate(frame_idx, frame):
                    break
        finally:
            reader.close()
    else:
        # Candidate features resize through cv2, not the reader's libswscale resize,
        # so single-, multi-, crop-, and imgstore paths share one filter and produce
        # comparable k-means features (RESIZE_INTERPOLATION, monkeypatchable).
        with VideoReader(
            resolved,
            facts=facts if facts is not None else probe_media(resolved),
            start_frame=int(start_frame),
            end_frame=int(end_frame) + 1,
            frame_step=int(candidate_step),
        ) as reader:
            for frame_idx, frame in reader:
                if _accumulate(frame_idx, frame):
                    break

    if not indices:
        raise RuntimeError("No candidate frames available in the requested range.")

    return np.asarray(indices, dtype=np.int32), np.vstack(features).astype(np.float32, copy=False)


def save_frames_as_png(
    video_path: Path | str,
    frame_indices: np.ndarray,
    output_dir: Path | str,
    crop_rect: Optional[tuple[int, int, int, int]],
    facts: MediaFacts | None = None,
) -> list[dict[str, Any]]:
    """Decode selected frames and save each as PNG.

    When *facts* is supplied (a dataset-scoped caller holding the media index
    row) the plain-video reader injects them instead of re-probing; a bare-path
    caller passes ``None`` and ``VideoReader`` probes the file as before.
    """
    from .imgstore_io import ImgStoreCapture, is_imgstore

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = Path(video_path).expanduser().resolve()
    target_indices = sorted(set(int(i) for i in frame_indices))
    records: list[dict[str, Any]] = []
    if not target_indices:
        return records

    def _write(frame_idx: int, frame: np.ndarray) -> None:
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

    if is_imgstore(resolved):
        # imgstore is always randomly addressable by frame_index, so a plain
        # seek-then-read per target mirrors the old seekable branch exactly.
        capture: SupportsSeekRead = ImgStoreCapture(resolved)
        try:
            for frame_idx in target_indices:
                capture.seek(frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    message = f"Failed to decode frame {frame_idx} from {video_path}"
                    raise MediaProbeError(message)
                _write(frame_idx, frame)
        finally:
            capture.close()
    else:
        # read_frames groups targets by GOP and seeks/decodes in one pass,
        # regardless of whether the container is seekable metadata-wise.
        with VideoReader(resolved, facts=facts) as reader:
            for frame_idx, frame in reader.read_frames(target_indices):
                _write(frame_idx, frame)

    # Restore original order from frame_indices
    record_map = {r["frame_index"]: r for r in records}
    records = [record_map[int(i)] for i in frame_indices if int(i) in record_map]

    return records


# --- Multi-video support ---


@dataclass(frozen=True)
class VideoSegment:
    """Metadata for one video file within a multi-video sequence."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    start_frame: int  # global frame index where this segment starts


class _ImgStoreMultiReader:
    """Unified read interface across N ordered imgstore sequences.

    Provides a single virtual frame index space across N imgstore
    directories. Store 0 has frames [0, N0), store 1 has frames
    [N0, N0+N1), etc. Every segment is imgstore-backed and therefore
    always randomly addressable by frame index.

    For single-store sequences, accepts a single Path and works as a
    thin wrapper with minimal overhead.

    Parameters
    ----------
    video_paths : list[Path] | Path | str
        One or more imgstore directory paths, in playback order.
    """

    def __init__(self, video_paths: list[Path] | Path | str):
        # Set _closed early so __del__ is safe if __init__ fails partway
        self._closed: bool = False
        self._current_cap: Optional[SupportsSeekRead] = None

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
            # Every segment is imgstore-backed: stores support random access
            # by frame_index and are always seekable.
            seg = VideoSegment(
                path=meta.path,
                frame_count=meta.frame_count,
                fps=meta.fps,
                width=meta.width,
                height=meta.height,
                start_frame=cumulative,
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

        # Property mismatch (fps) raises the same way the plain multi-video
        # reader does (mosaic_media.io.multi.uniform_properties), so callers
        # see one consistent failure mode across both backends instead of a
        # silent fall-back to the first video's rate.
        fps_vals = sorted({round(s.fps, 4) for s in self._segments})
        if len(fps_vals) > 1:
            message = f"property mismatch across sequence: fps {fps_vals[0]} vs {fps_vals[1]}"
            raise ValueError(message)

    # -- Properties --

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

    # -- Frame-to-segment mapping --

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

    # -- Open / close helpers --

    def _open_segment(self, seg_idx: int) -> None:
        from .imgstore_io import ImgStoreCapture  # local: breaks the video_io <-> imgstore_io cycle

        if self._current_cap is not None:
            self._current_cap.close()
        seg = self._segments[seg_idx]
        self._current_cap = ImgStoreCapture(seg.path)
        self._current_seg_idx = seg_idx
        self._current_local_frame = 0

    def _ensure_open(self) -> SupportsSeekRead:
        """Return the current segment's capture, opening segment 0 lazily if needed."""
        if self._current_cap is None:
            self._open_segment(self._current_seg_idx)
        cap = self._current_cap
        if cap is None:
            # _open_segment always assigns _current_cap; unreachable in practice.
            message = "imgstore multi-reader failed to open a segment"
            raise MediaProbeError(message)
        return cap

    # -- Seeking --

    def seek(self, global_frame: int) -> None:
        """Seek to a global frame position.

        Every segment is imgstore-backed and therefore always randomly
        addressable by frame_index.
        """
        seg_idx, local_frame = self.segment_for_frame(global_frame)

        need_reopen = (
            self._current_cap is None
            or seg_idx != self._current_seg_idx
        )

        if need_reopen:
            self._open_segment(seg_idx)

        if need_reopen and local_frame == 0:
            # Freshly opened segment is already positioned at its first frame.
            pass
        else:
            self._ensure_open().seek(local_frame)
            self._current_local_frame = local_frame

        self._global_frame = global_frame

    # -- Reading --

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame, automatically transitioning between videos.

        Returns (success, frame) like cv2.VideoCapture.read().
        """
        if self._closed:
            return False, None
        if self._global_frame >= self.total_frames:
            return False, None

        ok, frame = self._ensure_open().read()
        if not ok:
            # Current segment exhausted -- try next
            next_idx = self._current_seg_idx + 1
            if next_idx >= len(self._segments):
                return False, None
            self._open_segment(next_idx)
            ok, frame = self._ensure_open().read()
            if not ok:
                return False, None

        self._global_frame += 1
        self._current_local_frame += 1
        return True, frame

    # -- Cleanup --

    def close(self) -> None:
        if not self._closed:
            if self._current_cap is not None:
                self._current_cap.close()
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


# Annotation alias for either multi-video backend; both expose
# seek/read/total_frames/width/height/fps.
type MultiVideoReaderLike = _ImgStoreMultiReader | _PlainMultiReader


def open_multi_video_reader(
    video_paths: list[Path] | Path | str,
    *,
    facts: Sequence[MediaFacts] | None = None,
    indices: Sequence[SeekIndex] | None = None,
) -> "MultiVideoReaderLike":
    """Dispatch a sequence of video paths to the matching multi-file reader.

    An all-imgstore sequence routes to :class:`_ImgStoreMultiReader`; an
    all-plain-video sequence routes to the ``mosaic_media`` reader. Mixing
    imgstore directories and plain video files within one sequence is not
    supported.
    """
    from .imgstore_io import is_imgstore  # local: breaks the video_io <-> imgstore_io cycle

    paths = (
        [Path(video_paths)]
        if isinstance(video_paths, (str, Path))
        else [Path(p) for p in video_paths]
    )
    stores = [is_imgstore(p) for p in paths]
    if any(stores) and not all(stores):
        raise ValueError("mixed video-plus-imgstore sequences are not supported")
    if all(stores):
        return _ImgStoreMultiReader(paths)
    return _PlainMultiReader(paths, facts=facts, indices=indices)


# The public name callers use for construction stays a callable with the same
# call shape; existing `MultiVideoReader(paths)` sites are unaffected.
MultiVideoReader = open_multi_video_reader


def open_frame_reader(
    video_path: Path | str,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    resize: tuple[int, int] | None = None,
    hwaccel: bool = False,
    facts: MediaFacts | None = None,
) -> FrameReader:
    """Open a high-throughput sequential reader, dispatching on the path type.

    Returns an :class:`mosaic.core.media.imgstore_io.ImgStoreFrameReader` for imgstore
    directories, otherwise a :class:`mosaic_media.io.VideoReader`. Both satisfy
    :class:`FrameReader`, so callers (e.g. tracking inference) are unchanged.

    When *facts* is supplied (a dataset-scoped caller holding the media index
    row) the plain-video reader injects them instead of re-probing; a bare-path
    caller passes ``None`` and the file is probed.
    """
    from .imgstore_io import ImgStoreFrameReader, is_imgstore

    if is_imgstore(video_path):
        return ImgStoreFrameReader(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            resize=resize,
            hwaccel=hwaccel,
        )
    return VideoReader(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        resize=resize,
        hwaccel=hwaccel,
        facts=facts if facts is not None else probe_media(Path(video_path)),
    )


def prefetch_batches(
    reader: FrameReader,
    queue: Any,
    batch_size: int,
) -> None:
    """Worker function for a prefetch thread: reads batches and puts them on a queue.

    Public because batched inference callers (e.g.
    :mod:`mosaic.tracking.pose_training.inference`) run it in a background
    thread to overlap decode with model inference.
    """
    try:
        while True:
            indices, batch = reader.read_batch(batch_size)
            if len(indices) == 0:
                break
            queue.put((indices, batch))
    finally:
        queue.put(None)  # sentinel


# --- Multi-video extraction helpers ---


def extract_candidate_features_multi(
    reader: "MultiVideoReaderLike",
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
            work = cv2.resize(work, resize, interpolation=RESIZE_INTERPOLATION)
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
    reader: "MultiVideoReaderLike",
    frame_indices: np.ndarray,
    output_dir: Path | str,
    crop_rect: Optional[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """Save selected frames from a multi-video reader as PNG files.

    Every backend (imgstore or plain video) seeks directly to each target
    frame; the reader owns any GOP-aware seek strategy internally.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_indices = sorted(set(int(i) for i in frame_indices))
    records: list[dict[str, Any]] = []

    for frame_idx in target_indices:
        reader.seek(frame_idx)
        ok, frame = reader.read()
        if not ok:
            message = f"Failed to decode global frame {frame_idx}"
            raise MediaProbeError(message)
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

    # Restore original order from frame_indices
    record_map = {r["frame_index"]: r for r in records}
    records = [record_map[int(i)] for i in frame_indices if int(i) in record_map]
    return records
