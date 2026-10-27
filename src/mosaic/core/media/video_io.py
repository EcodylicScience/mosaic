"""The media read facade: one entry point per kind of read, dispatching on the path.

:func:`open_frame_reader` (sequential, batched) and :data:`MultiVideoReader`
(seek and read across an entry's files) open an imgstore directory through
:mod:`mosaic.core.media.imgstore_io` and any other file through ``mosaic_media``.
Every read names a :data:`~mosaic.core.media.read_target.ReadTarget`, and the
facts it injects are gated against that target first.

A read of an imgstore may name a window of its chunks
(:class:`~mosaic.core.media.store_division.StoreWindow`), which a division of a
store reads, and frame 0 is then the window's first frame. A window is refused on
a plain video. The functions that read an entry's clips take ``windows``
parallel to the clips, as a required keyword.

A raw ``.h264`` elementary stream (Raspberry Pi) lacks a container. Its header
metadata is unreliable, and a seek corrupts the decoder. ``probe_media`` measures
its frame count and rate by scanning packets, and ``VideoReader`` decodes it
sequentially with those facts injected. Its verdict requires an analysis
transcode. An ``"analysis"`` read of one raises, and a ``"raw"`` read warns and
proceeds. Open one through this module.
"""

from __future__ import annotations

import bisect
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Sequence

import cv2
import numpy as np
from mosaic_media import (
    MeasuredVideoProperties,
    MediaFacts,
    MediaProbeError,
    probe_media,
    uniform_properties,
)
from mosaic.user_paths import user_path
from mosaic_media.io import FFmpegVideoWriter as FFmpegVideoWriter
from mosaic_media.io import MultiVideoReader as _PlainMultiReader
from mosaic_media.io import SeekIndex
from mosaic_media.io import VideoReader

from .read_target import ReadTarget, verified_read_facts
from .store_division import StoreWindow
from .store_rate import store_rate_mismatch

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
    :class:`mosaic.core.media.imgstore_io.ImgStoreFrameReader` both satisfy this,
    context manager included.
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
    def __enter__(self) -> "FrameReader": ...
    def __exit__(self, *args: object) -> None: ...


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


def get_video_metadata(
    video_path: Path | str, window: StoreWindow | None = None
) -> VideoMetadata:
    """Read basic metadata from a video file.

    imgstore directories dispatch to the imgstore metadata reader (no
    cv2/ffprobe), which reads a *window*'s chunks alone. Everything else is
    probed via ``mosaic_media.probe_media`` and returns display-oriented
    width/height (coded dimensions swapped on a quarter-turn rotation).
    """
    from .imgstore_io import imgstore_metadata, is_imgstore

    if is_imgstore(video_path):
        # Resolve happens inside imgstore_metadata; a store is a directory,
        # read through the imgstore index rather than mosaic_media's
        # ffprobe-based probe, so dispatch before that probe.
        return imgstore_metadata(video_path, window)

    path = user_path(video_path).resolve()
    refuse_plain_window(path, window)
    try:
        facts = probe_media(path)
    except MediaProbeError as exc:
        message = f"Failed to probe video: {path}"
        raise MediaProbeError(message) from exc

    return facts_to_video_metadata(path, facts)


def video_metadata_or_probe(
    video_path: Path | str,
    facts: MediaFacts | None,
    window: StoreWindow | None = None,
) -> VideoMetadata:
    """Metadata for *video_path*, from *facts* when the caller already holds them.

    A caller holding a measurement must not pay for a second one: the packet scan
    that mints identity reads every payload byte, so re-probing costs between
    1.5x and 2x the scan it repeats. Passing ``None`` probes, which is the only
    remaining reason to reach :func:`get_video_metadata` directly. The facts of
    a *window* of a store are the window's.
    """
    if facts is None:
        return get_video_metadata(video_path, window)
    return facts_to_video_metadata(user_path(video_path).resolve(), facts)


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
        raise ValueError(
            f"Invalid frame range after clamping: start={start}, end={end}"
        )
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


def apply_crop(
    frame: np.ndarray, crop_rect: Optional[tuple[int, int, int, int]]
) -> np.ndarray:
    """Apply an optional (x, y, w, h) crop rectangle."""
    if crop_rect is None:
        return frame
    x, y, w, h = crop_rect
    return frame[y : y + h, x : x + w]


def frame_feature(
    frame: np.ndarray,
    crop: tuple[int, int, int, int] | None,
    size: tuple[int, int],
    gray: bool,
) -> np.ndarray:
    """Return the k-means feature vector of one BGR frame.

    Args:
        frame: The BGR frame, ``uint8``.
        crop: The (x, y, w, h) rectangle cropped before the resize, or ``None``.
        size: The (width, height) that the frame is resized to, with
            :data:`RESIZE_INTERPOLATION`.
        gray: Whether the resized frame is converted to grayscale.

    Returns:
        The flattened frame as ``float32`` values in ``[0, 1]``.
    """
    work = apply_crop(frame, crop)
    work = cv2.resize(work, size, interpolation=RESIZE_INTERPOLATION)
    if gray:
        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    return work.reshape(-1).astype(np.float32, copy=False) / 255.0


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
    *,
    window: StoreWindow | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode candidate frames and return:
      - candidate frame indices (N,)
      - flattened feature vectors (N, D)

    *facts*, when supplied (a dataset-scoped caller holding the media index
    row), is gated against the fixed ``"analysis"`` target rather than injected
    verbatim; a bare-path caller passes ``None`` and the file is probed and
    gated. The target is fixed because these are k-means candidate frames
    addressed by index: a caller must not be able to declare ``"raw"`` for
    this read. A *window* reads those chunks of a store (:func:`open_frame_reader`).
    """
    from .imgstore_io import is_imgstore

    if candidate_step <= 0:
        raise ValueError("candidate_step must be > 0")
    if resize[0] <= 0 or resize[1] <= 0:
        raise ValueError("resize must be (width>0, height>0)")

    resolved = user_path(video_path).resolve()
    indices: list[int] = []
    features: list[np.ndarray] = []
    max_n = None if max_candidates is None else max(1, int(max_candidates))

    def _accumulate(frame_idx: int, frame: np.ndarray) -> bool:
        """Record one candidate frame's feature, and return True once max_n is hit."""
        indices.append(frame_idx)
        features.append(frame_feature(frame, crop_rect, resize, grayscale))
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
            target="analysis",
            window=window,
        )
        try:
            for frame_idx, frame in reader:
                if _accumulate(frame_idx, frame):
                    break
        finally:
            reader.close()
    else:
        refuse_plain_window(resolved, window)
        # Candidate features resize through cv2, not the reader's libswscale resize,
        # so single-, multi-, crop-, and imgstore paths share one filter and produce
        # comparable k-means features (RESIZE_INTERPOLATION, monkeypatchable).
        # Candidate frames become k-means training input addressed by frame
        # index, so the target is fixed at "analysis": a caller must not be
        # able to declare "raw" for a candidate scan.
        with VideoReader(
            resolved,
            facts=verified_read_facts(resolved, facts, "analysis")[0],
            start_frame=int(start_frame),
            end_frame=int(end_frame) + 1,
            frame_step=int(candidate_step),
        ) as reader:
            for frame_idx, frame in reader:
                if _accumulate(frame_idx, frame):
                    break

    if not indices:
        raise RuntimeError("No candidate frames available in the requested range.")

    return np.asarray(indices, dtype=np.int32), np.vstack(features).astype(
        np.float32, copy=False
    )


def save_frames_as_png(
    video_path: Path | str,
    frame_indices: np.ndarray,
    output_dir: Path | str,
    crop_rect: Optional[tuple[int, int, int, int]],
    facts: MediaFacts | None = None,
    *,
    window: StoreWindow | None = None,
) -> list[dict[str, Any]]:
    """Decode selected frames and save each as PNG.

    A *window* reads those chunks of a store (:func:`read_frames_at`).

    *facts*, when supplied (a dataset-scoped caller holding the media index
    row), is gated against the fixed ``"analysis"`` target rather than injected
    verbatim; a bare-path caller passes ``None`` and the file is probed and
    gated. The target is fixed because these frames are addressed by index and
    become pose-annotation training data: a caller must not be able to declare
    ``"raw"`` for this read.
    """
    out_dir = user_path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def _write(frame_idx: int, frame: np.ndarray) -> None:
        frame = apply_crop(frame, crop_rect)
        out_name = f"frame_{frame_idx:06d}.png"
        out_path = out_dir / out_name
        ok_write = cv2.imwrite(str(out_path), frame)
        if not ok_write:
            raise RuntimeError(f"Failed to write PNG frame: {out_path}")
        h, w = frame.shape[:2]
        records.append(
            {
                "frame_index": frame_idx,
                "path": out_name,
                "width": int(w),
                "height": int(h),
            }
        )

    for frame_idx, frame in read_frames_at(
        video_path, frame_indices, facts, window=window
    ):
        _write(frame_idx, frame)

    # Restore original order from frame_indices
    record_map = {r["frame_index"]: r for r in records}
    records = [record_map[int(i)] for i in frame_indices if int(i) in record_map]

    return records


def read_frames_at(
    video_path: Path | str,
    frame_indices: Iterable[int],
    facts: MediaFacts | None = None,
    *,
    window: StoreWindow | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield ``(frame, image)`` for each distinct index of *frame_indices*, ascending.

    An imgstore is read by a seek per frame. Any other file is read by
    ``VideoReader.read_frames``, which decodes each group of pictures once. *facts*,
    when supplied, is gated against the fixed ``"analysis"`` target, because the
    frames are addressed by index. A caller passing ``None`` has the file probed
    and gated. The reader is closed when the iteration ends or is abandoned.

    Args:
        video_path: The video file or imgstore directory.
        frame_indices: The frames to read.
        facts: The stored media facts of *video_path*, or ``None``.
        window: The chunks of a store to read, counted from the window's first
            frame, or ``None``.

    Raises:
        MediaProbeError: If a frame fails to decode, or lies outside the window.
        ValueError: If *window* is given for a plain video.
    """
    from .imgstore_io import ImgStoreCapture, is_imgstore

    resolved = user_path(video_path).resolve()
    targets = sorted({int(index) for index in frame_indices})
    if not targets:
        return
    if is_imgstore(resolved):
        capture: SupportsSeekRead = ImgStoreCapture(resolved, window)
        try:
            for frame_idx in targets:
                capture.seek(frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    message = f"Failed to decode frame {frame_idx} from {video_path}"
                    raise MediaProbeError(message)
                yield frame_idx, frame
        finally:
            capture.close()
        return
    refuse_plain_window(resolved, window)
    with VideoReader(
        resolved, facts=verified_read_facts(resolved, facts, "analysis")[0]
    ) as reader:
        yield from reader.read_frames(targets)


# --- Multi-video support ---


def refuse_plain_window(path: Path, window: StoreWindow | None) -> None:
    """Raise when a window of chunks is named for *path*, which is not a store.

    Raises:
        ValueError: If *window* is not ``None``.
    """
    if window is not None:
        message = (
            f"{path} is not an imgstore, and a read of it names the chunks "
            f"{window.chunk_first}-{window.chunk_last}"
        )
        raise ValueError(message)


def _clip_windows(
    paths: Sequence[Path], windows: Sequence[StoreWindow | None] | None
) -> list[StoreWindow | None]:
    """Return *windows* parallel to *paths*, ``None`` for each clip when absent.

    Raises:
        ValueError: If *windows* is not parallel to *paths*.
    """
    if windows is None:
        return [None] * len(paths)
    if len(windows) != len(paths):
        message = f"{len(windows)} windows for {len(paths)} clips; they are parallel"
        raise ValueError(message)
    return list(windows)


@dataclass(frozen=True)
class VideoSegment:
    """Metadata for one video file within a multi-video sequence.

    ``window`` is the chunks of a store that the segment reads, ``None`` for
    the whole file.
    """

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    start_frame: int  # global frame index where this segment starts
    window: StoreWindow | None = None


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
    windows : sequence of StoreWindow or None, or None
        The chunks of each store that its segment reads, parallel to
        *video_paths*. ``None`` reads every store whole.
    """

    def __init__(
        self,
        video_paths: list[Path] | Path | str,
        windows: Sequence[StoreWindow | None] | None = None,
    ):
        # Set _closed early so __del__ is safe if __init__ fails partway
        self._closed: bool = False
        self._current_cap: Optional[SupportsSeekRead] = None

        paths = entry_paths(video_paths)
        if not paths:
            raise ValueError("At least one video path is required.")

        self._segments: list[VideoSegment] = []
        self._seg_starts: list[int] = []  # for bisect lookup
        self._build_segments(paths, _clip_windows(paths, windows))

        self._current_seg_idx: int = 0
        self._current_local_frame: int = 0
        self._global_frame: int = 0

    def _build_segments(
        self, paths: list[Path], windows: Sequence[StoreWindow | None]
    ) -> None:
        cumulative = 0
        for p, window in zip(paths, windows, strict=True):
            meta = get_video_metadata(p, window)
            # Every segment is imgstore-backed: stores support random access
            # by frame_index and are always seekable.
            seg = VideoSegment(
                path=meta.path,
                frame_count=meta.frame_count,
                fps=meta.fps,
                width=meta.width,
                height=meta.height,
                start_frame=cumulative,
                window=window,
            )
            self._segments.append(seg)
            self._seg_starts.append(cumulative)
            cumulative += meta.frame_count

        # The plain multi-video reader's geometry check, asked with one rate
        # substituted, and the stores' own rate rule. A store's rate is measured
        # from its timestamps, so one rate measured twice differs in the last
        # digits, by an amount that does not grow with the store. Stores cannot be
        # joined, and the trackers and inference ops refuse such a sequence by the
        # same rule before they start.
        geometry = uniform_properties(
            [
                MeasuredVideoProperties(
                    fps=1.0,
                    width=segment.width,
                    height=segment.height,
                    frame_count=segment.frame_count,
                    duration=float(segment.frame_count),
                )
                for segment in self._segments
            ]
        )
        if geometry is not None:
            message = (
                f"property mismatch across sequence: {geometry.field} "
                f"{geometry.first} vs {geometry.other}"
            )
            raise ValueError(message)
        other = store_rate_mismatch([segment.fps for segment in self._segments])
        if other is not None:
            message = (
                f"property mismatch across sequence: fps {self._segments[0].fps} "
                f"vs {self._segments[other].fps}"
            )
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
        from .imgstore_io import (
            ImgStoreCapture,
        )  # local: breaks the video_io <-> imgstore_io cycle

        if self._current_cap is not None:
            self._current_cap.close()
        seg = self._segments[seg_idx]
        self._current_cap = ImgStoreCapture(seg.path, seg.window)
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

        need_reopen = self._current_cap is None or seg_idx != self._current_seg_idx

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
    windows: Sequence[StoreWindow | None] | None,
    facts: Sequence[MediaFacts] | None = None,
    indices: Sequence[SeekIndex] | None = None,
    target: ReadTarget,
) -> "MultiVideoReaderLike":
    """Dispatch a sequence of video paths to the matching multi-file reader.

    An all-imgstore sequence routes to :class:`_ImgStoreMultiReader`; an
    all-plain-video sequence routes to the ``mosaic_media`` reader, gating
    *facts* against *target* first. Mixing imgstore directories and plain video
    files within one sequence is not supported. *windows* are the chunks of each
    store that the reader reads, parallel to *video_paths*, and ``None`` reads
    each file whole. A window on a plain video raises ``ValueError``.

    *target* is required on both branches even though the imgstore branch
    constructs no ``mosaic_media`` reader and calls no gate: the parameter is
    the caller's declaration of intent, and a store's own chunk reads declare
    their target separately, where the chunk reader is actually built.
    """
    from .imgstore_io import (
        is_imgstore,
    )  # local: breaks the video_io <-> imgstore_io cycle

    paths = entry_paths(video_paths)
    clip_windows = _clip_windows(paths, windows)
    stores = [is_imgstore(p) for p in paths]
    if any(stores) and not all(stores):
        raise ValueError("mixed video-plus-imgstore sequences are not supported")
    if all(stores):
        return _ImgStoreMultiReader(paths, clip_windows)
    for path, window in zip(paths, clip_windows, strict=True):
        refuse_plain_window(path, window)
    return _PlainMultiReader(
        paths, facts=verified_read_facts(paths, facts, target), indices=indices
    )


# The public name callers use for construction: a plain alias, so every call
# site now names its target explicitly through the same call.
MultiVideoReader = open_multi_video_reader


def open_frame_reader(
    video_path: Path | str,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    resize: tuple[int, int] | None = None,
    hwaccel: bool = False,
    facts: MediaFacts | None = None,
    *,
    target: ReadTarget,
    window: StoreWindow | None = None,
) -> FrameReader:
    """Open a high-throughput sequential reader, dispatching on the path type.

    A *window* reads those chunks of a store, frame 0 being its first frame,
    and raises ``ValueError`` for a plain video.

    Returns an :class:`mosaic.core.media.imgstore_io.ImgStoreFrameReader` for imgstore
    directories, otherwise a :class:`mosaic_media.io.VideoReader`. Both satisfy
    :class:`FrameReader`, so callers (e.g. tracking inference) are unchanged.

    *facts*, when supplied (a dataset-scoped caller holding the media index
    row), is gated against *target* rather than injected verbatim; a bare-path
    caller passes ``None`` and the file is probed and gated. *target* is
    required on the imgstore branch as well, even though that branch calls no
    gate: the parameter is the caller's declaration of intent.
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
            window=window,
        )
    refuse_plain_window(Path(video_path), window)
    return VideoReader(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        resize=resize,
        hwaccel=hwaccel,
        facts=verified_read_facts(video_path, facts, target)[0],
    )


def entry_paths(video_paths: str | Path | Sequence[Path]) -> list[Path]:
    """Return an entry's clips as a list: *video_paths*, or one bare path in a list.

    A ``str`` is a sequence of its characters and a ``Path`` is not a sequence,
    so a caller naming one file without a list around it is read as that file.
    """
    if isinstance(video_paths, (str, Path)):
        return [Path(video_paths)]
    return [Path(path) for path in video_paths]


def read_entry_frames(
    video_paths: str | Path | Sequence[Path],
    *,
    windows: Sequence[StoreWindow | None] | None,
    facts: Sequence[MediaFacts] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    target: ReadTarget,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield ``(frame, image)`` for an entry's clips, read as one frame axis.

    Frame ``i`` is frame ``i`` of the entry, counted across its clips in order,
    which is the axis that ``MultiVideoReader`` and
    :class:`~mosaic.core.media.timeline.ConcatenatedTimeline` define. One file is
    read by :func:`open_frame_reader`, as a caller with one path reads it. Several
    are read through :func:`open_multi_video_reader`, which refuses clips whose
    frame rates differ. A caller holding such clips reads their join instead.

    The window is the one :func:`open_frame_reader` takes: from *start_frame*, up
    to *end_frame* exclusive, every *frame_step*-th frame. The reader is closed
    when the iteration ends or is abandoned.

    Args:
        video_paths: The clips in order, or one file. One file may be given as
            a bare path (:func:`entry_paths`).
        windows: The chunks of each store to read, parallel to *video_paths*,
            or ``None`` to read every file whole.
        facts: The clips' facts, parallel to *video_paths*, or ``None`` to probe.
        start_frame: The first frame read.
        end_frame: The frame the window stops before, or ``None`` for the end.
        frame_step: The stride between frames read.
        target: The read target the facts are gated against.
    """
    paths = entry_paths(video_paths)
    clip_windows = _clip_windows(paths, windows)
    if len(paths) == 1:
        single = open_frame_reader(
            paths[0],
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            facts=facts[0] if facts else None,
            target=target,
            window=clip_windows[0],
        )
        try:
            yield from single
        finally:
            single.close()
        return
    reader = open_multi_video_reader(
        paths, windows=clip_windows, facts=facts, target=target
    )
    try:
        total = reader.total_frames
        stop = total if end_frame is None else min(end_frame, total)
        if start_frame >= stop:
            return
        reader.seek(start_frame)
        for frame in range(start_frame, stop):
            ok, image = reader.read()
            if not ok or image is None:
                return
            if (frame - start_frame) % frame_step == 0:
                yield frame, image
    finally:
        reader.close()


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
        if not ok or frame is None:
            break
        if (frame_idx - start_frame) % candidate_step == 0:
            indices.append(frame_idx)
            features.append(frame_feature(frame, crop_rect, resize, grayscale))
            if max_n is not None and len(indices) >= max_n:
                break
        frame_idx += 1

    if not indices:
        raise RuntimeError("No candidate frames available in the requested range.")

    return np.asarray(indices, dtype=np.int32), np.vstack(features).astype(
        np.float32, copy=False
    )


def read_frames_at_multi(
    reader: "MultiVideoReaderLike", frame_indices: Iterable[int]
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield ``(frame, image)`` for each distinct global frame index, ascending.

    Each frame is read by a seek of *reader* to it. The reader stays open.

    Args:
        reader: The open reader over the sequence.
        frame_indices: The global frames to read, counted across the files.

    Raises:
        MediaProbeError: If a frame fails to decode.
    """
    for frame_idx in sorted({int(index) for index in frame_indices}):
        reader.seek(frame_idx)
        ok, frame = reader.read()
        if not ok or frame is None:
            message = f"Failed to decode global frame {frame_idx}"
            raise MediaProbeError(message)
        yield frame_idx, frame


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
    out_dir = user_path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for frame_idx, frame in read_frames_at_multi(reader, frame_indices):
        frame = apply_crop(frame, crop_rect)
        out_name = f"frame_{frame_idx:06d}.png"
        out_path = out_dir / out_name
        ok_write = cv2.imwrite(str(out_path), frame)
        if not ok_write:
            raise RuntimeError(f"Failed to write PNG frame: {out_path}")
        h, w = frame.shape[:2]
        records.append(
            {
                "frame_index": frame_idx,
                "path": out_name,
                "width": int(w),
                "height": int(h),
            }
        )

    # Restore original order from frame_indices
    record_map = {r["frame_index"]: r for r in records}
    records = [record_map[int(i)] for i in frame_indices if int(i) in record_map]
    return records
