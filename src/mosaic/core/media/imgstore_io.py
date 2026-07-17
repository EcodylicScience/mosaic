"""Native support for the imgstore video format (Motif / Loopbio).

An imgstore is a *directory* (not a single file): it contains a ``metadata.yaml``
descriptor (with a top-level ``__store`` key) plus zero-padded chunk video/image
files (``000000.mp4`` + ``000000.npz`` index, …). This module lets mosaic treat a
store like any other video source so the rest of the pipeline is unchanged:

* :func:`is_imgstore` — detect a store directory *without* importing imgstore.
* :func:`imgstore_metadata` / :func:`imgstore_probe` — read width/height/fps/frames.
* :class:`ImgStoreCapture` — adapt a store to the subset of the
  ``cv2.VideoCapture`` API that :class:`mosaic.core.media.video_io.MultiVideoReader`
  relies on, so every frame-consuming feature works unchanged.
* :class:`ImgStoreFrameReader` — mirror
  :class:`mosaic.core.media.video_io.FFmpegFrameReader` for high-throughput, batched
  sequential reads (tracking inference).

Frame addressing: mosaic treats the track-table ``frame`` column as the 0-based
*contiguous* video frame index, which maps to imgstore's ``frame_index`` — **not**
the camera-provided, possibly-sparse ``frame_number``. The adapters here address
frames by ``frame_index`` exclusively.

The ``imgstore`` package is an optional dependency
(``pip install 'mosaic-behavior[imgstore]'``); only :func:`is_imgstore` is
import-free, so detecting/indexing datasets without stores never forces the dep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol, cast

import cv2
import numpy as np
import yaml

from .video_io import FrameReader, SupportsCapture, VideoMetadata

# imgstore on-disk constants, inlined so detection needs no imgstore import.
_STORE_MD_FILENAME = "metadata.yaml"
_STORE_MD_KEY = "__store"


class _Store(Protocol):
    """Minimal typed view of an imgstore store (it ships no type stubs)."""

    frame_count: int

    @property
    def image_shape(self) -> tuple[int, ...]: ...
    @property
    def duration(self) -> float: ...
    def get_next_image(self) -> tuple[np.ndarray, tuple[int, float]]: ...
    def get_image(
        self,
        frame_number: int,
        exact_only: bool = ...,
        frame_index: int | None = ...,
    ) -> tuple[np.ndarray, tuple[int, float]]: ...
    def close(self, save_index: bool = ...) -> None: ...


@dataclass(frozen=True)
class _StoreMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str


def _fps_from_duration(frame_count: int, duration: float) -> float:
    """Estimate fps from the timestamp span.

    imgstore stores per-frame timestamps but no explicit fps. For ``N`` evenly
    spaced frames the span is ``(N-1) * frame_interval``, so the rate is
    ``(N-1) / duration`` — using ``N / duration`` would overestimate by
    ``N/(N-1)``. Returns ``0.0`` when fps cannot be determined.
    """
    if frame_count > 1 and duration > 0:
        return (frame_count - 1) / duration
    return 0.0


def _require_imgstore():
    """Import imgstore lazily, raising an actionable error if it is missing."""
    try:
        import imgstore  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "imgstore support requires the 'imgstore' package. "
            "Install it with: pip install 'mosaic-behavior[imgstore]'"
        ) from exc
    return imgstore


def _open_store(path: Path | str) -> _Store:
    """Open an imgstore for reading and return it as a typed :class:`_Store`."""
    imgstore = _require_imgstore()
    # imgstore ships no type stubs; cast the result to our typed view.
    store = imgstore.new_for_filename(str(path), mode="r")  # pyright: ignore[reportUnknownMemberType]
    return cast("_Store", store)


def is_imgstore(path: Path | str) -> bool:
    """Return ``True`` if *path* is an imgstore (a dir, or its ``metadata.yaml``).

    Detection only parses ``metadata.yaml`` and never imports the ``imgstore``
    package, so it is safe to call on every media path during indexing even when
    imgstore is not installed.
    """
    p = Path(path)
    if p.is_dir():
        md = p / _STORE_MD_FILENAME
    elif p.name == _STORE_MD_FILENAME:
        md = p
    else:
        return False
    if not md.is_file():
        return False
    try:
        with md.open("r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return False
    return isinstance(data, dict) and _STORE_MD_KEY in data


def _read_store_meta(path: Path | str) -> _StoreMeta:
    store = _open_store(path)
    try:
        shape = store.image_shape
        height = int(shape[0])
        width = int(shape[1])
        frame_count = int(store.frame_count)
        duration = float(store.duration)
        fps = _fps_from_duration(frame_count, duration)
        codec = str(getattr(store, "_format", "") or "")
    finally:
        store.close()
    return _StoreMeta(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec=codec,
    )


def imgstore_metadata(path: Path | str) -> VideoMetadata:
    """Read :class:`VideoMetadata` (width/height/fps/frame_count) from a store.

    ``fps`` is derived as ``frame_count / duration`` (imgstore does not store an
    explicit fps); it is ``0.0`` for stores with no usable timestamps.
    """
    p = Path(path).expanduser().resolve()
    m = _read_store_meta(p)
    return VideoMetadata(
        path=p,
        width=m.width,
        height=m.height,
        fps=m.fps,
        frame_count=m.frame_count,
    )


def imgstore_probe(path: Path | str) -> dict[str, object]:
    """Probe a store, returning a dict shaped like ``_probe_video_metadata``.

    Keys: ``width``, ``height``, ``fps``, ``codec`` (and ``frame_count``). Used by
    ``Dataset.index_media`` to populate ``media/index.csv``.
    """
    m = _read_store_meta(path)
    return {
        "width": m.width,
        "height": m.height,
        "fps": m.fps,
        "codec": m.codec,
        "frame_count": m.frame_count,
    }


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalize a decoded frame to 3-channel BGR (cv2.VideoCapture convention)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


class ImgStoreCapture:
    """Adapt an imgstore to the ``cv2.VideoCapture`` subset MultiVideoReader uses.

    Implements :class:`mosaic.core.media.video_io.SupportsCapture`:
    ``isOpened``/``read``/``set``/``get``/``release``. Frames are addressed by
    ``frame_index`` (0-based, contiguous). Sequential reads use imgstore's
    ``get_next_image`` fast path; a pending ``CAP_PROP_POS_FRAMES`` seek triggers
    a single random ``get_image(frame_index=...)`` on the next read. Grayscale
    frames are normalized to 3-channel BGR.
    """

    def __init__(self, path: Path | str):
        self._store: Optional[_Store] = _open_store(path)
        shape = self._store.image_shape
        self._frame_count = int(self._store.frame_count)
        self._height = int(shape[0])
        self._width = int(shape[1])
        duration = float(self._store.duration)
        self._fps = _fps_from_duration(self._frame_count, duration)
        self._pos = 0  # frame_index the next plain read() will return
        self._pending_seek: Optional[int] = None

    def isOpened(self) -> bool:
        return self._store is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        store = self._store
        if store is None:
            return False, None
        if self._pending_seek is not None:
            target = self._pending_seek
            self._pending_seek = None
            if target < 0 or target >= self._frame_count:
                return False, None
            img, _ = store.get_image(-1, frame_index=target)
            self._pos = target + 1
        else:
            if self._pos >= self._frame_count:
                return False, None
            try:
                img, _ = store.get_next_image()
            except EOFError:
                return False, None
            self._pos += 1
        return True, _to_bgr(img)

    def set(self, propId: int, value: float) -> bool:
        if propId == cv2.CAP_PROP_POS_FRAMES:
            self._pending_seek = int(value)
            return True
        return False

    def get(self, propId: int) -> float:
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if propId == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if propId == cv2.CAP_PROP_POS_FRAMES:
            pending = self._pending_seek
            return float(pending if pending is not None else self._pos)
        return 0.0

    def release(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

    def __del__(self):
        self.release()


class ImgStoreFrameReader:
    """High-throughput sequential reader for imgstores.

    Mirrors :class:`mosaic.core.media.video_io.FFmpegFrameReader` (same constructor and
    ``read``/``read_batch``/``__iter__``/``close`` + ``width``/``height``/``fps``/
    ``frame_count``) so tracking inference works unchanged. Backed by
    :class:`ImgStoreCapture`, so contiguous reads use the ``get_next_image`` fast
    path and only discontinuities (``frame_step > 1`` or seeks) pay a random
    access. ``hwaccel`` is accepted for signature parity but is a no-op (imgstore
    decodes via OpenCV).
    """

    def __init__(
        self,
        video_path: Path | str,
        start_frame: int = 0,
        end_frame: int | None = None,
        frame_step: int = 1,
        resize: tuple[int, int] | None = None,
        hwaccel: bool = False,
    ):
        self._path = Path(video_path).expanduser().resolve()
        self._cap = ImgStoreCapture(self._path)
        self._source_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._source_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._source_frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._start_frame = max(0, int(start_frame))
        self._end_frame = (
            min(int(end_frame), self._source_frame_count)
            if end_frame is not None
            else self._source_frame_count
        )
        self._frame_step = max(1, int(frame_step))

        if resize is not None:
            self._width, self._height = int(resize[0]), int(resize[1])
        else:
            self._width, self._height = self._source_width, self._source_height

        self._closed = False
        self._output_indices = list(
            range(self._start_frame, self._end_frame, self._frame_step)
        )
        self._output_pos = 0  # index into _output_indices
        self._next_index = 0  # frame_index the underlying cap will read next

    # ── Properties ──

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        """Number of frames this reader will output."""
        return len(self._output_indices)

    # ── Reading ──

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next selected frame as BGR uint8 ``(height, width, 3)``."""
        if self._closed or self._output_pos >= len(self._output_indices):
            return False, None

        target = self._output_indices[self._output_pos]
        if target != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        self._next_index = target + 1
        self._output_pos += 1

        if (self._width, self._height) != (self._source_width, self._source_height):
            frame = cv2.resize(frame, (self._width, self._height))
        return True, frame

    def read_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Read up to *batch_size* frames → ``(indices (N,), frames (N,H,W,3))``."""
        indices: list[int] = []
        frames: list[np.ndarray] = []
        for _ in range(batch_size):
            ok, frame = self.read()
            if not ok or frame is None:
                break
            indices.append(self._output_indices[self._output_pos - 1])
            frames.append(frame)

        if not frames:
            return np.array([], dtype=np.int64), np.empty(
                (0, self._height, self._width, 3), dtype=np.uint8
            )
        return np.array(indices, dtype=np.int64), np.stack(frames)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` tuples."""
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield self._output_indices[self._output_pos - 1], frame

    # ── Cleanup ──

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args: object):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self.frame_count


# Static structural checks: the adapters satisfy the shared protocols.
_: type[SupportsCapture] = ImgStoreCapture
__: type[FrameReader] = ImgStoreFrameReader
