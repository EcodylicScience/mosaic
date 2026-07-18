"""Native reader for the imgstore on-disk format (Motif / Loopbio).

An imgstore is a *directory* holding a ``metadata.yaml`` descriptor (under a
top-level ``__store`` key) plus zero-padded chunk files. Two on-disk layouts
exist:

* ``DirectoryImgStore`` writes one image file per frame into a per-chunk
  directory (``000000/000000.npy`` for the raw ``npy`` format, or
  ``000000/000000.png`` and friends for cv2 image formats), plus a per-chunk
  ``index.npz`` mapping within-chunk position to ``frame_number`` /
  ``frame_time``.
* ``VideoImgStore`` writes one video file per chunk (``000000.avi`` /
  ``.mkv`` / ``.mp4``) plus a sibling ``000000.npz`` index.

:class:`NativeStore` reads either layout without importing the ``imgstore``
package: raw/image chunks decode through numpy / cv2, and video chunks decode
through :class:`mosaic_media.io.VideoReader`.

Frame addressing follows mosaic's convention: the 0-based *contiguous*
``frame_index`` (imgstore's ``frame_index``), which is the running position
across chunks in ascending chunk order -- **not** the camera-provided,
possibly-sparse ``frame_number``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
from mosaic_media import MediaProbeError
from mosaic_media.io import VideoReader

# imgstore on-disk constants, inlined so reading needs no imgstore package.
_STORE_MD_FILENAME = "metadata.yaml"
_STORE_MD_KEY = "__store"
_FRAME_TIME_KEY = "frame_time"
_INDEX_EXTENSIONS = (".npz", ".yaml")

# Store classes. Loopbio's internal recorder tags video stores as
# "VideoImgStoreFFMPEG"; imgstore reads those with the plain video code path.
_DIRECTORY_STORE_CLASS = "DirectoryImgStore"
_VIDEO_STORE_CLASSES = frozenset({"VideoImgStore", "VideoImgStoreFFMPEG"})

# DirectoryImgStore image formats read through cv2.imread; "npy" through
# np.load. The "+color" variant is stripped from the stored format by the
# writer, so the metadata format is always one of these base names.
_NUMPY_IMAGE_FORMAT = "npy"
_CV2_IMAGE_FORMATS = frozenset({"tif", "png", "jpg", "ppm", "pgm", "bmp"})

# Matches cv2's "COLOR_<codec>2BGR[_<method>]" enum names.
_CV2_CODEC_PATTERN = re.compile(r"^COLOR_(?P<codec>[A-Za-z0-9_]+)2BGR($|(?P<method>_[A-Za-z0-9]*))")


def _build_encoding_enum_by_code() -> dict[str, int]:
    """Build the ``cv_<code> -> cv2 color-conversion enum`` mapping.

    Mirrors imgstore's ``ImageCodecProcessor.__init__`` (imgstore/util.py):
    scans ``dir(cv2)`` for ``COLOR_<codec>2BGR[_<method>]`` names and maps
    ``cv_<codec-lowercased-without-underscores>[_<method>]`` to the enum
    value, so a store's ``encoding`` string (as written by imgstore, e.g.
    ``cv_bayerrg`` or ``cv_yuv420p``) resolves to the same ``cv2.cvtColor``
    code imgstore itself would use. Method-suffixed bayer variants are
    skipped, matching imgstore's rule of keeping only the method-less bayer
    code per codec.
    """
    mapping: dict[str, int] = {}
    for name in dir(cv2):
        match = _CV2_CODEC_PATTERN.match(name)
        if match is None:
            continue
        enum = int(getattr(cv2, name))
        parts = match.groupdict()
        code = "cv_" + parts["codec"].lower().replace("_", "")
        method = parts.get("method")
        if method:
            if "bayer" in code:
                continue
            if method == "full":
                continue
            code += "_" + method.lower().replace("_", "")
        mapping[code] = enum
    return mapping


_ENCODING_ENUM_BY_CODE: dict[str, int] = _build_encoding_enum_by_code()


def _resolve_store_paths(path: Path | str) -> tuple[Path, Path]:
    """Return ``(basedir, metadata_path)`` for a store dir or its metadata file."""
    p = Path(path)
    if p.is_dir():
        basedir = p
        metadata_path = p / _STORE_MD_FILENAME
    elif p.name == _STORE_MD_FILENAME:
        basedir = p.parent
        metadata_path = p
    else:
        message = f"not an imgstore (no {_STORE_MD_FILENAME}): {p}"
        raise MediaProbeError(message)
    if not metadata_path.is_file():
        message = f"imgstore metadata not found: {metadata_path}"
        raise MediaProbeError(message)
    return basedir, metadata_path


def _ensure_mapping(value: object, source: Path) -> None:
    """Raise :class:`MediaProbeError` unless *value* is a mapping."""
    if not isinstance(value, dict):
        message = f"malformed imgstore metadata (expected a mapping): {source}"
        raise MediaProbeError(message)


def _ensure_sequence(value: object, source: Path) -> None:
    """Raise :class:`MediaProbeError` unless *value* is a list or tuple."""
    if not isinstance(value, (list, tuple)):
        message = f"malformed imgstore imgshape (expected a sequence): {source}"
        raise MediaProbeError(message)


def _load_chunk_frame_times(index_base: Path) -> list[float] | None:
    """Read the ``frame_time`` array from a chunk index (``.npz`` or ``.yaml``).

    *index_base* is the index path without its extension (``.../000000/index``
    for a directory store, ``.../000000`` for a video store). The length of the
    returned list is the number of frames in the chunk.

    Returns ``None`` if neither index file exists: an interrupted or crashed
    recording can leave its final chunk without a written index, and imgstore
    skips such a chunk rather than failing the whole store. Raises
    :class:`MediaProbeError` if an index file exists but is malformed or
    unreadable -- that is corruption, not an interrupted recording.
    """
    npz_path = Path(f"{index_base}{_INDEX_EXTENSIONS[0]}")
    if npz_path.is_file():
        with npz_path.open("rb") as f:
            data = np.load(f)
            if _FRAME_TIME_KEY not in data.files:
                message = f"chunk index missing {_FRAME_TIME_KEY}: {npz_path}"
                raise MediaProbeError(message)
            return [float(v) for v in data[_FRAME_TIME_KEY].tolist()]
    yaml_path = Path(f"{index_base}{_INDEX_EXTENSIONS[1]}")
    if yaml_path.is_file():
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)
        _ensure_mapping(data, yaml_path)
        times = data.get(_FRAME_TIME_KEY)
        if times is None:
            message = f"chunk index missing {_FRAME_TIME_KEY}: {yaml_path}"
            raise MediaProbeError(message)
        return [float(v) for v in times]
    return None


def _chunk_extension(descriptor: dict[str, object]) -> str:
    """Return a video store's chunk extension (``.avi`` / ``.mkv`` / ``.mp4``)."""
    extension = descriptor.get("extension")
    if isinstance(extension, str) and extension:
        return extension
    # Backward compatibility with old stores that predate the "extension" key.
    if descriptor.get("format") == "mjpeg":
        return ".avi"
    return ".mp4"


class NativeStore:
    """Random-access reader for a single imgstore, without the imgstore package.

    Frames are addressed by the 0-based contiguous ``frame_index``. Raw and cv2
    image chunks decode through numpy / cv2; video chunks decode through
    :class:`mosaic_media.io.VideoReader`, holding one reader for the current
    chunk and rolling to the next at the chunk boundary. Frames are returned in
    the same dtype / shape / channel order the imgstore package returns, so
    downstream BGR normalization is unaffected.
    """

    def __init__(self, path: Path | str) -> None:
        # Video-chunk reader state, set first so close() (reachable from
        # __del__) is safe even if construction raises before the rest of
        # __init__ runs (e.g. probing a malformed store).
        self._video_reader: VideoReader | None = None
        self._video_chunk_id: int | None = None
        self._video_next_offset = 0

        self._basedir, metadata_path = _resolve_store_paths(path)

        try:
            with metadata_path.open("r") as f:
                parsed = yaml.safe_load(f)
        except Exception as exc:
            message = f"could not read imgstore metadata: {metadata_path}"
            raise MediaProbeError(message) from exc
        _ensure_mapping(parsed, metadata_path)
        descriptor = parsed.get(_STORE_MD_KEY)
        # A missing __store key yields None, which is not a mapping.
        _ensure_mapping(descriptor, metadata_path)

        self._format = str(descriptor.get("format", "") or "")
        encoding = descriptor.get("encoding")
        self._encoding = encoding if isinstance(encoding, str) and encoding else None
        store_class = str(descriptor.get("class", "") or "")
        self._is_video = store_class in _VIDEO_STORE_CLASSES

        raw_shape = descriptor.get("imgshape", ())
        _ensure_sequence(raw_shape, metadata_path)
        img_shape = tuple(int(v) for v in raw_shape)
        if len(img_shape) < 2:
            message = f"malformed imgstore imgshape (need at least 2 dims): {metadata_path}"
            raise MediaProbeError(message)
        if self._is_video:
            # Video encoders only write even dimensions -- VideoImgStore
            # truncates height/width down to even (`int(x) & -2`) at write
            # time, and re-applies the same correction on read for legacy
            # stores whose metadata.yaml still holds odd values. Mirror that
            # so the reported shape matches the actual decoded frame size.
            img_shape = (img_shape[0] & -2, img_shape[1] & -2, *img_shape[2:])
        # A store native color frame has a trailing 3-channel axis.
        self._color = (img_shape[-1] == 3) and (len(img_shape) == 3)
        # An encoded store always decodes to 3-channel BGR (mirrors imgstore's
        # image_shape property).
        if self._encoding is not None:
            self._image_shape: tuple[int, ...] = (img_shape[0], img_shape[1], 3)
        else:
            self._image_shape = img_shape

        self._video_ext = _chunk_extension(descriptor) if self._is_video else ""

        # Build the global frame_index -> (chunk_id, within-chunk offset) map and
        # collect per-frame timestamps, iterating chunks in ascending order.
        self._frame_map: list[tuple[int, int]] = []
        frame_times: list[float] = []
        for chunk_id in self._discover_chunks(store_class):
            index_base = self._chunk_index_base(chunk_id)
            times = _load_chunk_frame_times(index_base)
            if times is None:
                message = f"missing index for imgstore chunk {chunk_id}, skipping: {index_base}"
                warnings.warn(message, stacklevel=2)
                continue
            for offset in range(len(times)):
                self._frame_map.append((chunk_id, offset))
            frame_times.extend(times)

        self.frame_count = len(self._frame_map)
        if frame_times:
            self.duration = max(frame_times) - min(frame_times)
        else:
            self.duration = 0.0

        # Sequential cursor: the frame_index the next next_frame() returns.
        self._cursor = 0

    # --- Public metadata surface ---

    @property
    def image_shape(self) -> tuple[int, ...]:
        return self._image_shape

    @property
    def format(self) -> str:
        return self._format

    # --- Chunk discovery ---

    def _discover_chunks(self, store_class: str) -> list[int]:
        if self._is_video:
            ids = [
                int(entry.stem)
                for entry in self._basedir.iterdir()
                if entry.is_file()
                and entry.suffix == self._video_ext
                and entry.stem.isdigit()
            ]
        elif store_class == _DIRECTORY_STORE_CLASS:
            ids = [
                int(entry.name)
                for entry in self._basedir.iterdir()
                if entry.is_dir() and entry.name.isdigit()
            ]
        else:
            message = f"unsupported imgstore class: {store_class!r}"
            raise MediaProbeError(message)
        return sorted(ids)

    def _chunk_index_base(self, chunk_id: int) -> Path:
        if self._is_video:
            return self._basedir / f"{chunk_id:06d}"
        return self._basedir / f"{chunk_id:06d}" / "index"

    # --- Reading ---

    def frame(self, frame_index: int) -> np.ndarray:
        """Return the frame at the 0-based contiguous *frame_index*."""
        if frame_index < 0 or frame_index >= self.frame_count:
            message = f"frame_index {frame_index} out of range (0, {self.frame_count})"
            raise MediaProbeError(message)
        chunk_id, offset = self._frame_map[frame_index]
        if self._is_video:
            image = self._decode_video(chunk_id, offset)
        else:
            image = self._decode_directory(chunk_id, offset)
        self._cursor = frame_index + 1
        return self._apply_encoding(image)

    def next_frame(self) -> np.ndarray:
        """Return the next frame in sequence, advancing the internal cursor."""
        if self._cursor >= self.frame_count:
            raise EOFError
        return self.frame(self._cursor)

    def _decode_directory(self, chunk_id: int, offset: int) -> np.ndarray:
        path = self._basedir / f"{chunk_id:06d}" / f"{offset:06d}.{self._format}"
        if not path.is_file():
            message = f"imgstore frame file not found: {path}"
            raise MediaProbeError(message)
        if self._format == _NUMPY_IMAGE_FORMAT:
            return np.load(path)
        if self._format in _CV2_IMAGE_FORMATS:
            flags = cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
            image = cv2.imread(str(path), flags)
            if image is None:
                message = f"could not decode imgstore frame: {path}"
                raise MediaProbeError(message)
            return image
        message = f"unsupported imgstore directory format: {self._format!r}"
        raise MediaProbeError(message)

    def _decode_video(self, chunk_id: int, offset: int) -> np.ndarray:
        if chunk_id != self._video_chunk_id:
            self._close_video_reader()
            chunk_path = self._basedir / f"{chunk_id:06d}{self._video_ext}"
            if not chunk_path.is_file():
                message = f"imgstore video chunk not found: {chunk_path}"
                raise MediaProbeError(message)
            self._video_reader = VideoReader(chunk_path)
            self._video_chunk_id = chunk_id
            self._video_next_offset = 0
        reader = self._video_reader
        if reader is None:
            message = f"imgstore video chunk reader unavailable: {chunk_id}"
            raise MediaProbeError(message)
        if offset != self._video_next_offset:
            reader.seek(offset)
            self._video_next_offset = offset
        ok, image = reader.read()
        if not ok or image is None:
            message = f"could not decode imgstore video frame: chunk {chunk_id} offset {offset}"
            raise MediaProbeError(message)
        self._video_next_offset = offset + 1
        # imgstore normalizes video frames to the store's native channel count.
        if not self._color and image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _apply_encoding(self, image: np.ndarray) -> np.ndarray:
        if self._encoding is None:
            return image
        enum = _ENCODING_ENUM_BY_CODE.get(self._encoding)
        if enum is None:
            message = (
                "imgstore store has an unrecognized image encoding "
                f"({self._encoding!r}); the native reader could not resolve it to a cv2 color code"
            )
            raise MediaProbeError(message)
        return cv2.cvtColor(image, enum)

    # --- Cleanup ---

    def _close_video_reader(self) -> None:
        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None
            self._video_chunk_id = None

    def close(self) -> None:
        self._close_video_reader()

    def __enter__(self) -> NativeStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
