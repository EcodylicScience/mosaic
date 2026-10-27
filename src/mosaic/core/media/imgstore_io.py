"""Native support for the imgstore video format (Motif / Loopbio).

An imgstore is a *directory* (not a single file): it contains a ``metadata.yaml``
descriptor (with a top-level ``__store`` key) plus zero-padded chunk video/image
files (``000000.mp4`` + ``000000.npz`` index, …). This module lets mosaic treat a
store like any other video source so the rest of the pipeline is unchanged:

* :func:`is_imgstore` — detect a store directory *without* importing imgstore.
* :func:`imgstore_metadata`, :func:`imgstore_facts` and :func:`imgstore_probe` read
  width/height/fps/frames; :func:`measure_store` measures the codec and timing
  that the facts state, and :func:`store_stat` the size and time of the store's
  files.
* :func:`read_store_clock` reads the clock that names a store's divisions.
* :class:`ImgStoreCapture` — adapt a store to the subset of the
  ``cv2.VideoCapture`` API that
  :class:`mosaic.core.media.video_io._ImgStoreMultiReader` relies on, so every
  frame-consuming feature works unchanged.
* :class:`ImgStoreFrameReader` — mirror
  :class:`mosaic_media.io.VideoReader` for high-throughput, batched
  sequential reads (tracking inference).

Frame addressing: mosaic treats the track-table ``frame`` column as the 0-based
*contiguous* video frame index, which maps to imgstore's ``frame_index`` — **not**
the camera-provided, possibly-sparse ``frame_number``. The adapters here address
frames by ``frame_index`` exclusively.

Reading is fully native: this module never imports the ``imgstore`` package.
Detection (:func:`is_imgstore`) parses ``metadata.yaml`` directly, and all
metadata and frame access goes through
:class:`mosaic.core.media.imgstore_native.NativeStore`, which decodes chunks
through numpy / cv2 (raw and image formats) and
:class:`mosaic_media.io.VideoReader` (video formats). The ``imgstore`` package
is only needed to *write* stores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from mosaic_media import MediaFacts, Verdict

from mosaic.core.narrowing import is_object_list, is_str_mapping
from mosaic.core.pipeline.media_index import StoreStat
from mosaic.core.pipeline.stream_copy import header_codec
from mosaic.media_probe_config import media_thresholds
from mosaic.user_paths import user_path

from .facts_columns import ProbeMetadata, facts_to_row, store_facts
from .imgstore_native import NativeStore
from .store_division import DivisionRefused, StoreClock, store_clock
from .video_io import FrameReader, SupportsCapture, SupportsSeekRead, VideoMetadata

# An imgstore never needs transcode negotiation (no client-dependent codec, no
# container/rotation/timing ambiguity), so its verdict carries no reasons and
# both transcode flags are unset.
STORE_VERDICT: Final = Verdict(
    playable=True,
    stream_transcode=None,
    analysis_transcode=None,
    stream_reasons=frozenset(),
    analysis_reasons=frozenset(),
    truncated=False,
)

# imgstore on-disk constants, inlined so detection needs no imgstore import.
_STORE_MD_FILENAME = "metadata.yaml"
_STORE_MD_KEY = "__store"


def fps_from_duration(frame_count: int, duration: float) -> float:
    """Estimate fps from the timestamp span.

    imgstore stores per-frame timestamps but no explicit fps. For ``N`` evenly
    spaced frames the span is ``(N-1) * frame_interval``, so the rate is
    ``(N-1) / duration`` — using ``N / duration`` would overestimate by
    ``N/(N-1)``. Returns ``0.0`` when fps cannot be determined.
    """
    if frame_count > 1 and duration > 0:
        return (frame_count - 1) / duration
    return 0.0


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


STORE_IDENTITY_SCHEME: Final = "imgstore/1"
"""The regime a store's ``video_uuid`` is produced under (open item O5).

Namespaced rather than a bare counter, and that is the whole point: a plain
video's ``identity_scheme`` is ``mosaic-media``'s own counter, meaning a value
*derived* from a packet scan, and this one means a value *declared* by whoever
wrote the store. One column can hold both because the value says which it is; a
reader that cannot tell a mint from a measurement would draw a conclusion neither
supports. The upstream counter is deliberately not quoted here -- it advances
whenever a measured digest moves, so naming a digit would date this paragraph
on the next bump.

**What the namespace advertises is a weakness, so state it.** A store's uuid is
minted once at creation and written into ``metadata.yaml``. It cannot be
re-derived, so a re-probe cannot audit it: chunks edited in place keep the uuid
and the change goes undetected, where a derived ``video_uuid`` would move. What
it *does* buy is a stable per-store identity at the right granularity -- per
store, matching the per-camera media composition (rule P3) -- where before there
was none at all. A store re-recorded into the same directory does get a new uuid,
so that much is detected.

A store still has no ``content_digest``: there is no elementary stream to hash,
duplicate detection over stores is a different question, and inventing a value
here would be the confident-wrong-answer the honest empty exists to avoid.
"""


@dataclass(frozen=True)
class StoreIdentity:
    """Motif/Loopbio identity fields read from a store's ``metadata.yaml``.

    Every field is an empty string for a non-Motif store, or when a key is
    absent or non-string -- which the media index reads as a single-camera
    plain store. ``sync_uuid`` is the document-root ``synchronizationuuid``
    (the recording id shared by every camera of one synchronized recording);
    ``store_uuid`` is ``__store.uuid``, which **differs per store** and is
    therefore the one that can name a single store. That difference used to be
    the reason to avoid ``__store.uuid``; under open item O5 it is the reason to
    read it, into the ``video_uuid`` a store had no value for. The two are not
    interchangeable and neither may stand in for the other: ``sync_uuid`` is
    deliberately shared and safe to collide, ``store_uuid`` is not.

    ``title`` is display-only -- it may contain spaces and differ from the
    directory name -- and is never used for identity.
    """

    camera_serial: str = ""
    sync_uuid: str = ""
    synchronization: str = ""
    title: str = ""
    store_uuid: str = ""


_EMPTY_STORE_IDENTITY = StoreIdentity()


def _store_metadata(path: Path | str) -> Mapping[str, object] | None:
    """Return the parsed ``metadata.yaml`` of a store or its metadata file.

    ``None`` for a path that is not a readable store's metadata.
    """
    p = Path(path)
    md = p / _STORE_MD_FILENAME if p.is_dir() else p
    if md.name != _STORE_MD_FILENAME or not md.is_file():
        return None
    try:
        with md.open("r") as f:
            loaded = yaml.safe_load(f)
    except Exception:
        return None
    return loaded if is_str_mapping(loaded) else None


def imgstore_store_identity(path: Path | str) -> StoreIdentity:
    """Read a store's Motif identity fields import-free from ``metadata.yaml``.

    Reuses the document-root YAML parse of :func:`is_imgstore` (no ``imgstore``
    import), reading the document-**root** keys ``camera_serial``,
    ``synchronizationuuid``, ``synchronization`` and ``title``, plus ``uuid``
    from the nested ``__store`` descriptor -- the one field here that is not a
    root key, because that is where imgstore writes it. A missing key, a
    non-string value, or an unreadable / non-store path yields an empty field; a
    non-Motif store returns all-empty except ``store_uuid``, which every
    imgstore has.
    """
    root = _store_metadata(path)
    if root is None:
        return _EMPTY_STORE_IDENTITY

    def _str(key: str) -> str:
        value = root.get(key)
        return value if isinstance(value, str) else ""

    store_descriptor = root.get(_STORE_MD_KEY)
    store_uuid = ""
    if is_str_mapping(store_descriptor):
        uuid_value = store_descriptor.get("uuid")
        store_uuid = uuid_value if isinstance(uuid_value, str) else ""

    return StoreIdentity(
        camera_serial=_str("camera_serial"),
        sync_uuid=_str("synchronizationuuid"),
        synchronization=_str("synchronization"),
        title=_str("title"),
        store_uuid=store_uuid,
    )


@dataclass(frozen=True)
class StoreMeasurement:
    """A store's facts, measured from its descriptor, chunk indexes and first chunk.

    Attributes:
        width: The width of a decoded frame.
        height: The height of a decoded frame.
        fps: The rate of the chunk indexes' frame times
            (:func:`fps_from_duration`).
        frame_count: The frames of the indexed chunks.
        codec: The codec in the header of the first chunk of a video store, or
            the ``format`` of an image store (``npy``, ``png``).
        duration: The span of the frame times, in seconds.
        constant_frame_rate: Whether every frame time lies within
            ``drift_frame_periods`` of its slot ``i / fps`` from the first.
        max_timestamp_gap_frame_periods: The widest step between neighboring
            frame times, in frame periods.
        max_instantaneous_fps: The rate of the narrowest step, for a store whose
            rate is not constant, and ``None`` for one whose rate is.
    """

    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    duration: float
    constant_frame_rate: bool
    max_timestamp_gap_frame_periods: float
    max_instantaneous_fps: float | None


@dataclass(frozen=True)
class _StoreTiming:
    constant_frame_rate: bool
    max_timestamp_gap_frame_periods: float
    max_instantaneous_fps: float | None


# Fewer than two frame times, or a span of zero, leave no step to measure. The
# values are mosaic-media's for a file whose packets carry no timestamps.
_UNMEASURED_TIMING: Final = _StoreTiming(
    constant_frame_rate=False,
    max_timestamp_gap_frame_periods=0.0,
    max_instantaneous_fps=None,
)


def _store_rate(store: NativeStore) -> float:
    """The frame rate of *store*'s frame times."""
    return fps_from_duration(int(store.frame_count), float(store.duration))


def _measure_timing(times: npt.NDArray[np.float64], fps: float) -> _StoreTiming:
    """Measure frame times *times*, in ``frame_index`` order, against a rate *fps*.

    The measures are the ones ``probe_media`` takes of a file's timestamps, in
    the order the store numbers its frames.
    """
    if times.size < 2 or fps <= 0:
        return _UNMEASURED_TIMING
    slots = np.arange(times.size, dtype=np.float64) / fps
    drift = float(np.max(np.abs((times - times[0]) - slots))) * fps
    constant = drift < media_thresholds().drift_frame_periods
    steps = np.diff(times)
    max_instantaneous_fps: float | None = None
    positive = steps[steps > 0]
    if not constant and positive.size:
        max_instantaneous_fps = 1.0 / float(positive.min())
    return _StoreTiming(
        constant_frame_rate=constant,
        max_timestamp_gap_frame_periods=float(steps.max()) * fps,
        max_instantaneous_fps=max_instantaneous_fps,
    )


def _store_codec(store: NativeStore) -> str:
    """The codec in the first chunk's header of a video store, or an image format."""
    if not store.is_video:
        return store.format
    chunks = store.chunk_paths()
    return header_codec(chunks[0]) if chunks else ""


def measure_store(path: Path | str) -> StoreMeasurement:
    """Measure the store at *path* from its descriptor, chunk indexes and first chunk.

    A video store reads the header of its first chunk, and no chunk is scanned.

    Raises:
        MediaProbeError: If the store does not open.
    """
    with NativeStore(path) as store:
        shape = store.image_shape
        frame_count = int(store.frame_count)
        fps = _store_rate(store)
        timing = _measure_timing(store.frame_times, fps)
        return StoreMeasurement(
            width=int(shape[1]),
            height=int(shape[0]),
            fps=fps,
            frame_count=frame_count,
            codec=_store_codec(store),
            duration=float(store.duration),
            constant_frame_rate=timing.constant_frame_rate,
            max_timestamp_gap_frame_periods=timing.max_timestamp_gap_frame_periods,
            max_instantaneous_fps=timing.max_instantaneous_fps,
        )


_EXTRA_DATA_FILENAME: Final = "000000.extra_data.json"


def _extra_recording_start(path: Path) -> float | None:
    """Return the ``recording_start`` of the first record in a chunk's extra data.

    Motif writes a list of records, one per sample of a data source. ``None``
    when the file is absent or unreadable, or its start is not a number.
    """
    try:
        loaded: object = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    first = loaded[0] if is_object_list(loaded) and loaded else loaded
    if not is_str_mapping(first):
        return None
    start = first.get("recording_start")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    return float(start)


def read_store_clock(path: Path | str) -> StoreClock:
    """Return the clock of the store at *path* (:func:`store_clock`).

    The clock is anchored at the first frame of chunk 0, and reads the store's
    ``metadata.yaml``, its chunk indexes and ``000000.extra_data.json``.

    Raises:
        MediaProbeError: If the store does not open.
        DivisionRefused: If chunk 0 lacks an indexed frame, or no source gives
            the store a start.
    """
    p = Path(path)
    basedir = p if p.is_dir() else p.parent
    with NativeStore(basedir) as store:
        table = store.chunk_table()
    first = table[0] if table and table[0].chunk_id == 0 else None
    if first is None or first.first_time is None:
        message = (
            f"chunk 0 of the store {basedir.name!r} lacks an indexed frame to "
            "anchor its clock at"
        )
        raise DivisionRefused("no_clock_source", message)
    metadata = _store_metadata(basedir) or {}
    return store_clock(
        metadata,
        basedir.name,
        first.first_time,
        _extra_recording_start(basedir / _EXTRA_DATA_FILENAME),
    )


def store_stat(path: Path | str) -> StoreStat:
    """Return the size and newest modification time of the files of a store.

    The files are ``metadata.yaml`` and every chunk file at the top of the store,
    each named for its chunk number (``000000.mp4``, ``000000.npz``,
    ``000000.extra_data.json``), and the index of each chunk directory of an
    image store (``000000/index.npz``). An image store's frame files are not
    read. A resource fork (``._000000.mp4``) is not named for a chunk number and is
    skipped.

    Raises:
        OSError: If the store's directory cannot be listed.
    """
    basedir = Path(path)
    covered: list[Path] = []
    for entry in sorted(basedir.iterdir()):
        name = entry.name
        if name == _STORE_MD_FILENAME:
            covered.append(entry)
        elif name.isdigit() and entry.is_dir():
            covered.extend(
                index
                for index in (entry / "index.npz", entry / "index.yaml")
                if index.is_file()
            )
        elif name.split(".", 1)[0].isdigit() and entry.is_file():
            covered.append(entry)
    stats = [file.stat() for file in covered]
    return StoreStat(
        size=sum(stat.st_size for stat in stats),
        mtime=max((stat.st_mtime for stat in stats), default=0.0),
    )


def imgstore_metadata(path: Path | str) -> VideoMetadata:
    """Read :class:`VideoMetadata` (width/height/fps/frame_count) from a store.

    ``fps`` is derived as ``(frame_count - 1) / duration`` (imgstore does not
    store an explicit fps; see :func:`fps_from_duration`); it is ``0.0`` for
    stores with no usable timestamps.
    """
    p = user_path(path).resolve()
    with NativeStore(p) as store:
        shape = store.image_shape
        return VideoMetadata(
            path=p,
            width=int(shape[1]),
            height=int(shape[0]),
            fps=_store_rate(store),
            frame_count=int(store.frame_count),
        )


def imgstore_facts(path: Path | str) -> MediaFacts:
    """Read a store's :class:`MediaFacts`, measured by :func:`measure_store`.

    The ``video_uuid`` is the store's ``__store.uuid`` under
    :data:`STORE_IDENTITY_SCHEME`, or ``""`` for both when the metadata records
    none. Raises :class:`~mosaic_media.MediaProbeError` on a store that does not
    open.
    """
    m = measure_store(path)
    # A second, cheap YAML read rather than plumbing the descriptor out of
    # NativeStore: that class parses ``__store`` for format/encoding/class/shape
    # and keeps none of it, and widening it to carry an identity would put a
    # decode-path type in the identity path for one string.
    store_uuid = imgstore_store_identity(path).store_uuid
    return store_facts(
        width=m.width,
        height=m.height,
        fps=m.fps,
        frame_count=m.frame_count,
        codec=m.codec,
        duration=m.duration,
        video_uuid=store_uuid,
        identity_scheme=STORE_IDENTITY_SCHEME if store_uuid else "",
        constant_frame_rate=m.constant_frame_rate,
        max_timestamp_gap_frame_periods=m.max_timestamp_gap_frame_periods,
        max_instantaneous_fps=m.max_instantaneous_fps,
    )


def imgstore_probe(path: Path | str) -> ProbeMetadata:
    """Probe a store, returning a dict shaped like ``probe_video_metadata``.

    Keys: ``width``, ``height``, ``fps``, ``codec``, plus the injectable
    MediaFacts JSON and flat verdict columns of :func:`imgstore_facts` and
    :data:`STORE_VERDICT` (see :mod:`mosaic.core.media.facts_columns`;
    ``frame_count`` is one of them, so it is not set separately here). Used by
    ``Dataset.index_media`` to populate ``media/index.csv``.
    """
    facts = imgstore_facts(path)
    return {
        "width": facts.width,
        "height": facts.height,
        "fps": facts.fps,
        "codec": facts.codec_name,
        **facts_to_row(facts, STORE_VERDICT),
    }


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalize a decoded frame to 3-channel BGR (cv2.VideoCapture convention)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


class ImgStoreCapture:
    """Adapt an imgstore to the ``cv2.VideoCapture`` subset _ImgStoreMultiReader uses.

    Implements :class:`mosaic.core.media.video_io.SupportsCapture`:
    ``isOpened``/``read``/``set``/``get``/``release``. It also implements the
    lean :class:`mosaic.core.media.video_io.SupportsSeekRead` surface
    (``seek``/``read``/``close`` plus ``width``/``height``/``fps``/``frame_count``)
    shared with :class:`mosaic_media.io.VideoReader`, so callers that only need
    absolute-frame seeking and sequential reads can use either interchangeably.
    Frames are addressed by ``frame_index`` (0-based, contiguous). Sequential
    reads use :meth:`NativeStore.next_frame`; a pending seek (whether set through
    ``seek`` or ``set(CAP_PROP_POS_FRAMES, ...)``) triggers a single random
    :meth:`NativeStore.frame` on the next read. Grayscale frames are normalized
    to 3-channel BGR.
    """

    def __init__(self, path: Path | str):
        self._store: NativeStore | None = NativeStore(path)
        shape = self._store.image_shape
        self._frame_count = int(self._store.frame_count)
        self._height = int(shape[0])
        self._width = int(shape[1])
        self._fps = _store_rate(self._store)
        self._pos = 0  # frame_index the next plain read() will return
        self._pending_seek: int | None = None

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
            img = store.frame(target)
            self._pos = target + 1
        else:
            if self._pos >= self._frame_count:
                return False, None
            try:
                img = store.next_frame()
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

    # --- Lean SupportsSeekRead surface (no CAP_PROP idiom) ---

    def seek(self, frame_index: int) -> None:
        self._pending_seek = int(frame_index)

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
        return self._frame_count

    def close(self) -> None:
        self.release()


class ImgStoreFrameReader:
    """High-throughput sequential reader for imgstores.

    Mirrors :class:`mosaic_media.io.VideoReader` (same constructor and
    ``read``/``read_batch``/``__iter__``/``close`` + ``width``/``height``/``fps``/
    ``frame_count``) so tracking inference works unchanged. Backed by
    :class:`ImgStoreCapture`, so contiguous reads use the
    :meth:`NativeStore.next_frame` fast path and only discontinuities
    (``frame_step > 1`` or seeks) pay a random access. ``hwaccel`` is accepted
    for signature parity but is a no-op (imgstore decodes via OpenCV / VideoReader).
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
        self._path = user_path(video_path).resolve()
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
___: type[SupportsSeekRead] = ImgStoreCapture
