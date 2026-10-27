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
* :func:`probe_store` reads a store once, and :func:`window_probe` gives the
  facts, size and time and ``video_uuid`` of the whole store or of one window of
  its chunks (:class:`~mosaic.core.media.store_division.StoreWindow`) from that
  read.
* :func:`read_store_clock` reads the clock that names a store's divisions, and
  :func:`store_plan_input` gives a probed store as the division planner reads
  it.
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
frames by ``frame_index`` exclusively. An adapter opened on a window counts the
window's frames from 0.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from mosaic_media import MediaFacts, MediaProbeError, Verdict

from mosaic.core.narrowing import is_object_list, is_str_mapping
from mosaic.core.pipeline.media_index import StoreStat, stat_cells
from mosaic.core.pipeline.stream_copy import header_codec
from mosaic.media_probe_config import media_thresholds
from mosaic.user_paths import user_path

from .facts_columns import ProbeMetadata, facts_to_row, store_facts
from .imgstore_native import NativeStore
from .store_division import (
    DIVISION_IDENTITY_SCHEME,
    ChunkRecord,
    DivisionRefused,
    StoreClock,
    StorePlanInput,
    StoreWindow,
    division_uid,
    store_clock,
)
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


@dataclass(frozen=True, slots=True)
class StoreFile:
    """One file of a store that its size and time cover (:func:`store_stat`).

    Attributes:
        chunk_id: The chunk that the file belongs to, ``None`` for
            ``metadata.yaml``.
        size: The file's size in bytes.
        mtime: The file's modification time, in seconds since the epoch.
    """

    chunk_id: int | None
    size: int
    mtime: float


@dataclass(frozen=True, eq=False)
class StoreProbe:
    """One read of a store: its identity, its chunks and the files they are in.

    :func:`window_probe` measures the whole store, or one window of its chunks,
    from it without reading the store again.

    Attributes:
        path: The store's directory.
        identity: The store's Motif identity fields.
        width: The width of a decoded frame.
        height: The height of a decoded frame.
        codec: As :attr:`StoreMeasurement.codec` states it.
        chunksize: The frames per chunk that the store declares, ``0`` without.
        chunks: Every chunk of the store (:meth:`NativeStore.chunk_table`).
        chunk_times: The frame times of each indexed chunk, by chunk id.
        files: The files that the store's size and time cover.
        metadata: The parsed ``metadata.yaml``.
    """

    path: Path
    identity: StoreIdentity
    width: int
    height: int
    codec: str
    chunksize: int
    chunks: tuple[ChunkRecord, ...]
    chunk_times: Mapping[int, npt.NDArray[np.float64]]
    files: tuple[StoreFile, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class WindowProbe:
    """The measured facts of a store, or of one window of its chunks.

    Attributes:
        facts: The facts. A window's frame count, duration, rate and timing are
            its own, measured from its chunks' frame times, and its
            ``video_uuid`` is :attr:`uid`.
        stat: The size and time of the files of the window's chunks, or of the
            store (:func:`store_stat`).
        uid: The store's ``__store.uuid``, or the window's division uid
            (:func:`~mosaic.core.media.store_division.division_uid`).
    """

    facts: MediaFacts
    stat: StoreStat
    uid: str

    def probe_metadata(self) -> ProbeMetadata:
        """Return the probe cells (:func:`imgstore_probe`) of the facts."""
        return {
            "width": self.facts.width,
            "height": self.facts.height,
            "fps": self.facts.fps,
            "codec": self.facts.codec_name,
            **facts_to_row(self.facts, STORE_VERDICT),
        }

    def row_cells(self) -> dict[str, object]:
        """Return the measured media index cells: size, time, dimensions and facts."""
        size_bytes, modified_at = stat_cells(self.stat)
        return {
            "size_bytes": size_bytes,
            "mtime_iso": modified_at,
            **self.probe_metadata(),
        }


def probe_store(path: Path | str) -> StoreProbe:
    """Read the store at *path* once, for :func:`window_probe`.

    It reads the store's metadata, every chunk index, the header of a video
    store's first chunk and the stat of each file that :func:`store_stat`
    covers.

    Raises:
        MediaProbeError: If the store does not open.
        OSError: If the store's directory cannot be listed.
    """
    p = Path(path)
    basedir = p if p.is_dir() else p.parent
    with NativeStore(basedir) as store:
        shape = store.image_shape
        chunks = store.chunk_table()
        chunk_times = {
            chunk.chunk_id: store.chunk_frame_times(chunk.chunk_id)
            for chunk in chunks
            if chunk.frame_count is not None
        }
        codec = _store_codec(store)
        chunksize = store.chunksize
    return StoreProbe(
        path=basedir,
        identity=imgstore_store_identity(basedir),
        width=int(shape[1]),
        height=int(shape[0]),
        codec=codec,
        chunksize=chunksize,
        chunks=chunks,
        chunk_times=chunk_times,
        files=tuple(_store_files(basedir)),
        metadata=_store_metadata(basedir) or {},
    )


def _window_times(
    probe: StoreProbe, window: StoreWindow | None
) -> npt.NDArray[np.float64]:
    """The frame times of *window*'s chunks in order, or of every indexed chunk.

    Raises:
        MediaProbeError: If a chunk of *window* is absent or has no index.
    """
    if window is None:
        ids = sorted(probe.chunk_times)
    else:
        ids = list(range(window.chunk_first, window.chunk_last + 1))
        missing = [chunk_id for chunk_id in ids if chunk_id not in probe.chunk_times]
        if missing:
            message = (
                f"chunk {missing[0]} of the imgstore {probe.path} is absent or has "
                f"no index, and the window of chunks {window.chunk_first}-"
                f"{window.chunk_last} reads it"
            )
            raise MediaProbeError(message)
    if not ids:
        return np.empty(0, np.float64)
    return np.concatenate([probe.chunk_times[chunk_id] for chunk_id in ids])


def _measured(probe: StoreProbe, window: StoreWindow | None) -> StoreMeasurement:
    """Measure the frames of *window*, or of the whole store, from *probe*."""
    times = _window_times(probe, window)
    duration = float(times.max() - times.min()) if times.size else 0.0
    fps = fps_from_duration(int(times.size), duration)
    timing = _measure_timing(times, fps)
    return StoreMeasurement(
        width=probe.width,
        height=probe.height,
        fps=fps,
        frame_count=int(times.size),
        codec=probe.codec,
        duration=duration,
        constant_frame_rate=timing.constant_frame_rate,
        max_timestamp_gap_frame_periods=timing.max_timestamp_gap_frame_periods,
        max_instantaneous_fps=timing.max_instantaneous_fps,
    )


def _window_uid(probe: StoreProbe, window: StoreWindow, frame_count: int) -> str:
    """The division uid of *window*, refusing a store without a uuid."""
    store_uuid = probe.identity.store_uuid
    if not store_uuid:
        message = (
            f"the imgstore {probe.path} records no __store.uuid to name the "
            f"division of chunks {window.chunk_first}-{window.chunk_last} by"
        )
        raise MediaProbeError(message)
    return division_uid(store_uuid, window.chunk_first, window.chunk_last, frame_count)


def window_probe(probe: StoreProbe, window: StoreWindow | None) -> WindowProbe:
    """Return the facts, size and time and uid of *window* of a probed store.

    ``None`` measures the whole store, whose ``video_uuid`` is its
    ``__store.uuid`` under :data:`STORE_IDENTITY_SCHEME`, or ``""`` for both
    when the store records none. A window's rate is ``(n - 1) / span`` over its
    own frame times, and its ``video_uuid`` is its division uid under
    :data:`~mosaic.core.media.store_division.DIVISION_IDENTITY_SCHEME`.

    Raises:
        MediaProbeError: If a chunk of *window* is absent or has no index, or
            the store records no ``__store.uuid`` to name a window by.
    """
    measured = _measured(probe, window)
    if window is None:
        uid = probe.identity.store_uuid
        scheme = STORE_IDENTITY_SCHEME if uid else ""
    else:
        uid = _window_uid(probe, window, measured.frame_count)
        scheme = DIVISION_IDENTITY_SCHEME
    facts = store_facts(
        width=measured.width,
        height=measured.height,
        fps=measured.fps,
        frame_count=measured.frame_count,
        codec=measured.codec,
        duration=measured.duration,
        video_uuid=uid,
        identity_scheme=scheme,
        constant_frame_rate=measured.constant_frame_rate,
        max_timestamp_gap_frame_periods=measured.max_timestamp_gap_frame_periods,
        max_instantaneous_fps=measured.max_instantaneous_fps,
    )
    return WindowProbe(facts=facts, stat=_files_stat(probe.files, window), uid=uid)


def measure_store(path: Path | str) -> StoreMeasurement:
    """Measure the store at *path* from its descriptor, chunk indexes and first chunk.

    A video store reads the header of its first chunk, and no chunk is scanned.

    Raises:
        MediaProbeError: If the store does not open.
    """
    return _measured(probe_store(path), None)


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


def _clock(
    basedir: Path, table: tuple[ChunkRecord, ...], metadata: Mapping[str, object]
) -> StoreClock:
    """The clock of the store at *basedir* whose chunks are *table*."""
    first = table[0] if table and table[0].chunk_id == 0 else None
    if first is None or first.first_time is None:
        message = (
            f"chunk 0 of the store {basedir.name!r} lacks an indexed frame to "
            "anchor its clock at"
        )
        raise DivisionRefused("no_clock_source", message)
    return store_clock(
        metadata,
        basedir.name,
        first.first_time,
        _extra_recording_start(basedir / _EXTRA_DATA_FILENAME),
    )


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
    return _clock(basedir, table, _store_metadata(basedir) or {})


def store_plan_input(probe: StoreProbe, abs_path: str) -> StorePlanInput:
    """Return the probed store as :func:`~mosaic.core.media.store_division.plan_division` reads it.

    Args:
        probe: The store, read once.
        abs_path: The store's path as its media index rows record it.

    Raises:
        DivisionRefused: As :func:`read_store_clock` raises it.
    """
    return StorePlanInput(
        abs_path=abs_path,
        store_uuid=probe.identity.store_uuid,
        camera_serial=probe.identity.camera_serial,
        chunksize=probe.chunksize,
        fps=_measured(probe, None).fps,
        clock=_clock(probe.path, probe.chunks, probe.metadata),
        chunks=probe.chunks,
    )


def _chunk_of(name: str) -> int | None:
    """The chunk number that a store file's name starts with, ``None`` without one."""
    head = name.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _store_files(basedir: Path) -> list[StoreFile]:
    """The files that :func:`store_stat` covers, each with its chunk and its stat."""
    covered: list[tuple[int | None, Path]] = []
    for entry in sorted(basedir.iterdir()):
        name = entry.name
        if name == _STORE_MD_FILENAME:
            covered.append((None, entry))
        elif name.isdigit() and entry.is_dir():
            covered.extend(
                (int(name), index)
                for index in (entry / "index.npz", entry / "index.yaml")
                if index.is_file()
            )
        elif (chunk_id := _chunk_of(name)) is not None and entry.is_file():
            covered.append((chunk_id, entry))
    files: list[StoreFile] = []
    for chunk_id, file in covered:
        stat = file.stat()
        files.append(StoreFile(chunk_id, stat.st_size, stat.st_mtime))
    return files


def _files_stat(files: Sequence[StoreFile], window: StoreWindow | None) -> StoreStat:
    """The size and newest time of *files*, or of the files of *window*'s chunks."""
    if window is None:
        kept = list(files)
    else:
        kept = [
            file
            for file in files
            if file.chunk_id is not None
            and window.chunk_first <= file.chunk_id <= window.chunk_last
        ]
    return StoreStat(
        size=sum(file.size for file in kept),
        mtime=max((file.mtime for file in kept), default=0.0),
    )


def store_stat(path: Path | str, window: StoreWindow | None = None) -> StoreStat:
    """Return the size and newest modification time of the files of a store.

    The files are ``metadata.yaml`` and every chunk file at the top of the store,
    each named for its chunk number (``000000.mp4``, ``000000.npz``,
    ``000000.extra_data.json``), and the index of each chunk directory of an
    image store (``000000/index.npz``). An image store's frame files are not
    read. A resource fork (``._000000.mp4``) is not named for a chunk number and is
    skipped.

    A *window* covers the files of its chunks alone and leaves out
    ``metadata.yaml``, which a growing store rewrites, so a closed window keeps
    its size and time.

    Raises:
        OSError: If the store's directory cannot be listed.
    """
    return _files_stat(_store_files(Path(path)), window)


def imgstore_metadata(
    path: Path | str, window: StoreWindow | None = None
) -> VideoMetadata:
    """Read :class:`VideoMetadata` (width/height/fps/frame_count) from a store.

    ``fps`` is derived as ``(frame_count - 1) / duration`` (imgstore does not
    store an explicit fps; see :func:`fps_from_duration`); it is ``0.0`` for
    stores with no usable timestamps. A *window* reads the frames of its chunks
    alone.
    """
    p = user_path(path).resolve()
    with NativeStore(p, window) as store:
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
    return window_probe(probe_store(path), None).facts


def imgstore_probe(
    path: Path | str, window: StoreWindow | None = None
) -> ProbeMetadata:
    """Probe a store, returning a dict shaped like ``probe_video_metadata``.

    Keys: ``width``, ``height``, ``fps``, ``codec``, plus the injectable
    MediaFacts JSON and flat verdict columns of :func:`window_probe` and
    :data:`STORE_VERDICT` (see :mod:`mosaic.core.media.facts_columns`;
    ``frame_count`` is one of them, so it is not set separately here). A
    *window* measures its chunks alone. Used by ``Dataset.index_media`` to
    populate ``media/index.csv``.

    Raises:
        MediaProbeError: As :func:`window_probe` raises it, or if the store
            does not open.
    """
    return window_probe(probe_store(path), window).probe_metadata()


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
    to 3-channel BGR. A *window* reads its chunks alone (:class:`NativeStore`).
    """

    def __init__(self, path: Path | str, window: StoreWindow | None = None):
        self._store: NativeStore | None = NativeStore(path, window)
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
    A *window* reads its chunks alone, frame 0 being the window's first frame.
    """

    def __init__(
        self,
        video_path: Path | str,
        start_frame: int = 0,
        end_frame: int | None = None,
        frame_step: int = 1,
        resize: tuple[int, int] | None = None,
        hwaccel: bool = False,
        window: StoreWindow | None = None,
    ):
        self._path = user_path(video_path).resolve()
        self._cap = ImgStoreCapture(self._path, window)
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
