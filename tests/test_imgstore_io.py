"""Tests for native imgstore support (detection, metadata, readers)."""

from __future__ import annotations

import importlib
import sys

import cv2
import numpy as np
import pytest
import yaml

from mosaic.core.media.imgstore_io import (
    ImgStoreCapture,
    ImgStoreFrameReader,
    imgstore_metadata,
    imgstore_probe,
    is_imgstore,
)
from mosaic.core.media.imgstore_native import NativeStore
from mosaic.core.media.video_io import (
    MultiVideoReader,
    get_video_metadata,
    open_frame_reader,
)
from mosaic_media import MediaProbeError
from mosaic_media.io import VideoReader

# Only the fixtures that write real stores need the imgstore package; the
# imports above are import-free at module scope, so this skip gate can follow
# them instead of forcing them below it.
pytest.importorskip("imgstore")


def _tag(frame: np.ndarray) -> int:
    """Read the per-frame identity tag written by the fixture."""
    return int(frame[0, 0, 0])


# ── Detection ──


def test_is_imgstore_true_for_store_and_metadata(make_imgstore):
    store_dir, _ = make_imgstore()
    assert is_imgstore(store_dir) is True
    assert is_imgstore(store_dir / "metadata.yaml") is True


def test_is_imgstore_false_for_non_store(tmp_path, make_imgstore):
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "x.txt").write_text("hi")
    assert is_imgstore(plain_dir) is False
    assert is_imgstore(plain_dir / "x.txt") is False
    assert is_imgstore(tmp_path / "does_not_exist") is False


def test_is_imgstore_is_import_free(make_imgstore, monkeypatch):
    """Detection must work even when the imgstore package cannot be imported."""
    store_dir, _ = make_imgstore()
    monkeypatch.setitem(sys.modules, "imgstore", None)  # block `import imgstore`
    with pytest.raises(ImportError):
        importlib.import_module("imgstore")
    assert is_imgstore(store_dir) is True


# ── Metadata ──


def test_imgstore_metadata(make_imgstore):
    store_dir, _ = make_imgstore(nframes=12, shape=(48, 64, 3))
    meta = imgstore_metadata(store_dir)
    assert meta.width == 64
    assert meta.height == 48
    assert meta.frame_count == 12
    assert meta.fps == pytest.approx(30.0, abs=0.5)


def test_imgstore_probe(make_imgstore):
    store_dir, _ = make_imgstore(nframes=8)
    probe = imgstore_probe(store_dir)
    assert probe["width"] == 64
    assert probe["height"] == 48
    assert probe["frame_count"] == 8
    assert probe["codec"] == "npy"


def test_get_video_metadata_dispatches_to_imgstore(make_imgstore):
    store_dir, _ = make_imgstore(nframes=10)
    meta = get_video_metadata(store_dir)
    assert meta.frame_count == 10
    assert (meta.width, meta.height) == (64, 48)


# ── ImgStoreCapture ──


def test_capture_sequential_order(make_imgstore):
    store_dir, _ = make_imgstore(nframes=12)
    cap = ImgStoreCapture(store_dir)
    tags = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tags.append(_tag(frame))
    cap.release()
    assert tags == list(range(12))


def test_capture_eof_returns_false(make_imgstore):
    store_dir, _ = make_imgstore(nframes=3)
    cap = ImgStoreCapture(store_dir)
    for _ in range(3):
        assert cap.read()[0] is True
    ok, frame = cap.read()
    assert ok is False and frame is None
    cap.release()


def test_capture_seek(make_imgstore):
    store_dir, _ = make_imgstore(nframes=12)
    cap = ImgStoreCapture(store_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
    ok, frame = cap.read()
    assert ok and _tag(frame) == 7
    # reading continues sequentially after a seek
    ok, frame = cap.read()
    assert ok and _tag(frame) == 8
    cap.release()


def test_capture_grayscale_to_bgr(make_imgstore):
    store_dir, _ = make_imgstore(nframes=4, shape=(48, 64))  # single-channel
    cap = ImgStoreCapture(store_dir)
    ok, frame = cap.read()
    assert ok
    assert frame.ndim == 3 and frame.shape[2] == 3  # normalized to BGR
    cap.release()


def test_capture_get_props(make_imgstore):
    store_dir, _ = make_imgstore(nframes=9)
    cap = ImgStoreCapture(store_dir)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 9
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 64
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 48
    cap.release()


# ── MultiVideoReader over imgstores ──


def test_mvr_single_store(make_imgstore):
    store_dir, _ = make_imgstore(nframes=12)
    reader = MultiVideoReader(store_dir)
    assert reader.total_frames == 12
    assert (reader.width, reader.height) == (64, 48)

    # Every imgstore segment is randomly seekable: jumping straight to frame 9
    # (not from a sequential read) must land on the right frame.
    reader.seek(9)
    ok, frame = reader.read()
    assert ok and _tag(frame) == 9

    reader.seek(0)
    swept = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        swept.append(_tag(frame))
    reader.close()
    assert swept == list(range(12))


def test_mvr_two_stores_boundary(make_imgstore):
    s1, _ = make_imgstore(name="a", nframes=12)
    s2, _ = make_imgstore(name="b", nframes=12)
    reader = MultiVideoReader([s1, s2])
    assert reader.total_frames == 24
    # second store starts at global frame 12 (its local frame 0 → tag 0)
    seg_idx, local = reader.segment_for_frame(12)
    assert (seg_idx, local) == (1, 0)
    reader.seek(12)
    ok, frame = reader.read()
    assert ok and _tag(frame) == 0
    reader.close()


def test_mvr_rejects_fps_mismatch(make_imgstore):
    """A frame-rate mismatch across imgstore segments raises, matching the
    plain multi-video reader's construction-time failure for the same case."""
    s1, _ = make_imgstore(name="a", nframes=12, fps=30.0)
    s2, _ = make_imgstore(name="b", nframes=12, fps=60.0)
    with pytest.raises(ValueError, match="property mismatch"):
        MultiVideoReader([s1, s2])


def test_mvr_rejects_mixed_sequence(make_imgstore, tmp_path):
    store_dir, _ = make_imgstore(nframes=4)
    mp4 = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(4):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    with pytest.raises(ValueError, match="mixed"):
        MultiVideoReader([mp4, store_dir])


# ── ImgStoreFrameReader / open_frame_reader ──


def test_frame_reader_read_batch(make_imgstore):
    store_dir, _ = make_imgstore(nframes=12)
    reader = ImgStoreFrameReader(store_dir)
    indices, frames = reader.read_batch(100)
    reader.close()
    assert indices.dtype == np.int64
    assert indices.tolist() == list(range(12))
    assert frames.shape == (12, 48, 64, 3)
    assert [_tag(f) for f in frames] == list(range(12))


def test_frame_reader_frame_step_and_range(make_imgstore):
    store_dir, _ = make_imgstore(nframes=12)
    reader = ImgStoreFrameReader(store_dir, start_frame=2, end_frame=10, frame_step=2)
    indices, frames = reader.read_batch(100)
    reader.close()
    assert indices.tolist() == [2, 4, 6, 8]
    assert [_tag(f) for f in frames] == [2, 4, 6, 8]


def test_frame_reader_resize(make_imgstore):
    store_dir, _ = make_imgstore(nframes=3)
    reader = ImgStoreFrameReader(store_dir, resize=(32, 24))
    assert (reader.width, reader.height) == (32, 24)
    ok, frame = reader.read()
    reader.close()
    assert ok and frame.shape == (24, 32, 3)


def test_open_frame_reader_dispatch(make_imgstore, tmp_path):
    store_dir, _ = make_imgstore(nframes=4)
    reader = open_frame_reader(store_dir)
    assert isinstance(reader, ImgStoreFrameReader)
    reader.close()

    # A plain mp4 -> mosaic_media VideoReader (in-process decode; no ffmpeg gate).
    mp4 = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(4):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    reader2 = open_frame_reader(mp4)
    assert isinstance(reader2, VideoReader)
    reader2.close()


# -- NativeStore equivalence --


def test_native_store_npy_random_and_sequential(make_imgstore):
    """NativeStore returns the exact stored pixels for the lossless npy format."""
    store_dir, frames = make_imgstore(nframes=12, chunksize=5)

    store = NativeStore(store_dir)
    assert store.frame_count == 12
    assert store.image_shape == (48, 64, 3)
    assert store.format == "npy"
    # random access, out of order and repeated, crossing chunk boundaries
    for i in [7, 0, 11, 4, 4, 5, 9]:
        assert np.array_equal(store.frame(i), frames[i])
    store.close()

    # sequential reads return every frame in order, then signal EOF
    store = NativeStore(store_dir)
    for i in range(12):
        assert np.array_equal(store.next_frame(), frames[i])
    with pytest.raises(EOFError):
        store.next_frame()
    store.close()


def test_native_store_video_matches_imgstore_package(tmp_path):
    """For a video-format store, NativeStore matches the imgstore package.

    Both decode the same chunk files through the same OpenCV / ffmpeg build, so
    the frames are byte-identical despite mjpeg being lossy. Only ``mjpeg/avi``
    is exercised at runtime: it is the one video format writable without an
    external h264 encoder. The h264/mp4 chunk paths share the same code and are
    implemented from the imgstore source.
    """
    imgstore = pytest.importorskip("imgstore")
    dest = tmp_path / "vstore"
    try:
        store = imgstore.new_for_format(
            "mjpeg/avi",
            path=str(dest),
            mode="w",
            imgshape=(48, 64, 3),
            imgdtype=np.uint8,
            chunksize=5,
        )
        for i in range(12):
            img = np.zeros((48, 64, 3), np.uint8)
            img[0, 0, 0] = i
            store.add_image(img, frame_number=i, frame_time=float(i) / 30.0)
        store.close()
    except Exception as exc:  # pragma: no cover - depends on the local codec set
        pytest.skip(f"mjpeg/avi not writable in this environment: {exc}")

    native = NativeStore(dest)
    package = imgstore.new_for_filename(str(dest), mode="r")
    try:
        assert native.frame_count == 12
        assert native.format == "mjpeg/avi"
        # random access, out of order, crossing the chunk boundary at frame 5/10
        for i in [9, 0, 11, 3, 3, 5]:
            expected, _ = package.get_image(-1, frame_index=i)
            assert np.array_equal(native.frame(i), expected)
    finally:
        native.close()
        package.close()

    # sequential reads roll across chunk boundaries and match the package
    native = NativeStore(dest)
    package = imgstore.new_for_filename(str(dest), mode="r")
    try:
        for _ in range(12):
            expected, _ = package.get_next_image()
            assert np.array_equal(native.next_frame(), expected)
    finally:
        native.close()
        package.close()


def test_native_store_decodes_bayer_encoding(tmp_path):
    """NativeStore decodes Bayer-encoded frames identically to the imgstore package.

    imgstore stores with a non-null ``encoding`` (Bayer/YUV sensor formats)
    keep the raw single-channel sensor data on disk and apply
    ``cv2.cvtColor`` at read time via ``ImageCodecProcessor.autoconvert``.
    This exercises that path end to end: a ``cv_bayerrg``-encoded ``npy``
    store is written, then read through both the imgstore package and
    :class:`NativeStore`, and the decoded BGR frames must be byte-identical.
    """
    imgstore = pytest.importorskip("imgstore")
    dest = tmp_path / "bayer_store"
    rng = np.random.default_rng(42)
    raw_frames = [rng.integers(0, 256, size=(48, 64), dtype=np.uint8) for _ in range(6)]

    store = imgstore.new_for_format(
        "npy",
        path=str(dest),
        mode="w",
        imgshape=(48, 64),
        imgdtype=np.uint8,
        chunksize=5,
        encoding="cv_bayerrg",
    )
    for i, raw in enumerate(raw_frames):
        store.add_image(raw, frame_number=i, frame_time=float(i) / 30.0)
    store.close()

    native = NativeStore(dest)
    package = imgstore.new_for_filename(str(dest), mode="r")
    try:
        assert native.frame_count == 6
        assert native.image_shape == (48, 64, 3)
        # random access, out of order, crossing the chunk boundary at frame 5
        for i in [4, 0, 5, 2, 2, 1]:
            expected, _ = package.get_image(-1, frame_index=i)
            assert np.array_equal(native.frame(i), expected)
    finally:
        native.close()
        package.close()

    # sequential reads roll across chunk boundaries and match the package
    native = NativeStore(dest)
    package = imgstore.new_for_filename(str(dest), mode="r")
    try:
        for _ in range(6):
            expected, _ = package.get_next_image()
            assert np.array_equal(native.next_frame(), expected)
    finally:
        native.close()
        package.close()


# -- NativeStore robustness --


def test_native_store_video_truncates_odd_dimensions_to_even(tmp_path):
    """A legacy video store with an odd ``imgshape`` in metadata.yaml reads back even.

    Mirrors imgstore's ``VideoImgStore._calculate_written_image_shape``, which
    truncates height/width down to even (``int(x) & -2``) because the video
    encoder only ever wrote even dimensions, and re-applies that correction on
    read for stores whose metadata.yaml still holds odd values.
    """
    store_dir = tmp_path / "legacy_vstore"
    store_dir.mkdir()
    metadata = {
        "__store": {
            "imgshape": [49, 65, 3],
            "imgdtype": "uint8",
            "chunksize": 10000,
            "format": "mjpeg/avi",
            "class": "VideoImgStore",
            "version": 1,
            "encoding": None,
            "extension": ".avi",
        }
    }
    with (store_dir / "metadata.yaml").open("w") as f:
        yaml.safe_dump(metadata, f)

    store = NativeStore(store_dir)
    assert store.image_shape == (48, 64, 3)
    store.close()

    # The truncated shape must flow through to imgstore_probe and
    # ImgStoreCapture.width/height, not just NativeStore.image_shape.
    probe = imgstore_probe(store_dir)
    assert (probe["width"], probe["height"]) == (64, 48)

    cap = ImgStoreCapture(store_dir)
    assert (cap.width, cap.height) == (64, 48)
    cap.release()


def test_native_store_directory_store_imgshape_not_truncated(make_imgstore):
    """Directory (npy/image) stores hold the exact imgshape -- no truncation."""
    store_dir, _ = make_imgstore(nframes=3, shape=(47, 63, 3))
    store = NativeStore(store_dir)
    assert store.image_shape == (47, 63, 3)
    store.close()


def test_native_store_skips_chunk_with_missing_index(make_imgstore):
    """A chunk without a written index (interrupted recording) is skipped, not fatal.

    Mirrors imgstore's own index loader (``ImgStoreIndex.new_from_chunks``),
    which catches ``IOError`` from a missing chunk index, warns, and continues
    rather than failing the whole store.
    """
    store_dir, frames = make_imgstore(nframes=12, chunksize=5)
    last_chunk_index = store_dir / "000002" / "index.npz"
    assert last_chunk_index.is_file()
    last_chunk_index.unlink()

    with pytest.warns(UserWarning, match="missing index"):
        store = NativeStore(store_dir)
    assert store.frame_count == 10  # the dropped chunk held the last 2 of 12 frames
    for i in range(10):
        assert np.array_equal(store.next_frame(), frames[i])
    store.close()

    with pytest.warns(UserWarning, match="missing index"):
        cap = ImgStoreCapture(store_dir)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    tags = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tags.append(_tag(frame))
    cap.release()
    assert tags == list(range(10))


def test_native_store_close_safe_after_init_failure(tmp_path):
    """close() (reachable from __del__) is safe even if __init__ raises early.

    __init__ must set its cleanup-relevant attributes (``_video_reader``,
    ``_video_chunk_id``) before any line that can raise -- e.g. probing a
    malformed store during ``index_media`` -- so a partially constructed
    instance does not raise ``AttributeError`` when garbage collected.
    """
    bad_dir = tmp_path / "malformed"
    bad_dir.mkdir()
    (bad_dir / "metadata.yaml").write_text("- 1\n- 2\n")

    store = object.__new__(NativeStore)
    with pytest.raises(MediaProbeError):
        NativeStore.__init__(store, bad_dir)
    store.close()
