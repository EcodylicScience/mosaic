"""Tests for native imgstore support (detection, metadata, readers)."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("imgstore")

from mosaic.core.media.imgstore_io import (  # noqa: E402
    ImgStoreCapture,
    ImgStoreFrameReader,
    imgstore_metadata,
    imgstore_probe,
    is_imgstore,
)
from mosaic.core.media.video_io import (  # noqa: E402
    FFmpegFrameReader,
    MultiVideoReader,
    get_video_metadata,
    open_frame_reader,
)


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
        import imgstore  # noqa: F401
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
    assert reader.segments[0].seekable is True

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

    # A plain mp4 → FFmpegFrameReader (skip if ffmpeg is unavailable).
    from mosaic.core.media.video_io import _ffmpeg_available

    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    mp4 = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(4):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    reader2 = open_frame_reader(mp4)
    assert isinstance(reader2, FFmpegFrameReader)
    reader2.close()
