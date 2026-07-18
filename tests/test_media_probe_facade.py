import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from mosaic.core.media.video_io import get_video_metadata, open_frame_reader


def _write_cfr_mp4(path: Path, n: int = 12, w: int = 64, h: int = 48) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


def _write_raw_h264(path: Path, n_frames: int = 15) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30",
            "-frames:v", str(n_frames), "-c:v", "libx264", "-bsf:v", "h264_mp4toannexb",
            "-f", "h264", str(path),
        ],
        check=True, capture_output=True,
    )


def test_get_video_metadata_constant_rate(tmp_path: Path) -> None:
    mp4 = tmp_path / "v.mp4"
    _write_cfr_mp4(mp4)
    meta = get_video_metadata(mp4)
    assert (meta.width, meta.height) == (64, 48)
    assert meta.fps == pytest.approx(30.0, rel=0.05)
    assert meta.frame_count == 12
    assert meta.path == mp4.resolve()


def test_open_frame_reader_reads_raw_h264(tmp_path: Path) -> None:
    raw = tmp_path / "raw.h264"
    _write_raw_h264(raw)
    reader = open_frame_reader(raw)
    count = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        count += 1
    reader.close()
    assert count >= 10  # sequential decode of a containerless stream works
