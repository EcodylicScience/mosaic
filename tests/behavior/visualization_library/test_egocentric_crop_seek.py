"""A frame-filtered crop must seek to its first frame, and read the same pixels.

``_process_single_id`` used to open the reader and decode from frame 0 to the
first row of the filtered table, discarding every frame on the way. On a session
whose individuals first appear ~13k frames in, a 300-frame preview window cost
~13k decodes per individual -- and four individuals are cropped independently, so
~55k decodes to produce 1,200 crops.

Both multi-readers expose ``seek(global_frame)`` and position off a keyframe
index, so the prefix costs one GOP instead of all of it. What these tests pin is
that the optimisation is invisible in the output: the same frames, the same
pixels, the same names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import write_tracks_row

pytestmark = pytest.mark.media


_TOTAL = 300


def _tracks(frames: range, size: tuple[int, int]) -> pd.DataFrame:
    """One individual on a slow diagonal, so each frame's crop is distinguishable.

    Every column is a pure function of the **absolute** frame number. A window
    must describe the same animal in the same place as the full table does, or a
    filtered run would crop different pixels for a reason that has nothing to do
    with seeking, and the comparison below would be meaningless.
    """
    f = np.fromiter(frames, dtype=int, count=len(frames))
    t = f / max(_TOTAL - 1, 1)
    return pd.DataFrame(
        {
            "frame": f,
            "time": f / 30.0,
            "id": 0,
            "group": "g",
            "sequence": "s",
            "X": size[0] * (0.3 + 0.4 * t),
            "Y": size[1] * (0.3 + 0.4 * t),
            "ANGLE": t,
        }
    )


def _crop_to(ds: Dataset, out: Path, table: pd.DataFrame) -> dict[int, np.ndarray]:
    import cv2

    feature = EgocentricCrop(
        params={
            "crop_size": (24, 24),
            "target_id": 0,
            "center_mode": "xy",
            "angle_col": "ANGLE",
            "grayscale": True,
            "output_mode": "frames",
            "output_root": str(out),
        }
    )
    feature._ds = ds  # pyright: ignore[reportPrivateUsage]
    feature.transform(table)
    d = out / "g__s" / "frames_id0"
    return {
        int(p.stem.split("_")[1]): cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        for p in sorted(d.glob("frame_*.png"))
    }


@pytest.fixture
def recording(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> tuple[Dataset, int, tuple[int, int]]:
    """A dataset holding one longer clip, so a late window has a real prefix."""
    from mosaic_media.io.writer import FFmpegVideoWriter

    size, n_frames = (96, 72), _TOTAL
    ds = make_media_dataset(tmp_path)
    media = Path(ds.get_root("media_raw"))
    media.mkdir(parents=True, exist_ok=True)
    path = media / "s.mp4"
    rng = np.random.default_rng(0)
    with FFmpegVideoWriter(path, width=size[0], height=size[1], fps=30.0) as writer:
        for i in range(n_frames):
            # Per-frame texture: two frames must never be confusable, or a
            # mis-seek by one frame would pass unnoticed.
            frame = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            frame[:, :, 0] = i % 256
            writer.write(frame)
    # The tracks row goes in FIRST: `index_media` reads the tracks index to
    # derive each media file's (group, sequence), so indexing before it exists
    # files the clip under group "" and `resolve_media` then finds nothing.
    #
    # A table with no recorded schema also reads as the legacy centimetre family
    # and `require_pixel_positions` refuses to crop it, so record a pixel schema
    # and keep this fixture about seeking.
    out = Path(ds.get_root("tracks")) / "g__s.parquet"
    table = _tracks(range(n_frames), size)
    table.to_parquet(out)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="g",
        sequence="s",
        out_path=out,
        producer="convert-x",
        std_format="mosaic_v1",
        n_rows=len(table),
    )
    ds.index_media([media])
    return ds, n_frames, size


def test_a_late_window_reads_the_same_pixels_as_a_full_read(
    tmp_path: Path, recording: tuple[Dataset, int, tuple[int, int]]
) -> None:
    """The seek must be invisible: identical frames, identical pixels."""
    ds, n_frames, size = recording
    window = range(n_frames - 40, n_frames - 10)

    # The whole table, then only the late window. The first crops the prefix
    # sequentially; the second seeks past it.
    full = _crop_to(ds, tmp_path / "full", _tracks(range(n_frames), size))
    late = _crop_to(ds, tmp_path / "late", _tracks(window, size))

    assert set(late) == set(window), "the filtered run cropped the wrong frames"
    for f in window:
        assert np.array_equal(late[f], full[f]), (
            f"frame {f} differs between a sequential read and a seek -- the seek "
            f"landed on the wrong frame"
        )


def test_the_first_frame_of_the_window_is_not_off_by_one(
    tmp_path: Path, recording: tuple[Dataset, int, tuple[int, int]]
) -> None:
    """The sharpest failure mode: a seek that lands one frame early or late.

    Each window starts one frame later than the last, so an off-by-one in the
    seek would make two windows agree where they must differ.
    """
    ds, n_frames, size = recording
    full = _crop_to(ds, tmp_path / "ref", _tracks(range(n_frames), size))
    for offset in range(3):
        start = n_frames - 30 + offset
        got = _crop_to(
            ds, tmp_path / f"w{offset}", _tracks(range(start, start + 4), size)
        )
        assert min(got) == start
        assert np.array_equal(got[start], full[start]), (
            f"a window starting at {start} did not decode frame {start}"
        )


def test_an_unfiltered_run_is_unchanged(
    tmp_path: Path, recording: tuple[Dataset, int, tuple[int, int]]
) -> None:
    """A table starting at frame 0 must not seek at all, and must still be whole."""
    ds, n_frames, size = recording
    got = _crop_to(ds, tmp_path / "all", _tracks(range(n_frames), size))
    assert set(got) == set(range(n_frames))
