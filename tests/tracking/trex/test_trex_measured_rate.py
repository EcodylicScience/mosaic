"""A TREx table is timed at the measured frame rate of the media TREx read.

TREx keeps a video's frame rate as an integer, so a 29.97 fps video is timed at
29 and every per-second column is computed against that time. The tracker knows
which media it handed TREx, so the bridge gives the converter that media's
measured rate, and the converter retimes the table at it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mosaic.tracking.trex.dataset_runs as dr
from mosaic.core.dataset import Dataset
from mosaic.tracking.trex.params import TrexParams
from tests.helpers import FakeTrex, MediaClip, scope_over, write_media_index

_TREX_RATE = 29.0
_FRAMES = 6


def _track_at(ds: Dataset, trex: FakeTrex, *, media_fps: float) -> pd.DataFrame:
    """Track ``vid1`` from media measured at *media_fps*, and read its table."""
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    write_media_index(
        ds, [MediaClip(sequence="vid1", filename="vid1.mp4", fps=media_fps)]
    )
    trex.npz_frames = _FRAMES
    trex.extra_fields = {
        "frame_rate": np.array([_TREX_RATE]),
        "time": np.arange(_FRAMES) / _TREX_RATE,
        "VX": np.full(_FRAMES, 2.0),
    }

    _ = dr.run_trex(ds, TrexParams(), scope_over(("", "vid1")))

    rows = read_tracks_index(ds)
    assert len(rows) == 1
    return pd.read_parquet(ds.resolve_path(str(rows.iloc[0]["abs_path"])))


def test_the_media_rate_reaches_the_converter(ds: Dataset, trex: FakeTrex) -> None:
    table = _track_at(ds, trex, media_fps=29.97)

    assert table["time"].to_numpy() == pytest.approx(np.arange(_FRAMES) / 29.97)
    assert table["frame_rate"].to_list() == [29.97] * _FRAMES
    assert table["VX"].to_numpy() == pytest.approx(
        np.full(_FRAMES, 2.0 * 29.97 / _TREX_RATE)
    )


def test_media_without_a_measured_rate_keeps_trex_timing(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The dataset's default rate is not a measurement, and is not applied."""
    table = _track_at(ds, trex, media_fps=0.0)

    assert table["time"].to_numpy() == pytest.approx(np.arange(_FRAMES) / _TREX_RATE)
    assert table["VX"].to_numpy() == pytest.approx(np.full(_FRAMES, 2.0))
