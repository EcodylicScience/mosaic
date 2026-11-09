"""The SLEAP, Lightning Pose and Ultralytics bridges time a table at the clip's measured rate.

A bridge passes the measured rate of the media the tool read, and the dataset's
``fps_default`` for a clip without one. A clip with neither fails its entry instead of
being timed at an assumed rate.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.track_converter import FrameRateUnknownError
from mosaic.tracking.common.bridge import tracks_table_path

from tests.helpers import (
    make_dataset,
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_ultralytics_predictions,
)

_GROUP, _SEQUENCE = "", "vid1"
_FRAMES = 6

Bridge = Callable[[Dataset, float | None, float | None], None]
"""Bridge a tool's output for the entry at a measured rate and a fallback rate."""


def _variant(kind: str) -> str:
    return f"{kind}.0.1-bbbbbbbbbb"


def _sleap(ds: Dataset, measured: float | None, fallback: float | None) -> None:
    from mosaic.tracking.sleap.dataset_runs import bridge_analysis_h5_to_tracks

    h5 = ds.get_root("sleap") / "run" / _SEQUENCE / f"{_SEQUENCE}.analysis.h5"
    write_sleap_analysis_h5(h5, np.random.default_rng(0).random((_FRAMES, 1, 1, 2)))
    _ = bridge_analysis_h5_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        h5,
        tracks_variant=_variant("sleap"),
        producer_run_id=_variant("sleap"),
        consumed_media=[],
        model_checkpoints=[],
        measured_fps=measured,
        fallback_fps=fallback,
        overwrite=True,
        axis=EntryAxis(),
        recorded_first_frame=None,
        carries_tracklets=True,
    )


def _litpose(ds: Dataset, measured: float | None, fallback: float | None) -> None:
    from mosaic.tracking.litpose.dataset_runs import bridge_csv_to_tracks

    csv = ds.get_root("litpose") / "run" / _SEQUENCE / f"{_SEQUENCE}.predictions.csv"
    _ = write_dlc_csv(csv, ["nose"], n_frames=_FRAMES, scorer="heatmap_tracker")
    _ = bridge_csv_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        csv,
        tracks_variant=_variant("litpose"),
        producer_run_id=_variant("litpose"),
        consumed_media=[],
        model_files=[],
        measured_fps=measured,
        fallback_fps=fallback,
        overwrite=True,
        axis=EntryAxis(),
        recorded_first_frame=None,
    )


def _ultralytics(ds: Dataset, measured: float | None, fallback: float | None) -> None:
    from mosaic.tracking.ultralytics_track.dataset_runs import (
        bridge_predictions_to_tracks,
    )

    predictions = (
        ds.get_root("_tracking") / "ultralytics" / _SEQUENCE / "predictions.parquet"
    )
    write_ultralytics_predictions(predictions, n_frames=_FRAMES, n_ids=1)
    _ = bridge_predictions_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        predictions,
        tracks_variant=_variant("ultralytics-track"),
        producer_run_id=_variant("ultralytics-track"),
        consumed_media=[],
        model_files=[],
        measured_fps=measured,
        fallback_fps=fallback,
        overwrite=True,
        axis=EntryAxis(),
        frames_read=None,
        recorded_first_frame=None,
    )


_BRIDGES: dict[str, tuple[str, Bridge]] = {
    "sleap": ("sleap", _sleap),
    "litpose": ("litpose", _litpose),
    "ultralytics": ("ultralytics-track", _ultralytics),
}


@pytest.fixture(params=sorted(_BRIDGES))
def bridge(request: pytest.FixtureRequest) -> tuple[str, Bridge]:
    name: str = request.param
    return _BRIDGES[name]


def _published(ds: Dataset, kind: str) -> pd.DataFrame:
    return pd.read_parquet(
        tracks_table_path(ds, _variant(kind), make_entry_key(_GROUP, _SEQUENCE))
    )


def _assert_timed_at(table: pd.DataFrame, rate: float) -> None:
    np.testing.assert_allclose(
        np.asarray(table["time"], dtype=float),
        np.asarray(table["frame"], dtype=float) / rate,
    )


def test_the_measured_rate_times_the_table(
    tmp_path: Path, bridge: tuple[str, Bridge]
) -> None:
    kind, run = bridge
    ds = make_dataset(tmp_path / "ds")

    run(ds, 60.0, None)

    _assert_timed_at(_published(ds, kind), 60.0)


def test_the_measured_rate_outranks_the_dataset_default(
    tmp_path: Path, bridge: tuple[str, Bridge]
) -> None:
    kind, run = bridge
    ds = make_dataset(tmp_path / "ds")

    run(ds, 60.0, 24.0)

    _assert_timed_at(_published(ds, kind), 60.0)


def test_the_dataset_default_times_a_clip_without_a_measured_rate(
    tmp_path: Path, bridge: tuple[str, Bridge]
) -> None:
    kind, run = bridge
    ds = make_dataset(tmp_path / "ds")

    run(ds, None, 24.0)

    _assert_timed_at(_published(ds, kind), 24.0)


def test_a_clip_with_no_rate_fails_its_entry(
    tmp_path: Path, bridge: tuple[str, Bridge]
) -> None:
    _, run = bridge
    ds = make_dataset(tmp_path / "ds")

    with pytest.raises(FrameRateUnknownError):
        run(ds, None, None)

    assert read_tracks_index(ds).empty
