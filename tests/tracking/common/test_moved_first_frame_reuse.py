"""Each tracker bridge reuses a published table only at the first frame it was numbered from.

A sequence of a continuous group starts after the frames of the members before
it. When one of them changes length, the sequence's first frame moves, and its
published table is numbered from the old one. The bridge then publishes the
table again from the tool output, which is still current.

Each case publishes a table at first frame 5 and bridges again. A recorded first
frame that equals the axis's leaves the table as it is, even though the axis
places it elsewhere: the bridge trusts the record. A recorded first frame that
differs publishes the table again at the axis's first frame.
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
from mosaic.tracking.common.bridge import BridgeCounts, tracks_table_path

from tests.helpers import (
    clip_facts,
    make_dataset,
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_trex_npz,
    write_ultralytics_predictions,
)

_GROUP, _SEQUENCE = "rec", "s2"
_FRAMES = 4

Bridge = Callable[[Dataset, EntryAxis, int | None, bool], BridgeCounts | None]
"""Bridge a tool's output for the entry at an axis, a recorded first frame and an overwrite."""


def _variant(kind: str) -> str:
    return f"{kind}.0.1-bbbbbbbbbb"


def _axis(first_frame: int) -> EntryAxis:
    """The axis of a tool that read the entry's one clip, placed at *first_frame*."""
    return EntryAxis.of_entry_media(
        [clip_facts(frame_count=_FRAMES)], windowed=False, first_frame=first_frame
    )


def _trex(
    ds: Dataset, axis: EntryAxis, recorded: int | None, overwrite: bool
) -> BridgeCounts | None:
    from mosaic.tracking.trex.dataset_runs import bridge_npz_to_tracks

    npz = ds.get_root("trex") / "run" / _SEQUENCE / "data" / f"{_SEQUENCE}_fish0.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    write_trex_npz(npz, n=_FRAMES)
    return bridge_npz_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        [npz],
        tracks_variant=_variant("trex"),
        producer_run_id=_variant("trex"),
        consumed_media=[],
        axis=axis,
        fps=None,
        conversion=None,
        overwrite=overwrite,
        recorded_first_frame=recorded,
    )


def _sleap(
    ds: Dataset, axis: EntryAxis, recorded: int | None, overwrite: bool
) -> BridgeCounts | None:
    from mosaic.tracking.sleap.dataset_runs import bridge_analysis_h5_to_tracks

    h5 = ds.get_root("sleap") / "run" / _SEQUENCE / f"{_SEQUENCE}.analysis.h5"
    write_sleap_analysis_h5(h5, np.random.default_rng(0).random((_FRAMES, 1, 1, 2)))
    return bridge_analysis_h5_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        h5,
        tracks_variant=_variant("sleap"),
        producer_run_id=_variant("sleap"),
        consumed_media=[],
        model_checkpoints=[],
        fps=30.0,
        overwrite=overwrite,
        axis=axis,
        recorded_first_frame=recorded,
    )


def _litpose(
    ds: Dataset, axis: EntryAxis, recorded: int | None, overwrite: bool
) -> BridgeCounts | None:
    from mosaic.tracking.litpose.dataset_runs import bridge_csv_to_tracks

    csv = ds.get_root("litpose") / "run" / _SEQUENCE / f"{_SEQUENCE}.predictions.csv"
    _ = write_dlc_csv(csv, ["nose"], n_frames=_FRAMES, scorer="heatmap_tracker")
    return bridge_csv_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        csv,
        tracks_variant=_variant("litpose"),
        producer_run_id=_variant("litpose"),
        consumed_media=[],
        model_files=[],
        fps=30.0,
        overwrite=overwrite,
        axis=axis,
        recorded_first_frame=recorded,
    )


def _ultralytics(
    ds: Dataset, axis: EntryAxis, recorded: int | None, overwrite: bool
) -> BridgeCounts | None:
    from mosaic.tracking.ultralytics_track.dataset_runs import (
        bridge_predictions_to_tracks,
    )

    predictions = (
        ds.get_root("_tracking") / "ultralytics" / _SEQUENCE / "predictions.parquet"
    )
    write_ultralytics_predictions(predictions, n_frames=_FRAMES, n_ids=1)
    return bridge_predictions_to_tracks(
        ds,
        _GROUP,
        _SEQUENCE,
        predictions,
        tracks_variant=_variant("ultralytics-track"),
        producer_run_id=_variant("ultralytics-track"),
        consumed_media=[],
        model_files=[],
        fps=30.0,
        overwrite=overwrite,
        axis=axis,
        frames_read=None,
        recorded_first_frame=recorded,
    )


_BRIDGES: dict[str, tuple[str, Bridge]] = {
    "trex": ("trex", _trex),
    "sleap": ("sleap", _sleap),
    "litpose": ("litpose", _litpose),
    "ultralytics": ("ultralytics-track", _ultralytics),
}


@pytest.fixture(params=sorted(_BRIDGES))
def bridge(request: pytest.FixtureRequest) -> tuple[str, Bridge]:
    name: str = request.param
    return _BRIDGES[name]


def _first_frame_of_table(ds: Dataset, kind: str) -> int:
    path = tracks_table_path(ds, _variant(kind), make_entry_key(_GROUP, _SEQUENCE))
    return int(pd.read_parquet(path, columns=["frame"])["frame"].min())


def _recorded_cell(ds: Dataset) -> str:
    (cell,) = read_tracks_index(ds)["first_frame"].astype(str)
    return cell


def test_a_table_at_its_recorded_first_frame_is_reused(
    tmp_path: Path, bridge: tuple[str, Bridge]
) -> None:
    kind, run = bridge
    ds = make_dataset(tmp_path / "ds")
    _ = run(ds, _axis(5), None, True)

    counts = run(ds, _axis(7), 7, False)

    assert counts is not None
    assert _first_frame_of_table(ds, kind) == 5


def test_a_table_numbered_from_another_first_frame_is_published_again(
    tmp_path: Path, bridge: tuple[str, Bridge]
) -> None:
    kind, run = bridge
    ds = make_dataset(tmp_path / "ds")
    _ = run(ds, _axis(5), None, True)
    assert _recorded_cell(ds) == "5"

    counts = run(ds, _axis(7), 5, False)

    assert counts is not None
    assert _first_frame_of_table(ds, kind) == 7
    assert _recorded_cell(ds) == "7"
