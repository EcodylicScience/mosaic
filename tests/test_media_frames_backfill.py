"""The backfill fills ``media_frames`` by the rule the bridge records it by.

``mosaic measure-tracks`` fills the cell for a table published before it existed.
It used to fill every row from the entry's media, so a run under a frame window,
or a converted or resampled table, gained a length it was never meant to span.
Now the backfill rebuilds what the row's run read from the variant's record
(:func:`~mosaic.core.pipeline.tracks_axis.recorded_axis`) and asks the axis the
bridge asks. Each case below runs a producer, blanks the cell, backfills it, and
expects the value the bridge recorded.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import mosaic.tracking.litpose.dataset_runs as litpose_runs
import mosaic.tracking.sleap.dataset_runs as sleap_runs
import mosaic.tracking.trex.dataset_runs as trex_runs
import mosaic.tracking.ultralytics_track.dataset_runs as ultralytics_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.tracks_identity import write_tracks_variant
from mosaic.core.pipeline.tracks_index import (
    backfill_media_frames,
    read_media_frames,
    read_tracks_index,
    tracks_index_path,
    write_tracks_row,
)
from mosaic.core.scope import Scope
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from tests.helpers import (
    MediaClip,
    install_fake_litpose,
    install_fake_pose_inference,
    install_fake_sleap,
    install_fake_ultralytics,
    make_dataset,
    stub_join,
    write_litpose_model,
    write_media_index,
    write_sleap_model,
)
from tests.helpers.trex import install_fake_trex

_FRAMES = 30
"""How many frames each clip holds."""

type Producer = Callable[[Dataset, Path, pytest.MonkeyPatch], None]
"""Run one producer over the dataset's entry, with a fake for its tool."""


def _dataset(tmp_path: Path, clips: int = 1) -> Dataset:
    """One entry, ``vid1``, of *clips* clips, with the join that several need."""
    ds = make_dataset(tmp_path / "ds")
    uids = [f"uid-{order}" for order in range(clips)]
    write_media_index(
        ds,
        [
            MediaClip(
                sequence="vid1",
                filename=f"c{order}.mp4",
                video_order=order,
                video_uuid=uid,
                frame_count=_FRAMES,
            )
            for order, uid in enumerate(uids)
        ],
    )
    if clips > 1:
        _ = stub_join(ds, uids)
    return ds


def _trex(**params: object) -> Producer:
    def run(ds: Dataset, _tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = install_fake_trex(monkeypatch)
        _ = trex_runs.run_trex(ds, TrexParams.model_validate(params))

    return run


def _ultralytics(**params: object) -> Producer:
    def run(ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = install_fake_ultralytics(monkeypatch)
        model = tmp_path / "yolo" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")
        stated = {"model_path": str(model), **params}
        _ = ultralytics_runs.run_ultralytics(
            ds, UltralyticsParams.model_validate(stated)
        )

    return run


def _sleap(**params: object) -> Producer:
    def run(ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = install_fake_sleap(monkeypatch)
        model = write_sleap_model(tmp_path / "sleap_model")
        stated = {"model_paths": [str(model)], **params}
        _ = sleap_runs.run_sleap(ds, SleapParams.model_validate(stated))

    return run


def _litpose(ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = install_fake_litpose(monkeypatch)
    model = write_litpose_model(tmp_path / "litpose_model")
    _ = litpose_runs.run_litpose(ds, LitposeParams(model_path=str(model)))


def _infer_pose(**params: object) -> Producer:
    def run(ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = install_fake_pose_inference(monkeypatch)
        model = tmp_path / "weights" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")
        stated = {"model": str(model), **params}
        _ = run_op(ds, "infer-pose", stated, scope=Scope(entries=[("", "vid1")]))

    return run


def _blank_media_frames(ds: Dataset) -> None:
    path = tracks_index_path(ds)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows.assign(media_frames="").to_csv(path, index=False)


def _recorded(ds: Dataset) -> int | None:
    rows = read_tracks_index(ds)
    assert len(rows) == 1
    return read_media_frames(rows.iloc[0])


@pytest.mark.parametrize(
    ("clips", "produce", "expected"),
    [
        (1, _trex(), _FRAMES),
        (2, _trex(), 2 * _FRAMES),
        (2, _trex(analysis_range=(0, 9)), None),
        (1, _trex(track_extra_settings={"analysis_range": [0, 9]}), None),
        (1, _trex(convert_extra_settings={"video_conversion_range": [0, 9]}), None),
        (1, _ultralytics(), _FRAMES),
        (1, _ultralytics(start_frame=5), None),
        (1, _sleap(), _FRAMES),
        (1, _sleap(analysis_range=(0, 9)), None),
        (1, _litpose, _FRAMES),
        (1, _infer_pose(), _FRAMES),
        (1, _infer_pose(start_frame=5), None),
    ],
    ids=[
        "trex-one-clip",
        "trex-two-clips",
        "trex-window-field",
        "trex-track-setting",
        "trex-convert-setting",
        "ultralytics",
        "ultralytics-window",
        "sleap",
        "sleap-window",
        "litpose",
        "infer-pose",
        "infer-pose-window",
    ],
)
def test_a_backfilled_row_holds_what_the_bridge_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clips: int,
    produce: Producer,
    expected: int | None,
) -> None:
    ds = _dataset(tmp_path, clips)
    produce(ds, tmp_path, monkeypatch)
    assert _recorded(ds) == expected, "the bridge"

    _blank_media_frames(ds)
    _ = backfill_media_frames(ds)

    assert _recorded(ds) == expected, "the backfill"


@pytest.mark.parametrize("producer", ["convert-trex_npz", "resample-tracks"])
def test_a_table_made_from_another_table_stays_blank(
    tmp_path: Path, producer: str
) -> None:
    """No tool read media to make it, so nothing is compared with the media."""
    ds = _dataset(tmp_path)
    variant = f"{producer}.0.1-0123456789"
    _ = write_tracks_variant(ds.get_root("tracks"), variant, producer, "0.1", {})
    out = ds.get_root("tracks") / variant / "vid1.parquet"
    pd.DataFrame({"frame": [0, 1], "id": [0, 0]}).to_parquet(out)
    write_tracks_row(
        ds,
        run_id=variant,
        group="",
        sequence="vid1",
        out_path=out,
        producer=producer,
        std_format="mosaic_v1",
        n_rows=2,
    )

    assert len(backfill_media_frames(ds)) == 0
    assert _recorded(ds) is None


def test_a_row_whose_variant_has_no_record_stays_blank(tmp_path: Path) -> None:
    """Whether its run read the whole entry cannot be established."""
    ds = _dataset(tmp_path)
    out = ds.get_root("tracks") / "trex.0.2-0123456789" / "vid1.parquet"
    out.parent.mkdir(parents=True)
    pd.DataFrame({"frame": [0, 1], "id": [0, 0]}).to_parquet(out)
    write_tracks_row(
        ds,
        run_id="trex.0.2-0123456789",
        group="",
        sequence="vid1",
        out_path=out,
        producer="trex",
        std_format="trex_v2",
        n_rows=2,
    )

    assert len(backfill_media_frames(ds)) == 0
    assert _recorded(ds) is None
