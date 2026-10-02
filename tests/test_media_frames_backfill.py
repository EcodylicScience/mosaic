"""The backfill rewrites ``media_frames`` to what the bridge's rule gives.

``mosaic measure-tracks`` fills the cell for a table published before it existed.
The backfill rebuilds what the row's run read from the variant's record
(:func:`~mosaic.core.pipeline.tracks_axis.recorded_media_frames`) and asks the
axis the bridge asks, so a run under a frame window and a converted or resampled
table get a blank cell. It clears a cell that the rule leaves blank, such as one an
earlier backfill filled from the entry's media, and keeps the value of a row whose
run cannot be established. Each case below runs a producer, sets the cell,
backfills it, and expects the value the bridge recorded.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
    backfill_frames_read,
    backfill_known_tail_loss,
    backfill_media_frames,
    read_media_frames,
    read_tracks_index,
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
    install_fake_trex,
    install_fake_ultralytics,
    make_dataset,
    set_tracks_cell,
    stub_join,
    write_litpose_model,
    write_media_index,
    write_sleap_model,
)

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
    set_tracks_cell(ds, "media_frames", "")


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

    # A stale count, as the backfill that filled every row from the media left it.
    set_tracks_cell(ds, "media_frames", "999")
    done = backfill_media_frames(ds)

    assert _recorded(ds) == expected, "the rewrite"
    assert (len(done.written), len(done.cleared)) == (
        (1, 0) if expected is not None else (0, 1)
    )
    assert len(done.not_established) == 0


@pytest.mark.parametrize("producer", ["convert-trex_npz", "resample-tracks", ""])
def test_a_table_made_from_another_table_is_cleared(
    tmp_path: Path, producer: str
) -> None:
    """No tool read media to make it, so nothing is compared with the media.

    An earlier backfill filled such rows from the entry's media. A blank producer
    is no tool's either: every bridge records the tracking root that published.
    """
    ds = _dataset(tmp_path)
    variant = f"{producer or 'legacy'}.0.1-0123456789"
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
        media_frames=_FRAMES,
    )

    would = backfill_media_frames(ds, dry_run=True)
    assert len(would.cleared) == 1
    assert _recorded(ds) == _FRAMES, "a dry run must not write"

    done = backfill_media_frames(ds)
    assert len(done.cleared) == 1
    assert done.cleared.iloc[0]["media_frames"] == str(_FRAMES), (
        "the report names the value cleared"
    )
    assert len(done.written) == 0
    assert _recorded(ds) is None
    assert len(backfill_media_frames(ds).cleared) == 0, "not idempotent"


def _trex_one_clip(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trex()(ds, tmp_path, monkeypatch)
    assert _recorded(ds) == _FRAMES


def _without_variant_record(ds: Dataset) -> None:
    for record in ds.get_root("tracks").rglob("params.json"):
        record.unlink()


def _without_media_index(ds: Dataset) -> None:
    (ds.get_root(ds.resolve_media_root()) / "index.csv").unlink()


@pytest.mark.parametrize(
    "unestablish",
    [_without_variant_record, _without_media_index],
    ids=["no-variant-record", "no-media-index"],
)
def test_a_row_whose_run_cannot_be_established_keeps_its_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unestablish: Callable[[Dataset], None],
) -> None:
    """An unmounted media root or a lost record must not wipe a cell."""
    ds = _dataset(tmp_path)
    _trex_one_clip(ds, tmp_path, monkeypatch)
    set_tracks_cell(ds, "media_frames", "999")
    unestablish(ds)

    done = backfill_media_frames(ds)

    assert _recorded(ds) == 999
    assert (len(done.written), len(done.cleared)) == (0, 0)
    assert len(done.not_established) == 1
    assert done.not_established.iloc[0]["media_frames"] == "999"


def test_a_row_whose_media_has_changed_since_its_run_keeps_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's media is not what the run read, so its length is no rule's value."""
    import mosaic.core.pipeline.tracks_axis as tracks_axis

    ds = _dataset(tmp_path)
    _trex_one_clip(ds, tmp_path, monkeypatch)
    set_tracks_cell(ds, "consumed_media_composition", "then")

    def now(_ds: Dataset, entries: object) -> dict[tuple[str, str], str]:
        return {("", "vid1"): "now"}

    monkeypatch.setattr(tracks_axis, "media_compositions_for", now)

    done = backfill_media_frames(ds)

    assert _recorded(ds) == _FRAMES
    assert len(done.not_established) == 1
    assert (len(done.written), len(done.cleared)) == (0, 0)


def test_a_row_is_reported_not_established_only_when_it_holds_a_value(
    tmp_path: Path,
) -> None:
    """A blank cell left blank loses nothing, so it is not reported as kept.

    Both rows name a variant that has no record, so neither run can be
    established.
    """
    ds = _dataset(tmp_path)
    for sequence, media_frames in (("held", _FRAMES), ("blank", None)):
        out = ds.get_root("tracks") / "trex.0.2-0123456789" / f"{sequence}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"frame": [0, 1], "id": [0, 0]}).to_parquet(out)
        write_tracks_row(
            ds,
            run_id="trex.0.2-0123456789",
            group="",
            sequence=sequence,
            out_path=out,
            producer="trex",
            std_format="trex_v2",
            n_rows=2,
            media_frames=media_frames,
        )

    done = backfill_media_frames(ds)

    assert done.not_established["sequence"].tolist() == ["held"]
    assert (len(done.written), len(done.cleared)) == (0, 0)


def test_a_row_whose_media_was_re_indexed_since_its_run_keeps_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry now holds another clip of another length, which its run never read.

    Its media composition is projected as a scan projects it, so the one the row
    recorded at its run and the one the media holds now are both known.
    """
    ds = _dataset(tmp_path)
    _ = ds.rebuild_sequence_index("media_raw")
    _trex_one_clip(ds, tmp_path, monkeypatch)
    write_media_index(
        ds,
        [
            MediaClip(
                sequence="vid1",
                filename="c0.mp4",
                video_uuid="uid-replaced",
                frame_count=2 * _FRAMES,
            )
        ],
    )
    _ = ds.rebuild_sequence_index("media_raw")

    done = backfill_media_frames(ds)

    assert _recorded(ds) == _FRAMES
    assert len(done.not_established) == 1
    assert (len(done.written), len(done.cleared)) == (0, 0)


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

    assert len(backfill_media_frames(ds).written) == 0
    assert _recorded(ds) is None


_CORE_ONLY_BACKFILL = """
import json
import sys

from mosaic.core.dataset import open_dataset

ds = open_dataset(sys.argv[1])
media = ds.measure_media_frames(dry_run=True)
read = ds.measure_frames_read(dry_run=True)
tail = ds.measure_known_tail_loss(dry_run=True)
print(json.dumps({
    "tracking": "mosaic.tracking" in sys.modules,
    "media": [len(media.written), len(media.not_established)],
    "read": [len(read.written), len(read.not_established)],
    "tail": [len(tail.written), len(tail.not_established)],
    "unregistered": [
        sorted(media.unregistered["producer"]),
        sorted(read.unregistered["producer"]),
        sorted(tail.unregistered["producer"]),
    ],
}))
"""


def test_a_row_whose_producer_is_not_registered_is_reported_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that never imported the producers cannot read their rows.

    Filling nothing there would read as "nothing to fill" when the answer is
    "this process cannot tell", so those rows are reported apart from the ones
    whose run cannot be established.

    Run in a subprocess, because registration is a process-global import side
    effect that this test module has already paid.
    """
    ds = _dataset(tmp_path)
    _trex_one_clip(ds, tmp_path, monkeypatch)
    set_tracks_cell(ds, "media_frames", "")
    set_tracks_cell(ds, "frames_read", "")

    completed = subprocess.run(
        [sys.executable, "-c", _CORE_ONLY_BACKFILL, str(ds.manifest_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    reported = json.loads(completed.stdout.strip().splitlines()[-1])

    assert reported["tracking"] is False, "the probe imported mosaic.tracking"
    assert reported["media"] == [0, 0]
    assert reported["read"] == [0, 0]
    assert reported["tail"] == [0, 0]
    assert reported["unregistered"] == [["trex"], ["trex"], ["trex"]]

    # The same passes in this process, where the producers are registered. The
    # fake TREx read a stub clip, whose header tells no loss.
    media = backfill_media_frames(ds, dry_run=True)
    read = backfill_frames_read(ds, dry_run=True)
    tail = backfill_known_tail_loss(ds, dry_run=True)
    assert (len(media.written), len(media.unregistered)) == (1, 0)
    assert (len(read.written), len(read.unregistered)) == (1, 0)
    assert (len(tail.written), len(tail.unregistered)) == (0, 0)
