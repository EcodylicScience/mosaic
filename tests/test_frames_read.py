"""Each tracks row records how many frames its producer's tool read.

The count is the tool's, not the table's. A table has rows only where the tool
reported something, and a recording that ends with no animal in view ends its
table early however many frames the tool read. So each producer passes what its
tool read: the frames of TREx's ``.pv``, the frames a runner decoded, the frames
Lightning Pose predicted. SLEAP's analysis export cannot say, so its cell stays
blank.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import mosaic.tracking.litpose.dataset_runs as litpose_runs
import mosaic.tracking.sleap.dataset_runs as sleap_runs
import mosaic.tracking.trex.dataset_runs as trex_runs
import mosaic.tracking.ultralytics_track.dataset_runs as ultralytics_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.tracks_index import (
    TRACKS_INDEX_COLUMNS,
    backfill_frames_read,
    read_frames_read,
    read_tracks_index,
    tracks_index_path,
    write_tracks_row,
)
from mosaic.core.scope import Scope
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.pose_training.localizer_inference import LocalizerDetection
from mosaic.tracking.pose_training.ultralytics_infer import INFER_RESPONSE_NAME
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from mosaic.tracking.ultralytics_track.run import TRACK_RESPONSE_NAME
from tests.helpers import (
    MediaClip,
    install_fake_litpose,
    install_fake_pose_inference,
    install_fake_sleap,
    install_fake_trex,
    install_fake_ultralytics,
    latest_events,
    latest_snapshot,
    make_dataset,
    paint_frame_code,
    set_tracks_cell,
    write_litpose_model,
    write_media_index,
    write_painted_entry,
    write_sleap_model,
)

_FRAMES = 30
"""How many frames the one clip of each dataset holds."""


def _dataset(tmp_path: Path, frames: int = _FRAMES) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(
                sequence="vid1",
                filename="vid1.mp4",
                video_uuid="uid-vid1",
                frame_count=frames,
            )
        ],
    )
    return ds


def _recorded(ds: Dataset) -> int | None:
    """Return the frames read that the dataset's one tracks row records."""
    rows = read_tracks_index(ds)
    assert len(rows) == 1
    return read_frames_read(rows.iloc[0])


class TestTheCell:
    def test_a_blank_cell_reads_as_unknown_not_as_zero(self) -> None:
        rows = pd.DataFrame({"frames_read": ["", "1798", "many"]})
        assert [read_frames_read(row) for _, row in rows.iterrows()] == [
            None,
            1798,
            None,
        ]

    def test_an_index_written_before_the_cell_is_adopted(self, tmp_path: Path) -> None:
        """Read without it as unknown, and widened by the next write."""
        ds = make_dataset(tmp_path / "ds")
        old = ds.get_root("tracks") / "old" / "a.parquet"
        new = ds.get_root("tracks") / "new" / "b.parquet"
        for path in (old, new):
            path.parent.mkdir(parents=True)
            pd.DataFrame({"frame": [0, 1], "id": [0, 0]}).to_parquet(path)
        write_tracks_row(
            ds,
            run_id="old",
            group="",
            sequence="a",
            out_path=old,
            producer="trex",
            std_format="trex_v2",
            n_rows=2,
        )
        index = tracks_index_path(ds)
        legacy = pd.read_csv(index, dtype=str, keep_default_na=False)
        legacy.drop(columns=["frames_read"]).to_csv(index, index=False)

        assert read_frames_read(read_tracks_index(ds).iloc[0]) is None

        write_tracks_row(
            ds,
            run_id="new",
            group="",
            sequence="b",
            out_path=new,
            producer="trex",
            std_format="trex_v2",
            n_rows=2,
            frames_read=40,
        )

        on_disk = pd.read_csv(index, dtype=str, keep_default_na=False)
        assert list(on_disk.columns) == TRACKS_INDEX_COLUMNS
        recorded = dict(zip(on_disk["run_id"], on_disk["frames_read"], strict=True))
        assert recorded == {"old": "", "new": "40"}


class TestEachProducerRecordsWhatItsToolRead:
    def test_trex_records_the_frames_of_its_conversion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its exports end at the last frame an animal was tracked in.

        The count is the ``.pv`` header's, which counts every frame converted.
        """
        ds = _dataset(tmp_path)
        trex = install_fake_trex(monkeypatch)
        trex.npz_frames, trex.pv_frames = 20, 28

        _ = trex_runs.run_trex(ds, TrexParams())

        assert _recorded(ds) == 28

    def test_a_trex_republish_records_them_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        trex = install_fake_trex(monkeypatch)
        trex.npz_frames, trex.pv_frames = 20, 28
        _ = trex_runs.run_trex(ds, TrexParams())
        set_tracks_cell(ds, "frames_read", "")

        _ = trex_runs.run_trex(ds, TrexParams(), republish=True)

        assert _recorded(ds) == 28

    def test_lightning_pose_records_the_frames_it_predicted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        litpose = install_fake_litpose(monkeypatch)
        litpose.frames = 25
        model = write_litpose_model(tmp_path / "litpose_model")

        _ = litpose_runs.run_litpose(ds, LitposeParams(model_path=str(model)))

        assert _recorded(ds) == 25

    def test_ultralytics_records_the_frames_its_runner_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Detections in the first twenty frames, of the thirty the runner read."""
        ds = _dataset(tmp_path)
        ultralytics = install_fake_ultralytics(monkeypatch)
        ultralytics.n_frames = 20
        model = tmp_path / "yolo" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")

        _ = ultralytics_runs.run_ultralytics(
            ds, UltralyticsParams.model_validate({"model_path": str(model)})
        )

        assert _recorded(ds) == _FRAMES

    def test_an_inference_run_records_the_frames_its_runner_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fake predicts four frames, of the thirty the runner read."""
        ds = _dataset(tmp_path)
        _ = install_fake_pose_inference(monkeypatch)
        model = tmp_path / "weights" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")

        _ = run_op(
            ds, "infer-pose", {"model": str(model)}, scope=Scope(entries=[("", "vid1")])
        )

        assert _recorded(ds) == _FRAMES

    def test_a_runner_one_frame_short_is_a_mismatch_not_a_tail_loss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only TREx declares a shortfall at the end of a file."""
        ds = _dataset(tmp_path)
        ultralytics = install_fake_ultralytics(monkeypatch)
        ultralytics.frames_read = _FRAMES - 1
        model = tmp_path / "yolo" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")

        _ = ultralytics_runs.run_ultralytics(
            ds, UltralyticsParams.model_validate({"model_path": str(model)})
        )

        assert ds.frame_tail_shortfalls() == ()
        (found,) = ds.frame_axis_mismatches()
        assert (found.read, found.media) == (_FRAMES - 1, _FRAMES)
        snapshot = latest_snapshot(ds)
        assert snapshot["entries_frame_axis_mismatch"] == 1
        assert snapshot["entries_frame_tail_short"] == 0

    def test_a_runner_that_read_past_its_media_is_a_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames past the end of the media are no tail loss and no agreement."""
        ds = _dataset(tmp_path, frames=60)
        ultralytics = install_fake_ultralytics(monkeypatch)
        ultralytics.frames_read = 62
        model = tmp_path / "yolo" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")

        _ = ultralytics_runs.run_ultralytics(
            ds, UltralyticsParams.model_validate({"model_path": str(model)})
        )

        reported = [
            (event["key"], event["read"], event["media"])
            for event in latest_events(ds, "frame_axis_mismatch")
        ]
        assert reported == [("vid1", 62, 60)]
        assert latest_events(ds, "frame_tail_short") == []
        (found,) = ds.frame_axis_mismatches()
        assert (found.read, found.media) == (62, 60)

    def test_sleap_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its analysis export does not say whether it spans the video."""
        ds = _dataset(tmp_path)
        _ = install_fake_sleap(monkeypatch)
        model = write_sleap_model(tmp_path / "sleap_model")

        _ = sleap_runs.run_sleap(ds, SleapParams(model_paths=[str(model)]))

        assert _recorded(ds) is None


@pytest.mark.media
def test_the_localizer_records_every_frame_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    """It detects nothing, and read all twenty frames."""
    import mosaic.tracking.pose_training.localizer_inference as localizer

    ds = make_dataset(tmp_path / "ds")
    _ = write_painted_entry(ds, "sess", [(20, 25.0)], paint_frame_code)

    def load(_model_path: object, **_kwargs: object) -> object:
        return object()

    def detect(
        _encoder: object, _image: object, *_args: object, **_kwargs: object
    ) -> list[LocalizerDetection]:
        return []

    monkeypatch.setattr(localizer, "_load_encoder", load)
    monkeypatch.setattr(localizer, "detect_locations", detect)
    model = tmp_path / "weights" / "best.pt"
    model.parent.mkdir(parents=True)
    _ = model.write_bytes(b"weights")

    _ = run_op(
        ds,
        "infer-localizer",
        {"model": str(model)},
        scope=Scope(entries=[("", "sess")]),
    )

    assert _recorded(ds) == 20


# --- a row published before the cell existed ----------------------------------


def _blank_frames_read(ds: Dataset) -> None:
    set_tracks_cell(ds, "frames_read", "")


def _work_dir(ds: Dataset) -> Path:
    """The working directory that the one tracks row's producer wrote."""
    rows = read_tracks_index(ds)
    return ds.resolve_path(str(rows.iloc[0]["source_abs_path"]))


def _write_response(path: Path, **counts: int) -> None:
    """Write a runner's response, as the Ultralytics runner writes it."""
    _ = path.write_text(json.dumps(counts))


class TestAPastRowIsFilledFromWhatItsRunLeft:
    def test_trex_from_its_pv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        trex = install_fake_trex(monkeypatch)
        trex.npz_frames, trex.pv_frames = 20, 28
        _ = trex_runs.run_trex(ds, TrexParams())
        _blank_frames_read(ds)

        would = backfill_frames_read(ds, dry_run=True)
        assert len(would.written) == 1
        assert _recorded(ds) is None, "a dry run must not write"

        assert len(backfill_frames_read(ds).written) == 1
        assert _recorded(ds) == 28
        assert len(backfill_frames_read(ds).written) == 0, "not idempotent"

    def test_trex_stays_blank_once_its_pv_is_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        _ = install_fake_trex(monkeypatch)
        _ = trex_runs.run_trex(ds, TrexParams())
        _blank_frames_read(ds)
        for pv in ds.get_root("trex-convert").rglob("*.pv"):
            pv.unlink()
        for pv in ds.get_root("trex").rglob("*.pv"):
            pv.unlink()

        assert len(backfill_frames_read(ds).written) == 0
        assert _recorded(ds) is None

    def test_trex_keeps_its_count_once_its_pv_is_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing file establishes nothing, so the recorded count stays."""
        ds = _dataset(tmp_path)
        trex = install_fake_trex(monkeypatch)
        trex.npz_frames, trex.pv_frames = 20, 28
        _ = trex_runs.run_trex(ds, TrexParams())
        for pv in ds.get_root("trex-convert").rglob("*.pv"):
            pv.unlink()
        for pv in ds.get_root("trex").rglob("*.pv"):
            pv.unlink()

        done = backfill_frames_read(ds)

        assert _recorded(ds) == 28
        assert (len(done.written), len(done.cleared)) == (0, 0)
        assert len(done.not_established) == 1

    def test_trex_count_that_disagrees_with_its_pv_is_rewritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        trex = install_fake_trex(monkeypatch)
        trex.npz_frames, trex.pv_frames = 20, 28
        _ = trex_runs.run_trex(ds, TrexParams())
        set_tracks_cell(ds, "frames_read", "5")

        done = backfill_frames_read(ds)

        assert _recorded(ds) == 28
        assert len(done.written) == 1

    def test_a_table_made_from_another_table_is_cleared(self, tmp_path: Path) -> None:
        """No tool read media to make it, so no count of frames read applies."""
        ds = _dataset(tmp_path)
        out = ds.get_root("tracks") / "convert-x.0.1-0123456789" / "vid1.parquet"
        out.parent.mkdir(parents=True)
        pd.DataFrame({"frame": [0, 1], "id": [0, 0]}).to_parquet(out)
        write_tracks_row(
            ds,
            run_id="convert-x.0.1-0123456789",
            group="",
            sequence="vid1",
            out_path=out,
            producer="convert-x",
            std_format="mosaic_v1",
            n_rows=2,
            frames_read=_FRAMES,
        )

        would = backfill_frames_read(ds, dry_run=True)
        assert len(would.cleared) == 1
        assert _recorded(ds) == _FRAMES, "a dry run must not write"

        assert len(backfill_frames_read(ds).cleared) == 1
        assert _recorded(ds) is None

    def test_lightning_pose_from_its_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        litpose = install_fake_litpose(monkeypatch)
        litpose.frames = 25
        model = write_litpose_model(tmp_path / "litpose_model")
        _ = litpose_runs.run_litpose(ds, LitposeParams(model_path=str(model)))
        _blank_frames_read(ds)

        _ = backfill_frames_read(ds)

        assert _recorded(ds) == 25

    def test_ultralytics_from_its_runner_s_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The response counts every frame read, the predictions only detections."""
        ds = _dataset(tmp_path)
        ultralytics = install_fake_ultralytics(monkeypatch)
        ultralytics.n_frames = 20
        model = tmp_path / "yolo" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")
        _ = ultralytics_runs.run_ultralytics(
            ds, UltralyticsParams.model_validate({"model_path": str(model)})
        )
        _blank_frames_read(ds)
        _ = backfill_frames_read(ds)
        assert _recorded(ds) == _FRAMES

        _write_response(_work_dir(ds) / TRACK_RESPONSE_NAME, n_frames=29, n_ids=2)
        _ = backfill_frames_read(ds)

        assert _recorded(ds) == 29

    def test_an_inference_run_from_its_runner_s_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        _ = install_fake_pose_inference(monkeypatch)
        model = tmp_path / "weights" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")
        _ = run_op(
            ds, "infer-pose", {"model": str(model)}, scope=Scope(entries=[("", "vid1")])
        )
        _blank_frames_read(ds)
        _write_response(_work_dir(ds) / INFER_RESPONSE_NAME, n_frames=27, n_rows=4)

        _ = backfill_frames_read(ds)

        assert _recorded(ds) == 27

    def test_sleap_stays_blank(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        _ = install_fake_sleap(monkeypatch)
        model = write_sleap_model(tmp_path / "sleap_model")
        _ = sleap_runs.run_sleap(ds, SleapParams(model_paths=[str(model)]))

        assert len(backfill_frames_read(ds).written) == 0
        assert _recorded(ds) is None


def test_a_reused_ultralytics_run_republishes_its_runner_s_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table is gone and the tracking is not, so only the bridge runs again."""
    ds = _dataset(tmp_path)
    ultralytics = install_fake_ultralytics(monkeypatch)
    model = tmp_path / "yolo" / "best.pt"
    model.parent.mkdir(parents=True)
    _ = model.write_bytes(b"weights")
    params = UltralyticsParams.model_validate({"model_path": str(model)})
    _ = ultralytics_runs.run_ultralytics(ds, params)
    _write_response(_work_dir(ds) / TRACK_RESPONSE_NAME, n_frames=29, n_ids=2)
    table = ds.resolve_path(str(read_tracks_index(ds).iloc[0]["abs_path"]))
    table.unlink()

    _ = ultralytics_runs.run_ultralytics(ds, params)

    assert len(ultralytics.tracked) == 1, "the tracking was reused"
    assert _recorded(ds) == 29
