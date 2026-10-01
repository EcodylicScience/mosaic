"""Every producer's table is placed on its entry's axes by the shared bridge.

A tool reads one file: the entry's one clip, the join of its clips, or a media
variant's file. :class:`EntryAxis` says which, and the bridge that every tracker
and inference op publishes through does two things with it. A table from a join
is timed by the clips' own rates, which a tool reading the join at one rate cannot
do. And the tracks row records how many frames the tool should have read, beside
how many it did, so a tool that read short is reported as a
``frame_axis_mismatch``. The table's own extent is not compared: a table with
rows only at detections ends at its last one, however many frames the tool read.

The rules are tested on the value itself. The runs are Lightning Pose and
Ultralytics tracker runs over real clips and a real join, with the recording fakes
standing in for the tools.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mosaic_media import MediaFacts

import mosaic.tracking.litpose.dataset_runs as litpose_runs
import mosaic.tracking.sleap.dataset_runs as sleap_runs
import mosaic.tracking.ultralytics_track.dataset_runs as ultralytics_runs
from mosaic.core.dataset import Dataset
from mosaic.core.media.preprocess import Placement, TrimStep
from mosaic.core.media.timeline import concatenated_timeline
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.placement import EntryAxis, SourceMapping
from mosaic.core.pipeline.tracks_index import read_media_frames, read_tracks_index
from mosaic.core.scope import Scope
from mosaic.runlog import reduce_run_log, run_log_dir
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from tests.helpers import (
    FakeLitpose,
    FakeUltralytics,
    clip_facts,
    install_fake_litpose,
    install_fake_pose_inference,
    install_fake_sleap,
    install_fake_ultralytics,
    make_dataset,
    write_litpose_model,
    write_painted_entry,
    write_sleap_model,
)

_CLIP = 300


def _variant(*rates: float, trim: bool) -> SourceMapping:
    """A variant of one clip of ``_CLIP`` frames per rate, trimmed or whole."""
    facts = [clip_facts(fps=rate, frame_count=_CLIP) for rate in rates]
    total = _CLIP * len(rates)
    placement = Placement.identity(64, 48, total, rates[0])
    if trim:
        placement = TrimStep(start=10, stop=total - 10).place(placement)
    return SourceMapping(placement, concatenated_timeline(facts))


class TestTheMediaAxisATableIsMeantToSpan:
    def test_the_join_of_two_clips_spans_both(self) -> None:
        axis = EntryAxis.of_entry_media(
            [clip_facts(frame_count=_CLIP), clip_facts(frame_count=_CLIP)],
            windowed=False,
        )
        assert axis.media_frames == 2 * _CLIP

    def test_one_clip_is_read_whole_too(self) -> None:
        """A tool can lose frames at the end of one file, as at the end of a join."""
        axis = EntryAxis.of_entry_media([clip_facts(frame_count=_CLIP)], windowed=False)
        assert axis.media_frames == _CLIP

    def test_a_frame_window_asks_no_question(self) -> None:
        axis = EntryAxis.of_entry_media([clip_facts(), clip_facts()], windowed=True)
        assert axis.media_frames is None

    def test_absent_facts_ask_no_question(self) -> None:
        assert EntryAxis.of_entry_media([], windowed=False).media_frames is None

    def test_a_clip_of_unknown_length_asks_no_question(self) -> None:
        """A partial sum would read as a measurement."""
        clips = [clip_facts(frame_count=_CLIP), clip_facts(frame_count=0)]
        assert EntryAxis.of_entry_media(clips, windowed=False).media_frames is None

    def test_a_variant_keeping_every_frame_of_two_clips_spans_the_source(
        self,
    ) -> None:
        axis = EntryAxis.of_variant(_variant(30.0, 30.0, trim=False))
        assert axis.media_frames == 2 * _CLIP

    def test_a_variant_keeping_every_frame_of_one_clip_spans_the_clip(
        self,
    ) -> None:
        axis = EntryAxis.of_variant(_variant(30.0, trim=False))
        assert axis.media_frames == _CLIP

    def test_a_trimmed_variant_asks_no_question(self) -> None:
        """Its table spans the trimmed range, so the source axis would misreport."""
        axis = EntryAxis.of_variant(_variant(30.0, 30.0, trim=True))
        assert axis.media_frames is None

    def test_an_axis_is_the_entry_media_or_a_variant(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            _ = EntryAxis(
                clips=(clip_facts(),), mapping=_variant(30.0, 30.0, trim=False)
            )


class TestPlacingATable:
    @staticmethod
    def _table(frames: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {"frame": frames, "time": [frame / 30.0 for frame in frames]}
        )

    def test_a_join_of_two_rates_is_timed_by_each_clip(self) -> None:
        axis = EntryAxis.of_entry_media(
            [
                clip_facts(fps=30.0, frame_count=_CLIP),
                clip_facts(fps=31.0, frame_count=_CLIP),
            ],
            windowed=False,
        )

        placed = axis.place(self._table([0, _CLIP - 1, _CLIP, _CLIP + 31]))

        assert placed.frame["time"].tolist() == pytest.approx(
            [0.0, (_CLIP - 1) / 30.0, _CLIP / 30.0, _CLIP / 30.0 + 1.0]
        )

    def test_one_clip_keeps_the_tool_s_time(self) -> None:
        table = self._table([0, 10])

        placed = EntryAxis.of_entry_media([clip_facts()], windowed=False).place(table)

        assert placed.frame is table
        assert placed.dropped == ()


# --- tracker runs over a real two-clip entry ---------------------------------


_FRAMES_PER_CLIP = 30
_ENTRY = ("", "sess")


def _paint(frame: int) -> npt.NDArray[np.uint8]:
    return np.full((48, 64, 3), 20 + 8 * (frame % 25), np.uint8)


@pytest.fixture
def session(tmp_path: Path, requires_ffmpeg: None) -> Dataset:
    """One entry of thirty frames at 30 fps, then thirty at 31 fps, and its join."""
    ds = make_dataset(tmp_path / "ds")
    _ = write_painted_entry(
        ds, "sess", [(_FRAMES_PER_CLIP, 30.0), (_FRAMES_PER_CLIP, 31.0)], _paint
    )
    _ = run_op(ds, "export-joined", {}, scope=Scope(entries=[_ENTRY]))
    return ds


@pytest.fixture
def ultralytics(monkeypatch: pytest.MonkeyPatch) -> FakeUltralytics:
    return install_fake_ultralytics(monkeypatch)


@pytest.fixture
def litpose(monkeypatch: pytest.MonkeyPatch) -> FakeLitpose:
    return install_fake_litpose(monkeypatch)


def _predict(ds: Dataset, tmp_path: Path, fake: FakeLitpose, frames: int) -> None:
    """Run Lightning Pose, which predicts *frames* frames, over the dataset."""
    fake.frames = frames
    model = write_litpose_model(tmp_path / "litpose_model")
    _ = litpose_runs.run_litpose(ds, LitposeParams(model_path=str(model)))


def _track(ds: Dataset, tmp_path: Path, fake: FakeUltralytics, frames: int) -> None:
    model = tmp_path / "yolo" / "best.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    _ = model.write_bytes(b"weights")
    fake.n_frames, fake.n_ids = frames, 1
    params = UltralyticsParams.model_validate({"model_path": str(model)})
    _ = ultralytics_runs.run_ultralytics(ds, params)


def _published(ds: Dataset) -> tuple[int | None, pd.DataFrame]:
    """Return the one tracks row's recorded media axis, and its table."""
    rows = read_tracks_index(ds)
    assert len(rows) == 1
    row = rows.iloc[0]
    table = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
    return read_media_frames(row), table


def _latest_events(ds: Dataset) -> list[dict[str, object]]:
    logs = sorted(run_log_dir(ds.base_dir).glob("*.jsonl"))
    latest = max(logs, key=lambda path: path.stat().st_mtime)
    return [json.loads(line) for line in latest.read_text().splitlines()]


def _mismatch_events(ds: Dataset) -> list[tuple[object, object]]:
    return [
        (event["read"], event["media"])
        for event in _latest_events(ds)
        if event["ev"] == "frame_axis_mismatch"
    ]


@pytest.mark.media
class TestATrackerOverAJoinedEntry:
    def test_its_row_records_the_clips_summed(
        self, session: Dataset, tmp_path: Path, litpose: FakeLitpose
    ) -> None:
        _predict(session, tmp_path, litpose, 2 * _FRAMES_PER_CLIP)

        assert litpose.predicted[0].name.endswith(".joined.mp4")
        assert _published(session)[0] == 2 * _FRAMES_PER_CLIP
        assert session.frame_axis_mismatches() == ()

    def test_its_time_follows_each_clip_s_own_rate(
        self, session: Dataset, tmp_path: Path, ultralytics: FakeUltralytics
    ) -> None:
        """The converter times every frame at the first clip's 30 fps."""
        _track(session, tmp_path, ultralytics, 2 * _FRAMES_PER_CLIP)

        _media_frames, table = _published(session)
        time_of = dict(zip(table["frame"].tolist(), table["time"].tolist()))
        first = _FRAMES_PER_CLIP / 30.0
        assert time_of[_FRAMES_PER_CLIP - 1] == pytest.approx(
            (_FRAMES_PER_CLIP - 1) / 30.0
        )
        assert time_of[_FRAMES_PER_CLIP + 15] == pytest.approx(first + 15 / 31.0)
        assert time_of[2 * _FRAMES_PER_CLIP - 1] == pytest.approx(
            first + (_FRAMES_PER_CLIP - 1) / 31.0
        )

    def test_a_short_read_is_reported_and_published(
        self, session: Dataset, tmp_path: Path, litpose: FakeLitpose
    ) -> None:
        _predict(session, tmp_path, litpose, 2 * _FRAMES_PER_CLIP - 10)

        (found,) = session.frame_axis_mismatches()
        assert (found.read, found.media) == (50, 60)
        assert _mismatch_events(session) == [(50, 60)]
        logs = sorted(run_log_dir(session.base_dir).glob("*.jsonl"))
        snapshot = reduce_run_log(max(logs, key=lambda path: path.stat().st_mtime))
        assert snapshot is not None
        assert snapshot["entries_frame_axis_mismatch"] == 1
        assert snapshot["entries_failed"] == 0

    def test_a_table_that_ends_before_its_media_is_not_a_short_read(
        self, session: Dataset, tmp_path: Path, ultralytics: FakeUltralytics
    ) -> None:
        """Ultralytics read every frame, and no animal was seen in the last ten."""
        _track(session, tmp_path, ultralytics, 2 * _FRAMES_PER_CLIP - 10)

        media_frames, table = _published(session)
        assert int(table["frame"].max()) == 2 * _FRAMES_PER_CLIP - 11
        assert media_frames == 2 * _FRAMES_PER_CLIP
        assert session.frame_axis_mismatches() == ()
        assert _mismatch_events(session) == []

    def test_a_runner_that_read_short_is_reported(
        self, session: Dataset, tmp_path: Path, ultralytics: FakeUltralytics
    ) -> None:
        """Its predictions look the same whether or not it read the last frames."""
        ultralytics.frames_read = 2 * _FRAMES_PER_CLIP - 2
        _track(session, tmp_path, ultralytics, 2 * _FRAMES_PER_CLIP - 10)

        (found,) = session.frame_axis_mismatches()
        assert (found.read, found.media) == (58, 60)
        assert _mismatch_events(session) == [(58, 60)]


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record whether each axis a producer builds of the entry media is windowed."""
    seen: list[bool] = []
    build = EntryAxis.of_entry_media

    def of_entry_media(clips: Sequence[MediaFacts], *, windowed: bool) -> EntryAxis:
        seen.append(windowed)
        return build(clips, windowed=windowed)

    monkeypatch.setattr(EntryAxis, "of_entry_media", of_entry_media)
    return seen


@pytest.mark.media
class TestAWindowedRunSaysSo:
    """A producer run under a frame window tells the bridge, whatever its table."""

    def test_sleap(
        self,
        session: Dataset,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        windows: list[bool],
    ) -> None:
        _ = install_fake_sleap(monkeypatch)
        model = write_sleap_model(tmp_path / "sleap_model")

        _ = sleap_runs.run_sleap(
            session, SleapParams(model_paths=[str(model)], analysis_range=(0, 10))
        )

        assert windows == [True]

    def test_ultralytics(
        self,
        session: Dataset,
        tmp_path: Path,
        ultralytics: FakeUltralytics,
        windows: list[bool],
    ) -> None:
        model = tmp_path / "yolo" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")

        _ = ultralytics_runs.run_ultralytics(
            session,
            UltralyticsParams.model_validate(
                {"model_path": str(model), "start_frame": 5}
            ),
        )

        assert windows == [True]

    def test_an_inference_op(
        self,
        session: Dataset,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        windows: list[bool],
    ) -> None:
        model = tmp_path / "weights" / "best.pt"
        model.parent.mkdir(parents=True)
        _ = model.write_bytes(b"weights")
        _ = install_fake_pose_inference(monkeypatch)

        _ = run_op(
            session,
            "infer-pose",
            {"model": str(model), "start_frame": 5},
            scope=Scope(entries=[_ENTRY]),
        )

        assert windows == [True]
