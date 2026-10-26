"""A tracker places a continuous group's sequence at its first frame.

The tool reads the sequence's own media, and the bridge numbers the table from
the sequence's first frame on its recording's axis: the media frames of the members
before it. The tracks row records that first frame. When an earlier member's
media changes length, the row reads as drifted, and the next run republishes the
table at the new first frame without running the tool again.

The fixture records ``rec`` as ``s1``, frames 0 to 4, and ``s2``, frames 5 to 8.
The SLEAP stand-in reports one track in each of four frames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import mosaic.tracking.sleap.dataset_runs as sleap_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.tracks_index import (
    media_drift,
    read_tracks_index,
    tracks_index_path,
    write_tracks_row,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.scope import Scope
from mosaic.tracking.common.scope import build_work_items
from mosaic.tracking.sleap.params import SleapParams

from tests.helpers import (
    FakeSleap,
    install_fake_sleap,
    make_dataset,
    paint_frame_code,
    scope_over,
    write_painted_entry,
    write_painted_recording,
    write_sleap_model,
)

pytestmark = pytest.mark.media

_SIZE = (64, 48)
_SECOND = ("rec", "s2")
_FIRST_FRAMES = 5
_SECOND_FRAMES = 4


@pytest.fixture
def undeclared(tmp_path: Path, requires_ffmpeg: None) -> Dataset:
    """A dataset whose group ``rec`` has two sequences with media, not yet continuous."""
    ds = make_dataset(tmp_path / "ds")
    _ = write_painted_recording(
        ds,
        "rec",
        [("s1", [(_FIRST_FRAMES, 30.0)]), ("s2", [(_SECOND_FRAMES, 30.0)])],
        size=_SIZE,
    )
    return ds


@pytest.fixture
def recording(undeclared: Dataset) -> Dataset:
    """A dataset whose continuous group ``rec`` has two sequences with media."""
    undeclared.set_continuous_groups(["rec"])
    return undeclared


@pytest.fixture
def sleap(monkeypatch: pytest.MonkeyPatch) -> FakeSleap:
    return install_fake_sleap(monkeypatch, FakeSleap(frames=_SECOND_FRAMES))


@pytest.fixture
def sleap_model(tmp_path: Path) -> Path:
    return write_sleap_model(tmp_path / "sleap_model")


def _variant(ds: Dataset) -> str:
    """Write a cropped variant of ``s2``, which keeps every frame, and return its id."""
    crop = {"step": "crop", "x": 8, "y": 4, "width": 32, "height": 24}
    return run_op(
        ds,
        "preprocess",
        {"steps": [crop], "codec": "h264"},
        scope=Scope(entries=[_SECOND]),
    )


def _sleap_rows(ds: Dataset) -> list[pd.Series]:
    rows = read_tracks_index(ds)
    return [row for _, row in rows[rows["producer"] == "sleap"].iterrows()]


def _frames(ds: Dataset, row: pd.Series) -> list[int]:
    table = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
    return sorted({int(frame) for frame in table["frame"]})


def _grow_the_first_member(ds: Dataset) -> None:
    """Write ``s1`` again, two frames longer, which moves ``s2``'s first frame to 7."""
    _ = write_painted_entry(
        ds,
        "s1",
        [(_FIRST_FRAMES + 2, 30.0)],
        lambda frame: paint_frame_code(frame, _SIZE),
        size=_SIZE,
        group="rec",
    )


def _set_first_frame_cells(ds: Dataset, cell: str) -> None:
    """Rewrite every tracks row's ``first_frame`` cell, as a row of another version has it."""
    rows = read_tracks_index(ds)
    rows["first_frame"] = cell
    rows.to_csv(tracks_index_path(ds), index=False)


class TestTheWorkItem:
    """Each item carries the sequence's first frame into its entry axis."""

    def test_an_entry_media_item_starts_at_the_sequence_s_first_frame(
        self, recording: Dataset
    ) -> None:
        built = build_work_items(
            recording, recording.resolve_media_scope([_SECOND]), kind="sleap"
        )

        (item,) = built.items
        assert item.first_frame == _FIRST_FRAMES
        assert item.entry_axis(windowed=False).first_frame == _FIRST_FRAMES

    def test_a_variant_item_starts_at_the_sequence_s_first_frame(
        self, recording: Dataset
    ) -> None:
        variant = _variant(recording)

        built = build_work_items(
            recording,
            recording.resolve_media_scope([_SECOND]),
            kind="sleap",
            media=variant,
        )

        assert built.failures == ()
        (item,) = built.items
        assert item.variant is not None
        assert item.first_frame == _FIRST_FRAMES
        assert item.entry_axis(windowed=False).first_frame == _FIRST_FRAMES


class TestAPublishedTable:
    """A tracker's table of a later member starts at the member's first frame."""

    def test_a_table_from_the_entry_media_starts_at_the_first_frame(
        self, recording: Dataset, sleap: FakeSleap, sleap_model: Path
    ) -> None:
        params = SleapParams(model_paths=[str(sleap_model)])

        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))

        (row,) = _sleap_rows(recording)
        assert _frames(recording, row) == [5, 6, 7, 8]
        assert str(row["first_frame"]) == str(_FIRST_FRAMES)

    def test_a_table_from_a_variant_starts_at_the_first_frame(
        self, recording: Dataset, sleap: FakeSleap, sleap_model: Path
    ) -> None:
        variant = _variant(recording)
        params = SleapParams(model_paths=[str(sleap_model)], media=variant)

        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))

        (row,) = _sleap_rows(recording)
        assert _frames(recording, row) == [5, 6, 7, 8]


class TestAMovedFirstFrame:
    """An earlier member that changes length moves the later member's first frame."""

    def test_the_table_reads_as_drifted_and_is_republished_at_the_new_first_frame(
        self, recording: Dataset, sleap: FakeSleap, sleap_model: Path
    ) -> None:
        params = SleapParams(model_paths=[str(sleap_model)])
        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))
        assert media_drift(recording, read_tracks_index(recording)) == ()

        _grow_the_first_member(recording)

        assert media_drift(recording, read_tracks_index(recording)) == (_SECOND,)

        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))

        assert len(sleap.tracked) == 1
        (row,) = _sleap_rows(recording)
        assert _frames(recording, row) == [7, 8, 9, 10]
        assert media_drift(recording, read_tracks_index(recording)) == ()


class TestAGroupDeclaredAfterTracking:
    """A table tracked before its group was declared continuous starts at frame 0.

    A row with a blank ``first_frame`` cell and a recorded media composition was
    written by a tracker before the cell existed, and that tracker numbered every
    table from frame 0.
    """

    @pytest.mark.parametrize("cell", ["0", ""], ids=["recorded", "blank"])
    def test_the_table_reads_as_drifted_and_is_republished_at_the_first_frame(
        self, undeclared: Dataset, sleap: FakeSleap, sleap_model: Path, cell: str
    ) -> None:
        params = SleapParams(model_paths=[str(sleap_model)])
        _ = sleap_runs.run_sleap(undeclared, params, scope_over(_SECOND))
        _set_first_frame_cells(undeclared, cell)
        (row,) = _sleap_rows(undeclared)
        assert str(row["consumed_media_composition"])
        assert _frames(undeclared, row) == [0, 1, 2, 3]

        undeclared.set_continuous_groups(["rec"])

        assert media_drift(undeclared, read_tracks_index(undeclared)) == (_SECOND,)

        _ = sleap_runs.run_sleap(undeclared, params, scope_over(_SECOND))

        assert len(sleap.tracked) == 1
        (row,) = _sleap_rows(undeclared)
        assert _frames(undeclared, row) == [5, 6, 7, 8]
        assert media_drift(undeclared, read_tracks_index(undeclared)) == ()


class TestAGroupNoLongerDeclared:
    """A table placed past frame 0 is stale once its group is no longer continuous."""

    def test_the_table_reads_as_drifted_and_is_republished_at_frame_0(
        self, recording: Dataset, sleap: FakeSleap, sleap_model: Path
    ) -> None:
        params = SleapParams(model_paths=[str(sleap_model)])
        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))

        recording.set_continuous_groups([])

        assert media_drift(recording, read_tracks_index(recording)) == (_SECOND,)

        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))

        assert len(sleap.tracked) == 1
        (row,) = _sleap_rows(recording)
        assert _frames(recording, row) == [0, 1, 2, 3]
        assert media_drift(recording, read_tracks_index(recording)) == ()


class TestAConversion:
    """A converted table keeps its source's frame numbers, so its blank cell is unknown."""

    def test_a_later_member_numbered_across_the_recording_is_not_drifted(
        self, recording: Dataset
    ) -> None:
        variant = "convert-numbered.0.1-aaaaaaaaaa"
        out = recording.get_root("tracks") / variant / "rec__s2.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = list(range(_FIRST_FRAMES, _FIRST_FRAMES + _SECOND_FRAMES))
        table = pd.DataFrame(
            {
                "frame": frames,
                "time": [frame / 30.0 for frame in frames],
                "id": [0] * len(frames),
                "group": ["rec"] * len(frames),
                "sequence": ["s2"] * len(frames),
                "X": [1.0] * len(frames),
                "Y": [2.0] * len(frames),
            }
        )
        _ = write_parquet_atomic(table, out)
        write_tracks_row(
            recording,
            run_id=variant,
            group="rec",
            sequence="s2",
            out_path=out,
            producer="convert-numbered",
            std_format="mosaic_v1",
            n_rows=len(frames),
            producer_run_id="",
            records_media=False,
        )
        (cell,) = read_tracks_index(recording)["first_frame"].astype(str)
        assert cell == ""

        assert media_drift(recording, read_tracks_index(recording)) == ()


class TestAFeatureOfTheTable:
    """A feature of a republished table is computed again from the new numbering."""

    def test_the_feature_follows_the_table_to_its_new_first_frame(
        self, recording: Dataset, sleap: FakeSleap, sleap_model: Path
    ) -> None:
        from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel

        params = SleapParams(model_paths=[str(sleap_model)])
        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))
        (row,) = _sleap_rows(recording)
        variant = str(row["run_id"])
        first = recording.run_feature(
            SpeedAngvel(), scope=Scope(entries=[_SECOND]), tracks_run_id=variant
        )
        assert _feature_frames(recording, first.run_id)[0] == _FIRST_FRAMES

        _grow_the_first_member(recording)
        _ = sleap_runs.run_sleap(recording, params, scope_over(_SECOND))
        second = recording.run_feature(
            SpeedAngvel(), scope=Scope(entries=[_SECOND]), tracks_run_id=variant
        )

        assert second.run_id == first.run_id
        assert second.cache_hit is False
        assert _feature_frames(recording, second.run_id)[0] == _FIRST_FRAMES + 2


def _feature_frames(ds: Dataset, run_id: str) -> list[int]:
    root = feature_run_root(ds, "speed-angvel__from__tracks", run_id)
    (path,) = sorted(root.glob("*s2*.parquet"))
    return sorted({int(frame) for frame in pd.read_parquet(path)["frame"]})
