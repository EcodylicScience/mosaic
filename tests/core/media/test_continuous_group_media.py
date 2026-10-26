"""A sequence of a continuous group resolves to its own media, placed on its recording.

The tracks of a continuous group count ``frame`` across the whole recording, and
each sequence's media contains only that sequence's frames. A resolution therefore
returns the sequence's files and the sequence's ``first_frame``: the frames
of the members before it, in member order.

The fixture records ``rec`` as two sequences: ``s1`` contains frames 0 to 4 in one
clip, and ``s2`` frames 5 to 8 in two clips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mosaic_media import MediaProbeError, probe_media

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline import entry_keys
from mosaic.core.pipeline.joined_export import (
    JoinedExportParams,
    joined_export_path,
    joined_recipe_hash,
    joined_source_uid,
)
from mosaic.core.pipeline.ops import run_op
from mosaic.core.recording import FrameOutsideSequenceError
from mosaic.core.scope import Scope
from tests.helpers import (
    add_tracks_table,
    make_dataset,
    track_table,
    write_painted_recording,
)

pytestmark = pytest.mark.media

_FIRST_FRAMES = 5
_SECOND_FRAMES = 4


def _write_members(dataset: Dataset, *, second_fps: float = 30.0) -> None:
    """Write and index the two sequences of ``rec``."""
    _ = write_painted_recording(
        dataset,
        "rec",
        [
            ("s1", [(_FIRST_FRAMES, 30.0)]),
            ("s2", [(2, second_fps), (2, second_fps)]),
        ],
    )


@pytest.fixture
def recording(tmp_path: Path, requires_ffmpeg: None) -> Dataset:
    """A dataset whose continuous group ``rec`` has two sequences with media."""
    dataset = make_dataset(tmp_path / "ds")
    _write_members(dataset)
    dataset.set_continuous_groups(["rec"])
    return dataset


def test_each_sequence_resolves_to_its_own_media(recording: Dataset) -> None:
    first = recording.resolve_media("rec", "s1")
    second = recording.resolve_media("rec", "s2")

    assert [path.parent.name for path in first.paths] == ["s1"]
    assert [path.name for path in second.paths] == ["clip0.mp4", "clip1.mp4"]
    assert {path.parent.name for path in second.paths} == {"s2"}
    assert first.first_frame == 0
    assert second.first_frame == _FIRST_FRAMES
    assert second.frame_count == _SECOND_FRAMES


def test_a_scoped_resolution_places_each_sequence_alike(recording: Dataset) -> None:
    scoped = {
        (entry.group, entry.sequence): entry.resolved
        for entry in recording.resolve_media_scope(None)
    }

    assert {entry: media.first_frame for entry, media in scoped.items()} == {
        ("rec", "s1"): 0,
        ("rec", "s2"): _FIRST_FRAMES,
    }
    assert [path.name for path in scoped["rec", "s2"].paths] == [
        "clip0.mp4",
        "clip1.mp4",
    ]


def test_a_group_that_is_not_declared_starts_each_sequence_at_frame_0(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _write_members(dataset)

    assert dataset.resolve_media("rec", "s2").first_frame == 0


def test_the_originals_rows_are_the_sequence_s_own(recording: Dataset) -> None:
    """The transcode, store export and tool-input lookups read these rows."""
    rows = recording.match_media_rows("rec", "s2")

    assert set(rows["sequence"]) == {"s2"}
    assert list(rows["name"]) == ["clip0.mp4", "clip1.mp4"]


def test_a_resolution_finds_the_members_in_the_rows_it_read(
    recording: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Placing ``s2`` reads neither the media index nor the tracks index again."""
    add_tracks_table(recording, "convert.a", track_table("rec", "s1"))
    monkeypatch.setattr(Dataset, "media_entries", _unexpected_read)
    monkeypatch.setattr(entry_keys, "index_entries", _unexpected_read)

    assert recording.resolve_media("rec", "s2").first_frame == _FIRST_FRAMES


def _unexpected_read(*_: object) -> set[Entry]:
    raise AssertionError("an index was read again")


def test_a_sequence_after_a_member_without_media_is_not_placed(
    recording: Dataset,
) -> None:
    """The member ``s0`` has tracks and no media, and its frame count is unknown."""
    add_tracks_table(recording, "convert.a", track_table("rec", "s0"))

    resolved = recording.resolve_media("rec", "s1")
    assert [path.parent.name for path in resolved.paths] == ["s1"]
    with pytest.raises(MediaProbeError, match="'s0'"):
        _ = resolved.first_frame
    errors: dict[Entry, MediaProbeError] = {}
    scoped = recording.resolve_media_scope(None, errors=errors)
    assert errors == {}
    assert {(entry.group, entry.sequence) for entry in scoped} == {
        ("rec", "s1"),
        ("rec", "s2"),
    }
    for entry in scoped:
        with pytest.raises(MediaProbeError, match="'s0'"):
            _ = entry.resolved.first_frame


def test_a_sequence_that_is_not_placed_is_still_joined(recording: Dataset) -> None:
    """A join reads the sequence's clips and never a tracks frame."""
    add_tracks_table(recording, "convert.a", track_table("rec", "s0"))

    _ = run_op(recording, "export-joined", {}, scope=Scope(entries=[("rec", "s2")]))

    resolved = recording.resolve_media("rec", "s2")
    joined = joined_export_path(
        recording,
        joined_source_uid(list(resolved.facts)),
        joined_recipe_hash(JoinedExportParams()),
    )
    assert int(probe_media(joined).frame_count) == _SECOND_FRAMES


def test_a_sequence_without_media_names_its_divided_recording(
    recording: Dataset,
) -> None:
    add_tracks_table(recording, "convert.a", track_table("rec", "s3"))

    with pytest.raises(FileNotFoundError, match="divided recording 'rec'"):
        _ = recording.resolve_media("rec", "s3")


def test_a_sequence_after_a_member_at_another_rate_is_not_placed(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    """A frame's time on the recording axis needs one rate."""
    dataset = make_dataset(tmp_path / "ds")
    _write_members(dataset, second_fps=25.0)
    dataset.set_continuous_groups(["rec"])

    assert dataset.resolve_media("rec", "s1").first_frame == 0
    second = dataset.resolve_media("rec", "s2")
    with pytest.raises(MediaProbeError, match="the sequences before it"):
        _ = second.first_frame


def test_the_first_sequence_is_placed_whatever_the_rates_of_its_clips(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = write_painted_recording(dataset, "rec", [("s1", [(2, 30.0), (3, 25.0)])])
    dataset.set_continuous_groups(["rec"])

    assert dataset.resolve_media("rec", "s1").first_frame == 0


def test_a_later_sequence_whose_clips_differ_in_rate_is_not_placed(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _ = write_painted_recording(
        dataset, "rec", [("s1", [(5, 30.0)]), ("s2", [(2, 30.0), (2, 25.0)])]
    )
    dataset.set_continuous_groups(["rec"])

    with pytest.raises(MediaProbeError, match=r"the clips of \('rec', 's2'\) differ"):
        _ = dataset.resolve_media("rec", "s2").first_frame


def test_a_rate_measured_within_the_store_tolerance_is_one_rate(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _write_members(dataset, second_fps=30.003)
    dataset.set_continuous_groups(["rec"])

    assert dataset.resolve_media("rec", "s2").first_frame == _FIRST_FRAMES


@pytest.mark.parametrize(("frame", "local"), [(5, 0), (8, 3)])
def test_a_recording_frame_is_a_frame_of_the_sequence(
    recording: Dataset, frame: int, local: int
) -> None:
    assert recording.resolve_media("rec", "s2").sequence_frame(frame) == local


@pytest.mark.parametrize("frame", [4, 9])
def test_a_recording_frame_outside_the_sequence_is_refused(
    recording: Dataset, frame: int
) -> None:
    with pytest.raises(FrameOutsideSequenceError, match="5 to 8"):
        _ = recording.resolve_media("rec", "s2").sequence_frame(frame)


def test_a_tracks_variant_that_orders_the_sequences_apart_leaves_them_unplaced(
    recording: Dataset,
) -> None:
    """The variant numbers ``s2`` before ``s1``, and the names sort ``s1`` first."""
    add_tracks_table(
        recording, "convert.a", track_table("rec", "s1", n_rows=5, start_frame=4)
    )
    add_tracks_table(recording, "convert.a", track_table("rec", "s2", n_rows=4))

    for sequence in ("s1", "s2"):
        resolved = recording.resolve_media("rec", sequence)
        with pytest.raises(MediaProbeError, match="Rename the sequences"):
            _ = resolved.first_frame


def test_a_tracks_variant_that_numbers_each_sequence_from_0_places_them(
    recording: Dataset,
) -> None:
    """Two sequences that both start at frame 0 give the variant no order."""
    add_tracks_table(recording, "convert.a", track_table("rec", "s1", n_rows=5))
    add_tracks_table(recording, "convert.a", track_table("rec", "s2", n_rows=4))

    assert recording.resolve_media("rec", "s2").first_frame == _FIRST_FRAMES


def test_a_frame_past_the_end_of_an_undivided_recording_is_its_own(
    tmp_path: Path, requires_ffmpeg: None
) -> None:
    """A reader stops at the end of the file, and no later sequence follows."""
    dataset = make_dataset(tmp_path / "ds")
    _write_members(dataset)

    resolved = dataset.resolve_media("rec", "s2")

    assert resolved.sequence_frame(_SECOND_FRAMES + 3) == _SECOND_FRAMES + 3
