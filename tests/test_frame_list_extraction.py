"""``extract-frames`` with ``method="list"``: the caller names the frames.

Targeted annotation picks frames where a tracker fails, which neither sampling
method would land on. A list names, per entry, the frame indices to write, on
the entry's media axis -- counted across its clips in order, as its tracks'
``frame`` column is. The list chooses the entries and the scope narrows it.

A recording whose clips differ in frame rate is read through its
``export-joined`` file, under every method: ``MultiVideoReader`` refuses such
clips and stays strict.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.joined_export import JoinedExportMissingError
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic.core.pipeline.ops import run_op
from mosaic.core.scope import Scope
from mosaic.tracking import extract_frames
from mosaic.tracking.frame_extraction import list_frame_runs
from mosaic.tracking.frame_extraction.dataset_runs import (
    ExtractFramesParams,
    ListedFramesRefused,
    frames_identity_payload,
    frames_run_id,
)
from tests.helpers import add_media_sequence, write_h264_mp4

runner = CliRunner()

LIST_A = {"method": "list", "frames": [{"sequence": "a", "indices": [1]}]}


# --- the params ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"method": "list"}, "frames"),
        ({"method": "list", "frames": []}, "frames"),
        ({**LIST_A, "n_frames": 2}, "n_frames"),
        ({**LIST_A, "start_frame": 3}, "start_frame"),
        ({**LIST_A, "end_frame": 3}, "end_frame"),
        ({"n_frames": 2, "frames": LIST_A["frames"]}, "frames"),
        ({"method": "kmeans", "n_frames": 2, "frames": LIST_A["frames"]}, "frames"),
        ({}, "n_frames"),
        ({"method": "kmeans"}, "n_frames"),
    ],
)
def test_a_field_that_disagrees_with_the_method_is_refused_by_name(
    values: dict[str, object], field: str
) -> None:
    """Each refusal names its own field, so a form can place it on a control."""
    with pytest.raises(ValidationError) as excinfo:
        _ = ExtractFramesParams.model_validate(values)
    ((location, kind),) = [(e["loc"], e["type"]) for e in excinfo.value.errors()]
    assert location == (field,)
    assert kind == "value_error"


def test_an_unknown_method_is_refused_alone() -> None:
    """No second refusal is raised on the fields that read the method."""
    with pytest.raises(ValidationError) as excinfo:
        _ = ExtractFramesParams.model_validate({"method": "banana"})
    assert [e["loc"] for e in excinfo.value.errors()] == [("method",)]


def test_an_entry_listed_twice_is_refused() -> None:
    twice = [
        {"sequence": "a", "indices": [1]},
        {"sequence": "a", "indices": [2]},
    ]
    with pytest.raises(ValidationError, match=r"names \(, a\) twice"):
        _ = ExtractFramesParams.model_validate({"method": "list", "frames": twice})


def test_a_negative_index_is_refused_where_it_sits() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _ = ExtractFramesParams.model_validate(
            {"method": "list", "frames": [{"sequence": "a", "indices": [3, -1]}]}
        )
    assert ("frames", 0, "indices", 1) in [e["loc"] for e in excinfo.value.errors()]


def test_a_consumer_can_remove_a_checked_field_from_a_subclass() -> None:
    """A field a caller owns is removed from its exposed model and supplied later.

    mosaic-api removes the fields it owns from a subclass and rebuilds it. A
    validator naming a field the subclass no longer has would refuse that
    rebuild, so the one field it checks being absent must leave the rest working.
    """

    class Exposed(ExtractFramesParams):
        pass

    for name in ("frames", "start_frame"):
        _ = Exposed.model_fields.pop(name)
    _ = Exposed.model_rebuild(force=True)

    assert Exposed.model_validate({"method": "list"}).method == "list"
    with pytest.raises(ValidationError, match="n_frames is not read"):
        _ = Exposed.model_validate({"method": "list", "n_frames": 2})


# --- the identifier ------------------------------------------------------------


def _list_id(frames: list[dict[str, object]], **extra: object) -> str:
    params = ExtractFramesParams.model_validate(
        {"method": "list", "frames": frames, **extra}
    )
    return frames_run_id(params.method, params)


class TestTheListIdentifier:
    def test_order_and_repeats_name_the_same_run(self) -> None:
        tidy: list[dict[str, object]] = [
            {"sequence": "a", "indices": [1, 5]},
            {"sequence": "b", "indices": [2]},
        ]
        messy: list[dict[str, object]] = [
            {"sequence": "b", "indices": [2, 2]},
            {"sequence": "a", "indices": [5, 1, 5]},
        ]
        assert _list_id(tidy) == _list_id(messy)

    def test_another_frame_or_entry_names_another_run(self) -> None:
        base = _list_id([{"sequence": "a", "indices": [1]}])
        assert _list_id([{"sequence": "a", "indices": [2]}]) != base
        assert _list_id([{"sequence": "b", "indices": [1]}]) != base
        assert _list_id([{"group": "g", "sequence": "a", "indices": [1]}]) != base

    def test_it_is_prefixed_by_the_method(self) -> None:
        assert _list_id([{"sequence": "a", "indices": [1]}]).startswith("list-")

    def test_revision_still_enters(self) -> None:
        frames: list[dict[str, object]] = [{"sequence": "a", "indices": [1]}]
        assert _list_id(frames, revision=1) != _list_id(frames)

    def test_a_sampling_method_hashes_no_frames_term(self) -> None:
        """The payload a uniform run always hashed, key for key."""
        params = ExtractFramesParams(n_frames=100)
        assert frames_identity_payload(params) == params.identity_dump()


# --- a run -----------------------------------------------------------------------

CLIP = 6
"""Frames in each of ``add_media_sequence``'s two clips."""


@pytest.fixture
def two_sequences(scenario_dataset_with_media: Dataset) -> Dataset:
    """``seq_a`` in two clips of six frames, and ``seq_c`` in one."""
    add_media_sequence(scenario_dataset_with_media, "seq_c", videos=("c.mp4",))
    return scenario_dataset_with_media


def _run_root(ds: Dataset, run_id: str) -> Path:
    return ds.get_root("frames") / "list" / run_id


def _level(png: Path) -> float:
    image = cv2.imread(str(png))
    assert image is not None, png
    return float(np.mean(image))


def test_exactly_the_listed_frames_are_written_across_the_clips(
    two_sequences: Dataset,
) -> None:
    """Frame 6 is the second clip's first: the index counts across the clips."""
    ds = two_sequences
    run_id = extract_frames(
        ds,
        method="list",
        frames={("", "seq_a"): [11, 0, 6, 5, 6]},
    )

    seq_dir = _run_root(ds, run_id) / "seq_a"
    names = sorted(p.name for p in seq_dir.glob("*.png"))
    assert names == [f"frame_{i:06d}.png" for i in (0, 5, 6, 11)]

    first, last = (
        _level(seq_dir / "frame_000005.png"),
        _level(seq_dir / "frame_000006.png"),
    )
    assert abs(first - last) > 1.0, "frames 5 and 6 come from different clips"
    assert _level(seq_dir / "frame_000000.png") == pytest.approx(first, abs=1.0)
    assert _level(seq_dir / "frame_000011.png") == pytest.approx(last, abs=1.0)

    manifest = json.loads((seq_dir / "run_info.json").read_text())
    assert manifest["selected_frame_indices"] == [0, 5, 6, 11]
    assert manifest["n_requested"] == 4

    rows = list_frame_runs(ds, method="list")
    assert list(rows["sequence"]) == ["seq_a"]
    assert int(rows.iloc[0]["n_frames_requested"]) == 4

    params = json.loads((_run_root(ds, run_id) / "run_params.json").read_text())
    assert params["_frames"] == [["", "seq_a", [0, 5, 6, 11]]]


def test_an_unset_scope_extracts_every_listed_entry_and_no_other(
    two_sequences: Dataset,
) -> None:
    run_id = extract_frames(two_sequences, method="list", frames={("", "seq_c"): [1]})
    written = sorted(p.name for p in _run_root(two_sequences, run_id).iterdir())
    assert "seq_c" in written
    assert "seq_a" not in written


def test_a_scope_narrows_the_list(two_sequences: Dataset) -> None:
    listed = {("", "seq_a"): [1], ("", "seq_c"): [1]}
    run_id = extract_frames(
        two_sequences,
        method="list",
        frames=listed,
        scope=Scope(entries=[("", "seq_a")]),
    )
    root = _run_root(two_sequences, run_id)
    assert (root / "seq_a").is_dir()
    assert not (root / "seq_c").exists()


def test_a_scoped_entry_the_list_does_not_name_is_skipped(
    two_sequences: Dataset,
) -> None:
    run_id = extract_frames(
        two_sequences,
        method="list",
        frames={("", "seq_a"): [1]},
        scope=Scope(entries=[("", "seq_a"), ("", "seq_c")]),
    )
    root = _run_root(two_sequences, run_id)
    assert (root / "seq_a").is_dir()
    assert not (root / "seq_c").exists()


def test_a_scope_sharing_no_entry_with_the_list_is_refused(
    two_sequences: Dataset,
) -> None:
    with pytest.raises(ListedFramesRefused, match="lists none of them"):
        _ = extract_frames(
            two_sequences,
            method="list",
            frames={("", "seq_a"): [1]},
            scope=Scope(entries=[("", "seq_c")]),
        )


def test_a_listed_entry_without_media_is_refused(two_sequences: Dataset) -> None:
    """``seq_b`` has tracks and no media: the list names it, so the run refuses."""
    with pytest.raises(ListedFramesRefused, match=r"\(, seq_b\)"):
        _ = extract_frames(
            two_sequences,
            method="list",
            frames={("", "seq_a"): [1], ("", "seq_b"): [1]},
        )
    assert not (two_sequences.get_root("frames") / "list").exists()


def test_a_frame_past_the_end_is_refused_before_anything_is_written(
    two_sequences: Dataset,
) -> None:
    """Refused up front, where a worker's failure would only be printed."""
    with pytest.raises(ListedFramesRefused) as excinfo:
        _ = extract_frames(
            two_sequences,
            method="list",
            frames={("", "seq_a"): [0, 2 * CLIP, 40]},
        )
    message = str(excinfo.value)
    assert "seq_a has 12 frames" in message
    assert "12, 40" in message
    assert not (two_sequences.get_root("frames") / "list").exists()


def test_the_command_line_reads_a_list_from_a_file(
    two_sequences: Dataset, tmp_path: Path
) -> None:
    """``--params @file`` is how a list too long for an argument is passed."""
    params = tmp_path / "frames.json"
    _ = params.write_text(
        json.dumps(
            {"method": "list", "frames": [{"sequence": "seq_c", "indices": [0, 3]}]}
        )
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(two_sequences.manifest_path),
            "--kind",
            "extract-frames",
            "--params",
            f"@{params}",
        ],
    )

    assert result.exit_code == 0, result.output
    pngs = sorted(p.name for p in two_sequences.get_root("frames").rglob("*.png"))
    assert pngs == ["frame_000000.png", "frame_000003.png"]


# --- a recording whose clips differ in frame rate ---------------------------------

RATE_CLIP = 300
"""Long enough that 30 beside 31 fps drifts past the reader's half-frame allowance."""

_LEVEL_STEP = 11
_LEVEL_PERIOD = 20


def _level_of(index: int) -> int:
    """The grey level global frame *index* is written with."""
    return 16 + (index % _LEVEL_PERIOD) * _LEVEL_STEP


@pytest.fixture
def mixed_rate(tmp_path: Path, requires_ffmpeg: None) -> Dataset:
    """One sequence, ``sess``: 300 frames at 30 fps, then 300 at 31 fps.

    Every frame is written at the grey level of its global index, so a frame read
    back names the frame it is.
    """
    from tests.helpers import make_dataset

    ds = make_dataset(tmp_path, roots=["media_raw", "media", "tracks", "frames"])
    directory = ds.get_root("media_raw") / "sess"
    for position, fps in enumerate((30.0, 31.0)):
        first = position * RATE_CLIP
        write_h264_mp4(
            directory / f"c{position}.mp4",
            fps=fps,
            levels=[_level_of(first + i) for i in range(RATE_CLIP)],
        )
    _ = ds.write_media_index(
        [
            MediaIndexScope(
                directory=directory,
                group="",
                sequence="sess",
                order_by_name={"c0.mp4": 0, "c1.mp4": 1},
            )
        ],
        extensions=(".mp4",),
    )
    return ds


@pytest.mark.media
@pytest.mark.slow
class TestAMixedRateRecording:
    def test_it_is_refused_until_it_is_joined(self, mixed_rate: Dataset) -> None:
        with pytest.raises(JoinedExportMissingError) as excinfo:
            _ = extract_frames(mixed_rate, n_frames=4)
        message = str(excinfo.value)
        assert "[extract-frames] (, sess)" in message
        assert "--kind export-joined" in message
        assert not (mixed_rate.get_root("frames") / "uniform").exists()

    def test_it_is_read_through_its_join(self, mixed_rate: Dataset) -> None:
        _ = run_op(mixed_rate, "export-joined", {}, scope=Scope(entries=[("", "sess")]))

        listed = [0, RATE_CLIP - 1, RATE_CLIP, RATE_CLIP + 1, 2 * RATE_CLIP - 1]
        run_id = extract_frames(
            mixed_rate, method="list", frames={("", "sess"): listed}
        )

        seq_dir = _run_root(mixed_rate, run_id) / "sess"
        for index in listed:
            level = _level(seq_dir / f"frame_{index:06d}.png")
            assert level == pytest.approx(_level_of(index), abs=4.0), index

        rows = list_frame_runs(mixed_rate, method="list")
        assert ".joined.mp4" in str(rows.iloc[0]["video_abs_path"])

    def test_a_sampling_method_reads_the_join_as_well(
        self, mixed_rate: Dataset
    ) -> None:
        _ = run_op(mixed_rate, "export-joined", {}, scope=Scope(entries=[("", "sess")]))

        run_id = extract_frames(mixed_rate, n_frames=4)

        pngs = list((mixed_rate.get_root("frames") / "uniform" / run_id).rglob("*.png"))
        assert len(pngs) == 4
