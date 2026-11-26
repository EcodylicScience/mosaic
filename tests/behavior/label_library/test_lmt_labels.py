"""The ``lmt_sqlite`` ``id_tags`` converter writes a database's animals as individual tags.

Each test writes a Live Mouse Tracker database, indexes it as a labels source and
converts it through a dataset. The files load without unpickling, and
``id-tag-columns`` attaches their text to the tracks of the same entry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.id_tag_columns import IdTagColumns
from mosaic.behavior.feature_library.label_statistics import (
    LabelStatistics,
    state_suffix,
)
from mosaic.behavior.label_library.lmt_sqlite import (
    LmtAnimalsConverter,
    LmtEventsConverter,
    LmtEventsParams,
    comma_free,
)
from mosaic.core.dataset import Dataset
from mosaic.core.label_converter import LABEL_CONVERTERS, LabelConvertParams
from mosaic.core.label_intervals import SEGMENTS_FORMAT, segments_from_npz
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.track_library.lmt import LMT_EVENT_NAMES
from tests.helpers import (
    DEFAULT_ANIMAL_COLUMNS,
    NINE_ANIMAL_COLUMNS,
    LmtAnimal,
    LmtEvent,
    LmtRecording,
    alter_lmt_sqlite,
    default_lmt_recording,
    file_digests,
    make_dataset,
    result_outputs,
    write_lmt_sqlite,
)

_RFIDS = ["000004395862", "000004395870", "000004395881"]


def _lmt_labels_dataset(
    base: Path,
    recording: LmtRecording | None = None,
    *,
    animal_columns: Sequence[str] = DEFAULT_ANIMAL_COLUMNS,
    tracks: bool = False,
    recordings: Mapping[str, LmtRecording] | None = None,
    alter: str = "",
) -> Dataset:
    """A dataset whose labels source, and with *tracks* its tracks source, is ``raw/``.

    *recordings* names each database file's stem and rows, and defaults to the one
    file ``recording`` that holds *recording*. *alter* is SQL run on each file
    before it is indexed.
    """
    ds = make_dataset(base)
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if recordings is None:
        recordings = {"recording": recording or default_lmt_recording()}
    for stem, rows in recordings.items():
        path = write_lmt_sqlite(
            raw / f"{stem}.sqlite",
            animals=rows.animals,
            frames=rows.frames,
            events=rows.events,
            detections=rows.detections,
            log=rows.log,
            animal_columns=animal_columns,
        )
        if alter:
            alter_lmt_sqlite(path, alter)
    _ = ds.index_labels_raw([raw], patterns=["*.sqlite"], src_format="lmt_sqlite")
    if tracks:
        _ = ds.index_tracks_raw([raw], patterns=["*.sqlite"], src_format="lmt_sqlite")
        assert ds.convert_all_tracks().failed == 0
    return ds


def _convert(ds: Dataset) -> Path:
    outcome = ds.convert_all_labels(kind="id_tags", source_format="lmt_sqlite")
    assert outcome.written == 1
    index = read_labels_index(ds, "id_tags")
    (path,) = index["abs_path"].tolist()
    return ds.resolve_path(str(path))


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_the_converter_declares_its_contract() -> None:
    converter = LABEL_CONVERTERS[("lmt_sqlite", "id_tags")]

    assert converter is LmtAnimalsConverter
    assert converter.label_format == "id_tags_v1"
    assert converter.version == "0.1"
    assert converter.Params is LabelConvertParams
    assert converter.refuses_shared_entry_keys


def test_the_animals_are_written_as_text_arrays_keyed_by_id(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")

    arrays = _arrays(_convert(ds))

    assert sorted(arrays) == ["genotype", "ids", "name", "rfid"]
    assert arrays["ids"].dtype == np.int64
    assert arrays["ids"].tolist() == [1, 2, 3]
    assert arrays["rfid"].dtype.kind == "U"
    assert arrays["rfid"].tolist() == _RFIDS
    assert arrays["name"].tolist() == ["mouse one", "mouse two", "mouse three"]


def test_a_null_cell_is_empty_text(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")

    genotype = _arrays(_convert(ds))["genotype"]

    assert genotype.dtype.kind == "U"
    assert genotype.tolist() == ["WT", "KO", ""]


def test_the_ids_ascend_whatever_the_row_order(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    reordered = replace(recording, animals=list(reversed(recording.animals)))
    ds = _lmt_labels_dataset(tmp_path / "ds", reordered)

    arrays = _arrays(_convert(ds))

    assert arrays["ids"].tolist() == [1, 2, 3]
    assert arrays["rfid"].tolist() == _RFIDS


def test_the_nine_column_table_writes_nine_arrays(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    extras = {
        "AGE": "12",
        "SEX": "male",
        "STRAIN": "C57",
        "SETUP": "setup 2",
        "TREATMENT": None,
    }
    animals: list[LmtAnimal] = [
        replace(animal, extra=extras) for animal in recording.animals
    ]
    ds = _lmt_labels_dataset(
        tmp_path / "ds",
        replace(recording, animals=animals),
        animal_columns=NINE_ANIMAL_COLUMNS,
    )

    arrays = _arrays(_convert(ds))

    assert sorted(arrays) == [
        "age",
        "genotype",
        "ids",
        "name",
        "rfid",
        "setup",
        "sex",
        "strain",
        "treatment",
    ]
    assert arrays["age"].tolist() == ["12", "12", "12"]
    assert arrays["setup"].tolist() == ["setup 2"] * 3
    assert arrays["treatment"].tolist() == ["", "", ""]
    assert arrays["rfid"].tolist() == _RFIDS


def test_the_index_row_records_the_format_and_the_vocabulary(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    _ = _convert(ds)

    (row,) = (row for _, row in read_labels_index(ds, "id_tags").iterrows())

    assert row["label_format"] == "id_tags_v1"
    assert row["label_kind"] == "id_tags"
    assert (row["group"], row["sequence"]) == ("", "recording")
    assert row["n_frames"] == "3"
    assert row["label_ids"] == "1,2,3"
    assert row["label_names"] == "genotype,name,rfid"


def test_id_tag_columns_attaches_the_rfid_as_text_with_its_zeros(
    tmp_path: Path,
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", tracks=True)
    _ = _convert(ds)

    result = run_feature(ds, IdTagColumns())

    (table,) = result_outputs(ds, result).values()
    by_id = table.groupby("id")[["rfid", "genotype"]].first()
    assert by_id["rfid"].tolist() == _RFIDS
    assert all(type(value) is str for value in by_id["rfid"])
    assert by_id["genotype"].tolist() == ["WT", "KO", ""]


def test_conversion_leaves_the_database_unchanged(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    path = raw / "recording.sqlite"
    digests = file_digests(raw)
    mtime_ns = path.stat().st_mtime_ns

    _ = _convert(ds)

    assert file_digests(raw) == digests
    assert path.stat().st_mtime_ns == mtime_ns
    assert sorted(item.name for item in raw.iterdir()) == ["recording.sqlite"]


@pytest.mark.parametrize("suffix", ["-journal", "-wal"])
def test_a_journal_sibling_refuses_the_conversion_and_writes_nothing(
    tmp_path: Path, suffix: str
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    sibling = ds.base_dir / "raw" / f"recording.sqlite{suffix}"
    _ = sibling.write_bytes(b"")

    with pytest.raises(ValueError, match=f"recording.sqlite{suffix}"):
        _ = ds.convert_all_labels(kind="id_tags", source_format="lmt_sqlite")

    assert read_labels_index(ds, "id_tags").empty


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("DROP TABLE ANIMAL", "no ANIMAL table"),
        ("DROP TABLE ANIMAL; CREATE TABLE ANIMAL (RFID TEXT)", "no ID column"),
    ],
)
def test_an_animal_table_that_cannot_be_read_refuses_the_conversion_and_writes_nothing(
    tmp_path: Path, script: str, message: str
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    path = ds.base_dir / "raw" / "recording.sqlite"
    alter_lmt_sqlite(path, script)

    with pytest.raises(ValueError, match=message) as raised:
        _ = ds.convert_all_labels(kind="id_tags", source_format="lmt_sqlite")

    assert str(path) in str(raised.value)
    assert read_labels_index(ds, "id_tags").empty


_SPEC_EVENT_NAMES = (
    "Approach",
    "Approach contact",
    "Approach rear",
    "Behind",
    "Break contact",
    "Center Zone",
    "Contact",
    "Detection",
    "Escape",
    "Follow",
    "FollowZone",
    "FollowZone Isolated",
    "Get away",
    "Group 3 break",
    "Group 3 make",
    "Group 4 break",
    "Group 4 make",
    "Group2",
    "Group3",
    "Group4",
    "Head detected",
    "Look down",
    "MACHINE LEARNING ASSOCIATION",
    "Move in contact",
    "Move isolated",
    "Nest3_",
    "Nest4_",
    "Oral-genital Contact",
    "Oral-oral Contact",
    "Periphery Zone",
    "RFID ASSIGN ANONYMOUS TRACK",
    "RFID MATCH",
    "RFID MISMATCH",
    "Rear at periphery",
    "Rear in centerWindow",
    "Rear in contact",
    "Rear isolated",
    "Rearing",
    "SAP",
    "Side by side Contact",
    "Side by side Contact, opposite way",
    "Social approach",
    "Social escape",
    "Stop",
    "Stop in contact",
    "Stop isolated",
    "Train2",
    "WallJump",
    "Water Stop",
    "Water Zone",
    "badIdentity",
    "badOrientation",
    "badSegmentation",
    "manualContact",
    "manualOralGenital",
    "manualOralOralContact",
    "manualSideSideOpposite",
    "manualSideSideSame",
    "seq oral geni - oral oral",
    "seq oral oral - oral genital",
    "Flickering",
    "night",
    "Undetected",
)


def test_the_vocabulary_is_the_specified_table_in_order() -> None:
    assert LMT_EVENT_NAMES == _SPEC_EVENT_NAMES
    assert len(LMT_EVENT_NAMES) == 63
    assert LMT_EVENT_NAMES.index("Detection") == 7


def test_the_first_sixty_names_are_in_sqlite_binary_order() -> None:
    first = list(LMT_EVENT_NAMES[:60])
    assert first == sorted(first, key=lambda name: name.encode("utf-8"))


def test_comma_free_replaces_each_comma_with_a_space() -> None:
    assert comma_free("Side by side Contact, opposite way") == (
        "Side by side Contact opposite way"
    )
    assert comma_free("a,b") == "a b"
    assert comma_free("Contact") == "Contact"


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"extra_event_names": ["Contact"]}, ["Contact", "extra_event_names"]),
        (
            {"extra_event_names": ["Side by side Contact opposite way"]},
            [
                "Side by side Contact opposite way",
                "Side by side Contact, opposite way",
            ],
        ),
        (
            {"extra_event_names": ["x"], "skip_event_names": ["x"]},
            ["x", "extra_event_names", "skip_event_names"],
        ),
        ({"quality_events": ["Nope"]}, ["Nope", "quality_events"]),
    ],
)
def test_a_vocabulary_that_cannot_hold_is_refused_naming_the_names(
    overrides: dict[str, list[str]], named: list[str]
) -> None:
    with pytest.raises(ValidationError) as raised:
        _ = LmtEventsParams.model_validate(overrides)
    for text in named:
        assert text in str(raised.value)


def test_two_extra_names_with_one_comma_free_spelling_are_refused() -> None:
    with pytest.raises(ValidationError) as raised:
        _ = LmtEventsParams(extra_event_names=["Look, up", "Look up"])
    assert "Look, up" in str(raised.value)
    assert "Look up" in str(raised.value)


def test_a_quality_event_is_written_as_stored_with_its_comma() -> None:
    params = LmtEventsParams(quality_events=["Side by side Contact, opposite way"])
    assert params.quality_events == ["Side by side Contact, opposite way"]


def test_extra_names_follow_the_constant_in_the_order_given() -> None:
    params = LmtEventsParams(extra_event_names=["Grooming", "Digging"])
    vocabulary = params.vocabulary()
    assert vocabulary[:63] == LMT_EVENT_NAMES
    assert vocabulary[63:] == ("Grooming", "Digging")
    assert LmtEventsParams().vocabulary() == LMT_EVENT_NAMES


def test_a_skipped_name_may_also_be_a_quality_event() -> None:
    params = LmtEventsParams(
        skip_event_names=["Detection"], quality_events=["Detection"]
    )
    assert params.skip_event_names == ["Detection"]
    assert params.quality_events == ["Detection"]


def test_a_quality_event_may_be_an_extra_name() -> None:
    params = LmtEventsParams(
        extra_event_names=["Grooming"], quality_events=["Grooming"]
    )
    assert params.quality_events == ["Grooming"]


def test_the_defaults_are_the_specified_ones() -> None:
    params = LmtEventsParams()
    assert params.extra_event_names == []
    assert params.skip_event_names == []
    assert params.quality_events == ["Flickering", "Detection", "Undetected"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra_event_names": ["Grooming"]},
        {"skip_event_names": ["coucou"]},
        {"quality_events": ["Detection"]},
    ],
)
def test_each_vocabulary_param_changes_the_label_identity(
    overrides: dict[str, list[str]],
) -> None:
    assert (
        LmtEventsParams.model_validate(overrides).identity_dump()
        != LmtEventsParams().identity_dump()
    )


def test_group_from_does_not_change_the_label_identity() -> None:
    assert (
        LmtEventsParams(group_from="infile").identity_dump()
        == LmtEventsParams().identity_dump()
    )


# --- the lmt_events converter ------------------------------------------------------

_EVENT_KIND = "lmt_events"
_ORIGIN = 101
_COMMA_NAME = "Side by side Contact, opposite way"


def _events_recording(extra: Sequence[LmtEvent] = ()) -> LmtRecording:
    """The default recording with *extra* events after its nine."""
    recording = default_lmt_recording()
    return replace(recording, events=(*recording.events, *extra))


def _with_events(*events: LmtEvent) -> LmtRecording:
    return replace(default_lmt_recording(), events=events)


def _convert_events(
    ds: Dataset,
    params: dict[str, object] | None = None,
    *,
    tracks_run_id: str | None = None,
) -> str:
    """Convert the labels source to ``lmt_events`` and return the variant."""
    outcome = ds.convert_all_labels(
        kind=_EVENT_KIND,
        source_format="lmt_sqlite",
        params=params,
        tracks_run_id=tracks_run_id,
    )
    assert outcome.written >= 1
    return outcome.variant


def _events_file(
    ds: Dataset, variant: str | None = None, sequence: str = "recording"
) -> dict[str, np.ndarray]:
    index = read_labels_index(ds, _EVENT_KIND)
    rows = index[index["sequence"] == sequence]
    if variant is not None:
        rows = rows[rows["run_id"] == variant]
    (path,) = rows["abs_path"].tolist()
    return _arrays(ds.resolve_path(str(path)))


def _tracks_table(ds: Dataset, run_id: str | None = None) -> pd.DataFrame:
    index = read_tracks_index(ds)
    if run_id is not None:
        index = index[index["run_id"] == run_id]
    (path,) = index["abs_path"].tolist()
    return pd.read_parquet(ds.resolve_path(str(path)))


def _frame_seconds(recording: LmtRecording) -> dict[int, float]:
    """Each frame number's second, as the tracks of the file carry it."""
    stamps = np.array([frame.timestamp for frame in recording.frames], dtype=np.float64)
    seconds = (stamps - stamps[0]) / 1000.0
    return {
        frame.number: float(second)
        for frame, second in zip(recording.frames, seconds, strict=True)
    }


def test_the_events_converter_declares_its_contract() -> None:
    converter = LABEL_CONVERTERS[("lmt_sqlite", _EVENT_KIND)]

    assert converter is LmtEventsConverter
    assert converter.label_format == SEGMENTS_FORMAT == "segments_v1"
    assert converter.version == "0.1"
    assert converter.Params is LmtEventsParams
    assert converter.refuses_shared_entry_keys


def test_individual_pair_and_group_events_round_trip_through_the_segments(
    tmp_path: Path,
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", _events_recording(), tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)
    segments = segments_from_npz(arrays)

    assert segments is not None
    assert segments.individual is not None
    assert arrays["seg_event_id"].tolist() == list(range(1, 10))
    assert segments.individual.tolist() == [1, 1, 2, 1, -1, 3, 1, 2, 2]
    assert arrays["seg_partner"].tolist() == [-1, 2, 1, 2, -1, -1, 3, 3, -1]
    assert arrays["seg_member_c"].tolist() == [-1, -1, -1, 3, -1, -1, -1, -1, -1]
    assert arrays["seg_member_d"].tolist() == [-1] * 9
    names = [
        "Move isolated",
        "Contact",
        "Contact",
        "Group 3 make",
        "night",
        "Flickering",
        _COMMA_NAME,
        "Move in contact",
        "Stop isolated",
    ]
    assert segments.label.tolist() == [LMT_EVENT_NAMES.index(name) for name in names]
    assert arrays["seg_label"].dtype == np.int32
    for key in ("seg_individual", "seg_partner", "seg_member_c", "seg_member_d"):
        assert arrays[key].dtype == np.int64
    assert arrays["seg_event_id"].dtype == np.int64


def test_the_payload_holds_the_vocabulary_and_the_entry_without_pickles(
    tmp_path: Path,
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", _events_recording(), tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)

    assert str(arrays["label_format"]) == "segments_v1"
    assert str(arrays["group"]) == ""
    assert str(arrays["sequence"]) == "recording"
    assert int(arrays["background_id"]) == -1
    assert arrays["background_id"].dtype == np.int32
    assert arrays["label_ids"].dtype == np.int32
    assert arrays["label_ids"].tolist() == list(range(63))
    assert arrays["dropped_event_ids"].dtype == np.int64
    assert arrays["dropped_event_ids"].size == 0
    assert arrays["quality_label_ids"].dtype == np.int32
    assert "labels" not in arrays
    assert "frames" not in arrays


def test_native_segments_start_and_end_on_the_frames_of_the_tracks(
    tmp_path: Path,
) -> None:
    recording = _events_recording()
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)

    starts = [event.start - _ORIGIN for event in recording.events]
    ends = [event.end - _ORIGIN + 1 for event in recording.events]
    assert arrays["seg_start_frame"].tolist() == starts
    assert arrays["seg_end_frame"].tolist() == ends
    frames = set(_tracks_table(ds)["frame"].tolist())
    assert all(start in frames for start in starts)
    assert all(end - 1 in frames for end in ends)
    (row,) = read_labels_index(ds, _EVENT_KIND).to_dict("records")
    assert row["label_format"] == "segments_v1"
    assert row["n_frames"] == str(max(frames) + 1)


def test_an_event_starting_on_the_paused_frame_starts_on_the_next_tracks_frame(
    tmp_path: Path,
) -> None:
    recording = _events_recording([LmtEvent("Stop", 180, 200, a=1, id=10)])
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)

    frames = set(_tracks_table(ds)["frame"].tolist())
    assert 180 - _ORIGIN not in frames
    assert arrays["seg_event_id"][-1] == 10
    assert arrays["seg_start_frame"][-1] == 181 - _ORIGIN
    assert arrays["seg_end_frame"][-1] == 200 - _ORIGIN + 1
    assert arrays["dropped_event_ids"].size == 0


def _between_grid_points(recording: LmtRecording, fps: float) -> LmtEvent:
    """An event from one frame of the file to the next with no *fps* grid time between."""
    seconds = _frame_seconds(recording)
    for first, second in pairwise(sorted(seconds)):
        if first > 300 and math.ceil(seconds[first] * fps) > math.floor(
            seconds[second] * fps
        ):
            return LmtEvent("Stop", first, second, a=1, id=99)
    raise AssertionError("every pair of frames holds a grid time")


def test_a_resampled_variant_places_events_by_their_seconds_and_lists_the_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fps = 15.0
    base = _events_recording()
    between = _between_grid_points(base, fps)
    recording = _events_recording([between])
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)
    native = str(read_tracks_index(ds)["run_id"].iloc[0])
    _ = run_op(
        ds,
        "resample-tracks",
        {"target_fps": fps, "source_tracks_run_id": native},
        track=False,
    )
    (grid,) = set(read_tracks_index(ds)["run_id"]) - {native}
    on_native = _convert_events(ds, tracks_run_id=native)
    _ = capsys.readouterr()

    on_grid = _convert_events(ds, tracks_run_id=str(grid))

    table = _tracks_table(ds, str(grid)).drop_duplicates("frame").sort_values("frame")
    frames = table["frame"].to_numpy()
    times = table["time"].to_numpy()
    seconds = _frame_seconds(recording)
    kept: list[tuple[int, int, int]] = []
    dropped: list[int] = []
    for event in recording.events:
        inside = np.flatnonzero(
            (times >= seconds[event.start]) & (times <= seconds[event.end])
        )
        if inside.size == 0:
            dropped.append(int(event.id or 0))
        else:
            kept.append(
                (
                    int(event.id or 0),
                    int(frames[inside[0]]),
                    int(frames[inside[-1]]) + 1,
                )
            )
    arrays = _events_file(ds, on_grid)
    assert on_grid != on_native
    assert between.id in dropped
    assert arrays["seg_event_id"].tolist() == [event for event, _, _ in kept]
    assert arrays["seg_start_frame"].tolist() == [start for _, start, _ in kept]
    assert arrays["seg_end_frame"].tolist() == [end for _, _, end in kept]
    assert arrays["dropped_event_ids"].tolist() == dropped
    assert arrays["dropped_event_ids"].dtype == np.int64
    total = len(recording.events)
    line = (
        f"[lmt_events] recording: kept {len(kept)} of {total} events; 0 with a "
        f"skipped name and {len(dropped)} without a tracks frame left out"
    )
    assert line in capsys.readouterr().err
    (row,) = (
        read_labels_index(ds, _EVENT_KIND)
        .query("run_id == @on_grid")
        .to_dict("records")
    )
    assert row["n_frames"] == str(int(frames.max()) + 1)


def test_overlapping_events_keep_their_listing_order(tmp_path: Path) -> None:
    recording = _with_events(
        LmtEvent("Contact", 140, 160, a=1, b=2, id=5),
        LmtEvent("Stop", 150, 170, a=1, id=3),
    )
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)

    assert arrays["seg_event_id"].tolist() == [3, 5]
    assert arrays["seg_label"].tolist() == [
        LMT_EVENT_NAMES.index("Stop"),
        LMT_EVENT_NAMES.index("Contact"),
    ]
    assert arrays["seg_start_frame"].tolist() == [150 - _ORIGIN, 140 - _ORIGIN]


def test_a_file_whose_only_event_is_skipped_writes_no_segment_and_records_tracks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = _with_events(LmtEvent("coucou", 110, 130, a=1, id=1))
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)
    _ = capsys.readouterr()

    _ = _convert_events(ds, {"skip_event_names": ["coucou"]})

    arrays = _events_file(ds)
    segments = segments_from_npz(arrays)
    assert segments is not None
    assert len(segments) == 0
    assert arrays["label_ids"].size == 63
    (row,) = read_labels_index(ds, _EVENT_KIND).to_dict("records")
    roots = str(row["consumed_source_roots"])
    assert "labels_raw" in roots
    assert "tracks_raw" in roots
    assert (
        "[lmt_events] recording: kept 0 of 1 events; 1 with a skipped name and 0 "
        "without a tracks frame left out"
    ) in capsys.readouterr().err


def test_two_files_with_different_event_sets_share_one_id_per_name(
    tmp_path: Path,
) -> None:
    contact = LmtEvent("Contact", 140, 160, a=1, b=2, id=1)
    stop = LmtEvent("Stop", 110, 120, a=1, id=2)
    ds = _lmt_labels_dataset(
        tmp_path / "ds",
        recordings={
            "alpha": _with_events(contact),
            "beta": _with_events(contact, stop),
        },
        tracks=True,
    )

    _ = _convert_events(ds)

    alpha = _events_file(ds, sequence="alpha")
    beta = _events_file(ds, sequence="beta")
    assert alpha["seg_label"].tolist() == [LMT_EVENT_NAMES.index("Contact")]
    assert beta["seg_label"].tolist() == [
        LMT_EVENT_NAMES.index("Contact"),
        LMT_EVENT_NAMES.index("Stop"),
    ]
    assert alpha["label_ids"].tolist() == beta["label_ids"].tolist() == list(range(63))
    assert alpha["label_names"].tolist() == beta["label_names"].tolist()
    rows = read_labels_index(ds, _EVENT_KIND)
    assert rows["label_ids"].nunique() == 1
    assert rows["label_names"].nunique() == 1


def test_an_unknown_name_refuses_naming_it_and_converts_once_accepted(
    tmp_path: Path,
) -> None:
    recording = _events_recording(
        [
            LmtEvent("zzz first", 300, 310, a=1, id=10),
            LmtEvent("zzz second", 310, 320, a=2, id=11),
        ]
    )
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)

    with pytest.raises(ValueError) as raised:
        _ = _convert_events(ds)

    message = str(raised.value)
    for text in ("zzz first", "zzz second", "recording.sqlite", "extra_event_names"):
        assert text in message
    assert "skip_event_names" in message
    assert read_labels_index(ds, _EVENT_KIND).empty
    assert list((ds.get_root("labels") / _EVENT_KIND).rglob("*.npz")) == []

    _ = _convert_events(ds, {"extra_event_names": ["zzz first", "zzz second"]})
    extended = _events_file(ds)
    assert extended["label_ids"].tolist() == list(range(65))
    assert extended["seg_label"][9:].tolist() == [63, 64]
    assert extended["label_names"][63:].tolist() == ["zzz first", "zzz second"]

    skipped = _convert_events(ds, {"skip_event_names": ["zzz first", "zzz second"]})
    left = _events_file(ds, skipped)
    assert left["seg_event_id"].tolist() == list(range(1, 10))
    assert left["label_ids"].size == 63


def test_a_name_with_a_comma_is_comma_free_in_the_index_and_stored_in_the_file(
    tmp_path: Path,
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", _events_recording(), tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)

    expected = {
        position: comma_free(name) for position, name in enumerate(LMT_EVENT_NAMES)
    }
    assert ds.get_label_map(_EVENT_KIND) == expected
    assert expected[40] == "Side by side Contact opposite way"
    (row,) = read_labels_index(ds, _EVENT_KIND).to_dict("records")
    assert str(row["label_names"]).split(",") == list(expected.values())
    assert arrays["label_names"].tolist() == list(expected.values())
    assert arrays["source_label_names"].tolist() == list(LMT_EVENT_NAMES)
    assert arrays["source_label_names"][40] == _COMMA_NAME
    assert arrays["label_names"].dtype.kind == "U"


def test_the_quality_ids_leave_label_statistics_without_a_column(
    tmp_path: Path,
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", _events_recording(), tracks=True)
    _ = _convert_events(ds)

    arrays = _events_file(ds)

    quality = arrays["quality_label_ids"].tolist()
    assert LMT_EVENT_NAMES.index("Flickering") in quality
    assert LMT_EVENT_NAMES.index("Contact") not in quality
    assert quality == [
        LMT_EVENT_NAMES.index(name)
        for name in ("Detection", "Flickering", "Undetected")
    ]
    result = run_feature(
        ds,
        LabelStatistics(params={"labels": {"kind": _EVENT_KIND}, "fps": 30.0}),
    )
    (table,) = result_outputs(ds, result).values()
    assert f"usage__{state_suffix('Contact')}" in table.columns
    assert not [column for column in table.columns if "Flickering" in column]


@pytest.mark.parametrize(
    ("event", "named"),
    [
        (LmtEvent("Contact", 250, 255, a=1, id=77), "event 77"),
        (LmtEvent("Contact", 255, 250, a=1, id=78), "event 78"),
        (LmtEvent("Contact", 240, 250, a=1, id=79), "event 79"),
        (LmtEvent("Contact", 160, 150, a=1, id=80), "event 80"),
    ],
)
def test_an_event_frame_without_a_row_or_a_reversed_event_refuses_naming_the_id(
    tmp_path: Path, event: LmtEvent, named: str
) -> None:
    default = default_lmt_recording()
    ds = _lmt_labels_dataset(
        tmp_path / "ds",
        _with_events(*default.events[:1], event),
        tracks=True,
    )
    path = ds.base_dir / "raw" / "recording.sqlite"

    with pytest.raises(ValueError) as raised:
        _ = _convert_events(ds)

    assert named in str(raised.value)
    assert str(path) in str(raised.value)
    assert read_labels_index(ds, _EVENT_KIND).empty
    assert list((ds.get_root("labels") / _EVENT_KIND).rglob("*.npz")) == []


def test_the_rebuild_log_holds_the_files_two_rows_in_id_order(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", _events_recording(), tracks=True)
    _ = _convert_events(ds)

    log = _events_file(ds)["rebuild_log"]

    assert log.dtype.kind == "U"
    assert log.shape == (2, 5)
    assert log.tolist() == [
        ["Build DataBase Index", "0", "2022-09-08 13:35:55", "0", "-1"],
        ["Build Event Detection", "0", "2022-09-08 13:36:40", "0", "1188000"],
    ]


def test_a_file_without_a_log_table_has_an_empty_rebuild_log(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(
        tmp_path / "ds", _events_recording(), tracks=True, alter="DROP TABLE LOG"
    )
    _ = _convert_events(ds)

    log = _events_file(ds)["rebuild_log"]

    assert log.shape == (0, 5)
    assert log.dtype.kind == "U"


def test_event_conversion_leaves_the_database_unchanged(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", _events_recording(), tracks=True)
    raw = ds.base_dir / "raw"
    path = raw / "recording.sqlite"
    digests = file_digests(raw)
    mtime_ns = path.stat().st_mtime_ns
    listing = sorted(item.name for item in raw.iterdir())

    _ = _convert_events(ds)

    assert file_digests(raw) == digests
    assert path.stat().st_mtime_ns == mtime_ns
    assert sorted(item.name for item in raw.iterdir()) == listing


def test_a_conversion_without_a_clock_numbers_by_the_origin(tmp_path: Path) -> None:
    recording = _events_recording()
    ds = _lmt_labels_dataset(tmp_path / "ds", recording, tracks=True)
    _ = _convert_events(ds)
    native = _events_file(ds)
    path = ds.base_dir / "raw" / "recording.sqlite"

    (entry,) = LmtEventsConverter().convert(
        path, LmtEventsParams(), {"group": "", "sequence": "recording"}
    )

    assert entry.n_frames == 401 - _ORIGIN + 1
    assert entry.label_ids == tuple(range(63))
    assert entry.label_names == tuple(comma_free(name) for name in LMT_EVENT_NAMES)
    assert (entry.group, entry.sequence) == ("", "recording")
    for key in ("seg_start_frame", "seg_end_frame", "seg_event_id", "seg_label"):
        assert np.asarray(entry.payload[key]).tolist() == native[key].tolist()
