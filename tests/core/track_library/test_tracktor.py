"""The Tracktor converters: a corrected ``.tsv`` and a raw ``.csv`` of one arena.

The release times each frame with a ``seconds_total`` column and drops frames, so
no frame rate gives a frame's time. These tests pin that the converted ``time`` is
that column, whatever rate the caller or the dataset knows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import ensure_track_schema
from mosaic.core.scope import Scope
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    FrameRateUnknownError,
    TimedTrackConvertParams,
    get_track_converter,
)
from mosaic.core.track_library.tracktor import (
    TracktorParams,
    TracktorRawCsvConverter,
    TracktorTsvConverter,
)
from tests.helpers import (
    TRACKTOR_STEM,
    TracktorRecording,
    make_dataset,
    result_outputs,
    write_tracktor_recording,
)

HINTS = EntryHints(group="", sequence="arena")


def _recording(
    tmp_path: Path,
    *,
    frames: int = 80,
    flies: int = 6,
    write_tsv: bool = True,
    missing_position: tuple[int, int] | None = None,
) -> TracktorRecording:
    return write_tracktor_recording(
        tmp_path,
        frames=frames,
        flies=flies,
        write_tsv=write_tsv,
        missing_position=missing_position,
    )


def _tsv(path: Path, hints: EntryHints = HINTS, **params: object) -> pd.DataFrame:
    return TracktorTsvConverter().convert(
        path, TracktorParams.from_overrides(dict(params)), hints
    )


def _csv(path: Path, hints: EntryHints = HINTS, **params: object) -> pd.DataFrame:
    return TracktorRawCsvConverter().convert(
        path, TracktorParams.from_overrides(dict(params)), hints
    )


def _positions(table: pd.DataFrame, fly: int) -> npt.NDArray[np.float64]:
    rows = table[table["id"] == fly]
    return rows[["X", "Y"]].to_numpy(dtype=np.float64)


# --- registration -------------------------------------------------------------


def test_the_two_formats_declare_pixels_and_what_their_ids_mean() -> None:
    corrected = TRACK_CONVERTERS["tracktor_corrected_tsv"]
    raw = TRACK_CONVERTERS["tracktor_raw_csv"]

    assert corrected is TracktorTsvConverter
    assert raw is TracktorRawCsvConverter
    for converter in (corrected, raw):
        assert converter.output_schema == "mosaic_v1"
        assert not converter.enumerable
        assert not converter.merges_per_sequence
    assert corrected.carries_tracklets is True
    assert raw.carries_tracklets is False


def test_the_params_carry_no_frame_rate() -> None:
    assert not issubclass(TracktorParams, TimedTrackConvertParams)
    assert set(TracktorParams.model_fields) == {"strict_schema", "first_frame"}
    assert TracktorParams().first_frame == 62


def test_a_negative_first_frame_is_refused() -> None:
    with pytest.raises(ValidationError):
        TracktorParams(first_frame=-1)


def test_the_first_frame_is_part_of_the_recipe() -> None:
    default = TracktorParams().identity_dump()

    assert default == {"first_frame": 62}
    assert TracktorParams(first_frame=3).identity_dump() == {"first_frame": 3}


# --- the corrected table ------------------------------------------------------


def test_the_corrected_table_validates_as_mosaic_v1(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv)
    _, report = ensure_track_schema(
        table, "mosaic_v1", strict=True, source=str(recording.tsv)
    )

    assert report["missing_required"] == []
    assert report["forbidden_present"] == []


def test_frames_before_the_first_frame_are_dropped(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv)

    assert table["frame"].min() == 62
    assert table["frame"].max() == recording.frames.max()
    assert len(table) == (recording.frames >= 62).sum() * 6


def test_a_lower_first_frame_keeps_the_frames_the_release_advises_against(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, first_frame=3)

    assert table["frame"].min() == 3
    assert len(table) == len(recording.frames) * 6


def test_ids_are_the_fly_numbers_and_follow_the_animals(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, first_frame=3)

    assert sorted(set(table["id"])) == [1, 2, 3, 4, 5, 6]
    for fly in range(1, 7):
        np.testing.assert_allclose(
            _positions(table, fly), recording.corrected[:, fly - 1], atol=1e-9
        )


def test_rows_are_sorted_by_frame_then_id_and_carry_the_hints(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, hints=EntryHints(group="g", sequence="s"))

    assert list(table["id"]) == [1, 2, 3, 4, 5, 6] * table["frame"].nunique()
    assert table["frame"].is_monotonic_increasing
    assert set(table["group"]) == {"g"}
    assert set(table["sequence"]) == {"s"}


def test_the_sequence_falls_back_to_the_recording_name(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, hints=EntryHints())

    assert set(table["sequence"]) == {TRACKTOR_STEM}
    assert set(table["group"]) == {""}


def test_a_row_without_a_position_has_no_row(tmp_path: Path) -> None:
    recording = _recording(tmp_path, missing_position=(70, 3))

    table = _tsv(recording.tsv)

    frame_70 = table[table["frame"] == 70]
    assert sorted(frame_70["id"]) == [1, 2, 4, 5, 6]
    assert len(table) == (recording.frames >= 62).sum() * 6 - 1


def test_smoothed_and_derived_columns_are_not_carried(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv)

    assert set(table.columns) == {
        "frame",
        "time",
        "id",
        "X",
        "Y",
        "group",
        "sequence",
        "stimuli",
        "strain",
        "sex",
        "n_inds",
        "place",
        "prefix",
    }


# --- time ---------------------------------------------------------------------


def test_time_is_the_seconds_total_column(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, first_frame=3)

    times = table["time"].to_numpy(dtype=np.float64)
    frames = table["frame"].to_numpy(dtype=np.int64)
    np.testing.assert_allclose(times, np.repeat(recording.seconds, 6))
    np.testing.assert_array_equal(frames, np.repeat(recording.frames, 6))


def test_time_is_not_the_frame_over_a_rate(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, first_frame=3)

    one_fly = table[table["id"] == 1]
    steps = np.diff(one_fly["time"].to_numpy(dtype=np.float64))
    assert len({round(float(step), 6) for step in steps}) == 3
    for rate in (30.0, 53.0, 60.0):
        assert not np.allclose(
            one_fly["time"].to_numpy(dtype=np.float64),
            one_fly["frame"].to_numpy(dtype=np.float64) / rate,
        )


def test_a_measured_media_rate_does_not_time_the_table(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    without = _tsv(recording.tsv)
    with_rate = _tsv(
        recording.tsv, hints=EntryHints(group="", sequence="arena", fps=25.0)
    )

    pd.testing.assert_frame_equal(without, with_rate)


def test_the_table_records_no_frame_rate_column(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    for table in (_tsv(recording.tsv), _csv(recording.csv)):
        assert "fps" not in table.columns
        assert "frame_rate" not in table.columns


def test_the_converters_never_ask_for_a_rate(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    for name in ("tracktor_corrected_tsv", "tracktor_raw_csv"):
        converter = get_track_converter(name)
        params = TracktorParams.from_overrides({})
        path = recording.tsv if name.endswith("tsv") else recording.csv

        try:
            _ = converter.convert(path, params, HINTS)
        except FrameRateUnknownError as error:  # pragma: no cover
            pytest.fail(f"{name} asked for a frame rate: {error}")


# --- the per-frame and per-recording columns -----------------------------------


def test_stimuli_is_carried_per_frame_as_the_file_spells_it(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv, first_frame=3)

    stimuli = table["stimuli"].to_numpy(dtype=object).reshape(-1, 6)
    assert (stimuli == stimuli[:, :1]).all()
    assert tuple(stimuli[:, 0].tolist()) == recording.stimuli
    assert set(table["stimuli"]) == {"Y", "N"}


def test_the_recording_columns_are_constant_text_from_the_file(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _tsv(recording.tsv)

    assert set(table["strain"]) == {"DGRP101"}
    assert set(table["sex"]) == {"female"}
    assert set(table["n_inds"]) == {"6"}
    assert set(table["place"]) == {"no12"}
    assert set(table["prefix"]) == {"20211117111002"}


def test_a_column_that_changes_within_a_file_is_refused(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    lines = recording.tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    column = header.index("strain")
    cells = lines[5].split("\t")
    cells[column] = "DGRP102"
    lines[5] = "\t".join(cells)
    _ = recording.tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strain"):
        _ = _tsv(recording.tsv)


def test_two_times_for_one_frame_are_refused(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    lines = recording.tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    column = header.index("seconds_total")
    cells = lines[2].split("\t")
    cells[column] = "99.000000"
    lines[2] = "\t".join(cells)
    _ = recording.tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seconds_total"):
        _ = _tsv(recording.tsv, first_frame=3)


def test_a_fly_id_that_is_not_a_fly_number_is_refused(tmp_path: Path) -> None:
    recording = _recording(tmp_path, flies=2)
    text = recording.tsv.read_text(encoding="utf-8").replace("Fly2", "Bee2")
    _ = recording.tsv.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Bee2"):
        _ = _tsv(recording.tsv)


def test_a_file_with_no_frame_at_or_after_the_first_frame_converts_to_no_rows(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path, frames=20)

    table = _tsv(recording.tsv)

    assert len(table) == 0
    assert {"frame", "time", "id", "X", "Y", "group", "sequence"} <= set(table.columns)


# --- the raw table -------------------------------------------------------------


def test_the_raw_table_keeps_the_trackers_ids_and_takes_time_from_the_sibling(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path)

    table = _csv(recording.csv, first_frame=3)

    for fly in range(1, 7):
        np.testing.assert_allclose(
            _positions(table, fly), recording.raw[:, fly - 1], atol=1e-9
        )
    swapped = table[(table["frame"] >= recording.swap_frame) & (table["id"] == 1)]
    corrected = _tsv(recording.tsv, first_frame=3)
    corrected_two = corrected[
        (corrected["frame"] >= recording.swap_frame) & (corrected["id"] == 2)
    ]
    np.testing.assert_allclose(
        swapped[["X", "Y"]].to_numpy(dtype=np.float64),
        corrected_two[["X", "Y"]].to_numpy(dtype=np.float64),
        atol=0.006,
    )
    for column in ("time", "stimuli", "strain", "sex", "n_inds", "place", "prefix"):
        assert list(table[column]) == list(corrected[column])


def test_the_raw_table_drops_the_same_frames_and_validates(tmp_path: Path) -> None:
    recording = _recording(tmp_path)

    table = _csv(recording.csv)
    _, report = ensure_track_schema(
        table, "mosaic_v1", strict=True, source=str(recording.csv)
    )

    assert report["missing_required"] == []
    assert table["frame"].min() == 62
    assert "center_x" not in table.columns


def test_the_raw_table_without_its_sibling_is_refused_and_never_timed_by_a_rate(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path, write_tsv=False)

    with pytest.raises(ValueError, match="_tracked_fixed_final.tsv") as raised:
        _ = _csv(recording.csv, hints=EntryHints(group="", sequence="a", fps=60.0))

    assert not isinstance(raised.value, FrameRateUnknownError)
    assert "no time column" in str(raised.value)


def test_a_raw_frame_the_sibling_lacks_is_refused(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    lines = recording.tsv.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith("70\t")]
    _ = recording.tsv.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frame 70"):
        _ = _csv(recording.csv)


def test_a_raw_file_that_does_not_end_in_tracked_is_refused(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    renamed = recording.csv.with_name("arena.csv")
    _ = recording.csv.rename(renamed)

    with pytest.raises(ValueError, match="_tracked.csv"):
        _ = _csv(renamed)


# --- sequence names -------------------------------------------------------------


def test_both_formats_name_a_recording_the_same_way() -> None:
    corrected = TracktorTsvConverter()
    raw = TracktorRawCsvConverter()

    assert corrected.sequence_from_stem(f"{TRACKTOR_STEM}_tracked_fixed_final") == (
        TRACKTOR_STEM
    )
    assert raw.sequence_from_stem(f"{TRACKTOR_STEM}_tracked") == TRACKTOR_STEM
    assert corrected.sequence_from_stem("other") == "other"


# --- a dataset ------------------------------------------------------------------


def _dataset_with(
    tmp_path: Path, src_format: str, pattern: str, *, write_tsv: bool = True
) -> tuple[Dataset, TracktorRecording]:
    dataset = make_dataset((tmp_path / "ds").resolve())
    recording = write_tracktor_recording(dataset.base_dir / "raw", write_tsv=write_tsv)
    _ = dataset.index_tracks_raw(
        [dataset.base_dir / "raw"], patterns=[pattern], src_format=src_format
    )
    return dataset, recording


def test_a_dataset_converts_a_corrected_file_with_no_frame_rate_anywhere(
    tmp_path: Path,
) -> None:
    dataset, recording = _dataset_with(
        tmp_path, "tracktor_corrected_tsv", "*_tracked_fixed_final.tsv"
    )

    _ = dataset.convert_all_tracks()

    index = read_tracks_index(dataset)
    assert len(index) == 1
    row = index.iloc[0]
    assert row["std_format"] == "mosaic_v1"
    assert row["sequence"] == TRACKTOR_STEM
    table = dataset.load_tracks("", TRACKTOR_STEM)
    assert table["frame"].min() == 62
    expected = recording.seconds[recording.frames >= 62]
    np.testing.assert_allclose(
        table[table["id"] == 1]["time"].to_numpy(dtype=np.float64), expected
    )


def test_a_dataset_converts_a_raw_file_beside_its_corrected_sibling(
    tmp_path: Path,
) -> None:
    dataset, _ = _dataset_with(tmp_path, "tracktor_raw_csv", "*_tracked.csv")

    _ = dataset.convert_all_tracks()

    table = dataset.load_tracks("", TRACKTOR_STEM)
    assert len(table) > 0


def test_a_dataset_fails_a_raw_file_without_its_sibling_and_names_it(
    tmp_path: Path,
) -> None:
    dataset, _ = _dataset_with(
        tmp_path, "tracktor_raw_csv", "*_tracked.csv", write_tsv=False
    )

    failures: list[tuple[str, BaseException]] = []
    outcome = dataset.convert_all_tracks(
        on_failure=lambda path, error: failures.append((path, error))
    )

    assert outcome.failed == 1
    assert len(failures) == 1
    assert "_tracked_fixed_final.tsv" in str(failures[0][1])
    assert read_tracks_index(dataset).empty


def test_speed_is_over_the_recorded_time_and_not_a_frame_rate(tmp_path: Path) -> None:
    dataset, recording = _dataset_with(
        tmp_path, "tracktor_corrected_tsv", "*_tracked_fixed_final.tsv"
    )
    _ = dataset.convert_all_tracks()

    result = run_feature(
        dataset, SpeedAngvel(), scope=Scope(entries=[("", TRACKTOR_STEM)])
    )

    (out,) = result_outputs(dataset, result).values()
    one = out[out["id"] == 1].sort_values("frame")
    keep = recording.frames >= 62
    positions = recording.corrected[keep, 0]
    seconds = recording.seconds[keep]
    expected = np.hypot(*np.diff(positions, axis=0).T) / np.diff(seconds)
    np.testing.assert_allclose(
        one["speed"].to_numpy(dtype=np.float64)[1:], expected, rtol=1e-9
    )
