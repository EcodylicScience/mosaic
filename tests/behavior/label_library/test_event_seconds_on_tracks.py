"""An event at ``t`` seconds lands on the last frame of the entry's tracks at or before ``t``.

A table that dropped frames is not timed by ``frame / fps``, so ``int(t * fps)``
puts an event after a gap on the wrong frame. When the entry has tracks, the
converter reads their ``time`` column, a floor that matches ``int(t * fps)`` on an
evenly timed table. An event before the first track time starts on the first
frame, one that ends before it is dropped, and one that starts after the last is
dropped. Without tracks it keeps ``int(t * fps)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.convert_ops import ConvertLabelsParams
from mosaic.core.pipeline.labels_identity import read_labels_variant
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import read_tracks_index

from tests.helpers import (
    entry_error_lines,
    latest_snapshot,
    add_tracks_table,
    invoke_json,
    dropped_frame_times,
    dropped_frame_tracks,
    make_dataset,
)

VARIANT = "convert-tracktor_corrected_tsv.0.1-aaaaaaaaaa"
OBSERVATION = "obs"
SEQUENCE = "obs__ant"
FPS = 30.0


def _boris_table(start: float, stop: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Observation id": [OBSERVATION],
            "Subject": ["ant"],
            "Behavior": ["approach"],
            "Behavior type": ["STATE"],
            "Start (s)": [start],
            "Stop (s)": [stop],
            "FPS": [FPS],
        }
    )


def _dataset(tmp_path: Path, *, with_tracks: bool) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    if with_tracks:
        table = dropped_frame_tracks(ids=(0,), sequence=SEQUENCE)
        add_tracks_table(ds, VARIANT, table, std_format="mosaic_v1")
    return ds


def _index_pickle(ds: Dataset, tmp_path: Path, table: pd.DataFrame) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    table.to_pickle(uploads / "boris.pkl")
    ds.index_labels_raw([uploads], patterns=["*.pkl"], src_format="boris_pandas_pickle")


def _convert_pickle(
    ds: Dataset,
    tmp_path: Path,
    table: pd.DataFrame,
    tracks_run_id: str | None = None,
) -> np.ndarray:
    _index_pickle(ds, tmp_path, table)
    _ = ds.convert_all_labels(
        kind="behavior",
        source_format="boris_pandas_pickle",
        tracks_run_id=tracks_run_id,
    )
    index = read_labels_index(ds, "behavior")
    (path,) = index["abs_path"]
    return np.load(ds.resolve_path(str(path)), allow_pickle=False)["frames"]


def test_an_event_after_a_gap_lands_on_the_frame_the_clock_says(
    tmp_path: Path,
) -> None:
    times = dropped_frame_times()
    inside_gap = float(times[89] + 0.7)
    ds = _dataset(tmp_path, with_tracks=True)

    frames = _convert_pickle(ds, tmp_path, _boris_table(inside_gap, float(times[120])))

    # The last frame at or before a time in the 1.45 s gap is the frame before it,
    # and the stop is the frame timed exactly there.
    assert frames.tolist() == list(range(89, 121))
    assert int(inside_gap * FPS) != 89


def test_an_event_past_the_last_tracked_time_is_dropped(tmp_path: Path) -> None:
    times = dropped_frame_times()
    ds = _dataset(tmp_path, with_tracks=True)

    frames = _convert_pickle(
        ds, tmp_path, _boris_table(float(times[-1]) + 5.0, float(times[-1]) + 9.0)
    )

    assert frames.size == 0


def test_a_stop_past_the_last_tracked_time_ends_on_the_last_frame(
    tmp_path: Path,
) -> None:
    times = dropped_frame_times()
    ds = _dataset(tmp_path, with_tracks=True)

    frames = _convert_pickle(ds, tmp_path, _boris_table(float(times[-5]), 99.0))

    assert frames.tolist() == list(range(len(times) - 5, len(times)))


def test_a_floor_on_an_even_table_is_int_of_seconds_times_rate(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    even = dropped_frame_tracks(ids=(0,), sequence=SEQUENCE)
    even["time"] = even["frame"] / FPS
    add_tracks_table(ds, VARIANT, even, std_format="mosaic_v1")

    frames = _convert_pickle(ds, tmp_path, _boris_table(2.04, 3.0))

    assert frames.tolist() == list(range(int(2.04 * FPS), int(3.0 * FPS) + 1))


def test_an_event_starting_before_late_tracks_starts_on_their_first_frame(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    late = dropped_frame_tracks(ids=(0,), sequence=SEQUENCE)
    late["time"] = late["time"] + 5.0
    add_tracks_table(ds, VARIANT, late, std_format="mosaic_v1")
    last = float(late["time"].iloc[40])

    frames = _convert_pickle(ds, tmp_path, _boris_table(2.0, last))

    assert frames.tolist() == list(range(0, 41))


def test_an_event_ending_before_late_tracks_is_dropped(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    late = dropped_frame_tracks(ids=(0,), sequence=SEQUENCE)
    late["time"] = late["time"] + 5.0
    add_tracks_table(ds, VARIANT, late, std_format="mosaic_v1")

    frames = _convert_pickle(ds, tmp_path, _boris_table(1.0, 4.0))

    assert frames.size == 0


def test_an_entry_without_tracks_keeps_the_rate_conversion(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, with_tracks=False)

    frames = _convert_pickle(ds, tmp_path, _boris_table(2.0, 3.0))

    assert frames.tolist() == list(range(int(2.0 * FPS), int(3.0 * FPS) + 1))


def test_a_label_row_made_from_tracks_records_the_tracks_it_read(
    tmp_path: Path,
) -> None:
    times = dropped_frame_times()
    ds = _dataset(tmp_path, with_tracks=True)
    _ = _convert_pickle(ds, tmp_path, _boris_table(float(times[10]), float(times[20])))

    (roots,) = read_labels_index(ds, "behavior")["consumed_source_roots"]

    assert "tracks_raw" in str(roots)
    assert "labels_raw" in str(roots)


OTHER_VARIANT = "convert-tracktor_raw_csv.0.1-bbbbbbbbbb"


def _two_variants(tmp_path: Path) -> Dataset:
    ds = _dataset(tmp_path, with_tracks=True)
    other = dropped_frame_tracks(ids=(0,), sequence=SEQUENCE)
    other["time"] = other["time"] + 100.0
    add_tracks_table(ds, OTHER_VARIANT, other, std_format="mosaic_v1")
    return ds


def test_an_entry_with_two_tracks_variants_is_refused_and_the_message_names_the_pin(
    tmp_path: Path,
) -> None:
    ds = _two_variants(tmp_path)

    with pytest.raises(ValueError, match=r"holds 2 variants.*tracks_run_id"):
        _ = _convert_pickle(ds, tmp_path, _boris_table(0.5, 1.0))


def test_a_pinned_tracks_run_id_picks_the_variant_the_clock_reads(
    tmp_path: Path,
) -> None:
    ds = _two_variants(tmp_path)
    times = dropped_frame_times()

    frames = _convert_pickle(
        ds,
        tmp_path,
        _boris_table(float(times[10]), float(times[20])),
        tracks_run_id=VARIANT,
    )

    assert frames.tolist() == list(range(10, 21))


def test_a_rerun_with_the_pin_skips_the_files_it_wrote(tmp_path: Path) -> None:
    ds = _two_variants(tmp_path)
    times = dropped_frame_times()
    table = _boris_table(float(times[10]), float(times[20]))
    _ = _convert_pickle(ds, tmp_path, table, tracks_run_id=VARIANT)

    outcome = ds.convert_all_labels(
        kind="behavior", source_format="boris_pandas_pickle", tracks_run_id=VARIANT
    )

    assert outcome.written == 0
    assert outcome.skipped == 1


def test_the_cli_reports_a_lost_entry_as_partial(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    _index_pickle(ds, tmp_path, _boris_observations(OBSERVATION, "other"))

    payload = invoke_json(
        [
            "convert-labels",
            "-m",
            str(ds.manifest_path),
            "--source-format",
            "boris_pandas_pickle",
            "--json",
        ]
    )

    assert (payload["status"], payload["written"], payload["failed"]) == (
        "partial",
        1,
        1,
    )


def test_the_cli_takes_a_tracks_run_id(tmp_path: Path) -> None:
    ds = _two_variants(tmp_path)
    times = dropped_frame_times()
    _index_pickle(ds, tmp_path, _boris_table(float(times[10]), float(times[20])))

    payload = invoke_json(
        [
            "convert-labels",
            "-m",
            str(ds.manifest_path),
            "--source-format",
            "boris_pandas_pickle",
            "--tracks-run-id",
            VARIANT,
            "--json",
        ]
    )

    assert payload["written"] == 1


def test_the_convert_labels_step_takes_a_tracks_run_id() -> None:
    params = ConvertLabelsParams(
        source_format="boris_pandas_pickle", tracks_run_id=VARIANT
    )

    assert params.tracks_run_id == VARIANT
    assert "tracks_run_id" not in params.params


# --- the variant names the tracks the labels are placed on ---------------------


def _label_variants(ds: Dataset) -> list[str]:
    return sorted(
        {str(variant) for variant in read_labels_index(ds, "behavior")["run_id"]}
    )


def _resample(ds: Dataset, source: str) -> str:
    """Resample *source* onto 30 fps and return the variant that wrote."""
    _ = run_op(
        ds,
        "resample-tracks",
        {"target_fps": 30.0, "source_tracks_run_id": source},
        track=False,
    )
    (produced,) = set(read_tracks_index(ds)["run_id"]) - {source}
    return str(produced)


def _convert_pinned(ds: Dataset, tracks_run_id: str | None) -> str:
    return ds.convert_all_labels(
        kind="behavior",
        source_format="boris_pandas_pickle",
        tracks_run_id=tracks_run_id,
    ).variant


def test_converting_against_two_tracks_variants_gives_two_label_variants(
    tmp_path: Path,
) -> None:
    ds = _two_variants(tmp_path)
    times = dropped_frame_times()
    _index_pickle(ds, tmp_path, _boris_table(float(times[10]), float(times[20])))

    first = _convert_pinned(ds, VARIANT)
    second = _convert_pinned(ds, OTHER_VARIANT)

    assert first != second
    assert _label_variants(ds) == sorted([first, second])


def test_the_pinned_tracks_variant_is_in_the_label_variant_and_its_record(
    tmp_path: Path,
) -> None:
    ds = _two_variants(tmp_path)

    on_first = ds.label_conversion_variant(
        "behavior", "boris_pandas_pickle", tracks_run_id=VARIANT
    )
    on_second = ds.label_conversion_variant(
        "behavior", "boris_pandas_pickle", tracks_run_id=OTHER_VARIANT
    )

    assert on_first != on_second
    times = dropped_frame_times()
    _index_pickle(ds, tmp_path, _boris_table(float(times[10]), float(times[20])))
    assert _convert_pinned(ds, VARIANT) == on_first
    record = read_labels_variant(ds.get_root("labels") / "behavior" / on_first)
    assert record is not None
    assert record.params["tracks_variant"] == VARIANT


def test_a_variant_placed_on_no_tracks_is_not_one_placed_on_tracks(
    tmp_path: Path,
) -> None:
    without = _dataset(tmp_path / "a", with_tracks=False)
    with_tracks = _dataset(tmp_path / "b", with_tracks=True)

    bare = without.label_conversion_variant("behavior", "boris_pandas_pickle")
    placed = with_tracks.label_conversion_variant("behavior", "boris_pandas_pickle")

    assert bare != placed


def test_a_rerun_against_other_tracks_never_skips_a_file_of_another_axis(
    tmp_path: Path,
) -> None:
    ds = _two_variants(tmp_path)
    times = dropped_frame_times()
    _index_pickle(ds, tmp_path, _boris_table(float(times[10]), float(times[20])))
    _ = _convert_pinned(ds, VARIANT)

    outcome = ds.convert_all_labels(
        kind="behavior",
        source_format="boris_pandas_pickle",
        tracks_run_id=OTHER_VARIANT,
    )

    assert outcome.written == 1
    assert outcome.skipped == 0


def _boris_observations(*observations: str) -> pd.DataFrame:
    """One event of the subject ``ant`` in each of *observations*."""
    return pd.concat(
        [
            _boris_table(2.0, 3.0).assign(**{"Observation id": name})
            for name in observations
        ],
        ignore_index=True,
    )


def _label_files(ds: Dataset) -> list[str]:
    return sorted(str(row) for row in read_labels_index(ds, "behavior")["sequence"])


def test_an_entry_without_tracks_in_the_variant_is_a_failed_entry_with_no_file(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    _index_pickle(ds, tmp_path, _boris_observations(OBSERVATION, "other"))
    failures: list[str] = []

    def on_failure(key: str, error: BaseException) -> None:
        failures.append(f"{key}: {error}")

    outcome = ds.convert_all_labels(
        kind="behavior", source_format="boris_pandas_pickle", on_failure=on_failure
    )

    assert (outcome.written, outcome.failed_entries) == (1, (("", "other__ant"),))
    assert not outcome.lost_everything
    assert _label_files(ds) == [SEQUENCE]
    (message,) = failures
    assert message.startswith("other__ant: ")
    assert "has no tracks in the variant" in message
    assert "aaaaaaaaaa" in message


def test_a_later_conversion_fills_an_entry_once_its_tracks_exist(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    _index_pickle(ds, tmp_path, _boris_observations(OBSERVATION, "other"))
    _ = ds.convert_all_labels(kind="behavior", source_format="boris_pandas_pickle")
    late = dropped_frame_tracks(ids=(0,), sequence="other__ant")
    add_tracks_table(ds, VARIANT, late, std_format="mosaic_v1")

    outcome = ds.convert_all_labels(
        kind="behavior", source_format="boris_pandas_pickle"
    )

    assert (outcome.written, outcome.skipped, outcome.failed_entries) == (1, 1, ())
    assert _label_files(ds) == ["obs__ant", "other__ant"]


def test_losing_every_entry_registers_no_variant_and_fails_the_step(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    _index_pickle(ds, tmp_path, _boris_observations("other"))

    outcome = ds.convert_all_labels(
        kind="behavior", source_format="boris_pandas_pickle"
    )

    assert outcome.lost_everything
    assert outcome.written == 0
    section = ds.manifest.meta.get("labels")
    assert not isinstance(section, dict) or "behavior" not in section
    with pytest.raises(AllEntriesFailed, match="has no tracks in the variant"):
        _ = run_op(ds, "convert-labels", {"source_format": "boris_pandas_pickle"})
    assert latest_snapshot(ds)["status"] == "failed"


def test_the_step_reports_a_lost_entry_as_an_entry_error_and_still_finishes(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    _index_pickle(ds, tmp_path, _boris_observations(OBSERVATION, "other"))

    _ = run_op(ds, "convert-labels", {"source_format": "boris_pandas_pickle"})

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "finished"
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 1
    [line] = entry_error_lines(ds, snapshot["execution_id"])
    assert "other__ant" in line


def test_entries_the_labels_name_on_different_tracks_variants_are_refused_without_a_pin(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    elsewhere = dropped_frame_tracks(ids=(0,), sequence="other__ant")
    add_tracks_table(ds, OTHER_VARIANT, elsewhere, std_format="mosaic_v1")
    _index_pickle(ds, tmp_path, _boris_observations(OBSERVATION, "other"))

    with pytest.raises(ValueError, match=r"aaaaaaaaaa.*bbbbbbbbbb.*tracks_run_id"):
        _ = ds.convert_all_labels(kind="behavior", source_format="boris_pandas_pickle")


def test_an_entry_the_labels_do_not_name_does_not_make_the_variant_ambiguous(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, with_tracks=True)
    unrelated = dropped_frame_tracks(ids=(0,), sequence="unrelated__ant")
    for variant in (VARIANT, OTHER_VARIANT):
        add_tracks_table(ds, variant, unrelated, std_format="mosaic_v1")
    _index_pickle(ds, tmp_path, _boris_observations(OBSERVATION))

    outcome = ds.convert_all_labels(
        kind="behavior", source_format="boris_pandas_pickle"
    )

    assert outcome.written == 1
    record = read_labels_variant(ds.get_root("labels") / "behavior" / outcome.variant)
    assert record is not None
    assert record.params["tracks_variant"] == VARIANT


def test_labels_without_tracks_beside_several_tracks_variants_name_none_to_use(
    tmp_path: Path,
) -> None:
    ds = _two_variants(tmp_path)
    _index_pickle(ds, tmp_path, _boris_observations("other"))

    with pytest.raises(ValueError, match=r"none of the entries.*tracks_run_id"):
        _ = ds.convert_all_labels(kind="behavior", source_format="boris_pandas_pickle")


def test_labels_converted_on_resampled_tracks_are_on_the_resampled_axis(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    table = dropped_frame_tracks(ids=(0,), sequence=SEQUENCE)
    add_tracks_table(ds, VARIANT, table, std_format="trex_v2")
    produced = _resample(ds, VARIANT)
    times = dropped_frame_times()
    start, stop = float(times[89] + 0.7), float(times[120])
    _index_pickle(ds, tmp_path, _boris_table(start, stop))

    on_native = _convert_pinned(ds, VARIANT)
    on_resampled = _convert_pinned(ds, produced)

    assert on_native != on_resampled
    by_variant = {
        str(row["run_id"]): np.load(
            ds.resolve_path(str(row["abs_path"])), allow_pickle=False
        )["frames"]
        for _, row in read_labels_index(ds, "behavior").iterrows()
    }
    assert by_variant[on_native].tolist() == list(range(89, 121))
    assert by_variant[on_resampled].tolist() == list(
        range(int(start * 30.0), int(stop * 30.0) + 1)
    )
