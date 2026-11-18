"""``mosaic describe``: a dataset's manifest fields, counts, track schemas and frame rates."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.dataset_summary import SUMMARY_KINDS, DatasetSummary, dataset_summary
from mosaic.core.manifest import TracksScanSource
from mosaic.core.raw_report import raw_report
from mosaic.core.selections import save_selection
from mosaic.tracking import dataset_inventory
from tests.helpers import (
    DIVISION_CREATED_UTC,
    MediaClip,
    add_tracks_variant,
    invoke_ok,
    make_dataset,
    store_dataset,
    video_store_maker,
    write_division_rows,
    write_media_index,
)

VARIANT = "convert-trex.0.2-1111111111"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds", name="described")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def test_json_is_the_library_summary(dataset: Dataset) -> None:
    result = invoke_ok(["describe", "--manifest", str(dataset.manifest_path), "--json"])

    printed = DatasetSummary.model_validate_json(result.stdout)
    expected = dataset_summary(
        dataset, found=dataset_inventory(dataset, kinds=SUMMARY_KINDS)
    )
    assert printed == expected
    assert printed.counts.sequences == 2
    assert printed.counts.models == 0


def test_the_plain_output_lists_the_fields_counts_and_schemas(
    dataset: Dataset,
) -> None:
    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert "name: described" in result.stdout
    assert "sequences: 2 with a tracks table" in result.stdout
    assert "tracks_variants: 1" in result.stdout
    assert "managed_by: none" in result.stdout
    assert "name\ttables\nmosaic_v1\t2" in result.stdout


def test_the_plain_output_lists_the_roles_of_the_groups(dataset: Dataset) -> None:
    dataset.set_splits("train", ["a"])
    dataset.set_splits("sealed", ["b"])

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])
    as_json = invoke_ok(["describe", "--manifest", str(dataset.base_dir), "--json"])

    assert "splits: a=train, b=sealed" in result.stdout
    assert DatasetSummary.model_validate_json(as_json.stdout).splits == {
        "a": "train",
        "b": "sealed",
    }


def test_the_plain_output_counts_the_recordings(dataset: Dataset) -> None:
    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert "recordings: 2 in the media or tracks index (0 divided)" in result.stdout
    assert "continuous_groups" not in result.stdout


def test_the_plain_output_names_the_divided_recordings(dataset: Dataset) -> None:
    write_media_index(
        dataset,
        [
            MediaClip(group="night", sequence="s1", filename="night_s1.mp4"),
            MediaClip(group="night", sequence="s2", filename="night_s2.mp4"),
            MediaClip(group="day", sequence="s1", filename="day_s1.mp4"),
        ],
    )
    dataset.set_continuous_groups(["night", "day"])

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert (
        "recordings: 4 in the media or tracks index (2 divided: day, night)"
        in result.stdout
    )


def test_a_media_only_dataset_counts_its_recordings_beside_no_sequence(
    tmp_path: Path,
) -> None:
    """Each count names what it counts, so ``0`` beside ``1`` reads as both true."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(sequence="20210219_112253")])

    result = invoke_ok(["describe", "--manifest", str(ds.base_dir)])

    assert "sequences: 0 with a tracks table" in result.stdout
    assert "recordings: 1 in the media or tracks index (0 divided)" in result.stdout


@pytest.mark.media
def test_the_plain_output_states_each_divided_recording_s_rule(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """A store of chunks of 5 frames at 4 fps, divided every 10 frames."""
    ds = store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker((5, 5, 5, 5, 2), fps=4.0, created_utc=DIVISION_CREATED_UTC),
    )
    _ = write_division_rows(ds, ds.get_root("media_raw") / "recordings" / "rec", 10)

    result = invoke_ok(["describe", "--manifest", str(ds.base_dir)])

    assert (
        "division rec: 3 sequences of 10 frames (chunks of 5) by store, camera "
        "none, from 2021-02-19T19:22:53.000000+00:00 to "
        "2021-02-19T19:22:58.000000+00:00, 0 breaks"
    ) in result.stdout.splitlines()


def test_the_plain_output_lists_the_selections(dataset: Dataset) -> None:
    saved = save_selection(dataset, "day1", [("", "seq_a")], note="first day")

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert "name\tentries\tcreated_at\tnote" in result.stdout
    assert f"day1\t1\t{saved.created_at}\tfirst day" in result.stdout


def test_the_json_lists_the_selections(dataset: Dataset) -> None:
    _ = save_selection(dataset, "day1", [("", "seq_a"), ("", "seq_b")])

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir), "--json"])

    printed = DatasetSummary.model_validate_json(result.stdout)
    assert [(item.name, item.n_entries) for item in printed.selections] == [("day1", 2)]


def test_the_plain_output_names_a_selection_file_that_cannot_be_read(
    dataset: Dataset,
) -> None:
    _ = save_selection(dataset, "day1", [("", "seq_a")])
    broken = dataset.base_dir / ".mosaic" / "selections" / "day1 (1).json"
    _ = broken.write_text("{", encoding="utf-8")

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert "day1 (1).json" in result.stderr


def test_the_plain_output_without_selections_prints_no_selection_table(
    dataset: Dataset,
) -> None:
    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert "created_at" not in result.stdout


def _declare_duplicated_tracks(ds: Dataset, folder: Path) -> None:
    folder.mkdir(parents=True)
    for name in ("one.npy", "two.npy"):
        _ = (folder / name).write_bytes(b"same bytes")
    ds.add_scan_source(
        TracksScanSource.model_validate(
            {"id": "cams", "path": str(folder), "patterns": ["*.npy"]}
        )
    )
    _ = ds.scan_tracks()


def test_raw_json_prints_the_raw_report(dataset: Dataset, tmp_path: Path) -> None:
    _declare_duplicated_tracks(dataset, tmp_path / "cams")

    result = invoke_ok(
        ["describe", "--manifest", str(dataset.manifest_path), "--raw", "--json"]
    )

    printed = DatasetSummary.model_validate_json(result.stdout)
    assert printed.raw_report == raw_report(dataset)
    assert printed.raw_report is not None
    assert printed.raw_report.duplicate_sets == 1


def test_the_json_without_raw_prints_null_for_the_report(dataset: Dataset) -> None:
    result = invoke_ok(["describe", "--manifest", str(dataset.manifest_path), "--json"])

    assert json.loads(result.stdout)["raw_report"] is None
    assert DatasetSummary.model_validate_json(result.stdout).raw_report is None


def test_the_plain_output_without_raw_prints_no_report(dataset: Dataset) -> None:
    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir)])

    assert "duplicate_sets" not in result.stdout


def test_the_plain_output_with_raw_prints_the_report_counts(
    dataset: Dataset, tmp_path: Path
) -> None:
    _declare_duplicated_tracks(dataset, tmp_path / "cams")

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir), "--raw"])

    assert "duplicate_sets: 1" in result.stdout
    assert "pattern_miss_sources: 0" in result.stdout
    assert "frame_axis_tables: 0" in result.stdout
    assert "span_outlier_tables: 0" in result.stdout
    assert "time_irregular_entries: 0" in result.stdout
    assert "missing_source_paths: 0" in result.stdout


def test_the_plain_output_with_raw_names_a_missing_source(
    dataset: Dataset, tmp_path: Path
) -> None:
    dataset.add_scan_source(
        TracksScanSource.model_validate(
            {"id": "gone", "path": str(tmp_path / "unmounted"), "patterns": ["*.npy"]}
        )
    )

    result = invoke_ok(["describe", "--manifest", str(dataset.base_dir), "--raw"])

    assert "missing_source_paths: 1" in result.stdout
    assert "tracks\tgone\t" in result.stdout
    assert not (tmp_path / "unmounted").exists()
