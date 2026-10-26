"""``mosaic describe``: a dataset's manifest fields, counts, track schemas and frame rates."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.dataset_summary import SUMMARY_KINDS, DatasetSummary, dataset_summary
from mosaic.tracking import dataset_inventory
from tests.helpers import (
    MediaClip,
    add_tracks_variant,
    invoke_ok,
    make_dataset,
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
