"""``mosaic sequences``: the entries of the tracks index, each with its label and recording."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from tests.helpers import (
    MediaClip,
    add_tracks_table,
    invoke_json,
    invoke_ok,
    make_dataset,
    track_table,
    write_media_index,
)

VARIANT = "convert-trex.0.2-1111111111"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Two groups sharing the sequence ``s1``, one of them a continuous group."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(group="night", sequence="s1", filename="night_s1.mp4"),
            MediaClip(group="night", sequence="s2", filename="night_s2.mp4"),
        ],
    )
    for group, sequence in (("night", "s1"), ("night", "s2"), ("arena", "s1")):
        add_tracks_table(ds, VARIANT, track_table(group, sequence))
    ds.set_continuous_groups(["night"])
    ds.set_display_name("arena", "s1", "arena one")
    ds.set_display_name("night", "s1", "first hour")
    return ds


def test_each_entry_is_listed_with_its_group_label_and_recording(
    dataset: Dataset,
) -> None:
    payload = invoke_json(["sequences", "-m", str(dataset.manifest_path), "--json"])

    assert payload["sequences"] == ["s1", "s2"]
    assert payload["entries"] == [
        {
            "group": "arena",
            "sequence": "s1",
            "recording": "arena:s1",
            "display_name": "arena one",
        },
        {
            "group": "night",
            "sequence": "s1",
            "recording": "night",
            "display_name": "first hour",
        },
        {
            "group": "night",
            "sequence": "s2",
            "recording": "night",
            "display_name": "s2",
        },
    ]
    assert payload["labelled"] == [
        {"sequence": "s1", "display_name": "arena one"},
        {"sequence": "s1", "display_name": "first hour"},
        {"sequence": "s2", "display_name": "s2"},
    ]


def test_a_group_narrows_the_entries_and_keeps_their_labels(dataset: Dataset) -> None:
    payload = invoke_json(
        ["sequences", "-m", str(dataset.manifest_path), "--group", "night", "--json"]
    )

    assert payload["sequences"] == ["s1", "s2"]
    assert payload["labelled"] == [
        {"sequence": "s1", "display_name": "first hour"},
        {"sequence": "s2", "display_name": "s2"},
    ]


def test_the_plain_output_names_each_entry_by_its_token(dataset: Dataset) -> None:
    result = invoke_ok(["sequences", "-m", str(dataset.manifest_path)])

    assert result.stdout.splitlines() == [
        "arena:s1  (arena one)",
        "night:s1  (first hour)",
        "night:s2",
    ]
