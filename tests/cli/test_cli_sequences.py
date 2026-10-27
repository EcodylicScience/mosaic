"""``mosaic sequences``: the entries of the media and tracks indexes, each with its label and recording."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from tests.helpers import (
    DIVISION_CREATED_UTC,
    MediaClip,
    add_tracks_table,
    invoke_json,
    invoke_ok,
    make_dataset,
    store_dataset,
    track_table,
    video_store_maker,
    write_division_rows,
    write_media_index,
)

VARIANT = "convert-trex.0.2-1111111111"
UNDIVIDED: dict[str, object] = {
    "first_frame": None,
    "frame_count": None,
    "start_utc": None,
    "follows_gap": None,
}
"""The division fields of an entry that is not a division of a store."""


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
            "has_tracks": True,
            **UNDIVIDED,
        },
        {
            "group": "night",
            "sequence": "s1",
            "recording": "night",
            "display_name": "first hour",
            "has_tracks": True,
            **UNDIVIDED,
        },
        {
            "group": "night",
            "sequence": "s2",
            "recording": "night",
            "display_name": "s2",
            "has_tracks": True,
            **UNDIVIDED,
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


def test_a_media_only_entry_is_listed_with_its_recording(tmp_path: Path) -> None:
    """A dataset holding media and no tracks lists its entries."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(sequence="20210219_112253"),
            MediaClip(group="night", sequence="s1", filename="night_s1.mp4"),
        ],
    )
    add_tracks_table(ds, VARIANT, track_table("night", "s2"))
    ds.set_continuous_groups(["night"])

    payload = invoke_json(["sequences", "-m", str(ds.manifest_path), "--json"])
    plain = invoke_ok(["sequences", "-m", str(ds.manifest_path)])

    assert payload["entries"] == [
        {
            "group": "",
            "sequence": "20210219_112253",
            "recording": "20210219_112253",
            "display_name": "20210219_112253",
            "has_tracks": False,
            **UNDIVIDED,
        },
        {
            "group": "night",
            "sequence": "s1",
            "recording": "night",
            "display_name": "s1",
            "has_tracks": False,
            **UNDIVIDED,
        },
        {
            "group": "night",
            "sequence": "s2",
            "recording": "night",
            "display_name": "s2",
            "has_tracks": True,
            **UNDIVIDED,
        },
    ]
    assert payload["sequences"] == ["20210219_112253", "s1", "s2"]
    assert plain.stdout.splitlines() == ["20210219_112253", "night:s1", "night:s2"]


@pytest.mark.media
def test_a_division_is_listed_with_its_frames_start_and_wall_clock_name(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """A store of chunks of 5 frames at 4 fps, divided every 10 frames."""
    ds = store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker((5, 5, 5, 5, 2), fps=4.0, created_utc=DIVISION_CREATED_UTC),
    )
    _ = write_division_rows(ds, ds.get_root("media_raw") / "recordings" / "rec", 10)

    payload = invoke_json(["sequences", "-m", str(ds.manifest_path), "--json"])
    plain = invoke_ok(["sequences", "-m", str(ds.manifest_path)])

    assert payload["entries"] == [
        {
            "group": "rec",
            "sequence": sequence,
            "recording": "rec",
            "display_name": display_name,
            "has_tracks": False,
            "first_frame": first_frame,
            "frame_count": frame_count,
            "start_utc": start_utc,
            "follows_gap": False,
        }
        for sequence, display_name, first_frame, frame_count, start_utc in (
            (
                "20210219T192253Z",
                "2021-02-19 19:22:53Z",
                0,
                10,
                "2021-02-19T19:22:53.000000+00:00",
            ),
            (
                "20210219T192255Z",
                "2021-02-19 19:22:55Z",
                10,
                10,
                "2021-02-19T19:22:55.500000+00:00",
            ),
            (
                "20210219T192258Z",
                "2021-02-19 19:22:58Z",
                20,
                2,
                "2021-02-19T19:22:58.000000+00:00",
            ),
        )
    ]
    assert (
        plain.stdout.splitlines()[0] == "rec:20210219T192253Z  (2021-02-19 19:22:53Z)"
    )
