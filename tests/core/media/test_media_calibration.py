"""A calibration is user knowledge, so no scan may overwrite it with silence.

``cm_per_pixel`` says how far apart two adjacent pixels are in the world. It
lives on the media row because that is what it is a property of -- the camera and
the rig -- rather than on a tracks table derived from it: a dataset may mix rigs,
and a reconversion must not have to be told the scale again.

Nothing measures it. ``ffprobe`` cannot report it and no scan can propose one, so
unlike ``(group, sequence)`` it has no guessed counterpart that a rescan might
legitimately re-derive. That is why it survives a rescan unconditionally, rather
than only when the row's identity was assigned.

Empty means *uncalibrated*, and the column is text so it can. As a number the
empty cell would arrive as ``0.0`` or NaN, and the first of those is a scale
factor rather than an absence -- exactly the confusion between "does not say" and
"says one" that the TRex conversion refuses for the same reason.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.manifest import MediaDivision
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    load_media_index_frame,
    write_media_index_rows,
)

from tests.helpers import (
    DIVISION_CREATED_UTC,
    MediaClip,
    add_media_sequence,
    make_dataset,
    video_store_maker,
    write_h264_mp4,
    write_media_index,
)


def _dataset_with_media(tmp_path: Path) -> tuple[Dataset, str, str]:
    """A dataset holding one indexed sequence, plus its (group, sequence)."""
    manifest = new_dataset_manifest(name="calib", base_dir=tmp_path / "dataset")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_media_sequence(ds, "seq_a", videos=("a.mp4",), frames=3)
    return ds, "", "seq_a"


def test_the_column_is_part_of_the_media_schema() -> None:
    assert "cm_per_pixel" in MEDIA_INDEX_COLUMNS


def test_an_uncalibrated_sequence_reports_none_rather_than_one(tmp_path: Path) -> None:
    """ "Nobody said" must not read as "one centimetre per pixel"."""
    ds, group, sequence = _dataset_with_media(tmp_path)
    assert ds.media_calibration(group, sequence) is None


def test_a_recorded_calibration_reads_back(tmp_path: Path) -> None:
    ds, group, sequence = _dataset_with_media(tmp_path)

    assert ds.set_media_calibration(0.0412) == 1
    assert ds.media_calibration(group, sequence) == pytest.approx(0.0412)


def test_clearing_returns_it_to_uncalibrated(tmp_path: Path) -> None:
    ds, group, sequence = _dataset_with_media(tmp_path)

    _ = ds.set_media_calibration(0.5)
    _ = ds.set_media_calibration(None)
    assert ds.media_calibration(group, sequence) is None


def test_a_non_positive_scale_refuses(tmp_path: Path) -> None:
    ds, _group, _sequence = _dataset_with_media(tmp_path)
    with pytest.raises(ValueError, match="not a usable scale"):
        _ = ds.set_media_calibration(0.0)


def test_a_rescan_does_not_clear_a_calibration(tmp_path: Path) -> None:
    """The invariant this column exists to keep.

    A rescan refreshes what it measured -- size, mtime, probe -- and a
    calibration is none of those. Losing it here would be silent: the column
    would simply be empty again, and a later run would refuse to produce
    centimetres with no indication that somebody had already said what they were.
    """
    ds, group, sequence = _dataset_with_media(tmp_path)
    _ = ds.set_media_calibration(0.25)

    _ = ds.index_media([ds.get_root("media_raw")], extensions=(".mp4",))

    assert ds.media_calibration(group, sequence) == pytest.approx(0.25)


# --- a recording and its cameras -----------------------------------------------


def _two_camera_dataset(tmp_path: Path) -> Dataset:
    """The continuous group ``night`` of two sequences, each seen by two cameras."""
    ds = make_dataset(tmp_path / "dataset")
    write_media_index(
        ds,
        [
            MediaClip(
                group="night",
                sequence=sequence,
                camera=camera,
                filename=f"{sequence}_{camera}.mp4",
            )
            for sequence in ("s1", "s2")
            for camera in ("left", "right")
        ]
        + [MediaClip(group="arena", sequence="s1", filename="arena_s1.mp4")],
    )
    ds.set_continuous_groups(["night"])
    return ds


def _scales(ds: Dataset) -> dict[tuple[str, str, str], str]:
    """Each media row's ``cm_per_pixel`` cell, by group, sequence and camera."""
    return {
        (row["group"], row["sequence"], row["camera"]): row["cm_per_pixel"]
        for row in ds.read_media_index()
    }


def test_a_recording_s_calibration_is_written_on_every_member_row(
    tmp_path: Path,
) -> None:
    ds = _two_camera_dataset(tmp_path)

    assert ds.set_media_calibration(0.25, recording="night") == 4

    assert _scales(ds) == {
        ("night", "s1", "left"): "0.25",
        ("night", "s1", "right"): "0.25",
        ("night", "s2", "left"): "0.25",
        ("night", "s2", "right"): "0.25",
        ("arena", "s1", ""): "",
    }
    assert ds.media_calibration("night", "s2") == pytest.approx(0.25)


def test_an_undivided_recording_is_named_by_its_entry_token(tmp_path: Path) -> None:
    ds = _two_camera_dataset(tmp_path)

    assert ds.set_media_calibration(0.5, recording="arena:s1") == 1
    assert ds.set_media_calibration(0.5, recording="arena") == 0

    assert ds.media_calibration("arena", "s1") == pytest.approx(0.5)
    assert ds.media_calibration("night", "s1") is None


def test_a_camera_narrows_the_rows_calibrated(tmp_path: Path) -> None:
    ds = _two_camera_dataset(tmp_path)

    assert ds.set_media_calibration(0.25, recording="night", camera="left") == 2
    assert ds.set_media_calibration(0.5, recording="night", camera="right") == 2

    assert ds.media_calibration("night", "s1", camera="left") == pytest.approx(0.25)
    assert ds.media_calibration("night", "s1", camera="right") == pytest.approx(0.5)


def test_cameras_that_differ_ask_for_a_camera(tmp_path: Path) -> None:
    ds = _two_camera_dataset(tmp_path)
    _ = ds.set_media_calibration(0.25, recording="night", camera="left")

    with pytest.raises(ValueError, match="camera="):
        _ = ds.media_calibration("night", "s1")


def test_cameras_that_agree_read_as_one_scale(tmp_path: Path) -> None:
    ds = _two_camera_dataset(tmp_path)
    _ = ds.set_media_calibration(0.25, recording="night")

    assert ds.media_calibration("night", "s1") == pytest.approx(0.25)


def test_two_scales_within_one_camera_refuse(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "dataset")
    write_media_index(
        ds,
        [
            MediaClip(sequence="sess", filename="a.mp4", video_order=0),
            MediaClip(sequence="sess", filename="b.mp4", video_order=1),
        ],
    )
    path = ds.get_root("media_raw") / "index.csv"
    frame = load_media_index_frame(path)
    frame["cm_per_pixel"] = ["0.25", "0.5"]
    write_media_index_rows(path, frame)

    with pytest.raises(ValueError, match="2 different cm_per_pixel"):
        _ = ds.media_calibration("", "sess")


def test_a_recording_with_a_group_or_sequence_refuses(tmp_path: Path) -> None:
    ds = _two_camera_dataset(tmp_path)

    with pytest.raises(ValueError, match="recording="):
        _ = ds.set_media_calibration(0.25, recording="night", group="night")


@pytest.mark.media
def test_a_later_store_of_a_camera_takes_its_recording_s_calibration(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """By camera, a store recorded later divides into the calibrated recording."""
    stores = tmp_path / "stores"
    make = video_store_maker((5, 5, 2), fps=4.0, created_utc=DIVISION_CREATED_UTC)
    later = video_store_maker((5, 5, 2), fps=4.0, created_utc="2021-02-19T20:22:53")
    first, _ = make(name="a", parent=stores, extra_metadata={"camera_serial": "123"})
    second, _ = later(name="b", parent=stores, extra_metadata={"camera_serial": "123"})
    ds = make_media_dataset(tmp_path / "dataset")
    division = MediaDivision(frames=10, identity="camera")
    _ = ds.write_media_index(
        [MediaIndexScope(directory=first, group="", sequence="", division=division)]
    )
    assert ds.set_media_calibration(0.25, recording="123") == 2

    _ = ds.write_media_index(
        [MediaIndexScope(directory=second, group="", sequence="", division=division)]
    )

    rows = ds.read_media_index()
    assert [row["group"] for row in rows] == ["123"] * 4
    assert [row["cm_per_pixel"] for row in rows] == ["0.25"] * 4


def _scan_entries(ds: Dataset, *entries: tuple[str, str]) -> None:
    """Write one clip per entry under ``media_raw/<group>__<sequence>`` and scan."""
    root = ds.get_root("media_raw")
    for shade, (group, sequence) in enumerate(entries):
        directory = root / f"{group}__{sequence}"
        directory.mkdir(parents=True)
        write_h264_mp4(directory / f"{sequence}.mp4", frames=3, shade=40 * shade)
    _ = ds.index_media([root], extensions=(".mp4",), media_layout="per_sequence")


@pytest.mark.media
def test_a_new_member_of_a_continuous_group_takes_its_calibration(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """A rescan finds ``night:s2``, which takes its recording's scale."""
    ds = make_media_dataset(tmp_path / "dataset")
    _scan_entries(ds, ("night", "s1"), ("arena", "s1"))
    ds.set_continuous_groups(["night"])
    _ = ds.set_media_calibration(0.25, recording="night")
    _ = ds.set_media_calibration(0.75, recording="arena:s1")

    _scan_entries(ds, ("night", "s2"), ("arena", "s2"))

    assert ds.media_calibration("night", "s2") == pytest.approx(0.25)
    assert ds.media_calibration("arena", "s1") == pytest.approx(0.75)
    assert ds.media_calibration("arena", "s2") is None
