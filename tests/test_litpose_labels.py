"""Exporting an annotation set as a Lightning Pose project.

Plain text on both sides, so unlike the SLEAP export this is fully checkable
here. The interesting cases are the two places the representation says more than
CollectedData can hold: an instance axis, and the difference between a keypoint
at the origin and one that was never placed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from mosaic.core.annotations import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)
from mosaic.tracking.litpose.labels import write_litpose_dataset

SCHEMA = KeypointSchema(names=("nose", "thorax", "tail"))


def _one(*points: Keypoint) -> AnnotationObject:
    return AnnotationObject(keypoints=points)


def _set(tmp_path: Path, *frames: AnnotationFrame) -> AnnotationSet:
    return AnnotationSet(schema=SCHEMA, frames=frames, image_root=tmp_path / "images")


def _frame(
    name: str, *objects: AnnotationObject, video: str = "clipA"
) -> AnnotationFrame:
    return AnnotationFrame(
        image_path=Path(name), width=160, height=120, objects=objects, video=video
    )


def _rows(out: Path) -> list[list[str]]:
    with (out / "CollectedData.csv").open() as handle:
        return list(csv.reader(handle))


def test_the_header_is_deeplabcuts_three_rows(tmp_path: Path) -> None:
    placed = _one(Keypoint(1.0, 2.0, 2), Keypoint(3.0, 4.0, 2), Keypoint(5.0, 6.0, 2))
    out = write_litpose_dataset(
        _set(tmp_path, _frame("a.png", placed)), tmp_path / "project"
    )
    rows = _rows(out)
    assert rows[0] == ["scorer"] + ["mosaic"] * 6
    assert rows[1] == ["bodyparts", "nose", "nose", "thorax", "thorax", "tail", "tail"]
    assert rows[2] == ["coords", "x", "y", "x", "y", "x", "y"]


def test_a_placed_keypoint_is_written_in_native_pixels(tmp_path: Path) -> None:
    placed = _one(
        Keypoint(40.5, 50.25, 2), Keypoint(52.0, 56.0, 1), Keypoint(1.0, 2.0, 2)
    )
    out = write_litpose_dataset(
        _set(tmp_path, _frame("a.png", placed)), tmp_path / "project"
    )
    row = _rows(out)[3]
    assert row[0] == "labeled-data/clipA/a.png"
    assert row[1:3] == ["40.500000", "50.250000"]
    assert row[3:5] == ["52.000000", "56.000000"], "occluded is still placed"


def test_an_unplaced_keypoint_is_an_empty_cell_not_a_zero(tmp_path: Path) -> None:
    """A zero there is a keypoint at the image origin, which is a different claim."""
    partial = _one(Keypoint(10.0, 20.0, 2), Keypoint.absent(), Keypoint(30.0, 40.0, 2))
    out = write_litpose_dataset(
        _set(tmp_path, _frame("a.png", partial)), tmp_path / "project"
    )
    row = _rows(out)[3]
    assert row[3:5] == ["", ""], "absent, not at the origin"
    assert row[1:3] == ["10.000000", "20.000000"]


def test_a_multi_instance_frame_is_refused_by_name(tmp_path: Path) -> None:
    """CollectedData has no instance axis, so taking the first would discard work."""
    two = _frame(
        "crowded.png",
        _one(*[Keypoint(1.0, 1.0, 2)] * 3),
        _one(*[Keypoint(2.0, 2.0, 2)] * 3),
    )
    with pytest.raises(ValueError, match="crowded.png"):
        _ = write_litpose_dataset(_set(tmp_path, two), tmp_path / "project")


def test_the_config_describes_the_dataset_it_was_written_for(tmp_path: Path) -> None:
    placed = _one(*[Keypoint(1.0, 1.0, 2)] * 3)
    out = write_litpose_dataset(
        _set(tmp_path, _frame("a.png", placed)), tmp_path / "project", train_prob=0.75
    )
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["data"]["num_keypoints"] == 3
    assert config["data"]["keypoint_names"] == ["nose", "thorax", "tail"]
    assert config["data"]["image_orig_dims"] == {"height": 120, "width": 160}
    assert config["data"]["csv_file"] == "CollectedData.csv"
    assert config["training"] == {"train_prob": 0.75, "val_prob": 0.25}


def test_images_land_under_their_video(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _ = (images / "a.png").write_bytes(b"png")
    placed = _one(*[Keypoint(1.0, 1.0, 2)] * 3)
    out = write_litpose_dataset(
        _set(tmp_path, _frame("a.png", placed, video="session3")), tmp_path / "project"
    )
    assert (out / "labeled-data" / "session3" / "a.png").exists()


def test_an_empty_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        _ = write_litpose_dataset(AnnotationSet(schema=SCHEMA), tmp_path / "project")
