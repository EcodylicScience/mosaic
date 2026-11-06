"""The YOLO image-suffix set and the image-to-label path rule, as Ultralytics applies them."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.annotations.yolo_layout import (
    YOLO_IMAGE_SUFFIXES,
    label_path_for_image,
)


def test_image_suffixes_are_the_union_of_both_trainers_formats() -> None:
    """Ultralytics 8.4.121 (train-pose) and the POLO fork's 8.4.84 (train-points)."""
    pose = "bmp dng jpeg jpg mpo png tif tiff webp pfm heic"
    points = "avif bmp dng heic heif jp2 jpeg jpg mpo png tif tiff webp"
    assert YOLO_IMAGE_SUFFIXES == frozenset(
        f".{name}" for name in (pose + " " + points).split()
    )


def test_label_path_split_directory_inside_images_parent(tmp_path: Path) -> None:
    image = tmp_path / "ds" / "train" / "images" / "x.jpg"
    assert label_path_for_image(image) == (
        tmp_path.resolve() / "ds" / "train" / "labels" / "x.txt"
    )


def test_label_path_images_directory_above_split(tmp_path: Path) -> None:
    image = tmp_path / "ds" / "images" / "train" / "x.jpg"
    assert label_path_for_image(image) == (
        tmp_path.resolve() / "ds" / "labels" / "train" / "x.txt"
    )


def test_label_path_replaces_only_the_last_images_part(tmp_path: Path) -> None:
    image = tmp_path / "images" / "ds" / "images" / "x.png"
    assert label_path_for_image(image) == (
        tmp_path.resolve() / "images" / "ds" / "labels" / "x.txt"
    )


def test_label_path_without_images_part_swaps_only_the_suffix(tmp_path: Path) -> None:
    image = tmp_path / "ds" / "frames" / "x.jpg"
    assert label_path_for_image(image) == tmp_path.resolve() / "ds" / "frames" / "x.txt"


def test_label_path_matches_only_whole_images_parts(tmp_path: Path) -> None:
    image = tmp_path / "my_images" / "x.jpg"
    assert label_path_for_image(image) == tmp_path.resolve() / "my_images" / "x.txt"


def test_label_path_keeps_the_stem_as_written(tmp_path: Path) -> None:
    image = tmp_path / "images" / "Frame.v2.JPG"
    assert label_path_for_image(image) == tmp_path.resolve() / "labels" / "Frame.v2.txt"


def test_label_path_stays_beside_a_linked_image(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    target = frames / "x.jpg"
    target.touch()
    images = tmp_path / "ds" / "images" / "train"
    images.mkdir(parents=True)
    link = images / "x.jpg"
    link.symlink_to(target)
    assert label_path_for_image(link) == (
        tmp_path.resolve() / "ds" / "labels" / "train" / "x.txt"
    )


def test_label_path_resolves_a_relative_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert label_path_for_image(Path("images/x.jpg")) == (
        tmp_path.resolve() / "labels" / "x.txt"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("x.jpg", True),
        ("x.JPG", True),
        ("x.Png", True),
        ("x.heic", True),
        ("x.gif", False),
        ("x.txt", False),
        ("x", False),
    ],
)
def test_suffix_check_is_case_insensitive_like_ultralytics(
    name: str, expected: bool
) -> None:
    assert (Path(name).suffix.lower() in YOLO_IMAGE_SUFFIXES) is expected
