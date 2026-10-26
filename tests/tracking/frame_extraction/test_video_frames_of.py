"""The recorded frame behind each annotated image.

An image that a frames run wrote takes the entry and camera of the frames-index
row whose directory contains it. Its frame index is the one in the run's
``run_info.json``, and its origin is the uuid of the dataset whose index lists
it. Any other image falls back to its revision's origin, the sequence its
revision recorded and that frame's index.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mosaic.core.annotations.model import AnnotationFrame
from mosaic.core.annotations.split import VideoFrame, frame_key
from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.manifest import LibraryLink
from mosaic.tracking.frame_extraction.frame_runs import (
    frame_run_files,
    video_frames_of,
)
from tests.helpers import (
    assert_tree_unchanged,
    extract_entry_camera,
    make_dataset,
    tree_snapshot,
)

RUN_ID = "run0"
SIZES = {100: (64, 48), 250: (64, 48)}
REVISION_ORIGIN = "revision-origin"


def _frame(image: Path, *, video: str = "", frame_index: int = -1) -> AnnotationFrame:
    return AnnotationFrame(
        image_path=image, width=64, height=48, video=video, frame_index=frame_index
    )


def _extract(ds: Dataset, *, camera: str = "cam1") -> dict[int, Path]:
    return extract_entry_camera(
        ds, run_id=RUN_ID, group="G", sequence="S", camera=camera, sizes=SIZES
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def test_an_extracted_image_takes_its_entry_camera_and_run_index(
    dataset: Dataset,
) -> None:
    """The revision's sequence and index do not override the extraction's."""
    images = _extract(dataset)
    served = {file.image for file in frame_run_files(dataset, RUN_ID, "G", "S", "cam1")}

    (located,) = video_frames_of(
        dataset,
        dataset.base_dir,
        [_frame(images[250], video="renamed", frame_index=7)],
        revision_origin=REVISION_ORIGIN,
    )

    assert images[250].as_posix() in served
    assert located == VideoFrame(
        origin_uuid=dataset.uuid or "",
        group="G",
        sequence="S",
        camera="cam1",
        frame_index=250,
    )


def test_an_imported_frame_without_an_index_gets_the_run_index(
    dataset: Dataset,
) -> None:
    images = _extract(dataset)

    (located,) = video_frames_of(
        dataset,
        dataset.base_dir,
        [_frame(images[100])],
        revision_origin=REVISION_ORIGIN,
    )

    assert located.frame_index == 100
    assert frame_key(located).endswith(":cam1:100")


def test_an_absolute_image_path_inside_the_dataset_is_matched(
    dataset: Dataset,
) -> None:
    images = _extract(dataset)

    (located,) = video_frames_of(
        dataset,
        dataset.base_dir,
        [_frame(dataset.base_dir / images[100])],
        revision_origin=REVISION_ORIGIN,
    )

    assert (located.group, located.camera, located.frame_index) == ("G", "cam1", 100)


def test_a_legacy_absolute_row_locates_the_frame_as_a_relative_row_does(
    tmp_path: Path,
) -> None:
    relative = make_dataset(tmp_path / "relative")
    legacy = make_dataset(tmp_path / "legacy")
    relative_images = _extract(relative)
    legacy_images = extract_entry_camera(
        legacy,
        run_id=RUN_ID,
        group="G",
        sequence="S",
        camera="cam1",
        sizes=SIZES,
        stored_absolute=True,
    )

    from_relative = video_frames_of(
        relative,
        relative.base_dir,
        [_frame(relative_images[100])],
        revision_origin=REVISION_ORIGIN,
    )
    from_legacy = video_frames_of(
        legacy,
        legacy.base_dir,
        [_frame(legacy_images[100])],
        revision_origin=REVISION_ORIGIN,
    )

    assert legacy_images == relative_images
    assert from_legacy[0].origin_uuid == legacy.uuid
    assert [replace(frame, origin_uuid="") for frame in from_legacy] == [
        replace(frame, origin_uuid="") for frame in from_relative
    ]
    assert (from_legacy[0].group, from_legacy[0].camera) == ("G", "cam1")
    assert from_legacy[0].frame_index == 100


def test_frames_come_back_one_per_input_in_input_order(dataset: Dataset) -> None:
    left = _extract(dataset, camera="left")
    right = _extract(dataset, camera="right")
    other = extract_entry_camera(
        dataset, run_id="run1", group="H", sequence="S", camera="", sizes={7: (64, 48)}
    )
    frames = [
        _frame(right[250]),
        _frame(Path("loose/a.png"), video="v", frame_index=3),
        _frame(left[100]),
        _frame(other[7]),
        _frame(right[250]),
    ]

    located = video_frames_of(
        dataset, dataset.base_dir, frames, revision_origin=REVISION_ORIGIN
    )

    assert [
        (frame.group, frame.sequence, frame.camera, frame.frame_index)
        for frame in located
    ] == [
        ("G", "S", "right", 250),
        ("", "v", "", 3),
        ("G", "S", "left", 100),
        ("H", "S", "", 7),
        ("G", "S", "right", 250),
    ]


def test_an_image_under_a_linked_library_takes_the_library_uuid(
    tmp_path: Path,
) -> None:
    project = make_dataset(tmp_path / "project")
    library = make_dataset(tmp_path / "library")
    images = _extract(library)
    _ = project.add_library(LibraryLink(id="shared", path="../library"))

    (located,) = video_frames_of(
        project,
        library.base_dir,
        [_frame(images[100])],
        revision_origin=project.uuid or "",
    )

    assert located == VideoFrame(
        origin_uuid=library.uuid or "",
        group="G",
        sequence="S",
        camera="cam1",
        frame_index=100,
    )


def test_an_image_of_a_dataset_that_is_not_linked_falls_back(tmp_path: Path) -> None:
    """Only this dataset's frames index and its linked libraries' are read."""
    project = make_dataset(tmp_path / "project")
    elsewhere = make_dataset(tmp_path / "elsewhere")
    images = _extract(elsewhere)

    (located,) = video_frames_of(
        project,
        elsewhere.base_dir,
        [_frame(images[100], video="S", frame_index=100)],
        revision_origin=REVISION_ORIGIN,
    )

    assert located == VideoFrame(
        origin_uuid=REVISION_ORIGIN,
        group="",
        sequence="S",
        camera="",
        frame_index=100,
    )


@pytest.mark.parametrize(
    ("video", "frame_index", "sequence", "key"),
    [
        ("", -1, "coco", f"{REVISION_ORIGIN}::coco::@img_7"),
        ("rec1", 12, "rec1", f"{REVISION_ORIGIN}::rec1::12"),
    ],
    ids=["no-sequence-no-index", "recorded"],
)
def test_a_foreign_image_falls_back_to_what_its_revision_recorded(
    dataset: Dataset, video: str, frame_index: int, sequence: str, key: str
) -> None:
    """The sequence is the recorded one, else the image's directory name."""
    _ = _extract(dataset)

    (located,) = video_frames_of(
        dataset,
        dataset.base_dir,
        [_frame(Path("imports/coco/img_7.png"), video=video, frame_index=frame_index)],
        revision_origin=REVISION_ORIGIN,
    )

    assert located == VideoFrame(
        origin_uuid=REVISION_ORIGIN,
        group="",
        sequence=sequence,
        camera="",
        frame_index=frame_index,
        image_stem="img_7" if frame_index == -1 else "",
    )
    assert frame_key(located) == key


def test_an_image_the_run_did_not_write_takes_its_entry_and_no_index(
    dataset: Dataset,
) -> None:
    """The extraction's directory names the entry, and its records lack the index."""
    images = _extract(dataset)
    stray = images[100].parent / "stray.png"

    (located,) = video_frames_of(
        dataset,
        dataset.base_dir,
        [_frame(stray, video="S", frame_index=5)],
        revision_origin=REVISION_ORIGIN,
    )

    assert located == VideoFrame(
        origin_uuid=dataset.uuid or "",
        group="G",
        sequence="S",
        camera="cam1",
        frame_index=-1,
        image_stem="stray",
    )


def test_a_read_only_dataset_gains_no_directory(dataset: Dataset) -> None:
    images = _extract(dataset)
    base = dataset.base_dir
    reopened = open_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    located = video_frames_of(
        reopened,
        base,
        [_frame(images[100]), _frame(Path("imports/x.png"))],
        revision_origin=REVISION_ORIGIN,
    )

    assert [frame.camera for frame in located] == ["cam1", ""]
    assert_tree_unchanged(base, before)


def test_a_read_only_dataset_without_a_frames_root_gains_no_directory(
    tmp_path: Path,
) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    reopened = open_dataset(base, ensure_roots=False)
    before = tree_snapshot(base)

    (located,) = video_frames_of(
        reopened,
        base,
        [_frame(Path("media/frames/uniform/run0/S/frame_000004.png"), frame_index=4)],
        revision_origin=REVISION_ORIGIN,
    )

    assert located == VideoFrame(
        origin_uuid=REVISION_ORIGIN,
        group="",
        sequence="S",
        camera="",
        frame_index=4,
    )
    assert_tree_unchanged(base, before)
