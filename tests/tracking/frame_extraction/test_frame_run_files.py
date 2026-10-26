"""The typed reader of an extraction run's frame files.

The reader serves each frame's dataset-relative image key and size from the
entry camera's ``run_info.json``. It reads the frames-index row's directory in
either stored form, and creates nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.scope import Scope
from mosaic.tracking.frame_extraction.frame_runs import (
    FrameFile,
    frame_run_files,
    frames_index_path,
    list_frame_runs,
)
from tests.helpers import make_dataset, record_frames_row, write_frames_run

RUN_ID = "run0"
SIZES = {100: (64, 48), 250: (32, 24), 40: (16, 12)}
EXPECTED_FRAME_INDICES = sorted(SIZES)


def _write_run(directory: Path) -> None:
    """A camera directory whose records are listed out of frame order."""
    write_frames_run(directory, SIZES)


def _record(ds: Dataset, stored: str) -> None:
    record_frames_row(ds, stored, run_id=RUN_ID, group="G", sequence="S", camera="cam1")


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _entry_directory(ds: Dataset) -> Path:
    return ds.get_root("frames") / "uniform" / RUN_ID / "G__S" / "cam1"


def _relative_directory(ds: Dataset) -> str:
    return ds.relative_to_root(_entry_directory(ds))


def test_files_come_back_sorted_by_frame_index_with_their_sizes(
    dataset: Dataset,
) -> None:
    _write_run(_entry_directory(dataset))
    _record(dataset, _relative_directory(dataset))

    files = frame_run_files(dataset, RUN_ID, "G", "S", "cam1")

    assert [file.frame_index for file in files] == EXPECTED_FRAME_INDICES
    assert [(file.width, file.height) for file in files] == [
        SIZES[index] for index in EXPECTED_FRAME_INDICES
    ]
    assert all(isinstance(file, FrameFile) for file in files)


def test_image_is_the_root_relative_directory_joined_with_the_file_name(
    dataset: Dataset,
) -> None:
    _write_run(_entry_directory(dataset))
    _record(dataset, _relative_directory(dataset))

    files = frame_run_files(dataset, RUN_ID, "G", "S", "cam1")

    assert files[1].image == (
        f"media/frames/uniform/{RUN_ID}/G__S/cam1/frame_000100.png"
    )
    for file in files:
        assert not file.image.startswith("/")
        assert str(dataset.manifest_path.parent) not in file.image
        assert "\\" not in file.image


def test_an_absolute_legacy_row_gives_the_same_images_and_creates_no_directory(
    dataset: Dataset, tmp_path: Path
) -> None:
    _write_run(_entry_directory(dataset))
    _record(dataset, _relative_directory(dataset))
    relative_images = [
        file.image for file in frame_run_files(dataset, RUN_ID, "G", "S", "cam1")
    ]

    legacy = make_dataset(tmp_path / "legacy")
    _write_run(_entry_directory(legacy))
    _record(legacy, str(_entry_directory(legacy)))
    before = sorted(legacy.manifest_path.parent.rglob("*"))

    legacy_images = [
        file.image for file in frame_run_files(legacy, RUN_ID, "G", "S", "cam1")
    ]

    assert legacy_images == relative_images
    assert sorted(legacy.manifest_path.parent.rglob("*")) == before


def test_an_absolute_cell_from_another_machine_resolves_through_the_path_map(
    dataset: Dataset, tmp_path: Path
) -> None:
    _write_run(_entry_directory(dataset))
    _record(dataset, _relative_directory(dataset))
    expected = [
        file.image for file in frame_run_files(dataset, RUN_ID, "G", "S", "cam1")
    ]

    moved = make_dataset(tmp_path / "moved")
    _write_run(_entry_directory(moved))
    old_base = "/old/machine/ds"
    base = str(moved.manifest_path.parent)
    # The row refuses a directory that does not exist, so the old machine's cell
    # is written into the index text after the row is recorded.
    _record(moved, str(_entry_directory(moved)))
    index_path = frames_index_path(moved, "uniform")
    index_path.write_text(index_path.read_text().replace(base, old_base))
    moved.remap_roots({old_base: str(moved.manifest_path.parent)})

    files = frame_run_files(moved, RUN_ID, "G", "S", "cam1")

    assert [file.image for file in files] == expected


def test_a_directory_outside_the_dataset_tree_raises_naming_run_and_directory(
    dataset: Dataset, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere" / "cam1"
    _write_run(outside)
    _record(dataset, str(outside))

    with pytest.raises(ValueError) as raised:
        frame_run_files(dataset, RUN_ID, "G", "S", "cam1")

    assert RUN_ID in str(raised.value)
    assert str(outside.resolve()) in str(raised.value)


def test_an_unknown_run_or_entry_answers_nothing(dataset: Dataset) -> None:
    _write_run(_entry_directory(dataset))
    _record(dataset, _relative_directory(dataset))

    assert frame_run_files(dataset, "other", "G", "S", "cam1") == ()
    assert frame_run_files(dataset, RUN_ID, "G", "S", "cam2") == ()
    assert frame_run_files(dataset, RUN_ID, "G", "T", "cam1") == ()


def test_a_dataset_with_no_frames_index_answers_nothing(dataset: Dataset) -> None:
    assert frame_run_files(dataset, RUN_ID, "G", "S", "cam1") == ()


def test_a_read_only_dataset_gains_no_directory(dataset: Dataset) -> None:
    _write_run(_entry_directory(dataset))
    _record(dataset, _relative_directory(dataset))
    base = dataset.manifest_path.parent
    reopened = open_dataset(base, ensure_roots=False)
    before = sorted(base.rglob("*"))

    files = frame_run_files(reopened, RUN_ID, "G", "S", "cam1")

    assert len(files) == len(SIZES)
    assert sorted(base.rglob("*")) == before


def test_a_read_only_dataset_without_a_frames_root_gains_no_directory(
    tmp_path: Path,
) -> None:
    base = tmp_path / "ro"
    _ = make_dataset(base, ensure_roots=False)
    reopened = open_dataset(base, ensure_roots=False)
    before = sorted(base.rglob("*"))

    files = frame_run_files(reopened, RUN_ID, "G", "S", "cam1")

    assert files == ()
    assert sorted(base.rglob("*")) == before


def test_an_extraction_run_serves_the_files_it_wrote(
    scenario_dataset_with_media: Dataset,
) -> None:
    from mosaic.tracking import extract_frames

    ds = scenario_dataset_with_media
    _ = extract_frames(
        ds, n_frames=2, method="uniform", scope=Scope(entries=[("", "seq_a")])
    )
    row = list_frame_runs(ds, method="uniform").iloc[0]

    files = frame_run_files(ds, str(row["run_id"]), "", "seq_a", str(row["camera"]))

    assert len(files) == 2
    assert [file.frame_index for file in files] == sorted(
        file.frame_index for file in files
    )
    for file in files:
        assert file.image.startswith("media/frames/uniform/")
        assert (ds.manifest_path.parent / file.image).is_file()
