"""``discover_datasets``: the datasets below a directory, found by their manifests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mosaic.core.dataset_discovery import (
    SKIPPED_DIRECTORIES,
    DiscoveredDataset,
    discover_datasets,
)
from mosaic.core.manifest import MANIFEST_VERSION
from tests.helpers import make_dataset


def _names(found: list[DiscoveredDataset]) -> list[str | None]:
    return [dataset.name for dataset in found]


def test_a_dataset_is_found_down_to_the_depth_limit(tmp_path: Path) -> None:
    _ = make_dataset(tmp_path / "a", name="one")
    _ = make_dataset(tmp_path / "x" / "b", name="two")
    _ = make_dataset(tmp_path / "x" / "y" / "c", name="three")
    _ = make_dataset(tmp_path / "x" / "y" / "z" / "d", name="four")

    assert _names(discover_datasets(tmp_path)) == ["one", "two", "three"]
    assert _names(discover_datasets(tmp_path, max_depth=1)) == ["one"]


def test_the_root_itself_is_checked(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="root")

    found = discover_datasets(tmp_path / "ds", max_depth=0)

    assert found == [
        DiscoveredDataset(
            manifest=str(ds.manifest_path.absolute()), uuid=ds.uuid, name="root"
        )
    ]


def test_hidden_directories_and_the_roots_of_a_dataset_are_not_entered(
    tmp_path: Path,
) -> None:
    _ = make_dataset(tmp_path / ".cache" / "ds", name="hidden")
    _ = make_dataset(tmp_path / "project" / "features" / "ds", name="in-features")
    _ = make_dataset(tmp_path / "project" / "_tracking" / "ds", name="in-tracking")
    _ = make_dataset(tmp_path / "project" / "library", name="nested-library")
    _ = make_dataset(tmp_path / "project", name="project")

    assert _names(discover_datasets(tmp_path)) == ["project", "nested-library"]
    assert {"_tracking", "features", "media", "models", "tracks"} <= SKIPPED_DIRECTORIES


def test_a_folder_named_as_a_root_outside_a_dataset_is_entered(tmp_path: Path) -> None:
    _ = make_dataset(tmp_path / "labels" / "ds", name="under-labels")
    _ = make_dataset(tmp_path / "media" / "ds", name="under-media")

    assert _names(discover_datasets(tmp_path)) == ["under-labels", "under-media"]


def test_a_file_under_a_manifest_name_that_another_tool_wrote_is_passed_over(
    tmp_path: Path,
) -> None:
    yolo = tmp_path / "yolo" / "dataset.yaml"
    (yolo.parent / "labels" / "ds").mkdir(parents=True)
    _ = yolo.write_text(
        "path: .\ntrain: images/train\nkpt_shape: [3, 3]\n", encoding="utf-8"
    )
    _ = make_dataset(yolo.parent / "labels" / "ds", name="inside-yolo")
    medical = tmp_path / "medical" / "dataset.json"
    medical.parent.mkdir()
    _ = medical.write_text('{"version": "1.0", "files": []}', encoding="utf-8")

    assert _names(discover_datasets(tmp_path)) == ["inside-yolo"]


def test_a_version_one_manifest_is_found(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy" / "dataset.yaml"
    legacy.parent.mkdir()
    _ = legacy.write_text(
        "name: legacy\nroots:\n  features: features\n", encoding="utf-8"
    )

    found = discover_datasets(tmp_path)

    assert found == [DiscoveredDataset(manifest=str(legacy), name="legacy")]


def test_the_roots_a_dataset_declares_are_not_entered(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "project", name="project")
    ds.manifest.roots["features"] = "out/features"
    ds.save()
    _ = make_dataset(tmp_path / "project" / "out" / "features" / "ds", name="in-out")

    assert _names(discover_datasets(tmp_path, max_depth=5)) == ["project"]


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs a directory that its owner cannot read",
)
def test_a_directory_that_cannot_be_read_is_passed_over(tmp_path: Path) -> None:
    _ = make_dataset(tmp_path / "readable", name="readable")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        found = discover_datasets(tmp_path)
    finally:
        locked.chmod(0o755)

    assert _names(found) == ["readable"]


def test_a_manifest_that_does_not_read_is_reported_with_its_error(
    tmp_path: Path,
) -> None:
    newer = tmp_path / "newer" / "dataset.yaml"
    newer.parent.mkdir()
    _ = newer.write_text(
        f"name: future\nmanifest_version: {MANIFEST_VERSION + 1}\n", encoding="utf-8"
    )
    broken = tmp_path / "broken" / "dataset.yaml"
    broken.parent.mkdir()
    _ = broken.write_text("name: [unclosed\n", encoding="utf-8")

    found = discover_datasets(tmp_path)

    assert [Path(dataset.manifest).parent.name for dataset in found] == [
        "broken",
        "newer",
    ]
    assert all(dataset.error and dataset.name is None for dataset in found)


def test_a_symbolic_link_is_followed_once(tmp_path: Path) -> None:
    _ = make_dataset(tmp_path / "data" / "ds", name="linked")
    (tmp_path / "alias").symlink_to(tmp_path / "data", target_is_directory=True)
    (tmp_path / "data" / "loop").symlink_to(tmp_path, target_is_directory=True)

    assert _names(discover_datasets(tmp_path, max_depth=6)) == ["linked"]


def test_a_root_that_is_not_a_directory_or_a_negative_depth_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(NotADirectoryError):
        _ = discover_datasets(tmp_path / "absent")
    with pytest.raises(ValueError, match="max_depth"):
        _ = discover_datasets(tmp_path, max_depth=-1)
