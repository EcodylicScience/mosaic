"""The catalog of example datasets is well formed.

Each entry names a pinned Hugging Face revision, the archives it fetches, and the
notebook that uses it. A malformed entry would fail only when a user fetches it, so
the shape is checked here without the network.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

from mosaic.examples.catalog import CATALOG, Example
from tests.helpers import REPO_ROOT

NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def test_the_catalog_lists_the_five_examples_in_order() -> None:
    assert [example.name for example in CATALOG] == [
        "calms21",
        "shiners",
        "zebrafish",
        "calms21-pose",
        "shiners-polo",
    ]


def test_names_are_unique_lowercase_with_hyphens() -> None:
    names = [example.name for example in CATALOG]

    assert len(set(names)) == len(names)
    assert all(NAME_PATTERN.fullmatch(name) for name in names)


@pytest.mark.parametrize("example", CATALOG, ids=lambda example: example.name)
class TestEachEntry:
    def test_the_repository_is_named_after_the_entry(self, example: Example) -> None:
        assert example.repository == f"EcodylicScience/mosaic-example-{example.name}"

    def test_the_revision_is_a_commit_hash(self, example: Example) -> None:
        assert REVISION_PATTERN.fullmatch(example.revision)

    def test_license_and_summary_are_present(self, example: Example) -> None:
        assert example.license.strip()
        assert example.summary.strip()

    def test_the_notebook_is_a_file_in_notebooks(self, example: Example) -> None:
        assert (REPO_ROOT / "notebooks" / example.notebook).is_file()

    def test_file_names_are_unique_within_the_entry(self, example: Example) -> None:
        files = [archive.file for archive in example.archives]

        assert files
        assert len(set(files)) == len(files)

    def test_each_archive_unpacks_to_one_path_component(self, example: Example) -> None:
        for archive in example.archives:
            parts = PurePosixPath(archive.unpacks_to).parts
            assert len(parts) == 1
            assert parts[0] not in {"..", "."}
            assert not PurePosixPath(archive.unpacks_to).is_absolute()

    def test_each_archive_has_a_positive_size(self, example: Example) -> None:
        assert all(archive.download_bytes > 0 for archive in example.archives)

    def test_exactly_one_archive_is_the_dataset(self, example: Example) -> None:
        datasets = [archive for archive in example.archives if archive.dataset]

        assert len(datasets) == 1
        assert example.dataset_archive is datasets[0]

    def test_the_manifest_lies_in_the_dataset_directory(self, example: Example) -> None:
        directory = example.dataset_archive.unpacks_to

        assert example.manifest_relative == f"{directory}/dataset.yaml"


def test_an_example_downloads_the_sum_of_its_archives() -> None:
    by_name = {example.name: example for example in CATALOG}

    assert by_name["shiners"].download_bytes == 349_818_205 + 8_942_902
    assert by_name["calms21"].download_bytes == 12_613_112


def test_only_calms21_needs_unpickling() -> None:
    needing = {example.name for example in CATALOG if example.raw_needs_unpickling}

    assert needing == {"calms21"}


def test_the_listed_sizes_and_layouts_match_the_published_archives() -> None:
    by_name = {example.name: example for example in CATALOG}

    shiners = by_name["shiners"]
    assert [
        (a.file, a.unpacks_to, a.download_bytes, a.dataset) for a in shiners.archives
    ] == [
        ("shiners-tracks.tar.gz", "shiners-tracks", 349_818_205, True),
        ("shiners-raw-sample.tar.gz", "shiners-raw-sample", 8_942_902, False),
    ]
    assert by_name["calms21-pose"].dataset_archive.download_bytes == 1_940_249_616
    assert by_name["calms21"].revision == "ee1eacd1febed2e2c89a332237d668548f16836a"
