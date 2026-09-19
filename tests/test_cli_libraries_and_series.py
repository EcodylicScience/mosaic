"""The two command-line gestures a shared model library is built from.

``mosaic libraries`` links the dataset a model resolves from, and ``mosaic sources
add --series`` claims the exact annotation revisions a library trains on.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    write_series_revision,
)
from tests.helpers import make_dataset

runner = CliRunner()


def _project_with_revisions(tmp_path: Path, count: int) -> Dataset:
    project = make_dataset(tmp_path / "52", name="project")
    for number in range(count):
        _ = write_series_revision(
            project,
            series="keypoints",
            key="17-mice",
            payload=json.dumps({"state": number}).encode(),
            origin={},
            n_records=number,
        )
    return project


def test_libraries_add_list_and_remove(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")
    manifest = str(project.manifest_path)

    added = runner.invoke(
        app,
        [
            "libraries",
            "add",
            "-m",
            manifest,
            "--id",
            "group",
            "--path",
            "../libraries/7",
        ],
    )
    assert added.exit_code == 0, added.output

    listed = runner.invoke(app, ["libraries", "list", "-m", manifest, "--json"])
    assert json.loads(listed.output)["libraries"] == [
        {
            "id": "group",
            "path": "../libraries/7",
            "uuid": library.uuid,
            "added_at": json.loads(listed.output)["libraries"][0]["added_at"],
        }
    ]

    removed = runner.invoke(
        app, ["libraries", "remove", "-m", manifest, "--id", "group"]
    )
    assert removed.exit_code == 0, removed.output
    assert Dataset(manifest_path=project.manifest_path).load().libraries == ()


def test_linking_a_path_that_holds_no_dataset_fails_clearly(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")

    result = runner.invoke(
        app,
        [
            "libraries",
            "add",
            "-m",
            str(project.manifest_path),
            "--id",
            "g",
            "--path",
            "../nope",
        ],
    )

    assert result.exit_code != 0
    assert "no dataset manifest" in result.output


def test_a_series_source_is_declared_and_scanned_from_the_cli(tmp_path: Path) -> None:
    project = _project_with_revisions(tmp_path, 3)
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    manifest = str(library.manifest_path)
    set_dir = str(project.get_root("labels_raw") / "keypoints" / "17-mice")

    declared = runner.invoke(
        app,
        [
            "sources", "add", "-m", manifest, "--kind", "labels", "--id", "p52-17",
            "--path", set_dir, "--series", "keypoints",
            "--file", "rev2/annotations.coco.json",
        ],
    )  # fmt: skip
    assert declared.exit_code == 0, declared.output

    scanned = runner.invoke(app, ["scan", "-m", manifest, "--kind", "labels"])
    assert scanned.exit_code == 0, scanned.output

    rows = read_label_series(
        Dataset(manifest_path=library.manifest_path).load(), "keypoints"
    )
    assert [int(revision) for revision in rows["revision"]] == [2]


def test_series_refuses_the_flags_that_describe_a_label_file(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")

    result = runner.invoke(
        app,
        [
            "sources", "add", "-m", str(library.manifest_path), "--kind", "labels",
            "--path", "/x", "--series", "keypoints", "--src-format", "boris_aggregated_csv",
        ],
    )  # fmt: skip

    assert result.exit_code != 0
    assert "a series fixes what its files are" in result.output


def test_series_belongs_to_labels_only(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")

    result = runner.invoke(
        app,
        [
            "sources", "add", "-m", str(library.manifest_path), "--kind", "tracks",
            "--path", "/x", "--series", "keypoints",
        ],
    )  # fmt: skip

    assert result.exit_code != 0
    assert "a series is a kind of label" in result.output


def test_removing_a_series_source_can_drop_its_rows(tmp_path: Path) -> None:
    project = _project_with_revisions(tmp_path, 2)
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    manifest = str(library.manifest_path)
    set_dir = str(project.get_root("labels_raw") / "keypoints" / "17-mice")
    _ = runner.invoke(
        app,
        [
            "sources", "add", "-m", manifest, "--kind", "labels", "--id", "p52-17",
            "--path", set_dir, "--series", "keypoints",
            "--file", "rev1/annotations.coco.json",
        ],
    )  # fmt: skip
    _ = runner.invoke(app, ["scan", "-m", manifest, "--kind", "labels"])

    removed = runner.invoke(
        app,
        ["sources", "remove", "-m", manifest, "--kind", "labels", "--id", "p52-17", "--drop-rows", "--json"],
    )  # fmt: skip

    assert removed.exit_code == 0, removed.output
    assert json.loads(removed.output)["dropped_rows"] == 1
    fresh = Dataset(manifest_path=library.manifest_path).load()
    assert read_label_series(fresh, "keypoints").empty
