"""The two command-line gestures a shared model library is built from.

``mosaic links`` links the dataset a model resolves from, and ``mosaic sources add
--series`` claims the exact annotation revisions a library trains on.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.annotations.projection import COCO_EXPORT_FILENAME
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.artifact_kinds import LINKED_KINDS
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    write_series_revision,
)
from tests.helpers import make_dataset, revision_file

runner = CliRunner()


def _project_with_revisions(tmp_path: Path, count: int) -> Dataset:
    project = make_dataset(tmp_path / "52", name="project")
    for number in range(count):
        _ = write_series_revision(
            project,
            series="keypoints",
            key="17-mice",
            payload=json.dumps({"state": number}).encode(),
            exports={COCO_EXPORT_FILENAME: b"{}"},
            origin={},
            n_records=number,
        )
    return project


def test_links_add_list_and_remove(tmp_path: Path) -> None:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")
    manifest = str(project.manifest_path)

    added = runner.invoke(
        app,
        [
            "links",
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

    listed = runner.invoke(app, ["links", "list", "-m", manifest, "--json"])
    emitted = json.loads(listed.output)
    assert "libraries" not in emitted
    assert emitted["links"] == [
        {
            "id": "group",
            "path": "../libraries/7",
            "uuid": library.uuid,
            "added_at": emitted["links"][0]["added_at"],
            "provides": list(LINKED_KINDS),
        }
    ]

    removed = runner.invoke(app, ["links", "remove", "-m", manifest, "--id", "group"])
    assert removed.exit_code == 0, removed.output
    assert Dataset(manifest_path=project.manifest_path).load().links == ()


def test_links_add_stores_the_kinds_it_is_given_in_the_order_kinds_are_listed(
    tmp_path: Path,
) -> None:
    _ = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")
    manifest = str(project.manifest_path)

    added = runner.invoke(
        app,
        [
            "links",
            "add",
            "-m",
            manifest,
            "--id",
            "group",
            "--path",
            "../libraries/7",
            "--provides",
            "tracks",
            "--provides",
            "models",
        ],
    )
    assert added.exit_code == 0, added.output

    listed = json.loads(
        runner.invoke(app, ["links", "list", "-m", manifest, "--json"]).output
    )
    assert listed["links"][0]["provides"] == ["models", "tracks"]
    table = runner.invoke(app, ["links", "list", "-m", manifest])
    assert "models,tracks" in table.output


def test_links_add_without_provides_stores_every_kind(tmp_path: Path) -> None:
    _ = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")

    added = runner.invoke(
        app,
        [
            "links",
            "add",
            "-m",
            str(project.manifest_path),
            "--id",
            "group",
            "--path",
            "../libraries/7",
        ],
    )

    assert added.exit_code == 0, added.output
    reread = Dataset(manifest_path=project.manifest_path).load()
    assert reread.links[0].provides == LINKED_KINDS


def test_links_add_refuses_an_unknown_kind_and_names_the_choices(
    tmp_path: Path,
) -> None:
    _ = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")

    result = runner.invoke(
        app,
        [
            "links",
            "add",
            "-m",
            str(project.manifest_path),
            "--id",
            "group",
            "--path",
            "../libraries/7",
            "--provides",
            "weights",
        ],
    )

    assert result.exit_code != 0
    for kind in LINKED_KINDS:
        assert kind in result.output
    assert Dataset(manifest_path=project.manifest_path).load().links == ()


def test_linking_a_path_that_holds_no_dataset_fails_clearly(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")

    result = runner.invoke(
        app,
        [
            "links",
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
            "--file", revision_file(2),
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
            "--file", revision_file(1),
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


def test_keeping_a_series_sources_rows_names_the_steps_that_drop_them(
    tmp_path: Path,
) -> None:
    project = _project_with_revisions(tmp_path, 2)
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    manifest = str(library.manifest_path)
    set_dir = str(project.get_root("labels_raw") / "keypoints" / "17-mice")
    add = [
        "sources", "add", "-m", manifest, "--kind", "labels", "--id", "p52-17",
        "--path", set_dir, "--series", "keypoints", "--file", revision_file(1),
    ]  # fmt: skip
    remove = [
        "sources", "remove", "-m", manifest, "--kind", "labels", "--id", "p52-17",
    ]  # fmt: skip
    _ = runner.invoke(app, add)
    _ = runner.invoke(app, ["scan", "-m", manifest, "--kind", "labels"])

    removed = runner.invoke(app, remove)

    assert removed.exit_code == 0, removed.output
    assert "1 index row(s)" in removed.output
    assert "Every scan keeps" in removed.output
    assert "declare the source again with 'mosaic sources add'" in removed.output
    assert "'mosaic sources remove --drop-rows'" in removed.output
    assert "--prune-unsourced" not in removed.output
    assert "same folder" not in removed.output
    kept = read_label_series(
        Dataset(manifest_path=library.manifest_path).load(), "keypoints"
    )
    assert len(kept) == 1

    assert runner.invoke(app, add).exit_code == 0
    dropped = runner.invoke(app, [*remove, "--drop-rows", "--json"])

    assert dropped.exit_code == 0, dropped.output
    assert json.loads(dropped.output)["dropped_rows"] == 1
    fresh = Dataset(manifest_path=library.manifest_path).load()
    assert read_label_series(fresh, "keypoints").empty
