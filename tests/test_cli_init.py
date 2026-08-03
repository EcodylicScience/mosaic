"""``mosaic init``: creating a dataset from the command line.

Creation was Python-only until this command existed, so the gesture that starts
every other one was the gesture the CLI could not do.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import MANIFEST_VERSION, default_roots

runner = CliRunner()


def read_manifest_file(directory: Path) -> dict[str, object]:
    return yaml.safe_load((directory / "dataset.yaml").read_text(encoding="utf-8"))


class TestInit:
    def test_it_writes_a_manifest_the_library_can_load(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        result = runner.invoke(app, ["init", str(target), "--name", "cage-a"])
        assert result.exit_code == 0, result.output

        dataset = Dataset(manifest_path=target / "dataset.yaml").load()
        assert dataset.name == "cage-a"
        assert dataset.manifest.manifest_version == MANIFEST_VERSION

    def test_a_fresh_manifest_is_at_the_current_version(self, tmp_path: Path) -> None:
        _ = runner.invoke(app, ["init", str(tmp_path / "ds")])
        assert read_manifest_file(tmp_path / "ds")["manifest_version"] == 2

    def test_the_name_defaults_to_the_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "cage-b"
        _ = runner.invoke(app, ["init", str(target)])
        assert read_manifest_file(target)["name"] == "cage-b"

    def test_every_declared_root_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        _ = runner.invoke(app, ["init", str(target)])
        for relative in default_roots.values():
            assert (target / relative).is_dir(), relative

    def test_it_refuses_over_an_existing_manifest(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        _ = runner.invoke(app, ["init", str(target), "--name", "first"])
        result = runner.invoke(app, ["init", str(target), "--name", "second"])
        assert result.exit_code != 0
        assert "already exists" in result.output
        assert read_manifest_file(target)["name"] == "first"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        _ = runner.invoke(app, ["init", str(target), "--name", "first"])
        result = runner.invoke(
            app, ["init", str(target), "--name", "second", "--force"]
        )
        assert result.exit_code == 0, result.output
        assert read_manifest_file(target)["name"] == "second"

    def test_a_root_override_lands_in_the_manifest(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        result = runner.invoke(
            app, ["init", str(target), "--root", "features=derived/features"]
        )
        assert result.exit_code == 0, result.output
        roots = read_manifest_file(target)["roots"]
        assert isinstance(roots, dict)
        assert roots["features"] == "derived/features"
        assert (target / "derived" / "features").is_dir()

    def test_a_root_outside_the_dataset_is_refused(self, tmp_path: Path) -> None:
        """The rule that replaced an external ``media_raw``, at the CLI boundary.

        A root outside the dataset would put that root's own ``index.csv``
        outside too. Storage elsewhere is reached with a scan source instead,
        which is what the error says.
        """
        result = runner.invoke(
            app,
            ["init", str(tmp_path / "ds"), "--root", f"media_raw={tmp_path / 'nas'}"],
        )
        assert result.exit_code != 0
        assert "outside the dataset" in result.output

    def test_a_malformed_root_says_what_it_wanted(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["init", str(tmp_path / "ds"), "--root", "features"]
        )
        assert result.exit_code != 0
        assert "KEY=VALUE" in result.output

    def test_notes_and_tags_are_written(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        result = runner.invoke(
            app,
            [
                "init",
                str(target),
                "--note",
                "cage A pilot",
                "--tag",
                "cohort=2026-spring",
                "--tag",
                "species=mus-musculus",
            ],
        )
        assert result.exit_code == 0, result.output

        dataset = Dataset(manifest_path=target / "dataset.yaml").load()
        assert dataset.manifest.notes == "cage A pilot"
        assert {tag.name for tag in dataset.manifest.tags} == {"cohort", "species"}
        cohort = dataset.manifest.tag("cohort")
        assert cohort is not None
        assert (cohort.type, cohort.value) == ("text", "2026-spring")

    def test_notes_can_come_from_a_file(self, tmp_path: Path) -> None:
        notes = tmp_path / "notes.md"
        _ = notes.write_text("line one\nline two\n", encoding="utf-8")
        target = tmp_path / "ds"
        result = runner.invoke(app, ["init", str(target), "--notes-file", str(notes)])
        assert result.exit_code == 0, result.output
        loaded = Dataset(manifest_path=target / "dataset.yaml").load()
        assert loaded.manifest.notes == "line one\nline two\n"

    def test_note_and_notes_file_together_are_refused(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["init", str(tmp_path / "ds"), "--note", "x", "--notes-file", "y.md"],
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_json_output_names_the_manifest(self, tmp_path: Path) -> None:
        target = tmp_path / "ds"
        result = runner.invoke(app, ["init", str(target), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert Path(payload["manifest"]) == (target / "dataset.yaml").resolve()
