"""``mosaic sources``, ``mosaic scan``, ``mosaic tags`` and ``mosaic notes``.

The command surface that replaced ``index-media`` and ``index-tracks``: declare
where the dataset draws from, then rescan exactly that with no arguments at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset, new_dataset_manifest

runner = CliRunner()
VideoWriter = Callable[..., None]


def make_dataset(tmp_path: Path) -> Path:
    return new_dataset_manifest(name="cli", base_dir=tmp_path / "ds")


def indexed(manifest: Path) -> set[str]:
    dataset = Dataset(manifest_path=manifest).load()
    return {
        Path(str(row["abs_path"])).name
        for row in dataset.read_media_index()
        if row.get("abs_path")
    }


class TestSources:
    def test_add_then_list(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                "/mnt/nas",
                "--id",
                "nas",
            ],
        )
        assert result.exit_code == 0, result.output

        listing = runner.invoke(app, ["sources", "list", "-m", str(manifest), "--json"])
        payload = json.loads(listing.output)
        assert payload["sources"][0]["id"] == "nas"
        assert payload["sources"][0]["mode"] == "directory"
        assert payload["sources"][0]["exists"] is False

    def test_the_id_defaults_to_the_directory_name(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                "/mnt/cage_a",
            ],
        )
        listing = runner.invoke(app, ["sources", "list", "-m", str(manifest), "--json"])
        assert json.loads(listing.output)["sources"][0]["id"] == "cage_a"

    def test_a_file_list_and_a_glob_together_are_refused(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                "/n",
                "--file",
                "a.mp4",
                "--extensions",
                ".mp4",
            ],
        )
        assert result.exit_code != 0
        assert "cannot both be given" in result.output

    def test_an_unknown_kind_says_what_it_wanted(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "videos", "--path", "/n"],
        )
        assert result.exit_code != 0
        assert "media, tracks or labels" in result.output

    def test_remove_keeps_rows_and_says_how_many(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        manifest = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "one.mp4")
        _ = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                str(folder),
                "--id",
                "nas",
            ],
        )
        _ = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert indexed(manifest) == {"one.mp4"}

        result = runner.invoke(
            app,
            [
                "sources",
                "remove",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--id",
                "nas",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "claimed by no source" in result.output
        assert indexed(manifest) == {"one.mp4"}

    def test_remove_drop_rows_deletes_them(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        manifest = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "one.mp4")
        _ = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                str(folder),
                "--id",
                "nas",
            ],
        )
        _ = runner.invoke(app, ["scan", "-m", str(manifest)])

        result = runner.invoke(
            app,
            [
                "sources",
                "remove",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--id",
                "nas",
                "--drop-rows",
            ],
        )
        assert result.exit_code == 0, result.output
        assert indexed(manifest) == set()


class TestScan:
    def test_a_bare_scan_rescans_everything_declared(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """The gesture the whole design is for."""
        manifest = make_dataset(tmp_path)
        first, second = tmp_path / "a", tmp_path / "b"
        write_cfr_mp4(first / "one.mp4")
        write_cfr_mp4(second / "two.mp4")
        for name, path in (("first", first), ("second", second)):
            _ = runner.invoke(
                app,
                [
                    "sources",
                    "add",
                    "-m",
                    str(manifest),
                    "--kind",
                    "media",
                    "--path",
                    str(path),
                    "--id",
                    name,
                ],
            )

        result = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert result.exit_code == 0, result.output
        assert indexed(manifest) == {"one.mp4", "two.mp4"}

    def test_only_restricts_the_pass(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        manifest = make_dataset(tmp_path)
        first, second = tmp_path / "a", tmp_path / "b"
        write_cfr_mp4(first / "one.mp4")
        write_cfr_mp4(second / "two.mp4")
        for name, path in (("first", first), ("second", second)):
            _ = runner.invoke(
                app,
                [
                    "sources",
                    "add",
                    "-m",
                    str(manifest),
                    "--kind",
                    "media",
                    "--path",
                    str(path),
                    "--id",
                    name,
                ],
            )

        _ = runner.invoke(app, ["scan", "-m", str(manifest), "--only", "first"])
        assert indexed(manifest) == {"one.mp4"}

    def test_scanning_with_nothing_declared_points_at_sources_add(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert result.exit_code != 0
        assert "mosaic sources add" in result.output

    def test_json_names_the_index_written(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        manifest = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "one.mp4")
        _ = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                str(folder),
                "--id",
                "nas",
            ],
        )
        result = runner.invoke(app, ["scan", "-m", str(manifest), "--json"])
        assert result.exit_code == 0, result.output
        # stdout, not output: the library's progress lines are redirected to
        # stderr precisely so --json leaves stdout parseable.
        assert Path(json.loads(result.stdout)["indexes"]["media"]).name == "index.csv"

    def test_a_file_source_scans_only_its_files(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        manifest = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            write_cfr_mp4(folder / name)
        _ = runner.invoke(
            app,
            [
                "sources",
                "add",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--path",
                str(folder),
                "--id",
                "import",
                "--file",
                "a.mp4",
                "--file",
                "b.mp4",
            ],
        )
        _ = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert indexed(manifest) == {"a.mp4", "b.mp4"}

        _ = runner.invoke(
            app,
            [
                "sources",
                "add-files",
                "-m",
                str(manifest),
                "--kind",
                "media",
                "--id",
                "import",
                "--file",
                "c.mp4",
            ],
        )
        _ = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert indexed(manifest) == {"a.mp4", "b.mp4", "c.mp4"}

    def test_the_retired_commands_are_gone(self, tmp_path: Path) -> None:
        """``mosaic scan`` replaced them; two ways to scan was one too many."""
        for retired in ("index-media", "index-tracks"):
            result = runner.invoke(app, [retired, "--help"])
            assert result.exit_code != 0


class TestTagsAndNotes:
    def test_define_set_and_list(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(
            app,
            [
                "tags",
                "define",
                "-m",
                str(manifest),
                "cohort",
                "--type",
                "categorical",
                "--options",
                "spring,fall",
            ],
        )
        result = runner.invoke(
            app, ["tags", "set", "-m", str(manifest), "cohort", "spring"]
        )
        assert result.exit_code == 0, result.output

        listing = runner.invoke(app, ["tags", "list", "-m", str(manifest), "--json"])
        tag = json.loads(listing.output)["tags"][0]
        assert (tag["name"], tag["type"], tag["value"]) == (
            "cohort",
            "categorical",
            "spring",
        )

    def test_the_two_constraint_spellings_agree(self, tmp_path: Path) -> None:
        """A shorthand must not express something the JSON form could not."""
        shorthand = make_dataset(tmp_path / "a")
        explicit = make_dataset(tmp_path / "b")
        _ = runner.invoke(
            app,
            [
                "tags",
                "define",
                "-m",
                str(shorthand),
                "c",
                "--type",
                "categorical",
                "--options",
                "x,y",
            ],
        )
        _ = runner.invoke(
            app,
            [
                "tags",
                "define",
                "-m",
                str(explicit),
                "c",
                "--type",
                "categorical",
                "--constraints",
                '{"options": ["x", "y"]}',
            ],
        )
        left = json.loads(
            runner.invoke(app, ["tags", "list", "-m", str(shorthand), "--json"]).output
        )
        right = json.loads(
            runner.invoke(app, ["tags", "list", "-m", str(explicit), "--json"]).output
        )
        assert left == right

    def test_a_value_outside_the_vocabulary_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(
            app,
            [
                "tags",
                "define",
                "-m",
                str(manifest),
                "cohort",
                "--type",
                "categorical",
                "--options",
                "spring,fall",
            ],
        )
        result = runner.invoke(
            app, ["tags", "set", "-m", str(manifest), "cohort", "winter"]
        )
        assert result.exit_code != 0
        assert "not in options" in result.output
        # Terse: the validator's sentence, not pydantic's scaffolding.
        assert "For further information visit" not in result.output

    def test_a_label_tag_refuses_a_value(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(
            app, ["tags", "define", "-m", str(manifest), "pilot", "--type", "label"]
        )
        result = runner.invoke(
            app, ["tags", "set", "-m", str(manifest), "pilot", "yes"]
        )
        assert result.exit_code != 0
        assert "has no value" in result.output

    def test_a_tag_can_be_declared_before_it_has_a_value(self, tmp_path: Path) -> None:
        """Declaring that a dataset *has* a cohort precedes knowing which."""
        manifest = make_dataset(tmp_path)
        result = runner.invoke(
            app,
            ["tags", "define", "-m", str(manifest), "n", "--type", "int", "--min", "1"],
        )
        assert result.exit_code == 0, result.output
        listing = runner.invoke(app, ["tags", "list", "-m", str(manifest), "--json"])
        assert json.loads(listing.output)["tags"][0]["value"] is None

    def test_setting_an_undefined_tag_says_to_define_it(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(app, ["tags", "set", "-m", str(manifest), "nope", "x"])
        assert result.exit_code != 0
        assert "Define it first" in result.output

    def test_a_bool_tag_parses_true_and_false(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(
            app, ["tags", "define", "-m", str(manifest), "done", "--type", "bool"]
        )
        _ = runner.invoke(app, ["tags", "set", "-m", str(manifest), "done", "false"])
        listing = runner.invoke(app, ["tags", "list", "-m", str(manifest), "--json"])
        assert json.loads(listing.output)["tags"][0]["value"] is False

    def test_remove(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(
            app, ["tags", "define", "-m", str(manifest), "t", "--type", "label"]
        )
        assert (
            runner.invoke(app, ["tags", "remove", "-m", str(manifest), "t"]).exit_code
            == 0
        )
        assert (
            runner.invoke(app, ["tags", "remove", "-m", str(manifest), "t"]).exit_code
            != 0
        )

    def test_notes_round_trip(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        _ = runner.invoke(app, ["notes", "set", "-m", str(manifest), "cage A pilot"])
        shown = runner.invoke(app, ["notes", "show", "-m", str(manifest)])
        assert "cage A pilot" in shown.output

        _ = runner.invoke(app, ["notes", "clear", "-m", str(manifest)])
        assert (
            runner.invoke(app, ["notes", "show", "-m", str(manifest)]).output.strip()
            == ""
        )

    def test_notes_from_a_file(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        notes = tmp_path / "n.md"
        _ = notes.write_text("line one\nline two\n", encoding="utf-8")
        result = runner.invoke(
            app, ["notes", "set", "-m", str(manifest), "--from-file", str(notes)]
        )
        assert result.exit_code == 0, result.output
        assert Dataset(manifest_path=manifest).load().notes == "line one\nline two\n"
