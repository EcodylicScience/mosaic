"""``mosaic sources``, ``mosaic scan``, ``mosaic tags`` and ``mosaic notes``.

The command surface that replaced ``index-media`` and ``index-tracks``: declare
where the dataset draws from, then rescan exactly that with no arguments at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.manifest import TracksScanSource

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


def declare_missing_sources(manifest: Path, kind: str, *ids: str) -> None:
    """Declare one source per id whose path does not exist, so a scan reads no file."""
    for source_id in ids:
        added = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", kind,
             "--path", str(manifest.parent / "unmounted" / source_id),
             "--id", source_id]
            + (["--patterns", "*.npy", "--src-format", "calms21_npy"]
               if kind == "tracks" else []),
        )  # fmt: skip
        assert added.exit_code == 0, added.output


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

    def test_a_dividing_source_is_declared_with_its_rule(self, tmp_path: Path) -> None:
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
                "/mnt/stores",
                "--divide-frames",
                "60000",
                "--recording-by",
                "camera",
                "--recording-name",
                "cam7",
            ],
        )
        assert result.exit_code == 0, result.output

        (source,) = Dataset(manifest_path=manifest).load().manifest.sources.media
        assert source.division is not None
        assert (source.division.frames, source.division.identity) == (60000, "camera")
        assert source.division.name == "cam7"
        assert "manifest_version: 4" in manifest.read_text()

    def test_a_recording_flag_without_a_division_names_the_flags(
        self, tmp_path: Path
    ) -> None:
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
                "/mnt/stores",
                "--recording-by",
                "camera",
            ],
        )
        assert result.exit_code != 0
        assert "--recording-by applies only beside --divide-frames" in result.output

    def test_a_recording_identity_outside_store_and_camera_is_refused(
        self, tmp_path: Path
    ) -> None:
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
                "/mnt/stores",
                "--divide-frames",
                "600",
                "--recording-by",
                "tank",
            ],
        )
        assert result.exit_code == 2
        assert "'store', 'camera'" in result.output
        assert Dataset(manifest_path=manifest).load().manifest.sources.media == ()

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


class TestGroupPathPattern:
    def test_a_path_pattern_groups_by_folder_and_is_declared_in_the_manifest(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        root = tmp_path / "release"
        for strain in ("strainA", "strainB"):
            folder = root / "Behavior_data" / strain
            folder.mkdir(parents=True)
            _ = (folder / "x.npy").write_bytes(b"\x00")
        added = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "tracks",
             "--path", str(root), "--id", "d", "--patterns", "*.npy",
             "--src-format", "calms21_npy",
             "--group-path-pattern", "Behavior_data/([^/]+)/"],
        )  # fmt: skip
        assert added.exit_code == 0, added.output
        assert "manifest_version: 5" in manifest.read_text()

        scanned = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert scanned.exit_code == 0, scanned.output

        rows = Dataset(manifest_path=manifest).load().read_tracks_raw_index()
        assert sorted(row["group"] for row in rows) == ["strainA", "strainB"]

    def test_the_two_group_options_are_refused_together_by_flag(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "tracks",
             "--path", str(tmp_path), "--group-pattern", "(a)",
             "--group-path-pattern", "(b)/"],
        )  # fmt: skip
        assert result.exit_code != 0
        assert "--group-pattern" in result.output
        assert "--group-path-pattern" in result.output

    def test_the_help_says_where_each_pattern_looks(self) -> None:
        result = runner.invoke(app, ["sources", "add", "--help"])
        flat = " ".join(result.output.replace("│", " ").split())
        assert "file stem" in flat
        assert "relative to the source" in flat


class TestDeclaringAMissingPath:
    def test_it_warns_and_still_declares(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        gone = tmp_path / "unmounted"
        result = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "tracks",
             "--path", str(gone), "--id", "s", "--patterns", "*.npy",
             "--src-format", "calms21_npy", "--json"],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert str(gone) in result.stderr
        payload = json.loads(result.stdout)
        assert payload["id"] == "s"
        (finding,) = payload["findings"]
        assert (finding["problem"], finding["path"]) == ("path_missing", str(gone))
        sources = Dataset(manifest_path=manifest).load().scan_sources("tracks")
        assert [source.id for source in sources] == ["s"]

    def test_a_present_path_warns_of_nothing(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        result = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "tracks",
             "--path", str(tmp_path), "--id", "s", "--json"],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["findings"] == []
        assert result.stderr == ""


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

    def test_json_lists_what_the_scan_dropped(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        gone = tmp_path / "unmounted"
        added = runner.invoke(
            app,
            [
                "sources", "add", "-m", str(manifest),
                "--kind", "tracks", "--path", str(gone), "--id", "s",
                "--patterns", "*.npy", "--src-format", "calms21_npy",
            ],
        )  # fmt: skip
        assert added.exit_code == 0, added.output

        result = runner.invoke(app, ["scan", "-m", str(manifest), "--json"])

        assert result.exit_code == 0, result.output
        (finding,) = json.loads(result.stdout)["findings"]
        assert finding["kind"] == "tracks"
        assert finding["source"] == "s"
        assert finding["problem"] == "path_missing"
        assert finding["path"] == str(gone)
        assert str(gone) in result.stderr

    def test_a_scan_writes_a_start_line_to_stderr_for_each_kind(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        declare_missing_sources(manifest, "media", "m1", "m2")
        declare_missing_sources(manifest, "tracks", "raw", "extra")

        result = runner.invoke(app, ["scan", "-m", str(manifest)])

        assert result.exit_code == 0, result.output
        starts = [
            line for line in result.stderr.splitlines() if line.startswith("scanning ")
        ]
        assert starts == [
            "scanning media: sources m1, m2",
            "scanning tracks: sources raw, extra",
        ]

    def test_a_scan_restricted_by_only_names_just_the_sources_it_reads(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        declare_missing_sources(manifest, "media", "m1", "m2")
        declare_missing_sources(manifest, "tracks", "raw", "extra")

        result = runner.invoke(app, ["scan", "-m", str(manifest), "--only", "m1"])

        assert result.exit_code == 0, result.output
        starts = [
            line for line in result.stderr.splitlines() if line.startswith("scanning ")
        ]
        assert starts == ["scanning media: sources m1"]

    def test_json_keeps_stdout_one_value_and_puts_the_start_lines_on_stderr(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        declare_missing_sources(manifest, "tracks", "raw", "extra")

        result = runner.invoke(app, ["scan", "-m", str(manifest), "--json"])

        assert result.exit_code == 0, result.output
        assert "indexes" in json.loads(result.stdout)
        assert "scanning tracks: sources raw, extra" in result.stderr
        assert "scanning" not in result.stdout

    def test_json_lists_no_findings_for_a_clean_scan(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        manifest = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "one.mp4")
        _ = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "media",
             "--path", str(folder), "--id", "nas"],
        )  # fmt: skip
        result = runner.invoke(app, ["scan", "-m", str(manifest), "--json"])
        assert json.loads(result.stdout)["findings"] == []

    def test_a_tracks_file_source_is_indexed_through_the_cli(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        folder.mkdir()
        for name in ("a.npy", "b.npy"):
            _ = (folder / name).write_bytes(b"\x00")
        added = runner.invoke(
            app,
            ["sources", "add", "-m", str(manifest), "--kind", "tracks",
             "--path", str(folder), "--file", "a.npy", "--id", "s",
             "--src-format", "calms21_npy"],
        )  # fmt: skip
        assert added.exit_code == 0, added.output
        scanned = runner.invoke(app, ["scan", "-m", str(manifest)])
        assert scanned.exit_code == 0, scanned.output
        rows = Dataset(manifest_path=manifest).load().read_tracks_raw_index()
        assert [Path(row["abs_path"]).name for row in rows] == ["a.npy"]

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


class TestRemoveATracksSource:
    """``sources remove --drop-rows`` drops what no other tracks source claims."""

    @staticmethod
    def two_sources_sharing_a_target(tmp_path: Path) -> Path:
        manifest = make_dataset(tmp_path)
        dataset = Dataset(manifest_path=manifest).load()
        target = tmp_path / "elsewhere" / "t.dat"
        for path in (tmp_path / "a" / "x.npy", tmp_path / "b" / "y.npy", target):
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(b"\x00")
        for name in ("a", "b"):
            (tmp_path / name / "link.npy").symlink_to(target)
            dataset.add_scan_source(
                TracksScanSource(
                    id=name, path=str(tmp_path / name), patterns=("*.npy",)
                )
            )
        _ = dataset.scan_tracks()
        return manifest

    @staticmethod
    def remove(manifest: Path, source_id: str) -> dict[str, object]:
        result = runner.invoke(
            app,
            ["sources", "remove", "-m", str(manifest), "--kind", "tracks",
             "--id", source_id, "--drop-rows", "--json"],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        payload: dict[str, object] = json.loads(result.output)
        return payload

    def test_drop_rows_keeps_a_target_another_source_shares(
        self, tmp_path: Path
    ) -> None:
        manifest = self.two_sources_sharing_a_target(tmp_path)

        payload = self.remove(manifest, "a")

        assert payload["unclaimed_rows"] == 1
        assert payload["dropped_rows"] == 1
        dataset = Dataset(manifest_path=manifest).load()
        names = {Path(str(r["abs_path"])).name for r in dataset.read_tracks_raw_index()}
        assert names == {"y.npy", "t.dat"}

    def test_a_removal_walks_each_source_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = self.two_sources_sharing_a_target(tmp_path)
        from mosaic.core import dataset as dataset_module

        walked: list[Path] = []
        real = dataset_module.walked_files

        def counting(
            base: Path,
            patterns: Sequence[str],
            *,
            recursive: bool,
            exclude_patterns: Sequence[str] = (),
        ) -> frozenset[Path]:
            walked.append(base)
            return real(
                base, patterns, recursive=recursive, exclude_patterns=exclude_patterns
            )

        monkeypatch.setattr(dataset_module, "walked_files", counting)

        _ = self.remove(manifest, "a")

        assert sorted(walked) == [tmp_path / "a", tmp_path / "b"]


class TestSourcesOverOneFolder:
    def test_scan_only_refuses_a_file_two_sources_take(self, tmp_path: Path) -> None:
        manifest = make_dataset(tmp_path)
        base = tmp_path / "data"
        base.mkdir()
        _ = (base / "a.npy").write_bytes(b"\x00")
        for source_id, pattern in (("all", "*.npy"), ("xs", "x*.npy")):
            added = runner.invoke(
                app,
                [
                    "sources",
                    "add",
                    "-m",
                    str(manifest),
                    "--kind",
                    "tracks",
                    "--path",
                    str(base),
                    "--id",
                    source_id,
                    "--patterns",
                    pattern,
                ],
            )
            assert added.exit_code == 0, added.output
        _ = (base / "x1.npy").write_bytes(b"\x00")

        result = runner.invoke(app, ["scan", "-m", str(manifest), "--only", "all"])

        assert result.exit_code != 0
        assert "x1.npy" in result.output
        dataset = Dataset(manifest_path=manifest).load()
        assert dataset.read_tracks_raw_index() == []

    def test_scan_only_with_prune_is_refused_beside_a_sibling(
        self, tmp_path: Path
    ) -> None:
        manifest = self.two_sources_over_one_folder(tmp_path)
        index = (
            Dataset(manifest_path=manifest).load().get_root("tracks_raw") / "index.csv"
        )
        before = index.read_bytes()

        result = runner.invoke(
            app,
            ["scan", "-m", str(manifest), "--only", "npy", "--prune-unsourced"],
        )

        assert result.exit_code != 0
        assert "'npz'" in result.output
        assert index.read_bytes() == before

    def test_scan_only_with_prune_prunes_without_a_sibling(
        self, tmp_path: Path
    ) -> None:
        manifest = make_dataset(tmp_path)
        dataset = Dataset(manifest_path=manifest).load()
        for name in ("one", "two"):
            (tmp_path / name).mkdir()
            _ = (tmp_path / name / f"{name}.npy").write_bytes(b"\x00")
            dataset.add_scan_source(
                TracksScanSource(
                    id=name, path=str(tmp_path / name), patterns=("*.npy",)
                )
            )
        _ = dataset.scan_tracks()

        result = runner.invoke(
            app, ["scan", "-m", str(manifest), "--only", "one", "--prune-unsourced"]
        )

        assert result.exit_code == 0, result.output
        rows = Dataset(manifest_path=manifest).load().read_tracks_raw_index()
        assert {Path(str(r["abs_path"])).name for r in rows} == {"one.npy"}

    @staticmethod
    def two_sources_over_one_folder(tmp_path: Path) -> Path:
        manifest = make_dataset(tmp_path)
        dataset = Dataset(manifest_path=manifest).load()
        base = tmp_path / "data"
        base.mkdir()
        for name in ("a.npy", "a2.npy", "b.npz"):
            _ = (base / name).write_bytes(b"\x00")
        for source_id, pattern in (("npy", "*.npy"), ("npz", "*.npz")):
            dataset.add_scan_source(
                TracksScanSource(id=source_id, path=str(base), patterns=(pattern,))
            )
        _ = dataset.scan_tracks()
        return manifest

    def test_drop_rows_beside_a_sibling_over_one_folder_drops_its_own(
        self, tmp_path: Path
    ) -> None:
        manifest = self.two_sources_over_one_folder(tmp_path)

        result = runner.invoke(
            app,
            ["sources", "remove", "-m", str(manifest), "--kind", "tracks",
             "--id", "npz", "--drop-rows", "--json"],
        )  # fmt: skip

        assert result.exit_code == 0, result.output
        payload: dict[str, object] = json.loads(result.output)
        assert payload["unclaimed_rows"] == 1
        assert payload["dropped_rows"] == 1
        rows = Dataset(manifest_path=manifest).load().read_tracks_raw_index()
        assert {Path(str(r["abs_path"])).name for r in rows} == {"a.npy", "a2.npy"}

    def test_keeping_rows_names_the_passes_that_drop_them(self, tmp_path: Path) -> None:
        manifest = self.two_sources_over_one_folder(tmp_path)

        result = runner.invoke(
            app,
            ["sources", "remove", "-m", str(manifest), "--kind", "tracks",
             "--id", "npz"],
        )  # fmt: skip

        assert result.exit_code == 0, result.output
        assert "1 index row(s) are now claimed by no source" in result.output
        assert "mosaic scan --prune-unsourced" in result.output
        assert "over the same folder" in result.output
        assert "reindex" not in result.output
        assert "prune-media" not in result.output
