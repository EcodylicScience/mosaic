"""``mosaic selection``: save, list, show and delete a dataset's saved selections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.manifest import MANAGED_BY_KEY
from mosaic.core.selections import (
    Selection,
    list_selections,
    read_selection,
    save_selection,
)
from tests.helpers import (
    add_tracks_table,
    invoke_ok,
    make_dataset,
    track_table,
    tree_snapshot,
)

ENTRIES: list[Entry] = [("g1", "s1"), ("g1", "s2"), ("g2", "s1")]


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """A dataset whose tracks hold ``ENTRIES`` and ``("", "solo")``."""
    dataset = make_dataset(tmp_path / "ds")
    for group, sequence in [*ENTRIES, ("", "solo")]:
        add_tracks_table(
            dataset, "trex", track_table(group, sequence, n_rows=4), n_rows=4
        )
    return dataset


def _selection(ds: Dataset, *arguments: str) -> Result:
    return invoke_ok(["selection", *arguments, "--manifest", str(ds.manifest_path)])


def _refused(ds: Dataset, *arguments: str) -> str:
    result = CliRunner().invoke(
        app, ["selection", *arguments, "--manifest", str(ds.manifest_path)]
    )
    assert result.exit_code == 1, result.stdout
    return result.stderr


def _directory(ds: Dataset) -> Path:
    return ds.base_dir / ".mosaic" / "selections"


def _entries(ds: Dataset, name: str) -> list[Entry]:
    return read_selection(ds, name).entries


class TestSave:
    def test_entry_flags_save_those_entries_with_a_note(self, ds: Dataset) -> None:
        _ = _selection(
            ds,
            "save",
            "day1",
            "--entry",
            "g1:s2",
            "--entry",
            "g1:s1",
            "--note",
            "first day",
        )

        saved = read_selection(ds, "day1")
        assert saved.entries == [("g1", "s1"), ("g1", "s2")]
        assert saved.note == "first day"

    def test_an_entry_token_without_a_colon_is_a_sequence_of_the_empty_group(
        self, ds: Dataset
    ) -> None:
        _ = _selection(ds, "save", "alone", "--entry", "solo")

        assert _entries(ds, "alone") == [("", "solo")]

    def test_group_flags_save_every_entry_of_the_groups(self, ds: Dataset) -> None:
        _ = _selection(ds, "save", "both", "--group", "g1", "--group", "g2")

        assert _entries(ds, "both") == ENTRIES

    def test_a_scope_saves_the_entries_it_selects(self, ds: Dataset) -> None:
        _ = _selection(ds, "save", "firsts", "--from-scope", '{"sequences": ["s1"]}')

        assert _entries(ds, "firsts") == [("g1", "s1"), ("g2", "s1")]

    def test_a_scope_file_is_read_as_a_scope(self, ds: Dataset, tmp_path: Path) -> None:
        scope = tmp_path / "scope.json"
        _ = scope.write_text('{"groups": ["g2"]}', encoding="utf-8")

        _ = _selection(ds, "save", "second", "--from-scope", f"@{scope}")

        assert _entries(ds, "second") == [("g2", "s1")]

    def test_a_scope_that_names_entries_saves_them_as_named(self, ds: Dataset) -> None:
        _ = _selection(
            ds, "save", "named", "--from-scope", '{"entries": [["g2", "s1"]]}'
        )

        assert _entries(ds, "named") == [("g2", "s1")]

    def test_a_scope_that_names_a_selection_copies_it(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])

        _ = _selection(ds, "save", "copy", "--from-scope", '{"selection": "day1"}')
        _ = _selection(ds, "save", "again", "--entry", "@day1")

        assert _entries(ds, "copy") == ENTRIES[:2]
        assert _entries(ds, "again") == ENTRIES[:2]

    def test_an_entries_file_holds_a_token_on_each_line(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        listing = tmp_path / "entries.txt"
        _ = listing.write_text("g2:s1\n\ng1:s1\nsolo\n", encoding="utf-8")

        _ = _selection(ds, "save", "listed", "--entries-file", str(listing))

        assert _entries(ds, "listed") == [("", "solo"), ("g1", "s1"), ("g2", "s1")]

    def test_an_entries_file_holds_a_json_list_of_pairs(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        listing = tmp_path / "entries.json"
        _ = listing.write_text('[["g2", "s1"], ["g1", "s2"]]', encoding="utf-8")

        _ = _selection(ds, "save", "paired", "--entries-file", str(listing))

        assert _entries(ds, "paired") == [("g1", "s2"), ("g2", "s1")]

    def test_json_prints_the_saved_selection(self, ds: Dataset) -> None:
        result = invoke_ok(
            [
                "selection",
                "save",
                "day1",
                "--entry",
                "g1:s1",
                "--json",
                "--manifest",
                str(ds.manifest_path),
            ]
        )

        printed = Selection.model_validate_json(result.stdout)
        assert printed == read_selection(ds, "day1")

    def test_the_plain_output_names_the_selection_and_its_entry_count(
        self, ds: Dataset
    ) -> None:
        result = invoke_ok(
            [
                "selection",
                "save",
                "day1",
                "--group",
                "g1",
                "--manifest",
                str(ds.manifest_path),
            ]
        )

        assert "day1" in result.stdout
        assert "2 entries" in result.stdout

    @pytest.mark.parametrize(
        "sources",
        [
            [],
            ["--entry", "g1:s1", "--group", "g1"],
            ["--entry", "g1:s1", "--from-scope", "{}"],
            ["--group", "g1", "--entries-file", "entries.txt"],
        ],
        ids=["none", "entry-and-group", "entry-and-scope", "group-and-file"],
    )
    def test_exactly_one_source_of_entries_is_required(
        self, ds: Dataset, sources: list[str]
    ) -> None:
        message = _refused(ds, "save", "day1", *sources)

        assert "--from-scope" in message
        assert "--entries-file" in message
        assert not _directory(ds).exists()

    def test_an_existing_name_is_refused_and_the_file_is_kept(
        self, ds: Dataset
    ) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:1])

        message = _refused(ds, "save", "day1", "--group", "g1")

        assert "frozen" in message
        assert _entries(ds, "day1") == ENTRIES[:1]

    def test_an_entry_outside_the_dataset_is_refused(self, ds: Dataset) -> None:
        message = _refused(ds, "save", "day1", "--entry", "g1:s1", "--entry", "g9:x")

        assert "g9:x" in message
        assert list_selections(ds) == []

    def test_a_scope_that_selects_no_entry_is_refused(self, ds: Dataset) -> None:
        message = _refused(ds, "save", "day1", "--group", "absent")

        assert "no entries" in message
        assert list_selections(ds) == []

    def test_an_invalid_name_is_refused(self, ds: Dataset) -> None:
        message = _refused(ds, "save", "day 1", "--group", "g1")

        assert "whitespace" in message

    def test_a_missing_entries_file_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        message = _refused(
            ds, "save", "day1", "--entries-file", str(tmp_path / "absent.txt")
        )

        assert "absent.txt" in message

    def test_a_managed_dataset_is_refused_and_written_nothing(
        self, ds: Dataset
    ) -> None:
        with ds.manifest_path.open("a", encoding="utf-8") as handle:
            _ = handle.write(f"{MANAGED_BY_KEY}: mosaic-api\n")
        before = tree_snapshot(ds.base_dir)

        message = _refused(ds, "save", "day1", "--group", "g1")

        assert "mosaic-api" in message
        assert tree_snapshot(ds.base_dir) == before


class TestList:
    def test_json_lists_each_selection_with_its_entry_count(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day2", ENTRIES[2:], note="late")
        saved = save_selection(ds, "day1", ENTRIES[:2])

        result = invoke_ok(
            ["selection", "list", "--json", "--manifest", str(ds.manifest_path)]
        )

        payload = json.loads(result.stdout)
        assert payload["selections"] == [
            {
                "name": "day1",
                "n_entries": 2,
                "created_at": saved.created_at,
                "note": "",
            },
            {
                "name": "day2",
                "n_entries": 1,
                "created_at": read_selection(ds, "day2").created_at,
                "note": "late",
            },
        ]
        assert payload["unreadable"] == []

    def test_the_plain_output_is_a_table(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2], note="first day")

        result = invoke_ok(["selection", "list", "--manifest", str(ds.manifest_path)])

        header, row = result.stdout.splitlines()[:2]
        assert header.split("\t") == ["name", "entries", "created_at", "note"]
        assert row.split("\t")[0::3] == ["day1", "first day"]
        assert row.split("\t")[1] == "2"

    def test_a_dataset_without_selections_says_so(self, ds: Dataset) -> None:
        result = invoke_ok(["selection", "list", "--manifest", str(ds.manifest_path)])

        assert "No selection" in result.stdout
        assert "mosaic selection save" in result.stdout

    def test_an_unreadable_file_is_listed_with_its_reason(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])
        _ = (_directory(ds) / "broken.json").write_text("{", encoding="utf-8")

        result = invoke_ok(
            ["selection", "list", "--json", "--manifest", str(ds.manifest_path)]
        )

        payload = json.loads(result.stdout)
        assert [row["name"] for row in payload["selections"]] == ["day1"]
        assert [row["file"] for row in payload["unreadable"]] == ["broken.json"]
        assert payload["unreadable"][0]["reason"]

    def test_the_plain_output_says_how_to_delete_an_unreadable_file(
        self, ds: Dataset
    ) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])
        conflict = _directory(ds) / "day1.sync-conflict.json"
        _ = conflict.write_text(
            (_directory(ds) / "day1.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        result = invoke_ok(["selection", "list", "--manifest", str(ds.manifest_path)])

        assert "day1.sync-conflict.json" in result.stdout
        assert "mosaic selection delete day1.sync-conflict" in result.stdout

    def test_the_plain_output_names_the_path_of_a_file_no_name_reaches(
        self, ds: Dataset
    ) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])
        copy = _directory(ds) / "day1 (1).json"
        _ = copy.write_text(
            (_directory(ds) / "day1.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        result = invoke_ok(["selection", "list", "--manifest", str(ds.manifest_path)])

        assert str(copy) in result.stdout
        assert "by hand" in result.stdout


class TestShow:
    def test_json_prints_the_file_fields_and_entries(self, ds: Dataset) -> None:
        saved = save_selection(ds, "day1", ENTRIES[:2], note="first day")

        result = invoke_ok(
            ["selection", "show", "day1", "--json", "--manifest", str(ds.manifest_path)]
        )

        assert Selection.model_validate_json(result.stdout) == saved

    def test_the_plain_output_gives_the_fields_then_each_entry(
        self, ds: Dataset
    ) -> None:
        saved = save_selection(ds, "day1", ENTRIES[:2], note="first day")

        result = invoke_ok(
            ["selection", "show", "day1", "--manifest", str(ds.manifest_path)]
        )

        lines = result.stdout.splitlines()
        assert "name: day1" in lines
        assert "note: first day" in lines
        assert f"digest: {saved.digest}" in lines
        assert "entries: 2" in lines
        assert lines[-2:] == ["g1:s1", "g1:s2"]

    def test_a_missing_selection_names_the_selections_present(
        self, ds: Dataset
    ) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])

        message = _refused(ds, "show", "day2")

        assert "No selection named 'day2'" in message
        assert "day1" in message

    def test_a_missing_selection_is_the_envelope_under_json(self, ds: Dataset) -> None:
        result = CliRunner().invoke(
            app,
            [
                "selection",
                "show",
                "day2",
                "--json",
                "--manifest",
                str(ds.manifest_path),
            ],
        )

        assert result.exit_code == 1
        assert json.loads(result.stdout)["reason"] == "selection_not_found"

    def test_an_unreadable_file_is_refused_with_its_reason(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])
        _ = (_directory(ds) / "day1.json").write_text("{", encoding="utf-8")

        message = _refused(ds, "show", "day1")

        assert "day1.json" in message


class TestDelete:
    def test_delete_removes_the_file_and_a_save_then_works(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])

        result = _selection(ds, "delete", "day1")

        assert "day1" in result.stdout
        assert list_selections(ds) == []
        _ = _selection(ds, "save", "day1", "--group", "g1")
        assert _entries(ds, "day1") == ENTRIES[:2]

    def test_a_missing_selection_is_refused_naming_those_present(
        self, ds: Dataset
    ) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])

        message = _refused(ds, "delete", "day2")

        assert "No selection named 'day2'" in message
        assert (_directory(ds) / "day1.json").exists()

    def test_an_unreadable_file_of_a_valid_name_is_deleted_by_name(
        self, ds: Dataset
    ) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])
        conflict = _directory(ds) / "day1.sync-conflict.json"
        _ = conflict.write_text("{", encoding="utf-8")

        _ = _selection(ds, "delete", "day1.sync-conflict")

        assert not conflict.exists()
        assert (_directory(ds) / "day1.json").exists()

    def test_a_managed_dataset_is_refused_and_keeps_the_file(self, ds: Dataset) -> None:
        _ = save_selection(ds, "day1", ENTRIES[:2])
        with ds.manifest_path.open("a", encoding="utf-8") as handle:
            _ = handle.write(f"{MANAGED_BY_KEY}: mosaic-api\n")
        before = tree_snapshot(ds.base_dir)

        message = _refused(ds, "delete", "day1")

        assert "mosaic-api" in message
        assert tree_snapshot(ds.base_dir) == before
