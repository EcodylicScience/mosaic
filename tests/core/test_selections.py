"""A saved selection names an entry set that does not grow.

A selection is a frozen file under ``<dataset>/.mosaic/selections/``. These pin its
shape, the names it accepts, the entries it refuses, the lock a writer holds and
that a read-only open leaves the tree as it found it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core import selections
from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline import _utils
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.scope import Scope
from mosaic.core.selections import (
    SELECTION_FORMAT,
    Selection,
    SelectionExists,
    SelectionNotFound,
    SelectionUnreadable,
    UnreadableSelection,
    delete_selection,
    list_selections,
    read_selection,
    save_selection,
    selection_scope,
    unreadable_selections,
    validate_selection_name,
)
from tests.helpers import (
    add_tracks_table,
    make_dataset,
    modules_loaded_by,
    observed,
    track_table,
)

ENTRIES: list[Entry] = [("g1", "s2"), ("g1", "s1"), ("g2", "s1")]


def _selections_dir(ds: Dataset) -> Path:
    return ds.base_dir / ".mosaic" / "selections"


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """A dataset whose tracks hold ``ENTRIES`` and ``("", "solo")``."""
    dataset = make_dataset(tmp_path / "ds")
    for group, sequence in [*ENTRIES, ("", "solo")]:
        add_tracks_table(
            dataset, "trex", track_table(group, sequence, n_rows=4), n_rows=4
        )
    return dataset


def test_a_save_writes_the_selection_file(ds: Dataset) -> None:
    saved = save_selection(ds, "day1", ENTRIES, note="first day")

    path = _selections_dir(ds) / "day1.json"
    payload = json.loads(path.read_text())
    assert list(payload) == [
        "format",
        "name",
        "created_at",
        "note",
        "entries",
        "digest",
    ]
    assert payload["format"] == SELECTION_FORMAT == "mosaic-selection/1"
    assert payload["name"] == "day1"
    assert payload["note"] == "first day"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", payload["created_at"])
    assert payload["entries"] == [["g1", "s1"], ["g1", "s2"], ["g2", "s1"]]
    canonical = json.dumps(payload["entries"], sort_keys=True, separators=(",", ":"))
    assert (
        payload["digest"]
        == "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    assert saved == read_selection(ds, "day1")
    assert isinstance(saved, Selection)


def test_entries_are_sorted_and_unique(ds: Dataset) -> None:
    saved = save_selection(ds, "dup", [*ENTRIES, ("g1", "s1")])

    assert saved.entries == [("g1", "s1"), ("g1", "s2"), ("g2", "s1")]


def test_the_digest_depends_on_the_entries_alone(ds: Dataset) -> None:
    one = save_selection(ds, "one", ENTRIES, note="a")
    two = save_selection(ds, "two", list(reversed(ENTRIES)), note="b")
    other = save_selection(ds, "other", ENTRIES[:2])

    assert one.digest == two.digest
    assert one.digest != other.digest


def test_selection_scope_returns_the_entries_as_a_scope(ds: Dataset) -> None:
    save_selection(ds, "day1", ENTRIES)

    assert selection_scope(ds, "day1") == Scope(
        entries=[("g1", "s1"), ("g1", "s2"), ("g2", "s1")]
    )


@pytest.mark.parametrize(
    "name",
    [
        "a/b",
        "a\\b",
        "a\x00b",
        "",
        ".hidden",
        " lead",
        "trail ",
        "in side",
        "tab\tbed",
        "a:b",
        "@day1",
        "d@y",
        "x" * 129,
        "é" * 65,
    ],
)
def test_a_bad_name_is_refused(ds: Dataset, name: str) -> None:
    with pytest.raises(ValueError):
        validate_selection_name(name)
    with pytest.raises(ValueError):
        save_selection(ds, name, ENTRIES)
    assert not _selections_dir(ds).exists()


@pytest.mark.parametrize("name", ["day1", "Day-1_a.b", "x" * 128, "é" * 64])
def test_a_good_name_is_accepted(name: str) -> None:
    assert validate_selection_name(name) == name


def test_a_name_equal_to_an_existing_one_ignoring_case_is_refused(
    ds: Dataset,
) -> None:
    save_selection(ds, "Day1", ENTRIES)

    with pytest.raises(SelectionExists, match="Day1"):
        save_selection(ds, "day1", ENTRIES)

    assert [s.name for s in list_selections(ds)] == ["Day1"]


def test_a_selection_is_frozen_until_it_is_deleted(ds: Dataset) -> None:
    first = save_selection(ds, "day1", ENTRIES)
    with pytest.raises(SelectionExists):
        save_selection(ds, "day1", ENTRIES[:1])
    assert read_selection(ds, "day1") == first

    delete_selection(ds, "day1")
    second = save_selection(ds, "day1", ENTRIES[:1])

    assert read_selection(ds, "day1") == second
    assert second.entries == [("g1", "s2")]


def test_an_empty_entry_list_is_refused(ds: Dataset) -> None:
    with pytest.raises(ValueError, match="no entries"):
        save_selection(ds, "none", [])

    assert list_selections(ds) == []


def test_entries_outside_the_universe_are_refused_naming_five(ds: Dataset) -> None:
    missing = [("gone", f"s{i}") for i in range(8)]

    with pytest.raises(ValueError) as caught:
        save_selection(ds, "ghost", [*ENTRIES, *missing])

    message = str(caught.value)
    assert all(f"s{i}" in message for i in range(5))
    assert "s5" not in message
    assert "8" in message
    assert list_selections(ds) == []


def test_a_failing_write_leaves_no_partial_file(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = _utils.atomic_write

    def failing(final_path: Path, write_fn: Callable[[Path], object]) -> None:
        def torn(temp: Path) -> None:
            _ = write_fn(temp)
            raise OSError("disk full")

        real(final_path, torn)

    monkeypatch.setattr(_utils, "atomic_write", failing)

    with pytest.raises(OSError, match="disk full"):
        save_selection(ds, "day1", ENTRIES)

    leftovers = [
        p.name
        for p in _selections_dir(ds).iterdir()
        if p.name not in {".guard", ".guard.lock"}
    ]
    assert leftovers == []
    monkeypatch.undo()
    assert list_selections(ds) == []


def test_list_skips_the_guard_and_its_lock(ds: Dataset) -> None:
    save_selection(ds, "b", ENTRIES)
    save_selection(ds, "a", ENTRIES[:1], note="n")

    directory = _selections_dir(ds)
    assert (directory / ".guard").exists()
    assert (directory / ".guard.lock").exists()
    assert [s.name for s in list_selections(ds)] == ["a", "b"]


def test_a_save_and_a_delete_each_lock_the_guard(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked: list[Path] = []

    def record(path: Path, *_: object, **__: object) -> None:
        locked.append(path)

    monkeypatch.setattr(
        selections,
        "index_lock",
        observed(selections.index_lock, record),
    )
    guard = _selections_dir(ds) / ".guard"

    save_selection(ds, "day1", ENTRIES)
    assert locked == [guard]

    delete_selection(ds, "day1")
    assert locked == [guard, guard]


def test_reads_take_no_lock(ds: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    save_selection(ds, "day1", ENTRIES)
    locked: list[Path] = []

    def record(path: Path, *_: object, **__: object) -> None:
        locked.append(path)

    monkeypatch.setattr(
        selections, "index_lock", observed(selections.index_lock, record)
    )

    _ = list_selections(ds)
    _ = read_selection(ds, "day1")
    _ = selection_scope(ds, "day1")

    assert locked == []


def test_a_missing_selection_names_the_ones_present(ds: Dataset) -> None:
    save_selection(ds, "day1", ENTRIES)
    save_selection(ds, "day2", ENTRIES[:1])

    with pytest.raises(SelectionNotFound) as caught:
        read_selection(ds, "day3")

    error = caught.value
    assert isinstance(error, ClassifiedError)
    assert isinstance(error, KeyError)
    assert error.error_kind == "not_found"
    assert error.error_reason() == "selection_not_found"
    assert error.error_detail() == {"name": "day3", "selections": ["day1", "day2"]}
    assert "day1" in str(error) and "day2" in str(error)


def test_a_missing_selection_in_a_dataset_with_none_says_so(ds: Dataset) -> None:
    with pytest.raises(SelectionNotFound) as caught:
        selection_scope(ds, "day1")

    assert caught.value.error_detail() == {"name": "day1", "selections": []}
    assert not _selections_dir(ds).exists()


def test_deleting_a_missing_selection_raises_not_found(ds: Dataset) -> None:
    with pytest.raises(SelectionNotFound):
        delete_selection(ds, "day1")


def test_a_read_only_open_lists_and_reads_without_creating_the_directory(
    ds: Dataset,
) -> None:
    saved = save_selection(ds, "day1", ENTRIES)
    fresh = make_dataset(ds.base_dir.parent / "other")
    read_only = open_dataset(fresh.manifest_path, ensure_roots=False)

    assert list_selections(read_only) == []
    with pytest.raises(SelectionNotFound):
        read_selection(read_only, "day1")
    assert not (fresh.base_dir / ".mosaic").exists()

    held = open_dataset(ds.manifest_path, ensure_roots=False)
    assert list_selections(held) == [saved]
    assert read_selection(held, "day1") == saved
    assert selection_scope(held, "day1") == Scope(entries=saved.entries)


def test_the_module_loads_no_pandas_and_imports_no_adapter_or_feature_layer() -> None:
    statement = "import mosaic.core.selections"

    assert modules_loaded_by(statement, ("pandas",)) == []
    assert (
        modules_loaded_by(
            statement, ("mosaic.agent", "mosaic.behavior", "mosaic.tracking")
        )
        == []
    )


def test_a_read_returns_what_the_file_holds(
    ds: Dataset,
) -> None:
    save_selection(ds, "day1", ENTRIES)
    path = _selections_dir(ds) / "day1.json"
    payload = json.loads(path.read_text())

    assert read_selection(ds, "day1").model_dump(mode="json") == payload


FUTURE = '{"format": "mosaic-selection/2", "name": "day2"}'


def _with_unreadable_files(ds: Dataset) -> Selection:
    """Save ``day1``, then add a future-format file, a mismatched copy and junk."""
    day1 = save_selection(ds, "day1", ENTRIES)
    directory = _selections_dir(ds)
    (directory / "day2.json").write_text(FUTURE)
    (directory / "copy.json").write_text((directory / "day1.json").read_text())
    (directory / "edited.json").write_text("not json at all")
    return day1


def test_an_unreadable_file_is_not_reported_as_missing(ds: Dataset) -> None:
    _ = _with_unreadable_files(ds)

    with pytest.raises(SelectionUnreadable) as caught:
        read_selection(ds, "day2")

    error = caught.value
    assert isinstance(error, ClassifiedError)
    assert error.error_kind == "invalid"
    assert error.error_reason() == "selection_unreadable"
    detail = error.error_detail()
    assert detail is not None
    assert detail["file"] == "day2.json"
    assert "format: Input should be 'mosaic-selection/1'" in error.reason
    assert "day2.json" in str(error)
    with pytest.raises(SelectionUnreadable):
        selection_scope(ds, "day2")
    with pytest.raises(SelectionUnreadable, match="edited.json"):
        read_selection(ds, "edited")


def test_a_file_whose_stored_name_differs_is_unreadable(ds: Dataset) -> None:
    _ = _with_unreadable_files(ds)

    with pytest.raises(SelectionUnreadable, match="day1") as caught:
        read_selection(ds, "copy")

    assert caught.value.error_detail() == {
        "file": "copy.json",
        "reason": caught.value.reason,
    }


def test_a_missing_name_does_not_list_an_unreadable_file_as_present(
    ds: Dataset,
) -> None:
    _ = _with_unreadable_files(ds)

    with pytest.raises(SelectionNotFound) as caught:
        read_selection(ds, "day3")

    assert caught.value.error_detail() == {"name": "day3", "selections": ["day1"]}


def test_a_listing_skips_unreadable_files_and_exposes_them(ds: Dataset) -> None:
    day1 = _with_unreadable_files(ds)

    assert list_selections(ds) == [day1]
    unreadable = unreadable_selections(ds)
    assert [item.file for item in unreadable] == [
        "copy.json",
        "day2.json",
        "edited.json",
    ]
    assert all(
        isinstance(item, UnreadableSelection) and item.reason for item in unreadable
    )
    assert "day1" in unreadable[0].reason


def test_no_unreadable_files_in_a_dataset_without_the_directory(ds: Dataset) -> None:
    assert unreadable_selections(ds) == []
    assert not _selections_dir(ds).exists()


def test_a_delete_removes_an_unreadable_file_under_the_lock(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _with_unreadable_files(ds)
    locked: list[Path] = []

    def record(path: Path, *_: object, **__: object) -> None:
        locked.append(path)

    monkeypatch.setattr(
        selections, "index_lock", observed(selections.index_lock, record)
    )

    delete_selection(ds, "day2")
    delete_selection(ds, "copy")

    guard = _selections_dir(ds) / ".guard"
    assert locked == [guard, guard]
    assert [item.file for item in unreadable_selections(ds)] == ["edited.json"]
    assert [s.name for s in list_selections(ds)] == ["day1"]


def test_an_unreadable_file_still_holds_its_name(ds: Dataset) -> None:
    _ = _with_unreadable_files(ds)

    with pytest.raises(SelectionExists):
        save_selection(ds, "day2", ENTRIES)
    delete_selection(ds, "day2")
    assert save_selection(ds, "day2", ENTRIES).name == "day2"
