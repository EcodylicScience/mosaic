"""One grammar for naming an entry, whichever surface a user names it through.

``--entries g:seq`` on the command line and ``{"entries": ["g:seq"]}`` inside an
op's ``--params`` are the same value arriving two ways, and they used to be
parsed by two functions that disagreed: the CLI rejected a token with no ``:``
while the three tracker ops read it as a bare sequence in the empty group. Since
every dataset the control plane creates has ``group=""``, the rejected spelling
was the common case, and a user had to type a colon to say nothing.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import mosaic.core.helpers as helpers_module
from mosaic.cli._io import parse_entries
from mosaic.core.entry import (
    Entry,
    EntryKeyCollisionError,
    entry_key_collision,
    entry_token,
    parse_entry_tokens,
)
from mosaic.core.helpers import is_nameless_entry, make_entry_key
from mosaic.core.pipeline import entry_keys
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.pipeline.tracks_index import read_tracks_index, tracks_index_path
from tests.helpers import (
    MediaClip,
    add_tracks_table,
    make_dataset,
    track_table,
    write_media_index,
)

_CASES: list[tuple[list[str], list[tuple[str, str]]]] = [
    (["g:seq"], [("g", "seq")]),
    ([":seq"], [("", "seq")]),
    # A bare token is a sequence in the empty group -- the spelling the CLI used
    # to reject and the ops accepted.
    (["seq"], [("", "seq")]),
    # Split on the first colon only, so a sequence name may contain one.
    (["g:a:b"], [("g", "a:b")]),
    (["g1:s1", "g2:s2"], [("g1", "s1"), ("g2", "s2")]),
    ([], []),
]


@pytest.mark.parametrize(("tokens", "expected"), _CASES)
def test_the_grammar(tokens: list[str], expected: list[tuple[str, str]]) -> None:
    assert parse_entry_tokens(tokens) == expected


@pytest.mark.parametrize(("tokens", "expected"), _CASES)
def test_the_cli_agrees_with_the_ops(
    tokens: list[str], expected: list[tuple[str, str]]
) -> None:
    """The regression the split guarded against: two surfaces, one answer."""
    assert parse_entries(tokens) == parse_entry_tokens(tokens) == expected


@pytest.mark.parametrize("entry", [("g", "seq"), ("", "seq"), ("g", "a:b"), ("g", "")])
def test_an_entry_token_reads_back_as_its_entry(entry: tuple[str, str]) -> None:
    assert parse_entry_tokens([entry_token(*entry)]) == [entry]


def test_none_is_empty_rather_than_an_error() -> None:
    """Unset scope means every indexed entry, not a malformed one."""
    assert parse_entry_tokens(None) == []
    assert parse_entries(None) == []


def test_a_bare_token_names_the_key_it_is_stored_under() -> None:
    """Why the permissive reading is the correct one, not merely the lenient one.

    ``make_entry_key("", "seq")`` is ``"seq"``, so a bare token already spells
    the directory and filename the entry lives at. Rejecting it would have meant
    the name a user reads off disk is not a name they may type.
    """
    [(group, sequence)] = parse_entry_tokens(["seq"])

    assert make_entry_key(group, sequence) == "seq"


@pytest.mark.parametrize(
    ("group", "sequence", "nameless"),
    [("", "", True), ("", "  ", True), ("", "seq", False), ("g", "", False)],
)
def test_an_entry_is_nameless_exactly_when_its_key_is_empty(
    group: str, sequence: str, nameless: bool
) -> None:
    assert is_nameless_entry(group, sequence) is nameless
    assert (make_entry_key(group, sequence) == "") is nameless


def test_the_parser_lives_beside_the_type_it_produces() -> None:
    """Assert the one import path ``core/entry.py``'s docstring declares.

    It moved off core/helpers.py because that module imports pandas, and
    mosaic-queue parses these tokens in a process that must not.
    """
    assert not hasattr(helpers_module, "parse_entry_tokens"), (
        "core/helpers.py must not re-export it. A second import path is the "
        "compatibility alias this move avoids"
    )


def test_two_entries_can_spell_one_key() -> None:
    """The collision a write refuses: the empty group joins nothing to the name."""
    assert make_entry_key("", "R__0900") == make_entry_key("R", "0900") == "R__0900"


def test_a_new_entry_that_spells_another_entrys_key_is_refused() -> None:
    assert "'R__0900'" in (
        entry_key_collision([("", "R__0900")], [("R", "0900")]) or ""
    )
    assert "'R__0900'" in (
        entry_key_collision([("R", "0900"), ("", "R__0900")], []) or ""
    )


def test_entries_already_held_are_not_new() -> None:
    """A collision recorded before the check is rewritten without a refusal."""
    held = [("", "R__0900"), ("R", "0900")]

    assert entry_key_collision(held, held) is None
    assert entry_key_collision([("R", "1000")], held) is None


def test_a_tracks_write_refuses_the_key_of_an_entry_with_media(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset, [MediaClip(group="R", sequence="0900", filename="R-0900.mp4")]
    )

    with pytest.raises(EntryKeyCollisionError, match="R__0900"):
        add_tracks_table(dataset, "convert.a", track_table("", "R__0900"))

    assert read_tracks_index(dataset).empty


def _no_entries(_path: Path) -> set[Entry]:
    """An index read that finds no entry."""
    return set()


def test_the_tracks_check_reads_the_index_under_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row appended while a writer waits for the lock is seen by its check.

    The main thread holds the tracks index lock and appends ``("R", "0900")``
    while a second thread is writing ``("", "R__0900")``. The check before the
    lock reads an empty index here. Only the check inside the lock can refuse
    the second row.
    """
    monkeypatch.setattr(entry_keys, "index_entries", _no_entries)
    dataset = make_dataset(tmp_path / "ds")
    _ = dataset.get_root("tracks")
    raised: list[BaseException] = []

    def write_flat() -> None:
        try:
            add_tracks_table(dataset, "convert.b", track_table("", "R__0900"))
        except EntryKeyCollisionError as error:
            raised.append(error)

    writer = threading.Thread(target=write_flat)
    with index_lock(tracks_index_path(dataset)):
        writer.start()
        add_tracks_table(dataset, "convert.a", track_table("R", "0900"))
    writer.join()

    assert len(raised) == 1
    assert list(read_tracks_index(dataset)["sequence"]) == ["0900"]


def test_a_collision_already_in_the_index_still_reads(tmp_path: Path) -> None:
    """Refusal is at the write. An index that contains both entries resolves each."""
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset,
        [
            MediaClip(group="R", sequence="0900", filename="R-0900.mp4"),
            MediaClip(group="", sequence="R__0900", filename="R__0900.mp4"),
        ],
    )

    assert list(dataset.match_media_rows("R", "0900")["name"]) == ["R-0900.mp4"]
    assert list(dataset.match_media_rows("", "R__0900")["name"]) == ["R__0900.mp4"]
