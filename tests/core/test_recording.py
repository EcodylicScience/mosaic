"""One rule names an entry's recording: its continuous group, else the entry itself."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mosaic
from mosaic.core.recording import (
    OrderFacts,
    Recording,
    RecordingKey,
    facts_from_frame_extents,
    order_members,
    recording_name,
    recording_of,
    recording_storage_name,
    recordings,
)
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import read_manifest
from tests.helpers import (
    MediaClip,
    MockDataset,
    add_track_sequences,
    make_dataset,
    source_tree,
    write_media_index,
)


def test_a_continuous_group_is_one_divided_recording() -> None:
    """Hourly sequences named by UTC start are the members of one recording."""
    found = recordings(
        ("rec",),
        [("rec", "20261021T100000Z"), ("rec", "20261021T090000Z")],
    )

    assert found == (
        Recording(
            kind="divided",
            key=RecordingKey("rec", ""),
            name="rec",
            members=(("rec", "20261021T090000Z"), ("rec", "20261021T100000Z")),
        ),
    )


def test_calms21_splits_are_namespaces_of_undivided_recordings() -> None:
    """A group not declared continuous is a namespace of undivided recordings."""
    found = recordings((), [("train", "seq1"), ("train", "seq2"), ("test", "seqA")])

    assert [(r.kind, r.key, r.name, r.members) for r in found] == [
        ("undivided", RecordingKey("test", "seqA"), "test:seqA", (("test", "seqA"),)),
        (
            "undivided",
            RecordingKey("train", "seq1"),
            "train:seq1",
            (("train", "seq1"),),
        ),
        (
            "undivided",
            RecordingKey("train", "seq2"),
            "train:seq2",
            (("train", "seq2"),),
        ),
    ]


def test_treatment_namespaces_beside_a_continuous_group() -> None:
    """Guppies-shaped treatments stay per entry beside a declared recording."""
    found = recordings(
        ("tank1",),
        [
            ("control", "fish01"),
            ("parasite", "fish01"),
            ("tank1", "s1"),
            ("tank1", "s2"),
            ("", "pilot"),
        ],
    )

    by_name = {recording.name: recording for recording in found}
    assert by_name["control:fish01"].kind == "undivided"
    assert by_name["parasite:fish01"].kind == "undivided"
    assert by_name["pilot"].key == RecordingKey("", "pilot")
    assert by_name["tank1"].members == (("tank1", "s1"), ("tank1", "s2"))
    assert len(found) == 4


def test_a_single_member_group_is_still_divided() -> None:
    """The first hour of a store still recording keys as its recording already."""
    [recording] = recordings(("rec",), [("rec", "20261021T090000Z")])

    assert recording.kind == "divided"
    assert recording.key == RecordingKey("rec", "")
    assert recording.members == (("rec", "20261021T090000Z"),)


def test_the_kind_comes_from_the_declaration_not_from_the_group() -> None:
    assert recording_of(("rec",), "rec", "s1") == RecordingKey("rec", "")
    assert recording_of(("rec",), "ns", "s1") == RecordingKey("ns", "s1")
    assert recording_of(("rec",), "", "s1") == RecordingKey("", "s1")


def test_a_recording_is_named_by_its_group_or_its_entry_token() -> None:
    found = recordings(("rec",), [("rec", "a"), ("", "a"), ("ns", "a")])

    assert [recording_name(recording.key) for recording in found] == [
        recording.name for recording in found
    ]
    assert [recording.name for recording in found] == ["a", "ns:a", "rec"]


def test_every_recording_has_a_unique_key_and_storage_name() -> None:
    found = recordings(
        ("rec",),
        [("rec", "a"), ("rec", "b"), ("", "a"), ("ns", "a"), ("ns", "rec")],
    )

    keys = [recording.key for recording in found]
    names = [recording_storage_name(key) for key in keys]
    assert len(set(keys)) == len(keys)
    assert len(set(names)) == len(names)
    assert sorted(names) == ["a", "ns__a", "ns__rec", "rec"]


def test_a_member_listed_twice_is_one_member() -> None:
    assert order_members([("g", "s1"), ("g", "s1")], {}) == (("g", "s1"),)


_FIRST = datetime(2026, 10, 21, 9, tzinfo=UTC)
_LATER = datetime(2026, 10, 21, 10, tzinfo=UTC)


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        pytest.param(
            {
                ("g", "b"): OrderFacts(first_frame=0, start_utc=_LATER, frame_min=50),
                ("g", "a"): OrderFacts(first_frame=10, start_utc=_FIRST, frame_min=0),
            },
            ("b", "a"),
            id="first_frame-before-start_utc",
        ),
        pytest.param(
            {
                ("g", "b"): OrderFacts(start_utc=_FIRST, frame_min=50),
                ("g", "a"): OrderFacts(start_utc=_LATER, frame_min=0),
            },
            ("b", "a"),
            id="start_utc-before-frame_min",
        ),
        pytest.param(
            {("g", "b"): OrderFacts(frame_min=0), ("g", "a"): OrderFacts(frame_min=50)},
            ("b", "a"),
            id="frame_min-before-name",
        ),
        pytest.param({}, ("a", "b"), id="name-alone"),
        pytest.param(
            {("g", "b"): OrderFacts(frame_min=50)},
            ("b", "a"),
            id="a-placed-member-before-an-unplaced-one",
        ),
    ],
)
def test_each_order_fact_takes_precedence_over_the_next(
    facts: dict[tuple[str, str], OrderFacts], expected: tuple[str, ...]
) -> None:
    ordered = order_members([("g", "a"), ("g", "b")], facts)

    assert tuple(sequence for _, sequence in ordered) == expected


def test_names_sort_naturally() -> None:
    """``seg2`` precedes ``seg10``, which a plain string sort reverses."""
    members = [("g", "seg10"), ("g", "seg2"), ("g", "seg1")]

    assert order_members(members, {}) == (("g", "seg1"), ("g", "seg2"), ("g", "seg10"))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("\u00b2", ("s1", "s2", "\u00b2")),
        ("s1\u00b2", ("s1", "s1\u00b2", "s2")),
        ("\u00b2s10", ("s1", "s2", "\u00b2s10")),
    ],
)
def test_a_digit_that_is_not_decimal_sorts_as_text(
    name: str, expected: tuple[str, ...]
) -> None:
    """A superscript is a digit to ``str.isdigit`` and not a number to ``int``.

    It sorts as text, after the decimal run it follows and after every name
    that starts with a letter.
    """
    members = [("g", name), ("g", "s2"), ("g", "s1")]

    ordered = order_members(members, {})

    assert tuple(sequence for _, sequence in ordered) == expected


def test_frame_extents_give_the_tracks_order() -> None:
    facts = facts_from_frame_extents(
        {("g", "late"): (100, 199), ("g", "early"): (0, 99)}
    )

    assert order_members(facts, facts) == (("g", "early"), ("g", "late"))


def test_the_dataset_lists_media_and_tracks_entries(tmp_path: Path) -> None:
    """The universe is the media index together with the tracks index."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(group="rec", sequence="s2", filename="rec-s2.mp4"),
            MediaClip(sequence="pilot"),
        ],
    )
    add_track_sequences(ds, ("rec", "s1"), ("ns", "x"))
    ds.set_continuous_groups(["rec"])

    found = ds.recordings()

    assert [(r.kind, r.name, r.members) for r in found] == [
        ("undivided", "pilot", (("", "pilot"),)),
        ("undivided", "ns:x", (("ns", "x"),)),
        ("divided", "rec", (("rec", "s1"), ("rec", "s2"))),
    ]
    assert ds.recording_members(RecordingKey("rec", "")) == (
        ("rec", "s1"),
        ("rec", "s2"),
    )
    assert ds.recording_members(RecordingKey("ns", "x")) == (("ns", "x"),)


@pytest.mark.parametrize(
    "key",
    [RecordingKey("ns", ""), RecordingKey("", ""), RecordingKey("rec", "s1")],
)
def test_a_key_that_names_no_recording_has_no_members(
    tmp_path: Path, key: RecordingKey
) -> None:
    """An empty sequence names a divided recording, and a divided one is ``(G, "")``."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("rec", "s1"), ("ns", "x"))
    ds.set_continuous_groups(["rec"])

    assert ds.recording_members(key) == ()


def test_the_test_double_answers_by_the_same_rule(tmp_path: Path) -> None:
    ds = MockDataset(tmp_path, continuous_groups=("rec",))
    add_track_sequences(ds, ("rec", "s2"), ("rec", "s1"), ("", "a"))

    assert ds.recordings() == recordings(
        ("rec",), [("rec", "s1"), ("rec", "s2"), ("", "a")]
    )
    assert ds.recording_of("rec", "s1") == RecordingKey("rec", "")


class TestSetContinuousGroups:
    """What a declaration refuses before it is written."""

    @pytest.fixture
    def dataset(self, tmp_path: Path) -> Dataset:
        """A dataset with media for ``a__b/s1``, ``rec/s1`` and the empty-group ``pilot``."""
        ds = make_dataset(tmp_path / "ds")
        write_media_index(
            ds,
            [
                MediaClip(group="a__b", sequence="s1", filename="ab-s1.mp4"),
                MediaClip(group="rec", sequence="s1", filename="rec-s1.mp4"),
                MediaClip(group="a:b", sequence="s1", filename="a-b-s1.mp4"),
                MediaClip(sequence="pilot"),
            ],
        )
        return ds

    def test_a_group_with_entries_is_declared(self, dataset: Dataset) -> None:
        dataset.set_continuous_groups(["rec"])

        assert read_manifest(dataset.manifest_path).continuous_groups == ("rec",)

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            pytest.param("a__b", "contains '__'", id="joined"),
            pytest.param("a:b", "contains ':'", id="token"),
            pytest.param("pilot", "empty-group sequence", id="sequence-name"),
            pytest.param("absent", "no entry", id="no-entries"),
        ],
    )
    def test_a_name_is_refused(self, dataset: Dataset, name: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            dataset.set_continuous_groups(["rec", name])

        assert read_manifest(dataset.manifest_path).continuous_groups == ()

    def test_tracks_entries_count_as_entries(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path / "ds")
        add_track_sequences(ds, ("night", "s1"))

        ds.set_continuous_groups(["night"])

        assert ds.continuous_groups == ("night",)


# Each module that may read the declaration, and the functions of it that may.
# ``None`` admits the whole module. The summary and ``describe`` display the
# declared list and decide nothing from it.
_DECLARATION_READERS: dict[str, frozenset[str] | None] = {
    "core/manifest.py": None,
    "core/recording.py": None,
    "core/dataset.py": frozenset(
        {
            "continuous_groups",
            "is_continuous_group",
            "recording_kind",
            "recording_of",
            "recordings",
        }
    ),
    "core/dataset_summary.py": frozenset({"dataset_summary"}),
}
_DECLARATION_NAMES = frozenset({"continuous_groups", "is_continuous_group"})


def _declaration_reads(tree: ast.Module) -> list[tuple[str, int]]:
    """Each read of the declaration, with the function containing it."""
    reads: list[tuple[str, int]] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Attribute) and child.attr in _DECLARATION_NAMES:
                reads.append((function, child.lineno))
            visit(child, function)

    visit(tree, "")
    return reads


def test_only_the_recording_rule_reads_the_continuous_declaration() -> None:
    """Grouping by ``continuous_groups`` anywhere else is a second rule."""
    package = Path(mosaic.__file__).parent
    stray: list[str] = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        allowed = _DECLARATION_READERS.get(relative, frozenset())
        if allowed is None:
            continue
        stray.extend(
            f"{relative}:{line} in {function or '<module>'}"
            for function, line in _declaration_reads(source_tree(path))
            if function not in allowed
        )

    assert not stray, stray
