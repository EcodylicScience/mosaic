"""The ``mosaic-tags`` payload: one table is one sequence of bytes, and what is
written reads back.

A revision of the ``tags`` series is the whole tag table of a dataset. Its digest
decides whether a save changed anything and what a tag-reading run's identity is
taken over, so the payload must not depend on the order a writer happened to hold
things in, and must refuse a table that does not say what it means.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.annotations.tag_table import (
    TAG_TABLE_FORMAT,
    TAG_TABLE_VERSION,
    ArenaRow,
    IndividualRow,
    SequenceTags,
    TagDefinition,
    TagDefinitions,
    TagRead,
    TagTable,
    check_tag_value,
    parse_tag_table,
    read_tag_table,
    sequence_tags_projection,
    tag_table_payload,
)
from mosaic.core.json_value import JsonValue
from mosaic.core.typed_attribute import (
    OPTION_MAX_LENGTH,
    TypedAttributeType,
    TypedAttributeValue,
)
from tests.helpers import modules_loaded_by

SKETCH: Final[dict[str, JsonValue]] = {
    "format": "mosaic-tags",
    "version": 1,
    "definitions": {
        "sequence": [{"name": "condition", "type": "categorical"}],
        "individual": [
            {"name": "focal", "type": "bool"},
            {"name": "sex", "type": "categorical"},
        ],
    },
    "sequences": [
        {
            "group": "",
            "sequence": "day1-tank3",
            "tags": {"condition": "treated"},
            "individuals": {
                "ultralytics.0.4-1a2b3c4d5e": [
                    {
                        "id": 1,
                        "individual": "fish 1",
                        "uid": "F1",
                        "tags": {"focal": True, "sex": "f"},
                    },
                    {
                        "id": 7,
                        "individual": "fish 1",
                        "uid": "F1",
                        "tags": {"focal": True, "sex": "f"},
                    },
                ]
            },
            "arenas": [],
        }
    ],
}


def _compact(document: dict[str, JsonValue]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _table() -> TagTable:
    """A table with two sequences, two variants, arena rows and every tag kind."""
    return TagTable(
        definitions=TagDefinitions(
            sequence=(
                TagDefinition("condition", "categorical"),
                TagDefinition("note", "text"),
                TagDefinition("batch", "int"),
            ),
            individual=(
                TagDefinition("sex", "categorical"),
                TagDefinition("focal", "label"),
                TagDefinition("weight", "float"),
            ),
        ),
        sequences=(
            SequenceTags(
                group="g2",
                sequence="b",
                tags={"condition": "control", "batch": 2},
                individuals={
                    "v-b": (
                        IndividualRow(3, "fish 3", None, {"sex": "m"}),
                        IndividualRow(2, "fish 2", "F2", {"focal": None}),
                    ),
                    "v-a": (IndividualRow(1, "fish 1", "F1", {"weight": 2}),),
                },
                arenas=(
                    ArenaRow("right", 1, "fish 3", None, {"sex": "m"}),
                    ArenaRow("left", 0, "fish 2", "F2", {"focal": None}),
                ),
            ),
            SequenceTags(
                group="",
                sequence="a",
                tags={"note": "first"},
                individuals={},
                arenas=(),
            ),
        ),
    )


def _reversed(table: TagTable) -> TagTable:
    """The same table with every ordering a writer could vary reversed."""
    return TagTable(
        definitions=TagDefinitions(
            sequence=tuple(reversed(table.definitions.sequence)),
            individual=tuple(reversed(table.definitions.individual)),
        ),
        sequences=tuple(
            replace(
                sequence,
                tags=dict(reversed(list(sequence.tags.items()))),
                individuals={
                    variant: tuple(reversed(rows))
                    for variant, rows in reversed(list(sequence.individuals.items()))
                },
                arenas=tuple(reversed(sequence.arenas)),
            )
            for sequence in reversed(table.sequences)
        ),
    )


def test_the_sketch_parses_and_its_payload_is_its_compact_sorted_form() -> None:
    payload = _compact(SKETCH)

    table = parse_tag_table(json.dumps(SKETCH, indent=2).encode())

    assert table.definitions.sequence == (TagDefinition("condition", "categorical"),)
    assert table.definitions.individual == (
        TagDefinition("focal", "bool"),
        TagDefinition("sex", "categorical"),
    )
    (sequence,) = table.sequences
    assert (sequence.group, sequence.sequence) == ("", "day1-tank3")
    assert sequence.tags == {"condition": "treated"}
    assert sequence.individuals["ultralytics.0.4-1a2b3c4d5e"][1] == IndividualRow(
        7, "fish 1", "F1", {"focal": True, "sex": "f"}
    )
    assert tag_table_payload(table) == payload


def test_the_payload_declares_its_format_and_version() -> None:
    document = json.loads(tag_table_payload(_table()))

    assert document["format"] == TAG_TABLE_FORMAT == "mosaic-tags"
    assert document["version"] == TAG_TABLE_VERSION == 1


def test_one_table_in_any_order_is_one_payload() -> None:
    table = _table()

    assert tag_table_payload(_reversed(table)) == tag_table_payload(table)


def test_the_payload_sorts_definitions_sequences_variants_and_rows() -> None:
    document = json.loads(tag_table_payload(_table()))

    assert [d["name"] for d in document["definitions"]["sequence"]] == [
        "batch",
        "condition",
        "note",
    ]
    assert [d["name"] for d in document["definitions"]["individual"]] == [
        "focal",
        "sex",
        "weight",
    ]
    assert [(s["group"], s["sequence"]) for s in document["sequences"]] == [
        ("", "a"),
        ("g2", "b"),
    ]
    second = document["sequences"][1]
    assert list(second["individuals"]) == ["v-a", "v-b"]
    assert [row["id"] for row in second["individuals"]["v-b"]] == [2, 3]
    assert [row["ordinal"] for row in second["arenas"]] == [0, 1]


def test_a_float_tag_serializes_as_a_float() -> None:
    document = json.loads(tag_table_payload(_table()))

    row = document["sequences"][1]["individuals"]["v-a"][0]
    assert row["tags"]["weight"] == 2.0
    assert isinstance(row["tags"]["weight"], float)


def test_an_arena_row_keeps_its_fields_through_a_round_trip() -> None:
    table = _table()

    again = parse_tag_table(tag_table_payload(table))

    (_, second) = again.sequences
    assert second.arenas[0] == ArenaRow("left", 0, "fish 2", "F2", {"focal": None})
    assert tag_table_payload(again) == tag_table_payload(table)


def test_renaming_an_individual_a_uid_or_a_definition_changes_the_payload() -> None:
    table = _table()
    base = tag_table_payload(table)
    second = table.sequences[0]
    rows = second.individuals["v-a"]

    renamed_individual = replace(
        table,
        sequences=(
            replace(
                second,
                individuals={
                    **second.individuals,
                    "v-a": (replace(rows[0], individual="fish one"),),
                },
            ),
            table.sequences[1],
        ),
    )
    renamed_uid = replace(
        table,
        sequences=(
            replace(
                second,
                individuals={
                    **second.individuals,
                    "v-a": (replace(rows[0], uid="F1b"),),
                },
            ),
            table.sequences[1],
        ),
    )
    renamed_definition = replace(
        table,
        definitions=replace(
            table.definitions,
            sequence=(
                TagDefinition("cohort", "categorical"),
                *table.definitions.sequence[1:],
            ),
        ),
        sequences=(
            replace(second, tags={"cohort": "control", "batch": 2}),
            table.sequences[1],
        ),
    )

    assert tag_table_payload(renamed_individual) != base
    assert tag_table_payload(renamed_uid) != base
    assert tag_table_payload(renamed_definition) != base


def test_a_value_without_a_definition_is_refused() -> None:
    table = _table()
    first, second = table.sequences
    row = first.individuals["v-a"][0]
    on_sequence = replace(first, tags={**first.tags, "undeclared": "x"})
    on_individual = replace(
        first,
        individuals={
            **first.individuals,
            "v-a": (replace(row, tags={"undeclared": 1}),),
        },
    )
    on_arena = replace(
        first, arenas=(replace(first.arenas[0], tags={"undeclared": 1}),)
    )

    for broken in (on_sequence, on_individual, on_arena):
        with pytest.raises(ValueError, match="undeclared"):
            tag_table_payload(replace(table, sequences=(broken, second)))


def test_a_value_its_type_refuses_is_refused() -> None:
    table = _table()
    first, second = table.sequences
    sequence_tags = replace(first, tags={"batch": True})
    individual_tags = replace(
        first,
        individuals={
            **first.individuals,
            "v-a": (replace(first.individuals["v-a"][0], tags={"sex": 3}),),
        },
    )

    for broken in (sequence_tags, individual_tags):
        with pytest.raises(ValueError, match="batch|sex"):
            tag_table_payload(replace(table, sequences=(broken, second)))


@pytest.mark.parametrize("vocabulary", ["sequence", "individual"])
def test_two_definitions_of_one_name_in_one_vocabulary_are_refused(
    vocabulary: str,
) -> None:
    table = _table()
    definitions = replace(
        table.definitions,
        **{
            vocabulary: (
                *getattr(table.definitions, vocabulary),
                TagDefinition("twice", "bool"),
                TagDefinition("twice", "text"),
            )
        },
    )

    with pytest.raises(ValueError, match="twice"):
        tag_table_payload(replace(table, definitions=definitions))


def test_one_name_may_be_defined_for_sequences_and_for_individuals() -> None:
    table = TagTable(
        definitions=TagDefinitions(
            sequence=(TagDefinition("sex", "text"),),
            individual=(TagDefinition("sex", "categorical"),),
        ),
        sequences=(),
    )

    assert parse_tag_table(tag_table_payload(table)) == table


def test_two_sequences_of_one_entry_are_refused() -> None:
    table = _table()
    first, _ = table.sequences

    with pytest.raises(ValueError, match="two sequences"):
        tag_table_payload(replace(table, sequences=(first, first)))


def test_two_rows_of_one_id_in_one_variant_are_refused() -> None:
    table = _table()
    first, second = table.sequences
    row = first.individuals["v-a"][0]
    twice = replace(first, individuals={"v-a": (row, replace(row, uid="other"))})
    broken = replace(table, sequences=(twice, second))

    with pytest.raises(ValueError, match=r"v-a.*id 1|id 1.*v-a"):
        tag_table_payload(broken)
    with pytest.raises(ValueError, match="'b'"):
        tag_table_payload(broken)


def test_parsing_refuses_two_rows_of_one_id_in_one_variant() -> None:
    row: JsonValue = {"id": 1, "individual": "x", "uid": None, "tags": {}}
    rows: JsonValue = [row, row]
    individuals: JsonValue = {"v": rows}
    sketched = SKETCH["sequences"]
    assert isinstance(sketched, list)
    only = sketched[0]
    assert isinstance(only, dict)
    twice: JsonValue = {**only, "individuals": individuals}

    with pytest.raises(ValueError, match="id 1"):
        parse_tag_table(_document(sequences=[twice]))


def test_one_id_may_recur_across_variants_and_one_ordinal_across_individuals() -> None:
    table = _table()
    first, second = table.sequences
    row = first.individuals["v-a"][0]
    arena = first.arenas[0]
    shared = replace(
        first,
        individuals={**first.individuals, "v-c": (row,)},
        arenas=(arena, replace(arena, individual="fish 9")),
    )

    again = parse_tag_table(
        tag_table_payload(replace(table, sequences=(shared, second)))
    )

    assert len(again.sequences[1].arenas) == 2


def test_a_float_tag_of_2_and_2_0_gives_one_payload_and_one_projection() -> None:
    def table(value: float) -> TagTable:
        return TagTable(
            definitions=TagDefinitions(
                sequence=(TagDefinition("dose", "float"),), individual=()
            ),
            sequences=(SequenceTags("", "a", {"dose": value}, {}, ()),),
        )

    assert tag_table_payload(table(2)) == tag_table_payload(table(2.0))
    assert sequence_tags_projection(table(2), ["dose"]) == sequence_tags_projection(
        table(2.0), ["dose"]
    )


def _document(
    *,
    sequences: list[JsonValue] | None = None,
    sequence_definitions: list[JsonValue] | None = None,
    sequence_tags: dict[str, JsonValue] | None = None,
    extra: dict[str, JsonValue] | None = None,
    **header: JsonValue,
) -> bytes:
    """The sketch with one part replaced, as the bytes of a file."""
    document: dict[str, JsonValue] = {**SKETCH, **header, **(extra or {})}
    definitions = SKETCH["definitions"]
    assert isinstance(definitions, dict)
    if sequence_definitions is not None:
        document["definitions"] = {**definitions, "sequence": sequence_definitions}
    sketched = SKETCH["sequences"]
    assert isinstance(sketched, list)
    if sequences is not None:
        document["sequences"] = sequences
    if sequence_tags is not None:
        (only,) = sketched
        assert isinstance(only, dict)
        document["sequences"] = [{**only, "tags": sequence_tags}]
    return _compact(document)


def test_parsing_refuses_another_format_and_a_newer_or_older_version() -> None:
    with pytest.raises(ValueError, match="mosaic-tags"):
        parse_tag_table(_document(format="mosaic-pose-annotations"))
    with pytest.raises(ValueError, match="version 2"):
        parse_tag_table(_document(version=2))
    with pytest.raises(ValueError, match="version"):
        parse_tag_table(_document(version=0))


def test_parsing_refuses_what_writing_refuses() -> None:
    sketched = SKETCH["sequences"]
    assert isinstance(sketched, list)
    twice: JsonValue = {"name": "x", "type": "bool"}
    refused = (
        _document(sequence_tags={"nothing": "x"}),
        _document(sequence_tags={"condition": 3}),
        _document(sequences=[sketched[0], sketched[0]]),
        _document(sequence_definitions=[twice, twice]),
    )

    for payload in refused:
        with pytest.raises(ValueError):
            parse_tag_table(payload)


def test_parsing_refuses_a_key_it_does_not_know_and_a_malformed_file() -> None:
    color: JsonValue = {"name": "x", "type": "color"}
    unknown_type = _document(sequence_definitions=[color])
    refused = (
        _document(extra={"surprise": 1}),
        unknown_type,
        _document(sequences=[{"group": "", "sequence": "a"}]),
        b"[]",
        b"not json",
    )

    for payload in refused:
        with pytest.raises(ValueError):
            parse_tag_table(payload)


def test_reading_a_file_names_the_path_in_a_refusal(tmp_path: Path) -> None:
    good = tmp_path / "tags.mosaic.json"
    good.write_bytes(tag_table_payload(_table()))
    bad = tmp_path / "bad.mosaic.json"
    bad.write_bytes(b'{"format": "other", "version": 1}')

    assert read_tag_table(good) == parse_tag_table(good.read_bytes())
    with pytest.raises(ValueError, match="bad.mosaic.json"):
        read_tag_table(bad)


@pytest.mark.parametrize(
    ("type_", "value"),
    [
        ("int", 3),
        ("int", -3),
        ("float", 0.5),
        ("float", 2),
        ("bool", False),
        ("categorical", "a"),
        ("categorical", "x" * OPTION_MAX_LENGTH),
        ("text", "any non-empty text, as long as it likes " * 20),
        ("label", None),
    ],
)
def test_check_tag_value_accepts(
    type_: TypedAttributeType, value: TypedAttributeValue
) -> None:
    check_tag_value(type_, value)


@pytest.mark.parametrize(
    ("type_", "value"),
    [
        ("categorical", "x" * (OPTION_MAX_LENGTH + 1)),
        ("categorical", ""),
        ("categorical", 1),
        ("text", ""),
        ("text", 1),
        ("int", True),
        ("int", 1.5),
        ("int", "1"),
        ("float", True),
        ("float", float("nan")),
        ("float", float("inf")),
        ("bool", 1),
        ("bool", None),
        ("label", True),
        ("label", ""),
    ],
)
def test_check_tag_value_refuses(
    type_: TypedAttributeType, value: TypedAttributeValue
) -> None:
    with pytest.raises(ValueError):
        check_tag_value(type_, value)


def test_the_projection_follows_the_named_tags_only() -> None:
    table = _table()
    first, second = table.sequences
    base = sequence_tags_projection(table, ["condition"])

    named_changed = replace(
        table,
        sequences=(replace(first, tags={**first.tags, "condition": "treated"}), second),
    )
    unnamed_changed = replace(
        table,
        sequences=(replace(first, tags={**first.tags, "batch": 9}), second),
    )
    individual_changed = replace(
        table,
        sequences=(replace(first, individuals={}), second),
    )
    added = replace(
        table,
        sequences=(
            *table.sequences,
            SequenceTags("g3", "c", {}, {}, ()),
        ),
    )

    assert sequence_tags_projection(named_changed, ["condition"]) != base
    assert sequence_tags_projection(unnamed_changed, ["condition"]) == base
    assert sequence_tags_projection(individual_changed, ["condition"]) == base
    assert sequence_tags_projection(added, ["condition"]) != base


def test_the_projection_does_not_depend_on_order_or_repeated_names() -> None:
    table = _table()

    assert sequence_tags_projection(
        _reversed(table), ["note", "condition"]
    ) == sequence_tags_projection(table, ["condition", "note", "condition"])


def test_the_projection_carries_the_definitions_of_the_named_tags() -> None:
    table = _table()
    retyped = replace(
        table,
        definitions=replace(
            table.definitions,
            sequence=(
                TagDefinition("condition", "text"),
                *table.definitions.sequence[1:],
            ),
        ),
    )

    assert sequence_tags_projection(retyped, ["condition"]) != (
        sequence_tags_projection(table, ["condition"])
    )


def test_the_projection_refuses_a_name_that_is_not_a_sequence_definition() -> None:
    table = _table()

    with pytest.raises(ValueError, match="sex"):
        sequence_tags_projection(table, ["condition", "sex"])
    with pytest.raises(ValueError, match="missing"):
        sequence_tags_projection(table, ["missing"])


def test_a_tag_read_names_the_tags_and_the_columns_a_step_writes() -> None:
    read = TagRead(sequence_tags=("condition",), writes=frozenset({"group"}))

    assert read.sequence_tags == ("condition",)
    assert read.writes == frozenset({"group"})
    assert read.renames == ()


def test_a_tag_read_holds_its_renames_as_pairs_and_stays_hashable() -> None:
    read = TagRead(
        sequence_tags=("condition",),
        writes=frozenset({"group"}),
        renames=(("condition", "Condition"),),
    )

    assert read.renames == (("condition", "Condition"),)
    assert hash(read) == hash(read)


def test_importing_the_payload_module_loads_neither_pandas_nor_the_pipeline() -> None:
    statement = "import mosaic.core.annotations.tag_table"

    loaded = modules_loaded_by(
        statement,
        (
            "pandas",
            "numpy",
            "pydantic",
            "mosaic.core.pipeline",
            "mosaic.core.pipeline.label_series_index",
        ),
    )

    assert loaded == []
