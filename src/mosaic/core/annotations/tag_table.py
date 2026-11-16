"""A dataset's tag table, and the file mosaic saves it as.

Tags are named, typed values that the API attaches to a sequence (an entry) or to an
individual. What a feature reads, and what its run identity is taken over, has to be
on disk and pinned to a version: a revision of the ``tags`` label series. This module
is the format of that revision. It holds, per entry, the tags of the sequence, and per
tracks variant, one row for each linked tracker id with the individual's name, its
UID's value and its effective tags. A recording whose arenas hold individuals also
has one ``arenas`` row for each such individual.

**Stdlib only.** The module imports the standard library and
:mod:`mosaic.core.typed_attribute`, so ``core.pipeline`` reads it without a cycle and
a reader of the table does not import pandas.

**Deterministic by construction.** The payload's digest decides whether a save
changed anything, so one table always serializes to the same bytes: definitions are
sorted by name, sequences by ``(group, sequence)``, variants by name, rows by id or
ordinal, keys are sorted, the form is compact, and a float tag is always a float.
Nothing time-dependent and no database row id belongs in the payload; provenance
goes in the revision's ``origin``. Renaming an individual, a UID or a definition
changes the payload, since each is a column name or a column value.

**One check for writing and reading.** The same check refuses a table on write and
on read. It refuses a value without a definition, a value its type refuses, two
definitions of one name in one vocabulary, two sequences of one entry, and two rows
of one tracker id in one variant of a sequence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic.core.json_value import JsonValue
from mosaic.core.typed_attribute import (
    OPTION_MAX_LENGTH,
    TypedAttributeType,
    TypedAttributeValue,
    as_typed_attribute_type,
    validate_typed_value,
)

__all__ = [
    "TAG_TABLE_FORMAT",
    "TAG_TABLE_VERSION",
    "ArenaRow",
    "IndividualRow",
    "SequenceTags",
    "TagDefinition",
    "TagDefinitions",
    "TagRead",
    "TagTable",
    "check_tag_value",
    "parse_tag_table",
    "read_tag_table",
    "reserved_name_message",
    "sequence_tags_projection",
    "tag_table_payload",
]

TAG_TABLE_FORMAT: Final = "mosaic-tags"
"""The ``format`` every file of this kind declares."""

TAG_TABLE_VERSION: Final = 1
"""The newest version this module reads, and the one it writes."""


@dataclass(frozen=True, slots=True)
class TagDefinition:
    """One tag a dataset defines.

    Attributes:
        name: The tag's name, which is the name of the column that carries it.
        type: The type every value of the tag has.
    """

    name: str
    type: TypedAttributeType


@dataclass(frozen=True, slots=True)
class TagDefinitions:
    """The two vocabularies of a tag table.

    Attributes:
        sequence: The tags a sequence may carry.
        individual: The tags an individual may carry.
    """

    sequence: tuple[TagDefinition, ...]
    individual: tuple[TagDefinition, ...]


@dataclass(frozen=True, slots=True)
class IndividualRow:
    """One tracker id linked to an individual.

    An individual linked to several tracker ids has a row for each.

    Attributes:
        id: The tracker id.
        individual: The individual's name.
        uid: The value of the individual's UID, or ``None`` when it has none.
        tags: The individual's effective tags by name. A definition without an
            assignment is absent. A ``label`` that is assigned maps to ``None``.
    """

    id: int
    individual: str
    uid: str | None
    tags: Mapping[str, TypedAttributeValue]


@dataclass(frozen=True, slots=True)
class ArenaRow:
    """One individual whose arena is set, in the stead of an :class:`IndividualRow`.

    Attributes:
        arena: The arena's name.
        ordinal: The arena's ordinal in the recording's stored layout.
        individual: The individual's name.
        uid: The value of the individual's UID, or ``None`` when it has none.
        tags: The individual's effective tags, as in :class:`IndividualRow`.
    """

    arena: str
    ordinal: int
    individual: str
    uid: str | None
    tags: Mapping[str, TypedAttributeValue]


@dataclass(frozen=True, slots=True)
class SequenceTags:
    """The tags of one entry.

    Attributes:
        group: The entry's group, empty for an ungrouped entry.
        sequence: The entry's sequence name.
        tags: The sequence's tags by name, as in :class:`IndividualRow`.
        individuals: The linked tracker ids of each tracks variant, by the
            variant's name.
        arenas: The individuals whose arena is set.
    """

    group: str
    sequence: str
    tags: Mapping[str, TypedAttributeValue]
    individuals: Mapping[str, tuple[IndividualRow, ...]]
    arenas: tuple[ArenaRow, ...]


@dataclass(frozen=True, slots=True)
class TagTable:
    """Every tag of a dataset.

    Attributes:
        definitions: The tags the table defines.
        sequences: One element for each entry that has tags to report.
    """

    definitions: TagDefinitions
    sequences: tuple[SequenceTags, ...]


@dataclass(frozen=True, slots=True)
class TagRead:
    """What a feature reads of the tag table.

    Attributes:
        sequence_tags: The sequence tags whose values the feature's output depends
            on.
        writes: The identity columns that the feature writes, which a sequence tag
            may not be named after.
        renames: The feature's renames of a tag to the name of its output column,
            as ``(tag, column)`` pairs. A column may not be named after a column
            the feature writes.
    """

    sequence_tags: tuple[str, ...]
    writes: frozenset[str]
    renames: tuple[tuple[str, str], ...] = ()


def check_tag_value(type_: TypedAttributeType, value: TypedAttributeValue) -> None:
    """Check that *value* is a legal value of a tag of *type_*.

    Args:
        type_: The tag's declared type.
        value: The value to check. A ``label`` has none, and a ``bool`` is not
            an ``int``.

    Raises:
        ValueError: The type is unknown, or the value is not one the type holds.
            A ``categorical`` is a string of 1 to
            :data:`~mosaic.core.typed_attribute.OPTION_MAX_LENGTH` characters, and
            a ``text`` a non-empty string.
    """
    if type_ == "categorical":
        if not isinstance(value, str):
            msg = "categorical value must be a string"
            raise ValueError(msg)
        if not 1 <= len(value) <= OPTION_MAX_LENGTH:
            msg = (
                f"categorical value must have 1 to {OPTION_MAX_LENGTH} characters, "
                f"got {len(value)}"
            )
            raise ValueError(msg)
        return
    if type_ == "text":
        if not isinstance(value, str):
            msg = "text value must be a string"
            raise ValueError(msg)
        if not value:
            msg = "text value must not be empty"
            raise ValueError(msg)
        return
    validate_typed_value(type_, {}, value)


def reserved_name_message(name: str) -> str:
    """Why a tag named *name* cannot be read: its name is a column a step writes."""
    return (
        f"tag '{name}' has the name of a column that this step writes; "
        "rename the tag to read it"
    )


def tag_table_payload(table: TagTable) -> bytes:
    """The exact bytes a revision of *table* holds: compact, keys sorted.

    Raises:
        ValueError: The table holds a value without a definition, a value its
            type refuses, two definitions of one name in one vocabulary, two
            sequences of one entry, or two rows of one id in one variant.
    """
    _check_table(table)
    return _dump(_document(table))


def parse_tag_table(payload: bytes) -> TagTable:
    """The table that *payload* holds.

    Raises:
        ValueError: The payload is not JSON, declares another format, declares a
            version this mosaic does not read, has a missing, unknown or
            mistyped key, or holds what :func:`tag_table_payload` refuses.
    """
    try:
        document: JsonValue = json.loads(payload)
    except ValueError as exc:
        msg = f"not JSON: {exc}"
        raise ValueError(msg) from exc
    root = _as_object(document, "the document")
    header_format = root.get("format")
    if header_format != TAG_TABLE_FORMAT:
        msg = f"format is {header_format!r}, expected {TAG_TABLE_FORMAT!r}"
        raise ValueError(msg)
    version = root.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        msg = "version must be an integer"
        raise ValueError(msg)
    if version > TAG_TABLE_VERSION:
        msg = (
            f"{TAG_TABLE_FORMAT} version {version}, and this mosaic reads up to "
            f"version {TAG_TABLE_VERSION}. Upgrade mosaic to read it"
        )
        raise ValueError(msg)
    if version != TAG_TABLE_VERSION:
        msg = f"{TAG_TABLE_FORMAT} version {version} is not a version this mosaic reads"
        raise ValueError(msg)
    _only_keys(root, {"format", "version", "definitions", "sequences"}, "the document")
    definitions = _as_object(_required(root, "definitions"), "definitions")
    _only_keys(definitions, {"sequence", "individual"}, "definitions")
    table = TagTable(
        definitions=TagDefinitions(
            sequence=_parse_definitions(_required(definitions, "sequence")),
            individual=_parse_definitions(_required(definitions, "individual")),
        ),
        sequences=tuple(
            _parse_sequence(item)
            for item in _as_array(_required(root, "sequences"), "sequences")
        ),
    )
    _check_table(table)
    return table


def read_tag_table(path: Path) -> TagTable:
    """The table in the file at *path*.

    Raises:
        ValueError: The file is not a valid ``mosaic-tags`` file. The message
            names *path*.
    """
    try:
        return parse_tag_table(path.read_bytes())
    except ValueError as exc:
        msg = f"{path} is not a {TAG_TABLE_FORMAT} file: {exc}"
        raise ValueError(msg) from exc


def sequence_tags_projection(table: TagTable, names: Iterable[str]) -> bytes:
    """The bytes that a run's identity over the sequence tags *names* is taken over.

    The projection holds the definitions of the named tags and, for every sequence
    of the table, its ``group``, its ``sequence`` and the values of the named tags
    that it holds. A change to an unnamed tag, an individual or an arena leaves it
    unchanged, and adding a sequence changes it.

    Args:
        table: The tag table.
        names: The sequence tags to project, in any order.

    Raises:
        ValueError: A name is not a sequence definition of *table*.
    """
    wanted = sorted(set(names))
    defined = {definition.name: definition for definition in table.definitions.sequence}
    undefined = [name for name in wanted if name not in defined]
    if undefined:
        msg = f"not sequence tags of the dataset: {undefined}"
        raise ValueError(msg)
    types: dict[str, TypedAttributeType] = {name: defined[name].type for name in wanted}
    document: dict[str, JsonValue] = {
        "definitions": [{"name": name, "type": types[name]} for name in wanted],
        "sequences": [
            {
                "group": sequence.group,
                "sequence": sequence.sequence,
                "tags": _tag_values(
                    {
                        name: sequence.tags[name]
                        for name in wanted
                        if name in sequence.tags
                    },
                    types,
                ),
            }
            for sequence in sorted(
                table.sequences, key=lambda item: (item.group, item.sequence)
            )
        ],
    }
    return _dump(document)


def _dump(document: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _check_table(table: TagTable) -> None:
    sequence_types = _vocabulary(table.definitions.sequence, "sequence")
    individual_types = _vocabulary(table.definitions.individual, "individual")
    entries: set[tuple[str, str]] = set()
    for sequence in table.sequences:
        entry = (sequence.group, sequence.sequence)
        if entry in entries:
            msg = f"two sequences of the entry {entry}"
            raise ValueError(msg)
        entries.add(entry)
        where = f"sequence {sequence.sequence!r} of group {sequence.group!r}"
        _check_tags(sequence.tags, sequence_types, where)
        for variant, rows in sequence.individuals.items():
            ids: set[int] = set()
            for row in rows:
                if row.id in ids:
                    msg = f"{where}, variant {variant!r} has two rows of id {row.id}"
                    raise ValueError(msg)
                ids.add(row.id)
                _check_tags(
                    row.tags,
                    individual_types,
                    f"{where}, variant {variant!r}, id {row.id}",
                )
        for arena in sequence.arenas:
            _check_tags(arena.tags, individual_types, f"{where}, arena {arena.arena!r}")


def _vocabulary(
    definitions: Sequence[TagDefinition], label: str
) -> dict[str, TypedAttributeType]:
    types: dict[str, TypedAttributeType] = {}
    for definition in definitions:
        if as_typed_attribute_type(definition.type) is None:
            msg = (
                f"{label} tag {definition.name!r} has unknown type {definition.type!r}"
            )
            raise ValueError(msg)
        if definition.name in types:
            msg = f"two definitions of the {label} tag {definition.name!r}"
            raise ValueError(msg)
        types[definition.name] = definition.type
    return types


def _check_tags(
    tags: Mapping[str, TypedAttributeValue],
    types: Mapping[str, TypedAttributeType],
    where: str,
) -> None:
    for name, value in tags.items():
        type_ = types.get(name)
        if type_ is None:
            msg = f"{where} has a value for the tag {name!r}, which is not defined"
            raise ValueError(msg)
        try:
            check_tag_value(type_, value)
        except ValueError as exc:
            msg = f"{where}, tag {name!r}: {exc}"
            raise ValueError(msg) from exc


def _tag_values(
    tags: Mapping[str, TypedAttributeValue], types: Mapping[str, TypedAttributeType]
) -> dict[str, JsonValue]:
    """*tags* as payload values, a float tag always a float."""
    return {
        name: float(value)
        if types[name] == "float" and isinstance(value, (int, float))
        else value
        for name, value in tags.items()
    }


def _document(table: TagTable) -> dict[str, JsonValue]:
    sequence_types: dict[str, TypedAttributeType] = {
        definition.name: definition.type for definition in table.definitions.sequence
    }
    individual_types: dict[str, TypedAttributeType] = {
        definition.name: definition.type for definition in table.definitions.individual
    }
    definitions: dict[str, JsonValue] = {
        "sequence": _definitions_document(table.definitions.sequence),
        "individual": _definitions_document(table.definitions.individual),
    }
    sequences: list[JsonValue] = [
        _sequence_document(sequence, sequence_types, individual_types)
        for sequence in sorted(
            table.sequences, key=lambda item: (item.group, item.sequence)
        )
    ]
    return {
        "format": TAG_TABLE_FORMAT,
        "version": TAG_TABLE_VERSION,
        "definitions": definitions,
        "sequences": sequences,
    }


def _definitions_document(definitions: Sequence[TagDefinition]) -> list[JsonValue]:
    return [
        {"name": definition.name, "type": definition.type}
        for definition in sorted(definitions, key=lambda definition: definition.name)
    ]


def _ordered(rows: Iterable[tuple[int, dict[str, JsonValue]]]) -> list[JsonValue]:
    """The rows by their id or ordinal, ties broken by their serialized form."""
    keyed = sorted((key, _dump(row).decode("utf-8")) for key, row in rows)
    return [json.loads(text) for _, text in keyed]


def _sequence_document(
    sequence: SequenceTags,
    sequence_types: Mapping[str, TypedAttributeType],
    individual_types: Mapping[str, TypedAttributeType],
) -> dict[str, JsonValue]:
    individuals: dict[str, JsonValue] = {
        variant: _ordered(
            (
                row.id,
                {
                    "id": row.id,
                    "individual": row.individual,
                    "uid": row.uid,
                    "tags": _tag_values(row.tags, individual_types),
                },
            )
            for row in rows
        )
        for variant, rows in sorted(sequence.individuals.items())
    }
    arenas = _ordered(
        (
            arena.ordinal,
            {
                "arena": arena.arena,
                "ordinal": arena.ordinal,
                "individual": arena.individual,
                "uid": arena.uid,
                "tags": _tag_values(arena.tags, individual_types),
            },
        )
        for arena in sequence.arenas
    )
    return {
        "group": sequence.group,
        "sequence": sequence.sequence,
        "tags": _tag_values(sequence.tags, sequence_types),
        "individuals": individuals,
        "arenas": arenas,
    }


def _as_object(value: JsonValue, where: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        msg = f"{where} must be an object"
        raise ValueError(msg)
    return value


def _as_array(value: JsonValue, where: str) -> list[JsonValue]:
    if not isinstance(value, list):
        msg = f"{where} must be an array"
        raise ValueError(msg)
    return value


def _only_keys(mapping: Mapping[str, JsonValue], allowed: set[str], where: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        msg = f"{where} has keys this mosaic does not read: {extra}"
        raise ValueError(msg)


def _required(mapping: Mapping[str, JsonValue], key: str) -> JsonValue:
    if key not in mapping:
        msg = f"missing {key!r}"
        raise ValueError(msg)
    return mapping[key]


def _as_str(value: JsonValue, where: str) -> str:
    if not isinstance(value, str):
        msg = f"{where} must be a string"
        raise ValueError(msg)
    return value


def _as_optional_str(value: JsonValue, where: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, where)


def _as_int(value: JsonValue, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{where} must be an integer"
        raise ValueError(msg)
    return value


def _parse_definitions(value: JsonValue) -> tuple[TagDefinition, ...]:
    definitions: list[TagDefinition] = []
    for item in _as_array(value, "a vocabulary"):
        fields = _as_object(item, "a definition")
        _only_keys(fields, {"name", "type"}, "a definition")
        name = _as_str(_required(fields, "name"), "a definition's name")
        type_ = _as_str(_required(fields, "type"), "a definition's type")
        declared = as_typed_attribute_type(type_)
        if declared is None:
            msg = f"tag {name!r} has unknown type {type_!r}"
            raise ValueError(msg)
        definitions.append(TagDefinition(name, declared))
    return tuple(definitions)


def _parse_tags(value: JsonValue, where: str) -> dict[str, TypedAttributeValue]:
    tags: dict[str, TypedAttributeValue] = {}
    for name, item in _as_object(value, f"{where} tags").items():
        if item is None or isinstance(item, (str, int, float, bool)):
            tags[name] = item
        else:
            msg = f"{where} has a value for the tag {name!r} that is not a scalar"
            raise ValueError(msg)
    return tags


def _parse_individual(item: JsonValue) -> IndividualRow:
    fields = _as_object(item, "an individual row")
    _only_keys(fields, {"id", "individual", "uid", "tags"}, "an individual row")
    return IndividualRow(
        id=_as_int(_required(fields, "id"), "an individual row's id"),
        individual=_as_str(_required(fields, "individual"), "an individual's name"),
        uid=_as_optional_str(_required(fields, "uid"), "an individual's uid"),
        tags=_parse_tags(_required(fields, "tags"), "an individual row"),
    )


def _parse_arena(item: JsonValue) -> ArenaRow:
    fields = _as_object(item, "an arena row")
    _only_keys(
        fields, {"arena", "ordinal", "individual", "uid", "tags"}, "an arena row"
    )
    return ArenaRow(
        arena=_as_str(_required(fields, "arena"), "an arena's name"),
        ordinal=_as_int(_required(fields, "ordinal"), "an arena's ordinal"),
        individual=_as_str(_required(fields, "individual"), "an individual's name"),
        uid=_as_optional_str(_required(fields, "uid"), "an individual's uid"),
        tags=_parse_tags(_required(fields, "tags"), "an arena row"),
    )


def _parse_sequence(item: JsonValue) -> SequenceTags:
    fields = _as_object(item, "a sequence")
    _only_keys(
        fields, {"group", "sequence", "tags", "individuals", "arenas"}, "a sequence"
    )
    individuals = _as_object(_required(fields, "individuals"), "individuals")
    return SequenceTags(
        group=_as_str(_required(fields, "group"), "a sequence's group"),
        sequence=_as_str(_required(fields, "sequence"), "a sequence's name"),
        tags=_parse_tags(_required(fields, "tags"), "a sequence"),
        individuals={
            variant: tuple(
                _parse_individual(row) for row in _as_array(rows, "a variant's rows")
            )
            for variant, rows in individuals.items()
        },
        arenas=tuple(
            _parse_arena(row)
            for row in _as_array(_required(fields, "arenas"), "arenas")
        ),
    )
