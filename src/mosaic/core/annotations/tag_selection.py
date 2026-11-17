"""Selecting entries by sequence tag: the table to read and the scope it lowers to.

A selection names sequence tags and the values wanted of each. It is lowered
against one revision of the tag table to the entries that hold them, so a run that
is asked for the selection has the identity of the same entries named one by one.

:func:`selection_tag_table` reads the revision. :func:`tagged_scope` lowers a
selection to a :class:`~mosaic.core.scope.Scope` of entries, and
:func:`tagged_recordings` lists the recordings those entries make up, refusing a
divided recording that the selection covers in part. A value given as text is read
by its tag's type before the entries are matched, so ``03`` selects the number 3 of
an ``int`` tag and the value ``03`` of a ``categorical`` tag. Every refusal of the
request is a :class:`TagSelectionInvalid`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Final

from mosaic.core.annotations.sequence_tags import (
    SequenceTagRevisionNotFound,
    read_sequence_tag_table,
    read_tag_text,
)
from mosaic.core.annotations.tag_table import TagTable, entries_where
from mosaic.core.entry import Entry
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.tag_reads import ResolvedTagTable
from mosaic.core.recording import RecordingKey
from mosaic.core.scope import Scope
from mosaic.core.typed_attribute import TypedAttributeType, TypedAttributeValue

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "TagSelectionInvalid",
    "selection_tag_table",
    "tagged_recordings",
    "tagged_scope",
]

_KIND: Final = "tag selection"


class TagSelectionInvalid(ClassifiedError, ValueError):
    """A tag selection that the request must change: it names no usable tags or no entry.

    Attributes:
        revision: The revision of the tag table that was read, or ``None`` when
            none was.
        tags: The wanted values by tag name, or ``None`` when the refusal is not
            about a selection.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(
        self,
        message: str,
        *,
        revision: int | None = None,
        tags: Mapping[str, Sequence[TypedAttributeValue]] | None = None,
    ) -> None:
        super().__init__(message)
        self.revision: int | None = revision
        self.tags: dict[str, list[JsonValue]] | None = (
            None
            if tags is None
            else {name: list(values) for name, values in tags.items()}
        )

    def error_detail(self) -> dict[str, JsonValue] | None:
        """Return the revision read and the tags selected, or ``None`` without a selection."""
        if self.tags is None:
            return None
        return {"revision": self.revision, "tags": dict(self.tags)}


def selection_tag_table(ds: Dataset, revision: int | None = None) -> ResolvedTagTable:
    """The revision of the tag table that a selection is lowered against.

    Args:
        ds: The dataset whose tags are read.
        revision: The revision to read, or ``None`` for the latest.

    Raises:
        TagSelectionInvalid: No revision is indexed, or *revision* is not.
        SequenceTagTableUnavailable: The table cannot be read as one revision of
            one dataset.
    """
    try:
        resolved = read_sequence_tag_table(ds, revision, kind=_KIND)
    except SequenceTagRevisionNotFound as exc:
        raise TagSelectionInvalid(str(exc)) from exc
    if resolved is None:
        msg = (
            "no revision of the sequence tags is indexed here; import them with "
            "`mosaic sequence-tags import`"
        )
        raise TagSelectionInvalid(msg)
    return resolved


def tagged_scope(
    resolved: ResolvedTagTable,
    where: Mapping[str, Sequence[TypedAttributeValue]],
    *,
    groups: Sequence[str] | None = None,
) -> Scope:
    """The scope of the entries that hold the tags in *where*.

    Names combine as AND and the values of one name as OR. A value given as text is
    read by its tag's type. A ``label`` takes no values and matches where assigned.

    Args:
        resolved: The tag table to select from.
        where: The wanted values by sequence tag name.
        groups: Keep only the selected entries of these groups.

    Raises:
        TagSelectionInvalid: *where* names no tag or an unknown one, a value is
            refused by its tag's type, or no entry is selected.
    """
    entries = _selected(resolved, where, groups)
    return Scope(entries=entries)


def tagged_recordings(
    ds: Dataset,
    resolved: ResolvedTagTable,
    where: Mapping[str, Sequence[TypedAttributeValue]],
) -> tuple[RecordingKey, ...]:
    """The recordings that the entries holding the tags in *where* make up.

    There is one key for each undivided entry and one for each divided recording.

    Args:
        ds: The dataset the entries belong to.
        resolved: The tag table to select from.
        where: The wanted values by sequence tag name, as for :func:`tagged_scope`.

    Raises:
        TagSelectionInvalid: As :func:`tagged_scope`, or a divided recording holds
            a sequence that the selection leaves out. The message names the
            recording and the values its sequences hold of each selected tag.
    """
    selected = set(_selected(resolved, where, None))
    keys = sorted({ds.recording_of(group, sequence) for group, sequence in selected})
    divided = [key for key in keys if ds.recording_kind(key.group) == "divided"]
    if not divided:
        return tuple(keys)
    universe = ds.recording_entries()
    for key in divided:
        members = ds.recording_members(key, universe)
        if all(member in selected for member in members):
            continue
        raise TagSelectionInvalid(
            _partial_message(key, members, selected, resolved.table, where),
            revision=resolved.revision,
            tags=where,
        )
    return tuple(keys)


def _selected(
    resolved: ResolvedTagTable,
    where: Mapping[str, Sequence[TypedAttributeValue]],
    groups: Sequence[str] | None,
) -> list[Entry]:
    """The selected entries, sorted, of *groups* when they are given.

    Raises:
        TagSelectionInvalid: The selection is refused, or it selects no entry.
    """
    try:
        typed = _typed(resolved.table, where)
        entries = entries_where(resolved.table, typed)
    except ValueError as exc:
        raise TagSelectionInvalid(
            str(exc), revision=resolved.revision, tags=where
        ) from exc
    if groups is not None:
        entries = [entry for entry in entries if entry[0] in groups]
    if not entries:
        msg = (
            f"no entry holds {_described(where)} in revision {resolved.revision} "
            "of the tag table"
        )
        if groups is not None:
            msg += f", among the groups {list(groups)}"
        raise TagSelectionInvalid(msg, revision=resolved.revision, tags=where)
    return entries


def _typed(
    table: TagTable, where: Mapping[str, Sequence[TypedAttributeValue]]
) -> dict[str, list[TypedAttributeValue]]:
    """*where* with each text value read as its tag's type.

    A name that is not a sequence tag, a ``label`` and a value that is not text
    are left for :func:`entries_where` to check.

    Raises:
        ValueError: A text value is not a value of its tag's type, naming the tag
            and its type.
    """
    types: dict[str, TypedAttributeType] = {
        definition.name: definition.type for definition in table.definitions.sequence
    }
    typed: dict[str, list[TypedAttributeValue]] = {}
    for name, values in where.items():
        type_ = types.get(name)
        typed[name] = [
            _read_text(name, type_, value)
            if isinstance(value, str) and type_ is not None and type_ != "label"
            else value
            for value in values
        ]
    return typed


def _read_text(name: str, type_: TypedAttributeType, value: str) -> TypedAttributeValue:
    """*value* read as tag *name*'s type, refused naming the tag and its type."""
    try:
        return read_tag_text(type_, value)
    except ValueError as exc:
        msg = f"tag {name!r} of type {type_} refuses the value {value!r}: {exc}"
        raise ValueError(msg) from exc


def _described(where: Mapping[str, Sequence[TypedAttributeValue]]) -> str:
    """The selection as ``tag 'a' of 'x' or 'y' and tag 'b'`` for a message."""
    parts = [
        f"tag {name!r} of {' or '.join(repr(value) for value in values)}"
        if values
        else f"tag {name!r}"
        for name, values in where.items()
    ]
    return " and ".join(parts)


def _partial_message(
    key: RecordingKey,
    members: Sequence[Entry],
    selected: set[Entry],
    table: TagTable,
    where: Mapping[str, Sequence[TypedAttributeValue]],
) -> str:
    """Why the selection covers only part of the divided recording *key*."""
    held = {(item.group, item.sequence): item.tags for item in table.sequences}
    left_out = [
        sequence for group, sequence in members if (group, sequence) not in selected
    ]
    labels = {
        definition.name
        for definition in table.definitions.sequence
        if definition.type == "label"
    }
    per_tag: list[str] = []
    for name in where:
        values = ", ".join(
            f"{sequence}={_held_value(held.get((group, sequence), {}), name, name in labels)}"
            for group, sequence in members
        )
        per_tag.append(f"{name!r}: {values}")
    return (
        f"recording {key.group!r} is divided into sequences and the selection "
        f"covers only part of it; it leaves out {left_out}. Its sequences hold "
        f"{'; '.join(per_tag)}. Select every sequence of the recording"
    )


def _held_value(
    tags: Mapping[str, TypedAttributeValue], name: str, is_label: bool
) -> str:
    """What an entry holds of tag *name*, as a message names it."""
    if is_label:
        return "assigned" if name in tags else "not assigned"
    return repr(tags.get(name))
