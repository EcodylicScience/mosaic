"""Resolving the tag table that a feature reads, and what its identity covers of it.

A feature that reads the ``tags`` series declares what it reads (``tag_read``).
This module resolves the one revision a run reads, checks the declaration against
that revision, and builds the identity term over the part of the table that the
feature names. The planner, ``run_feature`` and ``pipeline verify`` all resolve
through it, so a predicted identifier and an executed one are built alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.annotations.tag_table import (
    TagRead,
    TagTable,
    parse_tag_table,
    read_tag_table,
    reserved_name_message,
    sequence_tags_projection,
)
from mosaic.core.pipeline._utils import LabelRevisionTerm
from mosaic.core.pipeline.label_series import (
    TAG_TABLE_KEY,
    TAGS_SERIES,
    LabelRevisionRef,
    refuse_other_tag_key,
)
from mosaic.core.pipeline.label_series_index import (
    payload_digest,
    resolve_series_refs,
)
from mosaic.core.pipeline.types.feature import Feature, ReadsTags

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "ResolvedTagTable",
    "feature_tag_read",
    "refuse_tag_read",
    "resolve_tag_ref",
    "resolve_tag_table",
    "tag_read_problems",
    "tag_term",
]


@dataclass(frozen=True, slots=True)
class ResolvedTagTable:
    """The tag table one run reads, with the revision it came from.

    Attributes:
        origin_uuid: The manifest uuid of the dataset that saved the revision, or
            of the dataset a passed payload is for.
        revision: The revision's number, or ``None`` for a payload that has not
            been written.
        payload_digest: The digest of the table's whole payload.
        path: The revision's payload file, or ``None`` for a payload that has not
            been written.
        table: The parsed table.
    """

    origin_uuid: str
    revision: int | None
    payload_digest: str
    path: Path | None
    table: TagTable

    @property
    def ref(self) -> LabelRevisionRef | None:
        """The reference that names this revision, or ``None`` if it is unwritten."""
        if self.revision is None:
            return None
        return LabelRevisionRef(
            key=TAG_TABLE_KEY, origin_uuid=self.origin_uuid, revision=self.revision
        )


def resolve_tag_ref(
    ds: Dataset, ref: LabelRevisionRef, *, kind: str
) -> LabelRevisionRef:
    """The reference that names the revision *ref* resolves to.

    Args:
        ds: The dataset whose ``tags`` series is read.
        ref: The reference to resolve. Without a revision it means the latest
            indexed one, and without an origin the one dataset that has it.
        kind: What is resolving, named in a deferral.

    Raises:
        ValueError: *ref* names a key other than ``dataset``.
        IdentityDeferred: No revision is indexed, or the one named is not.
        KeyError: Several datasets' revisions match and *ref* names no origin.
        LabelSeriesTamperedError: The revision's bytes changed after it was
            indexed.
    """
    refuse_other_tag_key(ref)
    (revision,) = resolve_series_refs(ds, TAGS_SERIES, (ref,), kind=kind)
    return LabelRevisionRef(
        key=TAG_TABLE_KEY,
        origin_uuid=revision.origin_uuid,
        revision=revision.revision,
    )


def resolve_tag_table(
    ds: Dataset,
    ref: LabelRevisionRef | None,
    *,
    payload: bytes | None = None,
    kind: str,
) -> ResolvedTagTable:
    """The revision of the tag table that *ref* names, read once.

    Args:
        ds: The dataset whose ``tags`` series is read.
        ref: The revision to read, naming the key ``dataset``. ``None``, and a
            reference without a revision, mean the latest indexed revision.
        payload: A table that its caller has built and not written. It stands in
            for the latest revision when *ref* names no revision, and the
            series is not read.
        kind: What is resolving, named in a deferral.

    Raises:
        IdentityDeferred: No revision is indexed, or the one named is not.
        KeyError: Several datasets' revisions match and *ref* names no origin.
        LabelSeriesTamperedError: The revision's bytes changed after it was
            indexed.
        ValueError: *payload*, or the revision file, is not a tag table.
    """
    ref = ref or LabelRevisionRef(key=TAG_TABLE_KEY)
    if payload is not None and ref.revision is None:
        return ResolvedTagTable(
            origin_uuid=ds.uuid or "",
            revision=None,
            payload_digest=payload_digest(payload),
            path=None,
            table=parse_tag_table(payload),
        )
    (revision,) = resolve_series_refs(ds, TAGS_SERIES, (ref,), kind=kind)
    return ResolvedTagTable(
        origin_uuid=revision.origin_uuid,
        revision=revision.revision,
        payload_digest=revision.digest,
        path=revision.payload,
        table=read_tag_table(revision.payload),
    )


def tag_read_problems(table: TagTable, read: TagRead) -> list[tuple[str, str]]:
    """What stops a step from reading *table* as *read* declares, as ``(where, message)``.

    The ``where`` is the params path that the problem is at, so a planner reports
    it as the step's problem and a run raises it.

    Args:
        table: The table the step reads.
        read: What the step declares that it reads, with its renames.
    """
    problems: list[tuple[str, str]] = []
    if not read.sequence_tags:
        problems.append(
            ("params.fields", "fields names no tag; name at least one sequence tag")
        )
    defined = {definition.name for definition in table.definitions.sequence}
    problems.extend(
        (
            "params.fields",
            reserved_name_message(name),
        )
        for name in sorted(defined & read.writes)
    )
    problems.extend(
        ("params.fields", f"'{name}' is not a sequence tag of the dataset")
        for name in read.sequence_tags
        if name not in defined
    )
    problems.extend(
        (
            "params.field_renames",
            f"rename '{name}' to '{target}', the name of a column that this step "
            "writes",
        )
        for name, target in sorted(read.renames)
        if target in read.writes
    )
    problems.extend(
        ("params.field_renames", message) for message in _colliding_columns(read)
    )
    return problems


def _colliding_columns(read: TagRead) -> list[str]:
    """One message for each output column that two or more of the fields write."""
    renamed = dict(read.renames)
    writers: dict[str, list[str]] = {}
    for name in dict.fromkeys(read.sequence_tags):
        writers.setdefault(renamed.get(name, name), []).append(name)
    return [
        f"fields {_joined(names)} both write the column '{column}'"
        for column, names in sorted(writers.items())
        if len(names) > 1
    ]


def _joined(names: list[str]) -> str:
    quoted = [f"'{name}'" for name in names]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def refuse_tag_read(feature_name: str, table: TagTable, read: TagRead) -> None:
    """Raise the first of :func:`tag_read_problems` of *table* and *read*.

    Raises:
        ValueError: The table does not suit the declaration. The message names
            the feature, the params path and the problem.
    """
    problems = tag_read_problems(table, read)
    if problems:
        where, message = problems[0]
        msg = f"{feature_name}: {where}: {message}"
        raise ValueError(msg)


def tag_term(resolved: ResolvedTagTable, read: TagRead) -> LabelRevisionTerm:
    """The identity term of *read* over *resolved*.

    Its digest covers the named sequence tags over every sequence of the table, so
    an edit of another tag leaves it unchanged. A table that has not been written
    has revision ``0``, which is recorded and never hashed.
    """
    return LabelRevisionTerm(
        series=TAGS_SERIES,
        origin_uuid=resolved.origin_uuid,
        key=TAG_TABLE_KEY,
        digest=payload_digest(
            sequence_tags_projection(resolved.table, read.sequence_tags)
        ),
        revision=resolved.revision or 0,
        payload_digest=resolved.payload_digest,
    )


def feature_tag_read(
    ds: Dataset,
    feature: Feature,
    ref: LabelRevisionRef | None,
    payload: bytes | None = None,
) -> tuple[ResolvedTagTable, LabelRevisionTerm] | None:
    """The table *feature* reads and its identity term, or ``None`` if it reads none.

    The one place a feature's declaration is checked against the table. Run and
    prediction both call it, so the two refuse the same steps.

    Raises:
        ValueError: The table does not suit the declaration; the message is the
            first of :func:`tag_read_problems`.
        IdentityDeferred, KeyError, LabelSeriesTamperedError: As
            :func:`resolve_tag_table`.
    """
    if not isinstance(feature, ReadsTags):
        return None
    read = feature.tag_read()
    resolved = resolve_tag_table(ds, ref, payload=payload, kind=feature.name)
    refuse_tag_read(feature.name, resolved.table, read)
    return resolved, tag_term(resolved, read)
