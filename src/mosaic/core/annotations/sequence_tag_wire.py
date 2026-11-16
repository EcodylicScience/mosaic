"""The shapes a sequence-tag import and listing report.

The command line prints them and the agent tool contract declares them, so this
module imports pydantic, :mod:`mosaic.core.strict_model`,
:mod:`mosaic.core.scan_findings` and :mod:`mosaic.core.typed_attribute` only. The
contract then loads no pandas.
"""

from __future__ import annotations

from typing import Literal

from mosaic.core.scan_findings import ScanFinding
from mosaic.core.strict_model import StrictModel
from mosaic.core.typed_attribute import TypedAttributeType

__all__ = [
    "SequenceTagChange",
    "SequenceTagImport",
    "SequenceTagListed",
    "SequenceTagListing",
    "SequenceTagValueCount",
]


class SequenceTagValueCount(StrictModel):
    """One value of a tag and the entries that hold it.

    Attributes:
        value: The value. A ``label`` tag has one value, ``true``.
        entries: The number of entries that hold the value.
    """

    value: str | int | float | bool
    entries: int


class SequenceTagListed(StrictModel):
    """One sequence tag of the latest revision.

    Attributes:
        name: The tag's name.
        type: The tag's type.
        entries: The number of entries that hold a value of the tag.
        values: Each value with its entry count, sorted by value.
    """

    name: str
    type: TypedAttributeType
    entries: int
    values: tuple[SequenceTagValueCount, ...]


class SequenceTagListing(StrictModel):
    """The sequence tags of the latest revision of a dataset's tag table.

    Attributes:
        revision: The latest revision, or ``None`` for a dataset without one.
        digest: The digest of that revision's payload, or ``None``.
        entries: The number of entries the revision lists.
        tags: The sequence tags, sorted by name.
    """

    revision: int | None
    digest: str | None
    entries: int
    tags: tuple[SequenceTagListed, ...]


class SequenceTagChange(StrictModel):
    """What an import does to one sequence tag.

    Attributes:
        name: The tag's name.
        type: The tag's type after the import, or before it for a removed tag.
        change: ``added`` for a tag the table lacked, ``removed`` for a tag the
            import drops, ``changed`` for a tag whose type or any entry's value
            differs, else ``unchanged``.
        entries_set: The entries that hold a value after the import that they
            did not hold before, or held with another value or type.
        entries_cleared: The entries that held a value before the import and hold
            none after it.
    """

    name: str
    type: TypedAttributeType
    change: Literal["added", "changed", "unchanged", "removed"]
    entries_set: int
    entries_cleared: int


class SequenceTagImport(StrictModel):
    """The outcome of importing a CSV of sequence tags.

    Attributes:
        revision: The latest revision of the tag table after the call: the one
            the import wrote, or the one already there when it wrote nothing.
            ``None`` when the dataset has none.
        written: Whether the call wrote a revision.
        dry_run: Whether the call only reported the changes.
        rows_matched: The rows that named at least one entry.
        changes: Each sequence tag the import touches or keeps, sorted by name.
        findings: The rows that named no entry.
    """

    revision: int | None
    written: bool
    dry_run: bool
    rows_matched: int
    changes: tuple[SequenceTagChange, ...]
    findings: tuple[ScanFinding, ...]
