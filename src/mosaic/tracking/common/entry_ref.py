"""Declares the ``(group, sequence)`` pair that a parameter model names an entry by."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mosaic.core.entry import Entry
from mosaic.core.params import Declared, DeclaredModel

__all__ = ["EntryRef"]

_GROUP_DESCRIPTION = "The entry's group; empty for a dataset with none."

_SEQUENCE_DESCRIPTION = "The entry's sequence."


class EntryRef(DeclaredModel):
    """Names one entry of a dataset."""

    group: Annotated[str, Declared(_GROUP_DESCRIPTION)] = ""
    sequence: Annotated[str, Field(min_length=1), Declared(_SEQUENCE_DESCRIPTION)]

    @property
    def entry(self) -> Entry:
        """The ``(group, sequence)`` pair this item names."""
        return (self.group, self.sequence)
