"""What a caller asks a run to cover, before anything resolves it.

A selector, never a resolution. It names entries, or it names groups and
sequences whose cross product an index enumerates, or it names nothing and
means every indexed entry. What a run resolved to, and what it read while
resolving, is :class:`~mosaic.core.dataset.ResolvedScopeEntry`, returned by
:meth:`~mosaic.core.dataset.Dataset.resolve_media_scope`.

A leaf beside :mod:`mosaic.core.entry`, importing that module and
:mod:`mosaic.core.strict_model` and nothing else. The manifest, the command
line, the graph and an external consumer all need to name a scope, and none of
them should pay for the loader and artifact machinery to do it.

**The camera is part of the entry key, never a parallel narrowing.** One op
reads a camera and sixteen do not, and nine of those discard a camera narrowing
by design -- a working directory is keyed without one. The arity an op declares
counts entries. A ``cameras`` list beside ``entries`` therefore changes what a
single-entry op covers while the count it is checked against stays at one.
:data:`~mosaic.core.entry.CameraEntry` states the rule this follows: the cameras
of one recording share a ``(group, sequence)``, and the camera belongs in the
key.

``entries`` is a union of two lists rather than a list of a union. One scope
addresses entries or camera-entries. A mixed list matches neither member and is
refused, because a pair means every camera of an entry and a triple means one of
them, and a selection holding both says nothing about whether the pair includes
the camera the triple names.
"""

from __future__ import annotations

from typing import ClassVar, Self, TypeIs

from pydantic import ConfigDict, field_validator, model_validator

from mosaic.core.entry import CameraEntry, Entry
from mosaic.core.strict_model import StrictModel

__all__ = ["Scope"]


def _is_camera_grain(
    entries: list[Entry] | list[CameraEntry],
) -> TypeIs[list[CameraEntry]]:
    """Whether *entries* holds ``(group, sequence, camera)`` triples.

    A pydantic union already refused a list mixing the two grains, so the
    first element states the grain of every element.
    """
    return bool(entries) and len(entries[0]) == 3


def _deduplicate[T: tuple[str, ...]](items: list[T]) -> list[T]:
    """Collapse a repeated item, keeping the order it first appeared in."""
    seen: set[T] = set()
    unique: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


class Scope(StrictModel):
    """The entries a run covers, as the caller named them.

    Attributes:
        entries: The entries to cover, as ``(group, sequence)`` pairs or
            ``(group, sequence, camera)`` triples. Excludes *groups* and
            *sequences*. An empty list names no entry, which is not the same as
            naming none at all.
        groups: Cover every sequence in these groups. Combines with *sequences*
            as a cross product.
        sequences: Cover this sequence name in every group.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    entries: list[Entry] | list[CameraEntry] | None = None
    groups: list[str] | None = None
    sequences: list[str] | None = None

    @field_validator("entries")
    @classmethod
    def _deduplicate_entries(
        cls, entries: list[Entry] | list[CameraEntry] | None
    ) -> list[Entry] | list[CameraEntry] | None:
        """Collapse a repeated entry, keeping the order the caller gave.

        A duplicate names one entry twice rather than covering more. An
        identity payload sorts its members and does not collapse them. Collapsing
        here keeps one run's name independent of how many times a caller repeated
        an entry.
        """
        if entries is None:
            return None
        if _is_camera_grain(entries):
            return _deduplicate(entries)
        return _deduplicate(entries)

    @model_validator(mode="after")
    def _entries_exclude_the_pair(self) -> Self:
        """Refuse ``entries`` given together with ``groups`` or ``sequences``.

        The three used to intersect, which narrowed an enumeration the caller
        had already written out. Two ways to say one thing, and the narrower
        one is what ``entries`` alone already says.
        """
        if self.entries is None:
            return self
        also = [
            name
            for name, value in (("groups", self.groups), ("sequences", self.sequences))
            if value is not None
        ]
        if also:
            joined = " and ".join(also)
            message = (
                f"entries names the exact entries to cover and cannot be "
                f"combined with {joined}. Give entries alone, or give "
                f"{joined} alone and let the index enumerate them."
            )
            raise ValueError(message)
        return self

    @property
    def is_unset(self) -> bool:
        """Whether the caller named no selector at all.

        Distinct from a selector that names nothing. ``Scope()`` covers every
        indexed entry. ``Scope(entries=[])`` and ``Scope(groups=["absent"])``
        cover none. An op that refuses an unscoped run needs to tell them
        apart, and a resolved entry count cannot.
        """
        return self.entries is None and self.groups is None and self.sequences is None

    @property
    def addresses_cameras(self) -> bool:
        """Whether ``entries`` holds ``(group, sequence, camera)`` triples."""
        return self.entries is not None and _is_camera_grain(self.entries)
