"""What a caller asks a run to cover, before anything resolves it.

A selector, never a resolution. It names entries, or it names groups and
sequences whose cross product an index enumerates, or it names nothing and
means every indexed entry. What a run resolved to, and what it read while
resolving, is :class:`~mosaic.core.pipeline._utils.ResolvedScope`. This is a
different thing from :class:`~mosaic.core.dataset.ResolvedScopeEntry`, which
is the per-camera result of resolving a media scope.

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

from collections.abc import Mapping, Sequence
from typing import ClassVar, Self

from pydantic import ConfigDict, field_validator, model_validator
from typing_extensions import TypeIs

from mosaic.core.entry import CameraEntry, Entry
from mosaic.core.strict_model import StrictModel

__all__ = [
    "SCOPE_PARAM_KEYS",
    "Scope",
    "camera_grain_refusal",
    "entries_exclude_pair_refusal",
    "scope_in_params_refusal",
]


def _is_camera_grain(
    entries: list[Entry] | list[CameraEntry],
) -> TypeIs[list[CameraEntry]]:
    """Whether *entries* holds ``(group, sequence, camera)`` triples.

    A pydantic union already refused a list mixing the two grains. The first
    element already states the grain of every element.
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


def entries_exclude_pair_refusal(
    entries: Sequence[object] | None,
    groups: Sequence[object] | None,
    sequences: Sequence[object] | None,
    *,
    prefix: str = "",
) -> str:
    """Why *entries* may not be given beside *groups* or *sequences*, or ``""``.

    The rule :meth:`Scope._entries_exclude_the_pair` enforces, written once so
    that a caller naming the same selector in another vocabulary refuses in the
    same words. ``prefix="--"`` spells the three names as the flags a person
    types at a command line.

    Presence decides it. This reads whether each selector was given and never
    what it names, so a caller holding unparsed command-line tokens asks without
    parsing them first.

    Args:
        entries: The entries selector, or ``None`` where it was not given.
        groups: The groups selector, or ``None`` where it was not given.
        sequences: The sequences selector, or ``None`` where it was not given.
        prefix: Prepended to each name.

    Returns:
        The refusal, or ``""`` where the three describe one selector.
    """
    if entries is None:
        return ""
    also = [
        f"{prefix}{name}"
        for name, value in (("groups", groups), ("sequences", sequences))
        if value is not None
    ]
    if not also:
        return ""
    joined = " and ".join(also)
    return (
        f"{prefix}entries names the exact entries to cover and cannot be "
        f"combined with {joined}. Give {prefix}entries alone, or give "
        f"{joined} alone and let the index enumerate them."
    )


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
        # Both arms call the same function on the same value, and still cannot
        # collapse into one call outside the branch. ``entries`` has the static
        # type ``list[Entry] | list[CameraEntry]``, a union of two list types.
        # ``_deduplicate`` is generic over one list type at a time. Each branch
        # binds its type parameter to ``Entry`` or to ``CameraEntry`` instead of
        # to their union. Widening ``_deduplicate``'s parameter to
        # ``Sequence[T]`` to admit the union in one call binds its type
        # parameter to ``Entry | CameraEntry`` instead. The resulting
        # ``list[Entry | CameraEntry]`` does not satisfy the declared
        # ``list[Entry] | list[CameraEntry]`` return type.
        if _is_camera_grain(entries):
            return _deduplicate(entries)
        return _deduplicate(entries)

    @model_validator(mode="after")
    def _entries_exclude_the_pair(self) -> Self:
        """Refuse ``entries`` given together with ``groups`` or ``sequences``.

        The three used to intersect, which narrowed an enumeration the caller
        had already written out. Two ways to say one thing, and the narrower
        one is what ``entries`` alone already says.

        The sentence is :func:`entries_exclude_pair_refusal`'s, so a caller
        naming this selector as command-line flags refuses in the same words.
        """
        refusal = entries_exclude_pair_refusal(
            self.entries, self.groups, self.sequences
        )
        if refusal:
            raise ValueError(refusal)
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

    @property
    def cameras(self) -> set[str]:
        """The cameras a camera-addressed selector narrows to.

        Empty where the selector names pairs or nothing, which both mean every
        camera of every entry named. The one op that reads a camera filters its
        rows by this set. Naming two cameras of one entry narrows to those
        two without changing the entry count an arity declaration is
        checked against.
        """
        if self.entries is None or not _is_camera_grain(self.entries):
            return set()
        return {entry[2] for entry in self.entries}

    @property
    def entry_pairs(self) -> set[Entry] | None:
        """The ``(group, sequence)`` pairs ``entries`` names, or ``None``.

        A camera-addressed selector reduces to its pairs. Every index whose row
        is keyed without a camera narrows on these, and there are enough of
        them -- the tracks index, each feature index, the labels index, a
        feature manifest -- that the reduction belongs here instead of at each
        one.

        ``None`` where ``entries`` is unset, which keeps the empty selection
        (a set naming no entry) distinct from the absent one.
        """
        if self.entries is None:
            return None
        return {(entry[0], entry[1]) for entry in self.entries}


SCOPE_PARAM_KEYS = frozenset(Scope.model_fields)
"""The selector field names, refused inside a params mapping.

No feature and no op declares a field under any of these names. A run accepting
one would take a narrowing its own model never validated, and a caller naming a
scope there has named it somewhere the model never reads.

Derived from the model so that a fourth selector is covered the day it is
declared. Both command lines that refuse it -- ``mosaic run`` and
``mosaic-queue submit`` -- read it from here.
"""


def scope_in_params_refusal(params: Mapping[str, object], *, prefix: str = "") -> str:
    """Why *params* names a scope it may not, or ``""`` when it names settings alone.

    A feature's or an op's model validates a params mapping, and none of them
    declares a field under a selector name. A scope belongs to the attempt and
    arrives through the selector the caller states beside it.

    Every offending key is named, and each is answered with the name that
    replaces it. Reporting the first alone sends a caller round the loop once per
    key, and each round costs a scheduled job.

    Args:
        params: The mapping as the caller wrote it.
        prefix: Prepended to each name. ``"--"`` for a command line.

    Returns:
        The refusal, or ``""`` where *params* names settings alone.
    """
    named = sorted(SCOPE_PARAM_KEYS & set(params))
    if not named:
        return ""
    listed = ", ".join(named)
    flags = ", ".join(f"{prefix}{key}" for key in named)
    return (
        f"{prefix}params names the scope key(s) {listed}, which no feature and "
        f"no op declares. Name the scope with {flags} instead, and leave "
        f"{prefix}params to the settings the model validates."
    )


def camera_grain_refusal(scope: Scope | None) -> str:
    """Why *scope* may not narrow to a camera, or ``""`` where it does not.

    The camera axis is modeled here. One op narrows on it (``export-store``,
    through :attr:`Scope.cameras`) and one produces per-camera output without
    narrowing (``extract-frames``). Every other op reduces a camera-addressed
    entry to its ``(group, sequence)`` pair, and a grain accepted on a wire then
    covers every camera of the entry under a selector that named one, reporting
    success.

    The sentence is audience-neutral. A caller reaching it is a command line, a
    queue building an argv, or a control plane constructing a spec in Python,
    and it names the axis instead of a flag.

    The Python API is unaffected. ``run_op`` with a camera-addressed scope still
    exports one camera through ``export-store``. This refuses the wire, not the
    capability.

    Args:
        scope: The selector a caller sent, or ``None`` where none was given.

    Returns:
        The refusal, or ``""`` where the selector names no camera.
    """
    if scope is None or not scope.addresses_cameras:
        return ""
    named = sorted(scope.cameras)
    noun = "camera" if len(named) == 1 else "cameras"
    listed = ", ".join(named)
    return (
        f"This scope narrows to the {noun} {listed}, which a submitted run "
        f"cannot honor. Name the entry without a camera. Only export-store "
        f"reads the camera axis, and every other op covers every camera of the "
        f"entry and reports success."
    )
