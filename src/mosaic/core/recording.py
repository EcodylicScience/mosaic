"""Which recording an entry belongs to, importing :mod:`mosaic.core.entry` only.

A recording is divided when its group is declared in ``continuous_groups``. Its
sequences are then time divisions of one recording, named by the group. Every
other entry is an undivided recording of its own, whatever its group: an empty
group, or a namespace such as a CalMS21 split or a treatment. The kind comes
from the declaration. An empty group does not decide it.

Every function takes plain inputs. A dataset, a test double and a division
writer answer the question by one rule.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal, TypedDict

from .entry import Entry, entry_token, make_entry_key, to_safe_name

__all__ = [
    "DivisionPlacement",
    "FrameOutsideSequenceError",
    "OrderFacts",
    "Recording",
    "RecordingKey",
    "RecordingKind",
    "adjacent",
    "count_breaks",
    "division_placement",
    "facts_from_frame_extents",
    "order_members",
    "recording_kind",
    "recording_name",
    "recording_name_collision",
    "recording_of",
    "recording_storage_name",
    "recordings",
    "sequence_frame",
    "sequence_window",
    "utc_text",
]

RecordingKind = Literal["divided", "undivided"]
"""A divided recording is a declared continuous group. An undivided one is one entry."""


@dataclass(frozen=True, slots=True, order=True)
class RecordingKey:
    """The ``(group, sequence)`` pair that names one recording.

    A divided recording is ``(group, "")``. An undivided recording is its
    entry's own pair.

    Attributes:
        group: The entry's group, or the continuous group.
        sequence: The entry's sequence, or ``""`` for a divided recording.
    """

    group: str
    sequence: str


@dataclass(frozen=True, slots=True)
class OrderFacts:
    """What places one member in its recording, each ``None`` when unknown.

    A division record gives every field but ``frame_min``.

    Attributes:
        first_frame: The member's first frame on the recording's frame axis,
            from a division record.
        start_utc: When the member starts, timezone-aware.
        frame_min: The first frame of the member's tracks table, read under a
            pinned tracks run id.
        frame_count: The member's frames, from a division record.
        follows_gap: Whether a gap or a store boundary precedes the member, so
            that the member before it is not its neighbor.
        time_offset: The member's first frame on the recording's time axis, in
            seconds, ``None`` to take ``first_frame`` over the frame rate.
    """

    first_frame: int | None = None
    start_utc: datetime | None = None
    frame_min: int | None = None
    frame_count: int | None = None
    follows_gap: bool = False
    time_offset: float | None = None


def adjacent(before: OrderFacts | None, after: OrderFacts | None) -> bool:
    """Return whether *after* continues *before* on the recording's frame axis.

    Two divisions are adjacent when the second starts at the frame after the
    first ends and no gap or store boundary precedes it. A member without a
    division record is adjacent to the members beside it in recording order.
    """
    if before is None or after is None:
        return True
    if before.first_frame is None or after.first_frame is None:
        return True
    if after.follows_gap or before.frame_count is None:
        return not after.follows_gap
    return before.first_frame + before.frame_count == after.first_frame


def count_breaks(
    members: Iterable[Entry], order_facts: Mapping[Entry, OrderFacts]
) -> int:
    """Return how many neighboring *members* are not :func:`adjacent`.

    *members* are in recording order. A gap, a store boundary or a division
    missing between two members is one break.
    """
    ordered = [order_facts.get(entry) for entry in members]
    return sum(not adjacent(before, after) for before, after in pairwise(ordered))


def utc_text(moment: datetime) -> str:
    """Return *moment* in UTC as ISO 8601 with microseconds.

    The media index records a division's ``start_utc`` in this form, and the
    summary and the listings print it.
    """
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


class DivisionPlacement(TypedDict):
    """Where a sequence lies in its recording, each ``None`` unless it is a division.

    Attributes:
        first_frame: The sequence's first frame on the recording frame axis.
        frame_count: The sequence's frames.
        start_utc: When its first frame was recorded (:func:`utc_text`).
        follows_gap: Whether a gap or a store boundary precedes it.
    """

    first_frame: int | None
    frame_count: int | None
    start_utc: str | None
    follows_gap: bool | None


def division_placement(facts: OrderFacts | None) -> DivisionPlacement:
    """Return the placement that a division's *facts* record, all ``None`` without."""
    if facts is None:
        return DivisionPlacement(
            first_frame=None, frame_count=None, start_utc=None, follows_gap=None
        )
    return DivisionPlacement(
        first_frame=facts.first_frame,
        frame_count=facts.frame_count,
        start_utc=None if facts.start_utc is None else utc_text(facts.start_utc),
        follows_gap=facts.follows_gap,
    )


@dataclass(frozen=True, slots=True)
class Recording:
    """One recording and its member sequences, in recording order.

    Attributes:
        kind: Whether the recording is divided into sequences.
        key: The recording's key.
        name: The continuous group for a divided recording. For an undivided
            one, its entry token: the sequence, or ``group:sequence`` when the
            group is non-empty.
        members: The member entries in recording order. An undivided recording
            has its one entry.
    """

    kind: RecordingKind
    key: RecordingKey
    name: str
    members: tuple[Entry, ...]


def recording_kind(continuous_groups: Collection[str], group: str) -> RecordingKind:
    """Return whether an entry of *group* belongs to a divided recording."""
    return "divided" if group in continuous_groups else "undivided"


def recording_of(
    continuous_groups: Collection[str], group: str, sequence: str
) -> RecordingKey:
    """Return the key of the recording that ``(group, sequence)`` belongs to."""
    if recording_kind(continuous_groups, group) == "divided":
        return RecordingKey(group, "")
    return RecordingKey(group, sequence)


def recording_name(key: RecordingKey) -> str:
    """Return the name of the recording that *key* names.

    A divided recording is named by its continuous group, and an undivided
    one by its entry token, ``group:sequence`` or the bare sequence.
    """
    return entry_token(key.group, key.sequence) if key.sequence else key.group


def recording_storage_name(key: RecordingKey) -> str:
    """Return the one path component that per-recording storage is named by.

    A divided recording is named by its safe group name, and an undivided one
    by its entry key, so ``("", "a")`` and ``("ns", "a")`` stay apart. A
    continuous group named like an empty-group sequence would share the name.
    ``Dataset.set_continuous_groups`` refuses such a group when it is declared,
    and an index write refuses such a sequence written after it
    (:func:`recording_name_collision`).
    """
    if not key.sequence:
        return to_safe_name(key.group)
    return make_entry_key(key.group, key.sequence)


def recording_name_collision(
    continuous_groups: Collection[str],
    written: Iterable[Entry],
    existing: Iterable[Entry],
) -> str | None:
    """Return why a new entry of *written* would take a divided recording's name.

    An undivided recording is stored under its entry key, and a divided one under
    its safe group name (:func:`recording_storage_name`). The empty-group
    sequence ``R`` therefore shares the storage name of the continuous group
    ``R``. ``Dataset.set_continuous_groups`` refuses the group when the sequence
    is indexed first. This refuses the sequence when the group is declared
    first. An entry of *existing* is not new and keeps resolving.

    Returns:
        A message naming the entry and the group, or ``None`` when no new entry
        takes a divided recording's name.
    """
    divided = {
        recording_storage_name(RecordingKey(group, "")): group
        for group in continuous_groups
        if group
    }
    for group, sequence in sorted(set(written) - set(existing)):
        key = recording_of(continuous_groups, group, sequence)
        if not key.sequence:
            continue
        taken = divided.get(recording_storage_name(key))
        if taken is not None:
            return (
                f"entry {(group, sequence)!r} would be the recording "
                f"{recording_name(key)!r}. The continuous group {taken!r} has "
                "that name, and the two recordings would share it. "
                "Rename the sequence, or declare the group under another name."
            )
    return None


_DIGITS = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple[tuple[tuple[int, int, str], ...], str]:
    """Sort key that orders digit runs by value, so ``s2`` sorts before ``s10``.

    The odd parts of the split are the captured decimal runs. A character such
    as ``\u00b2`` that ``str.isdigit`` accepts and ``int`` refuses stays text.
    """
    parts = tuple(
        (0, int(part), "") if index % 2 else (1, 0, part)
        for index, part in enumerate(_DIGITS.split(name))
        if part
    )
    return parts, name


def _member_key(
    entry: Entry, facts: OrderFacts | None
) -> tuple[int, int, float, tuple[tuple[tuple[int, int, str], ...], str]]:
    """Sort key placing *entry* by the first fact it has, then by name.

    The leading rank keeps a partly recorded group total. A member placed by a
    stronger fact sorts before one placed by a weaker fact.
    """
    name = _natural_key(entry[1])
    if facts is not None:
        if facts.first_frame is not None:
            return (0, facts.first_frame, 0.0, name)
        if facts.start_utc is not None:
            return (1, 0, facts.start_utc.timestamp(), name)
        if facts.frame_min is not None:
            return (2, facts.frame_min, 0.0, name)
    return (3, 0, 0.0, name)


def order_members(
    members: Iterable[Entry], order_facts: Mapping[Entry, OrderFacts]
) -> tuple[Entry, ...]:
    """Return *members* in recording order, each once.

    A member sorts by its division's ``first_frame``, then by ``start_utc``,
    then by its tracks ``frame_min``, and last by a natural sort on its
    sequence name. A member with a fact sorts before one without it.
    """
    unique = set(members)
    return tuple(
        sorted(unique, key=lambda entry: _member_key(entry, order_facts.get(entry)))
    )


def facts_from_frame_extents(
    extents: Mapping[Entry, tuple[int, int]],
) -> dict[Entry, OrderFacts]:
    """Return the order facts that recorded tracks frame extents give."""
    return {entry: OrderFacts(frame_min=extent[0]) for entry, extent in extents.items()}


def recordings(
    continuous_groups: Collection[str],
    entries: Iterable[Entry],
    order_facts: Mapping[Entry, OrderFacts] | None = None,
) -> tuple[Recording, ...]:
    """Return every recording that *entries* make up, sorted by key.

    Each entry of a continuous group is a member of that group's divided
    recording. Every other entry is an undivided recording. A continuous group
    without entries in *entries* is left out.
    """
    facts = order_facts or {}
    divided: dict[str, list[Entry]] = {}
    found: list[Recording] = []
    for group, sequence in sorted(set(entries)):
        if recording_kind(continuous_groups, group) == "divided":
            divided.setdefault(group, []).append((group, sequence))
            continue
        key = RecordingKey(group, sequence)
        found.append(
            Recording(
                kind="undivided",
                key=key,
                name=recording_name(key),
                members=((group, sequence),),
            )
        )
    found.extend(
        Recording(
            kind="divided",
            key=RecordingKey(group, ""),
            name=group,
            members=order_members(members, facts),
        )
        for group, members in divided.items()
    )
    return tuple(sorted(found, key=lambda recording: recording.key))


class FrameOutsideSequenceError(ValueError):
    """A frame on the recording's axis lies outside the sequence whose media is read."""


def sequence_frame(frame: int, *, first_frame: int, frame_count: int | None) -> int:
    """Return the frame of a sequence's media that recording frame *frame* is.

    The tracks of a divided recording count ``frame`` across the recording, and a
    sequence's media counts from the sequence's first frame, *first_frame* on the
    recording's axis. An undivided recording starts at 0, and its frames are
    the media's.

    Args:
        frame: The frame on the recording's axis.
        first_frame: The sequence's first frame on that axis.
        frame_count: The frames of the sequence's media, ``None`` when unknown,
            which bounds the sequence below only.

    Raises:
        FrameOutsideSequenceError: If *frame* is before the sequence, or at or
            past its end.
    """
    local = frame - first_frame
    if local < 0 or (frame_count is not None and local >= frame_count):
        message = f"frame {frame} is {_outside(first_frame, frame_count)}"
        raise FrameOutsideSequenceError(message)
    return local


def sequence_window(
    start: int, end: int | None, *, first_frame: int, frame_count: int | None
) -> tuple[int, int | None]:
    """Return the media frames of a sequence that recording frames *start* to *end* are.

    The window is the part of the requested frames that the sequence contains, so
    a *start* before the sequence reads from its first frame. *end* is inclusive,
    and ``None`` reads to the end of the media, as does an *end* past it.

    Args:
        start: The first recording frame to read.
        end: The last recording frame to read, or ``None``.
        first_frame: The sequence's first frame on the recording's axis.
        frame_count: The frames of the sequence's media, ``None`` when unknown.

    Returns:
        The first and last media frames, the last ``None`` to read to the end.

    Raises:
        FrameOutsideSequenceError: If the sequence contains none of the frames.
    """
    first = max(start, first_frame) - first_frame
    past_end = frame_count is not None and first >= frame_count
    if past_end or (end is not None and end < max(start, first_frame)):
        requested = f"frames {start} to {'the end' if end is None else end}"
        message = f"{requested} are {_outside(first_frame, frame_count)}"
        raise FrameOutsideSequenceError(message)
    return first, None if end is None else end - first_frame


def _outside(first_frame: int, frame_count: int | None) -> str:
    """Say which recording frames a sequence's media contains, for a refusal."""
    contained = (
        f"from frame {first_frame}"
        if frame_count is None
        else f"frames {first_frame} to {first_frame + frame_count - 1}"
    )
    return f"outside the sequence, whose media contains {contained} of its recording."
