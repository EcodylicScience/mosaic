"""What a run covers: the ``(group, sequence)`` pair, importing the standard library only.

A leaf module on purpose, for the reason :mod:`mosaic.core.json_value` is one.
Both aliases began in :mod:`mosaic.core.pipeline.inventory.model`, which a
parameter model cannot import without pointing ``types`` at a package ``graph``
and ``tracking`` already import. Declaring the pair here lets a scope name the
entries a run covers while every edge keeps pointing one way.

Every consumer imports the aliases and the token parser from here,
:mod:`mosaic.core.pipeline.inventory.model` included. That module re-exports
neither alias, and :mod:`mosaic.core.helpers` does not re-export the parser.
One import path per name.

The entry key and the safe-name encoding it is built from are defined here too, so
a reader of run-logs and failure records can name an entry without loading pandas.
:mod:`mosaic.core.helpers` re-exports those four functions, because mosaic-api
imports them from there.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import quote, unquote

__all__ = [
    "CameraEntry",
    "Entry",
    "EntryKeyCollisionError",
    "entry_key_collision",
    "entry_token",
    "from_safe_name",
    "make_entry_key",
    "parse_entry_key",
    "parse_entry_tokens",
    "refuse_entry_key_collision",
    "to_safe_name",
]


type Entry = tuple[str, str]
"""``(group, sequence)`` -- what a feature run, a tracks table or a tracker covers.

A tuple rather than a model: it is a key, used in a ``frozenset`` and as a
dictionary key throughout ``inventory``, ``graph`` and ``tracking``. It renders
in a JSON Schema as a two-item ``prefixItems`` array.
"""

type CameraEntry = tuple[str, str, str]
"""``(group, sequence, camera)`` -- what a frame run covers.

The camera axis is part of the key rather than a detail: the cameras of one
recording share a ``(group, sequence)``, so without it a run that extracted one
camera would read as covering the entry and the other camera would never be seen
as missing.
"""


def parse_entry_tokens(tokens: Iterable[str] | None) -> list[Entry]:
    """``["group:sequence", ...]`` as ``[(group, sequence), ...]``.

    The grammar a user types when naming entries on a command line or in an op's
    ``entries`` parameter, as distinct from
    :func:`parse_entry_key`, which reads the ``__``-joined
    key those entries are *stored* under.

    Splits on the **first** ``:``, so a sequence name containing one keeps it.
    A token with no ``:`` is a bare sequence in the empty group, which is the
    common case rather than the edge: every dataset the control plane creates has
    ``group=""``, and ``make_entry_key("", seq)`` is just ``seq``. Rejecting it
    would mean a user has to type a colon to say nothing.
    """
    pairs: list[Entry] = []
    for token in tokens or []:
        group, separator, sequence = token.partition(":")
        pairs.append((group, sequence) if separator else ("", group))
    return pairs


def entry_token(group: str, sequence: str) -> str:
    """Return the token that names an entry to a person: ``group:sequence``.

    A sequence in the empty group is its bare name, as
    :func:`parse_entry_tokens` reads a token without a ``:``.
    """
    return f"{group}:{sequence}" if group else sequence


def to_safe_name(s: str) -> str:
    """Return *s* stripped and percent-encoded as one path component."""
    return quote(s.strip(), safe="")


def from_safe_name(safe: str) -> str:
    """Return the name that :func:`to_safe_name` encoded as *safe*."""
    return unquote(safe)


def make_entry_key(group: str, sequence: str) -> str:
    """Return the composite key of a (group, sequence) dataset entry.

    The key is the filename stem that the pipeline uses throughout:
    ``{safe_group}__{safe_seq}`` when group is non-empty, else ``{safe_seq}``.

    ``group`` is a namespace, or the recording that a continuous group's
    sequences divide. An empty one yields a flat ``{safe_seq}`` key. Sequences
    are categorized by tags (mosaic-api) and selected with
    ``run_feature(scope=Scope(entries=...))``.

    Two entries can share a key: ``("", "a__b")`` and ``("a", "b")`` both key
    as ``a__b``. A write refuses the second through
    :func:`refuse_entry_key_collision`.
    """
    safe_group = to_safe_name(group) if group else ""
    safe_seq = to_safe_name(sequence)
    return f"{safe_group + '__' if safe_group else ''}{safe_seq}"


def parse_entry_key(key: str) -> tuple[str, str]:
    """Return the ``(group, sequence)`` that :func:`make_entry_key` built *key* from.

    A key without ``__`` is a sequence in the empty group. Every dataset the
    control plane creates has ``group=""``. A key written before collisions were
    refused may belong to ``("", "a__b")`` and parses as ``("a", "b")``.

    The key is split on the first ``__``, as it was built. The group is one level,
    and a sequence containing ``__`` (which ``parse_hierarchy`` reads as further
    levels) keeps it. Both halves are percent-decoded, and a name that needed
    encoding round-trips.
    """
    safe_group, separator, safe_sequence = key.partition("__")
    if not separator:
        return "", from_safe_name(key)
    return from_safe_name(safe_group), from_safe_name(safe_sequence)


class EntryKeyCollisionError(ValueError):
    """Raised when a written entry would share its key with a different entry."""


def entry_key_collision(
    written: Iterable[Entry], existing: Iterable[Entry]
) -> str | None:
    """Return why a new entry's :func:`make_entry_key` names a different entry.

    An entry of *written* that *existing* already contains is not new. A
    collision recorded before this check therefore keeps resolving, and a
    rewrite of its rows is accepted. A new entry is checked against *existing*
    and against the rest of *written*.

    Returns:
        A message naming both entries and their shared key, or ``None`` when
        no new entry collides.
    """
    held = set(existing)
    batch = set(written)
    by_key: dict[str, set[Entry]] = {}
    for entry in held | batch:
        by_key.setdefault(make_entry_key(*entry), set()).add(entry)
    for entry in sorted(batch - held):
        key = make_entry_key(*entry)
        others = sorted(by_key[key] - {entry})
        if others:
            return (
                f"entry {entry!r} would be stored under the key {key!r}, which "
                f"entry {others[0]!r} already names. A key joins the group and "
                "the sequence with '__', and the two entries cannot be told "
                "apart on disk. Rename one of them."
            )
    return None


def refuse_entry_key_collision(
    written: Iterable[Entry], existing: Iterable[Entry]
) -> None:
    """Refuse the collision that :func:`entry_key_collision` reports.

    Raises:
        EntryKeyCollisionError: Naming both entries and their shared key.
    """
    reason = entry_key_collision(written, existing)
    if reason is not None:
        raise EntryKeyCollisionError(reason)
