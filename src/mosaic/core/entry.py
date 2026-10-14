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
    "from_safe_name",
    "make_entry_key",
    "parse_entry_key",
    "parse_entry_tokens",
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

    ``group`` is an optional coarse namespace, and an empty one yields a flat
    ``{safe_seq}`` key. It is part of the on-disk identity, kept for back-compat
    and to tell apart sequences that share a name. Sequences are categorized by
    tags (mosaic-api) and selected with ``run_feature(scope=Scope(entries=...))``.
    """
    safe_group = to_safe_name(group) if group else ""
    safe_seq = to_safe_name(sequence)
    return f"{safe_group + '__' if safe_group else ''}{safe_seq}"


def parse_entry_key(key: str) -> tuple[str, str]:
    """Return the ``(group, sequence)`` that :func:`make_entry_key` built *key* from.

    A key without ``__`` is a sequence in the empty group. Every dataset the
    control plane creates has ``group=""``.

    The key is split on the first ``__``, as it was built. The group is one level,
    and a sequence containing ``__`` (which ``parse_hierarchy`` reads as further
    levels) keeps it. Both halves are percent-decoded, and a name that needed
    encoding round-trips.
    """
    safe_group, separator, safe_sequence = key.partition("__")
    if not separator:
        return "", from_safe_name(key)
    return from_safe_name(safe_group), from_safe_name(safe_sequence)
