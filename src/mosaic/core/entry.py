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
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["CameraEntry", "Entry", "parse_entry_tokens"]


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
    :func:`mosaic.core.helpers.parse_entry_key`, which reads the ``__``-joined
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
