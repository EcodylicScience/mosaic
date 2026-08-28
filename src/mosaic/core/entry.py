"""What a run covers: the ``(group, sequence)`` pair, with no dependencies at all.

A leaf module on purpose, for the reason :mod:`mosaic.core.json_value` is one.
Both aliases began in :mod:`mosaic.core.pipeline.inventory.model`, which a
parameter model cannot import without pointing ``types`` at a package ``graph``
and ``tracking`` already import. Declaring the pair here lets a scope name the
entries a run covers while every edge keeps pointing one way.

Every consumer imports them from here, :mod:`mosaic.core.pipeline.inventory.model`
included. That module re-exports neither, so this is the one import path.
"""

from __future__ import annotations

__all__ = ["CameraEntry", "Entry"]


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
