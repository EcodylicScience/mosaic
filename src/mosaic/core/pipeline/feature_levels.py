"""The entity levels a feature declares: what its output is keyed by and what it relates.

This leaf imports only the standard library. A reader of a feature's declared
levels, such as a definition's sidecar, loads neither pandas nor the feature library.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["RELATES_LEVELS", "EmitsLevel", "RelatesLevel"]

type EmitsLevel = Literal["individual", "pair", "unidentified", "as-input"]
"""At what entity level a feature's output is keyed. See ``Feature.emits``."""

type RelatesLevel = Literal["none", "arena-aware", "arena-blind"]
"""How a feature relates individuals to one another, read from its ``relates``.

- ``"none"``: it measures each individual alone, or reads a table without
  individuals.
- ``"arena-aware"``: it relates only individuals of one arena, through
  :func:`~mosaic.core.individuals.individual_pairs`, or reads pairs that its
  input formed so.
- ``"arena-blind"``: it relates individuals across arenas, such as a nearest
  neighbor or a group metric. ``run_feature`` refuses it on an entry whose input
  holds two arenas, or whose recording has a layout of several arenas while the
  input has no ``arena``.

``relates`` is a class attribute outside the protocol, as ``category`` is.
Every registered feature declares it
(``tests/behavior/feature_library/test_feature_conformance.py``), and an
undeclared one reads as ``"none"``.
"""

RELATES_LEVELS: tuple[RelatesLevel, ...] = ("none", "arena-aware", "arena-blind")
"""Every :data:`RelatesLevel`, in a stable order."""
