"""Reading a loaded YAML or JSON document without losing what it holds.

``yaml.safe_load`` answers ``Any``, so a test that indexes into one is asserting
against values the checker knows nothing about. These narrow a loaded document
to a mapping and flatten it to the dotted keys a Hydra override names, which is
the form both training suites compare against: the assignments an op injects are
already dotted, so a flattened document can be compared to them directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard

__all__ = ["dotted_values", "is_section"]


def is_section(value: object) -> TypeGuard[Mapping[str, object]]:
    """Is *value* a nested section of a document rather than a leaf?"""
    return isinstance(value, dict)


def dotted_values(node: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    """Every dotted key in *node*, at every depth, with the value it carries."""
    flat: dict[str, object] = {}
    for name, value in node.items():
        dotted = f"{prefix}{name}"
        flat[dotted] = value
        if is_section(value):
            flat.update(dotted_values(value, f"{dotted}."))
    return flat
