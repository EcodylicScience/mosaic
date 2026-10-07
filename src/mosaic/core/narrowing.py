"""Narrowing a value of unknown shape, for the type checker.

This leaf module imports only the standard library and ``typing_extensions``.
Parsed JSON and YAML documents, class attributes read with ``getattr`` and a
library's metadata mappings are typed ``object``. Their readers narrow them
through these predicates.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import TypeIs

__all__ = ["is_object_list", "is_object_tuple", "is_str_mapping"]


def is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    """Whether *value* is a mapping of values by name, as a JSON object decodes to.

    Only the mapping is checked, not its keys: a decoded JSON or YAML object's
    keys are strings, and a reader through ``.get`` with a string returns the
    default from a mapping keyed otherwise.
    """
    return isinstance(value, Mapping)


def is_object_list(value: object) -> TypeIs[list[object]]:
    """Whether *value* is a list, as a JSON array decodes to."""
    return isinstance(value, list)


def is_object_tuple(value: object) -> TypeIs[tuple[object, ...]]:
    """Whether *value* is a tuple, read as a tuple of anything."""
    return isinstance(value, tuple)
