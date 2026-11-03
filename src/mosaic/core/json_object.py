"""The adapter that validates a parsed JSON document as an object.

A leaf beside :mod:`mosaic.core.json_value`. An importer of ``JsonValue`` alone
loads the standard library and builds this adapter only when it imports this module.
"""

from __future__ import annotations

from typing import Final

from pydantic import TypeAdapter

from mosaic.core.json_value import JsonValue

__all__ = ["JSON_OBJECT"]

JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(
    dict[str, JsonValue]
)
"""Validates a parsed JSON document as an object, turning tuples into lists."""
