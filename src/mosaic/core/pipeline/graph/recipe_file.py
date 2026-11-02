"""Reading the JSON document in a file that a caller names.

One reader for the CLI's ``@file.json`` arguments, its ``--recipe <path>``, and
the MCP server's ``{"path": ...}`` recipe argument, so a file that does not read
is reported the same way by each. The module imports the error base only, so a
CLI argument read loads neither the validator nor pydantic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from ..error_kind import ClassifiedError, ErrorKind

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.json_value import JsonValue

__all__ = ["JsonFileUnreadable", "read_json_file"]


class JsonFileUnreadable(ClassifiedError, ValueError):
    """A named JSON file that is missing, cannot be read, or is not valid JSON.

    Attributes:
        path: The file named.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path: Path = path

    def error_reason(self) -> str:
        """Return ``json_file_unreadable``."""
        return "json_file_unreadable"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the file named."""
        return {"path": str(self.path)}


def read_json_file(path: Path) -> object:
    """Return the JSON document that the file at *path* holds.

    Raises:
        JsonFileUnreadable: There is no file at *path*, it cannot be read, or it
            is not valid JSON.
    """
    if not path.exists():
        raise JsonFileUnreadable(path, f"JSON file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise JsonFileUnreadable(path, f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonFileUnreadable(path, f"Invalid JSON ({path}): {exc}") from exc
