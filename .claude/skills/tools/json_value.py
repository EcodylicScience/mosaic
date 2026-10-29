"""Type JSON values as the type checker sees them, and write them to
files."""

import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
# A value to write as JSON. Sequence and Mapping accept list[str] and
# dict[str, str], which the invariant list and dict of Json reject.
JsonOutput = str | int | float | bool | None | Sequence["JsonOutput"] | Mapping[str, "JsonOutput"]


def parse_json(text: str) -> Json:
    """Return the value parsed from `text`. Raise ValueError when it is not
    JSON, including when it nests deeper than the parser can follow.

    This is the one place that touches untyped data. Every caller gets a value
    the checker understands.
    """
    try:
        return json.loads(text)
    except RecursionError as error:
        raise ValueError(str(error)) from error


def json_integer(value: Json) -> int | None:
    """Return `value` when it is a JSON integer, None for a boolean or any
    other value."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def write_json(path: Path, value: JsonOutput) -> None:
    """Replace `path` with `value` as JSON in one rename. A reader sees the
    previous file or the complete new one. Create the parent directory when it
    is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    _ = temporary.write_text(json.dumps(value), encoding="ascii")
    _ = temporary.replace(path)
