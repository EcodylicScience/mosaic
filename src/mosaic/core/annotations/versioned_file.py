"""Reads a JSON file of one of mosaic's own versioned formats.

Each such file declares its ``format`` and ``version``, and its model refuses a key
it does not know. A newer version may add keys, so the version is read first,
leniently, and a newer file is refused as newer before the strict model sees it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ValidationError

from mosaic.core.strict_model import terse

__all__ = ["read_versioned_file"]


class _Header(BaseModel):
    """The two fields every version of every format declares. Other keys pass."""

    format: str
    version: int


def read_versioned_file[ModelT: BaseModel](
    path: Path, model: type[ModelT], *, format_name: str, newest_version: int
) -> ModelT:
    """Return *path* validated as *model*, a file of *format_name*.

    Args:
        path: The file.
        model: The strict model of the newest version this mosaic reads.
        format_name: The ``format`` the file must declare.
        newest_version: The newest ``version`` this mosaic reads.

    Raises:
        ValueError: The file declares *format_name* at a version newer than
            *newest_version*, or *model* refuses it. The message names *path*.
    """
    raw = path.read_bytes()
    try:
        header = _Header.model_validate_json(raw)
    except ValidationError:
        header = None
    if (
        header is not None
        and header.format == format_name
        and header.version > newest_version
    ):
        msg = (
            f"{path} is {format_name} version {header.version}, and this mosaic "
            f"reads up to version {newest_version}. Upgrade mosaic to read it."
        )
        raise ValueError(msg)
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        msg = f"{path} is not a {format_name} file: {terse(exc)}"
        raise ValueError(msg) from exc
