"""Declares mosaic's strict pydantic base and renders the failures its models raise.

Imports pydantic and the standard library only. :func:`terse` and :func:`located` live
here because they render a pydantic validation failure, and this module declares
the pydantic base mosaic's models share. Both mosaic's command line and
mosaic-queue's submit commands print one.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = ["StrictModel", "located", "terse"]


class StrictModel(BaseModel):
    """BaseModel with extra="forbid" to reject unknown fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


def terse(exc: Exception) -> str:
    """The part of *exc* a person needs, without the validator's scaffolding.

    A pydantic ``ValidationError`` renders as several lines carrying the input
    that failed, its type, and a documentation URL. Those help someone debugging
    a model. Someone who mistyped a tag value wants the one sentence saying so.
    Non-pydantic exceptions are returned as they are.
    """
    if not isinstance(exc, ValidationError):
        return str(exc)
    messages = [_without_prefix(entry["msg"]) for entry in exc.errors()]
    return "; ".join(m for m in messages if m) or str(exc)


def located(exc: Exception) -> str:
    """Each failure of *exc* with the dotted location of the field it names.

    :func:`terse` drops where a failure happened, which suits a single value. A
    file with many keys needs the key: an unknown key or an out-of-range number
    reads as ``motion.mean_speed: Input should be greater than or equal to 0``.
    A failure of a whole model has no location and renders as its bare message.
    Failures are joined with ``"; "``. Non-pydantic exceptions are returned as
    they are.

    Args:
        exc: The exception to render.

    Returns:
        The rendered failures.
    """
    if not isinstance(exc, ValidationError):
        return str(exc)
    parts: list[str] = []
    for entry in exc.errors():
        message = _without_prefix(entry["msg"])
        location = ".".join(str(part) for part in entry["loc"])
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or str(exc)


def _without_prefix(message: object) -> str:
    """The message of one failure, without the prefix pydantic adds to a raised error."""
    # Pydantic prefixes a raised ValueError with "Value error, ". The rest is the
    # message the model's own validator wrote.
    return str(message).removeprefix("Value error, ")
