"""Declares mosaic's strict pydantic base and renders the failures its models raise.

Imports pydantic and the standard library only. :func:`terse` lives here because
it renders a pydantic validation failure, and this module declares the pydantic
base mosaic's models share. Both mosaic's command line and mosaic-queue's submit
commands print one.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = ["StrictModel", "terse"]


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
    messages = [
        # Pydantic prefixes a raised ValueError with "Value error, ". The rest
        # is the message the model's own validator wrote.
        str(entry["msg"]).removeprefix("Value error, ")
        for entry in exc.errors()
    ]
    return "; ".join(m for m in messages if m) or str(exc)
