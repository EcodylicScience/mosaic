"""Declares the pydantic base every strict model in mosaic shares.

Imports pydantic and the standard library only.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """BaseModel with extra="forbid" to reject unknown fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
