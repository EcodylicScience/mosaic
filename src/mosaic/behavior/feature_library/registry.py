from __future__ import annotations

from typing_extensions import TypeVar

from mosaic.core.pipeline.types import Feature

__all__ = [
    "FEATURES",
    "register_feature",
]

FEATURES: dict[str, type[Feature]] = {}

_F = TypeVar("_F", bound=Feature)


def register_feature(cls: type[_F]) -> type[_F]:
    FEATURES[cls.__name__] = cls
    return cls
