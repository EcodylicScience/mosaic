"""The JSON-representable value type, with no dependencies at all.

A leaf module on purpose. ``JsonValue`` began in
:mod:`mosaic.core.pipeline.types.params`, which cannot be imported without
pulling the loader and artifact machinery -- and through them pandas and pyarrow.
The dataset manifest needs the same type, and a manifest read must stay cheap
enough that ``mosaic init`` and ``--help`` do not pay for a dataframe library.

:mod:`mosaic.core.pipeline.types.params` imports the alias from here and
re-exports it, so ``from mosaic.core.pipeline.types import JsonValue`` keeps
working and there is exactly one definition.
"""

from __future__ import annotations

__all__ = ["JsonValue"]


type JsonValue = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
"""Anything JSON can represent exactly.

The annotation for a field that is deliberately open-ended -- a pass-through
settings dictionary handed to an external tool, a list of user-supplied specs,
the manifest's ``meta`` mapping. Prefer a concrete model wherever the shape is
actually known; this is for the cases where it genuinely is not.

It accepts every value that survives ``identity_ready`` intact and rejects
exactly those that would not, so an unrepresentable value fails at construction
with pydantic naming the field, rather than deep inside ``hash_params`` with only
a type name to go on. Recursive, so nested lists and dictionaries validate.
"""
