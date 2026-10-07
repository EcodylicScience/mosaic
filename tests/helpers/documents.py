"""Reading a loaded YAML or JSON document without losing what it holds.

``yaml.safe_load`` answers ``Any``, so a test that indexes into one is asserting
against values the checker knows nothing about. A test narrows a loaded document
to a mapping with ``mosaic.core.narrowing.is_str_mapping``, and this flattens it
to the dotted keys a Hydra override names, which is the form both training
suites compare against: the assignments an op injects are already dotted, so a
flattened document can be compared to them directly.
"""

from __future__ import annotations

from collections.abc import Mapping

from mosaic.core.narrowing import is_str_mapping

__all__ = ["dotted_values"]


def dotted_values(node: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    """Every dotted key in *node*, at every depth, with the value it carries."""
    flat: dict[str, object] = {}
    for name, value in node.items():
        dotted = f"{prefix}{name}"
        flat[dotted] = value
        if is_str_mapping(value):
            flat.update(dotted_values(value, f"{dotted}."))
    return flat
