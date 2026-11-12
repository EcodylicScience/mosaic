"""The values that a run summary's arguments take: key values and test names.

:func:`~mosaic.core.pipeline.run_summary.summarize_run`, ``mosaic summarize`` and the
``summary`` mode of the MCP tool ``preview_output`` share them. The module is a
leaf. It imports the standard library only, so the tool contract declares its
arguments without loading the analysis stack.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Final, Literal

__all__ = ["SUMMARY_TESTS", "KeyValue", "SummaryTest", "read_where", "readings"]

type KeyValue = str | int | float | bool
"""One value of a group key."""

type SummaryTest = Literal["kruskal", "mannwhitney"]
"""A test between groups: Kruskal-Wallis over all groups, or Mann-Whitney per pair."""

SUMMARY_TESTS: Final[tuple[SummaryTest, ...]] = ("kruskal", "mannwhitney")
"""Every test between groups, in the order a summary runs them."""

_INTEGER: Final = re.compile(r"[+-]?\d+")
"""The text of a whole number."""


def readings(text: str) -> list[KeyValue]:
    """Return *text* and each number or boolean that it reads as.

    The readings are the text itself, an integer for a whole number, a float for
    any other finite number, and a boolean for ``true`` or ``false``. A key holds
    one family of value (text, number or boolean), so a summary matches the
    readings of that family and no other, and ``1`` matches ``1.0``.
    """
    found: list[KeyValue] = [text]
    if _INTEGER.fullmatch(text):
        found.append(int(text))
    else:
        try:
            number = float(text)
        except ValueError:
            pass
        else:
            if math.isfinite(number):
                found.append(number)
    if text in ("true", "false"):
        found.append(text == "true")
    return found


def read_where(
    where: Mapping[str, Sequence[KeyValue]],
) -> dict[str, list[KeyValue]]:
    """Return *where* with each text value read as :func:`readings` gives.

    ``{"n_inds": ["1"]}`` keeps the rows whose integer ``n_inds`` is 1, and
    ``{"code": ["1"]}`` those whose text ``code`` is ``"1"``. A number or a
    boolean stays as it is.
    """
    return {
        name: [
            reading
            for value in values
            for reading in (readings(value) if isinstance(value, str) else [value])
        ]
        for name, values in where.items()
    }
