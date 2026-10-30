"""Reading the release numbers of a tool's version string.

A leaf module that imports only the standard library. A tracker's version
module can use it at module scope without importing the tracker.
"""

from __future__ import annotations

import re
from typing import Final

_RELEASE: Final = re.compile(r"\d+(?:\.\d+)*")
"""The leading release numbers of a version, such as ``2.3.1`` of ``2.3.1.post1``."""


def release_numbers(version: str) -> tuple[int, ...]:
    """Return the leading release numbers of *version*, empty when it has none.

    A pre-release reads as its release, so ``0.3.5rc1`` is ``(0, 3, 5)``.

    Args:
        version: A version string such as ``2.3.1.post1``.

    Returns:
        The dotted integers at the start of *version*.
    """
    found = _RELEASE.match(version)
    if found is None:
        return ()
    return tuple(int(part) for part in found.group(0).split("."))
