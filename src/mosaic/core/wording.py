"""Phrases that mosaic's messages share. A stdlib-only leaf."""

from __future__ import annotations

__all__ = ["counted"]


def counted(count: int, noun: str) -> str:
    """Return *count* and *noun* as prose, the noun plural unless the count is one.

    Args:
        count: How many there are.
        noun: The singular of a noun whose plural adds ``s``.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
