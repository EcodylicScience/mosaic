"""A byte count as the decimal size a person reads, such as ``12.6 MB``.

This module imports nothing, so the example downloader and the CLI share it without
loading anything else.
"""

from __future__ import annotations

__all__ = ["format_size"]


def format_size(size: int) -> str:
    """*size* in bytes as ``12 B``, ``3.4 KB``, ``12.6 MB`` or ``1.9 GB``.

    Units are powers of 1000. The largest unit is ``GB``, so a size of a terabyte or
    more prints as a number of ``GB``.
    """
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"
