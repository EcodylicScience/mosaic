"""The lookup a converter is handed to find a dataset root by its key.

A converter reads a root, such as a tracker's working root, without importing
``dataset.py``: the dataset passes a :data:`RootLookup`, which returns the root's
directory, or ``None`` for a root the dataset does not declare. This module
imports the standard library only.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

__all__ = ["RootLookup"]

RootLookup: TypeAlias = Callable[[str], Path | None]
