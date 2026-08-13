"""The installed toolkit version, read from distribution metadata.

A deliberate leaf, imported by anything that has to record *which mosaic*
produced an artifact. It takes no import from ``core``, ``behavior`` or
``tracking``.

**The distribution name is not the import package name.** mosaic is imported as
``mosaic`` and distributed as ``mosaic-behavior``, so the lookup has to name the
latter; asking for ``mosaic`` finds nothing, or worse finds an unrelated project
of that name in the same environment.

**Absent is a legitimate answer, not an error.** A source tree on ``sys.path``
with nothing installed -- a checkout run from its own directory, a worktree
reached through ``PYTHONPATH`` -- has no distribution metadata at all. Recording
a run from one is a normal thing to do, so an unknown version is written as the
empty string and read the way every other unestablishable provenance cell in
this toolkit is read: empty means *unknown*, never *none*.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _distribution_version
from typing import Final

__all__ = ["DISTRIBUTION_NAME", "installed_version"]

DISTRIBUTION_NAME: Final = "mosaic-behavior"
"""What mosaic is distributed as, which is not what it is imported as."""


def installed_version() -> str:
    """The installed ``mosaic-behavior`` version, or ``""`` when it is unknown.

    Returns:
        The version string from distribution metadata, e.g. ``"0.11.0"``. Empty
        when mosaic is not installed in the running environment, which is what
        an uninstalled source tree looks like.
    """
    try:
        return _distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return ""
