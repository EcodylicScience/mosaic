"""The body of a package ``__getattr__`` that binds public names on access.

A package that imports its public names at module scope charges every consumer
of every submodule for them, because Python executes the package ``__init__``
first. ``mosaic.core.pipeline``, ``mosaic.tracking`` and the tracking packages
that re-export a surface declare a table of sources instead and call
:func:`lazy_export` from their ``__getattr__``. This module imports only the
standard library.
"""

from collections.abc import Mapping
from importlib import import_module

__all__ = ["lazy_export"]


def lazy_export(package: str, sources: Mapping[str, str], name: str) -> object:
    """Return the public *name* of *package*, importing its module on first access.

    ``import_module`` imports the module. A ``from . import`` inside a package
    ``__getattr__`` would probe the package with ``hasattr`` and re-enter it.

    Args:
        package: The package's ``__name__``, which anchors a relative source.
        sources: Each public name's module, absolute or relative to *package*.
            A source ends in ``:<attribute>`` when the module binds the object
            under another name.
        name: The attribute asked of the package.

    Raises:
        AttributeError: *sources* names no module for *name*. ``from <package>
            import <submodule>`` then falls through to the submodule import.
    """
    source = sources.get(name)
    if source is None:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    module, _, attribute = source.partition(":")
    return getattr(import_module(module, package), attribute or name)
