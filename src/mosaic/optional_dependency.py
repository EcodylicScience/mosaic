"""One message for "this needs something the install did not bring".

A deliberate leaf, like :mod:`mosaic.runlog` and :mod:`mosaic.media_probe_config`:
it imports nothing from mosaic, so any layer may reach it without creating a
cycle.

Every optional dependency used to raise its own hand-written ``ImportError``, in
four different shapes across ten modules. They disagreed about what to tell the
user -- some named the mosaic extra, some named the raw PyPI package, one named
an extra that has never existed (``mosaic[movement]``, wrong on the distribution
name too) -- and nothing tied any of them to what ``pyproject.toml`` actually
declares. A message that names the wrong extra is worse than no message: it sends
someone to install a thing that will not fix their problem.

So the extra name is passed in and rendered one way, and
``tests/test_optional_dependency_messages.py`` checks every name used here
against the declared extras. A rename that misses a call site fails there rather
than in a user's terminal.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["MissingOptionalDependency", "require"]


class MissingOptionalDependency(ImportError):
    """An optional dependency is not importable.

    An ``ImportError`` subclass so that the call sites which already catch
    ``ImportError`` -- notably the CLI's ``mosaic run`` handler -- keep working
    unchanged, while code that wants to tell this apart from a genuine broken
    import can.
    """

    def __init__(self, module: str, extra: str, purpose: str) -> None:
        self.module = module
        self.extra = extra
        self.purpose = purpose
        super().__init__(
            f"{module} is required for {purpose}, and is not installed. "
            f'Install it with: pip install "mosaic-behavior[{extra}]"'
        )


def require(module: str, extra: str, purpose: str) -> ModuleType:
    """Import ``module``, or raise naming the extra that provides it.

    Args:
        module: The import name to load, e.g. ``"torch"``. Not the distribution
            name -- they differ often enough (``tables`` for PyTables,
            ``lightning_action`` for lightning-action) that using the wrong one
            is the mistake this argument exists to avoid.
        extra: The mosaic extra that installs it, e.g. ``"deep-learning"``. Must
            be a name declared in ``pyproject.toml``; the test above enforces it.
        purpose: What needs it, as a noun phrase completing "X is required for
            ...". Written for someone who does not know mosaic's internals, so
            "the identity models" rather than "TimmBackbone._build".

    Returns:
        The imported module.

    Raises:
        MissingOptionalDependency: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingOptionalDependency(module, extra, purpose) from None
