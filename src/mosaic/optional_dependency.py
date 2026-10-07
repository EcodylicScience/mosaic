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
``tests/meta/test_optional_dependency_messages.py`` checks every name used here
against the declared extras. A rename that misses a call site fails there rather
than in a user's terminal.

A feature declares the packages and setups beyond the base install that it
needs as ``prerequisites``, a tuple of :class:`Prerequisite`. The pipeline catalog
and recipe validation read them through :func:`unmet`, which imports none of the
modules it checks. A step type lacking one is offered as unavailable, with a
sentence naming the package to install or the variable to set, and a recipe step
running it is a problem.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, runtime_checkable

__all__ = [
    "MissingOptionalDependency",
    "OptionalModule",
    "Prerequisite",
    "require",
    "unmet",
]


def _install_hint(module: str, extra: str, purpose: str) -> str:
    """The one sentence that names a missing module and the extra installing it."""
    return (
        f"{module} is required for {purpose}, and is not installed. "
        f'Install it with: pip install "mosaic-behavior[{extra}]"'
    )


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
        super().__init__(_install_hint(module, extra, purpose))


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


@runtime_checkable
class Prerequisite(Protocol):
    """Something a feature needs that the base install does not provide.

    A package an extra installs, or a setup the user enables, such as an accepted
    license. A feature lists its prerequisites in a ``prerequisites`` class
    attribute, and declares none when the base install runs it.
    """

    def missing(self) -> str | None:
        """A sentence naming the missing package or setup and how to provide it.

        ``None`` when the prerequisite is met. Answered without importing or
        starting the package or program it checks, because the pipeline catalog
        asks it of every feature.
        """
        ...


@dataclass(frozen=True, slots=True)
class OptionalModule:
    """A module that one of mosaic's extras installs.

    The arguments of :func:`require`, declared once. A feature lists it among its
    prerequisites and imports the module through :meth:`load`. The catalog's
    sentence and the run's error then name the same extra.

    Attributes:
        module: The top-level import name, e.g. ``"movement"``.
        extra: The mosaic extra that installs it.
        purpose: What needs it, as :func:`require` phrases it.
    """

    module: str
    extra: str
    purpose: str

    def missing(self) -> str | None:
        """The install hint when the module cannot be found, else ``None``.

        ``find_spec`` locates a top-level module without importing it, and a
        finder that refuses the name reads as missing. ``sys.modules`` is asked
        first, because ``find_spec`` raises for a module built in place, as a
        test stub is, whose ``__spec__`` is ``None``.
        """
        if self.module in sys.modules:
            return None
        try:
            spec = importlib.util.find_spec(self.module)
        except ImportError:
            spec = None
        if spec is not None:
            return None
        return _install_hint(self.module, self.extra, self.purpose)

    def load(self) -> ModuleType:
        """Import the module, or raise :class:`MissingOptionalDependency`."""
        return require(self.module, self.extra, self.purpose)


def unmet(prerequisites: Iterable[Prerequisite]) -> str | None:
    """The unmet prerequisites' sentences, in declaration order, or ``None``."""
    lacking = [
        message
        for message in (prerequisite.missing() for prerequisite in prerequisites)
        if message is not None
    ]
    return " ".join(lacking) or None
