"""Find the entry points that installed packages declare, and load one of them.

A package extends mosaic by declaring an entry point in a group such as
``mosaic.sequence_heads``. :func:`installed_entry_points` lists a group's entry
points by name without importing any of them. :func:`load_entry_point` imports one
and checks what it holds against the type the group expects, and raises
:class:`EntryPointRefused` naming the entry point's :class:`InstalledOrigin` when
the import fails or the object is not of that type. A caller decides what a refusal
costs: it raises it, or records it and carries on.

This module imports the standard library only, so ``core`` and the feature library
share it without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Literal, Self, overload

__all__ = [
    "EntryPointRefused",
    "InstalledOrigin",
    "installed_entry_points",
    "load_entry_point",
]


@dataclass(frozen=True, slots=True)
class InstalledOrigin:
    """Where an installed object comes from: its entry point and its distribution.

    Attributes:
        group: The entry point group.
        name: The entry point's name.
        distribution: The name of the distribution that declares it, blank when
            unknown.
        version: The distribution's version, blank when unknown.
        value: The object reference the entry point names, ``module:attribute``.
    """

    group: str
    name: str
    distribution: str
    version: str
    value: str

    @classmethod
    def from_entry_point(cls, found: EntryPoint) -> Self:
        """Read the origin of *found*, with a blank distribution when it has none."""
        distribution = found.dist
        return cls(
            group=found.group,
            name=found.name,
            distribution=distribution.name if distribution is not None else "",
            version=distribution.version if distribution is not None else "",
            value=found.value,
        )

    def describe(self) -> str:
        """Return the entry point and its distribution, as a refusal names them."""
        owner = (
            f"distribution {self.distribution!r}"
            if self.distribution
            else "an unknown distribution"
        )
        return f"entry point {self.name!r} of {owner}"


class EntryPointRefused(ValueError):
    """An entry point cannot be loaded or holds an object of the wrong type."""


def installed_entry_points(group: str) -> dict[str, tuple[EntryPoint, ...]]:
    """Return the entry points of *group* by name, sorted by name, none loaded.

    A name that two installed packages declare maps to both entry points, so the
    caller can refuse the ambiguity and name each origin.
    """
    by_name: dict[str, list[EntryPoint]] = {}
    for found in entry_points(group=group):
        by_name.setdefault(found.name, []).append(found)
    return {name: tuple(by_name[name]) for name in sorted(by_name)}


@overload
def load_entry_point[Expected](
    found: EntryPoint, expected: type[Expected], *, subclass: Literal[False] = False
) -> Expected: ...


@overload
def load_entry_point[Expected](
    found: EntryPoint, expected: type[Expected], *, subclass: Literal[True]
) -> type[Expected]: ...


def load_entry_point[Expected](
    found: EntryPoint, expected: type[Expected], *, subclass: bool = False
) -> Expected | type[Expected]:
    """Load *found* and return what it names when that is what *expected* asks for.

    Args:
        found: The entry point to import.
        expected: The type of the object, or, with *subclass*, the base class of
            the class, that the entry point must name.
        subclass: Whether the entry point names a class that subclasses *expected*
            instead of an instance of it.

    Returns:
        The loaded object: an instance of *expected*, or with *subclass* a class
        that subclasses it.

    Raises:
        EntryPointRefused: The import raises, or the object is not what *expected*
            asks for. The message names the entry point and its distribution.
    """
    origin = InstalledOrigin.from_entry_point(found).describe()
    try:
        loaded = found.load()
    except Exception as error:
        msg = f"{origin} could not be loaded: {type(error).__name__}: {error}"
        raise EntryPointRefused(msg) from error
    if not subclass:
        if not isinstance(loaded, expected):
            msg = f"{origin} is a {type(loaded).__name__}, not a {expected.__name__}"
            raise EntryPointRefused(msg)
        return loaded
    if not isinstance(loaded, type):
        msg = (
            f"{origin} is a {type(loaded).__name__}, not a {expected.__name__} subclass"
        )
        raise EntryPointRefused(msg)
    if not issubclass(loaded, expected):
        msg = f"{origin} is the class {loaded.__name__}, not a {expected.__name__} subclass"
        raise EntryPointRefused(msg)
    return loaded
