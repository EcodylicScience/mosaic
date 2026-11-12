"""Wrappers that let a test watch calls of what it runs, keeping each signature."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import pytest

Params = ParamSpec("Params")
Result = TypeVar("Result")
Owner = TypeVar("Owner")


def observed(
    real: Callable[Params, Result], on_call: Callable[Params, object]
) -> Callable[Params, Result]:
    """*real* with the same signature, calling *on_call* with its arguments first."""

    def wrapper(*args: Params.args, **kwargs: Params.kwargs) -> Result:
        _ = on_call(*args, **kwargs)
        return real(*args, **kwargs)

    return wrapper


def count_property_reads(
    monkeypatch: pytest.MonkeyPatch, owner: type[Owner], name: str
) -> list[int]:
    """Replace the property *name* of *owner* with one that records each read.

    The replacement returns what the original returns. The list holds a 1 for
    each read, until the test ends and *monkeypatch* restores the original.
    """
    original = inspect.getattr_static(owner, name)
    assert isinstance(original, property)
    getter = original.fget
    assert getter is not None
    reads: list[int] = []

    def counting(instance: Owner) -> object:
        reads.append(1)
        value: object = getter(instance)
        return value

    monkeypatch.setattr(owner, name, property(counting))
    return reads
