"""Wrappers that let a test watch calls of what it runs, keeping each signature."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
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


def record_path_calls(
    monkeypatch: pytest.MonkeyPatch, owner: ModuleType, name: str
) -> list[Path]:
    """Replace the function *name* of *owner* with one that records each path it receives.

    The replacement returns what the original returns. The list holds the first
    argument of each call, in call order, until the test ends and *monkeypatch*
    restores the original.
    """
    paths: list[Path] = []
    real: Callable[[Path], object] = getattr(owner, name)
    monkeypatch.setattr(owner, name, observed(real, paths.append))
    return paths


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


def raise_high_water_mark_once_applied(
    monkeypatch: pytest.MonkeyPatch, owner: type[object], amount: int
) -> None:
    """Make the process's high-water mark read *amount* bytes higher once *owner* applies.

    ``owner.apply`` is wrapped to note that it ran in this process. From then on
    :func:`mosaic.core.pipeline.memory_watch.max_rss_bytes` reads *amount* more than
    the real figure, so a run's watch sees its mark rise past the figure it read
    when it began. Both patches last until the test ends and *monkeypatch*
    restores them. A worker process, which imports its own copy of the module,
    is unaffected.
    """
    from mosaic.core.pipeline import memory_watch

    applied = threading.Event()
    real_max_rss_bytes = memory_watch.max_rss_bytes

    def raised() -> int:
        return real_max_rss_bytes() + (amount if applied.is_set() else 0)

    monkeypatch.setattr(memory_watch, "max_rss_bytes", raised)
    apply = getattr(owner, "apply")
    monkeypatch.setattr(owner, "apply", observed(apply, lambda *_args: applied.set()))
