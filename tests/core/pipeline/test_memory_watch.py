"""The measurement of a run's memory and of what the host has left."""

from __future__ import annotations

import sys
import types

import pytest

from mosaic.core.pipeline.memory_watch import MemoryWatch, host_memory_bytes

GIB = 1024**3


class _Error(Exception):
    """Stands in for ``psutil.Error``."""


class _Process:
    def __init__(self, rss: int, children: list[_Process] | None = None) -> None:
        self._rss = rss
        self._children = children or []

    def children(self, recursive: bool = False) -> list[_Process]:
        _ = recursive
        return self._children

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=self._rss)


class _Vanished(_Process):
    def memory_info(self) -> types.SimpleNamespace:
        raise _Error


def _install_psutil(
    monkeypatch: pytest.MonkeyPatch,
    process: _Process,
    *,
    available: int = 32 * GIB,
    total: int = 64 * GIB,
) -> None:
    host = types.SimpleNamespace(available=available, total=total)
    module = types.SimpleNamespace(
        Process=lambda: process,
        virtual_memory=lambda: host,
        Error=_Error,
    )
    monkeypatch.setitem(sys.modules, "psutil", module)


def test_a_sample_adds_the_process_and_every_process_below_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = [_Process(2 * GIB), _Process(3 * GIB)]
    _install_psutil(monkeypatch, _Process(1 * GIB, workers))

    sample = MemoryWatch().sample()

    assert sample is not None
    assert sample.tree_rss_bytes == 6 * GIB
    assert (sample.available_bytes, sample.total_bytes) == (32 * GIB, 64 * GIB)


def test_a_process_that_ended_during_the_sample_is_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(1 * GIB, [_Vanished(5 * GIB)]))

    sample = MemoryWatch().sample()

    assert sample is not None
    assert sample.tree_rss_bytes == 1 * GIB


def test_the_watch_keeps_the_largest_resident_memory_and_the_least_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch = MemoryWatch()
    _install_psutil(monkeypatch, _Process(4 * GIB), available=20 * GIB)
    _ = watch.sample()
    _install_psutil(monkeypatch, _Process(2 * GIB), available=30 * GIB)
    _ = watch.sample()

    assert watch.peak_rss_bytes == 4 * GIB
    assert watch.min_available_bytes == 20 * GIB
    assert watch.total_bytes == 64 * GIB


def test_a_sample_inside_the_interval_is_not_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(GIB))
    watch = MemoryWatch(interval_seconds=3600)

    assert watch.sample_if_due() is True
    assert watch.sample_if_due() is False


def test_the_free_fraction_is_the_available_share_of_the_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(GIB), available=16 * GIB, total=64 * GIB)

    assert MemoryWatch().free_fraction() == 0.25
    assert host_memory_bytes() == (64 * GIB, 16 * GIB)


def test_a_host_without_psutil_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    watch = MemoryWatch()

    assert watch.sample() is None
    assert watch.sample_if_due() is False
    assert watch.free_fraction() is None
    assert host_memory_bytes() == (0, 0)
