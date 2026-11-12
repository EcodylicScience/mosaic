"""The measurement of a run's memory and of what the limit that applies has left."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline.memory_limit import MemoryLimit, usable_memory_bytes
from mosaic.core.pipeline.memory_watch import MemoryWatch

GIB = 1024**3


class _Error(Exception):
    """Stands in for ``psutil.Error``."""


_SPAWNED = [
    "python",
    "-c",
    "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)",
    "--multiprocessing-fork",
]
"""The command line of a worker that ``multiprocessing`` spawned."""

_pids = iter(range(1000, 100_000))


class _Process:
    def __init__(
        self,
        rss: int,
        children: list[_Process] | None = None,
        *,
        cmdline: list[str] | None = None,
    ) -> None:
        self._rss = rss
        self.child_processes = children or []
        self._cmdline = cmdline or ["python"]
        self.pid = next(_pids)

    def cmdline(self) -> list[str]:
        return self._cmdline

    def cpu_times(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            user=0.0, system=0.0, children_user=0.0, children_system=0.0
        )

    def children(self, recursive: bool = False) -> list[_Process]:
        if not recursive:
            return self.child_processes
        below: list[_Process] = []
        for child in self.child_processes:
            below.extend([child, *child.children(recursive=True)])
        return below

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=self._rss)


class _Vanished(_Process):
    def memory_info(self) -> types.SimpleNamespace:
        raise _Error


def _no_cgroup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the reader at a tree with no cgroup files, as on a host or macOS."""
    from mosaic.core.pipeline import memory_limit

    real = memory_limit.read_memory_limit
    monkeypatch.setattr(
        memory_limit,
        "read_memory_limit",
        lambda: real(
            cgroup_root=tmp_path / "cgroup", proc_self_cgroup=tmp_path / "proc"
        ),
    )
    monkeypatch.setattr(
        memory_watch, "read_memory_limit", memory_limit.read_memory_limit
    )


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


def test_a_sample_counts_the_spawned_workers_started_since_measuring_began(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    earlier = _Process(GIB, cmdline=_SPAWNED)
    root = _Process(GIB, [earlier])
    _install_psutil(monkeypatch, root)
    watch = MemoryWatch()
    watch.begin()
    tracker = _Process(
        GIB,
        cmdline=[
            "python",
            "-c",
            "from multiprocessing.resource_tracker import main;main(5)",
        ],
    )
    root.child_processes = [earlier, tracker, _Process(GIB, cmdline=_SPAWNED)]

    sample = watch.sample()

    assert sample is not None
    assert sample.spawned_workers == 1


def test_a_child_that_a_worker_forked_is_not_a_worker_but_counts_in_the_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fork keeps its parent's command line, so a worker's own child reads as a
    # spawned worker by its command line alone.
    root = _Process(GIB)
    _install_psutil(monkeypatch, root)
    watch = MemoryWatch()
    watch.begin()
    forked = _Process(2 * GIB, cmdline=_SPAWNED)
    root.child_processes = [_Process(GIB, [forked], cmdline=_SPAWNED)]

    sample = watch.sample()

    assert sample is not None
    assert sample.spawned_workers == 1
    assert sample.tree_rss_bytes == 4 * GIB


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


def _cgroup(*, limit: int, in_use: int, peak: int = 0) -> MemoryLimit:
    return MemoryLimit(
        limit_bytes=limit,
        source="cgroup_v2",
        in_use_bytes=in_use,
        peak_bytes=peak,
        host_total_bytes=64 * GIB,
        host_available_bytes=32 * GIB,
    )


def test_the_free_fraction_on_a_host_is_the_available_share_of_the_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_psutil(monkeypatch, _Process(GIB), available=16 * GIB, total=64 * GIB)
    _no_cgroup(monkeypatch, tmp_path)

    assert MemoryWatch().free_fraction() == 0.25
    assert usable_memory_bytes() == (64 * GIB, 16 * GIB)


def test_the_free_fraction_follows_the_cgroup_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(GIB), available=60 * GIB, total=64 * GIB)
    monkeypatch.setattr(
        memory_watch,
        "read_memory_limit",
        lambda: _cgroup(limit=8 * GIB, in_use=6 * GIB),
    )

    assert MemoryWatch().free_fraction() == 0.25


def test_a_sample_keeps_the_limit_beside_the_host_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(GIB), available=20 * GIB, total=64 * GIB)
    limits = iter(
        [
            _cgroup(limit=8 * GIB, in_use=2 * GIB),
            _cgroup(limit=8 * GIB, in_use=7 * GIB),
            _cgroup(limit=8 * GIB, in_use=4 * GIB),
        ]
    )
    monkeypatch.setattr(memory_watch, "read_memory_limit", lambda: next(limits))
    watch = MemoryWatch()

    samples = [watch.sample() for _ in range(3)]

    sample = samples[-1]
    assert sample is not None
    assert sample.limit.in_use_bytes == 4 * GIB
    assert watch.limit == sample.limit
    assert watch.min_available_in_limit_bytes == 1 * GIB
    assert watch.min_available_bytes == 20 * GIB
    assert watch.total_bytes == 64 * GIB


def test_the_watch_has_no_limit_before_its_first_sample() -> None:
    watch = MemoryWatch()

    assert watch.limit is None
    assert watch.min_available_in_limit_bytes == 0


def test_a_host_without_psutil_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    watch = MemoryWatch()

    assert watch.sample() is None
    assert watch.sample_if_due() is False
    assert watch.free_fraction() is None
    assert usable_memory_bytes() == (0, 0)
