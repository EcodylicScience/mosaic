"""The measurement of a run's memory and of what the limit that applies has left."""

from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline.memory_limit import MemoryLimit, usable_memory_bytes
from mosaic.core.pipeline.memory_watch import (
    MemoryWatch,
    peak_nonreclaimable_while,
    process_nonreclaimable_bytes,
)

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


def _cgroup(
    *, limit: int, in_use: int, peak: int = 0, nonreclaimable: int = 0
) -> MemoryLimit:
    return MemoryLimit(
        limit_bytes=limit,
        source="cgroup_v2",
        in_use_bytes=in_use,
        peak_bytes=peak,
        host_total_bytes=64 * GIB,
        host_available_bytes=32 * GIB,
        nonreclaimable_bytes=nonreclaimable,
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


def _status(proc_root: Path, pid: int | str, anon_kb: int, shmem_kb: int) -> None:
    """Write a ``/proc/<pid>/status`` with the resident split lines among others."""
    path = proc_root / str(pid) / "status"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Name:\tpython\n"
        f"VmRSS:\t{(anon_kb + shmem_kb + 999)} kB\n"
        "RssAnon:\t" + f"{anon_kb:>8} kB\n"
        "RssFile:\t     999 kB\n"
        "RssShmem:\t" + f"{shmem_kb:>7} kB\n"
        "Threads:\t4\n"
    )


def test_a_process_reads_its_anonymous_and_shared_resident_memory(
    tmp_path: Path,
) -> None:
    _status(tmp_path, 4242, anon_kb=100, shmem_kb=20)

    assert process_nonreclaimable_bytes(4242, proc_root=tmp_path) == 120 * 1024


def test_a_process_without_a_pid_reads_its_own_status(tmp_path: Path) -> None:
    _status(tmp_path, "self", anon_kb=7, shmem_kb=1)

    assert process_nonreclaimable_bytes(proc_root=tmp_path) == 8 * 1024


def test_a_missing_proc_or_a_gone_process_reads_none(tmp_path: Path) -> None:
    assert process_nonreclaimable_bytes(proc_root=tmp_path / "absent") is None
    assert process_nonreclaimable_bytes(31337, proc_root=tmp_path) is None


def test_a_status_without_the_resident_split_reads_none(tmp_path: Path) -> None:
    path = tmp_path / "5" / "status"
    path.parent.mkdir()
    path.write_text("Name:\tkthreadd\nVmRSS:\t 4 kB\n")

    assert process_nonreclaimable_bytes(5, proc_root=tmp_path) is None


def _host_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Process:
    """A host with a parent and two workers, and no cgroup limit."""
    workers = [_Process(2 * GIB), _Process(3 * GIB)]
    root = _Process(GIB, workers)
    _install_psutil(monkeypatch, root)
    _no_cgroup(monkeypatch, tmp_path)
    return root


def test_a_host_sample_sums_the_status_of_every_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _host_tree(monkeypatch, tmp_path)
    proc_root = tmp_path / "procfs"
    for member, anon in zip([root, *root.child_processes], [10, 20, 30], strict=True):
        _status(proc_root, member.pid, anon_kb=anon, shmem_kb=1)

    sample = MemoryWatch(proc_root=proc_root).sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes == (10 + 20 + 30 + 3) * 1024


def test_a_process_that_has_gone_adds_nothing_to_the_sum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _host_tree(monkeypatch, tmp_path)
    proc_root = tmp_path / "procfs"
    _status(proc_root, root.pid, anon_kb=10, shmem_kb=0)
    _status(proc_root, root.child_processes[0].pid, anon_kb=20, shmem_kb=0)

    sample = MemoryWatch(proc_root=proc_root).sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes == 30 * 1024


def test_a_live_process_with_an_unreadable_status_leaves_the_figure_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _host_tree(monkeypatch, tmp_path)
    proc_root = tmp_path / "procfs"
    _status(proc_root, root.pid, anon_kb=10, shmem_kb=0)
    _status(proc_root, root.child_processes[0].pid, anon_kb=20, shmem_kb=0)
    bad = proc_root / str(root.child_processes[1].pid) / "status"
    bad.parent.mkdir()
    bad.write_text("Name:\tpython\n")

    sample = MemoryWatch(proc_root=proc_root).sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes is None


def test_a_zombie_or_dead_process_adds_nothing_to_the_sum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _host_tree(monkeypatch, tmp_path)
    proc_root = tmp_path / "procfs"
    _status(proc_root, root.pid, anon_kb=10, shmem_kb=0)
    for member, state in zip(root.child_processes, ["Z (zombie)", "X (dead)"]):
        path = proc_root / str(member.pid) / "status"
        path.parent.mkdir(parents=True)
        path.write_text(f"Name:\tpython\nState:\t{state}\nPid:\t{member.pid}\n")

    sample = MemoryWatch(proc_root=proc_root).sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes == 10 * 1024


def test_a_sample_without_proc_has_no_figure_on_a_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _host_tree(monkeypatch, tmp_path)
    watch = MemoryWatch(proc_root=tmp_path / "absent")

    sample = watch.sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes is None
    assert watch.nonreclaimable_bytes == 0
    assert watch.peak_nonreclaimable_bytes == 0


def test_a_cgroup_sample_carries_the_limits_figure_once_for_the_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _Process(GIB, [_Process(2 * GIB), _Process(3 * GIB)])
    _install_psutil(monkeypatch, root)
    monkeypatch.setattr(
        memory_watch,
        "read_memory_limit",
        lambda: _cgroup(limit=8 * GIB, in_use=GIB, nonreclaimable=5 * GIB),
    )
    proc_root = tmp_path / "procfs"
    for member in [root, *root.child_processes]:
        _status(proc_root, member.pid, anon_kb=1000, shmem_kb=0)

    sample = MemoryWatch(proc_root=proc_root).sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes == 5 * GIB


def test_a_cgroup_figure_of_zero_is_not_a_measurement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_psutil(monkeypatch, _Process(GIB))
    monkeypatch.setattr(
        memory_watch,
        "read_memory_limit",
        lambda: _cgroup(limit=8 * GIB, in_use=GIB, nonreclaimable=0),
    )
    watch = MemoryWatch(proc_root=tmp_path / "absent")

    sample = watch.sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes is None
    assert watch.nonreclaimable_bytes == 0


def test_a_cgroup_sample_needs_no_proc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_psutil(monkeypatch, _Process(GIB))
    monkeypatch.setattr(
        memory_watch,
        "read_memory_limit",
        lambda: _cgroup(limit=8 * GIB, in_use=GIB, nonreclaimable=GIB),
    )

    sample = MemoryWatch(proc_root=tmp_path / "absent").sample()

    assert sample is not None
    assert sample.nonreclaimable_bytes == GIB


def test_the_watch_keeps_the_latest_non_reclaimable_figure_and_the_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(GIB))
    figures = iter([2 * GIB, 6 * GIB, 4 * GIB])
    monkeypatch.setattr(
        memory_watch,
        "read_memory_limit",
        lambda: _cgroup(limit=8 * GIB, in_use=GIB, nonreclaimable=next(figures)),
    )
    watch = MemoryWatch()

    for _ in range(3):
        _ = watch.sample()

    assert watch.nonreclaimable_bytes == 4 * GIB
    assert watch.peak_nonreclaimable_bytes == 6 * GIB


def _wait_for(condition: Callable[[], bool], *, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.005)


def test_the_peak_while_a_block_runs_is_the_largest_read(tmp_path: Path) -> None:
    _status(tmp_path, "self", anon_kb=100, shmem_kb=0)
    with peak_nonreclaimable_while(interval=0.01, proc_root=tmp_path) as holder:
        _status(tmp_path, "self", anon_kb=900, shmem_kb=0)
        _wait_for(lambda: holder.peak == 900 * 1024)
        _status(tmp_path, "self", anon_kb=300, shmem_kb=0)
        time.sleep(0.05)
        assert holder.peak == 900 * 1024
    assert holder.peak == 900 * 1024


def test_a_read_on_exit_can_set_the_peak(tmp_path: Path) -> None:
    _status(tmp_path, "self", anon_kb=100, shmem_kb=0)
    with peak_nonreclaimable_while(interval=3600, proc_root=tmp_path) as holder:
        assert holder.peak == 100 * 1024
        _status(tmp_path, "self", anon_kb=500, shmem_kb=0)

    assert holder.peak == 500 * 1024


def test_the_peak_is_none_when_no_read_succeeded(tmp_path: Path) -> None:
    with peak_nonreclaimable_while(
        interval=0.01, proc_root=tmp_path / "absent"
    ) as holder:
        pass

    assert holder.peak is None


def _sampler_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "mosaic-worker-memory-sampler"
    ]


def test_the_sampler_thread_runs_inside_the_block_and_is_gone_after_it(
    tmp_path: Path,
) -> None:
    _status(tmp_path, "self", anon_kb=1, shmem_kb=0)
    with peak_nonreclaimable_while(interval=0.01, proc_root=tmp_path):
        threads = _sampler_threads()
        assert len(threads) == 1
        assert threads[0].daemon

    assert _sampler_threads() == []


def test_the_sampler_thread_is_gone_after_the_block_raises(tmp_path: Path) -> None:
    _status(tmp_path, "self", anon_kb=1, shmem_kb=0)

    with pytest.raises(RuntimeError, match="boom"):
        with peak_nonreclaimable_while(interval=0.01, proc_root=tmp_path):
            raise RuntimeError("boom")

    assert _sampler_threads() == []
