"""The measurement of a run's memory and of what the limit that applies has left."""

from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

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


class _NoSuchProcess(_Error):
    """Stands in for ``psutil.NoSuchProcess``."""


class _ZombieProcess(_NoSuchProcess):
    """Stands in for ``psutil.ZombieProcess``, which psutil derives from ``NoSuchProcess``."""


class _AccessDenied(_Error):
    """Stands in for ``psutil.AccessDenied``."""


_SPAWNED = [
    "python",
    "-c",
    "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)",
    "--multiprocessing-fork",
]
"""The command line of a worker that ``multiprocessing`` spawned."""

_pids = iter(range(1000, 100_000))


class _Process:
    walks: ClassVar[int] = 0
    """The calls of :meth:`children` on any process since the count was reset."""

    def __init__(
        self,
        rss: int,
        children: list[_Process] | None = None,
        *,
        cmdline: list[str] | None = None,
    ) -> None:
        self._rss = rss
        self._cmdline = cmdline or ["python"]
        self.pid = next(_pids)
        self.parent_pid = 1
        self.cpu = 0.0
        """The user and system time that :meth:`cpu_times` reads."""
        self.read_error: type[_Error] | None = None
        """What every read of this process raises, ``None`` while it reads."""
        self.cmdline_error: type[_Error] | None = None
        self.walk_error: type[_Error] | None = None
        self._children: list[_Process] = []
        self.child_processes = children or []

    @property
    def child_processes(self) -> list[_Process]:
        return self._children

    @child_processes.setter
    def child_processes(self, children: list[_Process]) -> None:
        for child in children:
            child.parent_pid = self.pid
        self._children = children

    def ppid(self) -> int:
        if self.read_error is not None:
            raise self.read_error
        return self.parent_pid

    def cmdline(self) -> list[str]:
        if self.read_error is not None:
            raise self.read_error
        if self.cmdline_error is not None:
            raise self.cmdline_error
        return self._cmdline

    def cpu_times(self) -> types.SimpleNamespace:
        if self.read_error is not None:
            raise self.read_error
        return types.SimpleNamespace(
            user=self.cpu, system=0.0, children_user=0.0, children_system=0.0
        )

    def children(self, recursive: bool = False) -> list[_Process]:
        _Process.walks += 1
        if self.walk_error is not None:
            raise self.walk_error
        return self._below() if recursive else list(self._children)

    def _below(self) -> list[_Process]:
        below: list[_Process] = []
        for child in self._children:
            below.extend([child, *child._below()])
        return below

    def memory_info(self) -> types.SimpleNamespace:
        if self.read_error is not None:
            raise self.read_error
        return types.SimpleNamespace(rss=self._rss)


class _Vanished(_Process):
    def memory_info(self) -> types.SimpleNamespace:
        raise _Error


class _Clock:
    """The membership's clock, which moves only when the test moves it."""

    def __init__(self) -> None:
        self.time = 0.0

    def __call__(self) -> float:
        return self.time

    def advance(
        self, seconds: float = memory_watch.ESTIMATE_SAMPLE_INTERVAL_SECONDS
    ) -> None:
        self.time += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """The time that the age of every watch's membership reads."""
    frozen = _Clock()
    monkeypatch.setattr(memory_watch, "now", frozen)
    _Process.walks = 0
    return frozen


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
        NoSuchProcess=_NoSuchProcess,
        ZombieProcess=_ZombieProcess,
        AccessDenied=_AccessDenied,
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


def test_a_sample_records_when_its_measurement_began(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_psutil(monkeypatch, _Process(GIB))

    before = time.monotonic()
    sample = MemoryWatch().sample()
    after = time.monotonic()

    assert sample is not None
    assert sample.started_at is not None
    assert before <= sample.started_at <= after


def test_a_sample_counts_the_spawned_workers_started_since_measuring_began(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
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
    clock.advance()

    sample = watch.sample()

    assert sample is not None
    assert sample.spawned_workers == 1


def test_a_child_that_a_worker_forked_is_not_a_worker_but_counts_in_the_tree(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    # A fork keeps its parent's command line, so a worker's own child reads as a
    # spawned worker by its command line alone.
    root = _Process(GIB)
    _install_psutil(monkeypatch, root)
    watch = MemoryWatch()
    watch.begin()
    forked = _Process(2 * GIB, cmdline=_SPAWNED)
    root.child_processes = [_Process(GIB, [forked], cmdline=_SPAWNED)]
    clock.advance()

    sample = watch.sample()

    assert sample is not None
    assert sample.spawned_workers == 1
    assert sample.tree_rss_bytes == 4 * GIB


def _begun(monkeypatch: pytest.MonkeyPatch, root: _Process) -> MemoryWatch:
    """A watch begun over *root*, with the walks counted from after its begin."""
    _install_psutil(monkeypatch, root)
    watch = MemoryWatch()
    watch.begin()
    _Process.walks = 0
    return watch


def _tree_rss(watch: MemoryWatch) -> int:
    sample = watch.sample()
    assert sample is not None
    return sample.tree_rss_bytes


def _workers(watch: MemoryWatch) -> int:
    sample = watch.sample()
    assert sample is not None
    return sample.spawned_workers


def test_samples_within_the_interval_read_the_known_members_without_a_walk(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB, [_Process(2 * GIB)])
    watch = _begun(monkeypatch, root)

    assert _tree_rss(watch) == 3 * GIB
    root.child_processes = [*root.child_processes, _Process(4 * GIB)]
    assert _tree_rss(watch) == 3 * GIB
    assert _Process.walks == 0

    clock.advance()

    assert _tree_rss(watch) == 7 * GIB
    assert _Process.walks == 1


def test_sighting_walks_until_the_running_calls_have_workers_found_since_the_mark(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    earlier_pool = _Process(GIB, cmdline=_SPAWNED)
    root.child_processes = [earlier_pool]
    clock.advance()
    _ = watch.sample()
    mark = watch.walks
    _Process.walks = 0
    first, second = _Process(GIB, cmdline=_SPAWNED), _Process(GIB, cmdline=_SPAWNED)

    root.child_processes = [earlier_pool, first]
    watch.sight_workers(since=mark, running=1)
    assert _Process.walks == 1
    watch.sight_workers(since=mark, running=1)
    assert _Process.walks == 1
    watch.sight_workers(since=mark, running=2)
    assert _Process.walks == 2

    root.child_processes = [earlier_pool, first, second]
    watch.sight_workers(since=mark, running=2)
    assert _Process.walks == 3
    watch.sight_workers(since=mark, running=2)
    assert _Process.walks == 3
    assert _workers(watch) == 3


def test_a_process_with_a_known_members_pid_that_is_another_process_is_a_new_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # psutil compares processes on their pid and creation time, so a reused pid
    # names another process.
    original = _Process(GIB, cmdline=_SPAWNED)
    root = _Process(GIB, [original])
    watch = _begun(monkeypatch, root)
    mark = watch.walks
    successor = _Process(GIB, cmdline=_SPAWNED)
    successor.pid = original.pid
    root.child_processes = [successor]

    watch.sight_workers(since=mark, running=1)
    watch.sight_workers(since=mark, running=1)

    assert _Process.walks == 1
    assert _workers(watch) == 1


def test_a_member_that_has_gone_leaves_the_sum_the_workers_and_the_members(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    gone, staying = (
        _Process(2 * GIB, cmdline=_SPAWNED),
        _Process(4 * GIB, cmdline=_SPAWNED),
    )
    root.child_processes = [gone, staying]
    clock.advance()
    assert _workers(watch) == 2
    _Process.walks = 0

    gone.read_error = _NoSuchProcess
    sample = watch.sample()
    gone.read_error = None

    assert sample is not None
    assert (sample.tree_rss_bytes, sample.spawned_workers) == (5 * GIB, 1)
    assert (_tree_rss(watch), _workers(watch)) == (5 * GIB, 1)
    assert _Process.walks == 0


def test_a_zombie_member_adds_nothing_and_stays_a_member(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    zombie = _Process(2 * GIB, cmdline=_SPAWNED)
    root.child_processes = [zombie]
    clock.advance()
    _ = watch.sample()

    zombie.read_error = _ZombieProcess
    assert (_tree_rss(watch), _workers(watch)) == (GIB, 1)
    zombie.read_error = None
    assert _tree_rss(watch) == 3 * GIB


def test_workers_that_all_end_at_once_leave_a_sample_of_the_root_alone(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    # A cancel signals the process group, and every worker dies together.
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    workers = [_Process(2 * GIB, cmdline=_SPAWNED) for _ in range(3)]
    root.child_processes = workers
    clock.advance()
    assert _workers(watch) == 3

    for worker in workers:
        worker.read_error = _NoSuchProcess
    sample = watch.sample()

    assert sample is not None
    assert (sample.tree_rss_bytes, sample.spawned_workers) == (GIB, 0)


def test_a_walk_that_fails_keeps_the_members_and_leaves_the_next_read_to_walk(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    known = _Process(2 * GIB, cmdline=_SPAWNED)
    root.child_processes = [known]
    clock.advance()
    assert _workers(watch) == 1
    _Process.walks = 0
    root.child_processes = [known, _Process(4 * GIB, cmdline=_SPAWNED)]
    root.walk_error = _AccessDenied
    clock.advance()

    sample = watch.sample()

    assert sample is not None
    assert (sample.tree_rss_bytes, sample.spawned_workers) == (3 * GIB, 1)
    assert _Process.walks == 1
    _ = watch.sample()
    assert _Process.walks == 2

    root.walk_error = None
    assert (_tree_rss(watch), _workers(watch)) == (7 * GIB, 2)
    assert _Process.walks == 3


def test_a_worker_whose_command_line_is_denied_is_not_counted(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    denied = _Process(GIB, cmdline=_SPAWNED)
    denied.cmdline_error = _AccessDenied
    root.child_processes = [denied]
    clock.advance()

    sample = watch.sample()

    assert sample is not None
    assert (sample.tree_rss_bytes, sample.spawned_workers) == (2 * GIB, 0)


class _OwnCpu:
    """This process's CPU and its reaped children's, which the test sets."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


@pytest.fixture
def own_cpu(monkeypatch: pytest.MonkeyPatch) -> _OwnCpu:
    """The CPU that every watch reads for this process and its reaped children."""
    counter = _OwnCpu()
    monkeypatch.setattr(memory_watch, "_own_cpu_seconds", counter)
    return counter


@pytest.mark.usefixtures("own_cpu")
def test_begin_walks_the_tree_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_psutil(monkeypatch, _Process(GIB, [_Process(GIB)]))

    MemoryWatch().begin()

    assert _Process.walks == 1


def test_the_cpu_before_begin_and_right_after_it_is_zero(
    monkeypatch: pytest.MonkeyPatch, own_cpu: _OwnCpu
) -> None:
    child = _Process(GIB)
    child.cpu = 5.0
    _install_psutil(monkeypatch, _Process(GIB, [child]))
    own_cpu.seconds = 3.0
    watch = MemoryWatch()

    assert watch.cpu_seconds() == 0.0
    watch.begin()
    assert watch.cpu_seconds() == 0.0

    own_cpu.seconds = 4.0
    child.cpu = 6.5
    assert watch.cpu_seconds() == pytest.approx(2.5)


@pytest.mark.usefixtures("own_cpu")
def test_a_member_counts_its_cpu_from_its_first_sighting(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    watch = _begun(monkeypatch, root)
    member = _Process(GIB)
    member.cpu = 5.0
    root.child_processes = [member]
    clock.advance()

    assert watch.cpu_seconds() == 0.0
    member.cpu = 7.0
    assert watch.cpu_seconds() == pytest.approx(2.0)


def _sighted(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, root: _Process, member: _Process
) -> MemoryWatch:
    """A watch begun over *root*, which has since sighted *member* below it."""
    watch = _begun(monkeypatch, root)
    root.child_processes = [member]
    clock.advance()
    assert watch.cpu_seconds() == 0.0
    return watch


def test_a_direct_member_reaped_between_walks_leaves_the_cpu_unchanged(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, own_cpu: _OwnCpu
) -> None:
    root = _Process(GIB)
    member = _Process(GIB)
    member.cpu = 5.0
    watch = _sighted(monkeypatch, clock, root, member)
    member.cpu = 7.0
    assert watch.cpu_seconds() == pytest.approx(2.0)

    root.child_processes = []
    own_cpu.seconds += 7.0
    clock.advance()

    assert watch.cpu_seconds() == pytest.approx(2.0)


def test_a_direct_member_whose_read_finds_it_gone_leaves_the_cpu_unchanged(
    monkeypatch: pytest.MonkeyPatch, own_cpu: _OwnCpu, clock: _Clock
) -> None:
    root = _Process(GIB)
    member = _Process(GIB)
    member.cpu = 5.0
    watch = _sighted(monkeypatch, clock, root, member)
    member.cpu = 7.0
    assert watch.cpu_seconds() == pytest.approx(2.0)

    member.read_error = _NoSuchProcess
    own_cpu.seconds += 7.0

    assert watch.cpu_seconds() == pytest.approx(2.0)
    member.read_error = None
    member.cpu = 9.0
    assert watch.cpu_seconds() == pytest.approx(2.0)


def test_a_direct_member_that_a_sample_finds_gone_leaves_the_cpu_unchanged(
    monkeypatch: pytest.MonkeyPatch, own_cpu: _OwnCpu, clock: _Clock
) -> None:
    root = _Process(GIB)
    member = _Process(GIB)
    member.cpu = 5.0
    watch = _sighted(monkeypatch, clock, root, member)
    member.cpu = 7.0
    assert watch.cpu_seconds() == pytest.approx(2.0)

    member.read_error = _NoSuchProcess
    _ = watch.sample()
    own_cpu.seconds += 7.0

    assert watch.cpu_seconds() == pytest.approx(2.0)


class _ReapedAfterRead(_OwnCpu):
    """Own CPU whose next read is followed by a reap and another thread's sample.

    The sample, which finds the member gone, gets *join_seconds* to finish before
    the read returns. A watch that reads its own CPU under the membership lock
    holds the sample back until the read is done.
    """

    def __init__(self, member: _Process, *, reaped_cpu: float) -> None:
        super().__init__()
        self.member = member
        self.reaped_cpu = reaped_cpu
        self.watch: MemoryWatch | None = None
        self.sampler: threading.Thread | None = None

    def __call__(self) -> float:
        value = self.seconds
        watch, self.watch = self.watch, None
        if watch is not None:
            self.member.read_error = _NoSuchProcess
            self.seconds += self.reaped_cpu
            self.sampler = threading.Thread(target=watch.sample)
            self.sampler.start()
            self.sampler.join(timeout=0.5)
        return value


def test_a_member_reaped_after_the_own_read_and_left_by_another_thread_is_not_lost(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    member = _Process(GIB)
    member.cpu = 5.0
    own = _ReapedAfterRead(member, reaped_cpu=7.0)
    monkeypatch.setattr(memory_watch, "_own_cpu_seconds", own)
    watch = _sighted(monkeypatch, clock, root, member)
    member.cpu = 7.0
    assert watch.cpu_seconds() == pytest.approx(2.0)

    own.watch = watch
    racing = watch.cpu_seconds()
    assert own.sampler is not None
    own.sampler.join()

    assert racing == pytest.approx(2.0)
    assert watch.cpu_seconds() == pytest.approx(2.0)


@pytest.mark.usefixtures("own_cpu")
@pytest.mark.parametrize("error", [_ZombieProcess, _AccessDenied])
def test_a_member_that_cannot_be_read_keeps_its_last_cpu(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, error: type[_Error]
) -> None:
    root = _Process(GIB)
    member = _Process(GIB)
    member.cpu = 5.0
    watch = _sighted(monkeypatch, clock, root, member)
    member.cpu = 7.0
    assert watch.cpu_seconds() == pytest.approx(2.0)

    member.read_error = error
    member.cpu = 9.0

    assert watch.cpu_seconds() == pytest.approx(2.0)
    member.read_error = None
    assert watch.cpu_seconds() == pytest.approx(4.0)


@pytest.mark.usefixtures("own_cpu")
def test_a_read_overtaken_by_a_later_one_never_moves_a_members_cpu_back(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    member = _Process(GIB)
    member.cpu = 5.0
    watch = _sighted(monkeypatch, clock, root, member)
    reads: list[float] = []

    def overtaken(_process: object) -> float:
        """The first read waits while a whole read of a later value goes by."""
        reads.append(1.0)
        if len(reads) > 1:
            times = member.cpu_times()
            return times.user + times.system
        member.cpu = 9.0
        _ = watch.cpu_seconds()
        return 7.0

    monkeypatch.setattr(memory_watch, "_process_cpu_seconds", overtaken)

    _ = watch.cpu_seconds()

    member.read_error = _AccessDenied
    assert watch.cpu_seconds() == pytest.approx(4.0)


@pytest.mark.usefixtures("own_cpu")
def test_a_grandchild_that_leaves_takes_its_share_out(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB)
    grandchild = _Process(GIB)
    grandchild.cpu = 1.0
    worker = _Process(GIB, [grandchild], cmdline=_SPAWNED)
    worker.cpu = 2.0
    watch = _sighted(monkeypatch, clock, root, worker)
    worker.cpu, grandchild.cpu = 3.0, 4.0
    assert watch.cpu_seconds() == pytest.approx(4.0)

    worker.child_processes = []
    clock.advance()

    assert watch.cpu_seconds() == pytest.approx(1.0)


@pytest.mark.usefixtures("own_cpu")
def test_cpu_reads_within_the_interval_walk_nothing(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    root = _Process(GIB, [_Process(GIB)])
    watch = _begun(monkeypatch, root)

    for _ in range(3):
        _ = watch.cpu_seconds()
    assert _Process.walks == 0

    clock.advance()
    for _ in range(3):
        _ = watch.cpu_seconds()
    assert _Process.walks == 1


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
