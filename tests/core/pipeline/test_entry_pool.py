"""The pool that computes one run's entries, one after another or several at once.

A run hands the pool one call per entry and hears back through two callbacks. The pool
owns the executor, the dispatch limit, the memory floor and the two ways a host ends a
run: a worker it kills and a floor with nothing in flight.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import override

import pytest

from mosaic.core.pipeline import entry_pool, memory_watch
from mosaic.core.pipeline.entry_pool import (
    EntryFailure,
    EntryPool,
    OrderedResults,
    worker_dataset,
)
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.memory_limit import MemoryLimit
from mosaic.core.pipeline.memory_watch import MemoryWatch
from mosaic.core.pipeline.parallelism import (
    MemoryFloorReached,
    ParallelMode,
    ParallelSettings,
    WorkerProcessDied,
)
from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.runlog import ResourceReport
from tests.helpers import (
    CHILD_TIMEOUT_SECONDS,
    CLOCK_TICK_SECONDS,
    count_process_tree_walks,
    process_alive,
)


def double(value: int) -> int:
    return value * 2


def fail_on_three(value: int) -> int:
    if value == 3:
        msg = "three is refused"
        raise ValueError(msg)
    return value


def die(_value: int) -> int:
    os._exit(1)


class _Heard:
    """What a pool's callbacks were told, in order."""

    def __init__(self) -> None:
        self.results: list[tuple[int, int]] = []
        self.errors: list[tuple[int, str]] = []
        self.resources: list[ResourceReport] = []

    def on_result(self, tag: int, result: int) -> None:
        self.results.append((tag, result))

    def on_error(self, tag: int, exc: Exception) -> None:
        self.errors.append((tag, str(exc)))

    def on_resources(self, report: ResourceReport) -> None:
        self.resources.append(report)


def _pool(
    heard: _Heard,
    *,
    workers: int = 1,
    mode: ParallelMode = "thread",
    floor: float = 0.0,
    on_error: Callable[[int, Exception], None] | None = None,
) -> EntryPool[int, int]:
    return EntryPool(
        workers=workers,
        mode=mode,
        min_free_memory_fraction=floor,
        on_result=heard.on_result,
        on_error=on_error or heard.on_error,
        on_resources=heard.on_resources,
    )


def _free_fractions(values: list[float]) -> Callable[[MemoryWatch], float]:
    remaining = list(values)

    def free_fraction(self: MemoryWatch) -> float:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return free_fraction


@pytest.mark.parametrize(
    ("workers", "mode"),
    [(1, "thread"), (3, "thread"), (2, "process")],
    ids=["inline", "threads", "processes"],
)
def test_every_call_is_answered_and_a_failure_does_not_stop_the_rest(
    workers: int, mode: ParallelMode
) -> None:
    heard = _Heard()
    pool = _pool(heard, workers=workers, mode=mode)

    for value in range(1, 6):
        pool.submit(value, fail_on_three, value)
    pool.finish()

    assert sorted(heard.results) == [(1, 1), (2, 2), (4, 4), (5, 5)]
    assert heard.errors == [(3, "three is refused")]


def test_one_worker_computes_each_call_before_the_next_is_taken() -> None:
    heard = _Heard()
    pool = _pool(heard)

    pool.submit(1, double, 1)
    assert heard.results == [(1, 2)]
    pool.submit(2, double, 2)
    assert heard.results == [(1, 2), (2, 4)]
    pool.finish()


def test_no_more_calls_are_in_flight_than_workers() -> None:
    heard = _Heard()
    lock = threading.Lock()
    running = 0
    peak = 0

    def hold(value: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        threading.Event().wait(0.02)
        with lock:
            running -= 1
        return value

    pool = _pool(heard, workers=2)
    for value in range(8):
        pool.submit(value, hold, value)
    pool.finish()

    assert len(heard.results) == 8
    assert peak == 2


def test_a_dead_worker_ends_the_pool_with_the_memory_it_used() -> None:
    heard = _Heard()
    pool = _pool(heard, workers=2, mode="process")

    with pytest.raises(WorkerProcessDied, match="workers=2, mode=process"):
        pool.submit(1, die, 1)
        pool.submit(2, die, 2)
        pool.finish()

    pool.close()
    assert heard.resources[-1].workers == 2


def test_a_pool_that_breaks_while_a_cancel_is_requested_raises_the_cancel() -> None:
    heard = _Heard()
    token = CancelToken()
    token.cancel()
    pool = EntryPool(
        workers=2,
        mode="process",
        on_result=heard.on_result,
        on_error=heard.on_error,
        cancel_check=token.raise_if_cancelled,
    )

    with pytest.raises(Cancelled):
        pool.submit(1, die, 1)
        pool.submit(2, die, 2)
        pool.finish()

    pool.close()


def test_a_cancel_that_arrives_after_the_pool_broke_is_still_the_cancel() -> None:
    heard = _Heard()
    token = CancelToken()
    pool = EntryPool(
        workers=2,
        mode="process",
        on_result=heard.on_result,
        on_error=heard.on_error,
        cancel_check=token.raise_if_cancelled,
    )
    handler = threading.Timer(0.3, token.cancel)
    handler.start()

    with pytest.raises(Cancelled):
        pool.submit(1, die, 1)
        pool.submit(2, die, 2)
        pool.finish()

    handler.join()
    pool.close()


def test_a_broken_pool_with_no_cancel_requested_is_a_worker_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entry_pool, "CANCEL_ARRIVAL_SECONDS", 0.2)
    heard = _Heard()
    pool = EntryPool(
        workers=2,
        mode="process",
        on_result=heard.on_result,
        on_error=heard.on_error,
        cancel_check=CancelToken().raise_if_cancelled,
    )

    with pytest.raises(WorkerProcessDied):
        pool.submit(1, die, 1)
        pool.submit(2, die, 2)
        pool.finish()

    pool.close()


def test_the_floor_holds_dispatch_until_the_calls_in_flight_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heard = _Heard()
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.9, 0.01, 0.9])
    )
    pool = _pool(heard, workers=3, floor=0.5)

    for value in (1, 2, 3):
        pool.submit(value, double, value)
    pool.finish()

    assert sorted(heard.results) == [(1, 2), (2, 4), (3, 6)]


def test_the_floor_ends_the_pool_when_nothing_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heard = _Heard()
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )
    pool = _pool(heard, workers=2, floor=0.5)

    with pytest.raises(MemoryFloorReached, match="fewer workers"):
        pool.submit(1, double, 1)
    pool.close()


def test_the_floor_is_not_checked_with_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heard = _Heard()
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )
    pool = _pool(heard, workers=1, floor=0.5)

    pool.submit(1, double, 1)
    pool.finish()

    assert heard.results == [(1, 2)]


def test_a_caller_that_holds_its_own_entries_back_is_not_held_by_the_floor_at_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heard = _Heard()
    monkeypatch.setattr(
        memory_watch.MemoryWatch, "free_fraction", _free_fractions([0.01])
    )
    pool: EntryPool[int, int] = EntryPool(
        workers=2,
        mode="thread",
        min_free_memory_fraction=0.5,
        on_result=heard.on_result,
        on_error=heard.on_error,
        floor_at_submit=False,
    )

    pool.submit(1, double, 1)
    pool.finish()
    assert heard.results == [(1, 2)]

    with pytest.raises(MemoryFloorReached, match="fewer workers"):
        pool.wait_for_memory()
    pool.close()


@pytest.mark.parametrize("workers", [1, 3], ids=["inline", "threads"])
def test_a_call_is_dispatched_before_it_computes_and_computed_before_its_outcome(
    workers: int,
) -> None:
    heard = _Heard()
    events: list[tuple[str, int]] = []
    pool: EntryPool[int, int] = EntryPool(
        workers=workers,
        mode="thread",
        on_result=lambda tag, result: events.append(("result", tag)),
        on_error=heard.on_error,
        on_dispatch=lambda tag: events.append(("dispatched", tag)),
        on_computed=lambda tag: events.append(("computed", tag)),
    )

    pool.submit(7, double, 7)
    pool.finish()

    kinds = [kind for kind, tag in events if tag == 7]
    assert kinds[0] == "dispatched"
    assert kinds.index("computed") < kinds.index("result")
    assert kinds.count("result") == 1


def test_a_pool_measures_with_the_watch_it_is_given() -> None:
    watch = MemoryWatch()
    heard = _Heard()
    pool: EntryPool[int, int] = EntryPool(
        workers=1,
        mode="thread",
        on_result=heard.on_result,
        on_error=heard.on_error,
        watch=watch,
    )

    assert pool.watch is watch
    assert pool.in_flight == 0


def test_an_error_callback_that_raises_cancels_the_calls_not_yet_started() -> None:
    heard = _Heard()
    started: list[int] = []
    release = threading.Event()

    def gate(value: int) -> int:
        started.append(value)
        if value == 1:
            msg = "first fails"
            raise ValueError(msg)
        release.wait(5)
        return value

    class _Stop(Exception):
        pass

    def stop(tag: int, exc: Exception) -> None:
        raise _Stop

    pool = _pool(heard, workers=2, on_error=stop)
    try:
        with pytest.raises(_Stop):
            pool.submit(1, gate, 1)
            pool.submit(2, gate, 2)
            pool.finish()
    finally:
        release.set()
        pool.close()

    assert 1 in started


def test_the_resources_name_the_workers_and_the_mode_only_when_there_are_several() -> (
    None
):
    inline = _Heard()
    several = _Heard()

    for heard, workers in ((inline, 1), (several, 2)):
        pool = _pool(heard, workers=workers)
        pool.submit(1, double, 1)
        pool.finish()
        pool.close()

    assert (inline.resources[-1].workers, inline.resources[-1].parallel_mode) == (
        1,
        "",
    )
    assert (
        several.resources[-1].workers,
        several.resources[-1].parallel_mode,
    ) == (2, "thread")


def _limit_holding(figure: list[int]) -> Callable[[], MemoryLimit]:
    def read() -> MemoryLimit:
        return MemoryLimit(
            limit_bytes=64 * 1024**3,
            source="cgroup_v2",
            in_use_bytes=1024**3,
            peak_bytes=0,
            host_total_bytes=128 * 1024**3,
            host_available_bytes=64 * 1024**3,
            nonreclaimable_bytes=figure[0],
        )

    return read


def test_an_ops_report_carries_the_watchs_latest_and_peak_non_reclaimable_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figure = [900]
    monkeypatch.setattr(memory_watch, "read_memory_limit", _limit_holding(figure))
    heard = _Heard()
    pool = _pool(heard)

    pool.submit(1, double, 1)
    pool.finish()
    assert pool.watch.sample() is not None
    figure[0] = 500
    pool.close()

    report = heard.resources[-1]
    assert report.nonreclaimable_bytes == 500
    assert report.peak_nonreclaimable_bytes == 900
    assert (
        report.base_nonreclaimable_bytes,
        report.worker_base_nonreclaimable_bytes,
        report.peak_grown_bytes,
        report.peak_in_flight_entries,
        report.peak_in_flight_rows,
    ) == (0, 0, 0, 0, 0)


def test_an_ops_report_of_a_sample_without_the_figure_reads_it_as_not_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_watch, "read_memory_limit", _limit_holding([0]))
    heard = _Heard()
    pool = _pool(heard)

    pool.submit(1, double, 1)
    pool.finish()
    pool.close()

    report = heard.resources[-1]
    assert (report.nonreclaimable_bytes, report.peak_nonreclaimable_bytes) == (0, 0)


def test_a_settled_pool_hands_back_the_calls_in_flight_and_drops_what_a_callback_raises() -> (
    None
):
    heard = _Heard()
    release = threading.Event()

    def held(value: int) -> int:
        release.wait(5)
        return value

    def fail(tag: int, result: int) -> None:
        if tag == 1:
            msg = "the callback refuses"
            raise ValueError(msg)
        heard.on_result(tag, result)

    pool: EntryPool[int, int] = EntryPool(
        workers=3, mode="thread", on_result=fail, on_error=heard.on_error
    )
    for value in (1, 2, 3):
        pool.submit(value, held, value)
    release.set()

    pool.settle()
    pool.close()

    assert sorted(heard.results) == [(2, 2), (3, 3)]


def test_items_are_emitted_in_index_order_as_the_gaps_fill() -> None:
    emitted: list[str] = []
    ordered = OrderedResults[str](emitted.append)

    ordered.put(2, "c")
    ordered.put(1, "b")
    assert (emitted, ordered.held()) == ([], ["b", "c"])
    ordered.put(0, "a")
    ordered.put(4, "e")

    assert (emitted, ordered.held()) == (["a", "b", "c"], ["e"])


def test_an_item_whose_emission_raised_is_not_emitted_again() -> None:
    emitted: list[str] = []

    def emit(item: str) -> None:
        emitted.append(item)
        if item == "a":
            raise ValueError

    ordered = OrderedResults[str](emit)

    with pytest.raises(ValueError):
        ordered.put(0, "a")
    ordered.put(1, "b")

    assert emitted == ["a", "b"]


def _captured() -> EntryFailure:
    try:
        _ = fail_on_three(3)
    except ValueError as exc:
        return EntryFailure.capture("key", exc)
    msg = "fail_on_three(3) did not raise"
    raise AssertionError(msg)


def _record_text(failure: EntryFailure) -> str:
    seen: list[str] = []

    def record(_key: str, exc: BaseException) -> None:
        seen.append("".join(traceback.format_exception(exc)))

    failure.record(record)
    return seen[0]


def test_a_failure_is_recorded_from_inside_an_except_block_with_its_traceback() -> None:
    text = _record_text(_captured())

    assert "fail_on_three" in text
    assert "Raised in a worker" not in text


def test_a_failure_that_crossed_a_process_carries_the_workers_traceback() -> None:
    crossed = pickle.loads(pickle.dumps(_captured()))

    text = _record_text(crossed)

    assert "Raised in a worker" in text
    assert "fail_on_three" in text


class _Unpicklable(Exception):
    def __init__(self, first: str, second: str) -> None:
        super().__init__(first)
        self.second = second


def test_a_failure_whose_error_cannot_be_pickled_keeps_its_type_and_message() -> None:
    try:
        raise _Unpicklable("one", "two")
    except _Unpicklable as exc:
        failure = EntryFailure.capture("key", exc)

    crossed = pickle.loads(pickle.dumps(failure))

    assert str(crossed.error) == "_Unpicklable: one"


def _unpicklable_failure(*, portable: bool) -> EntryFailure:
    try:
        raise _Unpicklable("one", "two")
    except _Unpicklable as exc:
        return EntryFailure.capture("key", exc, portable=portable)


def test_a_failure_captured_as_raised_keeps_an_error_that_does_not_pickle() -> None:
    kept = _unpicklable_failure(portable=False)

    assert isinstance(kept.error, _Unpicklable)
    assert kept.error.second == "two"


def test_a_portable_failure_replaces_an_error_that_does_not_pickle() -> None:
    kept = _unpicklable_failure(portable=False)

    portable = kept.portable()
    crossed = pickle.loads(pickle.dumps(portable))

    assert type(portable.error) is RuntimeError
    assert str(crossed.error) == "_Unpicklable: one"
    assert (portable.key, portable.trace) == (kept.key, kept.trace)


def test_a_default_capture_is_the_portable_capture() -> None:
    default = _unpicklable_failure(portable=True)

    assert type(default.error) is RuntimeError
    assert str(default.error) == "_Unpicklable: one"
    picklable = _captured()
    assert picklable.portable() is picklable


def _reraised_text(failure: EntryFailure) -> str:
    with pytest.raises(ValueError) as raised:
        failure.reraise()
    return "".join(traceback.format_exception(raised.value))


def test_a_local_failure_reraises_with_its_traceback_and_no_note() -> None:
    text = _reraised_text(_captured())

    assert "fail_on_three" in text
    assert "Raised in a worker" not in text


def test_a_failure_that_crossed_a_process_reraises_with_the_workers_traceback() -> None:
    crossed = pickle.loads(pickle.dumps(_captured()))

    text = _reraised_text(crossed)

    assert "Raised in a worker" in text
    assert "fail_on_three" in text


def test_a_worker_process_opens_each_dataset_once(tmp_path: Path) -> None:
    from tests.helpers import make_dataset

    ds = make_dataset(tmp_path / "ds")
    ref = str(ds.manifest_path)

    assert worker_dataset(ds) is ds
    assert worker_dataset(ref) is worker_dataset(ref)
    assert worker_dataset(ref).manifest_path == ds.manifest_path


def wait_for_file(path: str) -> int:
    while not Path(path).exists():
        time.sleep(0.01)
    return 1


@pytest.fixture
def walks(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], tuple[MemoryWatch, list[int]]]:
    """Begin a watch, freeze its membership's clock, and count the walks after begin."""
    monkeypatch.setattr(memory_watch, "now", lambda: 0.0)

    def begun() -> tuple[MemoryWatch, list[int]]:
        watch = MemoryWatch()
        watch.begin()
        return watch, count_process_tree_walks(monkeypatch)

    return begun


def _process_pool(
    watch: MemoryWatch, *, workers: int = 2, mode: ParallelMode = "process"
) -> EntryPool[int, int]:
    heard = _Heard()
    return EntryPool(
        workers=workers,
        mode=mode,
        on_result=heard.on_result,
        on_error=heard.on_error,
        watch=watch,
    )


def test_a_process_pool_sees_each_worker_at_the_submission_that_starts_it(
    walks: Callable[[], tuple[MemoryWatch, list[int]]], tmp_path: Path
) -> None:
    release = tmp_path / "release"
    watch, counted = walks()
    pool = _process_pool(watch)
    try:
        pool.submit(1, wait_for_file, str(release))
        assert len(counted) == 1
        first = watch.sample()
        pool.submit(2, wait_for_file, str(release))
        assert len(counted) == 2
        second = watch.sample()
        assert len(counted) == 2
        assert first is not None and second is not None
        assert (first.spawned_workers, second.spawned_workers) == (1, 2)
        assert pool.in_flight == 2
        release.touch()
        for tag in range(3, 7):
            pool.submit(tag, wait_for_file, str(release))
        pool.finish()
    finally:
        release.touch()
        pool.close()

    assert len(counted) == 2


def test_a_process_pool_fed_one_call_at_a_time_walks_once(
    walks: Callable[[], tuple[MemoryWatch, list[int]]],
) -> None:
    watch, counted = walks()
    pool = _process_pool(watch)
    try:
        for tag in range(4):
            pool.submit(tag, double, tag)
            pool.drain_one()
        pool.finish()
    finally:
        pool.close()

    assert len(counted) == 1


@pytest.mark.parametrize(
    ("workers", "mode"), [(2, "thread"), (1, "process")], ids=["threads", "one-worker"]
)
def test_a_pool_without_process_workers_walks_nothing_on_submission(
    walks: Callable[[], tuple[MemoryWatch, list[int]]], workers: int, mode: ParallelMode
) -> None:
    watch, counted = walks()
    pool = _process_pool(watch, workers=workers, mode=mode)
    for tag in range(4):
        pool.submit(tag, double, tag)
    pool.finish()

    assert len(counted) == 0


def _reaped(pid: int) -> bool:
    """Whether the process is gone, which a zombie that nobody has joined is not."""
    import psutil

    return not psutil.pid_exists(pid)


def record_pid_then_sleep_until_the_other_worker_ends(
    pid_file: str, other_pid_file: str, seconds: float
) -> int:
    Path(pid_file).write_text(str(os.getpid()))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        other = Path(other_pid_file)
        if other.exists() and other.read_text().isdigit():
            if not process_alive(int(other.read_text())):
                break
        time.sleep(0.05)
    return 1


def record_pid_then_kill_the_worker(pid_file: str, seconds: float) -> int:
    Path(pid_file).write_text(str(os.getpid()))
    time.sleep(seconds)
    os._exit(1)


ENDED_DEADLINE_SECONDS = 30.0

SLEEPER_SECONDS = 20.0
"""The longest the sleeping call waits for the dying worker's pid to end.

The call returns as soon as that pid has ended. A pool that never noticed the death
would otherwise hold the test for as long as the call sleeps.
"""


def test_the_cpu_figure_does_not_fall_across_a_worker_that_dies_and_is_reaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [0.0]
    monkeypatch.setattr(memory_watch, "now", lambda: clock[0])
    watch = MemoryWatch()
    watch.begin()
    pool = _process_pool(watch)
    pid_files = [tmp_path / "sleeper", tmp_path / "killed"]
    try:
        pool.submit(
            1,
            record_pid_then_sleep_until_the_other_worker_ends,
            str(pid_files[0]),
            str(pid_files[1]),
            SLEEPER_SECONDS,
        )
        pool.submit(2, record_pid_then_kill_the_worker, str(pid_files[1]), 1.5)
        deadline = time.monotonic() + ENDED_DEADLINE_SECONDS
        while not all(path.exists() for path in pid_files):
            assert time.monotonic() < deadline, "a worker never began its call"
            time.sleep(0.05)
        before = watch.cpu_seconds()

        with pytest.raises(WorkerProcessDied):
            pool.finish()

        pids = [int(path.read_text()) for path in pid_files]
        deadline = time.monotonic() + ENDED_DEADLINE_SECONDS
        while not all(_reaped(pid) for pid in pids):
            assert time.monotonic() < deadline, "a worker was not reaped by the pool"
            time.sleep(0.05)
        clock[0] += memory_watch.ESTIMATE_SAMPLE_INTERVAL_SECONDS + 1.0
        sample = watch.sample()
        after = watch.cpu_seconds()
    finally:
        pool.close()

    assert sample is not None
    assert sample.spawned_workers == 0
    assert after >= before - CLOCK_TICK_SECONDS


class _HashedFuture(Future[int]):
    """A future whose hash is fixed and whose equality is its identity.

    CPython iterates a small set of small integer hashes in ascending order, so the
    ``done`` set that ``wait`` returns hands these futures out in hash order.
    """

    def __init__(self, order: int) -> None:
        super().__init__()
        self.order: int = order

    @override
    def __hash__(self) -> int:
        return self.order

    @override
    def __eq__(self, other: object) -> bool:
        return self is other


class _HandingExecutor:
    """An executor whose submissions return the futures a test resolves, in order."""

    def __init__(self, futures: list[_HashedFuture]) -> None:
        self._futures: list[_HashedFuture] = list(futures)

    def submit(self, fn: Callable[[int], int], *args: int) -> Future[int]:
        return self._futures.pop(0)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None


def _hand_out(monkeypatch: pytest.MonkeyPatch, futures: list[_HashedFuture]) -> None:
    executor = _HandingExecutor(futures)

    def make(max_workers: int) -> _HandingExecutor:
        return executor

    monkeypatch.setattr(entry_pool, "ThreadPoolExecutor", make)


class _Events:
    """The results and errors a pool handed back, in the order it handed them."""

    def __init__(self) -> None:
        self.heard: list[tuple[str, int]] = []

    def on_result(self, tag: int, result: int) -> None:
        self.heard.append(("result", tag))

    def on_error(self, tag: int, exc: Exception) -> None:
        self.heard.append(("error", tag))


def _broken_set() -> list[_HashedFuture]:
    """A broken call that hashes first, two calls that returned and one that raised."""
    broken, first, failed, second = (_HashedFuture(order) for order in range(4))
    broken.set_exception(BrokenProcessPool("a worker ended"))
    first.set_result(10)
    failed.set_exception(ValueError("refused"))
    second.set_result(30)
    return [broken, first, failed, second]


def _cancelled(order: int) -> _HashedFuture:
    future = _HashedFuture(order)
    _ = future.cancel()
    _ = future.set_running_or_notify_cancel()
    return future


def _handing_pool(
    events: _Events,
    *,
    on_error: Callable[[int, Exception], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> EntryPool[int, int]:
    return EntryPool(
        workers=8,
        mode="thread",
        on_result=events.on_result,
        on_error=on_error or events.on_error,
        cancel_check=cancel_check,
        hand_back_on_break=True,
    )


def _submit_all(pool: EntryPool[int, int], count: int) -> None:
    for tag in range(count):
        pool.submit(tag, double, tag)


def test_a_pool_that_hands_back_on_a_break_hears_the_finished_calls_results_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = _HashedFuture(4)
    interrupted.set_exception(KeyboardInterrupt())
    _hand_out(monkeypatch, [*_broken_set(), interrupted])
    events = _Events()
    pool = _handing_pool(events)
    _submit_all(pool, 5)

    with pytest.raises(WorkerProcessDied):
        pool.finish()
    pool.close()

    assert events.heard == [("result", 1), ("result", 3), ("error", 2)]
    assert pool.in_flight == 0


def test_the_hand_back_on_a_break_skips_a_cancelled_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_out(monkeypatch, [*_broken_set(), _cancelled(4)])
    events = _Events()
    pool = _handing_pool(events)
    _submit_all(pool, 5)

    with pytest.raises(WorkerProcessDied):
        pool.finish()
    pool.close()

    assert events.heard == [("result", 1), ("result", 3), ("error", 2)]


def test_a_cancel_during_the_hand_back_keeps_the_results_and_records_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_out(monkeypatch, _broken_set())
    events = _Events()
    token = CancelToken()
    token.cancel()

    def failed_unless_cancelled(tag: int, exc: Exception) -> None:
        if token.is_cancelled():
            raise Cancelled from exc
        events.on_error(tag, exc)

    pool = _handing_pool(
        events,
        on_error=failed_unless_cancelled,
        cancel_check=token.raise_if_cancelled,
    )
    _submit_all(pool, 4)

    with pytest.raises(Cancelled):
        pool.finish()
    pool.close()

    assert events.heard == [("result", 1), ("result", 3)]


def test_an_unavailable_tool_raised_in_the_hand_back_follows_the_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_out(monkeypatch, _broken_set())
    events = _Events()

    def unavailable(tag: int, exc: Exception) -> None:
        raise ToolUnavailable("fake-tool", "environment", "fake-tool is not set up")

    pool = _handing_pool(events, on_error=unavailable)
    _submit_all(pool, 4)

    with pytest.raises(ToolUnavailable):
        pool.finish()
    pool.close()

    assert events.heard == [("result", 1), ("result", 3)]


def test_a_pool_built_for_settings_hands_back_nothing_on_a_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_out(monkeypatch, _broken_set())
    events = _Events()
    pool = EntryPool[int, int].for_settings(
        ParallelSettings(workers=8),
        on_result=events.on_result,
        on_error=events.on_error,
    )
    _submit_all(pool, 4)

    with pytest.raises(WorkerProcessDied):
        pool.finish()
    pool.close()

    assert events.heard == []


def test_a_submission_to_a_broken_pool_hands_back_the_call_that_returned() -> None:
    computed = {tag: threading.Event() for tag in (1, 2)}
    heard = _Heard()

    def stamp(tag: int) -> None:
        if tag in computed:
            computed[tag].set()

    pool: EntryPool[int, int] = EntryPool(
        workers=3,
        mode="process",
        on_result=heard.on_result,
        on_error=heard.on_error,
        on_computed=stamp,
        hand_back_on_break=True,
    )
    try:
        pool.submit(1, double, 1)
        assert computed[1].wait(CHILD_TIMEOUT_SECONDS)
        pool.submit(2, die, 2)
        assert computed[2].wait(CHILD_TIMEOUT_SECONDS)

        with pytest.raises(WorkerProcessDied):
            pool.submit(3, double, 3)
    finally:
        pool.close()

    assert heard.results == [(1, 2)]


def test_a_results_only_settle_hears_the_results_and_returns_what_their_callback_raised() -> (
    None
):
    release = threading.Event()
    errors: list[int] = []

    def held(value: int) -> int:
        _ = release.wait(5)
        if value == 2:
            msg = "two is refused"
            raise ValueError(msg)
        return value

    refused = ValueError("the callback refuses")
    heard: list[tuple[int, int]] = []

    def on_result(tag: int, result: int) -> None:
        if tag == 3:
            raise refused
        heard.append((tag, result))

    pool: EntryPool[int, int] = EntryPool(
        workers=3,
        mode="thread",
        on_result=on_result,
        on_error=lambda tag, exc: errors.append(tag),
    )
    for value in (1, 2, 3):
        pool.submit(value, held, value)
    release.set()

    dropped = pool.settle(results_only=True)
    pool.close()

    assert dropped == [refused]
    assert heard == [(1, 1)]
    assert errors == []
    assert pool.in_flight == 0


def test_a_results_only_settle_skips_a_call_cancelled_by_an_earlier_error_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed, waiting, returned = (_HashedFuture(order) for order in range(3))
    failed.set_exception(ValueError("refused"))
    returned.set_result(20)
    _hand_out(monkeypatch, [failed, waiting, returned])
    events = _Events()

    class _Stop(Exception):
        pass

    def stop(tag: int, exc: Exception) -> None:
        raise _Stop

    pool = _handing_pool(events, on_error=stop)
    _submit_all(pool, 3)
    with pytest.raises(_Stop):
        pool.drain_one()
    assert waiting.cancelled()

    dropped = pool.settle(results_only=True)
    pool.close()

    assert dropped == []
    assert events.heard == [("result", 2)]
    assert pool.in_flight == 0


def interrupt(_value: int) -> int:
    raise KeyboardInterrupt


def test_a_results_only_settle_skips_a_worker_interrupted_by_a_signal() -> None:
    heard = _Heard()
    pool = _pool(heard, workers=2, mode="process")
    try:
        pool.submit(1, interrupt, 1)
        pool.submit(2, double, 2)
        try:
            dropped = pool.settle(results_only=True)
        except KeyboardInterrupt:
            pytest.fail("the settle raised the worker's KeyboardInterrupt")
    finally:
        pool.close()

    assert dropped == []
    assert heard.results == [(2, 4)]
    assert heard.errors == []


def die_once_file_exists(path: str) -> int:
    while not Path(path).exists():
        time.sleep(0.01)
    os._exit(1)


def test_a_results_only_settle_of_a_pool_that_breaks_returns_with_the_results(
    tmp_path: Path,
) -> None:
    computed = tmp_path / "computed-1"
    heard = _Heard()

    def stamp(tag: int) -> None:
        if tag == 1:
            computed.touch()

    pool: EntryPool[int, int] = EntryPool(
        workers=2,
        mode="process",
        on_result=heard.on_result,
        on_error=heard.on_error,
        on_computed=stamp,
    )
    try:
        pool.submit(2, die_once_file_exists, str(computed))
        pool.submit(1, double, 1)
        started = time.monotonic()
        dropped = pool.settle(results_only=True)
        elapsed = time.monotonic() - started
    finally:
        pool.close()

    assert dropped == []
    assert heard.results == [(1, 2)]
    assert heard.errors == []
    assert elapsed < CHILD_TIMEOUT_SECONDS


class _BreaksAfterAnEarlierResult(_HashedFuture):
    """A broken future that, once armed, sets an earlier result when asked if done.

    It stands for the executor that sets a call's result between the settle's
    look at that call and its look at this one, and then breaks.
    """

    def __init__(self, order: int) -> None:
        super().__init__(order)
        self._earlier: _HashedFuture | None = None
        self._earlier_result: int = 0
        self.set_exception(BrokenProcessPool("a worker ended"))

    def arm(self, earlier: _HashedFuture, result: int) -> None:
        self._earlier = earlier
        self._earlier_result = result

    @override
    def done(self) -> bool:
        earlier = self._earlier
        if earlier is not None:
            self._earlier = None
            earlier.set_result(self._earlier_result)
        return super().done()


def test_a_results_only_settle_hears_a_result_set_while_it_found_the_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    earlier, never = _HashedFuture(0), _HashedFuture(2)
    broken = _BreaksAfterAnEarlierResult(1)
    _hand_out(monkeypatch, [earlier, broken, never])
    events = _Events()
    pool = _handing_pool(events)
    _submit_all(pool, 3)
    broken.arm(earlier, 10)

    dropped = pool.settle(results_only=True)
    pool.close()

    assert dropped == []
    assert events.heard == [("result", 0)]
    assert pool.in_flight == 1
