"""The pool that computes one run's entries, one after another or several at once.

A run hands the pool one call per entry and hears back through two callbacks. The pool
owns the executor, the dispatch limit, the memory floor and the two ways a host ends a
run: a worker it kills and a floor with nothing in flight.
"""

from __future__ import annotations

import os
import pickle
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.pipeline import memory_watch
from mosaic.core.pipeline.entry_pool import (
    EntryFailure,
    EntryPool,
    OrderedResults,
    worker_dataset,
)
from mosaic.core.pipeline.memory_watch import MemoryWatch
from mosaic.core.pipeline.parallelism import (
    MemoryFloorReached,
    ParallelMode,
    WorkerProcessDied,
)
from mosaic.runlog import ResourceReport


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


def test_a_worker_process_opens_each_dataset_once(tmp_path: Path) -> None:
    from tests.helpers import make_dataset

    ds = make_dataset(tmp_path / "ds")
    ref = str(ds.manifest_path)

    assert worker_dataset(ds) is ds
    assert worker_dataset(ref) is worker_dataset(ref)
    assert worker_dataset(ref).manifest_path == ds.manifest_path
