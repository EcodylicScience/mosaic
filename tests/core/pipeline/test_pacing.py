"""A repeated pass runs as often as it pays for, not once for each entry."""

from __future__ import annotations

from mosaic.core.pipeline.pacing import PacedCollector, Pacer


class _Time:
    """A clock that moves only when a test or a collection moves it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.collections = 0
        self.cost = 0.0

    def __call__(self) -> float:
        return self.now

    def collect(self) -> None:
        self.collections += 1
        self.now += self.cost


def _collector(time: _Time, byte_limit: int = 1000) -> PacedCollector:
    return PacedCollector(byte_limit=byte_limit, clock=time, collect=time.collect)


def test_the_first_entry_collects_and_times_the_collection() -> None:
    time = _Time()
    collector = _collector(time)

    assert collector.finished(10)
    assert time.collections == 1


def test_entries_wait_nine_times_the_cost_of_the_last_collection() -> None:
    time = _Time()
    time.cost = 2.0
    collector = _collector(time)
    assert collector.finished(10)

    time.now += 17.0
    assert not collector.finished(10)
    time.now += 1.0

    assert collector.finished(10)
    assert time.collections == 2


def test_many_cheap_entries_collect_a_small_share_of_the_time() -> None:
    time = _Time()
    time.cost = 1.0
    collector = _collector(time)

    for _ in range(1000):
        time.now += 0.1
        _ = collector.finished(1)

    assert time.collections < 20


def test_a_run_of_large_outputs_collects_once_they_pass_the_limit() -> None:
    time = _Time()
    time.cost = 100.0
    collector = _collector(time, byte_limit=1000)
    assert collector.finished(10)

    assert not collector.finished(400)
    assert not collector.finished(400)
    assert collector.finished(400)
    assert time.collections == 2


def test_a_pass_is_due_after_the_interval_and_after_a_multiple_of_the_last_cost() -> (
    None
):
    time = _Time()
    pacer = Pacer(interval=10.0, cost_factor=4.0, clock=time)

    time.now += 9.0
    assert not pacer.due()
    time.now += 1.0
    assert pacer.due()

    time.cost = 8.0
    pacer.timed(time.collect)
    time.now += 31.0
    assert not pacer.due()
    time.now += 1.0
    assert pacer.due()


def test_a_pass_that_raises_still_counts_what_it_cost() -> None:
    time = _Time()
    pacer = Pacer(interval=1.0, cost_factor=4.0, clock=time)

    def failing() -> None:
        time.now += 5.0
        raise RuntimeError("pass failed")

    try:
        pacer.timed(failing)
    except RuntimeError:
        pass
    time.now += 19.0

    assert not pacer.due()
