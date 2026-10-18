"""Waiting for a request to change, and the brief report.

A wait looks at the run-logs on a short interval until a step of the request
ends, the request does, or the wait runs out. These tests inject the clock and the
sleep, and end a step from inside the sleep, so none of them waits in real time.
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable
from pathlib import Path

from mosaic.core.pipeline.graph import (
    MAX_WAIT_SECONDS,
    WAIT_POLL_SECONDS,
    CleanupSpec,
    Request,
    request_status_wire,
)
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import SLEEPER_COMMAND, exited_pid


def _log(base: Path, execution_id: str) -> JsonlRunLog:
    return JsonlRunLog(run_log_path(base, execution_id), execution_id)


def _request(**steps: str) -> Request:
    return Request(request_id="R1", recipe_digest="d" * 64, step_executions=steps)


def _start(base: Path, execution_id: str) -> None:
    with _log(base, execution_id) as log:
        log.started(kind="feature", target=execution_id)


def _finish(base: Path, execution_id: str, *, lost: int = 0) -> None:
    with _log(base, execution_id) as log:
        for index in range(lost):
            log.entry_failed(f"g__s{index}", "{}")
        log.finished()


def _fail(base: Path, execution_id: str) -> None:
    with _log(base, execution_id) as log:
        log.started(kind="feature", target=execution_id)
        log.failed('{"type": "ValueError", "message": "bad"}')


class _Clock:
    """A clock that only the sleep advances, and a hook that runs in each sleep."""

    def __init__(self) -> None:
        self.now: float = 0.0
        self.slept: list[float] = []
        self.on_sleep: Callable[[int], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(len(self.slept))


def test_a_read_without_a_wait_does_not_sleep(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()
    request = _request(speed="E1", templates="E2")

    waited = request_status_wire(
        tmp_path, request, sleep=clock.sleep, monotonic=clock.monotonic
    )

    assert clock.slept == []
    assert waited == request_status_wire(tmp_path, request)


def test_a_wait_returns_as_soon_as_a_step_ends(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()
    clock.on_sleep = lambda count: _finish(tmp_path, "E1") if count == 3 else None

    report = request_status_wire(
        tmp_path,
        _request(speed="E1", templates="E2"),
        wait_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.slept == [WAIT_POLL_SECONDS] * 3
    assert (report.verdict, report.terminal) == ("running", False)
    assert [(step.step, step.outcome) for step in report.steps] == [
        ("speed", "finished"),
        ("templates", "pending"),
    ]


def test_a_wait_returns_when_the_request_ends(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()
    clock.on_sleep = lambda _count: _finish(tmp_path, "E1")

    report = request_status_wire(
        tmp_path,
        _request(speed="E1"),
        wait_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert len(clock.slept) == 1
    assert (report.verdict, report.terminal) == ("finished", True)
    assert report.poll_after_seconds is None


def test_a_wait_with_nothing_to_report_ends_at_the_time_asked(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()

    report = request_status_wire(
        tmp_path,
        _request(speed="E1", templates="E2"),
        wait_seconds=7.25,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert sum(clock.slept) == 7.25
    assert max(clock.slept) <= WAIT_POLL_SECONDS
    assert report.verdict == "running"
    assert report.steps[0].outcome == "running"


def test_a_wait_is_capped(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()

    _ = request_status_wire(
        tmp_path,
        _request(speed="E1"),
        wait_seconds=10 * MAX_WAIT_SECONDS,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert sum(clock.slept) == MAX_WAIT_SECONDS


def test_a_step_that_ended_before_the_call_does_not_end_the_wait(
    tmp_path: Path,
) -> None:
    _finish(tmp_path, "E1")
    _start(tmp_path, "E2")
    clock = _Clock()

    report = request_status_wire(
        tmp_path,
        _request(speed="E1", templates="E2"),
        wait_seconds=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert sum(clock.slept) == 5
    assert [step.outcome for step in report.steps] == ["finished", "running"]


def test_a_request_that_has_ended_is_returned_without_waiting(tmp_path: Path) -> None:
    _finish(tmp_path, "E1")
    clock = _Clock()

    report = request_status_wire(
        tmp_path,
        _request(speed="E1"),
        wait_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.slept == []
    assert report.terminal


def test_a_wait_returns_when_the_cleanup_of_a_request_whose_steps_ended_ends(
    tmp_path: Path,
) -> None:
    """No step ends in the wait, so only the request ending can end it."""
    _finish(tmp_path, "E1")
    with _log(tmp_path, "C1") as log:
        log.started(kind="feature-cleanup", target="cleanup")
    request = Request(
        request_id="R1",
        recipe_digest="d" * 64,
        step_executions={"speed": "E1"},
        cleanup=CleanupSpec(execution_id="C1"),
    )
    clock = _Clock()

    def end_the_cleanup(_count: int) -> None:
        with _log(tmp_path, "C1") as log:
            log.finished()

    clock.on_sleep = end_the_cleanup

    report = request_status_wire(
        tmp_path,
        request,
        wait_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert len(clock.slept) == 1
    assert (report.verdict, report.terminal) == ("finished", True)


def test_an_orphaned_request_is_returned_without_waiting(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=exited_pid()
        )
    clock = _Clock()

    report = request_status_wire(
        tmp_path,
        _request(speed="E1", templates="E2"),
        wait_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.slept == []
    assert report.verdict == "orphaned"


def test_a_wait_returns_when_a_request_turns_orphaned(tmp_path: Path) -> None:
    step = subprocess.Popen([*SLEEPER_COMMAND, "600"])
    try:
        with _log(tmp_path, "E1") as log:
            log.started(
                kind="feature", target="speed", host=socket.gethostname(), pid=step.pid
            )
        clock = _Clock()

        def lose_the_process(count: int) -> None:
            if count == 2:
                step.kill()
                _ = step.wait()

        clock.on_sleep = lose_the_process

        report = request_status_wire(
            tmp_path,
            _request(speed="E1", templates="E2"),
            wait_seconds=30,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    finally:
        step.kill()
        _ = step.wait()

    assert len(clock.slept) == 2
    assert report.verdict == "orphaned"


def _five_steps(base: Path) -> Request:
    """One step in each state: finished, partial, failed, running and not started."""
    _finish(base, "E1")
    _finish(base, "E2", lost=2)
    _fail(base, "E3")
    _start(base, "E4")
    return _request(clean="E1", lossy="E2", broken="E3", moving="E4", waiting="E5")


def test_a_brief_report_lists_the_steps_that_need_attention_and_counts_the_rest(
    tmp_path: Path,
) -> None:
    request = _five_steps(tmp_path)

    brief = request_status_wire(tmp_path, request, brief=True)
    full = request_status_wire(tmp_path, request)

    assert [(step.step, step.outcome) for step in brief.steps] == [
        ("lossy", "partial"),
        ("broken", "failed"),
        ("moving", "running"),
    ]
    assert (brief.omitted_finished, brief.omitted_pending) == (1, 1)
    assert [step.step for step in full.steps] == [
        "clean",
        "lossy",
        "broken",
        "moving",
        "waiting",
    ]
    assert (full.omitted_finished, full.omitted_pending) == (0, 0)
    # What is listed is the full report's own entry for that step.
    assert list(brief.steps) == [full.steps[1], full.steps[2], full.steps[3]]
    assert brief.model_dump(
        exclude={"steps", "omitted_finished", "omitted_pending"}
    ) == (full.model_dump(exclude={"steps", "omitted_finished", "omitted_pending"}))


def test_a_brief_wait_lists_the_step_that_ended_in_it_though_it_finished_cleanly(
    tmp_path: Path,
) -> None:
    request = _five_steps(tmp_path)
    clock = _Clock()
    clock.on_sleep = lambda _count: _finish(tmp_path, "E4")

    report = request_status_wire(
        tmp_path,
        request,
        brief=True,
        wait_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert [(step.step, step.outcome) for step in report.steps] == [
        ("lossy", "partial"),
        ("broken", "failed"),
        ("moving", "finished"),
    ]
    assert (report.omitted_finished, report.omitted_pending) == (1, 1)


def test_a_brief_report_of_a_request_that_finished_lists_no_step(
    tmp_path: Path,
) -> None:
    _finish(tmp_path, "E1")
    _finish(tmp_path, "E2")

    report = request_status_wire(
        tmp_path, _request(speed="E1", templates="E2"), brief=True
    )

    assert (report.verdict, report.terminal) == ("finished", True)
    assert report.steps == ()
    assert (report.omitted_finished, report.omitted_pending) == (2, 0)
