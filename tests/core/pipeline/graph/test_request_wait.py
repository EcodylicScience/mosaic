"""Waiting for several requests to end, past the cap of one status read.

These tests inject the clock and the sleep, and end a step from inside the
sleep, so none of them waits in real time.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path

from mosaic.core.pipeline.graph import (
    MAX_WAIT_SECONDS,
    Request,
    wait_for_requests,
)
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import exited_pid


def _log(base: Path, execution_id: str) -> JsonlRunLog:
    return JsonlRunLog(run_log_path(base, execution_id), execution_id)


def _request(request_id: str, **steps: str) -> Request:
    return Request(request_id=request_id, recipe_digest="d" * 64, step_executions=steps)


def _start(base: Path, execution_id: str) -> None:
    with _log(base, execution_id) as log:
        log.started(kind="feature", target=execution_id)


def _finish(base: Path, execution_id: str) -> None:
    with _log(base, execution_id) as log:
        log.finished()


def _fail(base: Path, execution_id: str) -> None:
    with _log(base, execution_id) as log:
        log.failed('{"type": "ValueError", "message": "bad"}')


class _Clock:
    """A clock that only the sleep advances, and a hook run at each total slept."""

    def __init__(self) -> None:
        self.now: float = 0.0
        self.at: dict[float, Callable[[], None]] = {}

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        for when in sorted(self.at):
            if when <= self.now:
                self.at.pop(when)()


def test_a_wait_for_requests_already_ended_reads_once(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    _finish(tmp_path, "E1")
    clock = _Clock()

    waited = wait_for_requests(
        tmp_path,
        [_request("R1", speed="E1")],
        until="all",
        timeout=None,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.now == 0
    assert waited.outcome == "finished"
    assert [(report.request_id, report.verdict) for report in waited.requests] == [
        ("R1", "finished")
    ]


def test_a_wait_runs_past_the_cap_of_one_read(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()
    clock.at[3 * MAX_WAIT_SECONDS + 10] = lambda: _finish(tmp_path, "E1")

    waited = wait_for_requests(
        tmp_path,
        [_request("R1", speed="E1")],
        until="all",
        timeout=None,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert waited.outcome == "finished"
    assert 3 * MAX_WAIT_SECONDS + 10 <= clock.now <= 3 * MAX_WAIT_SECONDS + 11


def test_any_ends_with_the_first_request_to_end(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    _start(tmp_path, "E2")
    clock = _Clock()
    clock.at[100] = lambda: _fail(tmp_path, "E2")

    waited = wait_for_requests(
        tmp_path,
        [_request("R1", speed="E1"), _request("R2", speed="E2")],
        until="any",
        timeout=None,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert waited.outcome == "failed"
    assert [(report.request_id, report.verdict) for report in waited.requests] == [
        ("R1", "running"),
        ("R2", "failed"),
    ]
    assert clock.now <= 100 + MAX_WAIT_SECONDS / 2


def test_all_waits_for_every_request_and_fails_on_one_failure(tmp_path: Path) -> None:
    _start(tmp_path, "E1")
    _start(tmp_path, "E2")
    clock = _Clock()
    clock.at[30] = lambda: _fail(tmp_path, "E1")
    clock.at[200] = lambda: _finish(tmp_path, "E2")

    waited = wait_for_requests(
        tmp_path,
        [_request("R1", speed="E1"), _request("R2", speed="E2")],
        until="all",
        timeout=None,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert waited.outcome == "failed"
    assert [report.verdict for report in waited.requests] == ["failed", "finished"]
    assert clock.now >= 200


def test_a_wait_that_runs_out_reports_the_requests_as_they_stand(
    tmp_path: Path,
) -> None:
    _start(tmp_path, "E1")
    clock = _Clock()

    waited = wait_for_requests(
        tmp_path,
        [_request("R1", speed="E1")],
        until="all",
        timeout=100,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert waited.outcome == "timed_out"
    assert clock.now == 100
    assert waited.requests[0].verdict == "running"


def test_an_orphaned_request_ends_the_wait(tmp_path: Path) -> None:
    """A request that nothing drives ends the wait, though no run-log ended it."""
    with _log(tmp_path, "E1") as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=exited_pid()
        )
    clock = _Clock()

    waited = wait_for_requests(
        tmp_path,
        [_request("R1", speed="E1")],
        until="all",
        timeout=10,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert waited.requests[0].verdict == "orphaned"
    assert waited.outcome == "failed"
