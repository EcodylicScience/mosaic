"""Run and request status reports, built from run-logs alone."""

from __future__ import annotations

import datetime
import json
import os
import socket
from collections.abc import Mapping
from pathlib import Path

from mosaic.core.pipeline.graph import (
    FAILED_ENTRY_SAMPLE,
    POLL_AFTER_MAX_SECONDS,
    POLL_AFTER_MIN_SECONDS,
    Request,
    RequestRollup,
    RequestStatus,
    RunStatusWire,
    StepAttempt,
    poll_after_seconds,
    recorded_error,
    request_status_wire,
    request_summary,
    run_status_wire,
)
from mosaic.core.pipeline.refusal import Refusal
from mosaic.runlog import JsonlRunLog, RunLogSnapshot, run_log_path
from tests.helpers import exited_pid, write_driver_lease

OTHER_HOST = "another-host.invalid"

NOW = datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.UTC)


def _log(base: Path, execution_id: str) -> JsonlRunLog:
    return JsonlRunLog(run_log_path(base, execution_id), execution_id)


def _crash(message: str) -> str:
    return json.dumps({"type": "ValueError", "message": message, "traceback": "..."})


def _request(**steps: str) -> Request:
    return Request(request_id="R1", recipe_digest="d" * 64, step_executions=steps)


def test_the_report_contains_every_field_of_the_snapshot() -> None:
    assert set(RunLogSnapshot.__annotations__) <= set(RunStatusWire.model_fields)


def test_a_finished_run_that_lost_entries_reports_partial_and_samples_them(
    tmp_path: Path,
) -> None:
    lost = FAILED_ENTRY_SAMPLE + 3
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.set_total(lost + 1)
        for index in range(lost):
            log.entry_failed(f"g__s{index}", _crash(f"bad {index}"))
        log.heartbeat(lost + 1, lost + 1)
        log.entries_written(1)
        log.finished()

    report = run_status_wire(tmp_path, "E1")

    assert report is not None
    assert (report.status, report.outcome, report.terminal) == (
        "finished",
        "partial",
        True,
    )
    assert report.entries_failed == lost
    assert len(report.failed_entries) == FAILED_ENTRY_SAMPLE
    assert report.failed_entries_has_more
    first = report.failed_entries[0]
    assert first.key == "g__s0"
    assert first.error is not None
    assert (first.error.type, first.error.message) == ("ValueError", "bad 0")
    assert report.progress_fraction == 1.0
    assert report.progress is None


def test_a_refusal_is_parsed_into_its_envelope(tmp_path: Path) -> None:
    refusal = Refusal(
        "short by 2",
        reason="coverage_shortfall",
        step_id="templates",
        detail={"covered": 3, "target": 5},
    )
    with _log(tmp_path, "E2") as log:
        log.started(kind="graph-step", target="templates")
        log.failed(refusal.error_json())

    report = run_status_wire(tmp_path, "E2")

    assert report is not None
    assert report.outcome == "failed"
    assert report.error is not None
    envelope = report.error.refusal
    assert envelope is not None
    assert envelope.kind == "refused"
    assert envelope.reason == "coverage_shortfall"
    assert envelope.step == "templates"
    assert envelope.message == "short by 2"
    assert envelope.detail == {"covered": 3, "target": 5}


def test_a_blob_that_is_not_an_object_is_reported_as_its_text() -> None:
    error = recorded_error("killed by the queue")
    assert error is not None
    assert (error.type, error.message, error.refusal) == (
        "",
        "killed by the queue",
        None,
    )
    assert recorded_error("") is None


def test_an_absent_run_reads_as_none(tmp_path: Path) -> None:
    assert run_status_wire(tmp_path, "MISSING") is None


def test_progress_events_are_listed_when_asked_for(tmp_path: Path) -> None:
    with _log(tmp_path, "E3") as log:
        log.started(kind="train-pose", target="pose")
        log.on_epoch_end(0, 4, {"loss": 1.0})

    report = run_status_wire(tmp_path, "E3", progress=True)

    assert report is not None
    assert report.outcome == "running"
    assert report.progress_fraction == 0.25
    assert report.progress is not None
    assert [event["step_type"] for event in report.progress] == ["epoch"]


def test_a_request_reports_each_step_with_its_outcome(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.entry_failed("g__s1", _crash("bad"))
        log.finished()

    report = request_status_wire(tmp_path, _request(speed="E1", templates="E2"))

    assert (report.status, report.verdict, report.terminal) == (
        "running",
        "running",
        False,
    )
    speed, templates = report.steps
    assert (speed.outcome, speed.entries_failed, speed.failed_entries_has_more) == (
        "partial",
        1,
        False,
    )
    assert [entry.key for entry in speed.failed_entries] == ["g__s1"]
    assert (templates.status, templates.outcome, templates.run_id) == (
        "",
        "pending",
        "",
    )


def test_a_finished_request_with_lost_entries_is_partial(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.entry_failed("g__s1", _crash("bad"))
        log.finished()

    report = request_status_wire(tmp_path, _request(speed="E1"))

    assert (report.status, report.verdict, report.terminal) == (
        "finished",
        "partial",
        True,
    )


def test_a_running_request_whose_driver_lease_is_stale_is_orphaned(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")

    request = _request(speed="E1")

    assert request_status_wire(tmp_path, request).verdict == "running"
    _ = write_driver_lease(tmp_path, "R1", stale=False)
    assert request_status_wire(tmp_path, request).verdict == "running"
    _ = write_driver_lease(tmp_path, "R1", stale=True)
    orphaned = request_status_wire(tmp_path, request)
    assert (orphaned.status, orphaned.verdict) == ("running", "orphaned")
    assert request_summary(tmp_path, request).verdict == "orphaned"


def test_a_stale_lease_is_not_orphaned_while_a_step_process_runs(
    tmp_path: Path,
) -> None:
    """A killed driver leaves its step running in a session of its own."""
    with _log(tmp_path, "E1") as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=os.getpid()
        )
    _ = write_driver_lease(tmp_path, "R1", stale=True)

    report = request_status_wire(tmp_path, _request(speed="E1", templates="E2"))

    assert (report.status, report.verdict) == ("running", "running")


def test_a_pid_reused_after_the_step_started_does_not_keep_a_request_running(
    tmp_path: Path,
) -> None:
    """This process began after the recorded start, as one reusing a pid would."""
    with _log(tmp_path, "E1") as log:
        log.started(
            kind="feature",
            target="speed",
            host=socket.gethostname(),
            pid=os.getpid(),
            created_at="2000-01-01T00:00:00+00:00",
        )
    _ = write_driver_lease(tmp_path, "R1", stale=True)

    report = request_status_wire(tmp_path, _request(speed="E1"))

    assert report.verdict == "orphaned"


def test_a_terminal_request_is_not_orphaned(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.cancelled()
    _ = write_driver_lease(tmp_path, "R1", stale=True)

    report = request_status_wire(tmp_path, _request(speed="E1"))

    assert report.verdict == "cancelled"


def _write_log(base: Path, execution_id: str, *records: Mapping[str, object]) -> None:
    """Write a run-log whose lines carry the times the records give."""
    path = run_log_path(base, execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _ago(seconds: float) -> str:
    return (NOW - datetime.timedelta(seconds=seconds)).isoformat()


def test_a_request_without_a_lease_whose_started_step_is_dead_is_orphaned(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=exited_pid()
        )

    report = request_status_wire(tmp_path, _request(speed="E1", templates="E2"))

    assert (report.status, report.verdict) == ("running", "orphaned")
    assert report.steps[0].liveness == "dead"
    assert report.poll_after_seconds is None


def test_a_request_without_a_lease_between_queue_steps_is_running(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed", pid=exited_pid())
        log.finished()

    report = request_status_wire(tmp_path, _request(speed="E1", templates="E2"))

    assert (report.status, report.verdict) == ("running", "running")
    assert report.poll_after_seconds == POLL_AFTER_MIN_SECONDS


def test_a_request_without_a_lease_whose_step_is_unknown_stays_running(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed", host=OTHER_HOST, pid=4242)

    report = request_status_wire(tmp_path, _request(speed="E1"))

    assert (report.verdict, report.steps[0].liveness) == ("running", "unknown")


def test_a_request_whose_step_went_silent_in_another_container_is_orphaned(
    tmp_path: Path,
) -> None:
    """A Docker container with a new host name for every run, as the CalMS21 one."""
    started = {
        "t": _ago(3600),
        "ev": "started",
        "kind": "feature",
        "target": "speed",
        "host": "3f9a2c1d7e4b",
        "pid": 7,
        "created_at": _ago(3600),
        "liveness_interval": 60.0,
    }
    _write_log(tmp_path, "E1", started, {"t": _ago(1200), "ev": "alive"})

    report = request_status_wire(tmp_path, _request(speed="E1"), now=NOW)

    assert (report.status, report.verdict) == ("running", "orphaned")
    assert request_summary(tmp_path, _request(speed="E1")).verdict == "orphaned"


def test_each_step_reports_its_start_heartbeat_phase_and_liveness(
    tmp_path: Path,
) -> None:
    started = {
        "t": _ago(300),
        "ev": "started",
        "kind": "feature",
        "target": "speed",
        "host": OTHER_HOST,
        "pid": 7,
        "created_at": _ago(300),
        "liveness_interval": 60.0,
    }
    _write_log(
        tmp_path,
        "E1",
        started,
        {"t": _ago(200), "ev": "phase", "phase": "fit", "message": "2 entries"},
        {"t": _ago(30), "ev": "alive"},
    )

    report = request_status_wire(
        tmp_path, _request(speed="E1", templates="E2"), now=NOW
    )

    speed, templates = report.steps
    assert (
        speed.host,
        speed.started_at,
        speed.heartbeat_at,
        speed.phase,
        speed.liveness,
    ) == (OTHER_HOST, _ago(300), _ago(30), "fit", "alive")
    assert (templates.liveness, templates.host) == (None, "")
    run = run_status_wire(tmp_path, "E1", now=NOW)
    assert run is not None
    assert (run.liveness, run.phase) == ("alive", "fit")


def _rollup(*steps: StepAttempt, status: RequestStatus = "running") -> RequestRollup:
    return RequestRollup(request_id="R1", status=status, steps=steps)


def _running(started_ago: float, done: int = 0, total: int = 0) -> StepAttempt:
    return StepAttempt(
        step_id="speed",
        execution_id="E1",
        status="running",
        started_at=_ago(started_ago),
        progress_done=done,
        progress_total=total,
    )


def test_poll_after_scales_with_elapsed_time_and_progress() -> None:
    def wait(*steps: StepAttempt) -> int | None:
        return poll_after_seconds(_rollup(*steps), "running", now=NOW)

    assert wait(StepAttempt(step_id="speed", execution_id="E1")) == 10
    assert wait(_running(30)) == POLL_AFTER_MIN_SECONDS
    assert wait(_running(1000)) == 100
    assert wait(_running(100, done=1, total=4)) == 150
    assert wait(_running(36000)) == POLL_AFTER_MAX_SECONDS
    assert wait(_running(36000), _running(1000)) == 100


def test_a_terminal_or_orphaned_request_has_no_poll_after() -> None:
    finished = StepAttempt(step_id="speed", execution_id="E1", status="finished")

    assert poll_after_seconds(_rollup(finished, status="finished"), "finished") is None
    assert poll_after_seconds(_rollup(_running(60)), "orphaned", now=NOW) is None
