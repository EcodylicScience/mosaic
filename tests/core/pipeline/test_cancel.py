"""Cancelling an attempt or a pipeline request from another process.

A sleeper stands in for a step. A driver runs as ``mosaic pipeline run
--request`` in a separate session, with sleepers in place of its steps. The tests
check where SIGTERM is delivered and how each run-log ends.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.cancel import (
    AttemptNotFound,
    CannotSignal,
    RequestNotFound,
    cancel_attempt,
    cancel_request,
)
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.graph import (
    Request,
    driver_lease_path,
    driver_log_path,
    read_driver_lease,
    request_rollup,
    submit_request,
)
from mosaic.core.pipeline.markers import process_running
from mosaic.runlog import JsonlRunLog, new_execution_id, read_run, run_log_dir
from tests.helpers import (
    SLEEPER_COMMAND,
    add_tracks_variant,
    chain_recipe,
    exited_pid,
    make_dataset,
    wait_for,
    write_driver_lease,
)

VARIANT = "convert-trex.0.2-1111111111"

OTHER_HOST = "another-host.invalid"

STEP_SECONDS = 180.0
"""The longest a test waits for a driver, which imports mosaic, to act."""

DRIVER_PROGRAM = (
    "import sys\n"
    "from mosaic.core.pipeline.graph import run\n"
    f"run.MOSAIC_COMMAND = {(*SLEEPER_COMMAND, '600')!r}\n"
    "from mosaic.cli import app\n"
    "app(sys.argv[1:], prog_name='mosaic')\n"
)
"""``mosaic`` with every step that its driver starts replaced by a sleeper."""


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, with no feature computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


@pytest.fixture
def submitted(tracked: Dataset) -> Request:
    """The chain, submitted and not run."""
    return submit_request(tracked, chain_recipe()).request


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A process that sleeps for ten minutes unless SIGTERM ends it."""
    process = subprocess.Popen([*SLEEPER_COMMAND, "600"], stdin=subprocess.DEVNULL)
    yield process
    process.kill()
    _ = process.wait()


def _start(
    ds: Dataset,
    execution_id: str,
    *,
    pid: int,
    host: str = "",
    end: bool = False,
    created_at: str | None = None,
) -> None:
    """Write the run-log of an attempt that started as *pid*, finished when *end*.

    *created_at* is the instant the ``started`` record gives, now when ``None``.
    """
    path = run_log_dir(ds.base_dir) / f"{execution_id}.jsonl"
    with JsonlRunLog(path, execution_id) as log:
        log.started(
            kind="feature",
            target="speed",
            host=host or socket.gethostname(),
            pid=pid,
            created_at=created_at,
        )
        if end:
            log.finished()


def _status(ds: Dataset, execution_id: str) -> str:
    """The status of an attempt's run-log, ``""`` before it writes one."""
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    return "" if snapshot is None else snapshot["status"]


# --- one attempt ---------------------------------------------------------


def test_a_running_attempt_receives_sigterm_and_writes_its_own_ending(
    tracked: Dataset, sleeper: subprocess.Popen[bytes]
) -> None:
    execution_id = new_execution_id()
    _start(tracked, execution_id, pid=sleeper.pid)

    result = cancel_attempt(run_log_dir(tracked.base_dir), execution_id)

    assert (result.outcome, result.status, result.signalled, result.pid) == (
        "signalled",
        "cancelling",
        True,
        sleeper.pid,
    )
    assert sleeper.wait(timeout=10) == -signal.SIGTERM
    assert _status(tracked, execution_id) == "running"


def test_a_finished_attempt_is_left_as_it_is(tracked: Dataset) -> None:
    execution_id = new_execution_id()
    _start(tracked, execution_id, pid=exited_pid(), end=True)

    result = cancel_attempt(run_log_dir(tracked.base_dir), execution_id)

    assert (result.outcome, result.status, result.signalled) == (
        "ended",
        "finished",
        False,
    )


def test_an_attempt_on_another_host_is_not_signalled(tracked: Dataset) -> None:
    execution_id = new_execution_id()
    _start(tracked, execution_id, pid=exited_pid(), host=OTHER_HOST)

    result = cancel_attempt(run_log_dir(tracked.base_dir), execution_id)

    assert (result.outcome, result.host, result.signalled) == (
        "other_host",
        OTHER_HOST,
        False,
    )
    assert OTHER_HOST in result.message
    assert _status(tracked, execution_id) == "running"


def test_an_attempt_whose_process_vanished_is_recorded_cancelled(
    tracked: Dataset,
) -> None:
    execution_id = new_execution_id()
    _start(tracked, execution_id, pid=exited_pid())

    result = cancel_attempt(run_log_dir(tracked.base_dir), execution_id)

    assert (result.outcome, result.status, result.signalled) == (
        "recorded",
        "cancelled",
        False,
    )
    assert _status(tracked, execution_id) == "cancelled"


def test_a_pid_reused_after_the_attempt_started_is_not_signalled(
    tracked: Dataset, sleeper: subprocess.Popen[bytes]
) -> None:
    """A recorded pid that a later process holds, as after a reboot."""
    execution_id = new_execution_id()
    _start(tracked, execution_id, pid=sleeper.pid, created_at="2020-01-01T00:00:00Z")

    result = cancel_attempt(run_log_dir(tracked.base_dir), execution_id)

    assert (result.outcome, result.status, result.signalled) == (
        "recorded",
        "cancelled",
        False,
    )
    assert sleeper.poll() is None, "the process holding the reused pid was signalled"
    assert _status(tracked, execution_id) == "cancelled"


def test_an_attempt_without_a_run_log_is_not_found(tracked: Dataset) -> None:
    with pytest.raises(AttemptNotFound) as raised:
        _ = cancel_attempt(run_log_dir(tracked.base_dir), new_execution_id())

    assert error_envelope(raised.value).kind == "not_found"


def test_an_attempt_without_a_pid_cannot_be_signalled(tracked: Dataset) -> None:
    execution_id = new_execution_id()
    _start(tracked, execution_id, pid=0)

    with pytest.raises(CannotSignal) as raised:
        _ = cancel_attempt(run_log_dir(tracked.base_dir), execution_id)

    assert error_envelope(raised.value).kind == "conflict"
    assert _status(tracked, execution_id) == "running"


# --- one request ---------------------------------------------------------


def test_a_detached_request_is_cancelled_through_its_driver(
    tracked: Dataset, submitted: Request, tmp_path: Path
) -> None:
    """The driver receives SIGTERM, and forwards it to the step that it runs."""
    with (tmp_path / "driver.out").open("w") as output:
        driver = subprocess.Popen(
            [
                sys.executable,
                "-c",
                DRIVER_PROGRAM,
                "pipeline",
                "run",
                "--manifest",
                str(tracked.manifest_path),
                "--request",
                submitted.request_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    try:
        wait_for(
            lambda: _lease_step_pid(tracked, submitted) > 0,
            STEP_SECONDS,
            "the driver did not start its first step",
        )
        step_pid = _lease_step_pid(tracked, submitted)

        result = cancel_request(tracked, submitted.request_id)

        assert (result.outcome, result.driver_pid, result.steps) == (
            "signalled",
            driver.pid,
            (),
        )
        assert driver.wait(timeout=STEP_SECONDS) == 130
    finally:
        if driver.poll() is None:
            driver.kill()
            _ = driver.wait()

    log = driver_log_path(tracked.base_dir, submitted.request_id).read_text()
    assert f"step speed exited with code {-signal.SIGTERM}" in log
    assert not process_running(step_pid)
    rollup = request_rollup(tracked.base_dir, submitted)
    assert rollup.status == "cancelled"
    assert [attempt.status for attempt in rollup.steps] == ["cancelled", "cancelled"]
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()
    assert cancel_request(tracked, submitted.request_id).outcome == "ended"


def test_without_a_driver_the_running_step_is_signalled_and_the_rest_recorded(
    tracked: Dataset, submitted: Request, sleeper: subprocess.Popen[bytes]
) -> None:
    speed = submitted.execution_of("speed")
    templates = submitted.execution_of("templates")
    _start(tracked, speed, pid=sleeper.pid)

    result = cancel_request(tracked, submitted.request_id)

    assert result.outcome == "signalled"
    assert [(step.execution_id, step.outcome) for step in result.steps] == [
        (speed, "signalled"),
        (templates, "recorded"),
    ]
    assert sleeper.wait(timeout=10) == -signal.SIGTERM
    assert (_status(tracked, speed), _status(tracked, templates)) == (
        "running",
        "cancelled",
    )
    assert result.status == "cancelled"


def test_an_orphaned_request_has_its_vanished_and_unstarted_steps_recorded(
    tracked: Dataset, submitted: Request
) -> None:
    speed = submitted.execution_of("speed")
    _start(tracked, speed, pid=exited_pid())
    _ = write_driver_lease(
        tracked.base_dir, submitted.request_id, stale=True, pid=exited_pid()
    )

    result = cancel_request(tracked, submitted.request_id)

    assert (result.outcome, result.status, result.driver_pid) == (
        "recorded",
        "cancelled",
        0,
    )
    assert [step.outcome for step in result.steps] == ["recorded", "recorded"]
    assert [
        attempt.status for attempt in request_rollup(tracked.base_dir, submitted).steps
    ] == ["cancelled", "cancelled"]


def test_a_step_process_named_by_a_stale_lease_receives_sigterm(
    tracked: Dataset, submitted: Request, sleeper: subprocess.Popen[bytes]
) -> None:
    """A killed driver's step may not have written its run-log yet."""
    _ = write_driver_lease(
        tracked.base_dir,
        submitted.request_id,
        stale=False,
        pid=exited_pid(),
        step_id="speed",
        step_pid=sleeper.pid,
    )

    result = cancel_request(tracked, submitted.request_id)

    assert [(step.outcome, step.pid) for step in result.steps] == [
        ("signalled", sleeper.pid),
        ("recorded", 0),
    ]
    assert sleeper.wait(timeout=10) == -signal.SIGTERM


def test_a_finished_request_is_left_as_it_is(
    tracked: Dataset, submitted: Request
) -> None:
    for step_id in ("speed", "templates"):
        _start(tracked, submitted.execution_of(step_id), pid=exited_pid(), end=True)

    result = cancel_request(tracked, submitted.request_id)

    assert (result.outcome, result.status, result.steps) == ("ended", "finished", ())


def test_a_request_driven_on_another_host_is_left_as_it_is(
    tracked: Dataset, submitted: Request
) -> None:
    lease = write_driver_lease(
        tracked.base_dir, submitted.request_id, stale=False, host=OTHER_HOST
    )

    result = cancel_request(tracked, submitted.request_id)

    assert (result.outcome, result.status, result.steps) == (
        "other_host",
        "running",
        (),
    )
    assert OTHER_HOST in result.message
    assert read_driver_lease(tracked.base_dir, submitted.request_id) == lease
    assert _status(tracked, submitted.execution_of("speed")) == ""


def test_a_request_with_a_step_on_another_host_is_left_as_it_is(
    tracked: Dataset, submitted: Request
) -> None:
    _start(tracked, submitted.execution_of("speed"), pid=1, host=OTHER_HOST)

    result = cancel_request(tracked, submitted.request_id)

    assert result.outcome == "other_host"
    assert _status(tracked, submitted.execution_of("templates")) == ""


def test_a_request_without_a_file_is_not_found(tracked: Dataset) -> None:
    with pytest.raises(RequestNotFound) as raised:
        _ = cancel_request(tracked, new_execution_id())

    assert error_envelope(raised.value).kind == "not_found"


def _lease_step_pid(ds: Dataset, request: Request) -> int:
    """The step process that the request's driver lease names, 0 without one."""
    lease = read_driver_lease(ds.base_dir, request.request_id)
    return 0 if lease is None else lease.step_pid
