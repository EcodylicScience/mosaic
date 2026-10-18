"""Driving a submitted request one step process at a time, here or detached.

The driver runs each step as ``mosaic run --graph-request`` in a process of its
own, with the step's output in the request's driver log. These tests start real
processes, because what is under test is how the driver, its steps and its lease
behave when a process ends.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.graph import (
    MOSAIC_COMMAND,
    PARENT_FAILED_TYPE,
    STEP_CANCEL_GRACE_SECONDS,
    DriverRunning,
    Request,
    StepStillRunning,
    driver_lease_path,
    execute_step,
    read_driver_lease,
    record_step_ended,
    request_rollup,
    request_status_wire,
    run_request,
    save_request,
    start_detached_driver,
    step_argv,
    submit_request,
)
from mosaic.core.pipeline.graph import run as graph_run
from mosaic.core.pipeline.job import CancelToken
from mosaic.core.pipeline.markers import process_running
from mosaic.runlog import JsonlRunLog, read_run, run_log_dir, run_log_path
from tests.helpers import (
    CHAIN_RECIPE,
    SLEEPER_COMMAND,
    add_tracks_variant,
    cancel_when,
    chain_recipe,
    chain_with_sibling_recipe,
    exited_pid,
    make_dataset,
    wait_for,
    write_driver_lease,
)

VARIANT = "convert-trex.0.2-1111111111"

STEP_SECONDS = 180.0
"""The longest a test waits for a step process, which imports mosaic, to end."""


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


@pytest.fixture
def submitted(tracked: Dataset) -> Request:
    """The chain, submitted and not run."""
    return submit_request(tracked, chain_recipe()).request


def _driver_argv(ds: Dataset, request: Request, *flags: str) -> list[str]:
    return [
        *MOSAIC_COMMAND,
        "pipeline",
        "run",
        "--manifest",
        str(ds.manifest_path),
        "--request",
        request.request_id,
        *flags,
    ]


def _lease_step_pid(ds: Dataset, request: Request) -> int:
    """The step process that the request's driver lease names, 0 without one."""
    lease = read_driver_lease(ds.base_dir, request.request_id)
    return 0 if lease is None else lease.step_pid


def _status(ds: Dataset, request: Request, step_id: str) -> str:
    """The status of a step's run-log, ``""`` before it writes one."""
    snapshot = read_run(run_log_dir(ds.base_dir), request.execution_of(step_id))
    return "" if snapshot is None else snapshot["status"]


def test_a_driver_runs_every_step_and_writes_nothing_to_its_streams(
    tracked: Dataset, submitted: Request, capfd: pytest.CaptureFixture[str]
) -> None:
    done = run_request(tracked, submitted)

    assert (done.status, done.ran, done.skipped) == (
        "finished",
        ("speed", "templates"),
        (),
    )
    assert capfd.readouterr() == ("", "")
    assert "step templates exited with code 0" in done.log_path.read_text()
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()
    assert request_status_wire(tracked.base_dir, submitted).verdict == "finished"


def test_driving_a_finished_request_again_skips_every_step(
    tracked: Dataset, submitted: Request
) -> None:
    _ = run_request(tracked, submitted)

    again = run_request(tracked, submitted)

    assert (again.status, again.ran, again.skipped) == (
        "finished",
        (),
        ("speed", "templates"),
    )


def test_a_cancel_between_steps_records_the_unstarted_ones_cancelled(
    tracked: Dataset, submitted: Request
) -> None:
    _ = execute_step(tracked, submitted, "speed")
    token = CancelToken()
    token.cancel()

    done = run_request(tracked, submitted, cancel_token=token)

    assert (done.status, done.ran, done.cancelled) == ("cancelled", (), ("templates",))
    report = request_status_wire(tracked.base_dir, submitted)
    assert report.verdict == "cancelled"
    assert [step.status for step in report.steps] == ["finished", "cancelled"]
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()


def test_a_cancel_during_a_step_terminates_its_process(
    tracked: Dataset, submitted: Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step is a sleeper that only a forwarded SIGTERM ends early."""
    monkeypatch.setattr(graph_run, "MOSAIC_COMMAND", (*SLEEPER_COMMAND, "600"))
    step_pids: list[int] = []

    def step_started() -> bool:
        step_pids.append(_lease_step_pid(tracked, submitted))
        return step_pids[-1] > 0

    token = CancelToken()
    _ = cancel_when(step_started, token, message="the first step never started")
    began = time.monotonic()

    done = run_request(tracked, submitted, cancel_token=token)

    assert time.monotonic() - began < STEP_CANCEL_GRACE_SECONDS
    assert not process_running(step_pids[-1])
    assert (done.status, done.ran, done.cancelled) == (
        "cancelled",
        ("speed",),
        ("templates",),
    )
    assert _status(tracked, submitted, "speed") == "cancelled"


def test_a_refused_step_fails_the_request_with_its_refusal_parsed(
    tracked: Dataset, submitted: Request
) -> None:
    moved = submitted.model_copy(
        update={"step_versions": {**submitted.step_versions, "templates": "0.0"}}
    )
    _ = save_request(tracked.base_dir, moved)

    done = run_request(tracked, moved)

    assert (done.status, done.ran, done.cancelled) == ("failed", ("speed",), ())
    assert "step speed exited with code 65" in done.log_path.read_text()
    report = request_status_wire(tracked.base_dir, moved)
    speed, templates = report.steps
    assert (report.verdict, speed.status, templates.outcome) == (
        "failed",
        "failed",
        "pending",
    )
    assert speed.error is not None
    assert speed.error.refusal is not None
    assert speed.error.refusal.reason == "version_moved"


def test_a_failed_step_ends_its_branch_and_the_driver_runs_its_sibling(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``speed`` exits 1, so ``templates`` is not run, and ``sibling`` still is."""

    def speed_exits(
        manifest_path: Path | str,
        request: Request,
        step_id: str,
        *,
        program: str | Sequence[str],
    ) -> list[str]:
        if step_id == "speed":
            return [sys.executable, "-c", "raise SystemExit(1)"]
        return step_argv(manifest_path, request, step_id, program=program)

    monkeypatch.setattr(graph_run, "step_argv", speed_exits)
    request = submit_request(tracked, chain_with_sibling_recipe()).request

    done = run_request(tracked, request)

    assert (done.status, done.ran, done.blocked) == (
        "failed",
        ("speed", "sibling"),
        ("templates",),
    )
    assert _status(tracked, request, "sibling") == "finished"
    templates = read_run(
        run_log_dir(tracked.base_dir), request.execution_of("templates")
    )
    assert templates is not None
    assert templates["status"] == "failed"
    error = json.loads(templates["error_json"])
    assert (error["type"], error["reason"], error["parent"]) == (
        PARENT_FAILED_TYPE,
        "parent_failed",
        "speed",
    )
    assert error["parent_execution_id"] == request.execution_of("speed")
    assert not driver_lease_path(tracked.base_dir, request.request_id).exists()
    assert request_status_wire(tracked.base_dir, request).verdict == "failed"

    _ = write_driver_lease(tracked.base_dir, request.request_id, stale=True)

    again = run_request(tracked, request)

    assert (again.status, again.ran, again.blocked) == ("failed", (), ())
    assert again.skipped == ("sibling",)
    assert not driver_lease_path(tracked.base_dir, request.request_id).exists()


def test_a_request_that_has_ended_is_not_driven_again(tracked: Dataset) -> None:
    """``speed`` failed with no driver left to run ``sibling``. The request is closed."""
    request = submit_request(tracked, chain_with_sibling_recipe()).request
    record_step_ended(
        tracked,
        request.execution_of("speed"),
        step_id="speed",
        owner="",
        status="failed",
        error_json=json.dumps({"type": "RuntimeError", "message": "speed broke"}),
    )

    done = run_request(tracked, request)

    assert (done.status, done.ran, done.skipped, done.blocked) == (
        "failed",
        (),
        (),
        (),
    )
    for step_id in ("sibling", "templates"):
        assert _status(tracked, request, step_id) == ""
    assert not driver_lease_path(tracked.base_dir, request.request_id).exists()


def test_a_driver_refuses_while_a_step_process_runs(
    tracked: Dataset, submitted: Request
) -> None:
    execution_id = submitted.execution_of("speed")
    with JsonlRunLog(run_log_path(tracked.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="feature", target="speed", host=socket.gethostname(), pid=os.getpid()
        )

    with pytest.raises(StepStillRunning) as raised:
        _ = run_request(tracked, submitted)

    assert error_envelope(raised.value).kind == "conflict"
    assert raised.value.error_step() == "speed"
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()


def test_a_step_process_named_by_a_stale_lease_refuses_a_resume_until_it_ends(
    tracked: Dataset, submitted: Request
) -> None:
    """A killed driver's step may not have written its run-log yet."""
    step = subprocess.Popen([*SLEEPER_COMMAND, "60"], stdin=subprocess.DEVNULL)
    try:
        _ = write_driver_lease(
            tracked.base_dir,
            submitted.request_id,
            stale=False,
            pid=exited_pid(),
            step_id="speed",
            step_pid=step.pid,
        )
        assert _status(tracked, submitted, "speed") == ""

        with pytest.raises(StepStillRunning) as raised:
            _ = start_detached_driver(tracked, submitted)
        assert raised.value.pid == step.pid
        with pytest.raises(StepStillRunning):
            _ = run_request(tracked, submitted)
    finally:
        step.kill()
        _ = step.wait()

    resumed = run_request(tracked, submitted)

    assert (resumed.status, resumed.ran) == ("finished", ("speed", "templates"))


def test_a_second_driver_is_refused_while_a_live_one_holds_the_lease(
    tracked: Dataset, submitted: Request
) -> None:
    _ = write_driver_lease(tracked.base_dir, submitted.request_id, stale=False)

    with pytest.raises(DriverRunning):
        _ = start_detached_driver(tracked, submitted)
    with pytest.raises(DriverRunning):
        _ = run_request(tracked, submitted)


def test_sigterm_to_the_driver_between_steps_cancels_the_request(
    tracked: Dataset, submitted: Request, tmp_path: Path
) -> None:
    with (tmp_path / "driver.err").open("w") as errors:
        driver = subprocess.Popen(
            _driver_argv(tracked, submitted, "--json"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        wait_for(
            lambda: _status(tracked, submitted, "speed") == "finished",
            STEP_SECONDS,
            "the first step never finished",
        )
        driver.send_signal(signal.SIGTERM)
        out, _ = driver.communicate(timeout=STEP_SECONDS)

    assert driver.returncode == 130
    assert json.loads(out)["verdict"] == "cancelled"
    assert request_rollup(tracked.base_dir, submitted).status == "cancelled"
    assert _status(tracked, submitted, "templates") == "cancelled"
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()


def test_a_detached_driver_outlives_its_parent_and_the_parent_prints_one_value(
    tracked: Dataset, submitted: Request
) -> None:
    parent = subprocess.Popen(
        _driver_argv(tracked, submitted, "--detach", "--json"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    out, err = parent.communicate(timeout=STEP_SECONDS)
    # A driver left in the parent's process group would die here, as it would
    # when the terminal that ran the parent closes.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(parent.pid, signal.SIGKILL)

    assert parent.returncode == 0, err
    started = json.loads(out)
    assert started["request_id"] == submitted.request_id
    assert request_rollup(tracked.base_dir, submitted).status == "running"
    wait_for(
        lambda: request_rollup(tracked.base_dir, submitted).status == "finished",
        2 * STEP_SECONDS,
        "the detached driver never finished the request",
    )
    wait_for(
        lambda: not process_running(started["driver_pid"]),
        STEP_SECONDS,
        "the detached driver never exited",
    )
    assert not driver_lease_path(tracked.base_dir, submitted.request_id).exists()
    assert Path(started["log_path"]).read_text()


def test_a_killed_driver_reads_orphaned_once_its_step_ends(
    tracked: Dataset, submitted: Request
) -> None:
    driver = subprocess.Popen(
        _driver_argv(tracked, submitted),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for(
        lambda: _lease_step_pid(tracked, submitted) > 0,
        STEP_SECONDS,
        "the driver never started the first step",
    )
    driver.kill()
    _ = driver.wait()

    # The killed driver's step runs in a session of its own. The request is not
    # orphaned while it runs, and a resume is refused until it has ended.
    assert request_status_wire(tracked.base_dir, submitted).verdict == "running"
    with pytest.raises(StepStillRunning):
        _ = run_request(tracked, submitted)

    wait_for(
        lambda: _status(tracked, submitted, "speed") == "finished",
        STEP_SECONDS,
        "the orphaned step never finished",
    )
    report = request_status_wire(tracked.base_dir, submitted)
    assert (report.status, report.verdict) == ("running", "orphaned")
    resumed = run_request(tracked, submitted)

    assert (resumed.status, resumed.ran, resumed.skipped) == (
        "finished",
        ("templates",),
        ("speed",),
    )
    assert request_status_wire(tracked.base_dir, submitted).verdict == "finished"


def test_the_detach_flag_needs_a_request(tracked: Dataset) -> None:
    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "run",
            "--manifest",
            str(tracked.manifest_path),
            "--recipe",
            json.dumps(CHAIN_RECIPE),
            "--detach",
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "--detach is used with --request" in result.stderr


def _started_elsewhere(
    ds: Dataset, request: Request, step_id: str, *, interval: float
) -> None:
    """A run-log of another host whose last line is an hour old."""
    execution_id = request.execution_of(step_id)
    stamp = "2026-01-01T00:00:00+00:00"
    started: dict[str, object] = {
        "t": stamp,
        "ev": "started",
        "kind": "feature",
        "target": step_id,
        "host": "another-host.invalid",
        "pid": 7,
        "created_at": stamp,
    }
    if interval:
        started["liveness_interval"] = interval
    path = run_log_path(ds.base_dir, execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(started) + "\n")


def test_a_request_whose_step_went_silent_elsewhere_resumes(
    tracked: Dataset, submitted: Request
) -> None:
    """A container that died with its step, as the CalMS21 one did."""
    _started_elsewhere(tracked, submitted, "speed", interval=60.0)
    assert request_status_wire(tracked.base_dir, submitted).verdict == "orphaned"

    resumed = run_request(tracked, submitted)

    assert (resumed.status, resumed.ran) == ("finished", ("speed", "templates"))


def test_a_resume_refused_by_a_step_of_unknown_liveness_says_how_to_close_it(
    tracked: Dataset, submitted: Request
) -> None:
    _started_elsewhere(tracked, submitted, "speed", interval=0.0)

    with pytest.raises(StepStillRunning) as raised:
        _ = run_request(tracked, submitted)

    assert raised.value.liveness == "unknown"
    assert f"--request {submitted.request_id} --assume-exited" in str(raised.value)
    assert "submit the recipe again" in str(raised.value)
