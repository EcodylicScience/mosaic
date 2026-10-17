"""Whether a step attempt's process may still be running, as its run-log tells."""

from __future__ import annotations

import datetime
import os
import socket
from dataclasses import replace

import pytest

from mosaic.core.pipeline.graph import (
    StepAttempt,
    attempt_liveness,
    may_still_run,
    step_liveness,
)
from mosaic.core.pipeline.graph import liveness as liveness_module
from mosaic.core.pipeline.markers import boots_differ
from mosaic.runlog import LIVENESS_STALE_INTERVALS
from tests.helpers import exited_pid

OTHER_HOST = "another-host.invalid"

NOW = datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.UTC)


def _ago(seconds: float) -> str:
    return (NOW - datetime.timedelta(seconds=seconds)).isoformat()


def _attempt(
    *,
    host: str = OTHER_HOST,
    pid: int = 4242,
    status: str = "running",
    started_ago: float = 3600.0,
    heartbeat_ago: float = 3600.0,
    boot_id: str = "",
    pid_ns: str = "",
    liveness_interval: float = 0.0,
) -> StepAttempt:
    return StepAttempt(
        step_id="speed",
        execution_id="E1",
        status=status,
        host=host,
        pid=pid,
        started_at=_ago(started_ago),
        heartbeat_at=_ago(heartbeat_ago),
        boot_id=boot_id,
        pid_ns=pid_ns,
        liveness_interval=liveness_interval,
    )


def _here(pid: int, **fields: str) -> StepAttempt:
    """An attempt of this host, started just now as far as the pid check goes."""
    started = datetime.datetime.now(datetime.UTC).isoformat()
    attempt = replace(
        _attempt(host=socket.gethostname(), pid=pid),
        started_at=started,
        heartbeat_at=started,
    )
    return replace(attempt, **fields)


@pytest.mark.parametrize("status", ["finished", "failed", "cancelled"])
def test_a_terminal_attempt_has_ended(status: str) -> None:
    assert step_liveness(_attempt(status=status), now=NOW) == "ended"


def test_an_attempt_from_an_earlier_boot_of_this_host_is_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(liveness_module, "current_boot_id", lambda: "linux:now")

    attempt = _here(os.getpid(), boot_id="linux:before")

    assert step_liveness(attempt) == "dead"
    assert not may_still_run(attempt)


def test_an_attempt_whose_local_process_runs_is_alive() -> None:
    assert step_liveness(_here(os.getpid())) == "alive"


def test_an_attempt_whose_local_process_exited_is_dead() -> None:
    assert step_liveness(_here(exited_pid())) == "dead"


def test_a_pid_reused_after_the_attempt_started_is_dead() -> None:
    attempt = _here(os.getpid(), started_at="2000-01-01T00:00:00+00:00")

    assert step_liveness(attempt) == "dead"


def test_an_attempt_on_another_host_with_a_current_heartbeat_is_alive() -> None:
    attempt = _attempt(liveness_interval=60.0, heartbeat_ago=90.0)

    assert step_liveness(attempt, now=NOW) == "alive"
    assert may_still_run(attempt, now=NOW)


def test_an_attempt_on_another_host_silent_for_ten_intervals_is_dead() -> None:
    silent = LIVENESS_STALE_INTERVALS * 60.0 + 1
    attempt = _attempt(liveness_interval=60.0, heartbeat_ago=silent)

    assert step_liveness(attempt, now=NOW) == "dead"
    assert not may_still_run(attempt, now=NOW)


def test_an_attempt_on_another_host_without_an_interval_is_unknown() -> None:
    """A run-log written before liveness, as the CalMS21 attempt's was."""
    assert step_liveness(_attempt(), now=NOW) == "unknown"


def test_a_pid_in_another_namespace_is_judged_by_its_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two containers share a host name, and a pid of one names nothing in the other."""
    monkeypatch.setattr(liveness_module, "current_pid_namespace", lambda: "pid:[1]")
    attempt = _attempt(
        host=socket.gethostname(),
        pid=os.getpid(),
        pid_ns="pid:[2]",
        liveness_interval=60.0,
        heartbeat_ago=LIVENESS_STALE_INTERVALS * 60.0 + 1,
    )

    assert step_liveness(attempt, now=NOW) == "dead"


def test_a_local_attempt_without_a_pid_is_unknown() -> None:
    assert step_liveness(_here(0)) == "unknown"


def test_an_unknown_attempt_on_another_host_may_still_run_and_a_local_one_may_not() -> (
    None
):
    """Exactly the answers given before liveness was recorded."""
    assert may_still_run(_attempt(), now=NOW)
    assert not may_still_run(_here(0))


def test_an_attempt_without_any_readable_time_is_unknown() -> None:
    liveness = attempt_liveness(
        status="running",
        host=OTHER_HOST,
        pid=1,
        started_at="",
        heartbeat_at="",
        liveness_interval=60.0,
    )

    assert liveness == "unknown"


@pytest.mark.parametrize(
    ("recorded", "current", "differ"),
    [
        ("linux:a", "linux:b", True),
        ("linux:a", "linux:a", False),
        ("boot:1000", "boot:1100", False),
        ("boot:1000", "boot:5000", True),
        ("", "linux:b", False),
        ("linux:a", "boot:1000", False),
    ],
)
def test_boot_names_differ_only_when_they_prove_a_reboot(
    recorded: str, current: str, differ: bool
) -> None:
    assert boots_differ(recorded, current) is differ
