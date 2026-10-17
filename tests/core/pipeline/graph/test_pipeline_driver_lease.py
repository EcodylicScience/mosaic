"""The lease that a request's driver holds, and when it reads live."""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.pipeline import _utils as pipeline_utils
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.graph import (
    DRIVER_LEASE_GRACE_SECONDS,
    STALE_STEP_SECONDS,
    DriverRunning,
    LeaseKeeper,
    LeaseLost,
    LeaseNotRefreshed,
    RequestRollup,
    StepAttempt,
    driver_alive,
    driver_lease_path,
    lease_is_live,
    live_step_process,
    read_driver_lease,
    release_driver_lease,
    take_driver_lease,
)
from mosaic.core.pipeline.graph import driver_lease as driver_lease_module
from mosaic.core.pipeline.markers import parse_instant
from tests.helpers import exited_pid, wait_for, write_driver_lease


def test_a_request_without_a_lease_has_no_driver(tmp_path: Path) -> None:
    assert driver_alive(tmp_path, "R1") is None


def test_a_taken_lease_names_this_process_and_reads_live(tmp_path: Path) -> None:
    lease = take_driver_lease(tmp_path, "R1")

    assert (lease.request_id, lease.pid) == ("R1", os.getpid())
    assert read_driver_lease(tmp_path, "R1") == lease
    assert driver_alive(tmp_path, "R1") is True


def test_a_live_lease_refuses_a_second_driver_as_a_conflict(tmp_path: Path) -> None:
    held = take_driver_lease(tmp_path, "R1")

    with pytest.raises(DriverRunning) as raised:
        _ = take_driver_lease(tmp_path, "R1")

    envelope = error_envelope(raised.value)
    assert envelope.kind == "conflict"
    assert envelope.detail == {
        "request_id": "R1",
        "host": held.host,
        "pid": held.pid,
        "refreshed_at": held.refreshed_at,
    }


def test_a_lease_not_refreshed_within_the_grace_is_stale(tmp_path: Path) -> None:
    lease = write_driver_lease(tmp_path, "R1", stale=False)
    refreshed = parse_instant(lease.refreshed_at)
    assert refreshed is not None
    late = refreshed + datetime.timedelta(seconds=DRIVER_LEASE_GRACE_SECONDS + 1)

    assert lease_is_live(lease, now=refreshed)
    assert not lease_is_live(lease, now=late)


def test_a_lease_whose_process_is_gone_from_this_host_is_stale(
    tmp_path: Path,
) -> None:
    lease = write_driver_lease(tmp_path, "R1", stale=False)
    gone = lease.model_copy(update={"pid": exited_pid()})

    assert not lease_is_live(gone)


def test_a_stale_lease_is_taken_over(tmp_path: Path) -> None:
    _ = write_driver_lease(tmp_path, "R1", stale=True)
    assert driver_alive(tmp_path, "R1") is False

    lease = take_driver_lease(tmp_path, "R1")

    assert read_driver_lease(tmp_path, "R1") == lease
    assert driver_alive(tmp_path, "R1") is True


def test_a_lease_is_released_only_by_its_holder(tmp_path: Path) -> None:
    lease = take_driver_lease(tmp_path, "R1")
    other = lease.model_copy(update={"pid": lease.pid + 1})

    release_driver_lease(tmp_path, other)
    assert driver_lease_path(tmp_path, "R1").exists()

    release_driver_lease(tmp_path, lease)
    assert not driver_lease_path(tmp_path, "R1").exists()


def test_the_keeper_refreshes_the_lease_until_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver_lease_module, "DRIVER_LEASE_REFRESH_SECONDS", 0.05)
    lease = write_driver_lease(tmp_path, "R1", stale=True)
    keeper = LeaseKeeper(tmp_path, lease)

    keeper.start()
    wait_for(lambda: driver_alive(tmp_path, "R1") is True, 5.0, "never refreshed")
    keeper.stop()

    refreshed = read_driver_lease(tmp_path, "R1")
    assert refreshed is not None
    assert (refreshed.pid, refreshed.started_at) == (lease.pid, lease.started_at)


def test_the_keeper_reports_a_failed_refresh_and_refreshes_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(driver_lease_module, "DRIVER_LEASE_REFRESH_SECONDS", 0.05)
    write = pipeline_utils.atomic_write
    failures = [OSError("the mount went away")]

    def fail_once(path: Path, writer: Callable[[Path], object]) -> None:
        if failures:
            raise failures.pop()
        write(path, writer)

    monkeypatch.setattr(pipeline_utils, "atomic_write", fail_once)
    lease = write_driver_lease(tmp_path, "R1", stale=True)
    keeper = LeaseKeeper(tmp_path, lease)

    keeper.start()
    wait_for(lambda: driver_alive(tmp_path, "R1") is True, 5.0, "never refreshed")
    assert keeper.refreshing
    keeper.stop()

    assert failures == []
    assert "the mount went away" in capfd.readouterr().err


def test_a_driver_stops_once_its_lease_goes_unrefreshed_too_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver_lease_module, "DRIVER_LEASE_REFRESH_SECONDS", 0.05)
    monkeypatch.setattr(
        driver_lease_module, "DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS", 0.3
    )

    def always_fail(path: Path, writer: Callable[[Path], object]) -> None:
        raise OSError("the mount went away")

    monkeypatch.setattr(pipeline_utils, "atomic_write", always_fail)
    lease = take_driver_lease(tmp_path, "R1")
    keeper = LeaseKeeper(tmp_path, lease)
    keeper.start()

    def refused() -> bool:
        try:
            keeper.ensure_held()
        except LeaseNotRefreshed:
            return True
        return False

    wait_for(refused, 5.0, "the driver kept running on an unrefreshed lease")
    keeper.stop()
    with pytest.raises(LeaseNotRefreshed) as raised:
        keeper.ensure_held()
    envelope = error_envelope(raised.value)
    assert envelope.kind == "conflict"
    assert envelope.detail == {
        "request_id": "R1",
        "refreshed_at": lease.refreshed_at,
        "error": "OSError: the mount went away",
    }


def test_the_keeper_stops_once_another_driver_holds_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver_lease_module, "DRIVER_LEASE_REFRESH_SECONDS", 0.05)
    lease = take_driver_lease(tmp_path, "R1")
    keeper = LeaseKeeper(tmp_path, lease)
    other = lease.model_copy(update={"pid": lease.pid + 1})
    _ = driver_lease_path(tmp_path, "R1").write_text(
        other.model_dump_json(), encoding="utf-8"
    )

    keeper.start()
    wait_for(lambda: not keeper.refreshing, 5.0, "the keeper kept refreshing")

    assert read_driver_lease(tmp_path, "R1") == other
    keeper.stop()


def test_a_lease_taken_again_by_the_same_process_is_another_holder(
    tmp_path: Path,
) -> None:
    first = write_driver_lease(tmp_path, "R1", stale=True)
    second = take_driver_lease(tmp_path, "R1")

    assert second.pid == first.pid
    assert not second.same_holder(first)
    release_driver_lease(tmp_path, first)
    assert read_driver_lease(tmp_path, "R1") == second


def test_a_driver_learns_that_its_lease_was_taken_over_or_deleted(
    tmp_path: Path,
) -> None:
    lease = take_driver_lease(tmp_path, "R1")
    keeper = LeaseKeeper(tmp_path, lease)
    keeper.ensure_held()

    other = lease.model_copy(update={"pid": lease.pid + 1})
    _ = driver_lease_path(tmp_path, "R1").write_text(
        other.model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(DriverRunning) as raised:
        keeper.ensure_held()
    assert raised.value.lease == other

    driver_lease_path(tmp_path, "R1").unlink()
    with pytest.raises(LeaseLost) as lost:
        keeper.ensure_held()
    assert error_envelope(lost.value).kind == "conflict"


def test_the_lease_names_the_running_step_until_it_is_cleared(
    tmp_path: Path,
) -> None:
    lease = take_driver_lease(tmp_path, "R1")
    keeper = LeaseKeeper(tmp_path, lease)

    keeper.record_step("speed", 4242)
    named = read_driver_lease(tmp_path, "R1")
    keeper.record_step("", 0)
    cleared = read_driver_lease(tmp_path, "R1")

    assert named is not None and (named.step_id, named.step_pid) == ("speed", 4242)
    assert named.same_holder(lease)
    assert cleared is not None and (cleared.step_id, cleared.step_pid) == ("", 0)


def test_a_stalled_driver_does_not_overwrite_the_lease_that_replaced_its_own(
    tmp_path: Path,
) -> None:
    stalled = write_driver_lease(tmp_path, "R1", stale=True)
    keeper = LeaseKeeper(tmp_path, stalled)
    taken = take_driver_lease(tmp_path, "R1")

    keeper.record_step("speed", 4242)

    assert read_driver_lease(tmp_path, "R1") == taken


def test_a_stale_other_host_lease_naming_an_unstarted_step_is_dead_after_the_window(
    tmp_path: Path,
) -> None:
    """A step on another host that has not written its run-log yet."""
    lease = write_driver_lease(
        tmp_path,
        "R1",
        stale=True,
        host="another-host.invalid",
        step_id="speed",
        step_pid=7,
    )
    refreshed = parse_instant(lease.refreshed_at)
    assert refreshed is not None
    rollup = RequestRollup(
        request_id="R1",
        status="running",
        steps=(StepAttempt(step_id="speed", execution_id="E1"),),
    )
    within = refreshed + datetime.timedelta(seconds=STALE_STEP_SECONDS - 1)
    after = refreshed + datetime.timedelta(seconds=STALE_STEP_SECONDS + 1)

    running = live_step_process(rollup, lease, now=within)

    assert running is not None
    assert (running.step_id, running.pid, running.liveness) == (
        "speed",
        7,
        "unknown",
    )
    assert live_step_process(rollup, lease, now=after) is None
