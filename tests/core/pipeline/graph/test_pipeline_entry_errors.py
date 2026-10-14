"""The entities that one attempt lost, joined with their failure records."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from mosaic.core.pipeline.graph import (
    QUARANTINE_AFTER,
    FileFailureStore,
    entry_errors_wire,
)
from mosaic.core.pipeline.refusal import Refusal
from mosaic.runlog import JsonlRunLog, run_log_path

STORAGE = "speed-angvel__from__tracks"
RUN_ID = "0.2-abcdef0123"
LATER = datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc)


def _crash(message: str) -> str:
    return json.dumps({"type": "ValueError", "message": message, "traceback": "..."})


def _lose(base: Path, execution_id: str, *keys: str) -> None:
    """Write a finished attempt of the run that lost *keys*."""
    with JsonlRunLog(run_log_path(base, execution_id), execution_id) as log:
        log.started(kind="feature", target=STORAGE)
        log.set_run_id(RUN_ID)
        for key in keys:
            log.entry_failed(key, _crash(f"bad {key}"))
        log.finished()


def _fail(base: Path, group: str, sequence: str, times: int) -> None:
    store = FileFailureStore(base)
    for _ in range(times):
        _ = store.note_entry_failure((STORAGE, RUN_ID, group, sequence), error="bad")


def test_each_entity_reports_whether_a_request_attempts_it_again(
    tmp_path: Path,
) -> None:
    _lose(tmp_path, "E1", "g__fresh", "g__once", "g__often")
    _fail(tmp_path, "g", "once", 1)
    _fail(tmp_path, "g", "often", QUARANTINE_AFTER)

    now = entry_errors_wire(tmp_path, "E1")
    later = entry_errors_wire(tmp_path, "E1", now=LATER)

    assert now is not None
    assert later is not None
    assert [(e.sequence, e.attempts, e.retry) for e in now.entries] == [
        ("fresh", 0, "unrecorded"),
        ("once", 1, "waiting"),
        ("often", QUARANTINE_AFTER, "quarantined"),
    ]
    assert [e.retry for e in later.entries] == ["unrecorded", "ready", "quarantined"]
    assert now.held_back_by_quarantine


def test_a_waiting_entity_does_not_hold_the_step_back(
    tmp_path: Path,
) -> None:
    _lose(tmp_path, "E1", "g__once")
    _fail(tmp_path, "g", "once", 1)

    report = entry_errors_wire(tmp_path, "E1")

    assert report is not None
    assert report.entries[0].retry == "waiting"
    assert not report.held_back_by_quarantine


def test_an_unlisted_quarantined_entity_holds_the_step_back(
    tmp_path: Path,
) -> None:
    _lose(tmp_path, "E1", "g__once", "g__often")
    _fail(tmp_path, "g", "once", 1)
    _fail(tmp_path, "g", "often", QUARANTINE_AFTER)

    report = entry_errors_wire(tmp_path, "E1", limit=1)

    assert report is not None
    assert [e.retry for e in report.entries] == ["waiting"]
    assert report.has_more
    assert report.held_back_by_quarantine


def test_the_entities_are_parsed_and_sampled_up_to_the_limit(tmp_path: Path) -> None:
    keys = [f"g%2Fx__s{index}" for index in range(5)]
    _lose(tmp_path, "E1", *keys)

    report = entry_errors_wire(tmp_path, "E1", limit=2)

    assert report is not None
    assert (report.total, report.has_more, report.outcome) == (5, True, "partial")
    assert [e.key for e in report.entries] == keys[:2]
    first = report.entries[0]
    assert (first.group, first.sequence) == ("g/x", "s0")
    assert first.error is not None
    assert (first.error.type, first.error.message) == ("ValueError", "bad g%2Fx__s0")
    assert (report.execution_id, report.target, report.run_id) == (
        "E1",
        STORAGE,
        RUN_ID,
    )


def test_a_refused_entity_reports_its_refusal(tmp_path: Path) -> None:
    refusal = Refusal("codec av1", reason="undecodable_codec", step_id="track")
    with JsonlRunLog(run_log_path(tmp_path, "E1"), "E1") as log:
        log.started(kind="feature", target=STORAGE)
        log.entry_failed("s1", refusal.error_json())
        log.finished()

    report = entry_errors_wire(tmp_path, "E1")

    assert report is not None
    (entry,) = report.entries
    assert entry.error is not None
    assert entry.error.refusal is not None
    assert (entry.error.refusal.reason, entry.error.message) == (
        "undecodable_codec",
        "codec av1",
    )


def test_an_attempt_that_lost_nothing_lists_nothing(tmp_path: Path) -> None:
    _lose(tmp_path, "E1")

    report = entry_errors_wire(tmp_path, "E1")

    assert report is not None
    assert (report.total, report.has_more, report.entries) == (0, False, ())
    assert report.outcome == "finished"


def test_an_absent_run_log_reports_nothing(tmp_path: Path) -> None:
    assert entry_errors_wire(tmp_path, "MISSING") is None


def test_a_limit_below_one_is_refused(tmp_path: Path) -> None:
    _lose(tmp_path, "E1", "s1")

    with pytest.raises(ValueError, match="at least 1"):
        _ = entry_errors_wire(tmp_path, "E1", limit=0)
