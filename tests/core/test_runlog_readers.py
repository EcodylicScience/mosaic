"""The run-log readers that status reports and listings are built from, and the outcome rule."""

from __future__ import annotations

import builtins
import errno
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import mosaic.runlog as runlog
from mosaic.runlog import (
    JsonlRunLog,
    RunLogSnapshot,
    attempt_outcome,
    error_summary,
    list_runs,
    read_entry_error_keys,
    read_entry_errors,
    read_run_progress,
    read_runs,
    run_log_dir,
    run_log_path,
)
from tests.helpers import wait_for


def _log(base: Path, execution_id: str) -> JsonlRunLog:
    return JsonlRunLog(run_log_path(base, execution_id), execution_id)


@pytest.mark.parametrize(
    ("status", "entries_failed", "outcome"),
    [
        ("", 0, "pending"),
        ("running", 3, "running"),
        ("finished", 0, "finished"),
        ("finished", 2, "partial"),
        ("failed", 2, "failed"),
        ("cancelled", 0, "cancelled"),
        ("unheard-of", 0, "running"),
    ],
)
def test_the_outcome_names_a_finished_attempt_that_lost_entries_partial(
    status: str, entries_failed: int, outcome: str
) -> None:
    assert attempt_outcome(status, entries_failed) == outcome


def test_entry_errors_are_read_in_order_up_to_the_limit_and_counted(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        for index in range(5):
            log.entry_failed(
                f"g__s{index}", f'{{"type": "ValueError", "message": "boom {index}"}}'
            )
        log.finished()

    every = read_entry_errors(run_log_dir(tmp_path), "E1")
    first = read_entry_errors(run_log_dir(tmp_path), "E1", limit=2)

    assert [record["key"] for record in every["items"]] == [
        f"g__s{i}" for i in range(5)
    ]
    assert (every["total"], every["has_more"]) == (5, False)
    assert first["items"] == every["items"][:2]
    assert (first["total"], first["has_more"]) == (5, True)
    record = first["items"][0]
    assert record["error_json"] == '{"type": "ValueError", "message": "boom 0"}'
    assert (record["error_type"], record["message"]) == ("ValueError", "boom 0")
    assert record["recorded_at"]


def test_entry_error_keys_are_read_in_order_with_their_blobs_unparsed(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.entry_failed("g__s0", "not json")
        log.entry_failed("g__s1", '{"type": "ValueError"}')
        log.finished()

    assert read_entry_error_keys(run_log_dir(tmp_path), "E1") == ["g__s0", "g__s1"]
    assert read_entry_error_keys(run_log_dir(tmp_path), "MISSING") == []


def test_a_resumed_attempt_reports_what_its_latest_process_lost(
    tmp_path: Path,
) -> None:
    """A second ``started`` resumes an attempt whose process died."""
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.entry_failed("g__s0", '{"type": "ValueError", "message": "dead"}')
        log.entry_failed("g__s1", '{"type": "ValueError", "message": "dead"}')
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        log.entry_failed("g__s1", '{"type": "ValueError", "message": "again"}')
        log.finished()

    snapshot = runlog.read_run(run_log_dir(tmp_path), "E1")
    errors = read_entry_errors(run_log_dir(tmp_path), "E1")

    assert snapshot is not None
    assert (snapshot["status"], snapshot["entries_failed"]) == ("finished", 1)
    assert (errors["total"], [item["message"] for item in errors["items"]]) == (
        1,
        ["again"],
    )
    assert read_entry_error_keys(run_log_dir(tmp_path), "E1") == ["g__s1"]


def test_entry_errors_of_an_absent_run_log_are_empty(tmp_path: Path) -> None:
    assert read_entry_errors(run_log_dir(tmp_path), "MISSING") == {
        "items": [],
        "total": 0,
        "has_more": False,
    }


def test_an_entry_error_limit_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _ = read_entry_errors(run_log_dir(tmp_path), "E1", limit=0)


@pytest.mark.parametrize(
    ("error_json", "error_type", "message"),
    [
        ("", "", ""),
        ('{"type": "OSError", "message": "gone"}', "OSError", "gone"),
        ('{"reason": "upstream_empty", "message": "nothing"}', "", "nothing"),
        ('{"type": 3, "message": ["x"]}', "", ""),
        ("not json", "", "not json"),
        ('["a list"]', "", '["a list"]'),
    ],
)
def test_an_error_summary_reads_the_type_and_message_of_a_blob(
    error_json: str, error_type: str, message: str
) -> None:
    assert error_summary(error_json) == {"type": error_type, "message": message}


def test_an_error_summary_quotes_at_most_the_message_limit() -> None:
    long = "x" * (runlog.ERROR_MESSAGE_LIMIT + 10)

    assert error_summary(long)["message"] == long[: runlog.ERROR_MESSAGE_LIMIT]


def test_progress_events_keep_their_legacy_shape(tmp_path: Path) -> None:
    with _log(tmp_path, "E2") as log:
        log.started(kind="train-pose", target="pose")
        log.on_epoch_end(0, 3, {"loss": 0.5})
        log.on_phase("export", "writing weights")

    events = read_run_progress(run_log_dir(tmp_path), "E2")

    assert [event["step_type"] for event in events] == ["epoch", "phase"]
    assert events[0]["metrics"] == {"loss": 0.5}
    assert events[0]["step_total"] == 3
    assert events[1]["message"] == "export: writing weights"


def test_a_progress_field_of_the_wrong_type_reads_as_its_empty_value(
    tmp_path: Path,
) -> None:
    lines = [
        {"t": "T1", "ev": "entry_end", "index": None, "total": "4", "key": 7},
        {"t": "T2", "ev": "epoch", "epoch": True, "metrics": {"loss": "x", "acc": 1}},
        {"t": "T3", "ev": "phase", "phase": None, "message": "done"},
    ]
    path = run_log_path(tmp_path, "E3")
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    events = read_run_progress(run_log_dir(tmp_path), "E3")

    assert [(e["step_index"], e["step_total"], e["message"]) for e in events] == [
        (0, 0, ""),
        (0, 0, ""),
        (0, 0, ": done"),
    ]
    assert events[1]["metrics"] == {"acc": 1.0}


RUN_IDS = ("01A", "01B", "01C", "01D", "01E")
"""Execution ids in creation order, as ULIDs sort."""


def _finished_run(base: Path, execution_id: str, *, kind: str = "feature") -> None:
    with _log(base, execution_id) as log:
        log.started(kind=kind, target="speed")
        log.finished()


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    for execution_id in RUN_IDS:
        _finished_run(tmp_path, execution_id)
    return run_log_dir(tmp_path)


def _page_ids(run_dir: Path, *, limit: int, after: str | None = None) -> list[str]:
    return [
        snap["execution_id"]
        for snap in list_runs(run_dir, limit=limit, after=after)["items"]
    ]


def test_runs_are_listed_newest_first_in_pages(runs: Path) -> None:
    first = list_runs(runs, limit=2)
    second = list_runs(runs, limit=2, after=first["next_after"])
    third = list_runs(runs, limit=2, after=second["next_after"])

    assert [snap["execution_id"] for snap in first["items"]] == ["01E", "01D"]
    assert first["next_after"] == "01D"
    assert [snap["execution_id"] for snap in second["items"]] == ["01C", "01B"]
    assert [snap["execution_id"] for snap in third["items"]] == ["01A"]
    assert third["next_after"] is None


def test_an_unlimited_listing_is_every_run(runs: Path) -> None:
    page = list_runs(runs)

    assert page["items"] == read_runs(runs)
    assert [snap["execution_id"] for snap in page["items"]] == list(reversed(RUN_IDS))
    assert page["next_after"] is None


def test_a_run_started_between_pages_leaves_the_next_page_unchanged(
    runs: Path,
) -> None:
    first = list_runs(runs, limit=2)
    expected = _page_ids(runs, limit=2, after=first["next_after"])

    _finished_run(runs.parent.parent, "01F")

    assert _page_ids(runs, limit=2, after=first["next_after"]) == expected
    assert _page_ids(runs, limit=1) == ["01F"]


def test_only_the_run_logs_on_the_page_are_reduced(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduced: list[str] = []
    reduce = runlog.reduce_run_log

    def counting_reduce(path: Path) -> RunLogSnapshot | None:
        reduced.append(path.stem)
        return reduce(path)

    monkeypatch.setattr(runlog, "reduce_run_log", counting_reduce)

    _ = list_runs(runs, limit=2, after="01D")

    assert reduced == ["01C", "01B"]


def test_a_filter_fills_the_page_with_matching_runs(tmp_path: Path) -> None:
    for execution_id in RUN_IDS:
        kind = "trex" if execution_id in {"01B", "01D"} else "feature"
        _finished_run(tmp_path, execution_id, kind=kind)
    run_dir = run_log_dir(tmp_path)

    first = list_runs(run_dir, limit=2, kind="feature")
    second = list_runs(run_dir, limit=2, kind="feature", after=first["next_after"])

    assert [snap["execution_id"] for snap in first["items"]] == ["01E", "01C"]
    assert first["next_after"] == "01C"
    assert [snap["execution_id"] for snap in second["items"]] == ["01A"]
    assert second["next_after"] is None
    assert [
        s["execution_id"] for s in list_runs(run_dir, target="other")["items"]
    ] == []


def test_an_empty_run_log_is_skipped_and_not_counted(runs: Path) -> None:
    (runs / "01F.jsonl").write_text("")

    assert _page_ids(runs, limit=2) == ["01E", "01D"]


def test_runs_of_an_absent_directory_are_empty(tmp_path: Path) -> None:
    assert list_runs(run_log_dir(tmp_path), limit=3) == {
        "items": [],
        "next_after": None,
    }


def test_a_run_limit_below_one_is_refused(runs: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list_runs(runs, limit=0)


def _events(base: Path, execution_id: str) -> list[str]:
    path = run_log_path(base, execution_id)
    return [json.loads(line)["ev"] for line in path.read_text().splitlines()]


def _write_lines(base: Path, execution_id: str, *records: dict[str, object]) -> Path:
    path = run_log_path(base, execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_a_silent_attempt_appends_an_alive_event_each_interval(
    tmp_path: Path,
) -> None:
    log = _log(tmp_path, "E1")
    try:
        log.started(kind="feature", target="t", liveness_interval=0.05)
        log.start_liveness(0.05)
        wait_for(
            lambda: _events(tmp_path, "E1").count("alive") >= 2,
            5.0,
            "the liveness thread wrote fewer than two alive events",
        )
    finally:
        log.close()


def test_an_attempt_that_writes_events_appends_no_alive_event(tmp_path: Path) -> None:
    log = _log(tmp_path, "E1")
    try:
        log.started(kind="feature", target="t")
        log.start_liveness(30.0)
        for done in range(5):
            log.heartbeat(done, 5)
        log.finished()
    finally:
        log.close()

    assert "alive" not in _events(tmp_path, "E1")


def test_no_alive_event_follows_the_terminal_event(tmp_path: Path) -> None:
    log = _log(tmp_path, "E1")
    try:
        log.started(kind="feature", target="t")
        log.start_liveness(0.01)
        wait_for(lambda: "alive" in _events(tmp_path, "E1"), 5.0)
        log.finished()
        time.sleep(0.1)
        assert _events(tmp_path, "E1")[-1] == "finished"
    finally:
        log.close()


def test_closing_the_log_stops_its_liveness_thread(tmp_path: Path) -> None:
    log = _log(tmp_path, "E1")
    log.started(kind="feature", target="t")
    log.start_liveness(0.01)

    log.close()

    assert not any(thread.name == "runlog-alive-E1" for thread in threading.enumerate())
    written = _events(tmp_path, "E1")
    time.sleep(0.05)
    assert _events(tmp_path, "E1") == written


class _TearsOneLine:
    """A run-log file whose first write of one event lands half a line and raises."""

    def __init__(self, file: Any, event: str) -> None:
        self._file = file
        self._event = f'"ev": "{event}"'
        self.tears = 0

    def write(self, text: str) -> int:
        if self.tears == 0 and self._event in text:
            self.tears += 1
            _ = self._file.write(text[: len(text) // 2])
            self._file.flush()
            raise OSError(errno.ESTALE, "Stale file handle")
        return self._file.write(text)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _tearing_log(
    base: Path, monkeypatch: pytest.MonkeyPatch, event: str
) -> tuple[JsonlRunLog, _TearsOneLine]:
    files: list[_TearsOneLine] = []

    def tearing_open(*args: Any, **kwargs: Any) -> _TearsOneLine:
        files.append(_TearsOneLine(builtins.open(*args, **kwargs), event))
        return files[-1]

    monkeypatch.setattr(runlog, "open", tearing_open, raising=False)
    return _log(base, "E1"), files[0]


def _decoded_events(base: Path, execution_id: str) -> list[str]:
    events: list[str] = []
    for line in run_log_path(base, execution_id).read_text().splitlines():
        try:
            events.append(json.loads(line)["ev"])
        except json.JSONDecodeError:
            continue
    return events


def test_a_failed_alive_write_is_retried_after_the_next_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log, file = _tearing_log(tmp_path, monkeypatch, "alive")
    try:
        log.started(kind="feature", target="t", liveness_interval=0.02)
        log.start_liveness(0.02)
        wait_for(
            lambda: "alive" in _decoded_events(tmp_path, "E1"),
            5.0,
            "the liveness thread wrote no alive event after a failed write",
        )
    finally:
        log.close()

    assert file.tears == 1


def test_the_line_after_a_partial_line_still_reduces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log, file = _tearing_log(tmp_path, monkeypatch, "heartbeat")
    try:
        log.started(kind="feature", target="t")
        with pytest.raises(OSError, match="Stale file handle"):
            log.heartbeat(1, 2)
        log.finished()
    finally:
        log.close()

    snap = runlog.reduce_run_log(run_log_path(tmp_path, "E1"))
    assert file.tears == 1
    assert _decoded_events(tmp_path, "E1") == ["started", "finished"]
    assert snap is not None
    assert snap["status"] == "finished"


def test_an_exit_record_after_a_partial_last_line_still_reduces(
    tmp_path: Path,
) -> None:
    path = _write_lines(tmp_path, "E1", {"t": "2026-01-01T00:00:00Z", "ev": "started"})
    with path.open("a") as file:
        _ = file.write('{"t": "2026-01-01T00:00:01Z", "ev": "fini')

    runlog.record_exited_attempt(run_log_dir(tmp_path), "E1", "cancelled")

    snap = runlog.reduce_run_log(path)
    assert _decoded_events(tmp_path, "E1") == ["started", "cancelled"]
    assert snap is not None
    assert snap["status"] == "cancelled"


def test_a_log_that_ends_on_a_full_line_gains_no_blank_line(tmp_path: Path) -> None:
    path = _write_lines(tmp_path, "E1", {"t": "2026-01-01T00:00:00Z", "ev": "started"})

    runlog.record_exited_attempt(run_log_dir(tmp_path), "E1", "cancelled")

    assert "" not in path.read_text().splitlines()


def test_an_alive_event_only_advances_heartbeat_at(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path,
        "E1",
        {"t": "2026-01-01T00:00:00+00:00", "ev": "started", "kind": "feature"},
        {"t": "2026-01-01T00:00:01+00:00", "ev": "heartbeat", "done": 2, "total": 4},
        {"t": "2026-01-01T00:01:01+00:00", "ev": "alive"},
    )

    snap = runlog.reduce_run_log(path)

    assert snap is not None
    assert snap["heartbeat_at"] == "2026-01-01T00:01:01+00:00"
    assert (snap["status"], snap["progress_done"], snap["progress_total"]) == (
        "running",
        2,
        4,
    )


def test_the_started_event_records_boot_namespace_and_interval(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(
            kind="feature",
            target="t",
            boot_id="linux:abc",
            pid_ns="pid:[4026531836]",
            liveness_interval=60.0,
        )

    snap = runlog.read_run(run_log_dir(tmp_path), "E1")

    assert snap is not None
    assert (snap["boot_id"], snap["pid_ns"], snap["liveness_interval"]) == (
        "linux:abc",
        "pid:[4026531836]",
        60.0,
    )


def test_a_later_started_without_liveness_clears_it(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path,
        "E1",
        {
            "t": "2026-01-01T00:00:00+00:00",
            "ev": "started",
            "boot_id": "linux:abc",
            "pid_ns": "pid:[1]",
            "liveness_interval": 60.0,
        },
        {"t": "2026-01-01T00:05:00+00:00", "ev": "started", "host": "h"},
    )

    snap = runlog.reduce_run_log(path)

    assert snap is not None
    assert (snap["boot_id"], snap["pid_ns"], snap["liveness_interval"]) == (
        "",
        "",
        0.0,
    )


def test_phase_is_the_last_declared_phase_since_started(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.on_phase("fit", "3 entries")
        log.on_phase("apply", "")
    resumed = run_log_path(tmp_path, "E1")

    snap = runlog.reduce_run_log(resumed)
    assert snap is not None
    assert snap["phase"] == "apply"

    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
    snap = runlog.reduce_run_log(resumed)
    assert snap is not None
    assert snap["phase"] == ""


def _removal_log(base: Path) -> JsonlRunLog:
    log = _log(base, "C1")
    log.started(kind="feature-cleanup", target="R1")
    log.set_total(4)
    log.run_removing(
        feature="speed",
        run_id="0.1-a",
        reason="superseded",
        superseded_by="0.1-z",
        params={"_params": {"step_size": 1}},
    )
    log.run_removed(feature="speed", run_id="0.1-a", bytes=10, files=2, rows=1)
    log.run_removing(
        feature="speed",
        run_id="0.1-b",
        reason="superseded",
        superseded_by="0.1-z",
        params={"_params": {"step_size": 2}},
    )
    log.run_removing(
        feature="speed",
        run_id="0.1-c",
        reason="superseded",
        superseded_by="0.1-z",
        params={},
    )
    log.entry_failed("speed/0.1-c", '{"type": "OSError", "message": "busy"}')
    log.run_kept(feature="speed", run_id="0.1-d", reason="read_by", detail="read")
    return log


def test_removal_events_do_not_change_the_status(tmp_path: Path) -> None:
    with _removal_log(tmp_path) as log:
        running = runlog.read_run(run_log_dir(tmp_path), "C1")
        log.finished()
    finished = runlog.read_run(run_log_dir(tmp_path), "C1")

    assert running is not None
    assert finished is not None
    assert running["status"] == "running"
    assert finished["status"] == "finished"
    assert finished["entries_failed"] == 1
    assert "run_removed" not in runlog.TERMINAL_STATUSES


def test_an_intent_without_a_done_reads_as_interrupted(tmp_path: Path) -> None:
    with _removal_log(tmp_path):
        pass

    records = runlog.read_removal_records(run_log_dir(tmp_path), "C1")

    assert [(record["run_id"], record["state"]) for record in records] == [
        ("0.1-a", "removed"),
        ("0.1-b", "interrupted"),
        ("0.1-c", "failed"),
        ("0.1-d", "kept"),
    ]
    assert records[0]["bytes"] == 10
    assert records[1]["params"] == {"_params": {"step_size": 2}}
    assert (records[3]["reason"], records[3]["detail"]) == ("read_by", "read")


def test_a_resumed_cleanup_never_undoes_a_removal(tmp_path: Path) -> None:
    with _removal_log(tmp_path) as log:
        log.started(kind="feature-cleanup", target="R1")
        log.run_kept(feature="speed", run_id="0.1-a", reason="absent", detail="")
        log.run_removed(feature="speed", run_id="0.1-b", bytes=4, files=1, rows=0)

    records = runlog.read_removal_records(run_log_dir(tmp_path), "C1")

    assert [(record["run_id"], record["state"]) for record in records][:2] == [
        ("0.1-a", "removed"),
        ("0.1-b", "removed"),
    ]


def test_removal_records_of_an_absent_log_are_empty(tmp_path: Path) -> None:
    assert runlog.read_removal_records(run_log_dir(tmp_path), "missing") == []


def test_a_retired_removal_keeps_its_digest_through_the_done_event(
    tmp_path: Path,
) -> None:
    """The digest is on the intent, which ``run_removed`` does not repeat."""
    with _log(tmp_path, "C2") as log:
        log.started(kind="feature-cleanup", target="R1")
        log.run_removing(
            feature="speed",
            run_id="0.1-a",
            reason="retired",
            superseded_by="",
            params={},
            retired="51b2573baa1b4a84",
        )
        interrupted = runlog.read_removal_records(run_log_dir(tmp_path), "C2")
        log.run_removed(feature="speed", run_id="0.1-a", bytes=10, files=2, rows=1)

    [record] = runlog.read_removal_records(run_log_dir(tmp_path), "C2")

    assert interrupted[0]["state"] == "interrupted"
    assert interrupted[0]["retired"] == "51b2573baa1b4a84"
    assert (record["state"], record["reason"]) == ("removed", "retired")
    assert record["retired"] == "51b2573baa1b4a84"


def test_a_removal_without_a_retirement_writes_the_line_an_older_mosaic_wrote(
    tmp_path: Path,
) -> None:
    with _removal_log(tmp_path):
        pass

    lines = (run_log_dir(tmp_path) / "C1.jsonl").read_text().splitlines()
    removing = [json.loads(line) for line in lines if '"run_removing"' in line]

    assert removing
    assert all("retired" not in event for event in removing)
    records = runlog.read_removal_records(run_log_dir(tmp_path), "C1")
    assert {record["retired"] for record in records} == {""}
