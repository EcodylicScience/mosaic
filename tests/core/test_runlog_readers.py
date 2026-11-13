"""The run-log readers that status reports and listings are built from, and the outcome rule."""

from __future__ import annotations

import builtins
import dataclasses
import datetime
import errno
import os
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import mosaic.runlog as runlog
from mosaic.runlog import (
    EntryMeasure,
    JsonlRunLog,
    ResourceReport,
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


NUL_BLOCK = "\x00" * 700


def _write_with_nul_block_before(
    base: Path, execution_id: str, glued_index: int, *records: dict[str, object]
) -> Path:
    """Write the records, gluing a block of NUL bytes to the front of one line."""
    path = run_log_path(base, execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) + "\n" for record in records]
    lines[glued_index] = NUL_BLOCK + lines[glued_index]
    path.write_text("".join(lines))
    return path


def test_a_record_glued_behind_a_nul_block_is_still_read(tmp_path: Path) -> None:
    _write_with_nul_block_before(
        tmp_path,
        "E1",
        1,
        {"t": "2026-01-01T00:00:00Z", "ev": "started"},
        {"t": "2026-01-01T00:00:01Z", "ev": "resources", "workers": 2},
        {"t": "2026-01-01T00:00:02Z", "ev": "heartbeat"},
    )

    records = runlog.read_run_records(tmp_path / ".mosaic" / "runs", "E1")

    assert [record["ev"] for record in records] == ["started", "resources", "heartbeat"]
    assert records[1]["workers"] == 2


def test_a_finished_record_behind_a_nul_block_still_reduces_as_finished(
    tmp_path: Path,
) -> None:
    path = _write_with_nul_block_before(
        tmp_path,
        "E1",
        2,
        {"t": "2026-01-01T00:00:00Z", "ev": "started"},
        {"t": "2026-01-01T00:00:01Z", "ev": "heartbeat"},
        {"t": "2026-01-01T00:00:02Z", "ev": "finished"},
    )

    records = runlog.read_run_records(path.parent, "E1")
    snap = runlog.reduce_run_log(path)

    assert [record["ev"] for record in records] == ["started", "heartbeat", "finished"]
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


def test_the_started_event_records_the_leased_card(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t", gpu_card="1:GPU-abc")

    snap = runlog.read_run(run_log_dir(tmp_path), "E1")

    assert snap is not None
    assert snap["gpu_card"] == "1:GPU-abc"


def test_a_run_log_without_a_card_reads_an_empty_one(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
    path = run_log_path(tmp_path, "E1")

    snap = runlog.read_run(run_log_dir(tmp_path), "E1")

    assert snap is not None
    assert snap["gpu_card"] == ""
    assert "gpu_card" not in path.read_text(encoding="utf-8")


def test_a_resumed_attempt_takes_the_card_of_its_last_started(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path,
        "E1",
        {"t": "2026-01-01T00:00:00+00:00", "ev": "started", "gpu_card": "0:GPU-a"},
        {"t": "2026-01-01T00:05:00+00:00", "ev": "started", "gpu_card": "1:GPU-b"},
    )
    resumed_elsewhere = _write_lines(
        tmp_path,
        "E2",
        {"t": "2026-01-01T00:00:00+00:00", "ev": "started", "gpu_card": "0:GPU-a"},
        {"t": "2026-01-01T00:05:00+00:00", "ev": "started", "host": "h"},
    )

    snap = runlog.reduce_run_log(path)
    cleared = runlog.reduce_run_log(resumed_elsewhere)

    assert snap is not None and cleared is not None
    assert (snap["gpu_card"], cleared["gpu_card"]) == ("1:GPU-b", "")


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


def test_runs_changed_since_an_instant_are_those_that_began_or_ended_after_it(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="a")
        log.finished()
    with _log(tmp_path, "E2") as log:
        log.started(kind="feature", target="b")
    with _log(tmp_path, "E4") as log:
        log.started(kind="feature", target="d")
    since = datetime.datetime.now(datetime.UTC)
    with _log(tmp_path, "E2") as log:
        log.finished()
    with _log(tmp_path, "E3") as log:
        log.started(kind="feature", target="c")
    with _log(tmp_path, "E4") as log:
        log.heartbeat(1, 2)

    changed = list_runs(run_log_dir(tmp_path), changed_since=since)
    paged = list_runs(run_log_dir(tmp_path), changed_since=since, limit=1)

    assert [snap["execution_id"] for snap in changed["items"]] == ["E3", "E2"]
    assert [snap["execution_id"] for snap in paged["items"]] == ["E3"]
    assert paged["next_after"] == "E3"


def test_a_log_whose_mtime_reads_just_before_the_instant_is_still_read(
    tmp_path: Path,
) -> None:
    """A coarse or skewed file clock is not trusted to the second; the stamps decide."""
    since = datetime.datetime.now(datetime.UTC)
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="a")
    early = since.timestamp() - 1
    os.utime(run_log_path(tmp_path, "E1"), (early, early))

    changed = list_runs(run_log_dir(tmp_path), changed_since=since)

    assert [snap["execution_id"] for snap in changed["items"]] == ["E1"]


def _events_of(base: Path, execution_id: str, kind: str) -> list[dict[str, Any]]:
    path = run_log_path(base, execution_id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return [record for record in records if record["ev"] == kind]


def _full_report() -> ResourceReport:
    return ResourceReport(
        workers=4,
        parallel_mode="thread",
        peak_rss_bytes=300,
        min_available_bytes=500,
        total_memory_bytes=1000,
        feature="speed",
        feature_version="0.3",
        process_capable=True,
        cpu_seconds=12.5,
        wall_seconds=4.25,
        memory_limit_bytes=800,
        memory_limit_source="cgroup_v2",
        limit_peak_bytes=250,
        base_rss_bytes=40,
        fit_seconds=1.5,
        fit_peak_rss_bytes=120,
        peak_bytes_per_row=7,
        admission_margin=1.5,
        admission_waits=2,
        fit_entries=3,
        fit_input_rows=900,
        worker_base_rss_bytes=60,
        apply_wall_seconds=2.75,
    )


def test_a_resources_event_carries_every_field_of_its_report(tmp_path: Path) -> None:
    report = _full_report()
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.resources(report)

    (event,) = _events_of(tmp_path, "E1", "resources")

    assert {k: v for k, v in event.items() if k not in ("t", "ev")} == (
        dataclasses.asdict(report)
    )


def test_a_report_of_only_its_required_fields_reads_every_other_as_empty() -> None:
    report = ResourceReport(
        workers=1,
        parallel_mode="",
        peak_rss_bytes=1,
        min_available_bytes=2,
        total_memory_bytes=3,
    )

    assert (report.feature, report.feature_version, report.process_capable) == (
        "",
        "",
        False,
    )
    assert (report.cpu_seconds, report.wall_seconds, report.apply_wall_seconds) == (
        0.0,
        0.0,
        0.0,
    )
    assert (report.memory_limit_bytes, report.memory_limit_source) == (0, "")
    assert (report.admission_margin, report.admission_waits) == (0.0, 0)


def test_an_entry_end_carries_its_measure_only_when_given(tmp_path: Path) -> None:
    measure = EntryMeasure(
        wall_seconds=2.5, apply_seconds=1.5, input_rows=40, in_flight=2, cpu_seconds=3.0
    )
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.on_entry_end(1, 2, "a__1", measure=measure)
        log.on_entry_end(2, 2, "a__2")

    measured, bare = _events_of(tmp_path, "E1", "entry_end")

    assert {k: v for k, v in measured.items() if k not in ("t", "ev")} == {
        "index": 1,
        "total": 2,
        "key": "a__1",
        "wall_seconds": 2.5,
        "apply_seconds": 1.5,
        "input_rows": 40,
        "in_flight": 2,
        "cpu_seconds": 3.0,
    }
    assert {k: v for k, v in bare.items() if k not in ("t", "ev")} == {
        "index": 2,
        "total": 2,
        "key": "a__2",
    }


def test_a_probe_event_names_its_request_step_entries_and_kind(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.probe(request_id="R1", step="speed", entries=["g__a", "g__b"], kind="named")

    (event,) = _events_of(tmp_path, "E1", "probe")

    assert (event["request_id"], event["step"]) == ("R1", "speed")
    assert (event["entries"], event["kind"]) == (["g__a", "g__b"], "named")


def _comparable(snapshot: RunLogSnapshot | None) -> dict[str, Any]:
    assert snapshot is not None
    return {
        k: v
        for k, v in snapshot.items()
        if k
        not in (
            "execution_id",
            "created_at",
            "started_at",
            "heartbeat_at",
            "finished_at",
        )
    }


def test_a_log_with_the_new_fields_reduces_to_the_snapshot_of_one_without(
    tmp_path: Path,
) -> None:
    measure = EntryMeasure(
        wall_seconds=2.5, apply_seconds=1.5, input_rows=40, in_flight=2, cpu_seconds=3.0
    )
    report = _full_report()
    with _log(tmp_path, "NEW") as log:
        log.started(kind="feature", target="t")
        log.set_total(2)
        log.resources(report)
        log.on_entry_end(1, 2, "a__1", measure=measure)
        log.on_entry_end(2, 2, "a__2", measure=measure)
        log.finished()
    old_lines: list[dict[str, object]] = [
        {"ev": "started", "kind": "feature", "target": "t"},
        {"ev": "total", "total": 2},
        {
            "ev": "resources",
            "workers": 4,
            "parallel_mode": "thread",
            "peak_rss_bytes": 300,
            "min_available_bytes": 500,
            "total_memory_bytes": 1000,
        },
        {"ev": "entry_end", "index": 1, "total": 2, "key": "a__1"},
        {"ev": "entry_end", "index": 2, "total": 2, "key": "a__2"},
        {"ev": "finished"},
    ]
    old = _write_lines(tmp_path, "OLD", *old_lines)

    new_snapshot = _comparable(runlog.reduce_run_log(run_log_path(tmp_path, "NEW")))
    old_snapshot = _comparable(runlog.reduce_run_log(old))

    assert new_snapshot == old_snapshot
    assert old_snapshot["probe"] is False
    assert (old_snapshot["workers"], old_snapshot["peak_rss_bytes"]) == (4, 300)
    assert (old_snapshot["progress_done"], old_snapshot["progress_total"]) == (2, 2)


def test_a_probe_event_of_the_probe_kind_marks_the_attempt(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.probe(request_id="R1", step="speed", entries=["g__a"], kind="probe")

    snap = runlog.reduce_run_log(run_log_path(tmp_path, "E1"))

    assert snap is not None
    assert snap["probe"] is True


def test_a_named_subset_does_not_mark_the_attempt_a_probe(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.probe(request_id="R1", step="speed", entries=["g__a"], kind="named")

    snap = runlog.reduce_run_log(run_log_path(tmp_path, "E1"))

    assert snap is not None
    assert snap["probe"] is False


def test_a_later_started_clears_the_probe_mark(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.probe(request_id="R1", step="speed", entries=["g__a"], kind="probe")
        log.started(kind="feature", target="t")

    snap = runlog.reduce_run_log(run_log_path(tmp_path, "E1"))

    assert snap is not None
    assert snap["probe"] is False


def test_folding_the_records_of_a_log_gives_the_snapshot_read_run_reads(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="t")
        log.set_total(2)
        log.on_entry_end(1, 2, "a__1")
    path = run_log_path(tmp_path, "E1")
    path.write_text(path.read_text() + '{"ev": "entry_en')

    records = runlog.read_run_records(run_log_dir(tmp_path), "E1")

    assert [record["ev"] for record in records] == ["started", "total", "entry_end"]
    assert runlog.fold_run_records("E1", records) == runlog.read_run(
        run_log_dir(tmp_path), "E1"
    )
    assert runlog.fold_run_records("E1", []) is None
    assert runlog.read_run_records(run_log_dir(tmp_path), "ABSENT") == []


def test_a_record_field_of_the_wrong_type_reads_as_empty() -> None:
    record = {"n": True, "s": 3, "x": "1.5", "f": 1, "g": "yes"}

    assert runlog.record_count(record, "n") == 0
    assert runlog.record_text(record, "s") == ""
    assert runlog.record_seconds(record, "x") == 0.0
    assert runlog.record_flag(record, "f") is False
    assert runlog.record_flag(record, "g") is False
    assert runlog.record_flag({"f": True}, "f") is True


def test_error_fields_reads_a_json_object_and_nothing_else() -> None:
    assert runlog.error_fields('{"type": "ValueError"}') == {"type": "ValueError"}
    assert runlog.error_fields("[1]") is None
    assert runlog.error_fields("not json") is None
    assert runlog.error_fields("") is None


def test_the_last_attempt_is_the_records_from_the_last_started_on() -> None:
    first: dict[str, object] = {"ev": "started", "n": 1}
    second: dict[str, object] = {"ev": "started", "n": 2}
    tail: dict[str, object] = {"ev": "entry_end"}

    assert runlog.last_attempt_records([first, tail, second, tail]) == [second, tail]
    assert runlog.last_attempt_records([tail, tail]) == [tail, tail]
    assert runlog.last_attempt_records([]) == []
