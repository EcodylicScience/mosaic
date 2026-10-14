"""The run-log readers that a status report is built from, and the outcome rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.runlog import (
    JsonlRunLog,
    attempt_outcome,
    read_entry_errors,
    read_run_progress,
    run_log_dir,
    run_log_path,
)


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


def test_entry_errors_are_read_in_order_up_to_the_limit(tmp_path: Path) -> None:
    with _log(tmp_path, "E1") as log:
        log.started(kind="feature", target="speed")
        for index in range(5):
            log.entry_failed(f"g__s{index}", f'{{"message": "boom {index}"}}')
        log.finished()

    every = read_entry_errors(run_log_dir(tmp_path), "E1")
    first = read_entry_errors(run_log_dir(tmp_path), "E1", limit=2)

    assert [record["key"] for record in every] == [f"g__s{i}" for i in range(5)]
    assert first == every[:2]
    assert first[0]["error_json"] == '{"message": "boom 0"}'
    assert first[0]["recorded_at"]


def test_entry_errors_of_an_absent_run_log_are_empty(tmp_path: Path) -> None:
    assert read_entry_errors(run_log_dir(tmp_path), "MISSING") == []


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
