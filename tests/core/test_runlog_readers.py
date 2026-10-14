"""The run-log readers that status reports and listings are built from, and the outcome rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mosaic.runlog as runlog
from mosaic.runlog import (
    JsonlRunLog,
    RunLogSnapshot,
    attempt_outcome,
    list_runs,
    read_entry_errors,
    read_run_progress,
    read_runs,
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
