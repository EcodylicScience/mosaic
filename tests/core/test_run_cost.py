"""The cost reader: what a run-log says an attempt cost, read without parsing events."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from mosaic.run_cost import RunCost, read_run_cost
from mosaic.runlog import (
    EntryMeasure,
    JsonlRunLog,
    ResourceReport,
    run_log_dir,
    run_log_path,
)

GIB = 1024**3


def _log(base: Path, execution_id: str) -> JsonlRunLog:
    return JsonlRunLog(run_log_path(base, execution_id), execution_id)


_BASE_REPORT = ResourceReport(
    workers=2,
    parallel_mode="thread",
    peak_rss_bytes=3 * GIB,
    min_available_bytes=5 * GIB,
    total_memory_bytes=64 * GIB,
    feature="speed",
    feature_version="0.3",
    process_capable=True,
    cpu_seconds=30.0,
    wall_seconds=20.0,
    memory_limit_bytes=8 * GIB,
    memory_limit_source="cgroup_v2",
    limit_peak_bytes=3 * GIB,
    base_rss_bytes=GIB,
    fit_seconds=2.5,
    fit_peak_rss_bytes=2 * GIB,
    peak_bytes_per_row=1000,
    fit_entries=4,
    fit_input_rows=1200,
    worker_base_rss_bytes=GIB // 2,
    apply_wall_seconds=10.0,
)


def _report(**changes: object) -> ResourceReport:
    return dataclasses.replace(_BASE_REPORT, **changes)


def _measure(
    rows: int, wall: float, in_flight: int, cpu: float | None = None
) -> EntryMeasure:
    return EntryMeasure(
        wall_seconds=wall,
        apply_seconds=wall,
        input_rows=rows,
        in_flight=in_flight,
        cpu_seconds=wall if cpu is None else cpu,
    )


def _write_lines(base: Path, execution_id: str, *records: dict[str, object]) -> None:
    path = run_log_path(base, execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _read(base: Path, execution_id: str) -> RunCost:
    cost = read_run_cost(run_log_dir(base), execution_id)
    assert cost is not None
    return cost


def test_a_sequential_run_reads_its_costs_and_the_one_workers_cpu(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "SEQ") as log:
        log.started(kind="feature", target="speed")
        log.set_run_id("0.3-abcdef0123")
        log.set_total(2)
        log.resources(_report(workers=1, parallel_mode="", worker_base_rss_bytes=0))
        log.on_entry_end(
            1, 2, "g__a", measure=_measure(rows=100, wall=2.0, in_flight=1, cpu=1.5)
        )
        log.on_entry_end(
            2, 2, "g__b", measure=_measure(rows=300, wall=6.0, in_flight=1, cpu=4.5)
        )
        log.finished()

    cost = _read(tmp_path, "SEQ")

    assert (cost.execution_id, cost.kind, cost.target) == ("SEQ", "feature", "speed")
    assert (cost.run_id, cost.feature, cost.feature_version) == (
        "0.3-abcdef0123",
        "speed",
        "0.3",
    )
    assert (cost.status, cost.has_terminal_event) == ("finished", True)
    assert (cost.stopped_for_memory, cost.error_type) == (False, None)
    assert (cost.refusal_reason, cost.refusal_cause, cost.probe) == (None, None, False)
    assert (cost.workers, cost.parallel_mode, cost.process_capable) == (1, "", True)
    assert (cost.memory_limit_bytes, cost.memory_limit_source) == (8 * GIB, "cgroup_v2")
    assert (cost.limit_peak_bytes, cost.base_rss_bytes) == (3 * GIB, GIB)
    assert cost.peak_rss_bytes == 3 * GIB
    assert (cost.fit_seconds, cost.fit_peak_rss_bytes) == (2.5, 2 * GIB)
    assert (cost.fit_entries, cost.fit_input_rows) == (4, 1200)
    assert cost.worker_base_rss_bytes is None
    assert cost.peak_bytes_per_row == 1000
    assert (cost.entries_measured, cost.rows_measured) == (2, 400)
    assert cost.largest_entry_rows == 300
    assert cost.seconds_per_row == pytest.approx(8.0 / 400)
    assert cost.alone_cpu_per_wall_second == pytest.approx(6.0 / 8.0)
    assert cost.rows_per_second == pytest.approx(400 / 10.0)
    assert (cost.cpu_seconds, cost.wall_seconds) == (30.0, 20.0)
    assert cost.cpu_per_wall_second == pytest.approx(1.5)
    assert cost.thread_scaling is None


def test_a_probe_run_reads_the_alone_and_the_paired_phase_apart(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "PROBE") as log:
        log.started(kind="feature", target="speed")
        log.probe(
            request_id="R1", step="speed", entries=["g:a", "g:b", "g:c"], kind="probe"
        )
        log.set_total(3)
        log.resources(_report(worker_base_rss_bytes=GIB // 4))
        log.on_entry_end(
            1, 3, "g__a", measure=_measure(rows=200, wall=4.0, in_flight=1, cpu=3.0)
        )
        log.on_entry_end(
            2, 3, "g__b", measure=_measure(rows=100, wall=3.0, in_flight=2, cpu=4.5)
        )
        log.on_entry_end(
            3, 3, "g__c", measure=_measure(rows=100, wall=3.0, in_flight=2, cpu=4.5)
        )
        log.finished()

    cost = _read(tmp_path, "PROBE")

    assert cost.probe is True
    assert cost.worker_base_rss_bytes == GIB // 4
    assert cost.seconds_per_row == pytest.approx(4.0 / 200)
    assert cost.alone_cpu_per_wall_second == pytest.approx(3.0 / 4.0)
    scaling = cost.thread_scaling
    assert scaling is not None
    assert scaling.alone_seconds_per_row == pytest.approx(0.02)
    assert scaling.paired_seconds_per_row == pytest.approx(6.0 / 200)
    assert scaling.speedup == pytest.approx(2 * 0.02 / 0.03)
    assert scaling.alone_cpu_per_wall_second == pytest.approx(0.75)
    assert scaling.paired_cpu_per_wall_second == pytest.approx(9.0 / 6.0)


def test_entries_that_never_ran_in_pairs_give_no_thread_scaling(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "ALONE") as log:
        log.started(kind="feature", target="speed")
        log.resources(_report())
        log.on_entry_end(1, 2, "g__a", measure=_measure(rows=10, wall=1.0, in_flight=1))
        log.on_entry_end(2, 2, "g__b", measure=_measure(rows=10, wall=1.0, in_flight=3))
        log.finished()

    assert _read(tmp_path, "ALONE").thread_scaling is None


def test_a_cache_hit_entry_is_not_a_measurement(tmp_path: Path) -> None:
    with _log(tmp_path, "HIT") as log:
        log.started(kind="feature", target="speed")
        log.resources(_report(apply_wall_seconds=4.0))
        log.on_entry_end(1, 2, "g__a")
        log.on_entry_end(2, 2, "g__b", measure=_measure(rows=50, wall=2.0, in_flight=1))
        log.finished()

    cost = _read(tmp_path, "HIT")

    assert (cost.entries_measured, cost.rows_measured) == (1, 50)
    assert cost.largest_entry_rows == 50


def test_an_old_log_reads_every_measured_field_as_empty(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "OLD",
        {"ev": "started", "kind": "feature", "target": "speed"},
        {"ev": "run_id", "run_id": "0.1-1111111111"},
        {
            "ev": "resources",
            "workers": 4,
            "parallel_mode": "thread",
            "peak_rss_bytes": 300,
            "min_available_bytes": 500,
            "total_memory_bytes": 1000,
        },
        {"ev": "entry_end", "index": 1, "total": 1, "key": "g__a"},
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "OLD")

    assert (cost.status, cost.has_terminal_event, cost.run_id) == (
        "finished",
        True,
        "0.1-1111111111",
    )
    assert (cost.workers, cost.parallel_mode, cost.peak_rss_bytes) == (4, "thread", 300)
    assert (cost.feature, cost.feature_version, cost.process_capable) == ("", "", False)
    assert (cost.memory_limit_bytes, cost.memory_limit_source) == (0, "")
    assert (cost.limit_peak_bytes, cost.base_rss_bytes, cost.fit_entries) == (0, 0, 0)
    assert (cost.worker_base_rss_bytes, cost.peak_bytes_per_row) == (None, None)
    assert (cost.seconds_per_row, cost.alone_cpu_per_wall_second) == (None, None)
    assert (cost.rows_per_second, cost.cpu_per_wall_second) == (None, None)
    assert (cost.entries_measured, cost.rows_measured) == (0, 0)
    assert cost.thread_scaling is None


def test_a_log_cut_before_its_terminal_event_reads_as_running(tmp_path: Path) -> None:
    with _log(tmp_path, "CUT") as log:
        log.started(kind="feature", target="speed")
        log.resources(_report())
        log.on_entry_end(1, 3, "g__a", measure=_measure(rows=10, wall=1.0, in_flight=1))

    cost = _read(tmp_path, "CUT")

    assert (cost.status, cost.has_terminal_event) == ("running", False)
    assert (cost.stopped_for_memory, cost.error_type) == (False, None)
    assert cost.entries_measured == 1


@pytest.mark.parametrize(
    "error_type", ["WorkerProcessDied", "MemoryFloorReached", "EntryOverMemoryLimit"]
)
def test_a_memory_failure_marks_the_run_stopped_for_memory(
    tmp_path: Path, error_type: str
) -> None:
    with _log(tmp_path, "OOM") as log:
        log.started(kind="feature", target="speed")
        log.resources(_report(fit_peak_rss_bytes=7 * GIB))
        log.failed(json.dumps({"type": error_type, "message": "out of memory"}))

    cost = _read(tmp_path, "OOM")

    assert (cost.status, cost.has_terminal_event) == ("failed", True)
    assert cost.stopped_for_memory is True
    assert cost.error_type == error_type
    assert cost.refusal_reason is None
    assert cost.fit_peak_rss_bytes == 7 * GIB


def test_a_crash_reads_its_type_and_is_not_a_memory_stop(tmp_path: Path) -> None:
    with _log(tmp_path, "BOOM") as log:
        log.started(kind="feature", target="speed")
        log.failed(
            json.dumps({"type": "ValueError", "message": "bad", "traceback": ""})
        )

    cost = _read(tmp_path, "BOOM")

    assert cost.error_type == "ValueError"
    assert cost.stopped_for_memory is False
    assert (cost.refusal_reason, cost.refusal_cause) == (None, None)


def test_a_refusal_reads_its_reason_and_cause_and_no_error_type(
    tmp_path: Path,
) -> None:
    refusal = {
        "reason": "subset_refused",
        "step": "speed",
        "message": "the step reads the whole scope",
        "cause": "scope_dependent",
    }
    with _log(tmp_path, "REFUSED") as log:
        log.started(kind="feature", target="speed")
        log.probe(request_id="R1", step="speed", entries=["g:a"], kind="probe")
        log.failed(json.dumps(refusal))

    cost = _read(tmp_path, "REFUSED")

    assert cost.refusal_reason == "subset_refused"
    assert cost.refusal_cause == "scope_dependent"
    assert cost.error_type is None
    assert cost.stopped_for_memory is False
    assert (cost.entries_measured, cost.thread_scaling) == (0, None)
    assert (cost.memory_limit_bytes, cost.peak_bytes_per_row) == (0, None)


def test_a_failure_that_records_no_blob_reads_no_error(tmp_path: Path) -> None:
    with _log(tmp_path, "BARE") as log:
        log.started(kind="feature", target="speed")
        log.failed()

    cost = _read(tmp_path, "BARE")

    assert cost.status == "failed"
    assert (cost.error_type, cost.refusal_reason, cost.refusal_cause) == (
        None,
        None,
        None,
    )
    assert cost.stopped_for_memory is False


def test_a_resumed_attempt_reads_only_what_follows_its_last_start(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "RESUMED") as log:
        log.started(kind="feature", target="speed")
        log.probe(request_id="R1", step="speed", entries=["g:a"], kind="probe")
        log.resources(_report(feature_version="0.1", fit_entries=9, cpu_seconds=99.0))
        log.on_entry_end(
            1, 2, "g__a", measure=_measure(rows=500, wall=9.0, in_flight=1)
        )
        log.started(kind="feature", target="speed")
        log.resources(_report(fit_entries=3, cpu_seconds=10.0, wall_seconds=5.0))
        log.on_entry_end(1, 1, "g__b", measure=_measure(rows=20, wall=1.0, in_flight=1))
        log.finished()

    cost = _read(tmp_path, "RESUMED")

    assert cost.probe is False
    assert (cost.feature_version, cost.fit_entries) == ("0.3", 3)
    assert (cost.cpu_seconds, cost.wall_seconds) == (10.0, 5.0)
    assert (cost.entries_measured, cost.rows_measured) == (1, 20)
    assert cost.largest_entry_rows == 20


def test_the_last_resources_event_holds_the_attempts_figures(tmp_path: Path) -> None:
    with _log(tmp_path, "GROW") as log:
        log.started(kind="feature", target="speed")
        log.resources(_report(peak_rss_bytes=GIB, peak_bytes_per_row=0, workers=1))
        log.resources(
            _report(peak_rss_bytes=4 * GIB, peak_bytes_per_row=2000, workers=2)
        )
        log.finished()

    cost = _read(tmp_path, "GROW")

    assert (cost.peak_rss_bytes, cost.peak_bytes_per_row, cost.workers) == (
        4 * GIB,
        2000,
        2,
    )


def test_a_log_of_an_op_reads_with_every_measured_field_empty(tmp_path: Path) -> None:
    with _log(tmp_path, "OP") as log:
        log.started(kind="op", target="extract-frames")
        log.finished()

    cost = _read(tmp_path, "OP")

    assert (cost.kind, cost.target, cost.status) == ("op", "extract-frames", "finished")
    assert (cost.workers, cost.memory_limit_bytes, cost.entries_measured) == (0, 0, 0)
    assert (cost.peak_bytes_per_row, cost.worker_base_rss_bytes) == (None, None)
    assert cost.cpu_per_wall_second is None


def test_a_missing_or_empty_log_reads_as_none(tmp_path: Path) -> None:
    run_log_dir(tmp_path).mkdir(parents=True)
    run_log_path(tmp_path, "EMPTY").write_text("")

    assert read_run_cost(run_log_dir(tmp_path), "ABSENT") is None
    assert read_run_cost(run_log_dir(tmp_path), "EMPTY") is None


def test_fields_of_the_wrong_type_read_as_empty(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "ODD",
        {"ev": "started", "kind": "feature", "target": "speed"},
        {"ev": "resources", "workers": 1, "cpu_seconds": "many", "fit_entries": True},
        {"ev": "entry_end", "index": 1, "total": 1, "key": "g__a", "in_flight": "one"},
        {"ev": "failed", "error": "not json"},
    )

    cost = _read(tmp_path, "ODD")

    assert (cost.cpu_seconds, cost.fit_entries, cost.entries_measured) == (0.0, 0, 0)
    assert (cost.error_type, cost.refusal_reason) == (None, None)


def test_a_restarted_attempt_does_not_inherit_the_failure_of_the_one_before(
    tmp_path: Path,
) -> None:
    started: dict[str, object] = {"ev": "started", "kind": "feature", "target": "x"}
    _write_lines(
        tmp_path,
        "RESTART",
        started,
        {
            "ev": "failed",
            "error": json.dumps({"type": "WorkerProcessDied", "message": "m"}),
        },
        started,
        {
            "ev": "entry_end",
            "index": 1,
            "total": 2,
            "input_rows": 10,
            "wall_seconds": 1.0,
            "cpu_seconds": 1.0,
            "in_flight": 1,
        },
    )

    cost = _read(tmp_path, "RESTART")

    assert (cost.status, cost.has_terminal_event) == ("running", False)
    assert (cost.stopped_for_memory, cost.error_type) == (False, None)
    assert cost.entries_measured == 1


def test_a_restarted_attempt_reads_the_end_of_its_own_process(tmp_path: Path) -> None:
    started: dict[str, object] = {"ev": "started", "kind": "feature", "target": "x"}
    crash = json.dumps({"type": "ValueError", "message": "m"})
    _write_lines(
        tmp_path,
        "RESTARTED",
        started,
        {"ev": "failed", "error": json.dumps({"type": "WorkerProcessDied"})},
        started,
        {"ev": "failed", "error": crash},
    )

    cost = _read(tmp_path, "RESTARTED")

    assert (cost.status, cost.has_terminal_event) == ("failed", True)
    assert (cost.error_type, cost.stopped_for_memory) == ("ValueError", False)
