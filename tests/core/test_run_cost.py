"""The cost reader: what a run-log says an attempt cost, read without parsing events."""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

import pytest

from mosaic.run_cost import RunCost, read_run_cost
from mosaic.runlog import (
    EntryMeasure,
    InFlightLevel,
    JsonlRunLog,
    MachineFacts,
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
    peak_nonreclaimable_bytes=2 * GIB,
    base_nonreclaimable_bytes=GIB // 4,
    worker_base_nonreclaimable_bytes=GIB // 8,
)


def _report(**changes: object) -> ResourceReport:
    return dataclasses.replace(_BASE_REPORT, **changes)


def _measure(
    rows: int,
    wall: float,
    in_flight: int,
    cpu: float | None = None,
    mode: str = "thread",
) -> EntryMeasure:
    return EntryMeasure(
        wall_seconds=wall,
        apply_seconds=wall,
        input_rows=rows,
        in_flight=in_flight,
        cpu_seconds=wall if cpu is None else cpu,
        mode=mode,
    )


def _instant(seconds: float) -> str:
    """The ISO instant *seconds* after the epoch, as a run-log writes ``t``."""
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).isoformat()


def _entry_end(
    key: str,
    *,
    ended: float,
    wall: float,
    rows: int,
    cpu: float,
    in_flight: int = 1,
    **more: object,
) -> dict[str, object]:
    """One measured ``entry_end`` record that ended *ended* seconds after the epoch."""
    return {
        "t": _instant(ended),
        "ev": "entry_end",
        "index": 1,
        "total": 1,
        "key": key,
        "wall_seconds": wall,
        "apply_seconds": wall,
        "input_rows": rows,
        "in_flight": in_flight,
        "cpu_seconds": cpu,
        **more,
    }


def _resources(**fields: object) -> dict[str, object]:
    """A ``resources`` record as :meth:`JsonlRunLog.resources` writes one.

    A field given as ``None`` is left out of the record.
    """
    record = {"ev": "resources", **dataclasses.asdict(_report()), **fields}
    return {name: value for name, value in record.items() if value is not None}


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
        log.resources(
            _report(
                workers=1,
                parallel_mode="",
                worker_base_rss_bytes=0,
                worker_base_nonreclaimable_bytes=0,
            )
        )
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
    assert cost.peak_nonreclaimable_bytes == 2 * GIB
    assert cost.base_nonreclaimable_bytes == GIB // 4
    assert cost.worker_base_nonreclaimable_bytes is None
    assert cost.peak_bytes_per_row == 1000
    assert (cost.entries_measured, cost.rows_measured) == (2, 400)
    assert cost.largest_entry_rows == 300
    assert cost.seconds_per_row == pytest.approx(8.0 / 400)
    assert cost.alone_cpu_per_wall_second == pytest.approx(6.0 / 8.0)
    assert cost.rows_per_second == pytest.approx(400 / 10.0)
    assert (cost.cpu_seconds, cost.wall_seconds) == (30.0, 20.0)
    assert cost.cpu_per_wall_second == pytest.approx(1.5)
    assert cost.levels == ()


def test_a_run_with_the_non_reclaimable_figures_reads_each_of_them(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "NR") as log:
        log.started(kind="feature", target="speed")
        log.resources(_report())
        log.finished()

    cost = _read(tmp_path, "NR")

    assert (
        cost.peak_nonreclaimable_bytes,
        cost.base_nonreclaimable_bytes,
        cost.worker_base_nonreclaimable_bytes,
    ) == (2 * GIB, GIB // 4, GIB // 8)


def test_a_figure_the_run_did_not_measure_reads_none(tmp_path: Path) -> None:
    with _log(tmp_path, "RSS") as log:
        log.started(kind="feature", target="speed")
        log.resources(
            _report(
                peak_nonreclaimable_bytes=0,
                base_nonreclaimable_bytes=0,
                worker_base_nonreclaimable_bytes=0,
            )
        )
        log.finished()

    cost = _read(tmp_path, "RSS")

    assert (
        cost.peak_nonreclaimable_bytes,
        cost.base_nonreclaimable_bytes,
        cost.worker_base_nonreclaimable_bytes,
    ) == (None, None, None)


def test_a_probe_run_reads_the_alone_and_the_paired_phase_apart(
    tmp_path: Path,
) -> None:
    levels = (
        InFlightLevel(in_flight=1, seconds=4.0, rows=200.0, cpu_seconds=3.0),
        InFlightLevel(in_flight=2, seconds=3.0, rows=200.0, cpu_seconds=9.0),
    )
    with _log(tmp_path, "PROBE") as log:
        log.started(kind="feature", target="speed")
        log.probe(
            request_id="R1", step="speed", entries=["g:a", "g:b", "g:c"], kind="probe"
        )
        log.set_total(3)
        log.resources(_report(worker_base_rss_bytes=GIB // 4, in_flight_levels=levels))
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
    alone, paired = cost.levels
    assert (alone.mode, alone.in_flight, alone.rebuilt) == ("thread", 1, False)
    assert (paired.mode, paired.in_flight, paired.rebuilt) == ("thread", 2, False)
    assert (alone.seconds, alone.rows, alone.cpu_seconds) == (4.0, 200.0, 3.0)
    assert (paired.seconds, paired.rows, paired.cpu_seconds) == (3.0, 200.0, 9.0)
    speedup = (paired.rows / paired.seconds) / (alone.rows / alone.seconds)
    assert speedup == pytest.approx((200 / 3.0) / (200 / 4.0))


def test_entries_that_never_ran_in_pairs_give_level_one_alone(
    tmp_path: Path,
) -> None:
    with _log(tmp_path, "ALONE") as log:
        log.started(kind="feature", target="speed")
        log.resources(
            _report(
                in_flight_levels=(
                    InFlightLevel(in_flight=1, seconds=2.0, rows=20.0, cpu_seconds=2.0),
                )
            )
        )
        log.on_entry_end(1, 2, "g__a", measure=_measure(rows=10, wall=1.0, in_flight=1))
        log.on_entry_end(2, 2, "g__b", measure=_measure(rows=10, wall=1.0, in_flight=1))
        log.finished()

    cost = _read(tmp_path, "ALONE")

    assert [(level.in_flight, level.rebuilt) for level in cost.levels] == [(1, False)]


def test_a_cache_hit_entry_is_not_a_measurement(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "HIT",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(apply_wall_seconds=4.0, in_flight_levels=None),
        {
            "t": _instant(100.0),
            "ev": "entry_end",
            "index": 1,
            "total": 2,
            "key": "g__a",
        },
        _entry_end("g__b", ended=110.0, wall=2.0, rows=50, cpu=2.0),
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "HIT")

    assert (cost.entries_measured, cost.rows_measured) == (1, 50)
    assert cost.largest_entry_rows == 50
    assert [entry.key for entry in cost.entries] == ["g__b"]
    (level,) = cost.levels
    assert (level.in_flight, level.seconds, level.rows) == (1, 2.0, 50.0)


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
    assert (
        cost.peak_nonreclaimable_bytes,
        cost.base_nonreclaimable_bytes,
        cost.worker_base_nonreclaimable_bytes,
    ) == (None, None, None)
    assert (cost.seconds_per_row, cost.alone_cpu_per_wall_second) == (None, None)
    assert (cost.rows_per_second, cost.cpu_per_wall_second) == (None, None)
    assert (cost.entries_measured, cost.rows_measured) == (0, 0)
    assert (cost.entries, cost.peak_points, cost.levels) == ((), (), ())
    assert (cost.params_digest, cost.params_default, cost.emits) == (None, None, None)
    assert (cost.machine, cost.skipped_lines) == (None, 0)


def test_an_old_log_with_entry_costs_rebuilds_its_levels(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "OLDCOST",
        {"ev": "started", "kind": "feature", "target": "speed"},
        {
            "ev": "resources",
            "workers": 2,
            "parallel_mode": "thread",
            "peak_rss_bytes": 300,
            "min_available_bytes": 500,
            "total_memory_bytes": 1000,
        },
        _entry_end("g__a", ended=104.0, wall=4.0, rows=40, cpu=4.0),
        _entry_end("g__b", ended=106.0, wall=4.0, rows=40, cpu=8.0, in_flight=2),
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "OLDCOST")

    alone, paired = cost.levels
    assert [(level.mode, level.in_flight, level.rebuilt) for level in cost.levels] == [
        ("thread", 1, True),
        ("thread", 2, True),
    ]
    assert (alone.seconds, alone.rows, alone.cpu_seconds) == pytest.approx(
        (4.0, 40.0, 6.0)
    )
    assert (paired.seconds, paired.rows, paired.cpu_seconds) == pytest.approx(
        (2.0, 40.0, 3.0)
    )
    assert (cost.peak_points, cost.params_digest) == ((), None)


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
    assert (cost.entries_measured, cost.levels) == (0, ())
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


def test_a_nul_prefixed_record_is_read_and_a_torn_line_counted(tmp_path: Path) -> None:
    path = run_log_path(tmp_path, "NUL")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"ev": "started", "kind": "feature", "target": "speed"}),
        "\x00" * 700 + json.dumps({"ev": "finished"}),
        '{"ev": "entry_end", "key": "g__a", "wall_sec',
    ]
    path.write_text("\n".join(lines) + "\n")

    cost = _read(tmp_path, "NUL")

    assert (cost.status, cost.has_terminal_event) == ("finished", True)
    assert cost.skipped_lines == 1


def test_an_entry_reads_its_covariates_and_the_time_it_ended(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "COV",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(),
        _entry_end(
            "g__a",
            ended=500.5,
            wall=4.0,
            rows=100,
            cpu=3.0,
            peak_grown_bytes=GIB,
            ids_per_frame=2.5,
            pair_frames=900,
            pair_frames_bound=True,
            frames=40,
            keypoints=7,
            fps=30.0,
        ),
        {**_entry_end("g__b", ended=0.0, wall=1.0, rows=10, cpu=1.0), "t": "later"},
        {"ev": "finished"},
    )

    first, second = _read(tmp_path, "COV").entries

    assert (first.key, first.mode, first.ended_at) == ("g__a", "thread", 500.5)
    assert (first.wall_seconds, first.apply_seconds) == (4.0, 4.0)
    assert (first.cpu_seconds, first.input_rows, first.in_flight) == (3.0, 100, 1)
    assert first.peak_grown_bytes == GIB
    assert (first.ids_per_frame, first.pair_frames, first.pair_frames_bound) == (
        2.5,
        900,
        True,
    )
    assert (first.frames, first.keypoints, first.fps) == (40, 7, 30.0)
    assert second.ended_at is None
    assert (second.peak_grown_bytes, second.ids_per_frame, second.pair_frames) == (
        None,
        None,
        None,
    )
    assert (second.pair_frames_bound, second.frames, second.fps) == (False, None, None)


def test_an_entry_end_without_a_figure_reads_none_and_the_aggregates_skip_it(
    tmp_path: Path,
) -> None:
    _write_lines(
        tmp_path,
        "GAPS",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(),
        {"t": _instant(10.0), "ev": "entry_end", "key": "g__a", "in_flight": 1},
        _entry_end("g__b", ended=20.0, wall=2.0, rows=40, cpu=1.0),
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "GAPS")

    bare = cost.entries[0]
    assert (bare.wall_seconds, bare.apply_seconds, bare.cpu_seconds) == (
        None,
        None,
        None,
    )
    assert (bare.input_rows, bare.in_flight) == (None, 1)
    assert cost.entries_measured == 2
    assert (cost.rows_measured, cost.largest_entry_rows) == (40, 40)
    assert cost.seconds_per_row == pytest.approx(2.0 / 40)
    assert cost.alone_cpu_per_wall_second == pytest.approx(1.0 / 2.0)


def test_a_log_of_two_modes_reads_each_modes_levels_entries_and_peak(
    tmp_path: Path,
) -> None:
    thread_levels = [
        {"in_flight": 1, "seconds": 4.0, "rows": 200.0, "cpu_seconds": 3.0},
        {"in_flight": 2, "seconds": 3.0, "rows": 200.0, "cpu_seconds": 9.0},
    ]
    _write_lines(
        tmp_path,
        "TWO",
        {"ev": "started", "kind": "feature", "target": "speed"},
        {
            "ev": "probe",
            "kind": "probe",
            "entries": ["g:a", "g:b", "g:c", "g:d", "g:e"],
        },
        _resources(
            parallel_mode="thread",
            in_flight_levels=thread_levels,
            peak_in_flight_entries=2,
            peak_in_flight_rows=200,
            peak_in_flight_pair_frames=50,
            peak_grown_bytes=GIB,
        ),
        _entry_end("g__a", ended=104.0, wall=4.0, rows=200, cpu=3.0, mode="thread"),
        _entry_end(
            "g__b", ended=108.0, wall=3.0, rows=100, cpu=4.5, in_flight=2, mode="thread"
        ),
        _entry_end(
            "g__c", ended=108.0, wall=3.0, rows=100, cpu=4.5, in_flight=2, mode="thread"
        ),
        _resources(
            parallel_mode="process",
            in_flight_levels=None,
            peak_in_flight_entries=2,
            peak_in_flight_rows=80,
            peak_in_flight_pair_frames_bound=True,
            peak_grown_bytes=2 * GIB,
        ),
        _entry_end(
            "g__d",
            ended=210.0,
            wall=10.0,
            rows=50,
            cpu=6.0,
            in_flight=2,
            mode="process",
        ),
        _entry_end(
            "g__e",
            ended=210.0,
            wall=10.0,
            rows=30,
            cpu=10.0,
            in_flight=2,
            mode="process",
        ),
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "TWO")

    assert [(entry.key, entry.mode) for entry in cost.entries] == [
        ("g__a", "thread"),
        ("g__b", "thread"),
        ("g__c", "thread"),
        ("g__d", "process"),
        ("g__e", "process"),
    ]
    assert [(level.mode, level.in_flight, level.rebuilt) for level in cost.levels] == [
        ("thread", 1, False),
        ("thread", 2, False),
        ("process", 2, True),
    ]
    rebuilt = cost.levels[2]
    assert (rebuilt.seconds, rebuilt.rows, rebuilt.cpu_seconds) == pytest.approx(
        (10.0, 80.0, 8.0)
    )
    thread_peak, process_peak = cost.peak_points
    assert (thread_peak.mode, thread_peak.grown_bytes) == ("thread", GIB)
    assert (thread_peak.entries, thread_peak.rows) == (2, 200)
    assert (thread_peak.pair_frames, thread_peak.pair_frames_bound) == (50, False)
    assert (process_peak.mode, process_peak.grown_bytes) == ("process", 2 * GIB)
    assert (process_peak.entries, process_peak.rows) == (2, 80)
    assert (process_peak.pair_frames, process_peak.pair_frames_bound) == (0, True)


def test_an_empty_parallel_mode_reads_thread_everywhere(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "SEQ",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(
            parallel_mode="",
            workers=1,
            in_flight_levels=None,
            peak_in_flight_entries=1,
            peak_in_flight_rows=10,
        ),
        _entry_end("g__a", ended=10.0, wall=2.0, rows=10, cpu=1.0),
        {"ev": "finished"},
    )
    _write_lines(
        tmp_path,
        "PROC",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(parallel_mode="process", in_flight_levels=None),
        _entry_end("g__a", ended=10.0, wall=2.0, rows=10, cpu=1.0),
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "SEQ")
    processed = _read(tmp_path, "PROC")

    assert cost.parallel_mode == ""
    assert [entry.mode for entry in cost.entries] == ["thread"]
    assert [level.mode for level in cost.levels] == ["thread"]
    assert [point.mode for point in cost.peak_points] == ["thread"]
    assert [entry.mode for entry in processed.entries] == ["process"]
    assert [level.mode for level in processed.levels] == ["process"]


def test_params_default_reads_none_without_a_digest(tmp_path: Path) -> None:
    _write_lines(
        tmp_path,
        "NODIGEST",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(params_digest="", params_default=True, emits=""),
        {"ev": "finished"},
    )
    _write_lines(
        tmp_path,
        "DIGEST",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(params_digest="0123456789", params_default=True, emits="pair"),
        {"ev": "finished"},
    )
    _write_lines(
        tmp_path,
        "CHANGED",
        {"ev": "started", "kind": "feature", "target": "speed"},
        _resources(params_digest="0123456789", params_default=False),
        {"ev": "finished"},
    )

    unwritten = _read(tmp_path, "NODIGEST")
    default = _read(tmp_path, "DIGEST")
    changed = _read(tmp_path, "CHANGED")

    assert (unwritten.params_digest, unwritten.params_default) == (None, None)
    assert unwritten.emits is None
    assert (default.params_digest, default.params_default) == ("0123456789", True)
    assert default.emits == "pair"
    assert changed.params_default is False


def test_a_peak_point_without_pair_frames_reads_none(tmp_path: Path) -> None:
    with_key = _resources(peak_in_flight_entries=2, peak_in_flight_rows=30)
    without_key = {
        name: value
        for name, value in with_key.items()
        if not name.startswith("peak_in_flight_pair_frames")
    }
    none_in_flight = _resources(peak_in_flight_entries=0, peak_in_flight_rows=0)
    for name, record in (
        ("KEY", with_key),
        ("NOKEY", without_key),
        ("NONE", none_in_flight),
    ):
        _write_lines(
            tmp_path,
            name,
            {"ev": "started", "kind": "feature", "target": "speed"},
            record,
            {"ev": "finished"},
        )

    (present,) = _read(tmp_path, "KEY").peak_points
    (absent,) = _read(tmp_path, "NOKEY").peak_points

    assert (present.pair_frames, present.pair_frames_bound) == (0, False)
    assert (absent.pair_frames, absent.pair_frames_bound) == (None, False)
    assert (absent.entries, absent.rows) == (2, 30)
    assert _read(tmp_path, "NONE").peak_points == ()


def test_the_machine_of_the_attempt_reads_from_its_started_event(
    tmp_path: Path,
) -> None:
    machine = MachineFacts(
        cpu_model="Example Xeon 9000",
        logical_cpus=40,
        threads_per_core=2,
        memory_bytes=128 * GIB,
        machine_class="cpu-large",
    )
    with _log(tmp_path, "MACHINE") as log:
        log.started(kind="feature", target="speed", machine=machine)
        log.finished()

    assert _read(tmp_path, "MACHINE").machine == machine


def test_the_attempt_scalars_span_both_phases_and_the_phase_scalars_the_last(
    tmp_path: Path,
) -> None:
    _write_lines(
        tmp_path,
        "PHASES",
        {"ev": "started", "kind": "feature", "target": "speed"},
        {
            "ev": "probe",
            "kind": "probe",
            "entries": ["g:a", "g:b", "g:c", "g:d", "g:e"],
        },
        _resources(
            workers=2,
            parallel_mode="thread",
            worker_base_rss_bytes=GIB,
            peak_bytes_per_row=500,
            peak_rss_bytes=2 * GIB,
            cpu_seconds=12.0,
            wall_seconds=8.0,
            apply_wall_seconds=6.0,
        ),
        _entry_end("g__a", ended=104.0, wall=4.0, rows=200, cpu=3.0, mode="thread"),
        _entry_end(
            "g__b", ended=108.0, wall=3.0, rows=100, cpu=4.5, in_flight=2, mode="thread"
        ),
        _entry_end(
            "g__c", ended=108.0, wall=3.0, rows=100, cpu=4.5, in_flight=2, mode="thread"
        ),
        _resources(
            workers=2,
            parallel_mode="process",
            worker_base_rss_bytes=GIB // 2,
            peak_bytes_per_row=900,
            peak_rss_bytes=5 * GIB,
            cpu_seconds=40.0,
            wall_seconds=30.0,
            apply_wall_seconds=10.0,
        ),
        _entry_end(
            "g__d",
            ended=210.0,
            wall=10.0,
            rows=60,
            cpu=6.0,
            in_flight=2,
            mode="process",
        ),
        _entry_end(
            "g__e",
            ended=210.0,
            wall=10.0,
            rows=20,
            cpu=10.0,
            in_flight=2,
            mode="process",
        ),
        {"ev": "finished"},
    )

    cost = _read(tmp_path, "PHASES")

    assert (cost.workers, cost.parallel_mode) == (2, "process")
    assert cost.worker_base_rss_bytes == GIB // 2
    assert cost.peak_bytes_per_row == 900
    assert cost.rows_per_second == pytest.approx(80 / 10.0)
    assert (cost.peak_rss_bytes, cost.cpu_seconds, cost.wall_seconds) == (
        5 * GIB,
        40.0,
        30.0,
    )
    assert cost.cpu_per_wall_second == pytest.approx(40.0 / 30.0)
    assert (cost.entries_measured, cost.rows_measured) == (5, 480)
    assert cost.largest_entry_rows == 200
    assert cost.seconds_per_row == pytest.approx(4.0 / 200)
    assert cost.alone_cpu_per_wall_second == pytest.approx(3.0 / 4.0)
