"""What a run-log says an attempt cost: its memory, CPU, time and per-row rates.

A reader for the tools that size a run from earlier ones, such as a queue that
chooses a pod's memory and processors. :func:`read_run_cost` folds one attempt's
log into a :class:`RunCost`, so a caller never parses the events itself.

This module imports the standard library and :mod:`mosaic.runlog` only, so
reading a cost loads neither pydantic nor pandas. Every field defaults when the
log lacks it: a log written before a field existed reads as 0, ``False`` or
``None``, and the log of an op or of a refusal reads with every measured field
empty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic.runlog import (
    TERMINAL_STATUSES,
    error_fields,
    fold_run_records,
    last_attempt_records,
    read_run_records,
    record_count,
    record_flag,
    record_seconds,
    record_text,
)

__all__ = ["RunCost", "ThreadScaling", "read_run_cost"]

_MEMORY_FAILURES: Final = frozenset(
    {"WorkerProcessDied", "MemoryFloorReached", "EntryOverMemoryLimit"}
)
"""The exception types of a run that stopped for lack of memory."""


@dataclass(frozen=True, slots=True)
class ThreadScaling:
    """How a feature's entries fare when two run at once in threads.

    Measured over the entries of one attempt that ran alone and the entries that
    ran in a pair.

    Attributes:
        alone_seconds_per_row: Wall seconds per input row over the entries that
            ran alone.
        paired_seconds_per_row: The same over the entries that ran in a pair.
        speedup: Twice the alone figure over the paired one. 2 means a second
            thread doubles the throughput, 1 means it adds nothing.
        alone_cpu_per_wall_second: The CPU seconds over the wall seconds of the
            entries that ran alone, the processors one worker uses.
        paired_cpu_per_wall_second: The same over the entries that ran in a pair,
            both entries of a pair counting.
    """

    alone_seconds_per_row: float
    paired_seconds_per_row: float
    speedup: float
    alone_cpu_per_wall_second: float
    paired_cpu_per_wall_second: float


@dataclass(frozen=True, slots=True)
class RunCost:
    """What one attempt cost, read from its run-log.

    The status, ``has_terminal_event`` and the failure fields come from the
    terminal event after the last ``started``; ``kind``, ``target``, ``run_id``,
    ``probe``, ``workers`` and ``parallel_mode`` are the status snapshot's. The
    costs are those of the attempt's last process, the events after its last
    ``started``. ``None`` and 0 mean the log does not say.

    Attributes:
        execution_id: The attempt.
        kind: Its job kind, such as ``feature``.
        target: What it ran.
        run_id: The run it computed, ``""`` before it is known.
        feature: The feature's registered name, ``""`` for a log that records none.
        feature_version: The feature's version.
        status: The attempt's status, ``running`` without a terminal event.
        has_terminal_event: Whether the log ends in ``finished``, ``failed`` or
            ``cancelled``. A log without one, read after its process ended, is a
            kill.
        stopped_for_memory: The attempt failed with ``WorkerProcessDied``,
            ``MemoryFloorReached`` or ``EntryOverMemoryLimit``.
        error_type: The failure's exception type, ``None`` for a refusal or no
            failure.
        refusal_reason: A refusal's reason, ``None`` otherwise.
        refusal_cause: The ``cause`` in a refusal's detail, such as
            ``scope_dependent`` of a ``subset_refused``, ``None`` otherwise.
        probe: Whether the attempt measured a few entries of a step.
        workers: How many entries it computed at once.
        parallel_mode: How, ``""`` for one at a time.
        process_capable: Whether the feature may run in worker processes.
        memory_limit_bytes: The memory limit it ran under, a container's or the
            host's.
        memory_limit_source: Where that limit came from (``cgroup_v2``,
            ``cgroup_v1`` or ``host``).
        limit_peak_bytes: The most memory the limit's accounting reported in use.
        base_rss_bytes: The process tree's resident memory before the first entry
            loaded.
        peak_rss_bytes: The tree's largest resident memory.
        fit_seconds: The wall time of the fit.
        fit_peak_rss_bytes: The tree's largest resident memory during the fit.
        fit_entries: The entries of the stream the fit was offered.
        fit_input_rows: Their input rows. A fitter that fits from pinned artifacts
            ignores that stream, so for it both figures describe what it was
            offered and not what it read.
        worker_base_rss_bytes: One process worker's own memory, ``None`` when
            unmeasured.
        peak_bytes_per_row: The run's estimate of an entry's peak memory per input
            row, ``None`` until an observation with rows exceeds the fixed part:
            when no entry had input rows, or none rose above the run's base and
            its workers' own memory.
        seconds_per_row: Wall seconds per input row over the entries that ran
            alone.
        alone_cpu_per_wall_second: CPU seconds per wall second over the entries
            that ran alone, one worker's processors, probe or not.
        rows_per_second: The input rows of every measured entry over the wall time
            of the apply phase.
        cpu_seconds: The CPU time of the process tree since measuring began.
        wall_seconds: The wall time since measuring began.
        cpu_per_wall_second: The two above as a ratio, ``None`` without wall time.
        entries_measured: The computed entries that recorded a cost.
        rows_measured: Their input rows.
        largest_entry_rows: The input rows of the largest.
        thread_scaling: The alone and paired phases compared, ``None`` unless
            entries ended alone and in pairs both.
    """

    execution_id: str
    kind: str
    target: str
    run_id: str
    feature: str
    feature_version: str
    status: str
    has_terminal_event: bool
    stopped_for_memory: bool
    error_type: str | None
    refusal_reason: str | None
    refusal_cause: str | None
    probe: bool
    workers: int
    parallel_mode: str
    process_capable: bool
    memory_limit_bytes: int
    memory_limit_source: str
    limit_peak_bytes: int
    base_rss_bytes: int
    peak_rss_bytes: int
    fit_seconds: float
    fit_peak_rss_bytes: int
    fit_entries: int
    fit_input_rows: int
    worker_base_rss_bytes: int | None
    peak_bytes_per_row: int | None
    seconds_per_row: float | None
    alone_cpu_per_wall_second: float | None
    rows_per_second: float | None
    cpu_seconds: float
    wall_seconds: float
    cpu_per_wall_second: float | None
    entries_measured: int
    rows_measured: int
    largest_entry_rows: int
    thread_scaling: ThreadScaling | None


@dataclass(frozen=True, slots=True)
class _Measured:
    """The cost of one computed entry, as its ``entry_end`` event recorded it."""

    rows: int
    wall_seconds: float
    cpu_seconds: float
    in_flight: int


def read_run_cost(run_dir: Path | str, execution_id: str) -> RunCost | None:
    """Read what one attempt cost from its run-log.

    The log is read once and folded twice, so an append between the folds cannot
    split the attempt.

    Args:
        run_dir: The run-log directory, ``<dataset root>/.mosaic/runs``, which
            :func:`mosaic.runlog.run_log_dir` returns for the dataset root.
        execution_id: The attempt.

    Returns:
        Its cost, ``None`` for a missing or empty log.
    """
    records = read_run_records(run_dir, execution_id)
    snapshot = fold_run_records(execution_id, records)
    if snapshot is None:
        return None
    attempt = last_attempt_records(records)
    ending = _ending(attempt)
    resources = _last_resources(attempt)
    measured = _measured_entries(attempt)
    alone = [entry for entry in measured if entry.in_flight == 1]
    paired = [entry for entry in measured if entry.in_flight == 2]
    rows_measured = sum(entry.rows for entry in measured)
    cpu_seconds = record_seconds(resources, "cpu_seconds")
    wall_seconds = record_seconds(resources, "wall_seconds")
    apply_wall_seconds = record_seconds(resources, "apply_wall_seconds")
    error_type, refusal_reason, refusal_cause = _failure(ending)
    return RunCost(
        execution_id=execution_id,
        kind=snapshot["kind"],
        target=snapshot["target"],
        run_id=snapshot["run_id"],
        feature=record_text(resources, "feature"),
        feature_version=record_text(resources, "feature_version"),
        status=record_text(ending, "ev") or "running",
        has_terminal_event=bool(ending),
        stopped_for_memory=error_type in _MEMORY_FAILURES,
        error_type=error_type,
        refusal_reason=refusal_reason,
        refusal_cause=refusal_cause,
        probe=snapshot["probe"],
        workers=snapshot["workers"],
        parallel_mode=snapshot["parallel_mode"],
        process_capable=record_flag(resources, "process_capable"),
        memory_limit_bytes=record_count(resources, "memory_limit_bytes"),
        memory_limit_source=record_text(resources, "memory_limit_source"),
        limit_peak_bytes=record_count(resources, "limit_peak_bytes"),
        base_rss_bytes=record_count(resources, "base_rss_bytes"),
        peak_rss_bytes=snapshot["peak_rss_bytes"],
        fit_seconds=record_seconds(resources, "fit_seconds"),
        fit_peak_rss_bytes=record_count(resources, "fit_peak_rss_bytes"),
        fit_entries=record_count(resources, "fit_entries"),
        fit_input_rows=record_count(resources, "fit_input_rows"),
        worker_base_rss_bytes=(
            record_count(resources, "worker_base_rss_bytes") or None
        ),
        peak_bytes_per_row=record_count(resources, "peak_bytes_per_row") or None,
        seconds_per_row=_seconds_per_row(alone),
        alone_cpu_per_wall_second=_cpu_per_wall_second(alone),
        rows_per_second=_ratio(rows_measured, apply_wall_seconds),
        cpu_seconds=cpu_seconds,
        wall_seconds=wall_seconds,
        cpu_per_wall_second=_ratio(cpu_seconds, wall_seconds),
        entries_measured=len(measured),
        rows_measured=rows_measured,
        largest_entry_rows=max((entry.rows for entry in measured), default=0),
        thread_scaling=_thread_scaling(alone, paired),
    )


def _ending(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """The last terminal event of the attempt's own records, empty without one.

    The snapshot keeps the status of an earlier process through a later
    ``started``, so the end of a restarted attempt is read here.
    """
    for rec in reversed(records):
        if record_text(rec, "ev") in TERMINAL_STATUSES:
            return rec
    return {}


def _last_resources(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """The last ``resources`` event, which holds the attempt's extremes, else empty."""
    for rec in reversed(records):
        if rec.get("ev") == "resources":
            return rec
    return {}


def _measured_entries(records: Sequence[Mapping[str, object]]) -> list[_Measured]:
    """The computed entries that recorded a cost, in file order.

    A cache hit writes its ``entry_end`` without the fields, and an old log
    never wrote them, so neither is a measurement.
    """
    return [
        _Measured(
            rows=record_count(rec, "input_rows"),
            wall_seconds=record_seconds(rec, "wall_seconds"),
            cpu_seconds=record_seconds(rec, "cpu_seconds"),
            in_flight=record_count(rec, "in_flight"),
        )
        for rec in records
        if rec.get("ev") == "entry_end" and record_count(rec, "in_flight") >= 1
    ]


def _ratio(numerator: float, denominator: float) -> float | None:
    """*numerator* over *denominator*, ``None`` when the denominator is not positive."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _seconds_per_row(entries: Sequence[_Measured]) -> float | None:
    """Wall seconds per input row over *entries*, ``None`` without rows."""
    return _ratio(
        sum(entry.wall_seconds for entry in entries),
        sum(entry.rows for entry in entries),
    )


def _cpu_per_wall_second(entries: Sequence[_Measured]) -> float | None:
    """CPU seconds per wall second over *entries*, ``None`` without wall time."""
    return _ratio(
        sum(entry.cpu_seconds for entry in entries),
        sum(entry.wall_seconds for entry in entries),
    )


def _thread_scaling(
    alone: Sequence[_Measured], paired: Sequence[_Measured]
) -> ThreadScaling | None:
    """Compare the entries that ran alone with those that ran in a pair."""
    alone_rate = _seconds_per_row(alone)
    paired_rate = _seconds_per_row(paired)
    alone_cpu = _cpu_per_wall_second(alone)
    paired_cpu = _cpu_per_wall_second(paired)
    if (
        alone_rate is None
        or paired_rate is None
        or paired_rate <= 0
        or alone_cpu is None
        or paired_cpu is None
    ):
        return None
    return ThreadScaling(
        alone_seconds_per_row=alone_rate,
        paired_seconds_per_row=paired_rate,
        speedup=2 * alone_rate / paired_rate,
        alone_cpu_per_wall_second=alone_cpu,
        paired_cpu_per_wall_second=paired_cpu,
    )


def _failure(
    ending: Mapping[str, object],
) -> tuple[str | None, str | None, str | None]:
    """The error type, refusal reason and refusal cause of a failed attempt.

    A refusal's blob holds ``reason`` and its detail beside it, so it reads as a
    refusal and has no error type. A crash's holds ``type``. Anything else, or an
    attempt that did not fail, reads as none of them.
    """
    if ending.get("ev") != "failed":
        return None, None, None
    fields = error_fields(record_text(ending, "error"))
    if fields is None:
        return None, None, None
    reason = record_text(fields, "reason") or None
    if reason is not None:
        return None, reason, record_text(fields, "cause") or None
    return record_text(fields, "type") or None, None, None
