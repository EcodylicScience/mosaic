"""What a run-log says an attempt cost: its memory, CPU, time and per-row rates.

A reader for the tools that size a run from earlier ones, such as a queue that
chooses a pod's memory and processors. :func:`read_run_cost` folds one attempt's
log into a :class:`RunCost`, so a caller never parses the events itself.

This module imports the standard library and :mod:`mosaic.runlog` only, so
reading a cost loads neither pydantic nor pandas. Every field defaults when the
log lacks it: a log written before a field existed reads it as 0, ``False``,
``None`` or empty, and the log of an op or of a refusal reads with every measured
field empty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from mosaic.runlog import (
    TERMINAL_STATUSES,
    FlightSpan,
    MachineFacts,
    error_fields,
    fold_run_records,
    is_record,
    last_attempt_records,
    parse_instant,
    read_machine,
    read_run_log,
    record_count,
    record_flag,
    record_optional_count,
    record_optional_number,
    record_optional_seconds,
    record_seconds,
    record_text,
    sweep_levels,
)

__all__ = ["EntryCost", "LevelCost", "PeakPoint", "RunCost", "read_run_cost"]

_MEMORY_FAILURES: Final = frozenset(
    {"WorkerProcessDied", "MemoryFloorReached", "EntryOverMemoryLimit"}
)
"""The exception types of a run that stopped for lack of memory."""


@dataclass(frozen=True, slots=True)
class EntryCost:
    """What one computed entry cost, read from its ``entry_end`` event.

    A figure the event lacks reads ``None``.

    Attributes:
        key: The entry's key.
        mode: ``thread`` or ``process``, the mode the event names, else the run's
            ``parallel_mode``, with an empty one read as ``thread``.
        ended_at: When the event was written, in seconds since the epoch, ``None``
            where its time does not parse.
        wall_seconds: From the start of the entry's flight to its output written:
            its dispatch when its worker loads its input, else the run's load of
            it. The first entry each process worker computes contains the
            worker's build of the feature.
        apply_seconds: From its dispatch to the end of its apply, which contains
            the load when its worker loads its input, and the build of the first
            entry each process worker computes.
        cpu_seconds: The CPU time of the process tree over its flight, which
            counts every entry in flight with it, and the build of the first entry
            each process worker computes.
        input_rows: The rows of its input files.
        in_flight: The most entries in flight at once during its flight, itself
            included.
        peak_grown_bytes: The memory it added over its base, ``None`` when it was
            not measured.
        ids_per_frame: The distinct non-null ``(frame, id)`` pairs over the frames
            of its extent.
        pair_frames: Its ordered pairs summed over frames, or their lower bound
            when ``pair_frames_bound`` is true.
        pair_frames_bound: Whether ``pair_frames`` is a lower bound.
        frames: The length of its frame extent.
        keypoints: The keypoints of its tracks row.
        fps: The frame rate its indexed media measured.
        build_seconds: The seconds its process worker spent building the feature
            before computing it, ``None`` where the event records no build.
    """

    key: str
    mode: str
    ended_at: float | None
    wall_seconds: float | None
    apply_seconds: float | None
    cpu_seconds: float | None
    input_rows: int | None
    in_flight: int
    peak_grown_bytes: int | None
    ids_per_frame: float | None
    pair_frames: int | None
    pair_frames_bound: bool
    frames: int | None
    keypoints: int | None
    fps: float | None
    build_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class LevelCost:
    """What a run did while a given number of entries were in flight together.

    Throughput at a level is ``rows / seconds``.

    Attributes:
        mode: ``thread`` or ``process``, the mode of the phase that ran it.
        in_flight: The entries in flight, at least 1.
        seconds: The time spent at that count.
        rows: The input rows of the entries over that time, each entry's rows
            spread evenly over its own flight.
        cpu_seconds: The CPU time of the process tree over that time.
        rebuilt: Whether the level was rebuilt from the ``entry_end`` events of a
            log that records none, each entry spanning its time less its
            ``wall_seconds``, rather than read as the run measured it.
    """

    mode: str
    in_flight: int
    seconds: float
    rows: float
    cpu_seconds: float
    rebuilt: bool


@dataclass(frozen=True, slots=True)
class PeakPoint:
    """The memory sample with the most growth over the base, and the work then in flight.

    Attributes:
        mode: ``thread`` or ``process``, the mode of the phase that sampled it.
        grown_bytes: What that sample held over the run's base and its live
            workers' bases.
        entries: The entries in flight at it.
        rows: Their input rows.
        pair_frames: Their ordered pairs summed over frames, ``None`` for a log
            that predates the figure.
        pair_frames_bound: Whether an entry in flight carried a lower bound in
            place of a measured pair count.
    """

    mode: str
    grown_bytes: int
    entries: int
    rows: int
    pair_frames: int | None
    pair_frames_bound: bool


@dataclass(frozen=True, slots=True)
class RunCost:
    """What one attempt cost, read from its run-log.

    The status, ``has_terminal_event`` and the failure fields come from the
    terminal event after the last ``started``; ``kind``, ``target``, ``run_id``,
    ``probe``, ``workers`` and ``parallel_mode`` are the status snapshot's. The
    costs are those of the attempt's last process, the events after its last
    ``started``. ``None`` and 0 mean the log does not say.

    A probe of both modes writes each phase's own ``resources`` events. Every
    scalar describes the whole attempt, from the last ``resources`` event and the
    entries of every phase, except the six that describe the last phase:
    ``workers``, ``parallel_mode``, ``worker_base_rss_bytes``,
    ``worker_base_nonreclaimable_bytes``, ``peak_bytes_per_row`` and
    ``rows_per_second``. ``entries``, ``levels`` and ``peak_points`` hold each
    phase apart, by ``mode``.

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
        workers: How many entries it computed at once in the last phase.
        parallel_mode: How the last phase computed them, ``""`` for one at a time.
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
        worker_base_rss_bytes: One process worker's own memory in the last phase,
            ``None`` when unmeasured.
        peak_nonreclaimable_bytes: The largest memory the kernel cannot reclaim
            that a sample read, ``None`` when unmeasured.
        base_nonreclaimable_bytes: The same before the first entry loaded, ``None``
            when unmeasured.
        worker_base_nonreclaimable_bytes: One process worker's own anonymous and
            shared memory in the last phase, ``None`` when unmeasured.
        peak_bytes_per_row: The last phase's estimate of an entry's peak memory per
            input row, ``None`` until an observation with rows exceeds the fixed part:
            when no entry had input rows, or none rose above the run's base and
            its workers' own memory. It is in the run's measure: the
            non-reclaimable figure when ``base_nonreclaimable_bytes`` is set, else
            the tree's resident memory.
        seconds_per_row: Wall seconds per input row over the entries that ran
            alone.
        alone_cpu_per_wall_second: CPU seconds per wall second over the entries
            that ran alone, one worker's processors, probe or not.
        rows_per_second: The input rows of the last phase's measured entries over
            the wall time of its apply phase.
        cpu_seconds: The CPU time of the process tree since measuring began.
        wall_seconds: The wall time since measuring began.
        cpu_per_wall_second: The two above as a ratio, ``None`` without wall time.
        entries_measured: The computed entries that recorded a cost.
        rows_measured: Their input rows, an entry without the figure adding none.
        largest_entry_rows: The input rows of the largest.
        skipped_lines: The lines of the log that do not parse once NUL bytes are
            stripped.
        entries: The measured ``entry_end`` events of the attempt (``in_flight`` at
            least 1) in file order.
        levels: Each mode's levels, ascending by ``in_flight`` within a mode, from
            the last ``resources`` event of that mode, else rebuilt from its
            entries.
        peak_points: One peak point per mode that sampled an entry in flight.
        params_digest: The hash of the feature's params as they enter its run id,
            ``None`` where the log does not write it.
        params_default: Whether those params equal their class's defaults, ``None``
            wherever ``params_digest`` is ``None``.
        emits: The level the feature declares it emits, ``None`` where the log does
            not write it.
        machine: The machine the attempt's last process ran on, ``None`` where its
            ``started`` event names none.
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
    peak_nonreclaimable_bytes: int | None
    base_nonreclaimable_bytes: int | None
    worker_base_nonreclaimable_bytes: int | None
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
    skipped_lines: int = 0
    entries: tuple[EntryCost, ...] = ()
    levels: tuple[LevelCost, ...] = ()
    peak_points: tuple[PeakPoint, ...] = ()
    params_digest: str | None = None
    params_default: bool | None = None
    emits: str | None = None
    machine: MachineFacts | None = None


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
    read = read_run_log(run_dir, execution_id)
    records = read.records
    snapshot = fold_run_records(execution_id, records)
    if snapshot is None:
        return None
    attempt = last_attempt_records(records)
    ending = _ending(attempt)
    resources = _last_resources(attempt)
    run_mode = _mode(snapshot["parallel_mode"])
    entries = _entries(attempt, run_mode)
    last_phase = [entry for entry in entries if entry.mode == _phase_mode(resources)]
    alone = [entry for entry in entries if entry.in_flight == 1]
    rows_measured = _rows(entries)
    cpu_seconds = record_seconds(resources, "cpu_seconds")
    wall_seconds = record_seconds(resources, "wall_seconds")
    apply_wall_seconds = record_seconds(resources, "apply_wall_seconds")
    error_type, refusal_reason, refusal_cause = _failure(ending)
    params_digest = record_text(resources, "params_digest") or None
    phases = _phases(attempt, run_mode)
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
        peak_nonreclaimable_bytes=(
            record_count(resources, "peak_nonreclaimable_bytes") or None
        ),
        base_nonreclaimable_bytes=(
            record_count(resources, "base_nonreclaimable_bytes") or None
        ),
        worker_base_nonreclaimable_bytes=(
            record_count(resources, "worker_base_nonreclaimable_bytes") or None
        ),
        peak_bytes_per_row=record_count(resources, "peak_bytes_per_row") or None,
        seconds_per_row=_seconds_per_row(alone),
        alone_cpu_per_wall_second=_cpu_per_wall_second(alone),
        rows_per_second=_ratio(_rows(last_phase), apply_wall_seconds),
        cpu_seconds=cpu_seconds,
        wall_seconds=wall_seconds,
        cpu_per_wall_second=_ratio(cpu_seconds, wall_seconds),
        entries_measured=len(entries),
        rows_measured=rows_measured,
        largest_entry_rows=max((entry.input_rows or 0 for entry in entries), default=0),
        skipped_lines=read.skipped_lines,
        entries=tuple(entries),
        levels=tuple(
            level for mode, event in phases for level in _levels(mode, event, entries)
        ),
        peak_points=tuple(
            point
            for mode, event in phases
            if (point := _peak_point(mode, event)) is not None
        ),
        params_digest=params_digest,
        params_default=(
            record_flag(resources, "params_default") if params_digest else None
        ),
        emits=record_text(resources, "emits") or None,
        machine=read_machine(attempt),
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


def _mode(word: str) -> str:
    """``process`` for a process mode, else ``thread``, which a blank mode also reads as."""
    return "process" if word == "process" else "thread"


def _phase_mode(resources: Mapping[str, object]) -> str:
    """The normalized mode of a ``resources`` event's phase."""
    return _mode(record_text(resources, "parallel_mode"))


def _is_measured(rec: Mapping[str, object]) -> bool:
    """Whether *rec* is the ``entry_end`` of an entry that was computed."""
    return rec.get("ev") == "entry_end" and record_count(rec, "in_flight") >= 1


def _entry_mode(rec: Mapping[str, object], run_mode: str) -> str:
    """The normalized mode of an ``entry_end`` event, the run's when it names none."""
    word = record_text(rec, "mode")
    return _mode(word) if word else run_mode


def _entries(records: Sequence[Mapping[str, object]], run_mode: str) -> list[EntryCost]:
    """The computed entries that recorded a cost, in file order.

    A cache hit writes its ``entry_end`` without the fields, and an old log
    never wrote them, so neither is a measurement. An event without a mode reads
    the run's.
    """
    return [
        EntryCost(
            key=record_text(rec, "key"),
            mode=_entry_mode(rec, run_mode),
            ended_at=_epoch_seconds(record_text(rec, "t")),
            wall_seconds=record_optional_seconds(rec, "wall_seconds"),
            apply_seconds=record_optional_seconds(rec, "apply_seconds"),
            cpu_seconds=record_optional_seconds(rec, "cpu_seconds"),
            input_rows=record_optional_count(rec, "input_rows"),
            in_flight=record_count(rec, "in_flight"),
            peak_grown_bytes=record_optional_count(rec, "peak_grown_bytes"),
            ids_per_frame=record_optional_number(rec, "ids_per_frame"),
            pair_frames=record_optional_count(rec, "pair_frames"),
            pair_frames_bound=record_flag(rec, "pair_frames_bound"),
            frames=record_optional_count(rec, "frames"),
            keypoints=record_optional_count(rec, "keypoints"),
            fps=record_optional_number(rec, "fps"),
            build_seconds=record_optional_seconds(rec, "build_seconds"),
        )
        for rec in records
        if _is_measured(rec)
    ]


def _epoch_seconds(stamp: str) -> float | None:
    """An ISO instant as seconds since the epoch, ``None`` when it does not parse."""
    moment = parse_instant(stamp)
    return None if moment is None else moment.timestamp()


def _phases(
    records: Sequence[Mapping[str, object]], run_mode: str
) -> list[tuple[str, Mapping[str, object]]]:
    """Each mode the attempt ran in, with its last ``resources`` event.

    Modes come in the order the log first shows them. A mode whose entries wrote
    no ``resources`` event reads an empty one.
    """
    last: dict[str, Mapping[str, object]] = {}
    for rec in records:
        if rec.get("ev") == "resources":
            last[_phase_mode(rec)] = rec
        elif _is_measured(rec):
            last.setdefault(_entry_mode(rec, run_mode), {})
    return list(last.items())


def _levels(
    mode: str, resources: Mapping[str, object], entries: Sequence[EntryCost]
) -> tuple[LevelCost, ...]:
    """One mode's levels: those its last ``resources`` event wrote, else rebuilt."""
    written = resources.get("in_flight_levels")
    if not isinstance(written, list):
        return _rebuilt_levels(mode, [entry for entry in entries if entry.mode == mode])
    levels = [
        LevelCost(
            mode=mode,
            in_flight=record_count(level, "in_flight"),
            seconds=record_seconds(level, "seconds"),
            rows=record_seconds(level, "rows"),
            cpu_seconds=record_seconds(level, "cpu_seconds"),
            rebuilt=False,
        )
        for level in cast("list[object]", written)
        if is_record(level)
    ]
    return tuple(
        sorted((level for level in levels if level.in_flight >= 1), key=_by_count)
    )


def _by_count(level: LevelCost) -> int:
    """The sort key of a level, the entries in flight."""
    return level.in_flight


def _rebuilt_levels(mode: str, entries: Sequence[EntryCost]) -> tuple[LevelCost, ...]:
    """Levels of a log that wrote none, from the flights its entries ended.

    Each entry spans its ``t`` less its ``wall_seconds`` to its ``t``. An entry
    without a time or a wall time has no span. Each entry's CPU seconds count the
    whole tree, so they are spread evenly over its span, and a segment's CPU is the
    mean over the entries in flight there that recorded one.
    """
    flown = [
        (
            FlightSpan(
                start=entry.ended_at - entry.wall_seconds,
                end=entry.ended_at,
                rows=entry.input_rows or 0,
            ),
            entry.cpu_seconds,
        )
        for entry in entries
        if entry.ended_at is not None and entry.wall_seconds is not None
    ]
    spans = [span for span, _ in flown]

    def cpu_of_segment(start: float, end: float, covering: tuple[int, ...]) -> float:
        shares = [
            cpu * (end - start) / (spans[index].end - spans[index].start)
            for index in covering
            if (cpu := flown[index][1]) is not None
        ]
        return sum(shares) / len(shares) if shares else 0.0

    return tuple(
        LevelCost(
            mode=mode,
            in_flight=level.in_flight,
            seconds=level.seconds,
            rows=level.rows,
            cpu_seconds=level.cpu_seconds,
            rebuilt=True,
        )
        for level in sweep_levels(spans, cpu_of_segment)
    )


def _peak_point(mode: str, resources: Mapping[str, object]) -> PeakPoint | None:
    """The peak sample of a mode's last ``resources`` event, ``None`` without one."""
    entries = record_count(resources, "peak_in_flight_entries")
    if entries == 0:
        return None
    return PeakPoint(
        mode=mode,
        grown_bytes=record_count(resources, "peak_grown_bytes"),
        entries=entries,
        rows=record_count(resources, "peak_in_flight_rows"),
        pair_frames=record_optional_count(resources, "peak_in_flight_pair_frames"),
        pair_frames_bound=record_flag(resources, "peak_in_flight_pair_frames_bound"),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    """*numerator* over *denominator*, ``None`` when the denominator is not positive."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _rows(entries: Sequence[EntryCost]) -> int:
    """The input rows of *entries*, an entry without the figure adding none."""
    return sum(entry.input_rows or 0 for entry in entries)


def _seconds_per_row(entries: Sequence[EntryCost]) -> float | None:
    """Wall seconds per input row over the *entries* that recorded both."""
    both = [
        (entry.wall_seconds, entry.input_rows)
        for entry in entries
        if entry.wall_seconds is not None and entry.input_rows is not None
    ]
    return _ratio(sum(wall for wall, _ in both), sum(rows for _, rows in both))


def _cpu_per_wall_second(entries: Sequence[EntryCost]) -> float | None:
    """CPU seconds per wall second over the *entries* that recorded both."""
    both = [
        (entry.cpu_seconds, entry.wall_seconds)
        for entry in entries
        if entry.cpu_seconds is not None and entry.wall_seconds is not None
    ]
    return _ratio(sum(cpu for cpu, _ in both), sum(wall for _, wall in both))


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
