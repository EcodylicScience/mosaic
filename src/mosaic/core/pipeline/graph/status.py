"""How one attempt and one request stand, as the JSON a poller reads.

:class:`RunStatusWire` reports one attempt, :class:`RequestStatusWire` one pipeline
request and :class:`RequestPageWire` a page of a dataset's requests. ``mosaic status
--json``, ``mosaic pipeline status --json`` and ``mosaic pipeline list --json`` print
these models dumped, and other programmatic callers return the same models.

Each report reads run-logs, through the readers of :mod:`mosaic.runlog`, and the
request's driver lease (:mod:`~.driver_lease`). It judges whether each attempt's
process may still run by :mod:`~.liveness`. A poll costs a few small files. The
module and its imports leave the feature library, pandas, cv2 and av unloaded
(``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Final, Literal, TypeGuard, get_args

from pydantic import TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.runlog import (
    ERROR_MESSAGE_LIMIT,
    TERMINAL_STATUSES,
    AttemptLiveness,
    AttemptOutcome,
    EntryErrorRecord,
    ErrorSummary,
    ProgressEventRecord,
    attempt_outcome,
    error_summary,
    read_entry_errors,
    read_removal_records,
    read_run,
    read_run_progress,
    run_log_dir,
)

from ..error_envelope import ErrorEnvelope, error_envelope
from ..refusal import Refusal, RefusalReason
from ..markers import parse_instant
from .driver_lease import lease_is_live, live_step_process, read_driver_lease
from .liveness import snapshot_liveness, step_liveness
from .model import GraphModel
from .rollup import RequestStatus, request_rollup, request_verdict

if TYPE_CHECKING:
    from pathlib import Path

    from .model import Request
    from .rollup import RequestRollup, StepAttempt

__all__ = [
    "CLEANUP_RUN_SAMPLE",
    "FAILED_ENTRY_SAMPLE",
    "JSON_OBJECT",
    "POLL_AFTER_MAX_SECONDS",
    "POLL_AFTER_MIN_SECONDS",
    "CleanupRunWire",
    "CleanupStatusWire",
    "FailedEntryWire",
    "RecordedErrorWire",
    "RequestPageWire",
    "RequestStatusWire",
    "RequestSummaryWire",
    "RequestVerdict",
    "RunStatusWire",
    "StepStatusWire",
    "cleanup_status_wire",
    "poll_after_seconds",
    "recorded_error",
    "request_status_wire",
    "request_summary",
    "request_verdict_of",
    "rollup_verdict",
    "run_status_wire",
]

FAILED_ENTRY_SAMPLE: Final = 20
"""The most failed entries that one attempt's report lists."""

CLEANUP_RUN_SAMPLE: Final = 20
"""The most removed runs, and the most kept runs, that a cleanup's report lists."""

POLL_AFTER_MIN_SECONDS: Final = 10
"""The fewest seconds a request report asks a poller to wait."""

POLL_AFTER_MAX_SECONDS: Final = 300
"""The most seconds a request report asks a poller to wait."""

type RequestVerdict = Literal[
    "running", "finished", "partial", "failed", "cancelled", "orphaned"
]
"""How a request stands, as a report names it.

The rollup's status, with two derived words. ``partial`` is a finished request
with lost entries. ``orphaned`` is a running request that nothing drives any more:
its driving process let its lease go stale (:mod:`~.driver_lease`) and none of its
step processes may still run, or it has no lease and every step that started and
has not ended is provably dead (:mod:`~.liveness`). Neither is a run-log status.
"""

_REFUSAL_REASONS: Final[frozenset[str]] = frozenset(get_args(RefusalReason.__value__))
_ENVELOPE_KEYS: Final = frozenset({"reason", "step", "message"})
JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(
    dict[str, JsonValue]
)
"""Validates a parsed JSON document as an object."""


class RecordedErrorWire(GraphModel):
    """A failure that an attempt or one of its entries recorded, parsed.

    Attributes:
        type: The exception class name, ``""`` for a refusal or a blob that names
            none.
        message: The sentence a person reads.
        refusal: The refusal as an :class:`ErrorEnvelope` of kind ``refused``, when
            the blob names a ``RefusalReason``. Its detail contains every key of
            the blob beside ``reason``, ``step`` and ``message``.
    """

    type: str
    message: str
    refusal: ErrorEnvelope | None


class FailedEntryWire(GraphModel):
    """One entity that an attempt lost while it continued.

    Attributes:
        key: The entity, as the attempt named it.
        recorded_at: When the attempt recorded the failure.
        error: The failure it recorded.
    """

    key: str
    recorded_at: str
    error: RecordedErrorWire | None


class RunStatusWire(GraphModel):
    """One attempt, as its run-log reports it.

    The fields from ``execution_id`` to ``phase`` are those of
    :class:`~mosaic.runlog.RunLogSnapshot`, which documents them. The rest are
    derived from the same run-log.

    Attributes:
        outcome: ``status`` with ``partial`` for a finished attempt that lost
            entries.
        terminal: Whether the attempt is over.
        progress_fraction: ``progress_done`` over ``progress_total``, ``None``
            until a total is declared.
        error: The attempt's ``error_json`` parsed, ``None`` when it recorded none.
        failed_entries: The first of the entities it lost, at most
            :data:`FAILED_ENTRY_SAMPLE`, in the order it recorded them.
        failed_entries_has_more: Whether it lost more than ``failed_entries``
            lists. ``entries_failed`` counts them all.
        progress: Its progress events (:class:`~mosaic.runlog.ProgressEventRecord`),
            ``None`` unless asked for.
        liveness: Whether its process may still be running (:mod:`~.liveness`).
    """

    execution_id: str
    kind: str
    target: str
    run_id: str
    status: str
    owner: str
    host: str
    pid: int
    created_at: str
    started_at: str
    heartbeat_at: str
    finished_at: str
    error_json: str
    progress_done: int
    progress_total: int
    entries_failed: int
    entries_written: int
    entries_frame_axis_mismatch: int
    entries_frame_tail_short: int
    entries_columns_dropped: int
    cache_hit: bool
    tracks_variant: str
    boot_id: str
    pid_ns: str
    liveness_interval: float
    phase: str
    workers: int
    parallel_mode: str
    peak_rss_bytes: int
    min_available_bytes: int
    total_memory_bytes: int
    outcome: AttemptOutcome
    terminal: bool
    progress_fraction: float | None
    error: RecordedErrorWire | None
    failed_entries: tuple[FailedEntryWire, ...]
    failed_entries_has_more: bool
    progress: tuple[ProgressEventRecord, ...] | None = None
    liveness: AttemptLiveness | None = None


class StepStatusWire(GraphModel):
    """One step of a request, as its attempt's run-log reports it.

    Attributes:
        step: The step in the recipe.
        execution_id: The attempt the request assigned it.
        status: Its run-log status, ``""`` before it writes a run-log.
        run_id: The run it recorded producing, or ``""``.
        entries_failed: How many entities it lost while it continued.
        error_json: The failure it recorded, or ``""``.
        entries_written: How many entries it left with an output row.
        cache_hit: Whether it found the whole of its work already done.
        outcome: ``status`` with ``pending`` before a run-log and ``partial`` for
            a finished attempt that lost entries.
        progress_done: How many entries or epochs it has completed.
        progress_total: How many it declared it would run, 0 until declared.
        progress_fraction: ``progress_done`` over ``progress_total``, ``None``
            until a total is declared.
        error: ``error_json`` parsed, ``None`` when it recorded none.
        failed_entries: The first of the entities it lost, at most
            :data:`FAILED_ENTRY_SAMPLE`.
        failed_entries_has_more: Whether it lost more than ``failed_entries``
            lists.
        host: The host its process recorded, ``""`` before it starts.
        started_at: When that process started, ``""`` before.
        heartbeat_at: When the attempt last wrote a line, ``""`` before.
        phase: The last phase it declared, such as ``fit`` or ``apply``.
        workers: How many entries it computed at once, 0 before it reports.
        parallel_mode: How it computed them, ``""`` for one at a time or before it
            reports.
        peak_rss_bytes: The largest resident memory of its process tree that it
            measured, 0 before it reports. A process-mode worker counts toward it.
        min_available_bytes: The least memory the host had available when it
            measured, 0 before it reports.
        total_memory_bytes: The host's physical memory, 0 before it reports.
        liveness: Whether its process may still be running (:mod:`~.liveness`),
            ``None`` before it writes a run-log.
    """

    step: str
    execution_id: str
    status: str
    run_id: str
    entries_failed: int
    error_json: str
    entries_written: int
    cache_hit: bool
    outcome: AttemptOutcome
    progress_done: int
    progress_total: int
    progress_fraction: float | None
    error: RecordedErrorWire | None
    failed_entries: tuple[FailedEntryWire, ...]
    failed_entries_has_more: bool
    host: str = ""
    started_at: str = ""
    heartbeat_at: str = ""
    phase: str = ""
    workers: int = 0
    parallel_mode: str = ""
    peak_rss_bytes: int = 0
    min_available_bytes: int = 0
    total_memory_bytes: int = 0
    liveness: AttemptLiveness | None = None


class CleanupRunWire(GraphModel):
    """One run a request's cleanup was approved to remove, and how that ended.

    Attributes:
        storage: Its storage name under ``features/``.
        run_id: The run.
        reason: Why it was kept, ``""`` for a run removed.
        detail: What kept it, ``""`` for a run removed.
        bytes: The bytes its directory held, 0 for a run kept.
    """

    storage: str
    run_id: str
    reason: str = ""
    detail: str = ""
    bytes: int = 0


class CleanupStatusWire(GraphModel):
    """A request's cleanup of superseded runs, as its run-log reports it.

    Attributes:
        execution_id: The cleanup's attempt.
        status: ``pending`` while a step has not finished, ``running`` until
            its run-log ends, then ``finished``, ``failed`` or ``cancelled``.
            ``skipped`` when a step did not finish, so the cleanup never ran.
        approved_runs: How many runs the submission approved for removal.
        removed_runs: How many it removed.
        removed_bytes: The bytes they held.
        kept_runs: How many approved runs it kept, each for a reason.
        failed_runs: How many removals raised.
        interrupted_runs: How many removals began and did not end, as when its
            process was killed. Running the request again finishes them.
        removed: The first removed runs, at most :data:`CLEANUP_RUN_SAMPLE`.
        removed_omitted: How many more were removed.
        kept: The first kept runs with their reasons, at most
            :data:`CLEANUP_RUN_SAMPLE`.
        kept_omitted: How many more were kept.
        error: Why the cleanup as a whole failed, parsed, or ``None``.
    """

    execution_id: str
    status: Literal["pending", "running", "finished", "failed", "cancelled", "skipped"]
    approved_runs: int
    removed_runs: int = 0
    removed_bytes: int = 0
    kept_runs: int = 0
    failed_runs: int = 0
    interrupted_runs: int = 0
    removed: tuple[CleanupRunWire, ...] = ()
    removed_omitted: int = 0
    kept: tuple[CleanupRunWire, ...] = ()
    kept_omitted: int = 0
    error: RecordedErrorWire | None = None


class RequestStatusWire(GraphModel):
    """One pipeline request, as its steps' run-logs report it.

    Attributes:
        request_id: Which request this is.
        status: The rollup's one-shot status (``request_rollup``).
        terminal: Whether that status is final.
        verdict: ``status`` with the derived words of :data:`RequestVerdict`.
        steps: One report per step, in the request's order.
        poll_after_seconds: How long a poller should wait before reading the
            request again (:func:`poll_after_seconds`), ``None`` once nothing
            will change it by itself.
        cleanup: The request's removal of superseded runs, ``None`` for a
            request without one.
    """

    request_id: str
    status: RequestStatus
    terminal: bool
    verdict: RequestVerdict
    steps: tuple[StepStatusWire, ...]
    poll_after_seconds: int | None = None
    cleanup: CleanupStatusWire | None = None


class RequestSummaryWire(GraphModel):
    """One pipeline request in a listing, without its per-step reports.

    Attributes:
        request_id: Which request this is.
        recipe_digest: The recipe it submitted.
        owner: Who submitted it, ``""`` when unrecorded.
        created_at: When it was submitted, ``""`` when unrecorded.
        step_count: How many steps it assigned an attempt.
        status: The rollup's one-shot status (``request_rollup``).
        terminal: Whether that status is final.
        verdict: ``status`` with the derived words of :data:`RequestVerdict`.
    """

    request_id: str
    recipe_digest: str
    owner: str
    created_at: str
    step_count: int
    status: RequestStatus
    terminal: bool
    verdict: RequestVerdict


class RequestPageWire(GraphModel):
    """One page of a dataset's pipeline requests, newest first.

    Attributes:
        items: The requests on this page.
        next_after: The ``after`` that reads the next page, ``None`` on the last
            page. It is the request id of the last request on this page.
    """

    items: tuple[RequestSummaryWire, ...]
    next_after: str | None


def recorded_error(error_json: str) -> RecordedErrorWire | None:
    """Parse an ``error_json`` blob that an attempt or an entry recorded.

    :func:`~mosaic.runlog.error_summary` reads the type and the message. A blob
    naming a ``RefusalReason`` is a refusal, and :func:`error_envelope` renders it
    as an :class:`ErrorEnvelope`. A traceback is left out, and the raw blob still
    contains it.

    Args:
        error_json: The blob, ``""`` for an attempt or entry that recorded none.

    Returns:
        The parsed failure, or ``None`` for an empty blob.
    """
    return _recorded_error(error_json, error_summary(error_json))


def run_status_wire(
    base_dir: Path | str,
    execution_id: str,
    *,
    progress: bool = False,
    now: datetime.datetime | None = None,
) -> RunStatusWire | None:
    """Report one attempt from its run-log.

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        execution_id: The attempt.
        progress: Whether to list its progress events.
        now: The current instant that liveness is judged at. ``None`` reads the
            clock.

    Returns:
        The report, or ``None`` when the attempt's run-log is absent or
        unreadable.
    """
    logs = run_log_dir(base_dir)
    snapshot = read_run(logs, execution_id)
    if snapshot is None:
        return None
    failed, has_more = _failed_entries(logs, execution_id, snapshot["entries_failed"])
    events = tuple(read_run_progress(logs, execution_id)) if progress else None
    return RunStatusWire(
        **snapshot,
        outcome=attempt_outcome(snapshot["status"], snapshot["entries_failed"]),
        terminal=snapshot["status"] in TERMINAL_STATUSES,
        progress_fraction=_fraction(
            snapshot["progress_done"], snapshot["progress_total"]
        ),
        error=recorded_error(snapshot["error_json"]),
        failed_entries=failed,
        failed_entries_has_more=has_more,
        progress=events,
        liveness=snapshot_liveness(snapshot, now=now),
    )


def request_status_wire(
    base_dir: Path | str,
    request: Request,
    *,
    now: datetime.datetime | None = None,
) -> RequestStatusWire:
    """Report one request from the run-logs of the attempts it assigned.

    The verdict also reads the request's driver lease (:func:`request_verdict_of`).

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        request: The submission to report on.
        now: The current instant that liveness and the poll interval are judged
            at. ``None`` reads the clock.

    Returns:
        The request's status, its verdict, one report per step and when to poll
        again.
    """
    logs = run_log_dir(base_dir)
    rollup = request_rollup(base_dir, request)
    verdict = request_verdict_of(base_dir, rollup, now=now)
    return RequestStatusWire(
        request_id=rollup.request_id,
        status=rollup.status,
        terminal=rollup.is_terminal,
        verdict=verdict,
        steps=tuple(_step_status(logs, attempt, now=now) for attempt in rollup.steps),
        poll_after_seconds=poll_after_seconds(rollup, verdict, now=now),
        cleanup=cleanup_status_wire(base_dir, request, rollup),
    )


def cleanup_status_wire(
    base_dir: Path | str, request: Request, rollup: RequestRollup
) -> CleanupStatusWire | None:
    """Report *request*'s cleanup from its run-log, ``None`` for a request without one.

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        request: The submission.
        rollup: Its rollup (:func:`~.rollup.request_rollup`).
    """
    spec = request.cleanup
    attempt = rollup.cleanup
    if spec is None or attempt is None:
        return None
    steps = request_verdict(rollup.steps)
    if attempt.terminal:
        status = attempt.status
    elif attempt.started:
        status = "running"
    elif steps in ("failed", "cancelled"):
        status = "skipped"
    else:
        status = "pending"
    records = read_removal_records(run_log_dir(base_dir), spec.execution_id)
    removed = [record for record in records if record["state"] == "removed"]
    kept = [record for record in records if record["state"] == "kept"]
    return CleanupStatusWire(
        execution_id=spec.execution_id,
        status=_cleanup_status(status),
        approved_runs=len(spec.approved),
        removed_runs=len(removed),
        removed_bytes=sum(record["bytes"] for record in removed),
        kept_runs=len(kept),
        failed_runs=sum(record["state"] == "failed" for record in records),
        interrupted_runs=sum(record["state"] == "interrupted" for record in records),
        removed=tuple(
            CleanupRunWire(
                storage=record["feature"],
                run_id=record["run_id"],
                bytes=record["bytes"],
            )
            for record in removed[:CLEANUP_RUN_SAMPLE]
        ),
        removed_omitted=max(len(removed) - CLEANUP_RUN_SAMPLE, 0),
        kept=tuple(
            CleanupRunWire(
                storage=record["feature"],
                run_id=record["run_id"],
                reason=record["reason"],
                detail=record["detail"],
            )
            for record in kept[:CLEANUP_RUN_SAMPLE]
        ),
        kept_omitted=max(len(kept) - CLEANUP_RUN_SAMPLE, 0),
        error=recorded_error(attempt.error_json),
    )


def _cleanup_status(
    status: str,
) -> Literal["pending", "running", "finished", "failed", "cancelled", "skipped"]:
    """*status* narrowed to a cleanup status, ``running`` for any other word."""
    match status:
        case "pending" | "finished" | "failed" | "cancelled" | "skipped":
            return status
        case _:
            return "running"


def request_summary(base_dir: Path | str, request: Request) -> RequestSummaryWire:
    """Summarize one request for a listing, reading one run-log per step.

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        request: The submission to summarize.

    Returns:
        The request's identity, its status and its verdict, as for
        :func:`request_status_wire`.
    """
    rollup = request_rollup(base_dir, request)
    return RequestSummaryWire(
        request_id=request.request_id,
        recipe_digest=request.recipe_digest,
        owner=request.owner,
        created_at=request.created_at,
        step_count=len(request.step_executions),
        status=rollup.status,
        terminal=rollup.is_terminal,
        verdict=request_verdict_of(base_dir, rollup),
    )


def request_verdict_of(
    base_dir: Path | str,
    rollup: RequestRollup,
    *,
    now: datetime.datetime | None = None,
) -> RequestVerdict:
    """Return the verdict of *rollup*, with the request's driver lease read from disk.

    A running request with a stale lease reads ``orphaned`` only when none of its
    step processes may still run (:func:`~.driver_lease.live_step_process`). A
    killed driver leaves its step running in a session of its own, and that step
    still computes the request's work.

    A running request without a lease, as a queue or a shell loop runs one, reads
    ``orphaned`` only on positive evidence: at least one step has started and not
    ended, and every such step is ``dead`` (:func:`~.liveness.step_liveness`). A
    step of ``unknown`` liveness keeps it ``running``, and so does a request
    between two steps.

    Args:
        base_dir: The dataset root containing ``.mosaic/drivers``.
        rollup: The request's rollup.
        now: The current instant. ``None`` reads the clock.
    """
    if rollup.is_terminal:
        return rollup_verdict(rollup)
    lease = read_driver_lease(base_dir, rollup.request_id)
    if lease is None:
        unended = [
            attempt
            for attempt in rollup.steps
            if attempt.started and not attempt.terminal
        ]
        if unended and all(
            step_liveness(attempt, now=now) == "dead" for attempt in unended
        ):
            return rollup_verdict(rollup, driver_alive=False)
        return rollup_verdict(rollup)
    alive = (
        lease_is_live(lease, now=now)
        or live_step_process(rollup, lease, now=now) is not None
    )
    return rollup_verdict(rollup, driver_alive=alive)


def poll_after_seconds(
    rollup: RequestRollup,
    verdict: RequestVerdict,
    *,
    now: datetime.datetime | None = None,
) -> int | None:
    """Return how many seconds a poller should wait before reading a request again.

    A request that is over, or ``orphaned``, changes only when someone acts, and
    has no interval. Before any step runs the interval is the shortest. While a
    step runs it is half the time that step's progress says is left, or a tenth of
    the time it has run when it reports no progress, between
    :data:`POLL_AFTER_MIN_SECONDS` and :data:`POLL_AFTER_MAX_SECONDS`. Of several
    running steps the shortest interval wins.

    Args:
        rollup: The request's rollup.
        verdict: Its verdict (:func:`request_verdict_of`).
        now: The current instant. ``None`` reads the clock.
    """
    if rollup.is_terminal or verdict == "orphaned":
        return None
    moment = now or datetime.datetime.now(datetime.UTC)
    waits: list[float] = []
    for attempt in rollup.steps:
        if not attempt.started or attempt.terminal:
            continue
        began = parse_instant(attempt.started_at)
        if began is None:
            continue
        elapsed = max((moment - began).total_seconds(), 0.0)
        done, total = attempt.progress_done, attempt.progress_total
        if 0 < done < total:
            waits.append(elapsed * (total - done) / done / 2)
        else:
            waits.append(elapsed / 10)
    wait = min(waits, default=float(POLL_AFTER_MIN_SECONDS))
    return int(min(max(wait, POLL_AFTER_MIN_SECONDS), POLL_AFTER_MAX_SECONDS))


def rollup_verdict(
    rollup: RequestRollup, *, driver_alive: bool | None = None
) -> RequestVerdict:
    """Return the verdict of *rollup*: its status, with ``orphaned`` and ``partial``.

    Args:
        rollup: The request's rollup.
        driver_alive: Whether the request's driver, or a process of one of its
            steps, may still be running (:func:`request_verdict_of`). ``None``
            when the request has no lease, and the verdict then follows the
            run-logs alone. ``False`` turns a running request into ``orphaned``.
    """
    if rollup.status == "running" and driver_alive is False:
        return "orphaned"
    if rollup.status == "finished" and any(
        attempt.entries_failed for attempt in rollup.steps
    ):
        return "partial"
    return rollup.status


def _step_status(
    logs: Path, attempt: StepAttempt, *, now: datetime.datetime | None = None
) -> StepStatusWire:
    """One step's report, from its attempt's rollup record."""
    failed, has_more = _failed_entries(
        logs, attempt.execution_id, attempt.entries_failed
    )
    return StepStatusWire(
        step=attempt.step_id,
        execution_id=attempt.execution_id,
        status=attempt.status,
        run_id=attempt.run_id,
        entries_failed=attempt.entries_failed,
        error_json=attempt.error_json,
        entries_written=attempt.entries_written,
        cache_hit=attempt.cache_hit,
        outcome=attempt_outcome(attempt.status, attempt.entries_failed),
        progress_done=attempt.progress_done,
        progress_total=attempt.progress_total,
        progress_fraction=_fraction(attempt.progress_done, attempt.progress_total),
        error=recorded_error(attempt.error_json),
        failed_entries=failed,
        failed_entries_has_more=has_more,
        host=attempt.host,
        started_at=attempt.started_at,
        heartbeat_at=attempt.heartbeat_at,
        phase=attempt.phase,
        workers=attempt.workers,
        parallel_mode=attempt.parallel_mode,
        peak_rss_bytes=attempt.peak_rss_bytes,
        min_available_bytes=attempt.min_available_bytes,
        total_memory_bytes=attempt.total_memory_bytes,
        liveness=step_liveness(attempt, now=now) if attempt.started else None,
    )


def _failed_entries(
    logs: Path, execution_id: str, entries_failed: int
) -> tuple[tuple[FailedEntryWire, ...], bool]:
    """The first failed entries of one attempt, and whether it lost more.

    The run-log is read again only when the attempt counts a failed entry.
    """
    if entries_failed <= 0:
        return (), False
    page = read_entry_errors(logs, execution_id, limit=FAILED_ENTRY_SAMPLE)
    return tuple(failed_entry(record) for record in page["items"]), page["has_more"]


def failed_entry(record: EntryErrorRecord) -> FailedEntryWire:
    """Return one ``entry_error`` record of a run-log as a :class:`FailedEntryWire`."""
    summary: ErrorSummary = {"type": record["error_type"], "message": record["message"]}
    return FailedEntryWire(
        key=record["key"],
        recorded_at=record["recorded_at"],
        error=_recorded_error(record["error_json"], summary),
    )


def _recorded_error(error_json: str, summary: ErrorSummary) -> RecordedErrorWire | None:
    """*error_json* as a :class:`RecordedErrorWire`, with *summary* read from it."""
    if not error_json.strip():
        return None
    refusal = _recorded_refusal(error_json, summary["message"])
    if refusal is None:
        return RecordedErrorWire(
            type=summary["type"], message=summary["message"], refusal=None
        )
    return RecordedErrorWire(
        type="", message=summary["message"], refusal=error_envelope(refusal)
    )


def _recorded_refusal(error_json: str, message: str) -> Refusal | None:
    """The refusal that *error_json* records, ``None`` for any other blob."""
    try:
        blob = JSON_OBJECT.validate_json(error_json)
    except ValidationError:
        return None
    reason = blob.get("reason")
    if not _is_refusal_reason(reason):
        return None
    return Refusal(
        message,
        reason=reason,
        step_id=_text(blob.get("step")),
        detail={key: value for key, value in blob.items() if key not in _ENVELOPE_KEYS},
    )


def _fraction(done: int, total: int) -> float | None:
    """*done* over *total*, at most 1, or ``None`` before a total is declared."""
    if total <= 0:
        return None
    return min(done / total, 1.0)


def _is_refusal_reason(value: JsonValue | None) -> TypeGuard[RefusalReason]:
    """Whether a recorded ``reason`` names a ``RefusalReason``."""
    return isinstance(value, str) and value in _REFUSAL_REASONS


def _text(value: JsonValue | None) -> str:
    """A string field of a recorded blob, ``""`` when absent or not a string."""
    return value[:ERROR_MESSAGE_LIMIT] if isinstance(value, str) else ""
