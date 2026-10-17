"""How one attempt and one request stand, as the JSON a poller reads.

:class:`RunStatusWire` reports one attempt, :class:`RequestStatusWire` one pipeline
request and :class:`RequestPageWire` a page of a dataset's requests. ``mosaic status
--json``, ``mosaic pipeline status --json`` and ``mosaic pipeline list --json`` print
these models dumped, and other programmatic callers return the same models.

Each report reads run-logs alone, through the readers of :mod:`mosaic.runlog`. A
poll costs a few small files. The module and its imports leave the feature library,
pandas, cv2 and av unloaded (``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypeGuard, get_args

from pydantic import TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.runlog import (
    ERROR_MESSAGE_LIMIT,
    TERMINAL_STATUSES,
    AttemptOutcome,
    EntryErrorRecord,
    ErrorSummary,
    ProgressEventRecord,
    attempt_outcome,
    error_summary,
    read_entry_errors,
    read_run,
    read_run_progress,
    run_log_dir,
)

from ..error_envelope import ErrorEnvelope, error_envelope
from ..refusal import Refusal, RefusalReason
from .driver_lease import lease_is_live, live_step_process, read_driver_lease
from .model import GraphModel
from .rollup import RequestStatus, request_rollup

if TYPE_CHECKING:
    from pathlib import Path

    from .model import Request
    from .rollup import RequestRollup, StepAttempt

__all__ = [
    "FAILED_ENTRY_SAMPLE",
    "JSON_OBJECT",
    "FailedEntryWire",
    "RecordedErrorWire",
    "RequestPageWire",
    "RequestStatusWire",
    "RequestSummaryWire",
    "RequestVerdict",
    "RunStatusWire",
    "StepStatusWire",
    "recorded_error",
    "request_status_wire",
    "request_summary",
    "request_verdict_of",
    "rollup_verdict",
    "run_status_wire",
]

FAILED_ENTRY_SAMPLE: Final = 20
"""The most failed entries that one attempt's report lists."""

type RequestVerdict = Literal[
    "running", "finished", "partial", "failed", "cancelled", "orphaned"
]
"""How a request stands, as a report names it.

The rollup's status, with two derived words. ``partial`` is a finished request
with lost entries. ``orphaned`` is a running request whose driving process let its
lease go stale (:mod:`~.driver_lease`) and none of whose step processes may still
run. Neither is a run-log status.
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
    outcome: AttemptOutcome
    terminal: bool
    progress_fraction: float | None
    error: RecordedErrorWire | None
    failed_entries: tuple[FailedEntryWire, ...]
    failed_entries_has_more: bool
    progress: tuple[ProgressEventRecord, ...] | None = None


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


class RequestStatusWire(GraphModel):
    """One pipeline request, as its steps' run-logs report it.

    Attributes:
        request_id: Which request this is.
        status: The rollup's one-shot status (``request_rollup``).
        terminal: Whether that status is final.
        verdict: ``status`` with the derived words of :data:`RequestVerdict`.
        steps: One report per step, in the request's order.
    """

    request_id: str
    status: RequestStatus
    terminal: bool
    verdict: RequestVerdict
    steps: tuple[StepStatusWire, ...]


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
    base_dir: Path | str, execution_id: str, *, progress: bool = False
) -> RunStatusWire | None:
    """Report one attempt from its run-log.

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        execution_id: The attempt.
        progress: Whether to list its progress events.

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
    )


def request_status_wire(base_dir: Path | str, request: Request) -> RequestStatusWire:
    """Report one request from the run-logs of the attempts it assigned.

    The verdict also reads the request's driver lease (:func:`request_verdict_of`).

    Args:
        base_dir: The dataset root containing ``.mosaic/runs``.
        request: The submission to report on.

    Returns:
        The request's status, its verdict and one report per step.
    """
    logs = run_log_dir(base_dir)
    rollup = request_rollup(base_dir, request)
    return RequestStatusWire(
        request_id=rollup.request_id,
        status=rollup.status,
        terminal=rollup.is_terminal,
        verdict=request_verdict_of(base_dir, rollup),
        steps=tuple(_step_status(logs, attempt) for attempt in rollup.steps),
    )


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


def request_verdict_of(base_dir: Path | str, rollup: RequestRollup) -> RequestVerdict:
    """Return the verdict of *rollup*, with the request's driver lease read from disk.

    A running request with a stale lease reads ``orphaned`` only when none of its
    step processes may still run (:func:`~.driver_lease.live_step_process`). A
    killed driver leaves its step running in a session of its own, and that step
    still computes the request's work.

    Args:
        base_dir: The dataset root containing ``.mosaic/drivers``.
        rollup: The request's rollup.
    """
    if rollup.is_terminal:
        return rollup_verdict(rollup)
    lease = read_driver_lease(base_dir, rollup.request_id)
    if lease is None:
        return rollup_verdict(rollup)
    alive = lease_is_live(lease) or live_step_process(rollup, lease) is not None
    return rollup_verdict(rollup, driver_alive=alive)


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


def _step_status(logs: Path, attempt: StepAttempt) -> StepStatusWire:
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
