"""The entities that one attempt lost, each with its failure record.

A run-log records why each entity failed, and the failure records under
``<dataset>/.mosaic/claims`` record how often it has failed and whether a request
still attempts it. :func:`entry_errors_wire` joins the two for one attempt.
``mosaic runs errors --json`` prints the :class:`EntryErrorsWire` that it returns.

The join reads one run-log and up to one failure record per lost entity. Importing this
module leaves pandas, cv2, av and the feature library unloaded
(``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mosaic.core.entry import parse_entry_key
from mosaic.runlog import (
    AttemptOutcome,
    attempt_outcome,
    read_entry_error_keys,
    read_entry_errors,
    read_run,
    run_log_dir,
)

from .claims import FileFailureStore
from .model import GraphModel
from .status import FAILED_ENTRY_SAMPLE, FailedEntryWire, failed_entry

if TYPE_CHECKING:
    import datetime
    from pathlib import Path

    from .failures import FailureRecord, FailureStore

__all__ = ["EntryErrorWire", "EntryErrorsWire", "RetryState", "entry_errors_wire"]

type RetryState = Literal["unrecorded", "ready", "waiting", "quarantined"]
"""Whether a pipeline request attempts a failed entity again.

``unrecorded``
    The record counts zero failures. A later attempt succeeded and cleared it,
    or the attempt keeps no record of the entity. Attempts outside a pipeline
    request and op steps record no entity failures. A pipeline step counts the
    failure of an entity outside the scope that it asked for against the step.
``ready``
    The next attempt of its step retries it.
``waiting``
    It failed recently. The next attempt leaves it out until ``not_before``, and
    the one after retries it.
``quarantined``
    It failed ``QUARANTINE_AFTER`` times and is not attempted again. Its step is
    refused for a coverage shortfall until a request with ``allow_partial``
    proceeds without it.
"""


class EntryErrorWire(FailedEntryWire):
    """One entity that an attempt lost, with the failure record of its run.

    Attributes:
        group: The entity's group, parsed from ``key``.
        sequence: The entity's sequence, parsed from ``key``.
        attempts: How many failures the record counts, across requests.
        not_before: When a request may attempt it again, ISO-8601 UTC, or ``""``.
        last_execution_id: The attempt that recorded the latest failure, or ``""``.
        retry: Whether a request attempts it again.
    """

    group: str
    sequence: str
    attempts: int
    not_before: str
    last_execution_id: str
    retry: RetryState


class EntryErrorsWire(GraphModel):
    """The entities that one attempt lost while it continued.

    Attributes:
        execution_id: The attempt.
        kind: Its job kind.
        target: Its run-log target, the storage name for a feature run.
        run_id: The run it recorded producing, or ``""``.
        status: Its run-log status.
        outcome: ``status`` with ``partial`` for a finished attempt that lost
            entries.
        total: How many entities it lost.
        has_more: Whether *total* exceeds the entities listed.
        entries: The first entities it lost, in the order it recorded them.
        held_back_by_quarantine: Whether any entity that the attempt lost is
            quarantined, listed or not. A quarantined entity keeps its step back
            until a request with ``allow_partial`` proceeds without it. Every lost
            entity also leaves the run short, and a ``scope_dependent`` step that
            reads the run is refused until a request with ``allow_partial``
            answers that shortfall.
    """

    execution_id: str
    kind: str
    target: str
    run_id: str
    status: str
    outcome: AttemptOutcome
    total: int
    has_more: bool
    entries: tuple[EntryErrorWire, ...]
    held_back_by_quarantine: bool


def entry_errors_wire(
    base_dir: Path | str,
    execution_id: str,
    *,
    limit: int = FAILED_ENTRY_SAMPLE,
    store: FailureStore | None = None,
    now: datetime.datetime | None = None,
) -> EntryErrorsWire | None:
    """Report the entities that one attempt lost, each with its failure record.

    A failure record is keyed by the attempt's ``target`` and ``run_id`` and the
    entity's group and sequence, as a pipeline step records it. Only the listed
    entities parse their recorded blobs. The records of unlisted entities are read
    until one of them is quarantined.

    Args:
        base_dir: The dataset root containing ``.mosaic``.
        execution_id: The attempt.
        limit: The most entities to list, at least 1.
        store: Where the failure records are read. The dataset's records when
            ``None``.
        now: The moment that decides whether an entity still waits. The current
            time when ``None``.

    Returns:
        The report, or ``None`` when the attempt's run-log is absent or
        unreadable.

    Raises:
        ValueError: *limit* is below 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    logs = run_log_dir(base_dir)
    snapshot = read_run(logs, execution_id)
    if snapshot is None:
        return None
    target, run_id = snapshot["target"], snapshot["run_id"]
    page = read_entry_errors(logs, execution_id, limit=limit)
    records = FileFailureStore(base_dir) if store is None else store
    entries: list[EntryErrorWire] = []
    for record in page["items"]:
        failed = failed_entry(record)
        group, sequence = parse_entry_key(failed.key)
        failure = records.entry_record((target, run_id, group, sequence))
        entries.append(
            EntryErrorWire(
                key=failed.key,
                recorded_at=failed.recorded_at,
                error=failed.error,
                group=group,
                sequence=sequence,
                attempts=failure.attempts,
                not_before=failure.not_before,
                last_execution_id=failure.last_execution_id,
                retry=_retry_state(failure, now=now),
            )
        )
    unlisted = (
        read_entry_error_keys(logs, execution_id)[limit:] if page["has_more"] else []
    )
    quarantined = any(entry.retry == "quarantined" for entry in entries) or any(
        records.entry_record((target, run_id, *parse_entry_key(key))).quarantined
        for key in unlisted
    )
    return EntryErrorsWire(
        execution_id=execution_id,
        kind=snapshot["kind"],
        target=target,
        run_id=run_id,
        status=snapshot["status"],
        outcome=attempt_outcome(snapshot["status"], snapshot["entries_failed"]),
        total=page["total"],
        has_more=page["has_more"],
        entries=tuple(entries),
        held_back_by_quarantine=quarantined,
    )


def _retry_state(
    failure: FailureRecord, *, now: datetime.datetime | None
) -> RetryState:
    """Whether a request attempts the entity of *failure* again."""
    if failure.attempts == 0:
        return "unrecorded"
    if failure.quarantined:
        return "quarantined"
    return "ready" if failure.ready(now=now) else "waiting"
