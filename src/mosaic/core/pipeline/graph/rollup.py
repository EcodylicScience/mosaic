"""Whether one request is over, and how it ended.

**A request is one-shot.** It closes on a terminal rollup and is then no longer
walked, so deleting an artifact afterwards rebuilds nothing. Re-running is a new
request, which is cheap and incremental because everything still on disk is a
cache hit -- and the inventory is what makes the gap visible in the meantime. A
standing request, continuously maintained, is a calling policy rather than a
redesign, and it is deliberately not this: it would recreate the 200 GB output
somebody deleted to free space.

**A branch that failed while a sibling is still running is not terminal yet.**
That is the whole subtlety here. Reporting the request failed the moment one step
did would close it while work is in flight, and whatever that work produces would
land under a request nothing is watching. A local driver runs the steps that do
not descend from a failure. A failed request with an unstarted step therefore
reads ``running`` while its driver lease exists, and a stale lease then reads
``orphaned`` (:func:`~.status.request_verdict_of`).

**This reads attempts, not artifacts.** Whether every step's *work* is done is a
different question, answered by planning the recipe against the dataset -- which
costs the feature registry and is what ``mosaic pipeline plan`` is for. This one
costs a directory of small JSON files and, for a failed request, its driver
lease. Whatever polls it can poll it often.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mosaic.runlog import TERMINAL_STATUSES, read_run, run_log_dir

from .driver_lease import read_driver_lease

if TYPE_CHECKING:
    from pathlib import Path

    from .model import Request

__all__ = [
    "RequestRollup",
    "RequestStatus",
    "StepAttempt",
    "request_rollup",
    "request_verdict",
]

type RequestStatus = Literal["running", "finished", "failed", "cancelled"]
"""How a one-shot request ended, or that it has not.

``running`` covers "not started" too, deliberately: a request whose steps have no
run-logs yet is a request in flight, and a separate value for it would be a
stored status by another name -- the thing derived state exists to avoid.
"""


@dataclass(frozen=True, slots=True)
class StepAttempt:
    """What one step's attempt has recorded so far.

    Attributes:
        step_id: The step in the recipe.
        execution_id: The attempt the request assigned it.
        status: What its run-log says, or ``""`` where it has not written one.
        run_id: What it recorded producing, or ``""``.
        error_json: What it recorded going wrong, or ``""``.
        entries_failed: How many entities it lost while carrying on.
        entries_written: How many entries it left holding an output row.
        cache_hit: Whether it found the whole of its work already done.
        progress_done: How many entries or epochs it has completed.
        progress_total: How many it declared it would run, 0 until declared.
        host: The host its process recorded starting on, ``""`` before.
        pid: That process, 0 before it started or for an attempt recorded by
            another process.
        started_at: When that process started, ISO-8601 UTC.
        heartbeat_at: When the attempt last wrote a line, ISO-8601 UTC.
        boot_id: The boot of the host its process recorded, ``""`` when
            unrecorded.
        pid_ns: The pid namespace of that process, ``""`` when unrecorded.
        liveness_interval: The seconds between the ``alive`` events that the
            process promised, 0 for none.
        phase: The last phase the attempt declared, ``""`` before one.
        workers: How many entries it computed at once, 0 before it reports.
        parallel_mode: How it computed them, ``""`` for one at a time or before it
            reports.
        peak_rss_bytes: The largest resident memory of its process tree that it
            measured, 0 before it reports.
        min_available_bytes: The least memory the host had available when it
            measured, 0 before it reports.
        total_memory_bytes: The host's physical memory, 0 before it reports.
    """

    step_id: str
    execution_id: str
    status: str = ""
    run_id: str = ""
    error_json: str = ""
    entries_failed: int = 0
    entries_written: int = 0
    cache_hit: bool = False
    progress_done: int = 0
    progress_total: int = 0
    host: str = ""
    pid: int = 0
    started_at: str = ""
    heartbeat_at: str = ""
    boot_id: str = ""
    pid_ns: str = ""
    liveness_interval: float = 0.0
    phase: str = ""
    workers: int = 0
    parallel_mode: str = ""
    peak_rss_bytes: int = 0
    min_available_bytes: int = 0
    total_memory_bytes: int = 0

    @property
    def started(self) -> bool:
        """Has this step written anything at all?"""
        return bool(self.status)

    @property
    def terminal(self) -> bool:
        """Is this attempt over, whichever way it went?"""
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class RequestRollup:
    """One request, as its steps' attempts describe it.

    Attributes:
        request_id: Which request this is.
        status: The one-shot verdict.
        steps: One record per step, in the request's own order.
        cleanup: The attempt of the request's cleanup, with an empty
            ``step_id``, or ``None`` for a request without one.
    """

    request_id: str
    status: RequestStatus
    steps: tuple[StepAttempt, ...] = ()
    cleanup: StepAttempt | None = None

    @property
    def is_terminal(self) -> bool:
        """Is this request over?"""
        return self.status != "running"


def request_rollup(
    base_dir: Path | str, request: Request, *, read_lease: bool = True
) -> RequestRollup:
    """How *request* stands, from the run-logs of the attempts it assigned.

    A request with a cleanup whose steps all finished stays ``running`` until
    the cleanup's run-log ends, whichever way. A failed or cancelled cleanup
    leaves the verdict to the steps, and :func:`request_verdict` itself reads
    the steps alone.

    A failed request with a step that has not started stays ``running`` while
    the request's driver lease file exists, because its driver still starts
    the steps that do not descend from the failure. The lease need not be live.

    Args:
        base_dir: The dataset root holding ``.mosaic/runs``.
        request: The submission to report on.
        read_lease: ``False`` leaves the driver lease unread. A driver that has
            ended its steps passes it to decide whether to release the lease.

    Returns:
        The verdict and the per-step attempts behind it.
    """
    logs = run_log_dir(base_dir)
    steps = tuple(
        _attempt(logs, step_id, execution_id)
        for step_id, execution_id in request.step_executions.items()
    )
    status = request_verdict(steps)
    cleanup = (
        None
        if request.cleanup is None
        else _attempt(logs, "", request.cleanup.execution_id)
    )
    if status == "finished" and cleanup is not None and not cleanup.terminal:
        status = "running"
    if (
        status == "failed"
        and read_lease
        and any(not attempt.started for attempt in steps)
        and read_driver_lease(base_dir, request.request_id) is not None
    ):
        status = "running"
    return RequestRollup(
        request_id=request.request_id, status=status, steps=steps, cleanup=cleanup
    )


def _attempt(logs: Path, step_id: str, execution_id: str) -> StepAttempt:
    """One step's attempt, or an empty record where it has not written one."""
    snapshot = read_run(logs, execution_id)
    if snapshot is None:
        return StepAttempt(step_id=step_id, execution_id=execution_id)
    return StepAttempt(
        step_id=step_id,
        execution_id=execution_id,
        status=snapshot["status"],
        run_id=snapshot["run_id"],
        error_json=snapshot["error_json"],
        entries_failed=snapshot["entries_failed"],
        entries_written=snapshot["entries_written"],
        cache_hit=snapshot["cache_hit"],
        progress_done=snapshot["progress_done"],
        progress_total=snapshot["progress_total"],
        host=snapshot["host"],
        pid=snapshot["pid"],
        started_at=snapshot["started_at"],
        heartbeat_at=snapshot["heartbeat_at"],
        boot_id=snapshot["boot_id"],
        pid_ns=snapshot["pid_ns"],
        liveness_interval=snapshot["liveness_interval"],
        phase=snapshot["phase"],
        workers=snapshot["workers"],
        parallel_mode=snapshot["parallel_mode"],
        peak_rss_bytes=snapshot["peak_rss_bytes"],
        min_available_bytes=snapshot["min_available_bytes"],
        total_memory_bytes=snapshot["total_memory_bytes"],
    )


def request_verdict(steps: tuple[StepAttempt, ...]) -> RequestStatus:
    """The one-shot verdict over *steps*, by the rules in the order they decide.

    ``request_rollup`` calls it with each attempt as its run-log records it. A
    control plane can end an attempt before any process writes its run-log, when
    the dispatch fails or the attempt is stopped first. It passes that attempt
    with the status it recorded, and the request then closes.

    Cancellation first, because a cancelled request is over however its other
    branches ended. Then completion. Then anything still moving -- which is what
    keeps a failed branch from closing a request its siblings are still working
    on. Only then failure.

    A step that has written nothing counts as still moving *unless* something has
    already failed, and that exception is what closes the case a bare "has it
    started" rule leaves open: when a step refuses, the steps below it are never
    dispatched, so waiting for them to start is waiting forever.
    """
    if not steps:
        return "running"
    if any(attempt.status == "cancelled" for attempt in steps):
        return "cancelled"
    if all(attempt.status == "finished" for attempt in steps):
        return "finished"
    if any(attempt.started and not attempt.terminal for attempt in steps):
        return "running"
    if any(attempt.status == "failed" for attempt in steps):
        return "failed"
    return "running"
