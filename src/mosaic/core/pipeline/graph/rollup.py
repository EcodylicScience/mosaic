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
land under a request nothing is watching.

**This reads attempts, not artifacts.** Whether every step's *work* is done is a
different question, answered by planning the recipe against the dataset -- which
costs the feature registry and is what ``mosaic pipeline plan`` is for. This one
costs a directory of small JSON files, so whatever polls it can poll it often.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mosaic.runlog import TERMINAL_STATUSES, read_run, run_log_dir

if TYPE_CHECKING:
    from pathlib import Path

    from .model import Request

__all__ = ["RequestRollup", "RequestStatus", "StepAttempt", "request_rollup"]

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
    """

    step_id: str
    execution_id: str
    status: str = ""
    run_id: str = ""
    error_json: str = ""
    entries_failed: int = 0
    entries_written: int = 0
    cache_hit: bool = False

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
    """

    request_id: str
    status: RequestStatus
    steps: tuple[StepAttempt, ...] = ()

    @property
    def is_terminal(self) -> bool:
        """Is this request over?"""
        return self.status != "running"


def request_rollup(base_dir: Path | str, request: Request) -> RequestRollup:
    """How *request* stands, from the run-logs of the attempts it assigned.

    Args:
        base_dir: The dataset root holding ``.mosaic/runs``.
        request: The submission to report on.

    Returns:
        The verdict and the per-step attempts behind it.
    """
    logs = run_log_dir(base_dir)
    steps = tuple(
        _attempt(logs, step_id, execution_id)
        for step_id, execution_id in request.step_executions.items()
    )
    return RequestRollup(
        request_id=request.request_id, status=_verdict(steps), steps=steps
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
    )


def _verdict(steps: tuple[StepAttempt, ...]) -> RequestStatus:
    """The one-shot rules, in the order they decide.

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
