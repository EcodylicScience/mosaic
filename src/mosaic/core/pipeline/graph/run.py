"""Running a plan here, in this process, one step after another.

**Not a fallback and not a rejected alternative.** It is what a notebook uses,
what a bare HPC node with no queue uses, and what Dagster's own default executor
is. A pipeline that only runs under a control plane is a pipeline most of its
users cannot run at all.

**One step body, two drivers.** Each step goes through
:func:`~mosaic.core.pipeline.graph.step.execute_step`, which is also what one
``mosaic run --graph-request --step`` invocation calls. So the preflight, the
parent pinning and the failure record are properties of running a step rather
than of running it a particular way, and the loop below cannot drift from the
queued path by being edited separately.

**Each step is planned again at its own start**, which is what makes the loop
correct rather than optimistic: a step's coverage, and therefore what it should
be asked to compute, changes the moment its parent finishes.

**Planning does not submit.** ``plan_pipeline`` resolves; this module runs. They
are kept apart because the queue package already depends on this one, so a
planner that submitted would be a package cycle -- and because a preview that
could start work is a preview nobody dares ask for.

**Running here holds the request's lease too.** :func:`run_pipeline` takes an
``in_process`` driver lease (:mod:`~.driver_lease`), so a request run from a notebook
or by ``mosaic pipeline run --recipe`` reads ``orphaned`` once that process is gone,
and a second driver is refused while it lives.

**A submitted request is driven step by step, one process per step.**
:func:`run_request` runs each step's :func:`~.request.step_argv` in topological
order, as ``mosaic run --graph-request`` does under a queue, and holds the
request's driver lease (:mod:`~.driver_lease`) while it does. It submits nothing,
and running a request again resumes it: a finished step is skipped, and a step
without a terminal record runs. :func:`start_detached_driver` starts that driver in
a new session, which outlives the process that started it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, ClassVar, Final

from mosaic.core.scope import Scope
from mosaic.runlog import TERMINAL_STATUSES, now_iso, read_run, run_log_dir

from ..error_kind import ClassifiedError, ErrorKind
from ..job import Cancelled, CancelToken
from ..subprocess_util import terminate_group
from .driver_lease import (
    LeaseKeeper,
    driver_log_path,
    live_step_process,
    read_driver_lease,
    refuse_live_driver,
    release_driver_lease,
    take_driver_lease,
)
from .plan import plan_pipeline
from .request import load_recipe_for_request, step_argv, submit_request
from .rollup import request_rollup
from .step import StepOutcome, execute_step, record_step_ended
from .step_preflight import CoverageShortfall
from .topo import topological_order

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue

    from ..inventory.cache import InventoryCache
    from .compatibility import DeclarationCatalog
    from .failures import FailureStore
    from .model import Recipe, Request
    from .rollup import RequestStatus

__all__ = [
    "MOSAIC_COMMAND",
    "STEP_CANCEL_GRACE_SECONDS",
    "CoverageShortfall",
    "DetachedDriver",
    "PipelineRun",
    "RequestRun",
    "StepOutcome",
    "StepStillRunning",
    "run_pipeline",
    "run_request",
    "start_detached_driver",
]

MOSAIC_COMMAND: Final[tuple[str, ...]] = (sys.executable, "-m", "mosaic")
"""The command that starts mosaic in this interpreter, for each step and driver."""

STEP_CANCEL_GRACE_SECONDS: Final = 1800.0
"""How long a step has to stop after its driver forwards SIGTERM, before SIGKILL.

A feature step stops at its next cancel checkpoint, which ``run_feature`` reaches
between entries. A training op stops cooperatively at the end of its epoch and
kills its tool only after a grace of its own, so the driver plays the part a
substrate plays for a queued run: its grace equals the termination grace a
mosaic-queue pod is given by default and exceeds every in-step grace. A shorter
one would kill the step while its tool, in a session of its own, kept writing.
"""

_POLL_SECONDS: Final = 0.2
"""How often a driver checks its cancel token while a step runs."""

_DETACHED: list[subprocess.Popen[bytes]] = []
"""The detached drivers that this process started and has not reaped.

A ``Popen`` collected while its child runs warns, and a long-lived process reaps an
exited driver at its next start.
"""


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """What running one recipe against one dataset did.

    Attributes:
        recipe_digest: What names the recipe that was run.
        scope: The entries the graph was run over.
        outcomes: One record per step, in the order they were run.
        request_id: The submission this run was recorded as. Every run is one,
            so that a graph driven here is as legible afterwards as one driven
            by a queue.
    """

    recipe_digest: str
    scope: frozenset[Entry]
    outcomes: tuple[StepOutcome, ...] = ()
    request_id: str = ""

    @property
    def ran(self) -> tuple[StepOutcome, ...]:
        """The steps that did work, as against those served from cache."""
        return tuple(outcome for outcome in self.outcomes if outcome.state == "ran")

    @property
    def stalled(self) -> tuple[StepOutcome, ...]:
        """The steps that succeeded and produced nothing new."""
        return tuple(outcome for outcome in self.outcomes if outcome.state == "stalled")


def run_pipeline(
    ds: Dataset,
    recipe: Recipe,
    *,
    scope: Scope | None = None,
    request: Request | None = None,
    allow_partial: bool = False,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    store: FailureStore | None = None,
    owner: str = "",
    cancel_token: CancelToken | None = None,
) -> PipelineRun:
    """Run every step of *recipe* against *ds*, in order, in this process.

    The run holds the request's driver lease, marked ``in_process``, and names in
    it the step that runs. It deletes the lease when the request has ended and
    keeps it otherwise, so a run stopped by a shortfall reads ``orphaned`` once
    the lease goes stale, and ``mosaic pipeline run --request`` resumes it.

    Before each step the run stops when its cancel token has fired or a cancel
    recorded the step ``cancelled`` (:func:`~mosaic.core.pipeline.cancel.cancel_request`
    records the steps of an in-process run that have not started). A cancel, or a
    ``KeyboardInterrupt``, records every step without a terminal record
    ``cancelled`` before it propagates.

    Args:
        ds: The dataset to run against.
        recipe: The graph. Refused before anything runs if it is malformed.
        scope: What to run over, or ``None`` for the dataset's own scope.
        request: An existing submission to run. One is recorded when none is
            given, so the dataset holds the recipe and the request either way.
        allow_partial: Proceed through a shortfall or a stall rather than
            refusing. Ignored when *request* already says so.
        inventory: A held inventory, revalidated on every planning call.
        catalog: The declarations to resolve against.
        store: Where failures are recorded.
        owner: Recorded on each attempt.
        cancel_token: Fired to stop the run, as ``mosaic pipeline run --recipe``
            does on SIGTERM and SIGINT. Each step's work checks it.

    Returns:
        One :class:`~mosaic.core.pipeline.graph.step.StepOutcome` per step, in
        the order they were run.

    Raises:
        StepRefused: A step declined to run, naming why. ``CoverageShortfall`` is
            the case a caller most often means to catch.
        RecipeInvalid: The recipe is malformed.
        DriverRunning: A live driver holds the request's lease.
        StepStillRunning: A process of one of the request's steps is running.
        LeaseLost: The lease was deleted while this process ran the request.
        LeaseNotRefreshed: The lease could not be refreshed, checked before each
            step.
        Cancelled: The cancel token fired, or a cancel recorded a step.
    """
    if request is not None:
        _refuse_running_step(ds, request)
    submitted = request
    if submitted is None:
        submitted = submit_request(
            ds,
            recipe,
            scope=scope,
            allow_partial=allow_partial,
            owner=owner,
            inventory=inventory,
            catalog=catalog,
        ).request
    elif allow_partial and not submitted.allow_partial:
        submitted = submitted.model_copy(update={"allow_partial": True})

    token = CancelToken() if cancel_token is None else cancel_token
    order = tuple(step.id for step in topological_order(recipe))
    lease = take_driver_lease(ds.base_dir, submitted.request_id, in_process=True)
    keeper = LeaseKeeper(ds.base_dir, lease)
    keeper.start()
    outcomes: list[StepOutcome] = []
    try:
        for step_id in order:
            if (
                token.is_cancelled()
                or _recorded_status(ds, submitted, step_id) == "cancelled"
            ):
                raise Cancelled()
            keeper.ensure_held()
            keeper.record_step(step_id, os.getpid())
            outcome = execute_step(
                ds,
                submitted,
                step_id,
                store=store,
                inventory=inventory,
                catalog=catalog,
                owner=owner,
                cancel_token=token,
            )
            keeper.record_step("", 0)
            outcomes.append(outcome)
            if outcome.state == "stalled" and not submitted.allow_partial:
                # Stopped after one stall rather than after K attempts. A feature
                # legitimately producing fewer outputs than inputs never reads as
                # complete and would be resubmitted forever; an attempt counter
                # renders a *correct* pipeline red and invites a retry that
                # cannot succeed.
                raise CoverageShortfall(
                    step_id,
                    outcome.covered,
                    outcome.target,
                    "the attempt exited clean and coverage did not move",
                )
    except (Cancelled, KeyboardInterrupt):
        _ = _cancel_unended(ds, submitted, order)
        raise
    finally:
        keeper.stop()
        if request_rollup(ds.base_dir, submitted).is_terminal:
            release_driver_lease(ds.base_dir, lease)

    final = plan_pipeline(
        ds,
        recipe,
        request=submitted,
        inventory=inventory,
        catalog=catalog,
    )
    return PipelineRun(
        recipe_digest=final.recipe_digest,
        scope=final.scope,
        outcomes=tuple(outcomes),
        request_id=submitted.request_id,
    )


class StepStillRunning(ClassifiedError):
    """Refuse to drive a request while a process of one of its steps runs.

    A driver that was killed leaves its current step running. Driving the request
    is refused until that process ends.

    Attributes:
        step_id: The step.
        execution_id: Its attempt.
        host: The host its run-log, or the stale driver lease, names.
        pid: The process its run-log, or the stale driver lease, names.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, step_id: str, execution_id: str, *, host: str, pid: int) -> None:
        message = (
            f"step {step_id!r} is still running as attempt {execution_id} (pid "
            f"{pid} on host {host!r}). Run the request again once that attempt "
            f"has ended, or cancel it with 'mosaic cancel --execution-id "
            f"{execution_id}'."
        )
        super().__init__(message)
        self.step_id: str = step_id
        self.execution_id: str = execution_id
        self.host: str = host
        self.pid: int = pid

    def error_step(self) -> str:
        """Return the running step."""
        return self.step_id

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the attempt, its host and its pid."""
        return {"execution_id": self.execution_id, "host": self.host, "pid": self.pid}


@dataclass(frozen=True, slots=True)
class RequestRun:
    """What one driver of a submitted request did.

    Attributes:
        request_id: The request.
        status: The request's rollup status when the driver ended.
        ran: The steps that the driver started a process for, in order.
        skipped: The steps it found finished.
        cancelled: The steps without a terminal record that it recorded as
            cancelled when its cancel token fired.
        log_path: The log that the steps' output was appended to.
    """

    request_id: str
    status: RequestStatus
    ran: tuple[str, ...]
    skipped: tuple[str, ...]
    cancelled: tuple[str, ...]
    log_path: Path


@dataclass(frozen=True, slots=True)
class DetachedDriver:
    """A driver started in a new session.

    Attributes:
        request_id: The request it drives.
        pid: Its process.
        log_path: The log that it and its steps append to.
    """

    request_id: str
    pid: int
    log_path: Path


def run_request(
    ds: Dataset, request: Request, *, cancel_token: CancelToken | None = None
) -> RequestRun:
    """Run the steps of a submitted request that have not finished, in order.

    Each step runs as ``mosaic run --graph-request`` in its own process and
    session, with its standard input closed and its output appended to
    :func:`~.driver_lease.driver_log_path`. A step whose run-log is finished is
    skipped. A failed or cancelled step ends the run, and the steps after it are
    left as they are. A step that exits without a terminal record is recorded
    ``failed`` (:func:`~.step.record_step_ended`).

    When *cancel_token* fires, the running step's process group receives SIGTERM,
    and SIGKILL after :data:`STEP_CANCEL_GRACE_SECONDS`. Every step without a
    terminal record is then recorded ``cancelled``, and the request's rollup is
    ``cancelled``.

    The driver holds the request's lease while it runs, names in it the step whose
    process runs, and deletes it when the request has ended. A driver that raises
    keeps it, and the request then reads ``orphaned`` until it is run again. Before
    each step the driver checks that it still holds the lease.

    Args:
        ds: The dataset the request was submitted against.
        request: The request.
        cancel_token: Fired to stop the run, as ``mosaic pipeline run`` does on
            SIGTERM and SIGINT.

    Returns:
        What the driver ran, skipped and cancelled, and the request's status.

    Raises:
        StepRefused: The request's recipe is missing or changed
            (:func:`~.request.load_recipe_for_request`).
        StepStillRunning: A process of one of the request's steps is running.
        DriverRunning: A live driver holds the request's lease, or took it over
            while this one ran.
        LeaseLost: The lease was deleted while this driver ran.
        LeaseNotRefreshed: The lease could not be refreshed for
            :data:`~.driver_lease.DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS`, checked
            before each step.
    """
    order = _refuse_undrivable(ds, request)
    token = CancelToken() if cancel_token is None else cancel_token
    lease = take_driver_lease(ds.base_dir, request.request_id)
    keeper = LeaseKeeper(ds.base_dir, lease)
    keeper.start()
    log_path = driver_log_path(ds.base_dir, request.request_id)
    ran: list[str] = []
    skipped: list[str] = []
    cancelled: tuple[str, ...] = ()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for step_id in order:
                if token.is_cancelled():
                    break
                status = _recorded_status(ds, request, step_id)
                if status == "finished":
                    skipped.append(step_id)
                    continue
                if status in TERMINAL_STATUSES:
                    break
                keeper.ensure_held()
                ran.append(step_id)
                outcome = _run_step(
                    ds, request, step_id, log=log, token=token, keeper=keeper
                )
                if outcome != "finished":
                    break
        if token.is_cancelled():
            cancelled = _cancel_unended(ds, request, order)
    finally:
        keeper.stop()
    rollup = request_rollup(ds.base_dir, request)
    if rollup.is_terminal:
        release_driver_lease(ds.base_dir, lease)
    return RequestRun(
        request_id=request.request_id,
        status=rollup.status,
        ran=tuple(ran),
        skipped=tuple(skipped),
        cancelled=cancelled,
        log_path=log_path,
    )


def start_detached_driver(ds: Dataset, request: Request) -> DetachedDriver:
    """Start ``mosaic pipeline run --request`` for *request* in a new session.

    The driver's standard input is closed and its output is appended to
    :func:`~.driver_lease.driver_log_path`. It outlives this process. It takes the
    request's lease itself, and a driver started while another holds the lease
    writes :class:`~.driver_lease.DriverRunning` to the log and exits.

    Args:
        ds: The dataset the request was submitted against.
        request: The request.

    Returns:
        The driver's pid and log.

    Raises:
        StepRefused: The request's recipe is missing or changed
            (:func:`~.request.load_recipe_for_request`).
        StepStillRunning: A process of one of the request's steps is running.
        DriverRunning: A live driver holds the request's lease.
    """
    _ = _refuse_undrivable(ds, request)
    log_path = driver_log_path(ds.base_dir, request.request_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        *MOSAIC_COMMAND,
        "pipeline",
        "run",
        "--manifest",
        str(ds.manifest_path),
        "--request",
        request.request_id,
    ]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    _DETACHED[:] = [running for running in _DETACHED if running.poll() is None]
    _DETACHED.append(process)
    return DetachedDriver(
        request_id=request.request_id, pid=process.pid, log_path=log_path
    )


def _refuse_undrivable(ds: Dataset, request: Request) -> tuple[str, ...]:
    """Raise when *request* cannot be driven now, and return its steps in order.

    Raises:
        StepRefused: The request's recipe is missing or changed.
        DriverRunning: A live driver holds the request's lease.
        StepStillRunning: A process of one of the request's steps is running.
    """
    recipe = load_recipe_for_request(ds.base_dir, request)
    order = tuple(step.id for step in topological_order(recipe))
    refuse_live_driver(ds.base_dir, request.request_id)
    _refuse_running_step(ds, request)
    return order


def _refuse_running_step(ds: Dataset, request: Request) -> None:
    """Raise when a process of one of *request*'s steps may still be running.

    Raises:
        StepStillRunning: The process that a step's run-log, or a stale driver
            lease, names may still run (:func:`~.driver_lease.live_step_process`).
    """
    stale = read_driver_lease(ds.base_dir, request.request_id)
    running = live_step_process(request_rollup(ds.base_dir, request), stale)
    if running is not None:
        raise StepStillRunning(
            running.step_id, running.execution_id, host=running.host, pid=running.pid
        )


def _recorded_status(ds: Dataset, request: Request, step_id: str) -> str:
    """Return the status that a step's run-log records, ``""`` without one."""
    snapshot = read_run(run_log_dir(ds.base_dir), request.execution_of(step_id))
    return "" if snapshot is None else snapshot["status"]


def _run_step(
    ds: Dataset,
    request: Request,
    step_id: str,
    *,
    log: IO[str],
    token: CancelToken,
    keeper: LeaseKeeper,
) -> str:
    """Run one step's process to its end, and return the status it recorded.

    The lease names the step and its process while it runs. A step that exits
    without a terminal record is recorded ``cancelled`` after a cancel and
    ``failed`` otherwise.
    """
    execution_id = request.execution_of(step_id)
    argv = step_argv(ds.manifest_path, request, step_id, program=MOSAIC_COMMAND)
    _ = log.write(f"[mosaic] {now_iso()} step {step_id} ({execution_id}) started\n")
    log.flush()
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=sys.platform != "win32",
        text=True,
    )
    keeper.record_step(step_id, process.pid)
    while True:
        try:
            _ = process.wait(timeout=_POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            if token.is_cancelled():
                terminate_group(process, grace=STEP_CANCEL_GRACE_SECONDS)
                break
    code = process.poll()
    keeper.record_step("", 0)
    _ = log.write(f"[mosaic] {now_iso()} step {step_id} exited with code {code}\n")
    log.flush()
    status = _recorded_status(ds, request, step_id)
    if status in TERMINAL_STATUSES:
        return status
    if token.is_cancelled():
        record_step_ended(
            ds, execution_id, step_id=step_id, owner=request.owner, status="cancelled"
        )
        return "cancelled"
    message = (
        f"step {step_id!r} exited with code {code} without recording an outcome. "
        f"Its output is in {driver_log_path(ds.base_dir, request.request_id)}."
    )
    record_step_ended(
        ds,
        execution_id,
        step_id=step_id,
        owner=request.owner,
        status="failed",
        error_json=json.dumps({"type": "StepExited", "message": message}),
    )
    return "failed"


def _cancel_unended(
    ds: Dataset, request: Request, order: tuple[str, ...]
) -> tuple[str, ...]:
    """Record every step without a terminal record as cancelled, and return them."""
    unended = tuple(
        step_id
        for step_id in order
        if _recorded_status(ds, request, step_id) not in TERMINAL_STATUSES
    )
    for step_id in unended:
        record_step_ended(
            ds,
            request.execution_of(step_id),
            step_id=step_id,
            owner=request.owner,
            status="cancelled",
        )
    return unended
