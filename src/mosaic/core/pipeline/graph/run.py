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
without a terminal record runs. A step the request binds to a run is never run by
either driver: it is recorded as that run in the driver's process
(:func:`~.step.execute_step`), which refuses a bind the step could not have
written, and a failure above it does not reach the steps below it. A request
that has ended is not driven again.
:func:`start_detached_driver` starts that driver in a new session, which outlives
the process that started it.

**A failed step ends the branch below it.** Both drivers record each step below a
failed one ``failed``, with :data:`PARENT_FAILED_TYPE` naming the parent, and run
the steps that do not descend from it. A ``version_moved``, ``recipe_missing`` or
``digest_mismatch`` refusal, which every step of the request meets, and a cancel
still end the run.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, ClassVar, Final

from pydantic import ValidationError

from mosaic.core.json_object import JSON_OBJECT
from mosaic.core.scope import Scope
from mosaic.runlog import (
    TERMINAL_STATUSES,
    AttemptLiveness,
    now_iso,
    read_run,
    record_exited_attempt,
    run_log_dir,
)

from ..error_kind import ClassifiedError, ErrorKind
from ..job import Cancelled, CancelToken
from ..parallelism import ParallelSettings, parallel_flags
from ..subprocess_util import terminate_group
from .. import gpu_lease
from .driver_lease import (
    LeaseKeeper,
    driver_log_path,
    live_step_process,
    read_driver_lease,
    refuse_live_driver,
    release_driver_lease,
    take_driver_lease,
)
from .cleanup import run_cleanup
from ..gpu_lease import (
    CardLease,
    LeasableCards,
    LeaseDirectoryUnusable,
    card_warning,
    leasable_cards,
    no_lease_warning,
    step_environment,
    try_lease,
)
from .lanes import DEFAULT_LANE, GPU_LANES, lane_for_step
from .model import OpStepSpec
from .plan import plan_pipeline
from .request import load_recipe_for_request, step_argv, submit_request
from .rollup import blocked_by, request_rollup
from .step import StepOutcome, execute_step, record_step_ended
from .step_preflight import CoverageShortfall, StepRefused
from .store import load_computes
from .topo import ancestors_of, topological_order

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue
    from mosaic.runlog import RunLogSnapshot

    from ..inventory.cache import InventoryCache
    from .compatibility import DeclarationCatalog
    from .driver_lease import DriverLease
    from .failures import FailureStore
    from ..refusal import RefusalReason
    from .model import Recipe, Request
    from .rollup import RequestStatus

__all__ = [
    "MOSAIC_COMMAND",
    "PARENT_FAILED_TYPE",
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
    "step_lane",
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

PARENT_FAILED_TYPE: Final = "ParentFailed"
"""The error ``type`` recorded on a step that a failed ancestor kept from running.

Its ``error_json`` also has ``message``, ``reason`` (``parent_failed``, the word
the mosaic-queue gate records), ``parent`` and ``parent_execution_id``. The
``reason`` is not a ``RefusalReason``.
"""

_REFUSALS_OF_EVERY_STEP: Final[frozenset[RefusalReason]] = frozenset(
    {"version_moved", "recipe_missing", "digest_mismatch"}
)
"""The refusals that end a driver's run instead of only the refused step's branch.

Each is a property of the request or its recipe, and every step checks them at
its start. ``recipe_refused`` names problems at the refused step alone and ends
only the branch below it.
"""

_POLL_SECONDS: Final = 0.2
"""How often a driver checks its cancel token while a step runs or waits for a card."""

_DETACHED: list[subprocess.Popen[bytes]] = []
"""The detached drivers that this process started and that are still running.

A waiter thread per driver (:func:`_reap`) removes it from the list as it reaps it,
so a driver that exits leaves no zombie and no entry, whether or not this process
starts another.
"""

_IN_PROCESS_LEASES: set[tuple[str, str]] = set()
"""The leases that a :func:`run_pipeline` call in this process holds now.

Each is named by its request and the time it was taken. A lease of this process
missing from it was kept by a run that has returned, as after a shortfall.
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
        cleanup_status: How the request's cleanup ended (``finished``,
            ``failed`` or ``cancelled``), ``""`` when none ran.
    """

    recipe_digest: str
    scope: frozenset[Entry]
    outcomes: tuple[StepOutcome, ...] = ()
    request_id: str = ""
    cleanup_status: str = ""

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
    step_scopes: Mapping[str, Scope] | None = None,
    parallel: ParallelSettings | None = None,
    tracks_run_id: str | None = None,
    pins: Mapping[str, str] | None = None,
    applies_request_id: str | None = None,
) -> PipelineRun:
    """Run every step of *recipe* against *ds*, in order, in this process.

    A step the request binds to a run computes nothing: its outcome is
    ``bound``, or a refusal of a bind it could not have written. A step that
    fails ends the branch below it, down to a bound step, and the run
    goes on with the steps that do not descend from it. When the failed step's
    run-log records the failure, every step below it is recorded ``failed`` for it
    (:data:`PARENT_FAILED_TYPE`). A step that raises before its run-log records
    an end, and a stall without ``allow_partial``, leave the steps below them
    unrecorded. Once every other step has run, the first error is raised. A
    ``version_moved``, ``recipe_missing`` or ``digest_mismatch`` refusal, which
    every step meets, ends the run at once.

    The run holds the request's driver lease, marked ``in_process``, and names in
    it the step that runs. It deletes the lease when the request has ended and
    keeps it otherwise. A run that left steps unrecorded then reads ``orphaned``
    once the lease goes stale, and ``mosaic pipeline run --request`` resumes it,
    unless each such step is below a recorded failure: the run then releases
    its lease and the request reads ``failed``.
    A later call in this process takes over the lease that an earlier one kept.

    A request with a cleanup (``Request.cleanup``) runs it here once every step
    has finished (:func:`~.cleanup.run_cleanup`), under the same lease. A failed
    cleanup does not fail the run.

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
        step_scopes: Narrow single steps, keyed by step id. Recorded on the
            request that is submitted, and ignored when *request* is given.
        parallel: How many entries a step computes at once, and how
            (:class:`~mosaic.core.pipeline.parallelism.ParallelSettings`). ``None``
            computes them in turn.
        tracks_run_id: The tracks variant that each feature step reading the
            tracks without a ``tracks`` reference reads. Recorded on the
            request that is submitted, and ignored when *request* is given.
        pins: The finished run each pinned step serves, keyed by step id.
            Recorded on the request that is submitted, and ignored when
            *request* is given.
        applies_request_id: The finished request whose fit the run applies,
            as for :func:`~.request.submit_request`. Ignored when *request* is
            given.

    Returns:
        One :class:`~mosaic.core.pipeline.graph.step.StepOutcome` per step, in
        the order they were run.

    Raises:
        StepRefused: A step declined to run, naming why. ``CoverageShortfall`` is
            the case a caller most often means to catch. Raised after the steps
            that do not descend from it have run, as is any other error of a
            step.
        RecipeInvalid: The recipe is malformed.
        DriverRunning: A live driver holds the request's lease.
        StepStillRunning: A process of one of the request's steps is running.
        LeaseLost: The lease was deleted while this process ran the request.
        LeaseNotRefreshed: The lease could not be refreshed, checked before each
            step.
        Cancelled: The cancel token fired, or a cancel recorded a step.
    """
    if request is not None:
        _release_own_leftover_lease(ds, request.request_id)
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
            step_scopes=step_scopes,
            tracks_run_id=tracks_run_id,
            pins=pins,
            applies_request_id=applies_request_id,
        ).request
    elif allow_partial and not submitted.allow_partial:
        submitted = submitted.model_copy(update={"allow_partial": True})

    token = CancelToken() if cancel_token is None else cancel_token
    order = tuple(step.id for step in topological_order(recipe))
    lease = take_driver_lease(ds.base_dir, submitted.request_id, in_process=True)
    keeper = LeaseKeeper(ds.base_dir, lease)
    keeper.start()
    _IN_PROCESS_LEASES.add(_lease_key(lease))
    outcomes: list[StepOutcome] = []
    cleanup_status = ""
    below_failure: set[str] = set()
    unrecorded: set[str] = set()
    first_error: Exception | None = None
    driving_ended = False
    try:
        for step_id in order:
            if step_id in below_failure or step_id in unrecorded:
                continue
            keeper.ensure_held()
            keeper.record_step(step_id, os.getpid())
            before = _run_record(ds, submitted, step_id)
            try:
                # Checked once the lease names the step. A cancel that reads the
                # lease from then on signals this process instead of recording
                # the step, so only a cancel already under way can record it
                # behind this check.
                if (
                    token.is_cancelled()
                    or _recorded_status(ds, submitted, step_id) == "cancelled"
                ):
                    raise Cancelled()
                outcome = _execute_or_error(
                    ds,
                    submitted,
                    step_id,
                    store=store,
                    inventory=inventory,
                    catalog=catalog,
                    owner=owner,
                    cancel_token=token,
                    parallel=parallel,
                )
            finally:
                # Cleared however the step ended. A lease left naming a step
                # with this pid would read that step as running for as long as
                # this process lives, as a notebook kernel does after a step
                # raised before writing its run-log.
                keeper.record_step("", 0)
            if isinstance(outcome, Exception):
                first_error = first_error or outcome
                # A record left by an earlier call is not this call's failure.
                after = _run_record(ds, submitted, step_id)
                if after is None or after == before or after["status"] != "failed":
                    unrecorded |= blocked_by(recipe, submitted, step_id)
                elif _refuses_every_step(after):
                    break
                else:
                    _ = _record_blocked(ds, submitted, recipe, step_id)
                    below_failure |= blocked_by(recipe, submitted, step_id)
                continue
            outcomes.append(outcome)
            if outcome.state == "stalled" and not submitted.allow_partial:
                # Its branch stops after one stall rather than after K attempts.
                # A feature legitimately producing fewer outputs than inputs never
                # reads as complete and would be resubmitted forever; an attempt
                # counter renders a *correct* pipeline red and invites a retry
                # that cannot succeed.
                first_error = first_error or CoverageShortfall(
                    step_id,
                    outcome.covered,
                    outcome.target,
                    "the attempt exited clean and coverage did not move",
                )
                unrecorded |= blocked_by(recipe, submitted, step_id)
        if _cleanup_due(ds, submitted):
            keeper.ensure_held()
            cleanup_status = run_cleanup(
                ds, submitted, token=token, inventory=inventory, catalog=catalog
            ).status
        driving_ended = not unrecorded
    except (Cancelled, KeyboardInterrupt):
        _ = _cancel_unended(ds, submitted, order)
        raise
    finally:
        keeper.stop()
        _IN_PROCESS_LEASES.discard(_lease_key(lease))
        if request_rollup(
            ds.base_dir, submitted, read_lease=not driving_ended
        ).is_terminal:
            release_driver_lease(ds.base_dir, lease)
    if first_error is not None:
        raise first_error

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
        cleanup_status=cleanup_status,
    )


class StepStillRunning(ClassifiedError):
    """Refuse to drive a request while a process of one of its steps runs.

    A driver that was killed leaves its current step running. Driving the request
    is refused until that process ends. A step on another host whose run-log
    declares no heartbeat may have ended without this host being able to tell,
    and the message then says how to record it and start again.

    Attributes:
        step_id: The step.
        execution_id: Its attempt.
        host: The host its run-log, or the stale driver lease, names.
        pid: The process its run-log, or the stale driver lease, names.
        liveness: How its run-log reads (:mod:`~.liveness`), ``alive`` or
            ``unknown``.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(
        self,
        step_id: str,
        execution_id: str,
        *,
        host: str,
        pid: int,
        liveness: AttemptLiveness = "alive",
        request_id: str = "",
        manifest: str = "",
    ) -> None:
        if liveness == "unknown":
            close = (
                f"mosaic cancel --manifest {manifest or '<dataset.yaml>'} "
                f"--request {request_id or '<request>'} --assume-exited"
            )
            message = (
                f"step {step_id!r} may still be running as attempt {execution_id} "
                f"(pid {pid} on host {host!r}). This host cannot check host "
                f"{host!r}: the step's run-log declares no heartbeat, or the "
                f"step has not written one yet. If that process is gone, run "
                f"'{close}', then submit the recipe again; its finished steps "
                f"are cache hits."
            )
        else:
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
        self.liveness: AttemptLiveness = liveness

    def error_step(self) -> str:
        """Return the running step."""
        return self.step_id

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the attempt, its host, its pid and its liveness."""
        return {
            "execution_id": self.execution_id,
            "host": self.host,
            "pid": self.pid,
            "liveness": self.liveness,
        }


@dataclass(frozen=True, slots=True)
class RequestRun:
    """What one driver of a submitted request did.

    Attributes:
        request_id: The request.
        status: The request's rollup status when the driver ended.
        ran: The steps that the driver started a process for, in order.
        skipped: The steps it found finished, and the steps the request binds
            to a run, which it recorded as that run in its own process.
        cancelled: The steps without a terminal record that it recorded as
            cancelled when its cancel token fired.
        log_path: The log that the steps' output was appended to.
        cleanup_status: How the request's cleanup ended (``finished``,
            ``failed`` or ``cancelled``), ``""`` when none ran.
        blocked: The steps that it recorded ``failed`` because a step they
            descend from failed (:data:`PARENT_FAILED_TYPE`), in order.
    """

    request_id: str
    status: RequestStatus
    ran: tuple[str, ...]
    skipped: tuple[str, ...]
    cancelled: tuple[str, ...]
    log_path: Path
    cleanup_status: str = ""
    blocked: tuple[str, ...] = ()


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
    ds: Dataset,
    request: Request,
    *,
    cancel_token: CancelToken | None = None,
    parallel: ParallelSettings | None = None,
    gpu_cards: tuple[str, ...] | None = None,
) -> RequestRun:
    """Run the steps of a submitted request that have not finished, in order.

    Each step runs as ``mosaic run --graph-request`` in its own process and
    session, with its standard input closed and its output appended to
    :func:`~.driver_lease.driver_log_path`. A step whose run-log is finished is
    skipped. A step the request binds to a run starts no process: it is
    recorded as the pinned run here, or refused and recorded ``failed`` when its
    bind names a storage other than its own or a run that is gone. A step that
    exits without a terminal record is recorded ``failed``
    (:func:`~.step.record_step_ended`). A failed step ends the branch below it: every
    step below it is recorded ``failed`` for it (:data:`PARENT_FAILED_TYPE`), and
    the steps that do not descend from it run. A step that a resume finds
    ``failed`` ends its branch the same way. A cancelled step, and a
    ``version_moved``, ``recipe_missing`` or ``digest_mismatch`` refusal, which
    every step meets, end the run, and the steps after them are left as they
    are. A request whose rollup has ended is not driven, because a request is
    one-shot. Its stale driver lease, if one is left, is deleted.

    When *cancel_token* fires, the running step's process group receives SIGTERM,
    and SIGKILL after :data:`STEP_CANCEL_GRACE_SECONDS`. Every step without a
    terminal record is then recorded ``cancelled``, and the request's rollup is
    ``cancelled``.

    Once every step has finished, the driver runs the request's cleanup in its
    own process, under its lease (:func:`~.cleanup.run_cleanup`). A request
    whose cleanup has not ended is still running, so a driver killed during it
    leaves the request ``orphaned``, and running it again finishes the cleanup.

    The driver holds the request's lease while it runs, names in it the step whose
    process runs, and deletes it when the request has ended. A driver that raises
    keeps it, and the request then reads ``orphaned`` until it is run again. Before
    each step the driver checks that it still holds the lease.

    A step on a GPU lane (:func:`step_lane`) takes one GPU card
    (:mod:`~mosaic.core.pipeline.gpu_lease`) unless neither its run nor the run
    of any step above it is among the runs the submission recorded
    (:func:`~.store.load_computes`). Such a step's run, and the runs it reads,
    were complete at submit, so it only records a cache hit, and starts with no
    card and no wait, as a CPU step does. A step below a conversion, whose run
    id the submission could not resolve, takes a card. A request with no
    recorded runs, which an older mosaic submitted, leases a card to every GPU
    step. Before it takes its lease, the driver lists the cards it may lease
    once, when *gpu_cards* is given or a step that takes a card is still to
    run. When nothing can be leased, it writes one warning to the log and GPU
    steps start with its own environment. With every allowed card held, the
    driver waits for one, as :func:`_run_step` describes.

    Args:
        ds: The dataset the request was submitted against.
        request: The request.
        cancel_token: Fired to stop the run, as ``mosaic pipeline run`` does on
            SIGTERM and SIGINT.
        parallel: How many entries a step computes at once, and how
            (:class:`~mosaic.core.pipeline.parallelism.ParallelSettings`). ``None``
            computes them in turn.
        gpu_cards: The cards GPU steps may lease, as ``nvidia-smi`` indexes or
            UUIDs among the visible cards. ``None`` allows every visible card.

    Returns:
        What the driver ran, skipped, blocked and cancelled, and the request's
        status.

    Raises:
        StepRefused: The request's recipe is missing or changed
            (:func:`~.request.load_recipe_for_request`).
        StepStillRunning: A process of one of the request's steps is running.
        GpuCardsRefused: *gpu_cards* names a card that is not visible, raised
            before the driver takes its lease or runs a step.
        DriverRunning: A live driver holds the request's lease, or took it over
            while this one ran.
        LeaseLost: The lease was deleted while this driver ran.
        LeaseNotRefreshed: The lease could not be refreshed for
            :data:`~.driver_lease.DRIVER_LEASE_UNREFRESHED_LIMIT_SECONDS`, checked
            before each step.
    """
    recipe, order = _refuse_undrivable(ds, request)
    log_path = driver_log_path(ds.base_dir, request.request_id)
    ended = request_rollup(ds.base_dir, request)
    if ended.is_terminal:
        # A lease left here is stale, because a live one was refused above.
        stale = read_driver_lease(ds.base_dir, request.request_id)
        if stale is not None:
            release_driver_lease(ds.base_dir, stale)
        return RequestRun(
            request_id=request.request_id,
            status=ended.status,
            ran=(),
            skipped=tuple(
                attempt.step_id
                for attempt in ended.steps
                if attempt.status == "finished"
            ),
            cancelled=(),
            log_path=log_path,
        )
    computes = load_computes(ds.base_dir, request.request_id)
    computed = None if computes is None else frozenset(run.step for run in computes)
    gpu_steps = frozenset(
        step_id
        for step_id in order
        if step_id not in request.stand_ins
        and (
            computed is None
            or not computed.isdisjoint(
                ancestors_of(recipe, step_id, cut=request.stand_ins)
            )
        )
        and _recorded_status(ds, request, step_id) not in TERMINAL_STATUSES
        and step_lane(recipe, step_id) in GPU_LANES
    )
    cards = leasable_cards(gpu_cards) if gpu_cards is not None or gpu_steps else None
    token = CancelToken() if cancel_token is None else cancel_token
    lease = take_driver_lease(ds.base_dir, request.request_id)
    keeper = LeaseKeeper(ds.base_dir, lease)
    keeper.start()
    ran: list[str] = []
    skipped: list[str] = []
    blocked: list[str] = []
    below_failure: set[str] = set()
    cancelled: tuple[str, ...] = ()
    cleanup_status = ""
    try:
        with log_path.open("a", encoding="utf-8") as log:
            if cards is not None and cards.warning:
                _ = log.write(f"{card_warning(cards)}\n")
                log.flush()
            for step_id in order:
                if token.is_cancelled():
                    break
                if step_id in below_failure:
                    continue
                status = _recorded_status(ds, request, step_id)
                if status == "finished":
                    skipped.append(step_id)
                    continue
                if status not in TERMINAL_STATUSES and step_id in request.bind:
                    # Recorded here, in no process of its own: it is the run it
                    # names, unless the plan refuses the bind.
                    keeper.ensure_held()
                    status = _settle_bound(ds, request, step_id)
                    if status == "finished":
                        skipped.append(step_id)
                        continue
                if status not in TERMINAL_STATUSES:
                    keeper.ensure_held()
                    started = _run_step(
                        ds,
                        request,
                        step_id,
                        log=log,
                        token=token,
                        keeper=keeper,
                        parallel=parallel,
                        cards=cards if step_id in gpu_steps else None,
                    )
                    if started is None:
                        status = "cancelled"
                    else:
                        ran.append(step_id)
                        status = started
                if status == "finished":
                    continue
                if status != "failed" or _refuses_every_step(
                    _run_record(ds, request, step_id)
                ):
                    break
                blocked.extend(_record_blocked(ds, request, recipe, step_id))
                below_failure |= blocked_by(recipe, request, step_id)
        if not token.is_cancelled() and _cleanup_due(ds, request):
            keeper.ensure_held()
            cleanup_status = run_cleanup(ds, request, token=token).status
        if token.is_cancelled():
            cancelled = _cancel_unended(ds, request, order)
    finally:
        keeper.stop()
    # Read without the lease, because this driver has ended its steps.
    rollup = request_rollup(ds.base_dir, request, read_lease=False)
    if rollup.is_terminal:
        release_driver_lease(ds.base_dir, lease)
    return RequestRun(
        request_id=request.request_id,
        status=rollup.status,
        ran=tuple(ran),
        skipped=tuple(skipped),
        cancelled=cancelled,
        log_path=log_path,
        cleanup_status=cleanup_status,
        blocked=tuple(blocked),
    )


def start_detached_driver(
    ds: Dataset,
    request: Request,
    parallel: ParallelSettings | None = None,
    gpu_cards: tuple[str, ...] | None = None,
) -> DetachedDriver:
    """Start ``mosaic pipeline run --request`` for *request* in a new session.

    The driver's standard input is closed and its output is appended to
    :func:`~.driver_lease.driver_log_path`. It outlives this process. It takes the
    request's lease itself, and a driver started while another holds the lease
    writes :class:`~.driver_lease.DriverRunning` to the log and exits.

    Args:
        ds: The dataset the request was submitted against.
        request: The request.
        parallel: How many entries a step computes at once, and how
            (:class:`~mosaic.core.pipeline.parallelism.ParallelSettings`). ``None``
            computes them in turn.
        gpu_cards: The cards the driver may lease to GPU steps, as ``nvidia-smi``
            indexes or UUIDs, passed as ``--gpu-cards``. ``None`` allows every
            visible card.

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
        *parallel_flags(parallel),
        *([] if gpu_cards is None else ["--gpu-cards", ",".join(gpu_cards)]),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    _DETACHED.append(process)
    threading.Thread(
        target=_reap, args=(process,), name=f"mosaic-reap-{process.pid}", daemon=True
    ).start()
    return DetachedDriver(
        request_id=request.request_id, pid=process.pid, log_path=log_path
    )


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Wait for a detached driver to exit, so that its pid is released at once.

    A driver is a child of the process that started it, and an exited child is a
    zombie until that process waits for it. ``kill -0`` and ``/proc/<pid>`` then
    keep reporting a driver that ended, to every process watching its pid, for as
    long as the starting process lives. The thread is a daemon, so it never keeps
    a server or a notebook from exiting, and a driver outlives its parent.
    """
    _ = process.wait()
    with contextlib.suppress(ValueError):
        _DETACHED.remove(process)


def _refuse_undrivable(ds: Dataset, request: Request) -> tuple[Recipe, tuple[str, ...]]:
    """Raise when *request* cannot be driven now, and return its recipe and steps.

    The steps are in topological order.

    Raises:
        StepRefused: The request's recipe is missing or changed.
        DriverRunning: A live driver holds the request's lease.
        StepStillRunning: A process of one of the request's steps is running.
    """
    recipe = load_recipe_for_request(ds.base_dir, request)
    order = tuple(step.id for step in topological_order(recipe))
    refuse_live_driver(ds.base_dir, request.request_id)
    _refuse_running_step(ds, request)
    return recipe, order


def _lease_key(lease: DriverLease) -> tuple[str, str]:
    """Name one taking of a lease in :data:`_IN_PROCESS_LEASES`."""
    return (lease.request_id, lease.started_at)


def _release_own_leftover_lease(ds: Dataset, request_id: str) -> None:
    """Delete the lease that an earlier :func:`run_pipeline` call here kept.

    A run stopped by a shortfall keeps its lease, and for the grace that follows
    it reads as live. Running the request again from the same process, as a
    notebook does with ``allow_partial``, would then be refused as driven by this
    very process. A lease of this process that no running call holds is released
    instead.
    """
    held = read_driver_lease(ds.base_dir, request_id)
    if (
        held is not None
        and held.in_process
        and held.host == socket.gethostname()
        and held.pid == os.getpid()
        and _lease_key(held) not in _IN_PROCESS_LEASES
    ):
        release_driver_lease(ds.base_dir, held)


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
            running.step_id,
            running.execution_id,
            host=running.host,
            pid=running.pid,
            liveness=running.liveness,
            request_id=request.request_id,
            manifest=str(ds.manifest_path),
        )


def _run_record(ds: Dataset, request: Request, step_id: str) -> RunLogSnapshot | None:
    """Return what a step's run-log records, ``None`` without one."""
    return read_run(run_log_dir(ds.base_dir), request.execution_of(step_id))


def _recorded_status(ds: Dataset, request: Request, step_id: str) -> str:
    """Return the status that a step's run-log records, ``""`` without one."""
    snapshot = _run_record(ds, request, step_id)
    return "" if snapshot is None else snapshot["status"]


def _settle_bound(ds: Dataset, request: Request, step_id: str) -> str:
    """Record a step that *request* binds to a run, and return its status.

    :func:`~.step.execute_step` plans it, refuses a bind that the step could
    not have written, and otherwise records the pinned run. Either way it
    computes nothing.
    """
    try:
        _ = execute_step(ds, request, step_id, owner=request.owner)
    except StepRefused:
        # execute_step has recorded the refusal in the step's run-log.
        return "failed"
    return "finished"


def _execute_or_error(
    ds: Dataset,
    request: Request,
    step_id: str,
    *,
    store: FailureStore | None,
    inventory: InventoryCache | None,
    catalog: DeclarationCatalog | None,
    owner: str,
    cancel_token: CancelToken,
    parallel: ParallelSettings | None,
) -> StepOutcome | Exception:
    """Run one step in this process, and return the error that it raised.

    Raises:
        Cancelled: The step was cancelled.
    """
    try:
        return execute_step(
            ds,
            request,
            step_id,
            store=store,
            inventory=inventory,
            catalog=catalog,
            owner=owner,
            cancel_token=cancel_token,
            parallel=parallel,
        )
    except Cancelled:
        raise
    except Exception as error:
        return error


def _record_blocked(
    ds: Dataset, request: Request, recipe: Recipe, parent: str
) -> tuple[str, ...]:
    """Record each unended step below *parent* ``failed`` for it, and return them.

    Each record's error is a :data:`PARENT_FAILED_TYPE` naming *parent* and its
    attempt. The steps are returned in topological order.
    """
    below = blocked_by(recipe, request, parent)
    unended = tuple(
        step.id
        for step in topological_order(recipe)
        if step.id in below
        and _recorded_status(ds, request, step.id) not in TERMINAL_STATUSES
    )
    error_json = json.dumps(
        {
            "type": PARENT_FAILED_TYPE,
            "message": f"this step was not run, because step {parent!r} failed.",
            "reason": "parent_failed",
            "parent": parent,
            "parent_execution_id": request.execution_of(parent),
        }
    )
    for step_id in unended:
        record_step_ended(
            ds,
            request.execution_of(step_id),
            step_id=step_id,
            owner=request.owner,
            status="failed",
            error_json=error_json,
        )
    return unended


def _refuses_every_step(snapshot: RunLogSnapshot | None) -> bool:
    """Whether a step's run-log records a refusal that every later step meets."""
    if snapshot is None:
        return False
    try:
        blob = JSON_OBJECT.validate_json(snapshot["error_json"])
    except ValidationError:
        return False
    return blob.get("reason") in _REFUSALS_OF_EVERY_STEP


def step_lane(recipe: Recipe, step_id: str) -> str:
    """Return the lane of *recipe*'s step *step_id*, as its plan names it.

    :data:`~.lanes.DEFAULT_LANE` for a feature or op that this installation no
    longer registers, which the step's own process then reports.
    """
    step = recipe.step(step_id)
    name = step.kind if isinstance(step, OpStepSpec) else step.feature
    try:
        return lane_for_step(name, step.params)
    except KeyError:
        return DEFAULT_LANE


def _take_card(
    step_id: str,
    cards: LeasableCards,
    *,
    log: IO[str],
    token: CancelToken,
    keeper: LeaseKeeper,
) -> CardLease | None:
    """Lease one of *cards* for *step_id*, waiting while every one is held.

    The wait is named in the driver's lease
    (:meth:`~.driver_lease.LeaseKeeper.record_gpu_wait`) and written to the log
    once. Every :data:`~mosaic.core.pipeline.gpu_lease.LEASE_RETRY_SECONDS` the
    driver checks *token*, checks that it still holds its lease and tries again.

    Returns:
        The lease, or ``None`` when *token* fired during the wait or the lease
        directory is unusable, which the log then names.

    Raises:
        LeaseNotRefreshed: The driver's lease went unrefreshed during the wait.
        DriverRunning: Another driver took the driver's lease over.
        LeaseLost: The driver's lease was deleted.
    """
    waiting = False
    while True:
        try:
            lease = try_lease(cards.cards)
        except LeaseDirectoryUnusable as unusable:
            _ = log.write(f"{no_lease_warning(unusable.reason)}\n")
            log.flush()
            return None
        if lease is not None:
            _ = log.write(
                f"[mosaic] {now_iso()} step {step_id} took GPU {lease.card.label}\n"
            )
            log.flush()
            return lease
        if not waiting:
            waiting = True
            _ = log.write(f"[mosaic] {now_iso()} step {step_id} waits for a GPU\n")
            log.flush()
            keeper.record_gpu_wait(step_id)
        retry_at = time.monotonic() + gpu_lease.LEASE_RETRY_SECONDS
        while not token.is_cancelled() and time.monotonic() < retry_at:
            time.sleep(min(_POLL_SECONDS, max(0.0, retry_at - time.monotonic())))
        if token.is_cancelled():
            return None
        keeper.ensure_held()


def _run_step(
    ds: Dataset,
    request: Request,
    step_id: str,
    *,
    log: IO[str],
    token: CancelToken,
    keeper: LeaseKeeper,
    parallel: ParallelSettings | None,
    cards: LeasableCards | None,
) -> str | None:
    """Run one step's process to its end, and return the status it recorded.

    The lease names the step and its process while it runs. A step that exits
    without a terminal record is recorded ``cancelled`` after a cancel and
    ``failed`` otherwise.

    Given *cards* to lease, the step on a GPU lane first takes one
    (:func:`_take_card`), waiting while every one is held. The step's process
    starts with the card's environment
    (:func:`~mosaic.core.pipeline.gpu_lease.step_environment`) and the lock's
    descriptor, and the driver closes its own copy once the process has started,
    so the lock lives as long as the step and the tools it supervises. Every other
    step starts with the driver's environment, less the lease variables. A cancel
    during the wait starts no process and leaves the step unrecorded, so the driver
    records it ``cancelled`` with the other steps that have not started.

    Returns:
        The status, or ``None`` when a cancel during the wait for a card came
        before the step's process started.
    """
    execution_id = request.execution_of(step_id)
    argv = step_argv(
        ds.manifest_path,
        request,
        step_id,
        program=MOSAIC_COMMAND,
        parallel=parallel,
    )
    card: CardLease | None = None
    if cards is not None and cards.cards:
        card = _take_card(step_id, cards, log=log, token=token, keeper=keeper)
        if card is None and token.is_cancelled():
            keeper.record_step("", 0)
            return None
    _ = log.write(f"[mosaic] {now_iso()} step {step_id} ({execution_id}) started\n")
    log.flush()
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=sys.platform != "win32",
            text=True,
            env=step_environment(os.environ, card),
            pass_fds=() if card is None else (card.fd,),
        )
    finally:
        if card is not None:
            card.close()
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
    """Record every step without a terminal record as cancelled, and return them.

    A cleanup that the steps' finishing made due and that has not ended is
    recorded cancelled too, so the request ends instead of waiting for it.
    """
    unended = tuple(
        step_id
        for step_id in order
        # A bound step is the pinned run whether or not it was recorded.
        if step_id not in request.bind
        and _recorded_status(ds, request, step_id) not in TERMINAL_STATUSES
    )
    for step_id in unended:
        record_step_ended(
            ds,
            request.execution_of(step_id),
            step_id=step_id,
            owner=request.owner,
            status="cancelled",
        )
    if request.cleanup is not None and _cleanup_due(ds, request):
        record_exited_attempt(
            run_log_dir(ds.base_dir), request.cleanup.execution_id, "cancelled"
        )
    return unended


def _cleanup_due(ds: Dataset, request: Request) -> bool:
    """Whether *request* has a cleanup, every step finished, and it has not ended."""
    if request.cleanup is None:
        return False
    rollup = request_rollup(ds.base_dir, request)
    if any(attempt.status != "finished" for attempt in rollup.steps):
        return False
    return rollup.cleanup is None or not rollup.cleanup.terminal
