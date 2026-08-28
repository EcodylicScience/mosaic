"""Running one step of one request, wherever the process came from.

The unit of work. A queue gives it a whole process; the sequential runner calls
it in a loop; a shell script or a job array does the same with nothing else
installed. All three take this path, so the preflight and the failure record are
not features of one of them.

**A step re-plans itself at its own start.** The identity resolved at submit
drives the preview, the estimate and the decision to enqueue, and it is never
load-bearing here: a wrong prediction makes the preview wrong and costs nothing
else. What governs is what this step resolves once it has what the plan lacked --
which, for a ``scope_dependent`` step, is which entries actually completed.

**It pins its parents from their run-logs rather than by name.** Resolving an
input by *feature name* falls through to the latest-run rule, which is wall
clock, so two requests on one dataset running one feature with different params
would cross-bind. The request says which attempt each parent is, and that
attempt's run-log says what it produced.

**Only feature parents are pinned, and the asymmetry is not an omission.** An
op's identity is a function of its params, and the plan hands execution those
same params, so the two cannot differ. What an op parent *is* checked for is the
tracks variant it produced, which is a different identifier from its run id.

**A step does not enqueue its successors.** A worker dying between finishing and
enqueueing would stall the graph permanently, a diamond join would need whichever
worker finishes last to notice, and every worker would need the whole recipe and
the inventory. Reading its own parents and re-planning *itself* is bounded, needs
no graph-wide view, and creates nothing.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from mosaic.core.helpers import make_entry_key
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, read_run, run_log_dir, run_log_path

from ..inventory.cache import InventoryCache
from ..inventory.model import ArtifactRef, FeatureRunRef
from ..ops import run_op
from ..run import run_feature
from .claims import FileFailureStore
from .model import FeatureStepSpec
from .plan import (
    COMPLETE_STATUSES,
    MISSING_SAMPLE,
    Plan,
    PlannedStep,
    coverage_against,
    is_stalled,
    plan_pipeline,
)
from .preflight import CoverageShortfall, StepRefused, preflight, refuse_mixed_schemas
from .request import load_recipe_for_request
from .resolve import (
    build_step_feature,
    build_step_op_params,
    declared_version,
    op_class_for_kind,
)
from .topo import ancestors_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry

    from ..job import CancelToken
    from .compatibility import DeclarationCatalog
    from .failures import FailureStore
    from .model import Recipe, Request

__all__ = ["STEP_RUN_KIND", "StepOutcome", "execute_step"]

STEP_RUN_KIND: Final = "pipeline-step"
"""What a refusal records as its ``kind`` in the run-log.

A refusal runs neither a feature nor an op -- that is the point of it -- and some
refusals happen before the recipe that would say which is even readable. Naming
it for what it is beats guessing one of the two.
"""


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """What happened to one step of one request.

    Attributes:
        step_id: The step in the recipe.
        kind: Whether it ran a feature or an op.
        run_id: What the run is called, as *execution* recorded it -- the value
            that governs, and which a step whose identity was deferred has for
            the first time here.
        state: ``ran`` if work happened, ``cached`` if the artifact already
            answered for the whole scope, ``stalled`` if it exited clean and
            produced nothing new.
        planned_run_id: What the plan said it would be called, or ``None`` where
            nothing could say. Kept beside the recorded one so a divergence is
            visible rather than inferred.
        execution_id: Which attempt this was, so the artifact joins back to its
            run-log with no control plane in the way.
        failed_entries: The entities this attempt lost, by entry key.
        covered: How much of the target this step's artifact answers for, after.
        target: How much was wanted.
    """

    step_id: str
    kind: Literal["feature", "op"]
    run_id: str
    state: Literal["ran", "cached", "stalled"]
    planned_run_id: str | None = None
    execution_id: str = ""
    failed_entries: tuple[str, ...] = ()
    covered: int = 0
    target: int = 0

    @property
    def diverged(self) -> bool:
        """Did execution record something other than what was resolved?"""
        return self.planned_run_id is not None and self.planned_run_id != self.run_id


def execute_step(
    ds: Dataset,
    request: Request,
    step_id: str,
    *,
    execution_id: str = "",
    overwrite: bool = False,
    store: FailureStore | None = None,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    owner: str = "",
    cancel_token: CancelToken | None = None,
) -> StepOutcome:
    """Run one step of *request* against *ds*, and say what it did.

    Args:
        ds: The dataset. Also where the recipe, the run-logs and the failure
            records are read from.
        request: The submission this step belongs to.
        step_id: Which step to run.
        execution_id: This attempt's id. Empty takes the one the request assigned.
        overwrite: Recompute even where a cached run exists.
        store: Where failures are recorded. A dataset-backed one is built when
            none is given.
        inventory: A held inventory, revalidated on every planning call.
        catalog: The declarations to resolve against.
        owner: Recorded on the attempt.
        cancel_token: Cooperative cancellation, so a signal reaches the work
            rather than only the process around it.

    Returns:
        What this step did, and what it recorded doing it under.

    Raises:
        StepRefused: The step declined to run, naming why. Its ``error_json`` has
            already been written to this attempt's run-log.
        RecipeInvalid: The recipe this request names is malformed.
    """
    attempt = execution_id or request.execution_of(step_id)
    failures = FileFailureStore(ds.base_dir) if store is None else store
    try:
        return _execute(
            ds,
            request,
            step_id,
            attempt=attempt,
            overwrite=overwrite,
            store=failures,
            inventory=inventory,
            catalog=catalog,
            owner=owner,
            cancel_token=cancel_token,
        )
    except StepRefused as refusal:
        # Written here because nothing else will: a refusal happens before
        # ``run_feature`` opens the run-log it would otherwise own, and the queue
        # releases a child on its parent's terminal record. A refusal with no
        # record is a step that reads as still running forever.
        _record_terminal(
            ds,
            attempt,
            kind=STEP_RUN_KIND,
            target=step_id,
            owner=owner,
            error_json=refusal.error_json(),
        )
        raise


def _execute(
    ds: Dataset,
    request: Request,
    step_id: str,
    *,
    attempt: str,
    overwrite: bool,
    store: FailureStore,
    inventory: InventoryCache | None,
    catalog: DeclarationCatalog | None,
    owner: str,
    cancel_token: CancelToken | None,
) -> StepOutcome:
    """The body, with every refusal free to propagate to its recorder."""
    recipe = load_recipe_for_request(ds.base_dir, request)
    _refuse_moved_versions(recipe, request, step_id)
    # Before planning, because resolving any feature identity against a dataset
    # of mixed schemas already fails -- and failing there is an exception out of
    # the middle of a hash rather than a refusal naming the schemas.
    refuse_mixed_schemas(ds, step_id)

    held = InventoryCache(ds) if inventory is None else inventory
    excluded = set(store.exclusions(request.request_id).entries)
    recorded = _pinned_parents(ds, recipe, request, step_id)
    plan = _plan(ds, recipe, request, recorded, excluded, held, catalog)
    planned = plan.step(step_id)

    if planned.status in COMPLETE_STATUSES and not overwrite:
        return _cached(ds, planned, attempt=attempt, owner=owner)

    preflight(ds, plan, step_id, allow_partial=request.allow_partial)

    asked = asked_of(planned, plan)
    storage, run_id = planned.storage_name, planned.run_id or ""
    quarantined = store.quarantined_entries(storage, run_id, asked)
    if quarantined:
        if not request.allow_partial:
            raise CoverageShortfall(
                step_id,
                len(asked) - len(quarantined),
                len(asked),
                "held back after repeated failures: "
                + ", ".join(
                    sorted(str(entry) for entry in quarantined)[:MISSING_SAMPLE]
                ),
            )
        # Recorded before re-planning, because an exclusion narrows the scope and
        # a ``scope_dependent`` step's identity *is* its scope: a fit over what is
        # left has to be named for what is left.
        store.exclude(request.request_id, step_id, quarantined)
        excluded |= set(quarantined)
        plan = _plan(ds, recipe, request, recorded, excluded, held, catalog)
        planned = plan.step(step_id)
        if planned.status in COMPLETE_STATUSES:
            return _cached(ds, planned, attempt=attempt, owner=owner)
        asked = asked_of(planned, plan)
        storage, run_id = planned.storage_name, planned.run_id or ""

    # A wait is not a verdict, so it narrows this attempt and decides nothing.
    # An entry that has failed once and is not due yet needs a few more seconds,
    # not a permanent exclusion from a fit.
    waiting = store.waiting_entries(storage, run_id, asked)
    if waiting:
        asked = tuple(entry for entry in asked if entry not in waiting)
        if not asked:
            # Nothing to attempt *yet*, which is not the same as nothing to do:
            # an empty ``asked`` otherwise means "the whole scope", and a step
            # over an empty scope is a legitimate no-op rather than a stall.
            return _stalled(ds, planned, attempt=attempt, owner=owner)

    outcome = _run(
        ds,
        planned,
        asked,
        attempt=attempt,
        overwrite=overwrite,
        store=store,
        owner=owner,
        cancel_token=cancel_token,
    )
    return _measured(ds, held, plan, planned, outcome)


def _plan(
    ds: Dataset,
    recipe: Recipe,
    request: Request,
    recorded: Mapping[str, str],
    excluded: Iterable[Entry],
    inventory: InventoryCache,
    catalog: DeclarationCatalog | None,
) -> Plan:
    """This step's graph, resolved against what is on disk right now."""
    return plan_pipeline(
        ds,
        recipe,
        request=request,
        recorded=recorded,
        quarantined=excluded,
        inventory=inventory,
        catalog=catalog,
    )


def _measured(
    ds: Dataset,
    inventory: InventoryCache,
    plan: Plan,
    planned: PlannedStep,
    outcome: StepOutcome,
) -> StepOutcome:
    """The outcome with what the attempt actually achieved measured onto it.

    Measured off the artifact rather than by re-planning the graph, because the
    question is about one step and re-resolving every identity to answer it would
    cost more than the step that just ran.

    **A stall is reported, never raised here.** The queued path exits cleanly and
    lets the consumer refuse at its own preflight, which is where a shortfall
    belongs; the sequential runner reads this state and stops. Raising in both
    would put the refusal in two places with two different messages.
    """
    _ = inventory.revalidate()
    view = coverage_against(
        inventory.get(scope=Scope(entries=sorted(plan.scope)) if plan.scope else None),
        _produced_ref(planned, outcome.run_id),
        plan.scope,
    )
    measured = replace(outcome, covered=view.held, target=len(plan.scope))
    if is_stalled(
        exited_clean=True,
        terminal=True,
        covered_before=len(planned.coverage.covered),
        covered_after=view.held,
    ):
        return replace(measured, state="stalled")
    return measured


def _produced_ref(planned: PlannedStep, run_id: str) -> ArtifactRef | None:
    """What to look up what this attempt produced by.

    A feature's identifier can legitimately move between the plan and the run --
    that is what a ``scope_dependent`` step does whenever fewer entries complete
    than were intended -- so its reference is rebuilt from what was recorded. An
    op's cannot move, its identity being a function of the params the plan handed
    execution, so its planned reference still answers.
    """
    if planned.kind == "feature" and run_id:
        return FeatureRunRef(name=planned.storage_name, run_id=run_id)
    return planned.artifact


def _refuse_moved_versions(recipe: Recipe, request: Request, step_id: str) -> None:
    """Refuse when a producer's version has moved underneath an open request.

    A request completes against the versions it started with, and every request
    records which those were. Otherwise an upgrade partway through resolves the
    early steps under the old versions and the later ones under the new, so one
    submission silently spans two identity regimes and its later steps read as
    absent with no fault anywhere to point at.

    Reported rather than silently re-planned: a resubmit re-resolves every
    version and is the cheap way forward, because everything already on disk
    under the old identifiers stays a cache hit for whatever still names them.
    """
    for step in recipe.steps:
        pinned = request.step_versions.get(step.id)
        if not pinned:
            # A request that pinned nothing for this step, or a producer that
            # declares no version, has nothing to compare. Silence here is the
            # honest answer; the alternative would refuse every request written
            # before versions were pinned.
            continue
        is_feature = isinstance(step, FeatureStepSpec)
        name = step.feature if is_feature else step.kind
        installed = declared_version("feature" if is_feature else "op", name)
        if installed and installed != pinned:
            raise StepRefused(
                "version_moved",
                step_id,
                f"{name!r} is version {installed!r} here, but request "
                f"{request.request_id!r} was resolved against {pinned!r}. "
                f"Running now would put this request's steps under two identity "
                f"schemes; submit it again to resolve every version afresh.",
                {
                    "moved_step": step.id,
                    "runs": name,
                    "pinned": pinned,
                    "installed": installed,
                },
            )


def _pinned_parents(
    ds: Dataset, recipe: Recipe, request: Request, step_id: str
) -> dict[str, str]:
    """What each feature ancestor's attempt recorded producing.

    **Every ancestor, not only the immediate parents.** Identity chains: this
    step's identifier is a function of its parents', theirs of their parents',
    and so on. Planning an ancestor from a prediction while pinning a parent from
    a fact would give this step an identifier neither of them agrees with.

    An ancestor with no run-log contributes nothing and is resolved normally --
    that is a step served from an earlier request's cache, which is the ordinary
    state of a partly built graph, not a fault.
    """
    logs = run_log_dir(ds.base_dir)
    pinned: dict[str, str] = {}
    for ancestor in sorted(ancestors_of(recipe, step_id) - {step_id}):
        step = recipe.step(ancestor)
        if not isinstance(step, FeatureStepSpec):
            continue
        attempt = request.step_executions.get(ancestor, "")
        if not attempt:
            continue
        snapshot = read_run(logs, attempt)
        if snapshot is None:
            continue
        if snapshot["run_id"]:
            pinned[ancestor] = snapshot["run_id"]
            continue
        raise StepRefused(
            "parent_unrecorded",
            step_id,
            f"step {ancestor!r} ran as attempt {attempt!r} and recorded no "
            f"identifier, so nothing below it can name what it produced. Its "
            f"run-log says {snapshot['status']!r}.",
            {"parent": ancestor, "parent_execution_id": attempt},
        )
    return pinned


def asked_of(planned: PlannedStep, plan: Plan) -> tuple[Entry, ...]:
    """What this step should be asked to compute.

    The narrowed list where the plan produced one, and the whole scope otherwise
    -- which is what a cold step and an all-or-nothing op step both get.

    :attr:`Scope.is_unset` decides between the two. Every selector instance is
    truthy, and testing the selector itself keeps the whole-scope answer from
    ever being given.

    An op declaring ``scope_takes = "none"`` is the one step that computes no
    entry, and it returns ``()``. Its unset selector means it covers none of
    them, inverting the rule above. The callers of this function query the
    failure store by entry, and reading the whole plan scope here would
    quarantine a training step over entries it never touched.

    A selector naming groups or sequences is refused. ``plan_pipeline``
    enumerates one against the tracks universe before any step is planned, and
    neither answer below fits it. The unset answer covers every entry in the
    plan and the named one covers none.

    Raises:
        ValueError: ``planned.spec.entries`` names groups or sequences instead
            of entries.
    """
    if planned.kind == "op":
        op_cls = op_class_for_kind(planned.spec.op_kind)
        if op_cls is not None and op_cls.scope_takes == "none":
            return ()
    entries = planned.spec.entries
    if entries.groups is not None or entries.sequences is not None:
        raise ValueError(
            f"asked_of returns the entries a step will compute and does not "
            f"read an index to enumerate them. Step {planned.step_id!r} was "
            f"given {entries!r}, which names groups or sequences. Plan the "
            f"recipe first and pass each step the Scope(entries=[...]) that "
            f"plan_pipeline enumerates."
        )
    if entries.is_unset:
        return tuple(sorted(plan.scope))
    return tuple(sorted(entries.entry_pairs or ()))


def _run(
    ds: Dataset,
    planned: PlannedStep,
    asked: tuple[Entry, ...],
    *,
    attempt: str,
    overwrite: bool,
    store: FailureStore,
    owner: str,
    cancel_token: CancelToken | None,
) -> StepOutcome:
    """Do the work, and record what it lost.

    A step whose identity the plan could not resolve is run anyway: the deferral
    blocks the *preview*, and by the time the step starts the artifact its
    identity covers is on disk and it names itself.
    """
    if planned.kind == "op":
        run_id = run_op(
            ds,
            planned.spec.op_kind,
            build_step_op_params(planned.spec),
            # The selector the plan gave this step, rather than the entry tuple
            # *asked* enumerates from it. The planner resolved this same
            # selector for plan_identity, and a step cannot cover more than the
            # identifier it was minted for.
            scope=planned.spec.entries,
            overwrite=overwrite,
            execution_id=attempt,
            owner=owner,
            cancel_token=cancel_token,
        )
        return StepOutcome(
            step_id=planned.step_id,
            kind="op",
            run_id=run_id,
            state="ran",
            planned_run_id=planned.run_id,
            execution_id=attempt,
        )

    spec = planned.spec
    try:
        result = run_feature(
            ds,
            build_step_feature(spec),
            scope=Scope(entries=sorted(asked)) if asked else None,
            overwrite=overwrite,
            tracks_run_id=spec.tracks_run_id,
            overlap_frames=spec.overlap_frames,
            filter_start_frame=spec.filter_start_frame,
            filter_end_frame=spec.filter_end_frame,
            filter_start_time=spec.filter_start_time,
            filter_end_time=spec.filter_end_time,
            execution_id=attempt,
            owner=owner,
            cancel_token=cancel_token,
        )
    except Exception as exc:
        store.note_step_failure(
            (planned.storage_name, planned.run_id or ""),
            error=f"{type(exc).__name__}: {exc}",
            execution_id=attempt,
        )
        raise
    if result.run_id is None:
        # ``Result.run_id`` is optional because a *reference* may name the latest
        # run rather than one identifier; a Result a run just produced always
        # carries the name it produced it under. Refused rather than coerced,
        # because an outcome recording an empty identifier is a lie about what is
        # on disk, and every step below reads it.
        raise RuntimeError(
            f"step {planned.step_id!r} ran but recorded no run_id, so nothing "
            f"below it can name what it produced"
        )
    _record_entry_outcomes(store, planned, result.run_id, asked, result.failed_entries)
    store.clear_step((planned.storage_name, result.run_id))
    return StepOutcome(
        step_id=planned.step_id,
        kind="feature",
        run_id=result.run_id,
        state="ran",
        planned_run_id=planned.run_id,
        execution_id=attempt,
        failed_entries=result.failed_entries,
    )


def _record_entry_outcomes(
    store: FailureStore,
    planned: PlannedStep,
    run_id: str,
    asked: tuple[Entry, ...],
    failed_keys: tuple[str, ...],
) -> None:
    """Count what this attempt lost, and forget what it recovered.

    Keyed by entry rather than by step, because a deterministically bad sequence
    would otherwise quarantine a whole branch: one corrupt video is a reason to
    stop retrying that video, not to stop the pipeline.
    """
    by_key = {
        make_entry_key(group, sequence): (group, sequence) for group, sequence in asked
    }
    failed: set[Entry] = set()
    for key in failed_keys:
        entry = by_key.get(key)
        if entry is None:
            # An entry key nothing asked for: the run widened its own scope, or
            # the key spells something this map cannot reverse. Recorded against
            # the step rather than dropped, since losing the count would be
            # losing the bound on retrying it.
            store.note_step_failure(
                (planned.storage_name, run_id), error=f"entity {key} failed"
            )
            continue
        failed.add(entry)
        store.note_entry_failure(
            (planned.storage_name, run_id, entry[0], entry[1]),
            error=f"entity {key} failed",
        )
    for entry in asked:
        if entry not in failed:
            store.clear_entry((planned.storage_name, run_id, entry[0], entry[1]))


def _cached(
    ds: Dataset, planned: PlannedStep, *, attempt: str, owner: str
) -> StepOutcome:
    """The record for a step whose artifact already answered for its whole scope.

    It still writes a run-log. A queue releases a child on its parent's terminal
    record and a child pins its parent's identity from the same file, so a step
    that legitimately did nothing must be as legible as one that did everything.

    ``entries_written`` is the artifact's coverage -- the same number the outcome
    reports as ``covered``, by the same rule -- so the log and the outcome cannot
    disagree about one step. Deliberately not zero: the field counts what the scope
    holds at the end of the attempt, cache hits included, and a cached step holds
    all of it. Zero would read as a step that lost everything.
    """
    covered = len(planned.coverage.target)
    _record_terminal(
        ds,
        attempt,
        kind=planned.kind,
        target=planned.storage_name or planned.runs,
        owner=owner,
        run_id=planned.run_id or "",
        cache_hit=True,
        entries_written=covered,
    )
    return StepOutcome(
        step_id=planned.step_id,
        kind=planned.kind,
        run_id=planned.run_id or "",
        state="cached",
        planned_run_id=planned.run_id,
        execution_id=attempt,
        covered=covered,
        target=covered,
    )


def _stalled(
    ds: Dataset, planned: PlannedStep, *, attempt: str, owner: str
) -> StepOutcome:
    """The record for a step with nothing left it is allowed to attempt.

    Distinct from a failure on purpose. A step whose remaining entries are all
    held back has not gone wrong -- somebody decided to proceed without them --
    and reporting it as an error would invite a retry that cannot succeed.

    ``entries_written`` is what the scope already holds, matching the ``covered``
    on the outcome below. A stall is not a cache hit: the work was not found done,
    it was found unattemptable.
    """
    _record_terminal(
        ds,
        attempt,
        kind=planned.kind,
        target=planned.storage_name or planned.runs,
        owner=owner,
        run_id=planned.run_id or "",
        entries_written=len(planned.coverage.covered),
    )
    return StepOutcome(
        step_id=planned.step_id,
        kind=planned.kind,
        run_id=planned.run_id or "",
        state="stalled",
        planned_run_id=planned.run_id,
        execution_id=attempt,
        covered=len(planned.coverage.covered),
        target=len(planned.coverage.target),
    )


def _record_terminal(
    ds: Dataset,
    execution_id: str,
    *,
    kind: str,
    target: str,
    owner: str,
    run_id: str = "",
    error_json: str = "",
    cache_hit: bool = False,
    entries_written: int = 0,
) -> None:
    """Write a whole attempt record for a step that runs nothing itself.

    The three cases are a refusal, a cache hit and a stall, and each needs a
    run-log for the same reason: it is the release signal and the identity a child
    pins. Written directly rather than through ``job_context`` because that context
    brackets an attempt that *does* work, and a second writer on one file is what
    the run-log format rules out.

    Every fact is written *before* the terminal event. A queue projects an attempt
    the moment it sees one, so anything appended after it is a fact that exists on
    disk and never reaches the ledger.
    """
    log = JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id)
    try:
        log.started(kind=kind, target=target, owner=owner, host=socket.gethostname())
        if run_id:
            log.set_run_id(run_id)
        if cache_hit:
            log.cache_hit()
        log.entries_written(entries_written)
        if error_json:
            log.failed(error_json)
        else:
            log.finished()
    finally:
        log.close()
