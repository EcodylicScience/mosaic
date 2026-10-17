"""A plan as the bounded JSON an editor previews, and the fingerprint naming it.

The models it fills are defined in :mod:`.wire`, apart from the planning modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

from .digest import document_digest
from .plan import (
    CoverageShort,
    DepsIncomplete,
    HeldOnParents,
    IdentityUnresolved,
    Plan,
    PlannedStep,
    Reason,
    Stalled,
    WaitingOnResource,
    asked_of,
)
from ..inventory.wire import MISSING_SAMPLE_WIRE, coverage_wire
from .wire import (
    SUPERSEDED_SAMPLE,
    CleanupPlanWire,
    CoverageShortWire,
    DepsIncompleteWire,
    HeldOnParentsWire,
    IdentityUnresolvedWire,
    PlannedStepWire,
    PlanWire,
    Problem,
    ReasonWire,
    StalledWire,
    SupersededRunWire,
    WaitingOnResourceWire,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue
    from mosaic.core.scope import Scope

    from .cleanup import CleanupPlan, SupersededRun
    from .status import RequestSummaryWire

__all__ = ["LARGE_SCOPE_ENTRIES", "plan_fingerprint", "plan_wire"]

LARGE_SCOPE_ENTRIES: Final = 100
"""The most entries a step may be asked for before its plan cautions it."""


def plan_wire(
    plan: Plan,
    *,
    missing_sample: int = MISSING_SAMPLE_WIRE,
    open_requests: tuple[RequestSummaryWire, ...] = (),
) -> PlanWire:
    """Serialize *plan* with a structured reason and a bounded coverage summary.

    Full entry lists are left out. A step reports how many entries it holds and
    would be asked for, plus a sample of at most *missing_sample* of what is
    missing. The plan's problems are included. A plan that
    :class:`~.plan.RecipeRefused` contains has them. So are its cautions: a
    plan over the whole dataset, and each step asked for more than
    :data:`LARGE_SCOPE_ENTRIES` entries. The fingerprint depends on neither
    the cautions nor *missing_sample*.

    Args:
        plan: The plan.
        missing_sample: How many missing entries each step names at most.
        open_requests: The requests of the recipe that have not ended, as
            :func:`~.store.requests_matching` reads them. They are reported as
            given and do not enter the fingerprint.
    """
    superseded = _superseded_by_step(plan.cleanup)
    steps = tuple(
        _step_wire(planned, plan, missing_sample, superseded.get(planned.step_id, ()))
        for planned in plan.steps
    )
    return PlanWire(
        recipe_digest=plan.recipe_digest,
        fingerprint=plan_fingerprint(plan),
        scope_size=len(plan.scope),
        complete=plan.is_complete,
        steps=steps,
        problems=plan.problems,
        cautions=_plan_cautions(plan, steps),
        open_requests=open_requests,
        cleanup=_cleanup_wire(plan.cleanup),
    )


def _superseded_by_step(
    cleanup: CleanupPlan | None,
) -> dict[str, tuple[SupersededRun, ...]]:
    """The superseded runs of *cleanup*, grouped on the step that lists them."""
    grouped: dict[str, list[SupersededRun]] = {}
    for run in cleanup.runs if cleanup is not None else ():
        grouped.setdefault(run.step_id, []).append(run)
    return {step_id: tuple(runs) for step_id, runs in grouped.items()}


def _cleanup_wire(cleanup: CleanupPlan | None) -> CleanupPlanWire | None:
    """The totals of *cleanup*, ``None`` for a plan not asked for one."""
    if cleanup is None:
        return None
    return CleanupPlanWire(
        policy=cleanup.policy,
        removable_runs=len(cleanup.removable),
        removable_bytes=cleanup.removable_bytes,
        kept_runs=len(cleanup.kept),
        kept_bytes=cleanup.kept_bytes,
    )


def _superseded_wire(run: SupersededRun) -> SupersededRunWire:
    return SupersededRunWire(
        run_id=run.run_id,
        storage=run.storage,
        removable=run.removable,
        reason=run.reason,
        detail=run.detail,
        bytes=run.bytes,
        entries=run.entries,
        finished_at=run.finished_at,
        superseded_by=run.superseded_by,
    )


def _plan_cautions(
    plan: Plan, steps: tuple[PlannedStepWire, ...]
) -> tuple[Problem, ...]:
    """What the person submitting *plan* should know first. None refuses it.

    A plan asked for no narrowing covers the whole dataset. A step of *steps*
    is cautioned on the entries it is asked for (:func:`~.plan.asked_of`), its
    own and not the plan's scope.
    """
    cautions: list[Problem] = []
    if plan.covers_dataset:
        cautions.append(
            Problem(
                "",
                "scope",
                f"the plan covers all {len(plan.scope)} entries of the dataset",
            )
        )
    for step in steps:
        if step.asked > LARGE_SCOPE_ENTRIES:
            cautions.append(
                Problem(
                    step.step,
                    "scope",
                    f"asked for {step.asked} entries, more than {LARGE_SCOPE_ENTRIES}",
                )
            )
    return tuple(cautions)


def plan_fingerprint(plan: Plan) -> str:
    """Name what *plan* found on its dataset, for a submission to compare against.

    The digest covers the dataset's manifest ``uuid``, the recipe digest, and per
    step its id, ``run_id``, status, finished flag, coverage counts, drift count
    and the entries it would be asked for. A new feature run or a new tracks
    entry therefore changes it. A step's ``reason`` is left out, because a
    resource's occupancy changes from one second to the next. The plan's
    problems and cautions are left out too. A plan with problems is refused at
    submit whatever its fingerprint, and a caution refuses nothing.

    The document is canonical JSON, with steps sorted by id and entries sorted,
    so one plan has one fingerprint in every process. The plan's step scopes
    enter it only when there are any, which leaves the fingerprint of a plan
    without them unchanged. They are needed while a conversion defers every
    step's entries, when the counts above cannot tell two narrowings apart.

    A plan asked for a cleanup (:func:`~.cleanup.plan_with_cleanup`) adds the
    policy and the runs it may remove, so a submission removes only the runs
    its caller reviewed. Their sizes and the runs kept are left out. A plan not
    asked for one fingerprints as before.
    """
    document: dict[str, JsonValue] = {
        "dataset": plan.dataset_uuid,
        "recipe": plan.recipe_digest,
        "steps": [
            _step_fingerprint(planned, plan)
            for planned in sorted(plan.steps, key=lambda planned: planned.step_id)
        ],
    }
    if plan.step_scopes:
        document["step_scopes"] = {
            step_id: _scope_document(narrowed)
            for step_id, narrowed in sorted(plan.step_scopes.items())
        }
    if plan.cleanup is not None:
        policy = plan.cleanup.policy
        removable = sorted((run.storage, run.run_id) for run in plan.cleanup.removable)
        remove: list[JsonValue] = [[storage, run_id] for storage, run_id in removable]
        document["cleanup"] = {
            "policy": {
                "keep_newest": policy.keep_newest,
                "min_age_hours": policy.min_age_hours,
            },
            "remove": remove,
        }
    return document_digest(document)


def _scope_document(scope: Scope) -> JsonValue:
    """*scope* as JSON with each list sorted, so its order names nothing."""
    dumped: dict[str, JsonValue] = {}
    for name, value in scope.model_dump(mode="json").items():
        held = cast("JsonValue", value)
        dumped[name] = (
            sorted(held, key=lambda item: json.dumps(item))
            if isinstance(held, list)
            else held
        )
    return dumped


def _step_fingerprint(planned: PlannedStep, plan: Plan) -> dict[str, JsonValue]:
    coverage = planned.coverage
    return {
        "step": planned.step_id,
        "run_id": planned.run_id,
        "status": planned.status,
        "finished": planned.finished,
        "covers_all": coverage.covers_all,
        "covered": len(coverage.covered),
        "target": len(coverage.target),
        "drift": len(planned.drift),
        "asked": [[group, sequence] for group, sequence in asked_of(planned, plan)],
    }


def _step_wire(
    planned: PlannedStep,
    plan: Plan,
    missing_sample: int,
    superseded: tuple[SupersededRun, ...] = (),
) -> PlannedStepWire:
    return PlannedStepWire(
        step=planned.step_id,
        type=planned.kind,
        runs=planned.runs,
        run_id=planned.run_id,
        status=planned.status,
        finished=planned.finished,
        coverage=coverage_wire(planned.coverage, sample=missing_sample),
        asked=len(asked_of(planned, plan)),
        drift=len(planned.drift),
        lane=planned.lane,
        parents=planned.parents,
        storage_name=planned.storage_name,
        tracks_variant=planned.tracks_variant,
        model_run_id=planned.model_run_id,
        reason=_reason_wire(planned.reason),
        labels_variant=planned.labels_variant,
        step_scope=plan.step_scopes.get(planned.step_id),
        superseded=tuple(
            _superseded_wire(run) for run in superseded[:SUPERSEDED_SAMPLE]
        ),
        superseded_omitted=max(len(superseded) - SUPERSEDED_SAMPLE, 0),
    )


def _reason_wire(reason: Reason | None) -> ReasonWire | None:
    """The reason tagged by ``kind``, with each ``missing`` as ``missing_sample``."""
    match reason:
        case None:
            return None
        case DepsIncomplete():
            return DepsIncompleteWire(blocking=reason.blocking)
        case CoverageShort():
            return CoverageShortWire(
                covered=reason.covered,
                target=reason.target,
                missing_sample=reason.missing,
            )
        case HeldOnParents():
            return HeldOnParentsWire(parents=reason.parents)
        case WaitingOnResource():
            return WaitingOnResourceWire(
                resource_class=reason.resource_class,
                in_use=reason.in_use,
                capacity=reason.capacity,
            )
        case Stalled():
            return StalledWire(
                covered=reason.covered,
                target=reason.target,
                missing_sample=reason.missing,
            )
        case IdentityUnresolved():
            return IdentityUnresolvedWire(step=reason.step, because=reason.because)
