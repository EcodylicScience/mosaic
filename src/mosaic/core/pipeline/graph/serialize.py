"""A plan as the bounded JSON an editor previews.

A module of its own because it reads both :mod:`.plan` and :mod:`.step`, and
:mod:`.step` already imports :mod:`.plan`. The models it fills live in
:mod:`.wire`, which imports neither.
"""

from __future__ import annotations

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
)
from .step import asked_of
from ..inventory.wire import coverage_wire
from .wire import (
    CoverageShortWire,
    DepsIncompleteWire,
    HeldOnParentsWire,
    IdentityUnresolvedWire,
    PlannedStepWire,
    PlanWire,
    ReasonWire,
    StalledWire,
    WaitingOnResourceWire,
)

__all__ = ["plan_wire"]


def plan_wire(plan: Plan) -> PlanWire:
    """Serialize *plan* with a structured reason and a bounded coverage summary.

    Full entry lists are left out. A step reports how many entries it holds and
    would be asked for, plus a sample of what is missing. The plan's problems
    are included. A plan that :class:`~.plan.RecipeRefused` contains has them.
    """
    return PlanWire(
        recipe_digest=plan.recipe_digest,
        scope_size=len(plan.scope),
        complete=plan.is_complete,
        steps=tuple(_step_wire(planned, plan) for planned in plan.steps),
        problems=plan.problems,
    )


def _step_wire(planned: PlannedStep, plan: Plan) -> PlannedStepWire:
    return PlannedStepWire(
        step=planned.step_id,
        type=planned.kind,
        runs=planned.runs,
        run_id=planned.run_id,
        status=planned.status,
        finished=planned.finished,
        coverage=coverage_wire(planned.coverage),
        asked=len(asked_of(planned, plan)),
        drift=len(planned.drift),
        lane=planned.lane,
        parents=planned.parents,
        storage_name=planned.storage_name,
        tracks_variant=planned.tracks_variant,
        model_run_id=planned.model_run_id,
        reason=_reason_wire(planned.reason),
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
