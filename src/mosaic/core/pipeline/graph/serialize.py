"""A plan as the bounded JSON an editor previews, and the fingerprint naming it.

The models it fills are defined in :mod:`.wire`, apart from the planning modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

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
    CoverageShortWire,
    DepsIncompleteWire,
    HeldOnParentsWire,
    IdentityUnresolvedWire,
    PlannedStepWire,
    PlanWire,
    Problem,
    ReasonWire,
    StalledWire,
    WaitingOnResourceWire,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

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
    steps = tuple(_step_wire(planned, plan, missing_sample) for planned in plan.steps)
    return PlanWire(
        recipe_digest=plan.recipe_digest,
        fingerprint=plan_fingerprint(plan),
        scope_size=len(plan.scope),
        complete=plan.is_complete,
        steps=steps,
        problems=plan.problems,
        cautions=_plan_cautions(plan, steps),
        open_requests=open_requests,
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
    so one plan has one fingerprint in every process.
    """
    document: dict[str, JsonValue] = {
        "dataset": plan.dataset_uuid,
        "recipe": plan.recipe_digest,
        "steps": [
            _step_fingerprint(planned, plan)
            for planned in sorted(plan.steps, key=lambda planned: planned.step_id)
        ],
    }
    return document_digest(document)


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
    planned: PlannedStep, plan: Plan, missing_sample: int
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
