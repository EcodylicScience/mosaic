"""Running a plan here, in this process, one step after another.

**Not a fallback and not a rejected alternative.** It is what a notebook uses,
what a bare HPC node with no queue uses, and what Dagster's own default executor
is. A pipeline that only runs under a control plane is a pipeline most of its
users cannot run at all.

**Each step is planned again at its own start.** That is the rule a queued job
follows, applied here for the same reason: a ``run_id`` resolved before anything
ran drives the preview and nothing else, and the value that governs is the one
the step resolves once it has what the plan lacked. Re-planning per step is also
what makes the loop correct rather than optimistic -- a step's coverage, and
therefore what it should be asked to compute, changes the moment its parent
finishes.

**Planning does not submit.** This module runs; ``plan_pipeline`` resolves. They
are kept apart because the queue package already depends on this one, so a
planner that submitted would be a package cycle -- and because a preview that
could start work is a preview nobody dares ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..ops import run_op
from ..run import run_feature
from .plan import (
    COMPLETE_STATUSES,
    MISSING_SAMPLE,
    Plan,
    PlannedStep,
    is_stalled,
    plan_pipeline,
)
from .resolve import build_step_feature, build_step_op_params

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.dataset import Dataset

    from ..inventory.cache import InventoryCache
    from ..inventory.model import Entry
    from .compatibility import DeclarationCatalog
    from .model import Recipe, Request

__all__ = [
    "CoverageShortfall",
    "PipelineRun",
    "StepOutcome",
    "run_pipeline",
]


class CoverageShortfall(RuntimeError):
    """A step cannot be run over everything it was planned for.

    Raised rather than proceeded through, because for a ``scope_dependent`` step
    it is a scientific question rather than a maintenance one: a model fitted on
    89 sequences is a different model from one fitted on 90, and mosaic says so
    by giving it a different name. Proceeding is a decision, and
    ``allow_partial`` is where that decision is recorded.
    """

    def __init__(self, step_id: str, covered: int, target: int, detail: str) -> None:
        self.step_id: str = step_id
        self.covered: int = covered
        self.target: int = target
        super().__init__(
            f"step {step_id!r} would run over {covered} of {target} entries: "
            f"{detail}. Its identity covers the set it was fitted on, so this is "
            f"a different run from the one planned. Pass allow_partial to proceed "
            f"deliberately, or complete the steps above it first."
        )


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """What happened to one step of one run.

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
    """

    step_id: str
    kind: Literal["feature", "op"]
    run_id: str
    state: Literal["ran", "cached", "stalled"]
    planned_run_id: str | None = None

    @property
    def diverged(self) -> bool:
        """Did execution record something other than what was resolved?"""
        return self.planned_run_id is not None and self.planned_run_id != self.run_id


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """What running one recipe against one dataset did."""

    recipe_digest: str
    scope: frozenset[Entry]
    outcomes: tuple[StepOutcome, ...] = ()

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
    intended_entries: Iterable[Entry] | None = None,
    request: Request | None = None,
    allow_partial: bool = False,
    inventory: InventoryCache | None = None,
    catalog: DeclarationCatalog | None = None,
    owner: str = "",
) -> PipelineRun:
    """Run every step of *recipe* against *ds*, in order, in this process.

    Args:
        ds: The dataset to run against.
        recipe: The graph. Refused before anything runs if it is malformed.
        intended_entries: The entries to run over, or ``None`` for the dataset's
            own scope.
        request: One submission's choices. Its ``allow_partial`` is honoured, so
            a caller holding a request does not pass the flag twice.
        allow_partial: Proceed through a shortfall or a stall rather than
            refusing.
        inventory: A held inventory, revalidated on every planning call.
        catalog: The declarations to resolve against.
        owner: Recorded on each attempt.

    Returns:
        One :class:`StepOutcome` per step, in the order they were run.

    Raises:
        CoverageShortfall: A step would run over less than it was planned for, or
            produced nothing new, and *allow_partial* is not set.
        RecipeInvalid: The recipe is malformed.
    """
    proceed = allow_partial or (request is not None and request.allow_partial)
    # What each step turned out to be called. A resolved identifier is a
    # prediction and this is the fact, so every step below one that landed
    # somewhere else is planned against where it actually landed -- which is the
    # same rule a queued job follows by reading its parent's run-log, applied
    # here where the parent's answer is in hand.
    recorded: dict[str, str] = {}

    def replan() -> Plan:
        return plan_pipeline(
            ds,
            recipe,
            intended_entries=intended_entries,
            request=request,
            inventory=inventory,
            catalog=catalog,
            recorded=recorded,
        )

    plan = replan()
    order = [planned.step_id for planned in plan.steps]
    outcomes: list[StepOutcome] = []

    for step_id in order:
        planned = plan.step(step_id)
        if planned.status in COMPLETE_STATUSES:
            outcomes.append(_cached(planned))
            continue

        _refuse_shortfall(plan, planned, proceed)
        before = len(planned.coverage.covered)
        run_id = _execute(ds, planned, owner=owner)
        recorded[step_id] = run_id

        # One re-plan, read twice: it answers whether this step moved, and it is
        # what the next step is planned from.
        plan = replan()
        after = _covered(plan.step(step_id))
        stalled = is_stalled(
            exited_clean=True,
            terminal=True,
            covered_before=before,
            covered_after=after,
        )
        outcomes.append(
            StepOutcome(
                step_id=step_id,
                kind=planned.kind,
                run_id=run_id,
                state="stalled" if stalled else "ran",
                planned_run_id=planned.run_id,
            )
        )
        if stalled and not proceed:
            # Stopped after one stall rather than after K attempts. A feature
            # legitimately producing fewer outputs than inputs never reads as
            # complete and would be resubmitted forever; an attempt counter
            # renders a *correct* pipeline red and invites a retry that cannot
            # succeed.
            raise CoverageShortfall(
                step_id,
                after,
                len(planned.coverage.target),
                "the attempt exited clean and coverage did not move",
            )

    return PipelineRun(
        recipe_digest=plan.recipe_digest, scope=plan.scope, outcomes=tuple(outcomes)
    )


def _cached(planned: PlannedStep) -> StepOutcome:
    """The record for a step whose artifact already answered for its whole scope."""
    return StepOutcome(
        step_id=planned.step_id,
        kind=planned.kind,
        run_id=planned.run_id or "",
        state="cached",
        planned_run_id=planned.run_id,
    )


def _covered(planned: PlannedStep) -> int:
    """How much of its scope this step holds.

    A complete artifact answers for all of it, including the single-artifact
    kinds whose coverage is not keyed by entry at all -- counting *their* covered
    entries would report a finished model as zero and read every run as a stall.
    """
    if planned.status in COMPLETE_STATUSES:
        return len(planned.coverage.target)
    return len(planned.coverage.covered)


def _refuse_shortfall(plan: Plan, planned: PlannedStep, proceed: bool) -> None:
    """Refuse a step whose inputs do not cover what it was planned over.

    Only for a ``scope_dependent`` step, and the asymmetry is the point: a
    scope-free step running over 89 of 90 entries produces 89 correct outputs
    under the identifier they belong to, and the ninetieth arrives later under
    the same one. A scope-dependent step running over 89 produces *one* artifact
    that is not the artifact anyone asked for, under a name saying it is.
    """
    if proceed or planned.kind != "feature":
        return
    if not build_step_feature(planned.spec).scope_dependent:
        return
    short = [
        parent
        for parent in planned.parents
        if not _covers_scope(plan, parent, plan.scope)
    ]
    if not short:
        return
    covered = min(
        (len(plan.step(parent).coverage.covered) for parent in short), default=0
    )
    missing = sorted(
        str(entry) for parent in short for entry in plan.step(parent).coverage.missing
    )
    raise CoverageShortfall(
        planned.step_id,
        covered,
        len(plan.scope),
        "missing " + ", ".join(missing[:MISSING_SAMPLE]),
    )


def _covers_scope(plan: Plan, step_id: str, scope: frozenset[Entry]) -> bool:
    """Does this parent hold everything the graph is being run over?"""
    try:
        parent = plan.step(step_id)
    except KeyError:  # pragma: no cover - parents come from the same recipe
        return True
    return parent.status in COMPLETE_STATUSES or not (scope - parent.coverage.covered)


def _execute(ds: Dataset, planned: PlannedStep, *, owner: str) -> str:
    """Run one step and return the identifier it recorded.

    A step whose identity the plan could not resolve is run anyway. The deferral
    blocks the *preview*: by the time the step starts, the artifact its identity
    covers is on disk, and it names itself.
    """
    if planned.kind == "op":
        return run_op(
            ds, planned.spec.op_kind, build_step_op_params(planned.spec), owner=owner
        )
    spec = planned.spec
    result = run_feature(
        ds,
        build_step_feature(spec),
        entries=list(spec.entries) or None,
        tracks_run_id=spec.tracks_run_id,
        overlap_frames=spec.overlap_frames,
        filter_start_frame=spec.filter_start_frame,
        filter_end_frame=spec.filter_end_frame,
        filter_start_time=spec.filter_start_time,
        filter_end_time=spec.filter_end_time,
        owner=owner,
    )
    return result.run_id
