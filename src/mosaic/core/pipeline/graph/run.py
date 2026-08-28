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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mosaic.core.scope import Scope

from .plan import plan_pipeline
from .preflight import CoverageShortfall
from .request import submit_request
from .step import StepOutcome, execute_step
from .topo import topological_order

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry

    from ..inventory.cache import InventoryCache
    from .compatibility import DeclarationCatalog
    from .failures import FailureStore
    from .model import Recipe, Request

__all__ = [
    "CoverageShortfall",
    "PipelineRun",
    "StepOutcome",
    "run_pipeline",
]


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
) -> PipelineRun:
    """Run every step of *recipe* against *ds*, in order, in this process.

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

    Returns:
        One :class:`~mosaic.core.pipeline.graph.step.StepOutcome` per step, in
        the order they were run.

    Raises:
        StepRefused: A step declined to run, naming why. ``CoverageShortfall`` is
            the case a caller most often means to catch.
        RecipeInvalid: The recipe is malformed.
    """
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

    outcomes: list[StepOutcome] = []
    for step in topological_order(recipe):
        outcome = execute_step(
            ds,
            submitted,
            step.id,
            store=store,
            inventory=inventory,
            catalog=catalog,
            owner=owner,
        )
        outcomes.append(outcome)
        if outcome.state == "stalled" and not submitted.allow_partial:
            # Stopped after one stall rather than after K attempts. A feature
            # legitimately producing fewer outputs than inputs never reads as
            # complete and would be resubmitted forever; an attempt counter
            # renders a *correct* pipeline red and invites a retry that cannot
            # succeed.
            raise CoverageShortfall(
                step.id,
                outcome.covered,
                outcome.target,
                "the attempt exited clean and coverage did not move",
            )

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
