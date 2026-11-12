"""Which entries of a step a request has stopped attempting, and the plan without them.

Shared by the step that runs (``step.py``) and the report of a step's input size
(``input_size.py``), so both leave out the same entries. Nothing here writes: the
caller says how to plan again, and a caller that records the decision does so there.

An entry that failed ``QUARANTINE_AFTER`` times holds a step back until the request
proceeds without it (``allow_partial``). Proceeding narrows the scope, and a
``scope_dependent`` step's identity *is* its scope, so the step is planned again
over what is left and may name another run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .plan import MISSING_SAMPLE, Plan, PlannedStep, asked_of
from .step_preflight import CoverageShortfall

if TYPE_CHECKING:
    from mosaic.core.entry import Entry

    from .failures import FailureStore

__all__ = ["Narrowed", "narrow_by_quarantine"]


@dataclass(frozen=True, slots=True)
class Narrowed:
    """A step's plan after its quarantined entries are accounted for.

    Attributes:
        plan: The plan the step goes on with: planned again without the
            quarantined entries when the request proceeds without them, else the
            plan it was given.
        planned: The step in *plan*.
        asked: What the step was asked to compute in the plan it was given,
            quarantined entries included.
        quarantined: The entries of *asked* that are quarantined.
    """

    plan: Plan
    planned: PlannedStep
    asked: tuple[Entry, ...]
    quarantined: frozenset[Entry]

    def shortfall(self) -> CoverageShortfall:
        """The refusal of a step held back by *quarantined* when it may not proceed."""
        return CoverageShortfall(
            self.planned.step_id,
            len(self.asked) - len(self.quarantined),
            len(self.asked),
            "held back after repeated failures: "
            + ", ".join(
                sorted(str(entry) for entry in self.quarantined)[:MISSING_SAMPLE]
            ),
        )


def narrow_by_quarantine(
    store: FailureStore,
    plan: Plan,
    step_id: str,
    *,
    allow_partial: bool,
    replan: Callable[[frozenset[Entry]], Plan],
) -> Narrowed:
    """Find the quarantined entries of a step and, when allowed, plan without them.

    Args:
        store: The failure records.
        plan: The step's plan, with the step still to run.
        step_id: The step.
        allow_partial: Whether the request proceeds without quarantined entries.
        replan: Plans the step again with these entries added to those the request
            already excludes. Called only when there are some and *allow_partial*
            is set. A caller that records the exclusion does it in here.

    Returns:
        The step as it goes on. Without *allow_partial* the plan is the one given,
        and the caller decides what the quarantined entries mean.
    """
    planned = plan.step(step_id)
    asked = asked_of(planned, plan)
    quarantined = store.quarantined_entries(
        planned.storage_name, planned.run_id or "", asked
    )
    if quarantined and allow_partial:
        plan = replan(quarantined)
        planned = plan.step(step_id)
    return Narrowed(plan, planned, asked, quarantined)
