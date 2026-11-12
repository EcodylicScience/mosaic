"""What a step's feature ancestors recorded producing, read from their run-logs.

Shared by the step that runs (``step.py``) and the report of a step's input size
(``input_size.py``), which plan the step against the same pinned parents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.runlog import read_run, run_log_dir

from .model import FeatureStepSpec
from .step_preflight import StepRefused
from .topo import ancestors_of

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

    from .model import Recipe, Request

__all__ = ["pinned_parents"]


def pinned_parents(
    ds: Dataset, recipe: Recipe, request: Request, step_id: str
) -> dict[str, str]:
    """What each feature ancestor's attempt recorded producing.

    **Every ancestor, not only the immediate parents.** Identity chains: this
    step's identifier is a function of its parents', theirs of their parents',
    and so on. Planning an ancestor from a prediction while pinning a parent from
    a fact would give this step an identifier neither of them agrees with.

    A step the request binds or pins to a run is that run and stands for
    nothing above it, so neither it nor what is only above it is read
    (``cut``), and such a step pins nothing.

    An ancestor with no run-log contributes nothing and is resolved normally --
    that is a step served from an earlier request's cache, which is the ordinary
    state of a partly built graph, not a fault.
    """
    if step_id in request.stand_ins:
        return {}
    logs = run_log_dir(ds.base_dir)
    pinned: dict[str, str] = {}
    above = ancestors_of(recipe, step_id, cut=request.stand_ins) - {step_id}
    for ancestor in sorted(above):
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
