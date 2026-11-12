"""How much a step reads, and which of its entries a plain attempt would compute.

A caller that sizes a step before running it (a queue choosing memory, a script
cutting an array of attempts) asks this instead of planning the recipe itself. The
step is planned as :func:`~.step.execute_step` plans it: against the parents'
recorded runs and the request's exclusions, with nothing refused. Its feature is
built and its manifest resolved over the step's entries, and the rows come from the
input files' footers (:mod:`mosaic.core.pipeline.input_size`), so no table is read
and nothing is written.

**A step that cannot be sized says why.** An op step reads no table. A step below an
op or a feature that has not written what it reads has no manifest yet. Both report
``inputs_ready=False`` with the reason, and the counts of entries stay.

**``run_id`` is a prediction.** It is the plan's identity for the step, or ``None``
where the plan cannot resolve it yet because an op of the recipe has not written
what the step reads. The step plans its identity again when it starts, so for a
``scope_dependent`` step the value moves if entries are quarantined before then, and
a bound or pinned step gives the run it serves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mosaic.core.entry import Entry, entry_token
from mosaic.core.scope import Scope

from ..input_size import EntryInputSize, by_rows, manifest_input_sizes
from ..loading import resolve_sequence_identity, stacks_inputs
from ..manifest import UnrunFeatureInput, build_manifest
from ..resolve import resolve_references
from .claims import FileFailureStore
from .model import GraphModel
from .parents import pinned_parents
from .plan import asked_of, plan_pipeline, step_target
from .quarantine import narrow_by_quarantine
from .request import load_recipe_for_request
from .resolve import build_step_feature

if TYPE_CHECKING:
    from collections.abc import Collection

    from mosaic.core.dataset import Dataset

    from .model import Request
    from .plan import Plan, PlannedStep

__all__ = ["EntryRowsWire", "StepInputSizeWire", "step_entry_sizes", "step_input_size"]


class EntryRowsWire(GraphModel):
    """One entry of a step and the rows it reads.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        token: ``group:sequence`` (a bare sequence in the empty group), the form
            ``--step-entries`` takes.
        rows: The footer rows of the entry's input files. 0 for an entry the
            step's inputs do not hold, or whose footers cannot be read.
    """

    group: str
    sequence: str
    token: str
    rows: int


class StepInputSizeWire(GraphModel):
    """The input size of one step of one request.

    Attributes:
        request_id: The request.
        step: The step's id in the recipe.
        kind: Whether the step runs a feature or an op.
        inputs_ready: False when the step's inputs cannot be measured yet: an op
            step, a step whose inputs name a run that is not written, or a step
            the plan refuses.
        reason: Why the inputs are not ready, ``""`` otherwise.
        scope_dependent: Whether the step's identity covers its set of entries,
            so an attempt computes all of them or none.
        fits: Whether the step learns from the entries it covers.
        entries: The step's entries.
        entries_to_compute: The entries a plain attempt would compute: the
            uncomputed ones, the drifted ones of a complete-but-drifted step,
            and every entry of a ``scope_dependent`` step. 0 for a step that is
            complete, bound or pinned.
        rows: The footer rows of the step's input files over its entries.
        rows_to_compute: The same over the entries to compute.
        largest_entry_rows: The rows of the step's largest entry.
        columns_max: The most columns any entry's input files hold, summed over
            the files of that entry.
        entries_to_compute_by_rows: Every entry to compute with its rows, largest
            first and ties by ``(group, sequence)``. An entry quarantined and
            excluded by the request is left out, as the plan leaves it out. An
            entry waiting out a backoff is listed, because a backoff narrows one
            attempt only.
        run_id: The run the plan names for the step, or ``None`` while an op of
            the recipe has not written what it reads. A prediction: the step
            plans its identity again when it starts.
    """

    request_id: str
    step: str
    kind: Literal["feature", "op"]
    inputs_ready: bool
    reason: str
    scope_dependent: bool
    fits: bool
    entries: int
    entries_to_compute: int
    rows: int
    rows_to_compute: int
    largest_entry_rows: int
    columns_max: int
    entries_to_compute_by_rows: tuple[EntryRowsWire, ...]
    run_id: str | None


def step_input_size(ds: Dataset, request: Request, step_id: str) -> StepInputSizeWire:
    """Report what step *step_id* of *request* reads, and its entries to compute.

    Reads only: the plan is made with the request's exclusions as they are, the
    failure store is not written, and no lock is taken.

    A complete, bound or pinned step computes nothing, so it has no entries to
    compute. Entries quarantined by repeated failures are left out as the step's
    own attempt leaves them out: with ``request.allow_partial`` the step is planned
    again without them, which can change its ``run_id`` and, for a
    ``scope_dependent`` step, its entries. Without it the step would be refused
    for the shortfall, so ``inputs_ready`` is False with that reason.

    Args:
        ds: The dataset, which may be opened read-only.
        request: The submission the step belongs to.
        step_id: The step.

    Returns:
        The step's sizes. ``inputs_ready`` is False, with the reason, for a step
        that cannot be measured yet.

    Raises:
        KeyError: The recipe has no step *step_id*.
        StepRefused: A parent step's run-log records no run.
    """
    recipe = load_recipe_for_request(ds.base_dir, request)
    _ = recipe.step(step_id)
    store = FileFailureStore(ds.base_dir)
    excluded = set(store.exclusions(request.request_id).entries)
    recorded = {**pinned_parents(ds, recipe, request, step_id), **request.pins}

    def plan_without(extra: Collection[Entry]) -> Plan:
        return plan_pipeline(
            ds,
            recipe,
            request=request,
            recorded=recorded,
            quarantined=excluded | set(extra),
            refuse=False,
            ignore_unavailable=True,
        )

    plan = plan_without(())
    planned = plan.step(step_id)
    held_back = ""
    if planned.is_complete:
        # The step's own attempt serves its run and computes nothing.
        asked: tuple[Entry, ...] = ()
    else:
        narrowed = narrow_by_quarantine(
            store,
            plan,
            step_id,
            allow_partial=request.allow_partial,
            replan=plan_without,
        )
        plan, planned = narrowed.plan, narrowed.planned
        if planned.is_complete:
            asked = ()
        elif narrowed.quarantined and not request.allow_partial:
            asked = tuple(
                entry for entry in narrowed.asked if entry not in narrowed.quarantined
            )
            held_back = str(narrowed.shortfall())
        else:
            asked = asked_of(planned, plan)
    target = step_target(planned, plan)

    reason = _unready(planned, plan)
    sizes: dict[Entry, EntryInputSize] = {}
    if not reason:
        sizes, reason = step_entry_sizes(ds, planned, target)
    reason = reason or held_back

    rows_of = {entry: (sizes[entry].rows if entry in sizes else 0) for entry in asked}
    ordered = by_rows(rows_of)
    return StepInputSizeWire(
        request_id=request.request_id,
        step=step_id,
        kind=planned.kind,
        inputs_ready=not reason,
        reason=reason,
        scope_dependent=planned.scope_dependent,
        fits=planned.fits,
        entries=len(target),
        entries_to_compute=len(asked),
        rows=sum(size.rows for size in sizes.values()),
        rows_to_compute=sum(rows_of.values()),
        largest_entry_rows=max((size.rows for size in sizes.values()), default=0),
        columns_max=max((size.columns for size in sizes.values()), default=0),
        entries_to_compute_by_rows=tuple(
            EntryRowsWire(
                group=group,
                sequence=sequence,
                token=entry_token(group, sequence),
                rows=rows_of[(group, sequence)],
            )
            for group, sequence in ordered
        ),
        run_id=planned.run_id,
    )


def _unready(planned: PlannedStep, plan: Plan) -> str:
    """Why *planned* has no inputs to measure before any is read, or ``""``."""
    if planned.kind == "op":
        return f"step {planned.step_id!r} is an op, which reads no input table"
    own = [problem for problem in plan.problems if problem.step == planned.step_id]
    if own:
        return f"the plan refuses step {planned.step_id!r}: {own[0].message}"
    return ""


def step_entry_sizes(
    ds: Dataset, planned: PlannedStep, target: Collection[Entry]
) -> tuple[dict[Entry, EntryInputSize], str]:
    """Return the input size of each entry of *target* that the step's inputs hold.

    Args:
        ds: The dataset, which may be opened read-only.
        planned: The feature step, as its plan resolved it.
        target: The entries to measure.

    Returns:
        The sizes keyed by entry, and the reason they could not be read, which is
        ``""`` when they were.
    """
    feature = build_step_feature(planned.spec)
    if feature.inputs.is_empty:
        return {}, f"step {planned.step_id!r} reads no inputs"
    if not target:
        return {}, f"step {planned.step_id!r} covers no entries yet"
    _ = resolve_references(ds, feature)
    try:
        manifest, resolved = build_manifest(
            ds,
            feature.inputs,
            Scope(entries=sorted(target)),
            tracks_run_id=planned.spec.tracks_run_id,
            stack=stacks_inputs(feature),
        )
    except UnrunFeatureInput as exc:
        return {}, f"an input of step {planned.step_id!r} is not written: {exc}"
    if not manifest:
        return {}, f"the inputs of step {planned.step_id!r} hold none of its entries"
    return {
        resolve_sequence_identity(key, resolved.entry_map): size
        for key, size in manifest_input_sizes(manifest).items()
    }, ""
