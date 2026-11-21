"""A plan as the bounded JSON an editor previews, and the fingerprint naming it.

The models it fills are defined in :mod:`.wire`, apart from the planning modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

from .cpu_fit_advice import card_device_cautions, cpu_fit_cautions
from .digest import document_digest, recipe_digest
from .heldout import deferred_fit_cautions, listed
from .plan import (
    ConversionAdds,
    CoverageShort,
    DepsIncomplete,
    GrowthNote,
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
from ..memory_limit import usable_memory_bytes
from ..parallelism import (
    ParallelLimits,
    ParallelSettings,
    effective_parallel,
    host_cpus,
)
from .wire import (
    RETIRED_SAMPLE,
    SUPERSEDED_SAMPLE,
    CleanupPlanWire,
    CoverageShortWire,
    DepsIncompleteWire,
    HeldOnParentsWire,
    IdentityUnresolvedWire,
    NotRetiredRecipeWire,
    ParallelWire,
    PlannedStepWire,
    PlanWire,
    Problem,
    ReasonWire,
    RetirementPlanWire,
    StalledWire,
    SupersededRunWire,
    WaitingOnResourceWire,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.json_value import JsonValue
    from mosaic.core.scope import Scope

    from .cleanup import CleanupPlan, SupersededRun
    from .model import CleanupPolicy
    from .retire import RetirementPlan
    from .status import RequestSummaryWire, RetirableRecipeWire

__all__ = [
    "LARGE_SCOPE_ENTRIES",
    "cleanup_fingerprint_parts",
    "plan_fingerprint",
    "plan_wire",
    "retirement_wire",
]

LARGE_SCOPE_ENTRIES: Final = 100
"""The most entries a step may be asked for before its plan cautions it."""


def plan_wire(
    plan: Plan,
    *,
    missing_sample: int = MISSING_SAMPLE_WIRE,
    open_requests: tuple[RequestSummaryWire, ...] = (),
    superseded_sample: int = SUPERSEDED_SAMPLE,
    retired_sample: int = RETIRED_SAMPLE,
    retirable: tuple[RetirableRecipeWire, ...] = (),
    retirable_omitted: int = 0,
    parallel: ParallelLimits | None = None,
    extra_cautions: tuple[Problem, ...] = (),
) -> PlanWire:
    """Serialize *plan* with a structured reason and a bounded coverage summary.

    Full entry lists are left out. A step reports how many entries it holds and
    would be asked for, plus a sample of at most *missing_sample* of what is
    missing. The plan's problems are included. A plan that
    :class:`~.plan.RecipeRefused` contains has them. So are its cautions: a
    plan over the whole dataset, and each step asked for more than
    :data:`LARGE_SCOPE_ENTRIES` entries. The fingerprint depends on neither
    the cautions nor *missing_sample*, nor on *superseded_sample*.

    Args:
        plan: The plan.
        missing_sample: How many missing entries each step names at most.
        open_requests: The requests of the recipe that have not ended, as
            :func:`~.store.requests_matching` reads them. They are reported as
            given and do not enter the fingerprint.
        superseded_sample: How many superseded runs each step lists at most,
            when the plan was asked for a cleanup.
        retired_sample: How many retired runs of storages no step names the
            plan lists at most.
        retirable: The saved recipes whose requests wrote runs
            (:func:`~.retirement.retirable_recipes`), reported as given. They do
            not enter the fingerprint.
        retirable_omitted: How many more recipes there were than *retirable*.
        parallel: What a submission may ask for in parallelism. Each parallelizable
            step then reports the workers a submission that names no count
            receives, and the plan reports the limits and the memory limit that
            applies. The fingerprint depends on neither.
        extra_cautions: Cautions a caller worked out beyond the plan's own, such
            as what ``assume_added`` found. They follow the plan's cautions and
            join no field of the wire.
    """
    superseded = _superseded_by_step(plan.cleanup)
    default = None if parallel is None else parallel.settings_for()
    steps = tuple(
        _step_wire(
            planned,
            plan,
            missing_sample,
            superseded.get(planned.step_id, ()),
            superseded_sample,
            default,
        )
        for planned in plan.steps
    )
    return PlanWire(
        recipe_digest=plan.recipe_digest,
        fingerprint=plan_fingerprint(plan),
        scope_size=len(plan.scope),
        complete=plan.is_complete,
        steps=steps,
        problems=plan.problems,
        cautions=(*_plan_cautions(plan, steps), *extra_cautions),
        open_requests=open_requests,
        cleanup=_cleanup_wire(plan.cleanup, retired_sample),
        retirable=retirable,
        retirable_omitted=retirable_omitted,
        parallel=None if parallel is None else _parallel_wire(parallel),
        locked_steps=plan.locked,
    )


def _parallel_wire(limits: ParallelLimits) -> ParallelWire:
    """The limits a submission is held to, with the processors and memory limit that apply."""
    limit, available = usable_memory_bytes()
    return ParallelWire(
        default_workers=min(limits.settings_for().workers, host_cpus()),
        max_workers=limits.max_workers,
        allow_process_mode=limits.allow_process_mode,
        min_free_memory_fraction=limits.min_free_memory_fraction,
        cpus=host_cpus(),
        memory_total_bytes=limit,
        memory_available_bytes=available,
    )


def _superseded_by_step(
    cleanup: CleanupPlan | None,
) -> dict[str, tuple[SupersededRun, ...]]:
    """The superseded runs of *cleanup*, grouped on the step that lists them.

    A retired run in a storage no step names has no step and is listed by
    :func:`_cleanup_wire`.
    """
    grouped: dict[str, list[SupersededRun]] = {}
    for run in cleanup.runs if cleanup is not None else ():
        if run.step_id:
            grouped.setdefault(run.step_id, []).append(run)
    return {step_id: tuple(runs) for step_id, runs in grouped.items()}


def _cleanup_wire(
    cleanup: CleanupPlan | None, retired_sample: int
) -> CleanupPlanWire | None:
    """The totals of *cleanup*, ``None`` for a plan not asked for one.

    The retired runs of storages no step names are listed here, removable ones
    first and otherwise in the order *cleanup* reports them, at most
    *retired_sample* of them, so a sample never hides a run the cleanup would remove.
    """
    if cleanup is None:
        return None
    unplaced = sorted(
        (run for run in cleanup.runs if not run.step_id),
        key=lambda run: not run.removable,
    )
    return CleanupPlanWire(
        policy=cleanup.policy,
        removable_runs=len(cleanup.removable),
        removable_bytes=cleanup.removable_bytes,
        kept_runs=len(cleanup.kept),
        kept_bytes=cleanup.kept_bytes,
        retired=cleanup.retired,
        retired_runs=tuple(_superseded_wire(run) for run in unplaced[:retired_sample]),
        retired_omitted=max(len(unplaced) - retired_sample, 0),
    )


def retirement_wire(plan: RetirementPlan) -> RetirementPlanWire:
    """Serialize a retirement's plan, every run it judged included.

    Args:
        plan: The plan.

    Returns:
        The plan as its wire.
    """
    cleanup = plan.cleanup
    return RetirementPlanWire(
        fingerprint=plan.fingerprint,
        planned_at=plan.planned_at,
        policy=plan.policy,
        retired=plan.retired,
        not_retired=tuple(
            NotRetiredRecipeWire(
                digest=each.digest, reason=each.reason, message=each.message
            )
            for each in plan.not_retired
        ),
        keepers=tuple(recipe_digest(keeper.recipe) for keeper in plan.keepers),
        removable_runs=len(cleanup.removable),
        removable_bytes=cleanup.removable_bytes,
        kept_runs=len(cleanup.kept),
        kept_bytes=cleanup.kept_bytes,
        runs=tuple(_superseded_wire(run) for run in cleanup.runs),
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
        retired=run.retired_by,
    )


def _plan_cautions(
    plan: Plan, steps: tuple[PlannedStepWire, ...]
) -> tuple[Problem, ...]:
    """What the person submitting *plan* should know first. None refuses it.

    A plan asked for no narrowing covers the whole dataset, and when the dataset
    gives its groups roles the caution names the groups it covers with them, so
    a person submitting it sees that it takes the held-out ones. A step of *steps*
    is cautioned on the entries it is asked for (:func:`~.plan.asked_of`), its
    own and not the plan's scope.
    """
    cautions: list[Problem] = []
    if plan.covers_dataset and plan.deferred_by is not None:
        cautions.append(
            Problem(
                "",
                "scope",
                "the dataset holds no tracks yet, so the plan covers every entry "
                f"that step {plan.deferred_by!r} converts" + _roles_named(plan),
            )
        )
    elif plan.covers_dataset and not plan.scope:
        cautions.append(
            Problem(
                "",
                "scope",
                "the dataset holds no converted tracks, so the plan covers no "
                "entry. Begin the recipe with a convert-tracks step, or convert "
                "the raw tracks first. mosaic describe and describe_dataset list "
                "the next steps",
            )
        )
    elif plan.covers_dataset:
        cautions.append(
            Problem(
                "",
                "scope",
                _covered_sentence(plan) + _roles_covered(plan),
            )
        )
    if plan.growth:
        cautions.append(Problem("", "scope", _growth_sentence(plan.growth)))
    cautions.extend(deferred_fit_cautions(plan))
    cautions.extend(plan.reach_cautions)
    cautions.extend(
        cpu_fit_cautions(plan.steps, {step.step: step.asked for step in steps})
    )
    cautions.extend(card_device_cautions(plan.steps))
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


GROWTH_STEPS_NAMED: Final = 5
"""How many steps the growth caution names before it counts the rest."""


def _growth_sentence(growth: tuple[GrowthNote, ...]) -> str:
    """What the entries added since an earlier run mean for the steps that read it.

    Names the first :data:`GROWTH_STEPS_NAMED` steps, counts the others, and says
    once how to keep the earlier set.
    """
    parts = [_growth_note_sentence(note) for note in growth[:GROWTH_STEPS_NAMED]]
    rest = len(growth) - GROWTH_STEPS_NAMED
    if rest > 0:
        reads = "step reads" if rest == 1 else "steps read"
        parts.append(f"{rest} more {reads} an earlier set in the same way")
    parts.append("Pass a saved selection as the scope to keep the earlier set")
    return ". ".join(parts)


def _growth_note_sentence(note: GrowthNote) -> str:
    """One step's growth: the run it reads and what the plan adds to it.

    A run that was asked for every entry says the others were added since. A run
    asked for a chosen set says what it was asked for and that the others join it,
    since no entry was added.
    """
    added = note.target - note.covered
    joins = "joins" if added == 1 else "join"
    if note.made_run_id is None:
        if note.asked_all:
            return (
                f"step {note.step!r} reads run {note.read_run_id}, which covers "
                f"{_entries(note.covered)}. The plan covers {note.target}. The "
                f"{_entries(added)} added since {joins} it"
            )
        return (
            f"step {note.step!r} reads run {note.read_run_id}, which was asked for "
            f"{_entries(note.covered)}. The plan covers {note.target}. "
            f"{'The other entry' if added == 1 else f'The other {added}'} {joins} it"
        )
    return (
        f"step {note.step!r}: the plan makes run {note.made_run_id} over "
        f"{_entries(note.target)} beside run {note.read_run_id} over "
        f"{note.covered}. The {_entries(added)} added since {joins} the new run"
    )


def _covered_sentence(plan: Plan) -> str:
    """How many entries the plan covers, and which the converters add to them.

    A plan whose converters add nothing says it covers all of the dataset's
    entries. Otherwise the converted entries are counted apart from what each
    converting step adds, an entry two steps add counted once in the total.
    """
    adds = dict(sorted(plan.conversion_adds.items()))
    if not adds:
        return f"the plan covers all {len(plan.scope)} entries of the dataset"
    covered = f"the plan covers the {len(plan.scope)} converted entries of the dataset"
    if len(adds) == 1:
        [(step_id, added)] = adds.items()
        parts: list[str] = []
        if added.entries:
            parts.append(
                f"the {_entries(len(added.entries))} that step {step_id!r} adds"
            )
        if added.unnamed_files:
            parts.append(_unnamed_clause(step_id, added))
        return f"{covered} and {' and '.join(parts)}"
    total = {entry for added in adds.values() for entry in added.entries}
    named: list[str] = []
    for step_id, added in adds.items():
        parts = []
        if added.entries:
            parts.append(f"{len(added.entries)} from step {step_id!r}")
        if added.unnamed_files:
            parts.append(_unnamed_clause(step_id, added))
        named.append(" and ".join(parts))
    if total:
        return f"{covered} and {_entries(len(total))} that its converters add: " + (
            ", ".join(named)
        )
    return f"{covered} and {', '.join(named[:-1])} and {named[-1]}"


def _entries(count: int) -> str:
    """*count* with ``entry`` or ``entries``."""
    return f"{count} {'entry' if count == 1 else 'entries'}"


def _unnamed_clause(step_id: str, added: ConversionAdds) -> str:
    """The entries a step converts from raw files that name no sequence."""
    files = added.unnamed_files
    return (
        f"the entries step {step_id!r} converts from {files} raw "
        f"{'file' if files == 1 else 'files'} without a sequence name"
    )


def _roles_named(plan: Plan) -> str:
    """The roles the manifest gives its groups, for a plan with no entries yet."""
    if not plan.splits:
        return ""
    held = sorted(plan.splits.items())
    named = listed([f"{group!r} ({role})" for group, role in held])
    return f". The manifest gives the groups {named} a role"


def _roles_covered(plan: Plan) -> str:
    """The groups of *plan*'s scope that hold a role, with their roles.

    Empty for a dataset that gives no group a role, which leaves the caution as
    it was. Groups with no role are counted and not named.
    """
    if not plan.splits:
        return ""
    covered = sorted({group for group, _ in plan.scope})
    held = [(group, plan.splits[group]) for group in covered if group in plan.splits]
    unassigned = len(covered) - len(held)
    named = listed([f"{group!r} ({role})" for group, role in held])
    if not held:
        return f", in {unassigned} group(s) with no role"
    return f", in the groups {named}" + (
        f" and {unassigned} with no role" if unassigned else ""
    )


def plan_fingerprint(plan: Plan) -> str:
    """Name what *plan* found on its dataset, for a submission to compare against.

    The digest covers the dataset's manifest ``uuid``, the recipe digest, and per step
    its id, ``run_id``, status, finished flag, coverage counts, drift count and the
    entries it would be asked for. A new feature run or a new tracks entry therefore
    changes it, and so does drift appearing in a feature step's target, through its
    drift count and the drifted entries it is asked for. A step's ``reason`` is left
    out, because a resource's occupancy changes from one second to the next. The plan's
    problems and cautions are left out too. A plan with problems is refused at submit
    whatever its fingerprint, and a caution refuses nothing.

    The document is canonical JSON, with steps sorted by id and entries sorted,
    so one plan has one fingerprint in every process. The plan's step scopes
    enter it only when there are any, which leaves the fingerprint of a plan
    without them unchanged. They are needed while a conversion defers every
    step's entries, when the counts above cannot tell two narrowings apart.
    The request's tracks variant enters it only when there is one, for the
    same reason, and so do the request's pins, as sorted pairs of step id and
    pinned run id.

    A plan asked for a cleanup (:func:`~.cleanup.plan_with_cleanup`) adds the
    policy and the runs it may remove, so a submission removes only the runs
    its caller reviewed. Their sizes and the runs kept are left out. The
    recipes it retires are added only when there are any, so a submission that
    retires nothing fingerprints as it did before retirement existed. A plan not
    asked for a cleanup fingerprints as before.
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
    if plan.tracks_run_id is not None:
        document["tracks_run_id"] = plan.tracks_run_id
    if plan.pinned:
        document["pins"] = [
            [step_id, plan.step(step_id).run_id or ""]
            for step_id in sorted(plan.pinned)
        ]
    if plan.cleanup is not None:
        cleanup_document = cleanup_fingerprint_parts(
            plan.cleanup.policy, plan.cleanup.removable
        )
        if plan.cleanup.retired:
            cleanup_document["retire"] = list(plan.cleanup.retired)
        document["cleanup"] = cleanup_document
    return document_digest(document)


def cleanup_fingerprint_parts(
    policy: CleanupPolicy, removable: Iterable[SupersededRun]
) -> dict[str, JsonValue]:
    """The policy and the removable runs, as a fingerprint of a cleanup holds them.

    The runs are ``[storage, run_id]`` pairs, sorted. A submission's plan and a
    retirement's plan fingerprint their cleanup through this one form.

    Args:
        policy: The age and recency rules the runs were judged under.
        removable: The runs judged removable.

    Returns:
        The ``policy`` and ``remove`` parts of the fingerprinted document.
    """
    pairs = sorted((run.storage, run.run_id) for run in removable)
    remove: list[JsonValue] = [[storage, run_id] for storage, run_id in pairs]
    return {
        "policy": {
            "keep_newest": policy.keep_newest,
            "min_age_hours": policy.min_age_hours,
        },
        "remove": remove,
    }


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
    superseded_sample: int = SUPERSEDED_SAMPLE,
    default_parallel: ParallelSettings | None = None,
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
            _superseded_wire(run) for run in superseded[:superseded_sample]
        ),
        superseded_omitted=max(len(superseded) - superseded_sample, 0),
        parallelizable=planned.parallelizable,
        workers=effective_parallel(
            default_parallel, parallelizable=planned.parallelizable
        ).workers,
        pairs=planned.pairs,
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
