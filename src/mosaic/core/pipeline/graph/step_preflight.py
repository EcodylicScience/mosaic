"""What a step checks before it does any work, and how it says no.

A step that is about to compute something is the last party in a position to
notice that what it is about to read is not what was intended. Nothing is
re-planning the graph while it runs -- that is the price of deciding the whole
thing at submit -- so the observation moves to the party that cares, which is the
consumer, at its own start.

**Coverage counts cannot detect wrongness.** A hundred and twenty parquets of NaN
is a hundred and twenty of a hundred and twenty, and no count will ever notice.
That is the single weakness of deriving status from the record, and it is why
these checks are here: each is a predicate that already exists somewhere, run
where a refusal is still free. A general check registry, and any check that reads
the data itself, wait for a third caller.

A refusal is an ordinary failure carrying a reason, not a new terminal status;
:mod:`mosaic.core.pipeline.refusal` says why.

**``allow_partial`` answers exactly one of these.** A shortfall is a question
about *how much*, and a person may decide to proceed over less. A digest that
does not match, a version that moved, a variant that disagrees, an upstream
that produced nothing and a ``device`` that names a card the step was not
leased are not questions about how much, and no flag here unlocks them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.core.scope import Scope

from ..manifest import refuse_mixed_track_schemas
from ..refusal import Refusal, RefusalReason
from ..gpu_lease import leased_card, names_another_card
from ..track_space import (
    TablesRead,
    TrackSpaceRefused,
    refuse_unsupported_track_space,
    upstream_track_spaces,
    variant_track_spaces,
)
from ..track_universe import run_carries_z
from ..tracks_index import AmbiguousTracksVariant
from .model import (
    BOUND_STATUS,
    COMPLETE_STATUSES,
    TRACKS_INPUT,
    FeatureStepSpec,
    Recipe,
)
from .plan import (
    MISSING_SAMPLE,
    Plan,
    PlannedStep,
    RecipeRefused,
    step_target,
)
from .resolve import build_step_feature
from .topo import topological_order

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue
    from mosaic.core.schema import TrackSpace

    from .compatibility import DeclarationCatalog
    from .wire import Problem


__all__ = [
    "CoverageShortfall",
    "RecipeRefusedAtStep",
    "StepRefused",
    "preflight",
    "refuse_mixed_schemas",
    "track_space_refusals",
]


class StepRefused(Refusal, RuntimeError):
    """A step declined to run, and said why.

    Always names its step, unlike a :class:`Refusal` raised below one.
    """

    def __init__(
        self,
        reason: RefusalReason,
        step_id: str,
        message: str,
        detail: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message, reason=reason, step_id=step_id, detail=detail)


class CoverageShortfall(StepRefused):
    """A step cannot be run over everything it was planned for.

    Raised rather than proceeded through, because for a ``scope_dependent`` step
    it is a scientific question rather than a maintenance one: a model fitted on
    89 sequences is a different model from one fitted on 90, and mosaic says so
    by giving it a different name. Proceeding is a decision, and
    ``allow_partial`` is where that decision is recorded.
    """

    def __init__(self, step_id: str, covered: int, target: int, detail: str) -> None:
        self.covered: int = covered
        self.target: int = target
        super().__init__(
            "coverage_shortfall",
            step_id,
            f"step {step_id!r} would run over {covered} of {target} entries: "
            f"{detail}. Its identity covers the set it was fitted on, so this is "
            f"a different run from the one planned. Pass allow_partial to proceed "
            f"deliberately, or complete the steps above it first.",
            {"covered": covered, "target": target, "detail": detail},
        )


class RecipeRefusedAtStep(StepRefused):
    """A step that this dataset refuses when the step re-plans itself.

    The dataset can change after submit, as when an op of the graph writes the
    tracks that a later step reads. A problem found then refuses the step it
    names, and its attempt records the problems as a submission's refusal lists
    them.

    Attributes:
        plan: The step's plan, holding only the problems at this step.
    """

    def __init__(self, step_id: str, plan: Plan) -> None:
        self.plan: Plan = plan
        problems: list[JsonValue] = [problem.as_json() for problem in plan.problems]
        message = str(RecipeRefused(plan))
        super().__init__("recipe_refused", step_id, message, {"problems": problems})

    @property
    def problems(self) -> tuple[Problem, ...]:
        """The problems at this step, in the order the plan found them."""
        return self.plan.problems


def preflight(
    ds: Dataset, plan: Plan, step_id: str, *, allow_partial: bool = False
) -> None:
    """Refuse *step_id* if what it is about to read is not what was intended.

    Args:
        ds: The dataset, for the checks that read the tracks index.
        plan: This step's graph, resolved against *ds* moments ago.
        step_id: The step about to run.
        allow_partial: Whether a shortfall has been answered.

    Raises:
        StepRefused: Carrying the reason and the numbers behind it.
    """
    planned = plan.step(step_id)
    _refuse_card_device(planned)
    _refuse_empty_upstream(plan, planned)
    _refuse_variant_disagreement(ds, plan, planned)
    if not allow_partial:
        refuse_shortfall(plan, planned)


def refuse_mixed_schemas(
    ds: Dataset,
    step_id: str,
    *,
    recipe: Recipe | None = None,
    tracks_run_id: str | None = None,
    scope: Scope | None = None,
    only: str | None = None,
    bound: Collection[str] = (),
) -> None:
    """Refuse tracks tables a recipe would read that it cannot read as one.

    ``trex_v1``'s spatial columns are centimetres and its ``X`` is a head
    position, so mixing one with a ``mosaic_v1`` table is mixing units and
    landmarks. The predicate is the loader's own, called here so the refusal
    arrives before the work rather than partway through it.

    **An unpinned reader is asked of the whole dataset.** A feature step that
    reads the tracks with neither a ``tracks`` reference nor *tracks_run_id*
    hashes every entry's variant, whatever its scope: the term a feature
    identifier carries for the tables it reads is scope-free, so that a
    scope-free feature gets one identifier for every scope. Over such a
    reader both checks read every entry. A mixture of families raises
    ``schema_family_mismatch`` here instead of an exception out of the middle
    of a hash, and an entry of two variants raises ``variant_ambiguous``,
    because no identity of the reader is defined. Without such a reader, or
    given a variant, the family check reads *tracks_run_id* over *scope* alone.

    **Two variants of one entry are not a mixture.** A step that reads no
    tracks, such as a tracker, or one that names its variant, has no question
    to answer, so a tracker sweep of one family passes.

    Args:
        ds: The dataset whose tracks index to read.
        step_id: The step that refuses, empty for a submission. An empty one
            names the first unpinned reader in a ``variant_ambiguous`` refusal.
        recipe: The recipe whose feature steps read the tracks. ``None`` checks
            the families of the whole dataset and no variant.
        tracks_run_id: The request's variant, ``None`` for none.
        scope: What the request covers, ``None`` for everything.
        only: Ask about this step's reading alone, as a step does at its start.
        bound: The steps a request binds or pins to a run. Neither is
            executed, so neither reads tracks.

    Raises:
        StepRefused: ``schema_family_mismatch`` or ``variant_ambiguous``.
    """
    readers = (
        _unpinned_readers(recipe, only, bound)
        if recipe is not None and tracks_run_id is None
        else ()
    )
    try:
        if readers:
            refuse_mixed_track_schemas(ds)
        else:
            refuse_mixed_track_schemas(ds, tracks_run_id=tracks_run_id, scope=scope)
    except ValueError as exc:
        raise StepRefused("schema_family_mismatch", step_id, str(exc)) from exc
    if not readers:
        return
    from ..tracks_index import read_tracks_index, select_variant_rows

    try:
        _ = select_variant_rows(read_tracks_index(ds), None)
    except AmbiguousTracksVariant as ambiguous:
        raise _variant_ambiguous(step_id or readers[0], readers, ambiguous) from None


def track_space_refusals(
    ds: Dataset,
    recipe: Recipe,
    planned: Mapping[str, PlannedStep],
    *,
    catalog: DeclarationCatalog,
    targets: Mapping[str, frozenset[Entry]],
    exempt: Collection[str],
) -> dict[str, TrackSpaceRefused]:
    """The feature steps that would read tracks in a space they do not compute in.

    The plan-time half of the check ``run_feature`` makes: one refusal and one
    message for both. A step's spaces are those of the tracks tables it selects,
    through each recorded schema's space, and for each step it reads through
    ``inputs`` as tracks, ``world_m`` when its stored run carries ``Z`` and the
    spaces of the tracks behind that run. Each variant is read once per target,
    however many steps reach it.

    **What is not on disk yet is skipped**, and the step's own plan at its start
    checks it: a variant an op of the same recipe mints has no rows, and a parent
    without a run id or whose run holds no output has no footer. A schema this
    process does not register says nothing. An entry holding two variants under
    an unpinned reader records nothing, because the step's identity reports it.

    Args:
        ds: The dataset to read the tracks index and the stored runs of.
        recipe: The recipe planned.
        planned: Each step's plan, keyed by step id.
        catalog: The declarations of the step types.
        targets: Each step's target entries, keyed by step id.
        exempt: The steps a request binds or pins to a run. Neither is executed,
            so neither reads anything.

    Returns:
        The refusal of each refused step, keyed by step id, in topological order.
    """
    refusals: dict[str, TrackSpaceRefused] = {}
    read: TablesRead = {}
    for step in topological_order(recipe):
        if not isinstance(step, FeatureStepSpec) or step.id in exempt:
            continue
        declared = catalog.entries[step.feature]
        if not declared.track_spaces:
            continue
        target = targets[step.id]
        spaces = _tracks_spaces(ds, step, planned[step.id], target, read)
        if declared.consumes.requires_track_shape:
            spaces |= _stored_input_spaces(ds, step, planned, target, read)
        try:
            refuse_unsupported_track_space(
                frozenset(declared.track_spaces), step.feature, spaces
            )
        except TrackSpaceRefused as refused:
            refusals[step.id] = refused
    return refusals


def _tracks_spaces(
    ds: Dataset,
    step: FeatureStepSpec,
    planned: PlannedStep,
    target: frozenset[Entry],
    read: TablesRead,
) -> frozenset[TrackSpace]:
    """The spaces of the tracks tables *step* selects, empty when it reads none.

    *read* holds the spaces already read for a variant over a target, so steps
    sharing both read the tracks index once.
    """
    variant = planned.spec.tracks_run_id
    if TRACKS_INPUT not in step.inputs or (step.tracks is not None and not variant):
        # A tracks reference to a producer whose identity is unresolved names no
        # variant, and its tables cannot be read.
        return frozenset()
    return variant_track_spaces(ds, variant, target, read)


def _stored_input_spaces(
    ds: Dataset,
    step: FeatureStepSpec,
    planned: Mapping[str, PlannedStep],
    target: frozenset[Entry],
    read: TablesRead,
) -> frozenset[TrackSpace]:
    """The spaces of the stored runs that *step* reads through ``inputs`` as tracks.

    ``world_m`` for a run whose output carries ``Z``, and the spaces of the tracks
    behind each run (:func:`~..track_space.upstream_track_spaces`). A parent
    without a run is skipped.
    """
    spaces: set[TrackSpace] = set()
    for item in step.inputs:
        if item == TRACKS_INPUT:
            continue
        parent = planned[item.step]
        if not parent.run_id:
            continue
        if run_carries_z(ds, parent.storage_name, parent.run_id):
            spaces.add("world_m")
        spaces |= upstream_track_spaces(
            ds, parent.storage_name, parent.run_id, target, read
        )
    return frozenset(spaces)


def _unpinned_readers(
    recipe: Recipe, only: str | None, bound: Collection[str]
) -> tuple[str, ...]:
    """The executed feature steps of *recipe* reading the tracks with no variant."""
    return tuple(
        step.id
        for step in recipe.steps
        if isinstance(step, FeatureStepSpec)
        and TRACKS_INPUT in step.inputs
        and step.tracks is None
        and step.id not in bound
        and (only is None or step.id == only)
    )


def _variant_ambiguous(
    step_id: str, readers: Sequence[str], ambiguous: AmbiguousTracksVariant
) -> StepRefused:
    group, sequence = ambiguous.entry
    listing = ", ".join(repr(variant) for variant in ambiguous.variants)
    steps = ", ".join(repr(reader) for reader in readers)
    named = f"step {steps} reads" if len(readers) == 1 else f"steps {steps} read"
    return StepRefused(
        "variant_ambiguous",
        step_id,
        f"{named} the tracks without naming a variant, and the "
        f"dataset holds {len(ambiguous.variants)} of (group={group!r}, "
        f"sequence={sequence!r}): {listing}. Name one for the request with "
        f"tracks_run_id (--tracks-run-id), or pin the step to the step that "
        f'produces it with "tracks": {{"step": <id>}}.',
        {
            "entry": [group, sequence],
            "variants": list(ambiguous.variants),
            "steps": list(readers),
        },
    )


def refuse_shortfall(plan: Plan, planned: PlannedStep) -> None:
    """Refuse a step whose inputs do not cover what it was planned over.

    Only for a ``scope_dependent`` step, and the asymmetry is the point: a
    scope-free step running over 89 of 90 entries produces 89 correct outputs
    under the identifier they belong to, and the ninetieth arrives later under
    the same one. A scope-dependent step running over 89 produces *one* artifact
    that is not the artifact anyone asked for, under a name saying it is.

    Measured against the step's own target (:func:`~.plan.step_target`). A
    parent read through ``inputs`` or ``tracks`` must hold every entry of it. A
    parent read otherwise -- through a params reference such as templates or a
    label map, or by ``after`` alone (:attr:`~.plan.PlannedStep.status_parents`)
    -- answers for the whole of what it was planned over, and must only be
    complete: a test-scoped step reads a model fitted on the training entries.
    """
    if planned.kind != "feature":
        return
    if not build_step_feature(planned.spec).scope_dependent:
        return
    target = step_target(planned, plan)
    by_status = frozenset(planned.status_parents)
    stacked = planned.stacked_parents
    lacking: dict[str, frozenset[Entry]] = {}
    held: dict[str, int] = {}
    if stacked:
        found = _lacking_stack(plan, stacked, target)
        if found is not None:
            lacking[" and ".join(stacked)] = found
            held[" and ".join(stacked)] = sum(
                len(plan.step(parent).coverage.covered) for parent in stacked
            )
    for parent in planned.parents:
        if parent in stacked:
            continue
        found = _lacking(plan, parent, target, entry_by_entry=parent not in by_status)
        if found is not None:
            lacking[parent] = found
            held[parent] = len(plan.step(parent).coverage.covered)
    if not lacking:
        return
    covered = min(held.values(), default=0)
    missing = sorted(str(entry) for entries in lacking.values() for entry in entries)
    raise CoverageShortfall(
        planned.step_id,
        covered,
        len(target),
        "missing " + ", ".join(missing[:MISSING_SAMPLE]),
    )


def _lacking_stack(
    plan: Plan, parents: tuple[str, ...], target: frozenset[Entry]
) -> frozenset[Entry] | None:
    """What the inputs of a stacking step lack of *target*, taken together.

    Each entry is read from one input, so the inputs answer for *target* as a
    union: complete, and planned over every entry of it between them, or else
    the entries of it that none of them holds.
    """
    steps = [plan.step(parent) for parent in parents]
    if all(_answers_for_target(step) for step in steps) and target <= frozenset(
        entry for step in steps for entry in step.coverage.target
    ):
        return None
    outside = target - frozenset(
        entry for step in steps for entry in step.coverage.covered
    )
    return outside or None


def _lacking(
    plan: Plan, step_id: str, target: frozenset[Entry], *, entry_by_entry: bool
) -> frozenset[Entry] | None:
    """What a parent lacks of what its reader is planned over, ``None`` for nothing.

    A complete parent lacks nothing it was planned over. Read entry by entry, it
    must also have been planned over the whole of *target*, which a step scope
    can make wider than the parent's own target; the entries it lacks are then
    those of *target* it does not hold. A parent read otherwise lacks only what
    its own coverage misses. A parent bound to a run computes nothing, so read
    entry by entry it is complete only where that run holds its target.
    """
    try:
        parent = plan.step(step_id)
    except KeyError:  # pragma: no cover - parents come from the same recipe
        return None
    held = parent.coverage.covered
    complete = parent.status in COMPLETE_STATUSES
    if entry_by_entry:
        if _answers_for_target(parent) and target <= parent.coverage.target:
            return None
        outside = target - held
        return outside or None
    if complete or not (target - held):
        return None
    return parent.coverage.missing or (target - held)


def _answers_for_target(step: PlannedStep) -> bool:
    """Whether *step* holds every entry it was planned over, read entry by entry.

    A complete step does. A step bound to a run is complete without computing,
    so it does only where the pinned run covers its target.
    """
    if step.status == BOUND_STATUS:
        return step.coverage.covers_all or not step.coverage.missing
    return step.status in COMPLETE_STATUSES


def _refuse_card_device(planned: PlannedStep) -> None:
    """Refuse a leased step whose ``device`` names a card other than its own.

    A step that holds a card sees only that card, so its ``device`` index counts
    within it and ``0`` names it. Another index, or a list, would reach a card the
    step was not leased or fail inside the tool. A wrong card is not a shortfall,
    so no flag answers it.
    """
    card = leased_card()
    device = planned.spec.params.get("device")
    if not card or not names_another_card(device):
        return
    raise StepRefused(
        "gpu_card_not_leased",
        planned.step_id,
        f"step {planned.step_id!r} was leased GPU {card} alone, but its device is "
        f"{device!r}. Under a lease, device 0 names that card. Set device to 0, "
        f"'cuda' or leave it unset.",
        {"card": card, "device": device},
    )


def _refuse_empty_upstream(plan: Plan, planned: PlannedStep) -> None:
    """Refuse a step above which something ran and wrote nothing.

    A step whose manifest resolves empty finishes cleanly with zero outputs, and
    a consumer would happily chain onto the empty directory. Never-ran and
    ran-and-produced-nothing both read as ``absent``, so the finish is what tells
    them apart, and it is the one wrongness a count *can* see.
    """
    for parent_id in planned.parents:
        try:
            parent = plan.step(parent_id)
        except KeyError:  # pragma: no cover - parents come from the same recipe
            continue
        if not parent.finished or parent.coverage.covers_all:
            continue
        if parent.coverage.target and not parent.coverage.covered:
            raise StepRefused(
                "upstream_empty",
                planned.step_id,
                f"step {parent_id!r} finished having written no entries, so "
                f"step {planned.step_id!r} would read an empty artifact. Check "
                f"what {parent_id!r} was asked to compute before running this.",
                {"parent": parent_id, "target": len(parent.coverage.target)},
            )


def _refuse_variant_disagreement(ds: Dataset, plan: Plan, planned: PlannedStep) -> None:
    """Refuse when an op parent produced a tracks variant other than the pinned one.

    The variant an op step will write is *minted* from the recipe's settings at
    planning time, because a params-only payload is knowable before the op runs.
    That makes it a prediction, and this is where the prediction meets the index
    the op actually wrote. Reading it back is a verification, never the mechanism
    by which the edge resolves -- an index with no rows in it yet would otherwise
    hash an empty term where execution hashes a real one.
    """
    if planned.spec.tracks_run_id is None:
        return
    from ..tracks_index import read_tracks_index, variant_for_producer_run

    for parent_id in planned.parents:
        try:
            parent = plan.step(parent_id)
        except KeyError:  # pragma: no cover - parents come from the same recipe
            continue
        if parent.kind != "op" or not parent.tracks_variant or not parent.run_id:
            continue
        produced = variant_for_producer_run(read_tracks_index(ds), parent.run_id)
        if produced is None or produced == parent.tracks_variant:
            # ``None`` is not a disagreement: the op has written no rows this
            # step can check against, which the coverage check already reports.
            continue
        raise StepRefused(
            "variant_mismatch",
            planned.step_id,
            f"step {planned.step_id!r} is pinned to tracks variant "
            f"{parent.tracks_variant!r}, but run {parent.run_id!r} of step "
            f"{parent_id!r} produced {produced!r}. Re-plan rather than reading "
            f"tables under a name that does not describe them.",
            {
                "parent": parent_id,
                "pinned": parent.tracks_variant,
                "produced": produced,
                "producer_run_id": parent.run_id,
            },
        )
