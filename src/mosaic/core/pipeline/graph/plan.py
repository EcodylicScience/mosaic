"""What a recipe would do to a dataset: the identities, and what is already done.

**No step is left unresolved, and nothing waits on execution.** Step A's identity
is a function of its params; step B's is a function of its params plus A's
identity; step C's of B's. A topological walk is the whole mechanism, and it
closes because every term of an identifier is either in the recipe or on disk
before the graph runs: a feature-to-feature edge reads nothing at all, a tracks
variant is *minted* from the recipe's settings rather than read back from tables
an op has not written yet, and the entry set a ``scope_dependent`` step will hash
comes from :func:`~mosaic.core.pipeline.graph.scope.intended_scope`.

**A resolved ``run_id`` is never load-bearing at execution.** It drives the
preview, the cost estimate, validation and the decision to enqueue. It never
skips a step and never enters a downstream job's payload -- every step resolves
its own identity at its own start, and that is what makes prediction safe rather
than authoritative. A wrong resolution makes the preview wrong and the next call
corrects it; it cannot make a step read the wrong artifact. Do not optimize this
away by trusting a submitted identity.

**Granularity, said out loud because the opposite reading is natural and
expensive: failure records are per entry; jobs are per step with an entry list.**
One job per entry would multiply the multi-second feature-library import by the
entry count.

**Identity is a pure function of the recipe and the dataset, and the inventory is
read rather than handed over a wire.** A serialized inventory invites a stale or
fabricated snapshot, which is the one way this could produce a confidently wrong
answer. An injectable holder exists so a long-lived process can keep one between
calls, and so a test can build the world it is asserting about.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, Literal, cast

from mosaic.core.helpers import resolve_frame_range

from .._utils import Scope
from ..inventory.cache import InventoryCache
from ..inventory.model import (
    ArtifactRef,
    ArtifactStatus,
    Coverage,
    Entry,
    FeatureRunRef,
    MediaDerivativeRef,
    TracksVariantRef,
    TrainedModelRef,
    classify,
)
from ..manifest import tracks_variants_for
from ..ops import IdentityDeferred, OpIdentity
from ..resolve import resolve_references
from ..run import compute_run_id, resolve_labels_variants
from ..sequence_index import read_entry_compositions
from .digest import recipe_digest
from .lanes import lane_for
from .model import (
    TRACKS_INPUT,
    BoundRef,
    FeatureStepSpec,
    OpStepSpec,
    Recipe,
    Request,
    Step,
    StepRef,
    params_step_refs,
)
from .resolve import (
    ResolvedStep,
    StepSpec,
    build_step_feature,
    build_step_op_params,
    declaration_catalog,
    op_class_for_kind,
    resolve_step_spec,
)
from .scope import intended_scope
from .topo import parents_of, topological_order
from .validate import reject_unless_valid

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.types import Feature, Params

    from ..inventory.model import DatasetInventory
    from .compatibility import DeclarationCatalog

__all__ = [
    "COMPLETE_STATUSES",
    "CoverageShort",
    "DepsIncomplete",
    "HeldOnParents",
    "IdentityUnresolved",
    "MISSING_SAMPLE",
    "Plan",
    "PlannedStep",
    "Reason",
    "Stalled",
    "WaitingOnResource",
    "is_stalled",
    "plan_pipeline",
]

MISSING_SAMPLE: Final = 5
"""How many missing entries a shortfall names.

A shortfall is acted on by a person deciding whether to proceed, and a list of
several hundred entry names is not read. Naming a few and counting the rest is
what makes the count mean something.
"""

COMPLETE_STATUSES: Final[frozenset[ArtifactStatus]] = frozenset(
    {"complete", "complete-but-drifted"}
)
"""The statuses that mean there is nothing left to compute.

``complete-but-drifted`` is here deliberately: a drifted run is complete and
loadable, and superseded is not invalid. It is surfaced as drift so a caller can
choose to re-run it, and refusing to call it complete would make every dataset
whose sources have moved read as unfinished work.
"""


# --- why a step is not simply running ------------------------------------------


@dataclass(frozen=True, slots=True)
class DepsIncomplete:
    """Something this step reads does not hold everything this step wants."""

    blocking: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoverageShort:
    """This step holds some of its scope and not the rest."""

    covered: int
    target: int
    missing: tuple[Entry, ...] = ()
    """A sample of what is absent, at most :data:`MISSING_SAMPLE` entries."""


@dataclass(frozen=True, slots=True)
class HeldOnParents:
    """Its parents are running, or waiting to. Set by whatever schedules work."""

    parents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WaitingOnResource:
    """Its lane's resource class is at capacity. Set by whatever schedules work."""

    resource_class: str
    in_use: int
    capacity: int


@dataclass(frozen=True, slots=True)
class Stalled:
    """Its last attempt succeeded and produced nothing new. See :func:`is_stalled`."""

    covered: int
    target: int
    missing: tuple[Entry, ...] = ()


@dataclass(frozen=True, slots=True)
class IdentityUnresolved:
    """This step, or something above it, cannot say what it will be called yet.

    What an op's ``IdentityDeferred`` becomes once it is reported rather than
    raised -- a training op whose data directory is written by an earlier step in
    the same graph has nothing to fingerprint until that step runs. It blocks the
    *preview*, never execution: the step resolves its own identity at its own
    start, where the directory does exist.

    Named apart from that exception on purpose. Two things called
    ``IdentityDeferred`` in one package, one raised and one reported, would have
    to be disambiguated at every call site handling both -- and this module is
    exactly such a site.

    Attributes:
        step: The step whose identity is unresolvable, which for a cascade is the
            one above rather than the one reporting.
        because: What is missing, in the words of whatever could not resolve it.
    """

    step: str
    because: str


type Reason = (
    DepsIncomplete
    | CoverageShort
    | HeldOnParents
    | WaitingOnResource
    | Stalled
    | IdentityUnresolved
)
"""Why a step is not simply running, as a flat tagged union with fields.

Flat rather than nested: a nested tree would be a three-repository wire contract
needing a golden fixture to keep from rotting, for a nesting case that does not
exist. Every state a view renders carries one of these, because "waiting" with no
explanation is what makes people mark work done by hand.
"""


def is_stalled(
    *, exited_clean: bool, terminal: bool, covered_before: int, covered_after: int
) -> bool:
    """Did the last attempt succeed without producing anything new?

    **A distinct state, and not a failure.** A feature legitimately producing
    fewer outputs than inputs never reads as complete and would otherwise be
    resubmitted forever; bounding that with an attempt counter renders a
    *correct* pipeline red and invites a retry that cannot succeed. So the
    condition is stated by what happened rather than by how many times: a
    terminal attempt, a clean exit, and coverage that did not move.

    Stop after **one** stall rather than after K, report ``covered/target`` with
    the missing sample, and require the same explicit gesture a coverage
    shortfall does. Defined here and enforced by whatever runs the plan, which is
    what keeps it from becoming a retry loop under a different name.
    """
    return terminal and exited_clean and covered_after <= covered_before


# --- the plan ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannedStep:
    """One step of a recipe, resolved against one dataset.

    Attributes:
        step_id: The step's id in the recipe.
        kind: Whether it runs a feature or an op.
        runs: The feature slug or op kind it runs.
        spec: Exactly what it takes to run it, entries already narrowed.
        lane: Which queue this work is offered to.
        parents: The steps it references, ``after`` included.
        run_id: What the run will be called, or ``None`` when neither it nor
            something above it can say yet.
        storage_name: Where a feature step's outputs live.
        tracks_variant: What an op step's tables will be named by, empty
            otherwise. **A different identifier from** ``run_id``.
        model_run_id: What names a training step's model, empty otherwise.
        artifact: What to look this step's outputs up by, or ``None`` for a kind
            whose coverage nothing reports.
        coverage: Which of the planned entries already exist. ``covers_all`` is
            what a single-artifact kind -- a trained model, a media derivative --
            says instead, because those are not addressed by entry.
        drift: Entries whose recorded sources have moved underneath the run.
        status: Derived from coverage and consistency, never stored anywhere.
        reason: Why this step is not simply running, or ``None``.
    """

    step_id: str
    kind: Literal["feature", "op"]
    runs: str
    spec: StepSpec
    lane: str = ""
    parents: tuple[str, ...] = ()
    run_id: str | None = None
    storage_name: str = ""
    tracks_variant: str = ""
    model_run_id: str = ""
    artifact: ArtifactRef | None = None
    coverage: Coverage[Entry] = field(
        default_factory=lambda: Coverage[Entry](frozenset(), frozenset())
    )
    drift: tuple[Entry, ...] = ()
    status: ArtifactStatus = "absent"
    reason: Reason | None = None

    @property
    def is_complete(self) -> bool:
        """Is there nothing left for this step to compute?"""
        return self.status in COMPLETE_STATUSES


@dataclass(frozen=True, slots=True)
class Plan:
    """What a recipe would do to a dataset, resolved and not submitted.

    Attributes:
        recipe_digest: What names the recipe this plan is of.
        scope: The entries the graph is planned over.
        steps: Every step, in topological order.
    """

    recipe_digest: str
    scope: frozenset[Entry]
    steps: tuple[PlannedStep, ...] = ()

    def step(self, step_id: str) -> PlannedStep:
        """The planned step with this id, or ``KeyError``."""
        for planned in self.steps:
            if planned.step_id == step_id:
                return planned
        raise KeyError(f"no step {step_id!r} in this plan")

    @property
    def pending(self) -> tuple[PlannedStep, ...]:
        """Every step with work left, in topological order."""
        return tuple(planned for planned in self.steps if not planned.is_complete)

    @property
    def is_complete(self) -> bool:
        """Has every step of this graph been computed over this scope?"""
        return all(planned.is_complete for planned in self.steps)

    @property
    def run_ids(self) -> dict[str, str | None]:
        """What each step will be called, keyed by step id."""
        return {planned.step_id: planned.run_id for planned in self.steps}


@dataclass(frozen=True, slots=True)
class _Resolution:
    """One step's identity, plus the one fact narrowing needs beyond it."""

    step: PlannedStep
    scope_dependent: bool = False
    """Whether this step's identity covers the set of entries in scope.

    Carried out of resolution rather than asked again later, because answering it
    means building the feature, and the resolution has one in hand.
    """


def plan_pipeline(
    ds: Dataset,
    recipe: Recipe,
    *,
    intended_entries: Iterable[Entry] | None = None,
    request: Request | None = None,
    inventory: InventoryCache | None = None,
    quarantined: Iterable[Entry] = (),
    catalog: DeclarationCatalog | None = None,
    recorded: Mapping[str, str] | None = None,
) -> Plan:
    """Resolve *recipe* against *ds*. Returns what to run; never submits it.

    Args:
        ds: The dataset. Read only -- planning writes nothing, not even a cache.
        recipe: The graph. Refused before *ds* is touched if it is malformed.
        intended_entries: The entries to plan over. ``None`` falls back to the
            request's narrowing, and then to everything the dataset can process
            -- which, for a graph that produces its own tracks, is what there is
            media for.
        request: One submission's choices: its narrowing, and the out-of-graph
            pins in ``bind``, which resolve a step outright and are never
            overridden by anything derived here.
        inventory: A held inventory to read coverage from, revalidated first. One
            is built when none is given.
        quarantined: Entries to leave out. They narrow the *intended scope*, so a
            ``scope_dependent`` step's identity moves when entries are
            quarantined -- which is correct, and is the point: a model fitted on
            89 sequences is a different model from one fitted on 90.
        catalog: The declarations to resolve against. Built from the registries
            when not given.
        recorded: What a step's run was actually called, for the steps that have
            already run. **A resolved identifier is a prediction and a recorded
            one is a fact**, so where the two differ the fact governs -- for that
            step and for everything hashing it. It is how a caller executing a
            graph keeps the rest of it addressable after a step legitimately
            landed somewhere else, which is what a ``scope_dependent`` step does
            whenever fewer entries complete than were intended.

    Returns:
        A :class:`Plan`: one :class:`PlannedStep` per step, in topological order.

    Raises:
        RecipeInvalid: The recipe is malformed. Raised before *ds* is read.
    """
    catalog = declaration_catalog() if catalog is None else catalog
    reject_unless_valid(recipe, catalog)

    narrowing = intended_entries
    if narrowing is None and request is not None:
        narrowing = request.entries
    target = intended_scope(
        ds,
        recipe,
        narrowing,
        produces_tracks=[
            name
            for name, declared in catalog.entries.items()
            if declared.produces.writes_tracks
        ],
    ) - frozenset(quarantined)

    held = InventoryCache(ds) if inventory is None else inventory
    _ = held.revalidate()
    inv = held.get(entries=target or None)

    bind = request.bind if request is not None else {}
    resolved: dict[str, ResolvedStep] = {}
    planned: dict[str, PlannedStep] = {}
    steps: list[PlannedStep] = []
    for step in topological_order(recipe):
        step_plan = _plan_step(
            ds,
            recipe,
            step,
            catalog=catalog,
            inv=inv,
            target=target,
            bind=bind,
            resolved=resolved,
            planned=planned,
            recorded=recorded or {},
        )
        planned[step.id] = step_plan
        steps.append(step_plan)
        resolved[step.id] = ResolvedStep(
            step_id=step.id,
            storage_name=step_plan.storage_name,
            run_id=step_plan.run_id,
            tracks_variant=step_plan.tracks_variant,
            model_run_id=step_plan.model_run_id,
        )

    return Plan(recipe_digest=recipe_digest(recipe), scope=target, steps=tuple(steps))


def _plan_step(
    ds: Dataset,
    recipe: Recipe,
    step: Step,
    *,
    catalog: DeclarationCatalog,
    inv: DatasetInventory,
    target: frozenset[Entry],
    bind: Mapping[str, BoundRef],
    resolved: Mapping[str, ResolvedStep],
    planned: Mapping[str, PlannedStep],
    recorded: Mapping[str, str],
) -> PlannedStep:
    """Resolve one step against everything the walk has already established."""
    declared = catalog.entries[
        step.feature if isinstance(step, FeatureStepSpec) else step.kind
    ]
    parents = parents_of(recipe, step.id)

    inherited = _inherited_deferral(step, planned)
    if inherited is not None:
        # Short-circuited rather than resolved and then overwritten, and the
        # difference is not tidiness. A step whose input identity is unknown has
        # an unknown identity; asking it anyway means asking an op to resolve a
        # model reference that is still the empty string, or a feature to pin an
        # input from whatever the dataset happens to hold -- the first raises out
        # of a path that has nothing to do with this graph, and the second
        # silently answers a question nobody asked.
        resolution = _Resolution(step=_unresolved(step, target, recipe, resolved, bind))
        base = replace(resolution.step, reason=inherited)
    elif isinstance(step, OpStepSpec):
        resolution = _resolve_op_step(
            ds, recipe, step, target=target, bind=bind, resolved=resolved
        )
        base = resolution.step
    else:
        resolution = _resolve_feature_step(
            ds, recipe, step, target=target, bind=bind, resolved=resolved
        )
        base = resolution.step
        base = _as_recorded(base, recorded.get(step.id, ""))

    coverage, status, drift = _coverage_of(inv, base.artifact, target)
    return replace(
        base,
        parents=parents,
        lane=lane_for(declared),
        spec=replace(
            base.spec,
            entries=_entries_for(step, resolution, target, coverage, status),
        ),
        coverage=coverage,
        status=status,
        drift=drift,
        reason=_reason_for(base, parents, planned, coverage, status),
    )


def _as_recorded(base: PlannedStep, run_id: str) -> PlannedStep:
    """Replace a feature step's resolved identity with the one it actually got.

    Feature steps only, and the asymmetry is not an omission. An op's identity is
    a function of its params -- its settings, the recipe it encodes, the recorded
    identities of the videos it reads -- and the plan passes execution those same
    params, so the two cannot differ. A ``scope_dependent`` feature's identity
    covers the entry set it was fitted over, and *that* can differ from what was
    intended whenever some of the intended entries turn out not to be there.
    """
    if not run_id or base.kind != "feature" or run_id == base.run_id:
        return base
    return replace(
        base,
        run_id=run_id,
        artifact=FeatureRunRef(name=base.storage_name, run_id=run_id),
    )


def _unresolved(
    step: Step,
    target: frozenset[Entry],
    recipe: Recipe,
    resolved: Mapping[str, ResolvedStep],
    bind: Mapping[str, BoundRef],
) -> PlannedStep:
    """The record for a step whose identity nothing above it can supply.

    It still carries a spec, because what to run is known even when what the run
    will be called is not -- and a caller executing the graph reads the spec,
    resolving the identity itself at the moment it has what the plan lacked.
    """
    spec = resolve_step_spec(
        recipe,
        step.id,
        resolved,
        bind=bind,
        entries=tuple(sorted(target)) if isinstance(step, OpStepSpec) else (),
    )
    return PlannedStep(
        step_id=step.id,
        kind="op" if isinstance(step, OpStepSpec) else "feature",
        runs=step.kind if isinstance(step, OpStepSpec) else step.feature,
        spec=spec,
        run_id=None,
        storage_name="" if isinstance(step, OpStepSpec) else spec.storage_name,
    )


def _inherited_deferral(
    step: Step, planned: Mapping[str, PlannedStep]
) -> IdentityUnresolved | None:
    """Whether an unresolved identity above this step reaches its own.

    Only the references that *enter* an identity cascade -- inputs, the tracks
    variant, a params reference. An ``after`` edge carries no term in any payload,
    which is exactly why it exists and exactly why an unresolved identity above
    one leaves this step's own knowable.
    """
    for parent in _data_parents(step):
        above = planned.get(parent)
        if above is not None and above.run_id is None:
            because = (
                above.reason.because
                if isinstance(above.reason, IdentityUnresolved)
                else f"{parent!r} could not be resolved"
            )
            return IdentityUnresolved(step=parent, because=because)
    return None


def _data_parents(step: Step) -> tuple[str, ...]:
    """The steps whose identity reaches this one's, ``after`` excluded."""
    found: list[str] = []
    if isinstance(step, FeatureStepSpec):
        for item in step.inputs:
            if isinstance(item, StepRef) and item.step not in found:
                found.append(item.step)
        if step.tracks is not None and step.tracks.step not in found:
            found.append(step.tracks.step)
    for reference in params_step_refs(step.params).values():
        if reference.step not in found:
            found.append(reference.step)
    return tuple(found)


def _resolve_feature_step(
    ds: Dataset,
    recipe: Recipe,
    step: FeatureStepSpec,
    *,
    target: frozenset[Entry],
    bind: Mapping[str, BoundRef],
    resolved: Mapping[str, ResolvedStep],
) -> _Resolution:
    """One feature step's identity, from its params plus what is above it."""
    spec = resolve_step_spec(recipe, step.id, resolved, bind=bind)
    feature = build_step_feature(spec)
    # Pin every unpinned reference before hashing, exactly as execution does. A
    # params reference the recipe wrote literally -- a global fitter's templates,
    # named but not pinned -- is resolved from the dataset here, and a run that
    # hashed it as None would predict one identifier and execute another.
    _ = resolve_references(ds, feature)

    frame_start, frame_end = resolve_frame_range(
        ds.meta_float("fps_default"),
        spec.filter_start_frame,
        spec.filter_end_frame,
        spec.filter_start_time,
        spec.filter_end_time,
    )
    scope = Scope(
        entries=set(target),
        frame_start=frame_start,
        frame_end=frame_end,
        tracks_variants=_tracks_variants(ds, feature, spec),
        labels_variants=resolve_labels_variants(ds, feature),
        # Read from the source roots, which exist before any of this graph runs,
        # so the term is exact rather than predicted. A feature declaring no
        # roots -- which is every ``scope_dependent`` feature today -- reduces it
        # to the entry names alone.
        compositions=read_entry_compositions(ds, target),
    )
    run_id, _ = compute_run_id(
        feature, frame_start, frame_end, scope, overlap_frames=spec.overlap_frames
    )
    return _Resolution(
        step=PlannedStep(
            step_id=step.id,
            kind="feature",
            runs=step.feature,
            spec=spec,
            run_id=run_id,
            storage_name=spec.storage_name,
            artifact=FeatureRunRef(name=spec.storage_name, run_id=run_id),
        ),
        scope_dependent=bool(getattr(feature, "scope_dependent", False)),
    )


def _tracks_variants(ds: Dataset, feature: Feature, spec: StepSpec) -> tuple[str, ...]:
    """The ``_tracks`` term: which tracks recipes this step will read.

    Set **only** by a ``tracks`` input, which is the rule the manifest applies: a
    feature reading another feature's output inherits no variant, because its
    upstream already hashed the one it read.

    A step whose ``tracks`` reference names a producing step takes that step's
    **minted** variant rather than whatever the index holds. That is what lets an
    op-to-feature edge resolve before the op has run: the variant payload is
    params-only, so the identifier is knowable now, and reading it back off an
    index with no rows in it yet would hash an empty term where execution hashes
    a real one.
    """
    if not any(item == TRACKS_INPUT for item in feature.inputs.root):
        return ()
    if spec.tracks_run_id:
        return (spec.tracks_run_id,)
    return tracks_variants_for(ds, spec.tracks_run_id)


def _resolve_op_step(
    ds: Dataset,
    recipe: Recipe,
    step: OpStepSpec,
    *,
    target: frozenset[Entry],
    bind: Mapping[str, BoundRef],
    resolved: Mapping[str, ResolvedStep],
) -> _Resolution:
    """One op step's identity, asked of the op rather than reconstructed.

    An op step covers its whole scope or none of it, so its entries are settled
    before its identity is -- which is what a transcode needs, its identifier
    covering the recorded identities of the videos it will read.
    """
    spec = resolve_step_spec(
        recipe, step.id, resolved, bind=bind, entries=tuple(sorted(target))
    )
    op_cls = op_class_for_kind(step.kind)
    if op_cls is None:  # pragma: no cover - validation resolves every kind first
        raise KeyError(f"no registered op is named {step.kind!r}")
    params = build_step_op_params(spec)
    try:
        identity = op_cls().plan_identity(ds, params)
    except IdentityDeferred as exc:
        return _Resolution(
            step=PlannedStep(
                step_id=step.id,
                kind="op",
                runs=step.kind,
                spec=spec,
                run_id=None,
                reason=IdentityUnresolved(step=step.id, because=exc.because),
            )
        )
    return _Resolution(
        step=PlannedStep(
            step_id=step.id,
            kind="op",
            runs=step.kind,
            spec=spec,
            run_id=identity.run_id,
            tracks_variant=identity.tracks_variant,
            model_run_id=identity.model_run_id,
            artifact=_op_artifact(step.kind, identity, params),
        )
    )


def _op_artifact(kind: str, identity: OpIdentity, params: Params) -> ArtifactRef | None:
    """What to look an op step's output up by, or ``None`` when nothing reports it.

    Each op's coverage is asked of the artifact a downstream step would read,
    which is not always the op's own run: what follows a tracker is its
    ``tracks/`` variant, and what follows a training op is its model. Transcode
    has no run-addressed artifact at all -- its derivatives are named by recipe
    and source -- so it is looked up per target.

    ``None`` is honest rather than a gap: an op whose output nothing inventories
    reads as having no coverage answer, where reporting zero would say it had
    never run and invite it to be run again on every pass.
    """
    if identity.tracks_variant:
        return TracksVariantRef(run_id=identity.tracks_variant)
    if identity.model_run_id:
        return TrainedModelRef(op_kind=kind, run_id=identity.model_run_id)
    if kind == "transcode":
        wanted = getattr(params, "target", "analysis")
        return MediaDerivativeRef(
            target="playback" if wanted == "playback" else "analysis"
        )
    return None


def _coverage_of(
    inv: DatasetInventory, ref: ArtifactRef | None, target: frozenset[Entry]
) -> tuple[Coverage[Entry], ArtifactStatus, tuple[Entry, ...]]:
    """What the dataset already holds for one step, measured against its target.

    The coverage is rebuilt against the *plan's* target rather than reported as
    the inventory measured it, because those are two questions: an inventory
    measures a run against the dataset's universe, and a plan measures it against
    the entries this submission asked for.

    A kind not addressed by entry -- a trained model, a media derivative --
    reports ``covers_all`` instead, which is what a single artifact answering for
    everything means. Reading its entry coverage would report a complete model as
    zero of ninety.
    """
    record = inv.record(ref) if ref is not None else None
    if record is None:
        return Coverage[Entry](target=target, present=frozenset()), "absent", ()
    if isinstance(ref, FeatureRunRef | TracksVariantRef):
        coverage = Coverage[Entry](
            target=target,
            present=cast("frozenset[Entry]", record.coverage.present),
            covers_all=record.coverage.covers_all,
        )
    else:
        coverage = Coverage[Entry](
            target=target, present=frozenset(), covers_all=record.coverage.is_satisfied
        )
    # Re-derived rather than taken from the record, and the two are not the same
    # answer: the record's status is measured against the *dataset's* universe,
    # and this one against the entries the submission asked for. A run holding
    # every entry the dataset can process reads complete to an inventory and is
    # genuinely short of a scope naming one more -- and taking the record's word
    # for it would report a step as done while its coverage said two of three.
    status = classify(
        satisfied=coverage.is_satisfied,
        any_covered=bool(coverage.covered) or coverage.covers_all,
        orphan_rows=bool(record.orphan_rows),
        orphan_files=bool(record.orphan_files),
        drifted=bool(record.drift),
        finished=bool(record.finished_at),
    )
    return coverage, status, record.drift


def _entries_for(
    step: Step,
    resolution: _Resolution,
    target: frozenset[Entry],
    coverage: Coverage[Entry],
    status: ArtifactStatus,
) -> tuple[Entry, ...]:
    """What this step should be asked to compute, already narrowed.

    A step must not re-request its whole scope, so a scope-free feature is asked
    for the remainder. **A ``scope_dependent`` one is asked for all of it**, and
    that is not an oversight: its identity *is* its scope, so running it over the
    remainder would produce a different run from the one that was planned -- a
    fit over 31 sequences under the name of a fit over 120.

    An op step is all-or-nothing for the same reason turned around: its scope
    lives in its own params, and the ops that read one read the whole of it.
    """
    if isinstance(step, OpStepSpec) or resolution.scope_dependent:
        return tuple(sorted(target))
    if status in COMPLETE_STATUSES:
        return ()
    return tuple(sorted(coverage.missing))


def _reason_for(
    base: PlannedStep,
    parents: tuple[str, ...],
    planned: Mapping[str, PlannedStep],
    coverage: Coverage[Entry],
    status: ArtifactStatus,
) -> Reason | None:
    """Why this step is not simply running.

    Precedence, and each level answers a different question. An unresolved
    identity comes first, because nothing below it means anything. A step that is
    already complete has no reason at all, whatever is above it. Otherwise the
    graph comes before the artifact: a step whose parents are short cannot be
    expected to hold what they have not produced.
    """
    if base.reason is not None:
        return base.reason
    if status in COMPLETE_STATUSES:
        return None
    blocking = tuple(
        parent
        for parent in parents
        if parent in planned and not planned[parent].is_complete
    )
    if blocking:
        return DepsIncomplete(blocking=blocking)
    if coverage.covered:
        missing = tuple(sorted(coverage.missing))
        return CoverageShort(
            covered=len(coverage.covered),
            target=len(coverage.target),
            missing=missing[:MISSING_SAMPLE],
        )
    return None
