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

**A recipe that validates can still be refused by a dataset.** The plan records
each such problem at the place in the recipe that would change, and
:func:`plan_pipeline` raises them together in :class:`RecipeRefused`:

- A time filter on a dataset that does not record its frame rate, at
  ``run.filter_start_time`` or ``run.filter_end_time``. The step's identity is
  then unresolved.
- A column that a step requires and the tracks tables lack, at ``inputs``. The
  plan reads the column names of the tables that each step reads, from the
  parquet footers, and checks the recipe's columns again with them
  (:func:`~mosaic.core.pipeline.graph.validate.check_recipe`). Validation reads
  only the columns that the schemas allow, and ``ANGLE`` is a caution there on
  any tracks.
  Read off ``mosaic_v1`` tables, which forbid it, it is a problem.
- An op step reading media that requires an analysis transcode and lacks its
  analysis derivative, at ``after``, unless a ``transcode`` step of target
  ``analysis`` is upstream of the step. The plan routes the step's entries as
  the op does, and the message names the transcode step to add.
- A ``convert-tracks`` or ``convert-labels`` step whose source format has no row
  in the raw index it reads, at ``params.source_format``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast

from mosaic.core.helpers import resolve_frame_range
from mosaic.core.media.facts_columns import AnalysisDerivativeMissingError
from mosaic.core.scope import Scope

from ..manifest import tracks_table_columns
from ..inventory.cache import InventoryCache
from ..inventory.model import (
    ArtifactRef,
    ArtifactStatus,
    Coverage,
    Entry,
    FeatureRunRef,
    LabelsVariantRef,
    MediaDerivativeRef,
    MediaVariantRef,
    TracksVariantRef,
    TrainedModelRef,
    reclassify,
)
from ..inventory.scan import reportable_universe
from ..ops import IdentityDeferred, OpIdentity, check_scope_takes
from ..preprocess_layout import PREPROCESS_KIND
from ..resolve import resolve_references
from ..run import resolve_feature_identity
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
from .columns import ColumnSet, measured_columns
from .scope import intended_scope
from .topo import ancestors_of, parents_of, topological_order
from .validate import (
    MEDIA_FIELD,
    RecipeInvalid,
    check_recipe,
    one_run_problems,
    reject_unless_valid,
)
from .wire import Problem

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mosaic_media import MediaProbeError

    from mosaic.core.dataset import Dataset
    from mosaic.core.params import Params

    from ..inventory.model import DatasetInventory
    from .compatibility import DeclarationCatalog

__all__ = [
    "COMPLETE_STATUSES",
    "ArtifactView",
    "CoverageShort",
    "DepsIncomplete",
    "HeldOnParents",
    "IdentityUnresolved",
    "MISSING_SAMPLE",
    "Plan",
    "PlannedStep",
    "Reason",
    "RecipeRefused",
    "Stalled",
    "WaitingOnResource",
    "asked_of",
    "coverage_against",
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
        labels_variant: What names a label conversion step's labels variant,
            empty otherwise.
        labels_kind: The label kind that variant is of, empty otherwise.
        artifact: What to look this step's outputs up by, or ``None`` for a kind
            whose coverage nothing reports.
        coverage: Which of the planned entries already exist. ``covers_all`` is
            what a single-artifact kind -- a trained model, a media derivative --
            says instead, because those are not addressed by entry.
        drift: Entries whose recorded sources have moved underneath the run.
        status: Derived from coverage and consistency, never stored anywhere.
        finished: Whether the artifact records having finished. Beside the status
            rather than folded into it, because "never ran" and "ran and produced
            nothing" both read as ``absent`` and only one of them is a fault a
            consumer must refuse to chain onto.
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
    labels_variant: str = ""
    labels_kind: str = ""
    artifact: ArtifactRef | None = None
    coverage: Coverage[Entry] = field(
        default_factory=lambda: Coverage[Entry](frozenset(), frozenset())
    )
    drift: tuple[Entry, ...] = ()
    status: ArtifactStatus = "absent"
    finished: bool = False
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
        problems: Why this dataset refuses the recipe, in topological order of
            their steps. Empty in a plan that :func:`plan_pipeline` returns
            with ``refuse`` set, because it raises :class:`RecipeRefused`
            instead.
        dataset_uuid: The ``uuid`` of the planned dataset's manifest, empty
            when the manifest records none.
        covers_dataset: Whether the plan was asked for no narrowing, so that it
            covers every entry the dataset can process. A plan naming each of
            those entries does not set it.
    """

    recipe_digest: str
    scope: frozenset[Entry]
    steps: tuple[PlannedStep, ...] = ()
    problems: tuple[Problem, ...] = ()
    dataset_uuid: str = ""
    covers_dataset: bool = False

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


def asked_of(planned: PlannedStep, plan: Plan) -> tuple[Entry, ...]:
    """What this step should be asked to compute.

    The narrowed list where the plan produced one, and the whole scope otherwise.
    A cold step and an all-or-nothing op step both get the whole scope.

    :attr:`Scope.is_unset` decides between the two. Every selector instance is
    truthy, and testing the selector itself keeps the whole-scope answer from
    ever being given.

    An op declaring ``scope_takes = "none"`` is the one step that computes no
    entry, and it returns ``()``. Its unset selector means it covers none of
    them, inverting the rule above. The callers of this function query the
    failure store by entry, and reading the whole plan scope here would
    quarantine a training step over entries it never touched.

    A selector naming groups or sequences is refused. ``plan_pipeline``
    enumerates one against the tracks universe before any step is planned, and
    neither answer below fits it. The unset answer covers every entry in the
    plan and the named one covers none.

    Raises:
        ValueError: ``planned.spec.entries`` names groups or sequences instead
            of entries.
    """
    if planned.kind == "op":
        op_cls = op_class_for_kind(planned.spec.op_kind)
        if op_cls is not None and op_cls.scope_takes == "none":
            return ()
    entries = planned.spec.entries
    if entries.groups is not None or entries.sequences is not None:
        raise ValueError(
            f"asked_of returns the entries a step will compute and does not "
            f"read an index to enumerate them. Step {planned.step_id!r} was "
            f"given {entries!r}, which names groups or sequences. Plan the "
            f"recipe first and pass each step the Scope(entries=[...]) that "
            f"plan_pipeline enumerates."
        )
    if entries.is_unset:
        return tuple(sorted(plan.scope))
    return tuple(sorted(entries.entry_pairs or ()))


class RecipeRefused(RecipeInvalid):
    """A valid recipe that one dataset cannot run, with every problem found.

    A subclass of :class:`~.validate.RecipeInvalid`, because a caller refusing a
    malformed recipe refuses this one in the same way.

    Attributes:
        plan: The plan that found the problems, ``problems`` included. A step
            whose identity a problem leaves unresolved has no ``run_id``.
    """

    lead: ClassVar[str] = "this recipe cannot be run on this dataset"

    def __init__(self, plan: Plan) -> None:
        self.plan: Plan = plan
        super().__init__(plan.problems)


@dataclass
class _Findings:
    """What the walk finds beside the identities: problems, and the tracks it read.

    Attributes:
        problems: The plan-time problems found while walking the steps.
        tracks: The columns of the tracks that each step reads, keyed by step
            id, for the steps whose tables were read.
        read: The columns read for each tracks variant, ``None`` naming the
            variant of each entry. Each is read once per plan.
    """

    problems: list[Problem] = field(default_factory=list)
    tracks: dict[str, ColumnSet] = field(default_factory=dict)
    read: dict[str | None, ColumnSet | None] = field(default_factory=dict)

    def read_tracks(
        self, ds: Dataset, step_id: str, variant: str | None, target: frozenset[Entry]
    ) -> None:
        """Record the columns of the tracks that *step_id* reads, from their tables.

        A variant without tables yet, such as one that an op of this graph
        writes, records nothing, and the step's tracks stay as the schemas
        describe them.
        """
        if variant not in self.read:
            tables = tracks_table_columns(ds, variant, Scope(entries=sorted(target)))
            self.read[variant] = measured_columns(tables.values())
        columns = self.read[variant]
        if columns is not None:
            self.tracks[step_id] = columns


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
    scope: Scope | None = None,
    request: Request | None = None,
    inventory: InventoryCache | None = None,
    quarantined: Iterable[Entry] = (),
    catalog: DeclarationCatalog | None = None,
    recorded: Mapping[str, str] | None = None,
    refuse: bool = True,
) -> Plan:
    """Resolve *recipe* against *ds*. Returns what to run; never submits it.

    Args:
        ds: The dataset. Read only -- planning writes nothing, not even a cache.
        recipe: The graph. Refused before *ds* is touched if it is malformed.
        scope: What to plan over. ``None`` falls back to the request's
            narrowing, and then to everything the dataset can process -- which,
            for a graph that produces its own tracks, is what there is media for.
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
        refuse: Whether a problem this dataset finds with the recipe raises.
            Unset, the problems are recorded on the returned plan. A step that
            runs plans this way and refuses only the problems at its step.

    Returns:
        A :class:`Plan`: one :class:`PlannedStep` per step, in topological order.

    Raises:
        RecipeInvalid: The recipe is malformed. Raised before *ds* is read, and
            reporting every problem found rather than the first.
        RecipeRefused: The recipe validates and this dataset cannot run it, with
            every problem found. Raised only when *refuse* is set.
        ScopeRefused: An op step's scope is one that op's declaration does not
            accept. Raised on the first such step rather than collected,
            because it takes the dataset to enumerate the selector against and
            so cannot be answered alongside the recipe problems above.
    """
    catalog = declaration_catalog() if catalog is None else catalog
    reject_unless_valid(recipe, catalog)

    narrowing = scope
    if narrowing is None and request is not None:
        narrowing = request.scope
    # A converter of raw files takes no scope and writes whatever entries its
    # files hold, which need not be what there is media for, so it does not
    # make the media universe the target the way a tracker does.
    unscoped_writers = frozenset(
        name
        for name, declared in catalog.entries.items()
        if declared.produces.writes_tracks and declared.scope_takes == "none"
    )
    intended = intended_scope(
        ds,
        recipe,
        narrowing,
        produces_tracks=[
            name
            for name, declared in catalog.entries.items()
            if declared.produces.writes_tracks and name not in unscoped_writers
        ],
    )
    target = frozenset(intended.entries) - frozenset(quarantined)
    deferred_by = _deferring_converter(ds, recipe, narrowing, unscoped_writers)

    held = InventoryCache(ds) if inventory is None else inventory
    _ = held.revalidate()
    inv = held.get(scope=Scope(entries=sorted(target)) if target else None)

    bind = request.bind if request is not None else {}
    resolved: dict[str, ResolvedStep] = {}
    planned: dict[str, PlannedStep] = {}
    steps: list[PlannedStep] = []
    findings = _Findings()
    fps = ds.meta_float("fps_default")
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
            deferred_by=deferred_by,
        )
        if isinstance(step, FeatureStepSpec):
            findings.problems.extend(_feature_problems(recipe, step, fps, resolved))
            if TRACKS_INPUT in step.inputs and (
                step.tracks is None or step_plan.spec.tracks_run_id
            ):
                # A tracks reference to a producer whose identity is unresolved
                # does not name a variant, and its tables cannot be read.
                findings.read_tracks(ds, step.id, step_plan.spec.tracks_run_id, target)
        else:
            findings.problems.extend(_conversion_source_problems(ds, step))
            findings.problems.extend(
                _untranscoded_media_problems(
                    ds, recipe, step, step_plan, catalog=catalog, planned=planned
                )
            )
        planned[step.id] = step_plan
        steps.append(step_plan)
        resolved[step.id] = ResolvedStep(
            step_id=step.id,
            storage_name=step_plan.storage_name,
            run_id=step_plan.run_id,
            tracks_variant=step_plan.tracks_variant,
            model_run_id=step_plan.model_run_id,
            labels_variant=step_plan.labels_variant,
            labels_kind=step_plan.labels_kind,
        )

    problems = list(findings.problems)
    if findings.tracks:
        problems.extend(check_recipe(recipe, catalog, tracks=findings.tracks))
    order = {planned_step.step_id: index for index, planned_step in enumerate(steps)}
    plan = Plan(
        recipe_digest=recipe_digest(recipe),
        scope=target,
        steps=tuple(steps),
        problems=tuple(sorted(problems, key=lambda found: order.get(found.step, -1))),
        dataset_uuid=ds.uuid or "",
        covers_dataset=intended.selector.is_unset,
    )
    if refuse and plan.problems:
        raise RecipeRefused(plan)
    return plan


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
    deferred_by: str | None = None,
) -> PlannedStep:
    """Resolve one step against everything the walk has already established.

    *deferred_by* names the conversion step that writes the dataset's first
    tracks, when this graph has one. The entries of a step whose identity
    covers its scope are not known before that step runs, so such a step is
    left unresolved rather than named over no entries.
    """
    declared = catalog.entries[
        step.feature if isinstance(step, FeatureStepSpec) else step.kind
    ]
    parents = parents_of(recipe, step.id)

    inherited = _inherited_deferral(step, planned)
    if (
        inherited is None
        and deferred_by is not None
        and isinstance(step, OpStepSpec)
        and declared.scope_takes != "none"
    ):
        # Before the op is asked for an identity: a scoped op over the empty
        # target is refused by its scope check, or names its run from an index
        # with no rows yet.
        inherited = _awaiting_entries(deferred_by)
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
        if (
            deferred_by is not None
            and resolution.scope_dependent
            and base.run_id is not None
        ):
            # Its identity covers entries that only the conversion will name.
            # A scope-free feature keeps its identity, which the minted tracks
            # variant decides.
            base = replace(
                base, run_id=None, artifact=None, reason=_awaiting_entries(deferred_by)
            )
        base = _as_recorded(base, recorded.get(step.id, ""))

    view = coverage_against(inv, base.artifact, target)
    return replace(
        base,
        parents=parents,
        lane=lane_for(declared),
        spec=replace(
            base.spec,
            entries=_entries_for(step, resolution, target, view.coverage, view.status),
        ),
        coverage=view.coverage,
        status=view.status,
        drift=view.drift,
        finished=view.finished,
        reason=_reason_for(base, parents, planned, view.coverage, view.status),
    )


def _as_recorded(base: PlannedStep, run_id: str) -> PlannedStep:
    """Replace a feature step's resolved identity with the one it actually got.

    Feature steps only, and the asymmetry is not an omission. An op step
    covers its whole scope or none of it, so no op's identity is a function of
    how much of that scope turned out to be there. A ``scope_dependent``
    feature's identity covers the entry set it was fitted over, and *that* can
    differ from what was intended whenever some of the intended entries turn
    out not to be there.
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

    Its scope comes from :func:`_op_scope`. ``_plan_step`` overwrites this
    spec's entries through :func:`_entries_for`, which asks the same helper.
    One helper answers for both sites, and a scope-free op is given no scope
    on either.
    """
    spec = resolve_step_spec(
        recipe,
        step.id,
        resolved,
        bind=bind,
        entries=_op_scope(step.kind, target)
        if isinstance(step, OpStepSpec)
        else Scope(),
    )
    return PlannedStep(
        step_id=step.id,
        kind="op" if isinstance(step, OpStepSpec) else "feature",
        runs=step.kind if isinstance(step, OpStepSpec) else step.feature,
        spec=spec,
        run_id=None,
        storage_name="" if isinstance(step, OpStepSpec) else spec.storage_name,
    )


def _deferring_converter(
    ds: Dataset,
    recipe: Recipe,
    narrowing: Scope | None,
    unscoped_writers: frozenset[str],
) -> str | None:
    """The step that converts this graph's tracks before any exist, or ``None``.

    Set only when the submission names no entries, the dataset holds no tracks
    and the graph converts raw files into some. The entries the graph covers
    are then the ones that conversion writes, which nothing can enumerate
    before it runs.
    """
    if narrowing is not None and narrowing.entry_pairs is not None:
        return None
    converters = [
        step.id
        for step in topological_order(recipe)
        if isinstance(step, OpStepSpec) and step.kind in unscoped_writers
    ]
    if not converters or reportable_universe(ds, None):
        return None
    return converters[0]


def _awaiting_entries(deferred_by: str) -> IdentityUnresolved:
    """Why a step's identity waits on the conversion *deferred_by*."""
    return IdentityUnresolved(
        step=deferred_by,
        because=(
            "the dataset holds no tracks yet, and the entries this step covers "
            f"are the ones step {deferred_by!r} converts"
        ),
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
    """One feature step's identity, from its params plus what is above it.

    A problem that :func:`_feature_problems` finds leaves the identity
    unresolved, and :func:`plan_pipeline` reports it.
    """
    spec = resolve_step_spec(recipe, step.id, resolved, bind=bind)
    shared = _shared_run_problems(recipe, step, resolved)
    if shared:
        # Before the feature is built, because its inputs model refuses a run
        # listed twice.
        return _Resolution(step=_unresolved_feature(step, spec, shared))
    feature = build_step_feature(spec)
    # Pin every unpinned reference before hashing, exactly as execution does. A
    # params reference the recipe wrote literally -- a global fitter's templates,
    # named but not pinned -- is resolved from the dataset here, and a run that
    # hashed it as None would predict one identifier and execute another.
    _ = resolve_references(ds, feature)
    scope_dependent = bool(getattr(feature, "scope_dependent", False))

    fps = ds.meta_float("fps_default")
    unconvertible = _time_filter_problems(step, fps)
    if unconvertible:
        return _Resolution(
            step=_unresolved_feature(step, spec, unconvertible),
            scope_dependent=scope_dependent,
        )

    frame_start, frame_end = resolve_frame_range(
        fps,
        spec.filter_start_frame,
        spec.filter_end_frame,
        spec.filter_start_time,
        spec.filter_end_time,
    )
    run_id, _ = resolve_feature_identity(
        ds,
        feature,
        scope=Scope(entries=sorted(target)),
        # The upstream's **minted** variant where the step names a producer,
        # rather than whatever the index holds. That is what lets an
        # op-to-feature edge resolve before the op has run: the variant payload
        # is params-only, so the identifier is knowable now, and reading it back
        # off an index with no rows in it yet would hash an empty term where
        # execution hashes a real one.
        tracks_run_id=spec.tracks_run_id,
        # The minted labels variant where a labels reference names a
        # conversion step, for the reason the tracks variant is minted.
        labels_run_id=spec.labels_run_id,
        frame_start=frame_start,
        frame_end=frame_end,
        overlap_frames=spec.overlap_frames,
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
        scope_dependent=scope_dependent,
    )


def _unresolved_feature(
    step: FeatureStepSpec, spec: StepSpec, problems: Sequence[Problem]
) -> PlannedStep:
    """The record of a feature step whose identity *problems* leave unresolved."""
    return PlannedStep(
        step_id=step.id,
        kind="feature",
        runs=step.feature,
        spec=spec,
        run_id=None,
        storage_name=spec.storage_name,
        reason=IdentityUnresolved(step=step.id, because=problems[0].message),
    )


def _feature_problems(
    recipe: Recipe,
    step: FeatureStepSpec,
    fps: float | None,
    resolved: Mapping[str, ResolvedStep],
) -> list[Problem]:
    """What this dataset refuses in a feature step, found before its identity."""
    return [
        *_time_filter_problems(step, fps),
        *_shared_run_problems(recipe, step, resolved),
    ]


_RESOLVED_ALIKE: Final = (
    "with params, inputs and run arguments that name one run on this dataset"
)
"""Why two steps that validate as two runs resolve to one, as a frame filter and
the time filter equal to it at the dataset's frame rate do."""


def _shared_run_problems(
    recipe: Recipe, step: FeatureStepSpec, resolved: Mapping[str, ResolvedStep]
) -> list[Problem]:
    """A problem at *step*'s inputs for each run that several of them resolve to."""
    runs = {
        other.id: other.feature if isinstance(other, FeatureStepSpec) else other.kind
        for other in recipe.steps
    }
    return one_run_problems(step, resolved, runs, alike=_RESOLVED_ALIKE)


def _time_filter_problems(step: FeatureStepSpec, fps: float | None) -> list[Problem]:
    """A problem at each time filter of *step*, when the dataset lacks a frame rate.

    The plan converts a time filter to frames with the dataset's ``fps_default``,
    as a run does.
    """
    if fps is not None:
        return []
    message = (
        "a time filter needs the dataset's frame rate, and this dataset does not "
        "record one. Set meta.fps_default in its dataset.yaml, or filter by frame "
        "with run.filter_start_frame and run.filter_end_frame"
    )
    filters = {
        "filter_start_time": step.run.filter_start_time,
        "filter_end_time": step.run.filter_end_time,
    }
    return [
        Problem(step.id, f"run.{name}", message)
        for name, value in filters.items()
        if value is not None
    ]


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
    before its identity is -- which is what ``resample-tracks`` needs, its
    identifier carrying the tracks variant its scope resolves.
    """
    op_cls = op_class_for_kind(step.kind)
    if op_cls is None:  # pragma: no cover - validation resolves every kind first
        raise KeyError(f"no registered op is named {step.kind!r}")
    spec = resolve_step_spec(
        recipe,
        step.id,
        resolved,
        bind=bind,
        entries=_op_scope(step.kind, target),
    )
    params = build_step_op_params(spec)
    # Refused here as well as in ``run_op``, because a plan that cannot be run is
    # worth saying so before anything is enqueued. The scope is the one the step
    # will run under, and the message comes from the same checker the runner
    # uses. A plan and a run refuse the same scope in the same words.
    scope = ds.resolve_scope(spec.entries)
    check_scope_takes(step.kind, op_cls.scope_takes, scope)
    try:
        identity = op_cls().plan_identity(ds, params, scope)
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
            labels_variant=identity.labels_variant,
            labels_kind=str(getattr(params, "kind", ""))
            if identity.labels_variant
            else "",
            artifact=_op_artifact(step.kind, identity, params),
        )
    )


def _op_artifact(kind: str, identity: OpIdentity, params: Params) -> ArtifactRef | None:
    """What to look an op step's output up by, or ``None`` when nothing reports it.

    Each op's coverage is asked of the artifact a downstream step would read,
    which is not always the op's own run: what follows a tracker is its
    ``tracks/`` variant, and what follows a training op is its model. Transcode
    has no run-addressed artifact at all -- its derivatives are named by recipe
    and source -- so it is looked up per target. A media variant is a separate
    run with one file per entry.

    ``None`` is honest rather than a gap: an op whose output nothing inventories
    reads as having no coverage answer, where reporting zero would say it had
    never run and invite it to be run again on every pass.
    """
    if identity.tracks_variant:
        return TracksVariantRef(run_id=identity.tracks_variant)
    if identity.model_run_id:
        return TrainedModelRef(op_kind=kind, run_id=identity.model_run_id)
    if identity.labels_variant:
        # Looked up whole rather than by entry: a labels variant covers the
        # annotated entries, which need not be the plan's target, and a partly
        # labelled dataset would otherwise never read as converted.
        return LabelsVariantRef(
            label_kind=str(getattr(params, "kind", "behavior")),
            run_id=identity.labels_variant,
        )
    if kind == "transcode":
        wanted = getattr(params, "target", "analysis")
        return MediaDerivativeRef(
            target="playback" if wanted == "playback" else "analysis"
        )
    if kind == PREPROCESS_KIND:
        return MediaVariantRef(run_id=identity.run_id)
    return None


_CONVERSION_ROOTS: Final[dict[str, str]] = {
    "convert-tracks": "tracks_raw",
    "convert-labels": "labels_raw",
}
"""The raw root each conversion op reads its format's rows from."""


def _conversion_source_problems(ds: Dataset, step: OpStepSpec) -> list[Problem]:
    """A problem at a conversion step whose format has no raw row to convert.

    A conversion reads the rows a scan recorded, and an unscanned dataset would
    plan cleanly and fail when the step runs. Reads the raw index and writes
    nothing.
    """
    root = _CONVERSION_ROOTS.get(step.kind)
    if root is None:
        return []
    source_format = step.params.get("source_format")
    if not isinstance(source_format, str) or not source_format:
        return []
    if source_format in _raw_formats(ds, root):
        return []
    message = (
        f"{root}/index.csv has no row of source format {source_format!r}; scan "
        f"the dataset's sources first (mosaic scan), or name the format the "
        f"scan recorded"
    )
    return [Problem(step.id, "params.source_format", message)]


def _raw_formats(ds: Dataset, root: str) -> set[str]:
    """The ``src_format`` values that a raw index records, empty when it is absent."""
    import csv

    try:
        path = ds.get_root(root) / "index.csv"
    except KeyError:
        return set()
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(row.get("src_format") or "").strip() for row in csv.DictReader(handle)}


_ANALYSIS_DERIVATIVES: Final = MediaDerivativeRef(target="analysis")
"""The artifact of a ``transcode`` step whose target is ``analysis``."""

_ADD_AN_ANALYSIS_TRANSCODE: Final = (
    'Add the step {"id": "transcode", "type": "op", "kind": "transcode", '
    '"params": {"target": "analysis"}} and "after": ["transcode"] on this step'
)


def _untranscoded_media_problems(
    ds: Dataset,
    recipe: Recipe,
    step: OpStepSpec,
    planned_step: PlannedStep,
    *,
    catalog: DeclarationCatalog,
    planned: Mapping[str, PlannedStep],
) -> list[Problem]:
    """A problem at *step*'s ``after`` for its entries awaiting an analysis transcode.

    An op whose declaration sets ``reads_media``
    (:class:`~.compatibility.ConsumerDecl`) routes each entry through
    :meth:`~mosaic.core.dataset.Dataset.resolve_media_scope`, and an entry whose
    media requires an analysis transcode resolves to its analysis derivative.
    The plan routes the planned entries the same way and reports the entries
    refused with
    :class:`~mosaic.core.media.facts_columns.AnalysisDerivativeMissingError`.
    The run reports every other routing failure.

    The check is skipped for a step that names a media variant in ``media``,
    which it reads in place of the derivative. It is skipped for a step with a
    ``transcode`` step of target ``analysis`` among its ancestors, and for a
    complete step, which does not run.
    """
    if not catalog.entries[step.kind].consumes.reads_media:
        return []
    if step.params.get(MEDIA_FIELD) or planned_step.is_complete:
        return []
    above = ancestors_of(recipe, step.id) - {step.id}
    if any(planned[parent].artifact == _ANALYSIS_DERIVATIVES for parent in above):
        return []
    entries = planned_step.spec.entries.entry_pairs
    if not entries:
        return []
    unrouted: dict[Entry, MediaProbeError] = {}
    try:
        _ = ds.resolve_media_scope(sorted(entries), errors=unrouted)
    except FileNotFoundError:
        return []
    awaiting = sorted(
        entry
        for entry, error in unrouted.items()
        if isinstance(error, AnalysisDerivativeMissingError)
    )
    if not awaiting:
        return []
    named = ", ".join(str(entry) for entry in awaiting[:MISSING_SAMPLE])
    if len(awaiting) > MISSING_SAMPLE:
        named += f" and {len(awaiting) - MISSING_SAMPLE} more"
    counted = (
        "1 entry requires an analysis transcode and has"
        if len(awaiting) == 1
        else f"{len(awaiting)} entries require an analysis transcode and have"
    )
    unordered = _analysis_transcodes(recipe, excluding=step.id)
    fix = (
        f'Add "after": ["{unordered[0]}"] on this step, so that transcode runs first'
        if unordered
        else _ADD_AN_ANALYSIS_TRANSCODE
    )
    message = (
        f"{counted} no analysis derivative, which {step.kind} reads in place of "
        f"the original: {named}. {fix}"
    )
    return [Problem(step.id, "after", message)]


def _analysis_transcodes(recipe: Recipe, *, excluding: str) -> list[str]:
    """The ids of *recipe*'s ``transcode`` steps that make the analysis derivative.

    Read from the recipe rather than the plan, because a step outside the
    checked step's ancestry may not be planned yet. The rule is
    :func:`_op_artifact`'s: every target but ``playback`` makes ``analysis``.
    """
    return sorted(
        other.id
        for other in recipe.steps
        if isinstance(other, OpStepSpec)
        and other.kind == "transcode"
        and other.params.get("target", "analysis") != "playback"
        and other.id != excluding
    )


@dataclass(frozen=True, slots=True)
class ArtifactView:
    """What the dataset holds for one step, measured against that step's target."""

    coverage: Coverage[Entry]
    status: ArtifactStatus
    drift: tuple[Entry, ...] = ()
    finished: bool = False

    @property
    def held(self) -> int:
        """How much of the target this artifact answers for.

        A complete artifact answers for all of it, including the kinds whose
        coverage is not keyed by entry at all -- counting a trained model's
        covered entries would report a finished model as zero, and read every run
        that produced one as having made no progress.
        """
        if self.status in COMPLETE_STATUSES:
            return len(self.coverage.target)
        return len(self.coverage.covered)


def coverage_against(
    inv: DatasetInventory, ref: ArtifactRef | None, target: frozenset[Entry]
) -> ArtifactView:
    """What the dataset already holds for one step, measured against its target.

    The coverage is rebuilt against the *plan's* target rather than reported as
    the inventory measured it, because those are two questions: an inventory
    measures a run against the dataset's universe, and a plan measures it against
    the entries this submission asked for.

    A kind not addressed by entry -- a trained model, a media derivative --
    reports ``covers_all`` instead, which is what a single artifact answering for
    everything means. Reading its entry coverage would report a complete model as
    zero of ninety.

    A media variant is keyed by entry and camera, and an entry is covered when
    the camera that the variant was made from has a row and a file. The op writes
    only an entry's consumed camera, and the camera is dropped here. A camera that
    stops being the consumed one changes the entry's media composition, and the
    variant then reports that entry as drifted when both compositions are known.
    """
    record = inv.record(ref) if ref is not None else None
    if record is None:
        return ArtifactView(
            coverage=Coverage[Entry](target=target, present=frozenset()),
            status="absent",
        )
    if isinstance(ref, FeatureRunRef | TracksVariantRef):
        coverage = Coverage[Entry](
            target=target,
            present=cast("frozenset[Entry]", record.coverage.present),
            covers_all=record.coverage.covers_all,
        )
    elif isinstance(ref, MediaVariantRef):
        coverage = Coverage[Entry](
            target=target,
            present=frozenset(
                (group, sequence) for group, sequence, _ in inv.coverage(ref).present
            ),
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
    return ArtifactView(
        coverage=coverage,
        status=reclassify(record, coverage),
        drift=record.drift,
        finished=bool(record.finished_at),
    )


def _op_scope(kind: str, target: frozenset[Entry]) -> Scope:
    """What an op step of *kind* is asked to cover.

    An op that takes no scope is given none, whatever the graph is narrowed to.
    A training step reads a prepared directory, and the plan's entries name a
    coverage it cannot act on. Handing it one is refused by
    :func:`~mosaic.core.pipeline.ops.check_scope_takes`, in the planner and
    again in the runner.

    Every other op step is all-or-nothing. An op that reads a scope reads the
    whole of it, and the target therefore goes through unnarrowed by what is
    already covered.
    """
    op_cls = op_class_for_kind(kind)
    if op_cls is not None and op_cls.scope_takes == "none":
        return Scope()
    return _named(sorted(target))


def _entries_for(
    step: Step,
    resolution: _Resolution,
    target: frozenset[Entry],
    coverage: Coverage[Entry],
    status: ArtifactStatus,
) -> Scope:
    """What this step should be asked to compute, already narrowed.

    A step must not re-request its whole scope, so a scope-free feature is asked
    for the remainder. **A ``scope_dependent`` one is asked for all of it**, and
    that is not an oversight: its identity *is* its scope, so running it over the
    remainder would produce a different run from the one that was planned -- a
    fit over 31 sequences under the name of a fit over 120.

    An op step is all-or-nothing for the same reason turned around, and one that
    takes no scope is asked for nothing at all. :func:`_op_scope` decides both.

    The completeness branch is not the same test as an empty remainder, and the
    case that separates them is a global fit: one artifact answers for every
    entry, so it is complete over a scope naming an entry it never saw, and its
    *missing* set names that entry all the same. Asking it to compute the entry
    would ask it to be a different artifact.
    """
    if isinstance(step, OpStepSpec):
        return _op_scope(step.kind, target)
    if resolution.scope_dependent:
        return _named(sorted(target))
    if status in COMPLETE_STATUSES:
        return Scope()
    return _named(sorted(coverage.missing))


def _named(entries: list[Entry]) -> Scope:
    """*entries* as a selector, or an unset one where the list is empty.

    An empty narrowing has always meant "everything in scope" here, and the
    caller reads it through :attr:`Scope.is_unset`. Naming no entry explicitly
    is what a step covering nothing would say, and no step says it.
    """
    return Scope(entries=entries) if entries else Scope()


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
