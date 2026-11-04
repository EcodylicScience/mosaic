"""What a recipe file is, and what a request beside it holds.

Two documents, kept apart on purpose.

The **recipe** is the pipeline and nothing else: which steps, with which params,
wired to which other steps. It is portable across datasets, so it never holds a
resolved ``run_id`` -- those are dataset state -- and it never holds a
submission's choices, so two people running the same analysis over different
subsets are running the same recipe.

The **request** is one submission of one recipe against one dataset. It holds
what the recipe must not: which entries, whether a shortfall may proceed, who
asked, and the out-of-graph pins, each of which replaces a step with the run it
names, so the step is never executed. It is defined here rather than where it is
written because ``plan_pipeline`` takes ``request=`` and resolves each bound step
as its pinned run; writing the file and minting the execution ids into it belong
to the phase that executes.

**One place per fact.** Cross-step references live in the step body at the exact
site they will be substituted, so there is no separate ``edges`` array to drift
from the bodies -- :func:`~mosaic.core.pipeline.graph.topo.edges` is a derived
read-only view for a canvas. The one explicit list is ``after``, which is
ordering-only and corresponds to nothing in the payload.

**No pandas, no numpy, no feature registry.** Parsing a recipe has to be cheap
enough that a release gate and a status endpoint can do it on every tick, so this
module declares its own strict base rather than importing the one that lives
beside the loaders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, ClassVar, Final, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.label_controls import LabelMask, LabelRemap
from mosaic.core.scope import Scope

from .expand import expand_repeats

__all__ = [
    "BOUND_STATUS",
    "COMPLETE_STATUSES",
    "REQUEST_HOLD_FLOOR_HOURS",
    "BoundRef",
    "CleanupMode",
    "CleanupPolicy",
    "CleanupSpec",
    "CleanupTarget",
    "ComputedRun",
    "FeatureStepSpec",
    "GraphModel",
    "OpStepSpec",
    "Recipe",
    "Request",
    "SCHEMA_VERSION",
    "Step",
    "StepRef",
    "StepRun",
    "StepStatus",
    "TRACKS_INPUT",
    "canonical_number",
    "canonical_params",
    "params_step_refs",
    "references_of",
    "refuse_pins_beside_apply",
]

SCHEMA_VERSION: Final = 1
"""The recipe format this module reads and writes.

A file declaring a *newer* version is refused rather than read under the wrong
rules, the same call ``core.manifest`` makes. There is no older version yet.
"""

TRACKS_INPUT: Final = "tracks"
"""The literal naming the dataset's standardized tracks as a feature input.

Deliberately not a ``StepRef``: ``tracks`` is a dataset root rather than a step,
and a graph may read it without any step having produced it.
"""


_EXACT_INTEGER_BOUND: Final = 2**53
"""Below this magnitude every integer is exactly a float, and back.

RFC 8785 writes a number by its value, so ``5.0`` and ``5`` are one number. A
float at or past the bound may not name the integer it prints as, so it is left
a float rather than turned into a different number.
"""


def canonical_number(value: JsonValue) -> JsonValue:
    """*value* with every whole-number float written as an integer.

    A step's ``params`` are free JSON, so ``5.0`` and ``5`` would otherwise stay
    two values and name two recipes. Which one a recipe holds depends on who
    wrote it: a hand-written file keeps ``5.0``, and a client that serializes
    through JavaScript sends ``5``. This follows the number rule of RFC 8785:
    a finite float with an integral value below 2**53 becomes that integer, and
    ``-0.0`` becomes ``0``. A ``bool`` is left as it is, NaN and the
    infinities are left as they are, and lists and mappings are walked.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if (
            math.isfinite(value)
            and value.is_integer()
            and abs(value) < _EXACT_INTEGER_BOUND
        ):
            return int(value)
        return value
    if isinstance(value, list):
        return [canonical_number(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical_number(item) for key, item in value.items()}
    return value


def canonical_params(params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """A step's *params* with every whole-number float written as an integer.

    Applied when a recipe is parsed, so the plan, the saved file, the digest and
    the step that runs all see one value (:func:`canonical_number`).
    """
    return {key: canonical_number(value) for key, value in params.items()}


type StepParams = Annotated[dict[str, JsonValue], AfterValidator(canonical_params)]
"""A step's params, normalized as they are parsed."""


class GraphModel(BaseModel):
    """Strict base for every model in this package.

    ``extra="forbid"`` throughout, because a recipe is a document a human writes
    by hand and a typo silently ignored is a step that does not do what its
    author read.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class StepRef(GraphModel):
    """A reference to another step in this graph, at the site it substitutes.

    ``pattern`` is meaningful only where the reference lands in a params field
    typed ``ArtifactSpec``, whose ``_derive_pattern`` defaults to ``*.parquet``
    and would otherwise resolve the wrong file out of a producer that writes more
    than one -- ``extract-templates`` being the documented case. Empty everywhere
    else, and validated against the target field's type by ``validate``.
    """

    step: str
    pattern: str = ""

    @model_validator(mode="after")
    def _step_is_named(self) -> Self:
        if not self.step.strip():
            raise ValueError("a step reference must name a step")
        return self


class ParamsStepRef(StepRef):
    """A step reference in a params field, which a labels field may give controls.

    ``remap`` and ``mask`` are the controls of the labels source the reference
    becomes. They are meaningful only at a labels field, and ``validate`` refuses
    them anywhere else. A step's ``inputs`` and ``tracks`` are plain
    :class:`StepRef`, which accepts neither.
    """

    remap: LabelRemap = None
    mask: LabelMask = None


class StepRun(GraphModel):
    """The run arguments that reach a feature's identity, and only those.

    Typed rather than a free ``dict`` of run kwargs, because the distinction is
    load-bearing: ``compute_run_id`` covers the frame range and the overlap
    width, and covers nothing else a caller might put beside them. A throughput
    knob (worker counts, parallel mode) changes runtime rather than output, so it
    is a property of the machine a step lands on and not of the recipe;
    ``overwrite`` is refused outright, because it mutates content under a stable
    address and a concurrent downstream reader would get a mixed read its own
    ``run_id`` records nothing about.

    ``extra="forbid"`` is what turns each of those from a convention into a
    parse error.
    """

    filter_start_frame: int | None = None
    filter_end_frame: int | None = None
    filter_start_time: float | None = None
    filter_end_time: float | None = None
    overlap_frames: int = 0

    @model_validator(mode="after")
    def _one_filter_per_boundary(self) -> Self:
        """Each boundary is set as a frame or as a time, as a run reads it."""
        boundaries = (
            ("start", self.filter_start_frame, self.filter_start_time),
            ("end", self.filter_end_frame, self.filter_end_time),
        )
        for boundary, frame, time in boundaries:
            if frame is not None and time is not None:
                msg = f"set one of filter_{boundary}_frame and filter_{boundary}_time"
                raise ValueError(msg)
        return self


class _StepBase(GraphModel):
    """What every step carries, whatever it runs."""

    id: str
    after: list[str] = Field(default_factory=list)
    """Ordering-only edges: this step runs after those, with no data reference.

    ``transcode -> trex`` is the case -- the tracker's identity has no term for
    the media it read, so nothing in the payload records the dependency. No rule
    restricts where one may go. An edge enters no identity, so a step whose
    identity should cover what it reads takes a reference instead.
    """

    @model_validator(mode="after")
    def _id_is_named(self) -> Self:
        if not self.id.strip():
            raise ValueError("a step must have a non-empty id")
        return self


class FeatureStepSpec(_StepBase):
    """One feature run: a slug, its inputs, its params, its run arguments."""

    type: Literal["feature"] = "feature"
    feature: str
    inputs: list[Literal["tracks"] | StepRef] = Field(
        default_factory=lambda: [TRACKS_INPUT]
    )
    tracks: StepRef | None = None
    """Which tracks variant to read, as the step that produces it.

    Substituted to ``--tracks-run-id <variant>``. The substitution target is the
    **tracks variant**, which is a different identifier from the producing op's
    run id; they coincide today only because the tracker variant payload is an
    unwrapped passthrough of the settings.
    """
    params: StepParams = Field(default_factory=dict)
    run: StepRun = Field(default_factory=StepRun)


class OpStepSpec(_StepBase):
    """One op run: a kind and its settings.

    A step names no scope. Which entries it covers is the plan's answer, not the
    recipe's, and it is given to the op as an argument. One recipe therefore
    runs over several datasets.
    """

    type: Literal["op"] = "op"
    kind: str
    params: StepParams = Field(default_factory=dict)


type Step = Annotated[FeatureStepSpec | OpStepSpec, Field(discriminator="type")]


class Recipe(GraphModel):
    """A pipeline as a file: steps and the references between them.

    Structural validation only -- unique ids and references that resolve.
    Whether a slug exists, whether its params validate, whether the graph is
    acyclic and whether two steps may be connected are semantic questions that
    need the registries, and they live in ``validate``.
    """

    schema_version: int = SCHEMA_VERSION
    name: str = ""
    steps: list[Step] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _expand_repeats(cls, data: object) -> object:
        """Expand each ``repeat`` item, so the recipe holds its steps in full.

        Every reader parses through here, so the canonical form, the digest and
        the saved file are those of the expanded steps (``expand.py``).
        """
        return expand_repeats(data)

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        if self.schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"recipe schema_version {self.schema_version} is newer than this "
                f"mosaic understands ({SCHEMA_VERSION}); upgrade mosaic to read it"
            )
        known: set[str] = set()
        for step in self.steps:
            if step.id in known:
                raise ValueError(f"duplicate step id: {step.id!r}")
            known.add(step.id)
        for step in self.steps:
            for referenced, where in references_of(step):
                if referenced not in known:
                    raise ValueError(
                        f"step {step.id!r} references unknown step "
                        f"{referenced!r} at {where}. Declared: {sorted(known)}"
                    )
                if referenced == step.id:
                    raise ValueError(f"step {step.id!r} references itself at {where}")
        return self

    def step(self, step_id: str) -> Step:
        """The step with this id, or ``KeyError``."""
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"no step {step_id!r} in this recipe")

    @property
    def ids(self) -> tuple[str, ...]:
        """Every step id, in declaration order."""
        return tuple(step.id for step in self.steps)


def references_of(step: Step) -> list[tuple[str, str]]:
    """Every ``(referenced step id, where)`` one step names, ``after`` included.

    One walk, used by validation here and by the topology module, so a new
    reference site cannot be added to the model and forgotten by the ordering.
    """
    found: list[tuple[str, str]] = []
    if isinstance(step, FeatureStepSpec):
        for position, item in enumerate(step.inputs):
            if isinstance(item, StepRef):
                found.append((item.step, f"inputs[{position}]"))
        if step.tracks is not None:
            found.append((step.tracks.step, "tracks"))
    for field_name, ref in params_step_refs(step.params).items():
        found.append((ref.step, f"params.{field_name}"))
    for position, after in enumerate(step.after):
        found.append((after, f"after[{position}]"))
    return found


def params_step_refs(params: Mapping[str, JsonValue]) -> dict[str, ParamsStepRef]:
    """The top-level params fields holding a step reference.

    A params reference is a mapping carrying a ``step`` key, because ``params``
    is typed open -- an op passes arbitrary settings through to an external tool,
    and a feature's params are validated by its own model rather than by this
    one. Detecting the shape here means the recipe format needs no second
    declaration of which fields may be references, and no author-supplied label
    to mark one.

    **Top-level only**, matching ``resolve_references`` and
    ``Params.identity_dump()``. The dump asks a nested model for its own dump, and
    no reference is found below the top level. A reference buried inside a nested
    dict would be substituted by neither, so finding one here would promise
    something the resolver does not do.

    Raises:
        ValueError: A mapping carries ``step`` alongside a key ``ParamsStepRef``
            does not declare. Refused rather than ignored: it is a reference its
            author expected to be honoured.
    """
    found: dict[str, ParamsStepRef] = {}
    for name, value in params.items():
        if isinstance(value, dict) and "step" in value:
            try:
                found[name] = ParamsStepRef.model_validate(value)
            except Exception as exc:
                raise ValueError(
                    f"params.{name} looks like a step reference but is not a "
                    f"valid one: {exc}"
                ) from exc
    return found


class BoundRef(GraphModel):
    """An artifact this graph did not produce, pinned by the submission.

    What makes "run t-SNE over the speed feature I computed last month"
    expressible: bind the speed step to that run. A bound step is the run it
    names and is never executed. Its ``run_id`` is the pinned one, its storage is
    ``feature``, which must be the step's own output storage, and the steps below
    it read that run. It lives on the request rather than in the recipe, so the
    recipe stays portable and free of resolved identifiers while the binding is
    recorded where it is auditable.

    **It pins only.** Params are never bound, or the digest would stop describing
    the work it names.
    """

    feature: str
    run_id: str


BOUND_STATUS: Final = "bound"
"""The status of a step that a submission binds to a run.

The step is the pinned run: it is never executed, so it computes nothing, and
the steps below it read that run. It counts as complete
(:data:`COMPLETE_STATUSES`). A plan word, never an inventory status.
"""

type StepStatus = Literal[
    "absent", "partial", "complete", "complete-but-drifted", "inconsistent", "bound"
]
"""What a planned step's artifact adds up to (an ``ArtifactStatus``), or
:data:`BOUND_STATUS`.

Spelled flat rather than as a union with ``ArtifactStatus``, which a JSON Schema
renders as an ``anyOf`` that the ``plan_recipe`` output schema has no room for. A
test asserts it is ``ARTIFACT_STATUSES`` and ``bound``. Defined here, where
:class:`BoundRef` is, because a status read and the MCP tool models declare
fields over it and load no pandas.
"""


COMPLETE_STATUSES: Final[frozenset[StepStatus]] = frozenset(
    {"complete", "complete-but-drifted", BOUND_STATUS}
)
"""The statuses that mean there is nothing left to compute.

``bound`` is a step that the submission binds to a run: it is that run, and is
never executed.

``complete-but-drifted`` is here deliberately: a drifted run is complete and
loadable, and superseded is not invalid. It is surfaced as drift so a caller can
choose to re-run it, and refusing to call it complete would make every dataset
whose sources have moved read as unfinished work.
"""


REQUEST_HOLD_FLOOR_HOURS: Final = 1.0
"""The fewest hours a request that has ended holds the runs it recorded or bound.

A cleanup holds them for ``max(min_age_hours, this)``, so a policy of
``min_age_hours: 0`` cannot remove the output of a request that just finished.
"""

type CleanupMode = Literal["none", "superseded"]
"""Whether a plan reports, and a submission removes, the superseded runs."""


class CleanupPolicy(GraphModel):
    """Which superseded feature runs a cleanup keeps by age and by recency.

    Fixed by whoever runs the cleanup: the operator of an MCP server, or the
    person running the CLI. It is never a tool argument.

    Attributes:
        keep_newest: How many superseded runs of each storage to keep, the
            newest by ``finished_at`` first.
        min_age_hours: The youngest a run may be and be removed, in hours.
    """

    keep_newest: int = Field(default=1, ge=0)
    min_age_hours: float = Field(default=24.0, ge=0)


class CleanupTarget(GraphModel):
    """One feature run a submission approved for removal."""

    storage: str
    run_id: str


class CleanupSpec(GraphModel):
    """The cleanup a request runs after its steps, as it was approved at submit.

    Attributes:
        mode: Which runs it removes.
        policy: The policy the plan was computed under.
        execution_id: The cleanup's attempt, whose run-log records each run it
            removed or kept.
        approved: The runs the reviewed plan found removable. The cleanup
            removes no other run, and keeps any of these that is no longer
            removable when it runs.
        fingerprint: The fingerprint of the plan the submission reviewed.
        retired: The digests of the saved recipes this submission retired: the
            runs their requests wrote were judged as candidates, wherever they
            sit. It applies to this submission alone. Left out of the request
            file when empty, because an older mosaic refuses a key it does not
            know.
    """

    mode: Literal["superseded"] = "superseded"
    policy: CleanupPolicy = Field(default_factory=lambda: CleanupPolicy())
    execution_id: str
    approved: list[CleanupTarget] = Field(default_factory=list)
    fingerprint: str = ""
    retired: list[str] = Field(default_factory=list)


class ComputedRun(GraphModel):
    """One run that a request's plan had left to compute when it was submitted.

    A submission records its runs beside its request file
    (:func:`~.store.save_computes`), and a later submission compares its own
    with them, whatever its recipe, to find a request that computes the same
    run.

    Attributes:
        step: The step of the request that computes it.
        storage: Where a feature run's outputs live, empty for an op.
        run_id: The run's identifier at submit.
        whole: Whether the step computes the run as one piece, whatever its
            entries: a feature that fits or names its run by its entries, or an
            op that takes no scope. Two requests computing such a run duplicate
            it over any entries. Two computing any other run duplicate it only
            over entries they share.
    """

    step: str
    storage: str
    run_id: str
    whole: bool = False


def refuse_pins_beside_apply(
    pins: Mapping[str, str] | None, applies_request_id: str | None
) -> None:
    """Refuse pins given beside an ``applies_request_id``.

    An apply derives its pins from the request it names, so a call names one
    source of pins. Empty pins are none.

    Raises:
        ValueError: *pins* holds a pin and *applies_request_id* is given.
    """
    if pins and applies_request_id is not None:
        raise ValueError(
            "pins and applies_request_id are both given; an apply derives its "
            "pins from the request it names"
        )


class Request(GraphModel):
    """One submission of one recipe against one dataset.

    Holds no ``run_id`` of its own beyond the explicit ``bind`` and ``pins``: the
    execution ids are *assigned* before anything runs, which is what makes this
    document complete at submit rather than filled in as work lands.

    ``step_executions`` is the load-bearing part. A step that re-resolved its
    input by *feature name* at its own start would fall through to the
    latest-run rule, which is wall clock -- so two requests on one dataset
    running the same feature with different params would cross-bind, the second
    step of one picking up the other's output because its index row landed a
    second later. Naming the parent's attempt removes the ambiguity: a step reads
    *its own parent's* run-log for the identity to pin.

    ``step_versions`` is why a request survives an upgrade. The recipe digest
    identifies the *recipe*; only this map identifies *what ran*, because a
    mosaic upgrade that bumps a producer's ``version`` moves every identifier
    below it. Resolved once at request start and read from here by a step
    re-planning itself, so one request cannot span two identity regimes with its
    early steps under the old versions and its later ones under the new.

    It covers **every** step rather than only the feature ones. An op's version
    is a visible segment of its run identifier rather than a hash term, so a bump
    there makes the step read as absent instead of as complete -- waste rather
    than corruption, but the same request spanning two regimes either way.
    """

    request_id: str
    recipe_digest: str
    owner: str = ""
    created_at: str = ""
    scope: Scope = Field(default_factory=Scope)
    """Narrow the graph to what this selector names. An unset one is all."""
    bind: dict[str, BoundRef] = Field(default_factory=dict)
    allow_partial: bool = False
    """Whether a coverage shortfall may proceed.

    The explicit gesture a refusal asks for, and it lives here so a worker reads
    it with no control plane. For a ``scope_dependent`` step it answers a
    scientific question rather than a maintenance one: a model fitted on 89
    sequences is a different model from one fitted on 90.
    """
    max_concurrent_steps: int | None = None
    step_executions: dict[str, str] = Field(default_factory=dict)
    """Which attempt each step is, assigned before anything runs."""
    step_versions: dict[str, str] = Field(default_factory=dict)
    """The version each step's producer declared when this request was made."""
    step_scopes: dict[str, Scope] = Field(default_factory=dict)
    """Narrow one step, keyed by step id; intersected with ``scope``.

    A train/test split is two step scopes over one request: the fitting steps
    narrowed to the training groups and the scoring step to the test groups.
    Left out of the request file when empty, so a request without one reads
    in an older mosaic.
    """

    tracks_run_id: str | None = None
    """The tracks variant that each feature step without a ``tracks`` reference reads.

    ``None`` leaves each entry's one variant to the step, which refuses an entry
    of two. Left out of the request file when ``None``, so a request without one
    reads in an older mosaic.
    """

    pins: dict[str, str] = Field(default_factory=dict)
    """The run each pinned step serves, keyed by step id.

    A pin names a finished run of a step that fits over its scope, and keeps
    the step's inputs, which ``bind`` replaces. The pinned step is not
    executed, reads complete over any scope, and the steps below it derive
    their identity from the run. Left out of the request file when empty, so a
    request without one reads in an older mosaic.
    """

    applies_request_id: str | None = None
    """The finished request whose fit this request applies, or ``None``.

    The request then records the pins, the tracks variant and the locked steps
    derived from it (:func:`~.apply.derive_applied_fit`), and a plan read from
    it takes them from here and does not read the named request again. Left
    out of the request file when ``None``, so a request without one reads in
    an older mosaic.
    """

    locked: list[str] = Field(default_factory=list)
    """The steps an apply keeps as the named request ran them, in topological order.

    Every step that fits and every step one reads. Empty for a request that
    applies nothing, and left out of the request file then.
    """

    cleanup: CleanupSpec | None = None
    """The superseded runs to remove after every step finished, or ``None``.

    Left out of the request file when ``None``, so a request without one reads
    in an older mosaic. One with it does not, which is why a managed dataset
    refuses it.
    """

    @model_validator(mode="after")
    def _step_scopes_name_steps_by_entry(self) -> Self:
        for step_id, narrowed in self.step_scopes.items():
            if not step_id.strip():
                raise ValueError("step_scopes holds a scope under an empty step id")
            if narrowed.addresses_cameras:
                raise ValueError(
                    f"step_scopes[{step_id!r}] names cameras. A step scope "
                    f"narrows the (group, sequence) entries a step covers."
                )
        return self

    @model_validator(mode="after")
    def _tracks_variant_is_named(self) -> Self:
        if self.tracks_run_id is not None and not self.tracks_run_id.strip():
            raise ValueError(
                "tracks_run_id names no variant; leave it unset to read each "
                "entry's one variant"
            )
        return self

    @model_validator(mode="after")
    def _applied_request_is_named(self) -> Self:
        if self.applies_request_id is not None and not self.applies_request_id.strip():
            raise ValueError(
                "applies_request_id names no request; leave it unset for a "
                "request that applies no fit"
            )
        return self

    @model_validator(mode="after")
    def _bind_targets_are_named(self) -> Self:
        for step_id, bound in self.bind.items():
            if not bound.feature.strip():
                raise ValueError(f"bind[{step_id!r}] names no feature")
            if not bound.run_id.strip():
                raise ValueError(
                    f"bind[{step_id!r}] names no run_id; a bind exists to pin one"
                )
        return self

    @model_validator(mode="after")
    def _pins_name_a_step_and_a_run(self) -> Self:
        for step_id, run_id in self.pins.items():
            if not step_id.strip():
                raise ValueError("pins holds a run under an empty step id")
            if not run_id.strip():
                raise ValueError(
                    f"pins[{step_id!r}] names no run_id; a pin exists to name one"
                )
            if step_id in self.bind:
                raise ValueError(
                    f"step {step_id!r} is both bound and pinned. A bind replaces "
                    f"the step's inputs and a pin keeps them; give it one of the two"
                )
        return self

    def step_scope(self, step_id: str) -> Scope | None:
        """The scope narrowing *step_id*, or ``None`` when only ``scope`` does."""
        return self.step_scopes.get(step_id)

    @property
    def stand_ins(self) -> frozenset[str]:
        """The steps that a run stands in for: bound (``bind``) or pinned (``pins``).

        Neither is executed, so a step below one reads that run and nothing above
        it. A walk over the graph's failures stops at them (``cut``).
        """
        return frozenset(self.bind) | frozenset(self.pins)

    def execution_of(self, step_id: str) -> str:
        """Which attempt *step_id* is, or ``KeyError``.

        Raised rather than returning ``""``, because an empty execution id would
        be read as "no attempt yet" by a caller pinning a parent -- and a step
        this request never assigned an id to is a malformed request, not a step
        that has not started.
        """
        try:
            return self.step_executions[step_id]
        except KeyError:
            raise KeyError(
                f"request {self.request_id!r} assigns no execution id to step "
                f"{step_id!r}. Assigned: {sorted(self.step_executions)}"
            ) from None
