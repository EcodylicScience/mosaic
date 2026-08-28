"""What a recipe file is, and what a request beside it holds.

Two documents, kept apart on purpose.

The **recipe** is the pipeline and nothing else: which steps, with which params,
wired to which other steps. It is portable across datasets, so it never holds a
resolved ``run_id`` -- those are dataset state -- and it never holds a
submission's choices, so two people running the same analysis over different
subsets are running the same recipe.

The **request** is one submission of one recipe against one dataset. It holds
what the recipe must not: which entries, whether a shortfall may proceed, who
asked, and the out-of-graph pins. It is defined here rather than where it is
written because ``resolve_step_spec`` consumes ``request.bind`` and
``plan_pipeline`` takes ``request=``; writing the file and minting the execution
ids into it belong to the phase that executes.

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

from collections.abc import Mapping
from typing import Annotated, ClassVar, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.scope import Scope

__all__ = [
    "BoundRef",
    "FeatureStepSpec",
    "GraphModel",
    "OpStepSpec",
    "Recipe",
    "Request",
    "SCHEMA_VERSION",
    "Step",
    "StepRef",
    "StepRun",
    "TRACKS_INPUT",
    "params_step_refs",
    "references_of",
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


class _StepBase(GraphModel):
    """What every step carries, whatever it runs."""

    id: str
    after: list[str] = Field(default_factory=list)
    """Ordering-only edges: this step runs after those, with no data reference.

    ``transcode -> trex`` is the case -- the tracker's identity has no term for
    the media it read, so nothing in the payload records the dependency. They are
    admissible only where the consumer records and compares what it consumed, and
    ``validate`` is what enforces that.
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
    params: dict[str, JsonValue] = Field(default_factory=dict)
    run: StepRun = Field(default_factory=StepRun)


class OpStepSpec(_StepBase):
    """One op run: a kind and its params.

    An op declares its scope inside ``params`` rather than beside them, which is
    what the op registry already does and what ``mosaic run --kind`` already
    accepts.
    """

    type: Literal["op"] = "op"
    kind: str
    params: dict[str, JsonValue] = Field(default_factory=dict)


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


def params_step_refs(params: Mapping[str, JsonValue]) -> dict[str, StepRef]:
    """The top-level params fields holding a step reference.

    A params reference is a mapping carrying a ``step`` key, because ``params``
    is typed open -- an op passes arbitrary settings through to an external tool,
    and a feature's params are validated by its own model rather than by this
    one. Detecting the shape here means the recipe format needs no second
    declaration of which fields may be references, and no author-supplied label
    to mark one.

    **Top-level only**, matching ``resolve_references`` and
    ``Params.identity_dump()``, neither of which recurses into a nested model. A
    reference buried inside a nested dict would be substituted by neither, so
    finding one here would promise something the resolver does not do.

    Raises:
        ValueError: A mapping carries ``step`` alongside a key ``StepRef`` does
            not declare. Refused rather than ignored: it is a reference its
            author expected to be honoured.
    """
    found: dict[str, StepRef] = {}
    for name, value in params.items():
        if isinstance(value, dict) and "step" in value:
            try:
                found[name] = StepRef.model_validate(value)
            except Exception as exc:
                raise ValueError(
                    f"params.{name} looks like a step reference but is not a "
                    f"valid one: {exc}"
                ) from exc
    return found


class BoundRef(GraphModel):
    """An artifact this graph did not produce, pinned by the submission.

    What makes "run t-SNE over the speed feature I computed last month"
    expressible. It lives on the request rather than in the recipe, so the recipe
    stays portable and free of resolved identifiers while the binding is recorded
    where it is auditable.

    **It pins only.** Params are never bound, or the digest would stop describing
    the work it names.
    """

    feature: str
    run_id: str


class Request(GraphModel):
    """One submission of one recipe against one dataset.

    Holds no ``run_id`` of its own beyond the explicit ``bind`` pins: the
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
