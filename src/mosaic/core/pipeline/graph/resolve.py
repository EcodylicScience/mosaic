"""The one module in this package that reaches the registries.

Everything else here works on declarations -- small frozen records of strings and
booleans -- and this is where those come from. The split is not tidiness:
importing ``FEATURES`` imports every feature module and, through them, scipy,
sklearn and pywt, which is seconds of wall clock. Validating a recipe, ordering
it, listing a step's parents, deciding a lane, rendering a status view,
evaluating a release gate and cancelling must all work without paying it, and the
gate runs far more often than a submit does.

So the registry imports live **inside the functions**, not at module scope, and
``tests/test_graph_imports.py`` holds both halves of that line.

**Declarations are read off what a class already declares**, never off its name.
Whether a feature accepts tracks is answered by validating a probe payload
against its own ``Inputs`` model, so the answer cannot drift from what the
feature will actually accept; whether an op writes tracks is answered by the
tracking-roots registry, which is the table a producer has to fill in to exist at
all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, cast

from mosaic.core.json_value import JsonValue

if TYPE_CHECKING:
    from pydantic import ValidationError
    from pydantic.fields import FieldInfo

    from mosaic.core.pipeline.ops import Op
    from mosaic.core.pipeline.types import Feature, Params

from ..types.feature import EmitsLevel
from .compatibility import (
    TRACKS_DECLARATION,
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    ProducerDecl,
    resolve_emits,
)
from .model import (
    TRACKS_INPUT,
    BoundRef,
    OpStepSpec,
    Recipe,
    params_step_refs,
)
from .storage import storage_name_of

__all__ = [
    "ReferenceSite",
    "ResolvedStep",
    "StepBuildError",
    "StepSpec",
    "build_feature",
    "build_op_params",
    "build_step_feature",
    "build_step_op_params",
    "declaration_catalog",
    "feature_class_for_slug",
    "op_class_for_kind",
    "params_reference_site",
    "resolve_step_spec",
]

_EMITS_LEVELS: frozenset[str] = frozenset(
    {"individual", "pair", "unidentified", "as-input"}
)
"""The vocabulary a feature may declare, for narrowing what ``getattr`` returns."""

_PROBE_FEATURE: str = "__probe__"
"""A feature name no registry holds, used to ask an ``Inputs`` model a question.

Validating ``[{"feature": ...}]`` against a feature's own ``Inputs`` answers
"does this accept another feature's output" using the real validator rather than
by introspecting a generic parameter. The name never escapes this module.
"""


def feature_class_for_slug(slug: str) -> type | None:
    """The registered feature class whose ``name`` is *slug*, or ``None``.

    ``FEATURES`` is keyed by class name while a recipe and the CLI both name a
    feature by its ``name`` slug, so the lookup is a scan. One copy, here: it was
    spelled three times -- in the CLI, in this package's caller, and in
    mosaic-api's lane routing -- and a fourth would have been written the next
    time something needed to turn a slug into a class.

    Returns ``None`` rather than raising, because both callers want to phrase
    their own refusal: the CLI lists the available slugs, and recipe validation
    names the step the unknown slug sits in.
    """
    from mosaic.behavior.feature_library import FEATURES

    for cls in FEATURES.values():
        if getattr(cls, "name", None) == slug:
            return cls
    return None


def op_class_for_kind(kind: str) -> "type[Op[Params]] | None":
    """The registered op class for *kind*, or ``None``.

    Registers the tracking ops first, because the generic registry holds no
    tracking kinds until their subpackages are imported -- so a lookup made
    before anything else touched them would report a real op as unknown. The
    call is idempotent and is what every other reader of ``OPS`` does.
    """
    from mosaic.core.pipeline.ops import OPS
    from mosaic.tracking import register_ops

    register_ops()
    return OPS.get(kind)


class StepBuildError(ValueError):
    """A step cannot be constructed from what the recipe declares.

    Carries *stage* so a caller can phrase its own refusal without parsing the
    message: the CLI names the available slugs for an unknown one and points at
    ``--inputs`` or ``--params`` for the other two, while recipe validation names
    the step the fault sits in.
    """

    def __init__(self, stage: "BuildStage", name: str, detail: str) -> None:
        self.stage: BuildStage = stage
        self.name: str = name
        self.detail: str = detail
        super().__init__(f"{name}: {detail}")


type BuildStage = Literal["slug", "inputs", "params"]
"""Which half of a step's declaration was rejected."""


def _compact(exc: "ValidationError") -> str:
    """Render a pydantic ``ValidationError`` as ``field: message; ...``."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        message = str(err.get("msg", ""))
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts)


def build_feature(
    slug: str,
    inputs_payload: object | None,
    params: object | None,
) -> "Feature":
    """Construct a runnable feature from a slug plus JSON inputs and params.

    **The one place a feature is built from a document.** The CLI builds one
    from ``--inputs`` / ``--params``, recipe validation builds one to find out
    whether a step is well formed, and planning builds one to hash -- and three
    constructions would be three answers to what a step means, of which the
    validating one would be the answer nobody meets.

    Args:
        slug: The feature's ``name``.
        inputs_payload: The inputs as JSON -- a list of ``"tracks"`` literals and
            ``{"feature", "run_id"}`` mappings. ``None`` means the default
            ``["tracks"]``, which is what a feature reading tracks gets.
        params: The params as a mapping, or ``None`` for the defaults. Passed
            through ``from_overrides``, so a partial mapping for a field with a
            default factory is merged onto that default rather than replacing it
            -- which is what reconstructs an artifact reference from a bare
            ``{"feature", "run_id"}``.

    Raises:
        StepBuildError: The slug names nothing, the inputs are not admissible,
            or the params do not validate.
    """
    from pydantic import BaseModel, ValidationError

    cls = feature_class_for_slug(slug)
    if cls is None:
        raise StepBuildError("slug", slug, f"no registered feature is named {slug!r}")

    inputs_cls: object = getattr(cls, "Inputs", None)
    if not (isinstance(inputs_cls, type) and issubclass(inputs_cls, BaseModel)):
        raise StepBuildError("inputs", slug, "the feature declares no Inputs model")
    if inputs_payload is not None and not isinstance(inputs_payload, list):
        raise StepBuildError(
            "inputs",
            slug,
            'inputs must be a list, e.g. ["tracks"] or [{"feature": "speed-angvel"}]',
        )
    try:
        inputs_obj = inputs_cls.model_validate(
            ["tracks"] if inputs_payload is None else inputs_payload
        )
    except ValidationError as exc:
        raise StepBuildError("inputs", slug, _compact(exc)) from exc

    if params is not None and not isinstance(params, dict):
        raise StepBuildError("params", slug, "params must be a mapping")
    try:
        # The uniform feature constructor, which every feature follows:
        # ``FeatureCls(inputs, params: dict | None)``. ``Feature`` is a Protocol
        # and declares no ``__init__``, hence the targeted ignore.
        return cls(inputs_obj, params)  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        raise StepBuildError("params", slug, _compact(exc)) from exc


def build_op_params(
    kind: str,
    params: Mapping[str, JsonValue],
    entries: Sequence[tuple[str, str]] = (),
) -> "Params":
    """Validate one op step's params against the op's own model.

    The op half of :func:`build_feature`, and the same single-site rule: this is
    what ``run_op`` does with the same mapping, so a recipe that validates here is
    one that runs.

    *entries* is what the plan is running this step over, handed to the op's own
    :meth:`~mosaic.core.pipeline.ops.Op.scoped_params` so it lands in whichever
    field that op reads a scope from. An op step's scope comes from the plan
    rather than from the recipe, which is what lets one recipe run over several
    datasets.

    Raises:
        StepBuildError: The kind names nothing, or the params do not validate.
    """
    from pydantic import ValidationError

    op_cls = op_class_for_kind(kind)
    if op_cls is None:
        raise StepBuildError("slug", kind, f"no registered op is named {kind!r}")
    try:
        return op_cls.Params.model_validate(op_cls.scoped_params(params, entries))
    except ValidationError as exc:
        raise StepBuildError("params", kind, _compact(exc)) from exc


def build_step_feature(spec: "StepSpec") -> "Feature":
    """The feature one resolved feature step would run.

    Raises:
        StepBuildError: As :func:`build_feature`.
    """
    return build_feature(spec.feature, list(spec.inputs), dict(spec.params))


def build_step_op_params(spec: "StepSpec") -> "Params":
    """The validated params one resolved op step would run under.

    Raises:
        StepBuildError: As :func:`build_op_params`.
    """
    return build_op_params(spec.op_kind, spec.params, spec.entries)


def _accepts(inputs_cls: type, payload: list[object]) -> bool:
    """Would this ``Inputs`` model accept *payload*?

    Asked of the model rather than of its type parameter. A feature narrows its
    inputs by subclassing ``Inputs[...]`` and sometimes by a validator on top, so
    the specialization alone is not the whole answer -- and reading it back at
    run time means reading pydantic's generic metadata, which would be a second
    description of a rule the validator already holds.
    """
    validate = getattr(inputs_cls, "model_validate", None)
    if validate is None:
        return False
    try:
        _ = validate(payload)
    except Exception:
        return False
    return True


def _feature_declaration(cls: type) -> Declaration:
    """Read one feature class as a producer and a consumer."""
    from mosaic.core.pipeline.loading import CROSS_JOIN_FEATURES

    slug = str(getattr(cls, "name", cls.__name__))
    inputs_cls: type | None = getattr(cls, "Inputs", None)
    require = str(getattr(inputs_cls, "_require", "nonempty"))
    roots: tuple[str, ...] = tuple(
        str(root) for root in getattr(cls, "consumed_roots", ())
    )
    # Every registered feature declares this and a conformance test holds that,
    # so the fallback is for a class handed in from outside the registry.
    declared: object = getattr(cls, "emits", "as-input")
    emits: EmitsLevel = declared if declared in _EMITS_LEVELS else "as-input"  # pyright: ignore[reportAssignmentType]

    accepts_tracks = inputs_cls is not None and _accepts(inputs_cls, [TRACKS_INPUT])
    accepts_features = inputs_cls is not None and _accepts(
        inputs_cls, [{"feature": _PROBE_FEATURE}]
    )
    return Declaration(
        produces=ProducerDecl(
            name=slug,
            kind="feature",
            # Unresolved here on purpose: a chain resolves a passthrough from its
            # own upstream, and a producer offered to a canvas with nothing above
            # it reads as individual, which is what raw tracks give it.
            level=resolve_emits(emits),
            writes_tracks=False,
            writes_media=False,
        ),
        consumes=ConsumerDecl(
            name=slug,
            kind="feature",
            accepts_tracks=accepts_tracks,
            accepts_features=accepts_features,
            requires_track_shape=bool(getattr(inputs_cls, "_track_input", False)),
            takes_no_inputs=require == "empty",
            cross_joins=slug in CROSS_JOIN_FEATURES,
            reads_media="media_raw" in roots,
        ),
        emits=emits,
        category=str(getattr(cls, "category", "")),
        # Optional on a feature and defaulted to cpu, which is what every
        # feature but a declared-heavy one is.
        resource_class=str(getattr(cls, "resource_class", "") or "cpu"),
    )


def _op_declaration(kind: str, op_cls: type) -> Declaration:
    """Read one op as a producer and a consumer.

    An op is never a feature's *input* in the ``inputs`` sense -- what a feature
    reads from a tracker is the ``tracks/`` variant it produced, referenced from
    the step's ``tracks`` field -- so its producer record exists to say that, and
    to carry the two facts an ordering-only edge is checked against.

    ``writes_tracks`` is the op's own declaration, falling back to membership of
    the tracking-roots registry -- the table a producer must appear in to *bridge
    from a tracker run root*, which is what every tracks producer did until one
    arrived that reads a tracks table and writes another. ``writes_media`` comes
    from the op's declared category. Neither is a name list.

    ``reads_media`` stays tied to the registry rather than following
    ``writes_tracks``: it is true because bridging into tracks means opening the
    entry's video, which is exactly what a tracks-to-tracks producer does not do.
    """
    from mosaic.core.pipeline.ops import op_resource_class
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

    bridges_tracks = kind in TRACKING_ROOTS
    declared_tracks = getattr(op_cls, "writes_tracks", None)
    return Declaration(
        produces=ProducerDecl(
            name=kind,
            kind="op",
            level="individual",
            writes_tracks=bridges_tracks
            if declared_tracks is None
            else bool(declared_tracks),
            writes_media=str(getattr(op_cls, "category", "")) == "transcode",
        ),
        consumes=ConsumerDecl(
            name=kind,
            kind="op",
            # An op declares its scope in its params rather than taking pipeline
            # inputs, so nothing may be wired into it. What it depends on is
            # expressed as a params reference or as an ordering-only edge.
            takes_no_inputs=True,
            # Every producer that bridges into tracks opens video to do it, which
            # is what makes an ordering-only edge from a media writer meaningful.
            reads_media=bridges_tracks,
        ),
        emits="individual",
        category=str(getattr(op_cls, "category", "")),
        resource_class=op_resource_class(kind),
    )


@lru_cache(maxsize=1)
def declaration_catalog() -> DeclarationCatalog:
    """Every step type this installation knows, as declarations.

    Cached for the process, which is safe because both registries are filled by
    import side effect and are fixed once the modules that populate them have
    run. Anything registering a feature after the first call would not appear,
    and nothing does that outside a test.

    The result is JSON-serializable by construction, so an API can hand it
    straight to a browser and the canvas answers connection questions locally
    rather than asking per candidate wire.
    """
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.ops import OPS
    from mosaic.tracking import register_ops

    # The generic registry holds no tracking kinds until their subpackages are
    # imported; this is idempotent and is what every other reader of OPS does.
    register_ops()

    entries: dict[str, Declaration] = {TRACKS_INPUT: TRACKS_DECLARATION}
    for cls in FEATURES.values():
        declared = _feature_declaration(cls)
        entries[declared.produces.name] = declared
    for kind, op_cls in OPS.items():
        entries[kind] = _op_declaration(kind, op_cls)
    return DeclarationCatalog(entries=entries)


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    """What an earlier step in the walk turned out to be.

    What a later step substitutes its references from. Kept apart from the plan
    record because resolution needs only these four facts, and taking the whole
    record would let a substitution depend on coverage -- which would make the
    identity of a step depend on how much of its upstream happened to be done.

    Attributes:
        step_id: The step this describes.
        storage_name: Where a feature step's outputs live, which is what a
            downstream ``Result`` names -- not the slug. A chain three deep
            nests its suffixes, so this is carried rather than recomputed.
        run_id: The identifier, or ``None`` when it could not be resolved.
        tracks_variant: What an op step's tables are named by, empty otherwise.
        model_run_id: What names a training step's model, empty otherwise.
    """

    step_id: str
    storage_name: str = ""
    run_id: str | None = None
    tracks_variant: str = ""
    model_run_id: str = ""


@dataclass(frozen=True, slots=True)
class StepSpec:
    """Exactly what it takes to run one step, and nothing about how.

    The hand-off out of planning: mapped to ``mosaic run``'s arguments by
    anything that starts a process, and to a queue's job spec by anything that
    schedules one. It carries no coverage, no status and no lane -- those are the
    plan's, and a spec that carried them would invite a runner to act on a view
    of the world it did not read itself.
    """

    step_id: str
    kind: Literal["feature", "op"]
    feature: str = ""
    op_kind: str = ""
    inputs: tuple[JsonValue, ...] = ()
    params: Mapping[str, JsonValue] = field(
        default_factory=lambda: cast("dict[str, JsonValue]", {})
    )
    tracks_run_id: str | None = None
    entries: tuple[tuple[str, str], ...] = ()
    """What this step should compute, already narrowed.

    A step must not re-request its whole scope: it is the scope minus what is
    covered and minus what is quarantined. Empty means everything in scope, which
    is what a first run of a cold step gets.
    """
    filter_start_frame: int | None = None
    filter_end_frame: int | None = None
    filter_start_time: float | None = None
    filter_end_time: float | None = None
    overlap_frames: int = 0

    @property
    def storage_name(self) -> str:
        """Where this step's outputs land, for a feature step."""
        return storage_name_of(
            self.feature, [_input_name(item) for item in self.inputs]
        )


def _input_name(item: JsonValue) -> str:
    """The storage name an input contributes to a downstream storage suffix."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("feature", ""))
    return ""


def _params_field_types(kind: str, name: str) -> dict[str, object]:
    """The declared type of each top-level params field of a step.

    What decides how a reference is substituted. Read from the model rather than
    from a label the recipe author wrote, so a reference lands in the shape its
    consumer actually declares -- an ``ArtifactSpec`` field gets a reference with
    a pattern, a ``str`` field gets a bare identifier, and neither is a choice
    the file gets to make.
    """
    if kind == "feature":
        cls = feature_class_for_slug(name)
        params_cls = getattr(cls, "Params", None) if cls is not None else None
    else:
        from mosaic.core.pipeline.ops import OPS

        op_cls = OPS.get(name)
        params_cls = getattr(op_cls, "Params", None) if op_cls is not None else None
    declared: object = getattr(params_cls, "model_fields", None)
    if not isinstance(declared, dict):
        return {}
    typed = cast("dict[str, FieldInfo]", declared)
    return {key: info.annotation for key, info in typed.items()}


def _is_reference_field(annotation: object) -> bool:
    """Does this declared type hold a run reference rather than a bare value?"""
    from mosaic.core.pipeline.types import Result

    candidates = [annotation, *getattr(annotation, "__args__", ())]
    return any(
        isinstance(candidate, type) and issubclass(candidate, Result)
        for candidate in candidates
    )


def _is_identifier_field(annotation: object) -> bool:
    """Does this declared type hold a bare identifier -- how a model is named?"""
    candidates = [annotation, *getattr(annotation, "__args__", ())]
    return any(candidate is str for candidate in candidates)


type ReferenceSite = Literal["result", "identifier", "unsupported", "unknown"]
"""What a params field can be substituted with, read off its declared type.

``result`` takes ``{feature, run_id}`` -- a ``Result`` or an ``ArtifactSpec``.
``identifier`` takes the bare upstream identifier, which is how a model is named.
``unsupported`` is a field that holds neither, so a reference landing there would
be substituted into a number or a boolean and rejected by the model with a
message about the substituted value rather than about the reference. ``unknown``
is a field the model does not declare at all.
"""


def params_reference_site(kind: str, name: str, field_name: str) -> ReferenceSite:
    """How a reference in *field_name* of this step's params would substitute.

    The question ``validate`` asks before substituting anything, so a reference
    in a field that cannot hold one is refused as what it is rather than as a
    type error two layers down.
    """
    types = _params_field_types(kind, name)
    if field_name not in types:
        return "unknown"
    annotation = types[field_name]
    if _is_reference_field(annotation):
        return "result"
    if _is_identifier_field(annotation):
        return "identifier"
    return "unsupported"


def resolve_step_spec(
    recipe: Recipe,
    step_id: str,
    resolved: Mapping[str, ResolvedStep],
    *,
    bind: Mapping[str, BoundRef] | None = None,
    entries: Sequence[tuple[str, str]] = (),
) -> StepSpec:
    """Substitute every reference in one step from what is already resolved.

    Called in topological order, so every step this one names has been resolved
    before it is reached. Substitution is decided by the **declared field type**,
    never by a label in the file:

    - an ``inputs`` reference becomes ``{feature, run_id}`` naming the upstream's
      *storage* name, which is what a ``Result`` carries;
    - the step-level ``tracks`` reference becomes the upstream's tracks
      **variant**, which is a different identifier from its run id;
    - a params field declared as a ``Result`` or ``ArtifactSpec`` becomes
      ``{feature, run_id}`` plus the reference's pattern when it gave one;
    - a params field declared as a plain ``str`` becomes the bare identifier,
      which is how a model is named.

    Only top-level params fields are scanned, matching ``resolve_references`` and
    ``Params.identity_dump()``, neither of which recurses -- so finding a
    reference deeper would promise a substitution nothing performs.

    *bind* is applied first and is never overridden: an out-of-graph artifact the
    submission pinned is already resolved when the walk reaches it.
    """
    step = recipe.step(step_id)
    bound = (bind or {}).get(step_id)

    if isinstance(step, OpStepSpec):
        types = _params_field_types("op", step.kind)
        params = dict(step.params)
        for name, reference in params_step_refs(step.params).items():
            upstream = resolved.get(reference.step)
            if upstream is None:
                continue
            # An op names a model by a bare identifier: the training run when
            # there is one, which is what resolve_model returns for it.
            params[name] = (
                upstream.model_run_id or upstream.run_id or ""
                if not _is_reference_field(types.get(name))
                else {"feature": upstream.storage_name, "run_id": upstream.run_id}
            )
        return StepSpec(
            step_id=step_id,
            kind="op",
            op_kind=step.kind,
            params=params,
            entries=tuple(entries),
        )

    types = _params_field_types("feature", step.feature)
    inputs: list[JsonValue] = []
    for item in step.inputs:
        if isinstance(item, str):
            inputs.append(item)
            continue
        upstream = resolved.get(item.step)
        inputs.append(
            {
                # The storage name, not the slug: a Result names where the output
                # lives, and a chain three deep nests its suffixes.
                "feature": upstream.storage_name if upstream else "",
                "run_id": upstream.run_id if upstream else None,
            }
        )

    params = dict(step.params)
    for name, reference in params_step_refs(step.params).items():
        upstream = resolved.get(reference.step)
        if upstream is None:
            continue
        if _is_reference_field(types.get(name)):
            payload: dict[str, JsonValue] = {
                "feature": upstream.storage_name,
                "run_id": upstream.run_id,
            }
            # Pinned only when the recipe said so. ArtifactSpec defaults its glob
            # to *.parquet, which silently resolves the wrong file out of a
            # producer that writes more than one -- so the author pins it, and
            # leaving it unset here keeps the model's own default in force rather
            # than substituting a guess.
            if reference.pattern:
                payload["pattern"] = reference.pattern
            params[name] = payload
        else:
            params[name] = upstream.run_id or ""

    tracks_run_id: str | None = None
    if step.tracks is not None:
        upstream = resolved.get(step.tracks.step)
        # The variant, never the op run id. They coincide today only because the
        # tracker variant payload is an unwrapped passthrough of the settings.
        tracks_run_id = upstream.tracks_variant if upstream else None

    if bound is not None:
        # An out-of-graph pin resolves the step outright; nothing derived above
        # may override it, because the submission chose it deliberately.
        inputs = [{"feature": bound.feature, "run_id": bound.run_id}]

    return StepSpec(
        step_id=step_id,
        kind="feature",
        feature=step.feature,
        inputs=tuple(inputs),
        params=params,
        tracks_run_id=tracks_run_id,
        entries=tuple(entries),
        filter_start_frame=step.run.filter_start_frame,
        filter_end_frame=step.run.filter_end_frame,
        filter_start_time=step.run.filter_start_time,
        filter_end_time=step.run.filter_end_time,
        overlap_frames=step.run.overlap_frames,
    )
