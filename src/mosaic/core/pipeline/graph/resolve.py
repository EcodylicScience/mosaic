"""The one module in this package that reaches the registries.

Everything else here works on declarations -- small frozen records of strings and
booleans -- and this is where those come from. The split is not tidiness:
importing ``FEATURES`` imports every feature module and, through them, scipy,
sklearn and pywt, which is seconds of wall clock. Validating a recipe, ordering
it, listing a step's parents, deciding a lane, rendering a status view,
evaluating a release gate and cancelling must all work without paying it, and the
gate runs far more often than a submit does.

So the registry imports live **inside the functions**, not at module scope.
:func:`feature_class_for_slug` and :func:`declaration_catalog` import ``FEATURES``
from ``mosaic.behavior.feature_library``, and :func:`op_class_for_kind` and
:func:`declaration_catalog` import ``register_ops`` from ``mosaic.tracking``. Both
are imports from ``core`` upward, made only to fill a registry, and kept inside a
call so that importing this package stays cheap and ``core`` imports neither
``behavior`` nor ``tracking`` at module scope.
``tests/meta/test_graph_imports.py`` checks the feature-library imports. No test
checks the ``mosaic.tracking`` ones. :func:`toolkit_model_for_ref` imports the
module of a toolkit model that a feature definition names, also inside the call.
:func:`fill_builtin_registries` and :func:`registry_snapshot` fill and read every
registry, for the check that a definition's source registers nothing.

**Declarations are read off what a class already declares**, never off its name.
Whether a feature accepts tracks is answered by validating a probe payload
against its own ``Inputs`` model, so the answer cannot drift from what the
feature will actually accept; whether an op writes tracks is answered by the
tracking-roots registry, which is the table a producer has to fill in to exist at
all.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast
from weakref import WeakKeyDictionary

from pydantic import BaseModel

from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_object_tuple
from mosaic.core.pipeline.definitions.store import (
    DefinitionCorrupt,
    DefinitionInvalid,
    DefinitionToolkitModelMissing,
)
from mosaic.core.scope import Scope
from mosaic.optional_dependency import (
    Prerequisite,
    RestrictedUse,
    license_notes,
    unmet,
)

if TYPE_CHECKING:
    from pydantic import ValidationError
    from pydantic.fields import FieldInfo

    from mosaic.core.pipeline.definitions.sidecar import DefinitionSidecar
    from mosaic.core.pipeline.ops import Op
    from mosaic.core.pipeline.types import Feature
    from mosaic.core.params import DeclaredModel, Params

from ..column_contract import (
    PAIR_RULE_FIELD,
    Column,
    Provides,
    contract_fields,
    declared_provides,
    declared_requires,
)
from ..error_kind import ClassifiedError, ErrorKind
from ..label_controls import LABEL_CONTROL_FIELDS
from ..refusal import ToolUnavailable
from ..track_space import feature_track_spaces
from ..types.data_config import COLUMNS
from ..feature_levels import EmitsLevel
from .columns import provided_wire, requires_wire, tracks_provides
from .compatibility import (
    TRACKS_DECLARATION,
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    GpuWhen,
    ProducerDecl,
    resolve_emits,
)
from .lanes import lane_for
from .reach import class_reach, class_reach_by_option
from .model import (
    TRACKS_INPUT,
    FeatureStepSpec,
    OpStepSpec,
    Recipe,
    params_step_refs,
)
from .storage import storage_name_of
from .wire import (
    CatalogEntry,
    CatalogIndex,
    CatalogIndexEntry,
    ColumnNames,
    DefinitionWire,
    EditorCatalog,
    ProvidedColumns,
    ReferenceKind,
    StepTypeDetail,
    render_location,
)

__all__ = [
    "ReferenceSite",
    "RegistryName",
    "RegistrySnapshot",
    "ResolvedStep",
    "StepBuildError",
    "StepSpec",
    "StepTypeNotFound",
    "build_feature",
    "build_op_params",
    "build_step_feature",
    "build_step_op_params",
    "builtin_declaration_catalog",
    "builtin_editor_catalog",
    "catalog_index",
    "declaration_catalog",
    "declared_fit",
    "declared_version",
    "definition_declaration",
    "describe_step_type",
    "editor_catalog",
    "feature_class_for_slug",
    "fill_builtin_registries",
    "first_docstring_line",
    "fitting_steps",
    "fields_need_ground_truth",
    "op_class_for_kind",
    "op_writes_tracks",
    "params_reference_site",
    "registry_snapshot",
    "resolve_step_spec",
    "run_reference_fields",
    "step_type_detail",
    "step_type_lane",
    "toolkit_model_for_ref",
    "unavailable_tool",
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

    A ``def:`` slug names a feature definition of the ambient definition scope
    (:func:`~mosaic.core.pipeline.definitions.scope.definition_class_for_slug`),
    never a registered class.

    Raises:
        FeatureDefinitionsNotAllowed: *slug* is a ``def:`` slug and the scope is
            disabled.
    """
    from mosaic.core.pipeline.definitions.store import is_definition_slug

    if is_definition_slug(slug):
        from mosaic.core.pipeline.definitions.scope import definition_class_for_slug

        return definition_class_for_slug(slug)
    from mosaic.behavior.feature_library import FEATURES

    for cls in FEATURES.values():
        if getattr(cls, "name", None) == slug:
            return cls
    return None


def declared_fit(
    slug: str, params: Mapping[str, JsonValue], declared: Declaration
) -> tuple[bool, bool]:
    """Whether a feature step fits, and whether it fits over its scope.

    Read off the class and the step's *params* without building the feature,
    for a step whose feature cannot be built yet or need not be: a fitter given
    a ``model`` applies it and fits nothing, and its identity covers no
    entries. A built feature answers both on its instance.

    Returns:
        ``(fits, scope_dependent)``.
    """
    applies = params.get("model") is not None
    cls = feature_class_for_slug(slug)
    fits = declared.fits and not applies
    scope_dependent = bool(getattr(cls, "scope_dependent", False)) and not (
        declared.fits and applies
    )
    return fits, scope_dependent


def fitting_steps(recipe: Recipe, catalog: DeclarationCatalog) -> dict[str, bool]:
    """The feature steps of *recipe* that fit, and whether each fits over its scope.

    Applies :func:`declared_fit` to every feature step whose feature *catalog*
    declares. An op step, a step whose feature the catalog lacks and a step given
    a ``model`` to apply are left out.

    Args:
        recipe: The recipe to read.
        catalog: The declarations that say which features fit.

    Returns:
        Each fitting step's id, in recipe order, mapped to its ``scope_dependent``.
    """
    fitted: dict[str, bool] = {}
    for step in recipe.steps:
        if not isinstance(step, FeatureStepSpec):
            continue
        declared = catalog.get(step.feature)
        if declared is None:
            continue
        fits, scope_dependent = declared_fit(step.feature, step.params, declared)
        if fits:
            fitted[step.id] = scope_dependent
    return fitted


def feature_parallelizable(slug: str) -> bool:
    """Whether the registered feature *slug* can compute several entries at once.

    Returns ``False`` for a slug that names no feature.
    """
    cls = feature_class_for_slug(slug)
    return bool(getattr(cls, "parallelizable", False))


def op_parallelizable(kind: str) -> bool:
    """Whether the registered op *kind* can compute several entries at once.

    Returns ``False`` for a kind that names no op.
    """
    cls = op_class_for_kind(kind)
    return cls is not None and cls.parallelizable


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


def declared_version(kind: Literal["feature", "op"], name: str) -> str:
    """The version the producer behind a step declares, or ``""`` when unknown.

    What a request pins so it cannot span two identity regimes. A mosaic upgrade
    that bumps a version moves every identifier below it, and a request resolving
    its early steps under the old versions and its later ones under the new would
    read as half-absent with no fault anywhere -- so the version is resolved once,
    at submit, and a step re-planning itself checks against what was resolved
    rather than against what is installed now.

    ``""`` for a name no registry holds, rather than raising: the caller that
    cares about an unknown slug is recipe validation, which names the step.

    A ``def:`` slug reads the ``version`` of the sidecar of the version it names
    in the definition scope in force, and imports no source.

    Raises:
        FeatureDefinitionsNotAllowed: *name* is a ``def:`` slug and the scope is
            disabled.
        DefinitionDigestAmbiguous: *name* names a digest prefix that more than
            one version starts with.
    """
    from mosaic.core.pipeline.definitions.store import is_definition_slug

    if kind == "feature" and is_definition_slug(name):
        from mosaic.core.pipeline.definitions.scope import definition_sidecar_for_slug

        sidecar = definition_sidecar_for_slug(name)
        return "" if sidecar is None else sidecar.version
    if kind == "feature":
        cls = feature_class_for_slug(name)
    else:
        cls = op_class_for_kind(name)
    if cls is None:
        return ""
    version = getattr(cls, "version", "")
    return version if isinstance(version, str) else ""


def unavailable_tool(
    kind: Literal["feature", "op"], name: str, catalog: DeclarationCatalog
) -> ToolUnavailable | None:
    """The refusal for a producer that *catalog* reads as unavailable, else ``None``.

    The tool is the name that the producer's class declares in a ``tool``
    attribute, else the producer's own name. The cause is ``license`` when a
    prerequisite that restricts use is unmet, and ``environment`` otherwise.
    The message is the catalog's sentence naming what to provide.
    """
    declared = catalog.get(name)
    if declared is None or declared.unavailable is None:
        return None
    cls = feature_class_for_slug(name) if kind == "feature" else op_class_for_kind(name)
    prerequisites = () if cls is None else _prerequisites_of(cls)
    restricted = any(
        isinstance(prerequisite, RestrictedUse) and prerequisite.missing() is not None
        for prerequisite in prerequisites
    )
    declared_tool: object = getattr(cls, "tool", name)
    tool = declared_tool if isinstance(declared_tool, str) else name
    return ToolUnavailable(
        tool, "license" if restricted else "environment", declared.unavailable
    )


class StepBuildError(ClassifiedError, ValueError):
    """A step cannot be constructed from what the recipe declares.

    Carries *stage* so a caller can phrase its own refusal without parsing the
    message: the CLI names the available slugs for an unknown one and points at
    ``--inputs`` or ``--params`` for the other two, while recipe validation names
    the step the fault sits in.

    *errors* holds the validator's findings as ``(location, message)`` pairs,
    and is empty when no validator refused. Recipe validation reports one
    problem per pair, which lets an editor mark the field at fault.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(
        self,
        stage: "BuildStage",
        name: str,
        detail: str,
        errors: tuple[FieldError, ...] = (),
    ) -> None:
        self.stage: BuildStage = stage
        self.name: str = name
        self.detail: str = detail
        self.errors: tuple[FieldError, ...] = errors
        super().__init__(f"{name}: {detail}")

    def error_reason(self) -> str:
        """Return :attr:`stage`, the half of the declaration that was rejected."""
        return self.stage

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the step type's name and the validator's findings."""
        errors: list[JsonValue] = [
            {"location": location, "message": message}
            for location, message in self.errors
        ]
        return {"name": self.name, "errors": errors}


type BuildStage = Literal["slug", "inputs", "params"]
"""Which half of a step's declaration was rejected."""


type FieldError = tuple[str, str]
"""One validator finding: where in the model, rendered, and what is wrong."""


def _field_errors(exc: "ValidationError") -> tuple[FieldError, ...]:
    """A pydantic ``ValidationError`` as ``(location, message)`` pairs."""
    return tuple((render_location(err["loc"]), str(err["msg"])) for err in exc.errors())


def _build_error(
    stage: "BuildStage", name: str, exc: "ValidationError"
) -> StepBuildError:
    """A refusal carrying the validator's findings, compacted and itemized."""
    errors = _field_errors(exc)
    detail = "; ".join(
        f"{location}: {message}" if location else message
        for location, message in errors
    )
    return StepBuildError(stage, name, detail, errors)


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

    A ``def:`` slug under a definition scope that loads code builds the loaded
    class and the class declared from its sidecar alike, and the two must agree
    on every term of the run identity.

    Raises:
        StepBuildError: The slug names nothing, the inputs are not admissible,
            or the params do not validate.
        DefinitionMismatch: The loaded definition's instance differs from its
            declared one in a term of the run identity, each named.
    """
    cls = feature_class_for_slug(slug)
    if cls is None:
        raise StepBuildError("slug", slug, f"no registered feature is named {slug!r}")
    feature = _construct(cls, slug, inputs_payload, params)
    _refuse_a_differing_load(cls, slug, feature, inputs_payload, params)
    return feature


_IDENTITY_ATTRIBUTES: Final = (
    "version",
    "scope_dependent",
    "consumed_roots",
    "fits",
    "parallelizable",
    "emits",
)
"""The instance values that a run's identity and its plan read."""


def _refuse_a_differing_load(
    cls: type,
    slug: str,
    loaded: "Feature",
    inputs_payload: object | None,
    params: object | None,
) -> None:
    """Refuse a loaded definition whose identity its declared class does not predict.

    The declared instance is built from the same inputs and params. They must
    agree on the canonical JSON of each identity term and on the instance values
    of :data:`_IDENTITY_ATTRIBUTES`, read as ``compute_run_id`` reads them.

    Raises:
        DefinitionMismatch: They differ, naming each differing term.
    """
    from mosaic.core.pipeline.definitions.loader import definition_ref_of
    from mosaic.core.pipeline.definitions.scope import declared_class_of
    from mosaic.core.pipeline.definitions.store import (
        DefinitionMismatch,
        is_definition_slug,
    )

    if not is_definition_slug(slug):
        return
    declared_cls = declared_class_of(cls)
    ref = definition_ref_of(cls)
    if declared_cls is None or ref is None:
        return
    from mosaic.core.pipeline._utils import canonical_json
    from mosaic.core.pipeline.run import identity_terms

    declared = _construct(declared_cls, slug, inputs_payload, params)
    loaded_terms = identity_terms(loaded.params, loaded.inputs)
    declared_terms = identity_terms(declared.params, declared.inputs)
    differing = [
        term
        for term in sorted(loaded_terms)
        if canonical_json(loaded_terms[term])
        != canonical_json(declared_terms.get(term))
    ]
    differing.extend(
        attribute
        for attribute in _IDENTITY_ATTRIBUTES
        if canonical_json(getattr(loaded, attribute, None))
        != canonical_json(getattr(declared, attribute, None))
    )
    if differing:
        raise DefinitionMismatch(ref.name, ref.digest, differing)


def _construct(
    cls: type, slug: str, inputs_payload: object | None, params: object | None
) -> "Feature":
    """Build *cls* from JSON inputs and params, as :func:`build_feature` does.

    Raises:
        StepBuildError: The inputs are not admissible or the params do not
            validate.
    """
    from pydantic import BaseModel, ValidationError

    inputs_cls: object = getattr(cls, "Inputs", None)
    if not (isinstance(inputs_cls, type) and issubclass(inputs_cls, BaseModel)):
        raise StepBuildError("inputs", slug, "the feature declares no Inputs model")
    if inputs_payload is not None and not isinstance(inputs_payload, list):
        raise StepBuildError(
            "inputs",
            slug,
            'inputs must be a list, e.g. ["tracks"] or [{"feature": "speed-angvel__from__tracks"}]',
        )
    try:
        inputs_obj = inputs_cls.model_validate(
            ["tracks"] if inputs_payload is None else inputs_payload
        )
    except ValidationError as exc:
        raise _build_error("inputs", slug, exc) from exc

    if params is not None and not isinstance(params, dict):
        raise StepBuildError("params", slug, "params must be a mapping")
    try:
        # The uniform feature constructor, which every feature follows:
        # ``FeatureCls(inputs, params: dict | None)``. ``Feature`` is a Protocol
        # and declares no ``__init__``, hence the targeted ignore.
        return cls(inputs_obj, params)  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        raise _build_error("params", slug, exc) from exc


def build_op_params(kind: str, params: Mapping[str, JsonValue]) -> "Params":
    """Validate one op step's params against the op's own model.

    The op half of :func:`build_feature`, and the same single-site rule. This is
    what ``run_op`` does with the same mapping. A recipe that validates here is
    one that runs.

    The recipe's params are validated as written. An op step's scope comes from
    the plan rather than from the recipe, which lets one recipe run over several
    datasets. It is given to the op as ``run_op``'s ``scope`` argument rather
    than as a params field.

    Raises:
        StepBuildError: The kind names nothing, or the params do not validate.
    """
    from pydantic import ValidationError

    op_cls = op_class_for_kind(kind)
    if op_cls is None:
        raise StepBuildError("slug", kind, f"no registered op is named {kind!r}")
    try:
        return op_cls.Params.model_validate(params)
    except ValidationError as exc:
        raise _build_error("params", kind, exc) from exc


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
    return build_op_params(spec.op_kind, spec.params)


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


def _param_defaults(cls: type, fields: frozenset[str]) -> dict[str, JsonValue]:
    """The default of each of *fields* on *cls*'s params model, as JSON.

    A field with no default, or one whose default is not JSON, is left out: a
    reference reading it resolves as unknown until a step sets it.
    """
    from pydantic import TypeAdapter, ValidationError

    params_cls: object = getattr(cls, "Params", None)
    if not (isinstance(params_cls, type) and issubclass(params_cls, BaseModel)):
        return {}
    adapter: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
    defaults: dict[str, JsonValue] = {}
    for name in sorted(fields):
        info = params_cls.model_fields.get(name)
        if info is None or info.is_required():
            continue
        default: object = info.get_default(call_default_factory=True)
        if isinstance(default, BaseModel):
            # A nested params model, read as the object its fields dump to.
            # Validated as it is, a model iterates into its field pairs.
            default = default.model_dump(mode="json")
        try:
            defaults[name] = adapter.validate_python(default)
        except ValidationError:
            continue
    return defaults


def _prerequisites_of(cls: type) -> tuple[Prerequisite, ...]:
    """The prerequisites *cls* declares, empty for a feature the base install runs."""
    declared: object = getattr(cls, "prerequisites", ())
    if not is_object_tuple(declared):
        return ()
    return tuple(item for item in declared if isinstance(item, Prerequisite))


def _keeps_track_frame(provides: Provides | None) -> bool | None:
    """Whether a declared output keeps the track frame.

    ``None`` when undeclared, and when its payload families may include ``X``
    and ``Y``.
    """
    if provides is None:
        return None
    if provides.passthrough:
        return True
    written = {ref.name for ref in provides.adds if isinstance(ref, Column)}
    if {COLUMNS.x_col, COLUMNS.y_col} <= written:
        return True
    return None if provides.payload_families else False


def _feature_declaration(cls: type) -> Declaration:
    """Read one feature class as a producer and a consumer."""
    from mosaic.core.pipeline.loading import (
        CROSS_JOIN_FEATURES,
        expands_individuals,
        stacks_inputs,
    )

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
    requires = declared_requires(cls)
    provides = declared_provides(cls)
    declared_gpu_when: object = getattr(cls, "gpu_when", None)
    gpu_when = declared_gpu_when if isinstance(declared_gpu_when, GpuWhen) else None
    default_fields = contract_fields(requires, provides)
    if gpu_when is not None:
        default_fields |= {gpu_when.field}

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
            keeps_track_frame=_keeps_track_frame(provides),
        ),
        consumes=ConsumerDecl(
            name=slug,
            kind="feature",
            accepts_tracks=accepts_tracks,
            accepts_features=accepts_features,
            requires_track_shape=bool(getattr(inputs_cls, "_track_input", False)),
            takes_no_inputs=require == "empty",
            cross_joins=slug in CROSS_JOIN_FEATURES,
            expands_individuals=expands_individuals(cls),
            stacks_inputs=stacks_inputs(cls),
            reads_media="media_raw" in roots,
        ),
        emits=emits,
        category=str(getattr(cls, "category", "")),
        # Optional on a feature and defaulted to cpu, which is what every
        # feature but a declared-heavy one is.
        resource_class=str(getattr(cls, "resource_class", "") or "cpu"),
        gpu_when=gpu_when,
        requires=requires,
        provides=provides,
        param_defaults=_param_defaults(cls, default_fields),
        costly_fit=getattr(cls, "costly_fit", False) is True,
        fits=getattr(cls, "fits", False) is True,
        follows_ids=getattr(cls, "follows_ids", True) is True,
        track_spaces=tuple(sorted(feature_track_spaces(cls))),
        temporal_reach=class_reach(cls),
        temporal_reach_by_option=tuple(class_reach_by_option(cls).items()),
        pair_rule_column=_pair_rule_column(cls),
        unavailable=unmet(_prerequisites_of(cls)),
    )


def _pair_rule_column(cls: type) -> str | None:
    """The params field that names the column a feature's ``pairs`` rule is read of."""
    declared: object = getattr(cls, "pair_rule_column", None)
    return declared if isinstance(declared, str) and declared else None


def op_writes_tracks(kind: str, op_cls: type[Op[Params]]) -> bool:
    """Whether the op *kind* writes a tracks variant.

    The op's own ``writes_tracks`` declaration, falling back to membership of
    the tracking-roots registry for an op that declares none.
    """
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

    declared = getattr(op_cls, "writes_tracks", None)
    return kind in TRACKING_ROOTS if declared is None else bool(declared)


def _op_declaration(kind: str, op_cls: type[Op[Params]]) -> Declaration:
    """Read one op as a producer and a consumer.

    An op is never a feature's *input* in the ``inputs`` sense -- what a feature
    reads from a tracker is the ``tracks/`` variant it produced, referenced from
    the step's ``tracks`` field -- so its producer record exists to say that, and
    to carry the two facts an ordering-only edge is checked against.

    ``writes_tracks`` is the op's own declaration, falling back to membership of
    the tracking-roots registry -- the table a producer must appear in to *bridge
    from a tracker run root*, which is what every tracks producer did until one
    arrived that reads a tracks table and writes another. ``writes_media`` comes
    from the op's declared domain. A media op of a new category writes media
    without an entry here. Both come from declarations instead of name lists.

    ``reads_media`` stays tied to the registry rather than following
    ``writes_tracks``: it is true because bridging into tracks means opening the
    entry's video, which is exactly what a tracks-to-tracks producer does not do.
    """
    from mosaic.core.pipeline.ops import op_resource_class
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

    bridges_tracks = kind in TRACKING_ROOTS
    return Declaration(
        produces=ProducerDecl(
            name=kind,
            kind="op",
            level="individual",
            writes_tracks=op_writes_tracks(kind, op_cls),
            writes_media=str(getattr(op_cls, "domain", "")) == "media",
            writes_labels=bool(getattr(op_cls, "writes_labels", False)),
        ),
        consumes=ConsumerDecl(
            name=kind,
            kind="op",
            # An op reads a dataset root or a named artifact rather than a
            # pipeline input, and nothing may be wired into it. What it depends
            # on is expressed as a params reference or as an ordering-only edge.
            takes_no_inputs=True,
            # Every producer that bridges into tracks opens video to do it, which
            # is what makes an ordering-only edge from a media writer meaningful.
            reads_media=bridges_tracks,
        ),
        emits="individual",
        category=str(getattr(op_cls, "category", "")),
        resource_class=op_resource_class(kind),
        # ``register_op`` refuses by name a class that declares neither of
        # these. Every member of ``OPS`` therefore defines both.
        scope_takes=op_cls.scope_takes,
        scope_dependent=op_cls.scope_dependent,
    )


@lru_cache(maxsize=1)
def builtin_declaration_catalog() -> DeclarationCatalog:
    """Every built-in step type this installation knows, as declarations.

    Cached for the process, which is safe because both registries are filled by
    import side effect and are fixed once the modules that populate them have
    run. A feature's prerequisites are evaluated once as well. A package
    installed or a variable set after the first call is seen by the next process.

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

    tracks = replace(TRACKS_DECLARATION, requires=(), provides=tracks_provides())
    entries: dict[str, Declaration] = {TRACKS_INPUT: tracks}
    for cls in FEATURES.values():
        declared = _feature_declaration(cls)
        entries[declared.produces.name] = declared
    for kind, op_cls in OPS.items():
        entries[kind] = _op_declaration(kind, op_cls)
    return DeclarationCatalog(entries=entries)


def declaration_catalog() -> DeclarationCatalog:
    """Every step type this installation knows here, as declarations.

    The built-in catalog (:func:`builtin_declaration_catalog`) joined with the
    definitions of the scope in force, each listed as ``def:<name>`` under the
    version its bare slug resolves to. Without a scope, and under a disabled one,
    it is the built-in catalog. The join is assembled on each call from the
    dataset's listing and costs no source import. A definition added while a
    process runs appears at its next call. A definition whose files cannot be read
    is left out, and :meth:`DeclarationCatalog.get` of its slug raises the error.
    """
    builtin = builtin_declaration_catalog()
    usable = [listed for listed in _listed_definitions() if listed.usable]
    if not usable:
        return builtin
    return DeclarationCatalog(
        entries={
            **builtin.entries,
            **{listed.entry.name: listed.declaration for listed in usable},
        }
    )


def definition_declaration(slug: str) -> Declaration | None:
    """The declaration of the version a ``def:`` slug names in the scope in force.

    ``None`` under no scope, under a disabled one, and for a slug that matches no
    version. Derived from the version's sidecar and cached with its declared
    class, without a source import.

    Raises:
        DefinitionDigestAmbiguous: A prefix matches more than one version.
        DefinitionCorrupt: A stored file is not what its name says.
        DefinitionToolkitModelMissing: The definition names a toolkit model this
            installation lacks.
    """
    sidecar = _scope_sidecar(slug)
    return None if sidecar is None else _definition_entry(sidecar).declaration


def _scope_sidecar(slug: str) -> DefinitionSidecar | None:
    """The sidecar a ``def:`` slug names in the scope in force, ``None`` when none."""
    from mosaic.core.pipeline.definitions.scope import definition_sidecar_for_slug
    from mosaic.core.pipeline.definitions.store import FeatureDefinitionsNotAllowed

    try:
        return definition_sidecar_for_slug(slug)
    except FeatureDefinitionsNotAllowed:
        return None


@dataclass(frozen=True, slots=True)
class _DefinitionEntry:
    """What the catalogs derive from one definition version, once."""

    declaration: Declaration
    entry: CatalogEntry
    usable: bool = True
    """``False`` for a stand-in that lists a definition whose files cannot be read."""


_UNUSABLE: Final = (
    DefinitionCorrupt,
    DefinitionInvalid,
    DefinitionToolkitModelMissing,
)
"""The refusals that make one definition unreadable and no other."""


def _listed_definitions() -> list[_DefinitionEntry]:
    """The entry of each definition the scope in force lists, in name order.

    One definition that cannot be read is listed as an unavailable stand-in. Every
    other definition and every built-in step type stays readable.
    """
    from mosaic.core.pipeline.definitions.scope import scope_definition_names

    listed: list[_DefinitionEntry] = []
    for name in scope_definition_names():
        try:
            sidecar = _scope_sidecar(f"def:{name}")
            if sidecar is not None:
                listed.append(_definition_entry(sidecar))
        except _UNUSABLE as exc:
            listed.append(_unusable_entry(name, exc))
    return listed


def _unusable_entry(name: str, error: Exception) -> _DefinitionEntry:
    """The stand-in that lists the definition *name*, whose files raised *error*.

    It takes no inputs and carries the reason as its ``unavailable`` sentence, so
    validation reports a step that runs it and an attach candidate is refused with
    it.
    """
    slug = f"def:{name}"
    sentence = f"the feature definition {name!r} cannot be used: {error}"
    declaration = Declaration(
        produces=ProducerDecl(name=slug, kind="feature", level="individual"),
        consumes=ConsumerDecl(name=slug, kind="feature", takes_no_inputs=True),
        emits="individual",
        unavailable=sentence,
    )
    entry = CatalogEntry(
        name=slug,
        type="feature",
        version="",
        category="",
        summary="",
        declaration=declaration,
        params_schema=None,
        reference_fields={},
        requires=None,
        provides=None,
        unavailable=sentence,
        definition=DefinitionWire(name=name),
    )
    return _DefinitionEntry(declaration, entry, usable=False)


_DEFINITION_ENTRIES: Final[WeakKeyDictionary[type, _DefinitionEntry]] = (
    WeakKeyDictionary()
)
"""Each derived entry, by the declared class it was derived from.

A declared class is cached by digest and replaced when the definition caches are
cleared, which leaves the entry derived from the old class behind with it.
"""
_ENTRIES_LOCK: Final = threading.Lock()


def _definition_entry(sidecar: DefinitionSidecar) -> _DefinitionEntry:
    """The declaration and the editor-catalog entry of a definition version.

    Raises:
        DefinitionInvalid: A toolkit model the sidecar names is not one.
        DefinitionToolkitModelMissing: This installation lacks a toolkit model the
            sidecar names.
    """
    from mosaic.core.pipeline.definitions.scope import declared_class

    cls = declared_class(sidecar)
    with _ENTRIES_LOCK:
        cached = _DEFINITION_ENTRIES.get(cls)
    if cached is not None:
        return cached
    declaration = _feature_declaration(cls)
    entry = _entry_of_class(
        f"def:{sidecar.name}",
        declaration,
        cls,
        version=sidecar.version,
        definition=DefinitionWire(
            name=sidecar.name, digest=sidecar.digest, defined_on=sidecar.defined_on
        ),
    )
    with _ENTRIES_LOCK:
        return _DEFINITION_ENTRIES.setdefault(cls, _DefinitionEntry(declaration, entry))


_TRACKS_SUMMARY: Final = (
    "The dataset's standardized tracks, which every chain starts from."
)
"""What the editor catalog says of ``tracks``, which has no class to read."""


@lru_cache(maxsize=1)
def builtin_editor_catalog() -> EditorCatalog:
    """Every built-in step type a recipe may run, with what an editor draws it from.

    The declarations of :func:`builtin_declaration_catalog`, each joined to its
    params JSON Schema, its version, a one-line summary, and the params fields
    that accept a step reference. Kinds a graph may not run are left out, and an
    editor offers only steps that validate. Cached for the process for the same
    reason the declaration catalog is: both registries are fixed once their
    modules have been imported.
    """
    from mosaic.version import installed_version

    from .model import SCHEMA_VERSION
    from .validate import EXCLUDED_KINDS

    declarations = builtin_declaration_catalog()
    entries: list[CatalogEntry] = []
    for name in declarations.names():
        declared = declarations.get(name)
        if declared is None or name in EXCLUDED_KINDS:
            continue
        entries.append(_catalog_entry(name, declared))
    return EditorCatalog(
        schema_version=SCHEMA_VERSION,
        mosaic_version=installed_version(),
        entries=tuple(entries),
    )


def editor_catalog() -> EditorCatalog:
    """Every step type a recipe may run here, with what an editor draws it from.

    The built-in catalog (:func:`builtin_editor_catalog`) joined with the
    definitions of the scope in force, sorted by name. A definition's entry is
    derived from its sidecar by this installation and carries its
    ``definition``. A definition whose files cannot be read is listed as an
    unavailable entry that names the reason. Without a scope, and under a disabled
    one, it is the built-in catalog.
    """
    builtin = builtin_editor_catalog()
    listed = _listed_definitions()
    if not listed:
        return builtin
    entries = [*builtin.entries, *(each.entry for each in listed)]
    return builtin.model_copy(update={"entries": tuple(sorted(entries, key=_by_name))})


def _by_name(entry: CatalogEntry) -> str:
    return entry.name


def _column_wires(
    declared: Declaration,
) -> tuple[tuple[ColumnNames, ...] | None, ProvidedColumns | None]:
    """The columns *declared* reads and writes under its default params, as wires."""
    requires = (
        None
        if declared.requires is None
        else requires_wire(declared.requires, declared.param_defaults)
    )
    provides = (
        None
        if declared.provides is None
        else provided_wire(declared.provides, declared.param_defaults)
    )
    return requires, provides


def _catalog_entry(name: str, declared: Declaration) -> CatalogEntry:
    """One editor-catalog entry, read off the class behind *name*."""
    requires, provides = _column_wires(declared)
    kind = declared.produces.kind
    if kind == "tracks":
        return CatalogEntry(
            name=name,
            type="tracks",
            version="",
            category="",
            summary=_TRACKS_SUMMARY,
            declaration=declared,
            params_schema=None,
            reference_fields={},
            requires=requires,
            provides=provides,
            unavailable=declared.unavailable,
        )
    cls = feature_class_for_slug(name) if kind == "feature" else op_class_for_kind(name)
    return _entry_of_class(
        name, declared, cls, version=declared_version(kind, name), definition=None
    )


def _entry_of_class(
    name: str,
    declared: Declaration,
    cls: type | None,
    *,
    version: str,
    definition: DefinitionWire | None,
) -> CatalogEntry:
    """The editor-catalog entry of a feature or op class, read off *cls*."""
    requires, provides = _column_wires(declared)
    prerequisites = () if cls is None else _prerequisites_of(cls)
    params_cls: object = getattr(cls, "Params", None)
    schema: dict[str, JsonValue] | None = None
    if isinstance(params_cls, type) and issubclass(params_cls, BaseModel):
        schema = params_cls.model_json_schema()
    reference_fields: dict[str, ReferenceKind] = {}
    field_types = _field_types_of(getattr(cls, "Params", None))
    for field_name, annotation in field_types.items():
        site = _reference_site_of(annotation)
        if site == "result" or site == "identifier" or site == "labels":
            reference_fields[field_name] = site
    ground_truth_fields = tuple(
        sorted(
            field_name
            for field_name, annotation in field_types.items()
            if _accepts_ground_truth(annotation)
        )
    )
    return CatalogEntry(
        name=name,
        type=declared.produces.kind,
        version=version,
        category=declared.category,
        summary=first_docstring_line(cls),
        declaration=declared,
        params_schema=schema,
        reference_fields=reference_fields,
        requires=requires,
        provides=provides,
        unavailable=declared.unavailable,
        license_note=license_notes(prerequisites),
        keywords=_keywords_of(cls),
        ground_truth_fields=ground_truth_fields,
        needs_ground_truth=fields_need_ground_truth(field_types),
        definition=definition,
    )


def _keywords_of(cls: type | None) -> tuple[str, ...]:
    """The search keywords a feature or op class declares, ``()`` when none."""
    declared: object = getattr(cls, "keywords", ())
    if not is_object_tuple(declared):
        return ()
    return tuple(phrase for phrase in declared if isinstance(phrase, str))


def _field_forms(annotation: object) -> tuple[object, ...]:
    """The types a field declared as *annotation* takes, ``None`` left out."""
    args: tuple[object, ...] = getattr(annotation, "__args__", ())
    candidates = args if args else (annotation,)
    return tuple(candidate for candidate in candidates if candidate is not type(None))


def _accepts_ground_truth(annotation: object) -> bool:
    """Whether a field declared as *annotation* takes a ``GroundTruthLabelsSource``.

    The class itself. Its subclass ``PredictionsLabelsSource`` reads the
    ``predictions`` kind, a classifier's output.
    """
    from mosaic.core.pipeline.types.labels import GroundTruthLabelsSource

    return any(form is GroundTruthLabelsSource for form in _field_forms(annotation))


def fields_need_ground_truth(field_types: Mapping[str, object]) -> bool:
    """Whether a step type with these params fields depends on ground-truth labels.

    True when a field takes the ``GroundTruthLabelsSource`` form and no other and
    is required, or when a field takes labeled templates (``LabeledTemplatesRef``),
    which only a run over ground-truth labels writes. A field that also accepts
    ``None`` leaves the step runnable without ground truth, as ``event-response``
    does with its other onset sources, so it does not make the step dependent.
    """
    from mosaic.core.pipeline.types.artifacts import LabeledTemplatesRef

    for annotation in field_types.values():
        forms = _field_forms(annotation)
        optional = type(None) in getattr(annotation, "__args__", ())
        if (
            not optional
            and _accepts_ground_truth(annotation)
            and all(_accepts_ground_truth(form) for form in forms)
        ):
            return True
        if any(
            isinstance(form, type) and issubclass(form, LabeledTemplatesRef)
            for form in forms
        ):
            return True
    return False


class StepTypeNotFound(ClassifiedError, LookupError):
    """A name that no step type of the editor catalog has."""

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, name: str) -> None:
        super().__init__(f"no step type named {name!r} in the catalog")
        self.name: str = name

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name that was looked up."""
        return {"name": self.name}


def describe_step_type(name: str) -> CatalogEntry:
    """The editor-catalog entry of the step type *name*, params schema included.

    Raises:
        StepTypeNotFound: When the catalog has no step type of that name, which
            includes an op that a recipe may not run.
    """
    for entry in editor_catalog().entries:
        if entry.name == name:
            return entry
    raise StepTypeNotFound(name)


def catalog_index() -> CatalogIndex:
    """Every step type of :func:`editor_catalog`, summarized without params schemas.

    A row per entry, in the catalog's order. The full catalog dumps to about
    500 KB of JSON, almost all of it params schemas, and
    :func:`describe_step_type` returns one entry whole.
    """
    catalog = editor_catalog()
    return CatalogIndex(
        schema_version=catalog.schema_version,
        mosaic_version=catalog.mosaic_version,
        entries=tuple(_index_entry(entry) for entry in catalog.entries),
    )


def step_type_detail(
    name: str, *, detail: Literal["fields", "schema"] = "fields"
) -> StepTypeDetail:
    """The step type *name* in detail: its index row, what it reads, its params.

    The params are a flat table (:func:`~.param_table.param_rows`), a few
    kilobytes where the whole JSON Schema of :func:`describe_step_type` can run
    to tens. A field that takes a run reference is typed as a recipe writes
    one.

    Args:
        name: A feature slug, an op kind, or ``tracks``.
        detail: ``schema`` also returns the params JSON Schema, without titles
            or class docstrings.

    Raises:
        StepTypeNotFound: When the catalog has no step type of that name.
    """
    from .param_table import compact_params_schema, param_rows

    entry = describe_step_type(name)
    declared = entry.declaration
    consumes = declared.consumes
    labels = (
        ("tracks", consumes.accepts_tracks),
        ("features", consumes.accepts_features),
        ("media", consumes.reads_media),
        ("nothing", consumes.takes_no_inputs),
    )
    scope_dependent = declared.scope_dependent
    if entry.definition is not None:
        sidecar = _entry_sidecar(entry.definition)
        scope_dependent = sidecar is not None and sidecar.flags.scope_dependent
    elif entry.type == "feature":
        # A feature's declaration does not carry it; the planner reads the class.
        cls = feature_class_for_slug(name)
        scope_dependent = bool(getattr(cls, "scope_dependent", False))
    schema = entry.params_schema
    return StepTypeDetail(
        **_index_entry(entry).model_dump(),
        consumes=tuple(label for label, holds in labels if holds),
        resource_class=declared.resource_class,
        scope_dependent=scope_dependent,
        fits=declared.fits,
        follows_ids=declared.follows_ids,
        temporal_reach=declared.temporal_reach,
        temporal_reach_by_option=dict(declared.temporal_reach_by_option),
        params=() if schema is None else param_rows(schema, entry.reference_fields),
        params_schema=(
            compact_params_schema(schema)
            if detail == "schema" and schema is not None
            else None
        ),
    )


def _index_entry(entry: CatalogEntry) -> CatalogIndexEntry:
    """One catalog entry as its index row."""
    declared = entry.declaration
    provides = entry.provides
    return CatalogIndexEntry(
        name=entry.name,
        type=entry.type,
        version=entry.version,
        category=entry.category,
        summary=entry.summary,
        lane=step_type_lane(entry),
        emits=declared.emits,
        requires=None
        if entry.requires is None
        else tuple(names.label for names in entry.requires),
        provides=None
        if provides is None
        else tuple(names.label for names in provides.adds),
        passthrough=provides is not None and provides.passthrough,
        payload_families=provides is not None and provides.payload_families,
        unavailable=entry.unavailable,
        license_note=entry.license_note,
        parallelizable=_parallelizable(entry),
        definition=entry.definition,
        keywords=entry.keywords,
        track_spaces=declared.track_spaces,
    )


def _entry_sidecar(definition: DefinitionWire) -> DefinitionSidecar | None:
    """The sidecar of the version *definition* names, read in the scope in force.

    ``None`` when its files cannot be read.
    """
    slug = f"def:{definition.name}"
    if definition.digest is not None:
        slug = f"{slug}@{definition.digest}"
    try:
        return _scope_sidecar(slug)
    except _UNUSABLE:
        return None


def _parallelizable(entry: CatalogEntry) -> bool:
    """Whether a run of *entry* can compute several entries at once.

    A definition answers from its sidecar's flags without importing its source.
    """
    if entry.definition is not None:
        sidecar = _entry_sidecar(entry.definition)
        return sidecar is not None and sidecar.flags.parallelizable
    if entry.type == "feature":
        return feature_parallelizable(entry.name)
    return op_parallelizable(entry.name)


def step_type_lane(entry: CatalogEntry) -> str | None:
    """The lane *entry*'s work is offered to, or ``None`` for ``tracks``, which runs nothing."""
    return None if entry.type == "tracks" else lane_for(entry.declaration)


def first_docstring_line(cls: object) -> str:
    """The summary line of *cls*'s docstring, or ``""`` when it has none."""
    lines = (inspect.getdoc(cls) or "").strip().splitlines()
    return lines[0] if lines else ""


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
        labels_variant: What names a label conversion's variant, empty
            otherwise.
        labels_kind: The label kind that variant is of, empty otherwise.
    """

    step_id: str
    storage_name: str = ""
    run_id: str | None = None
    tracks_variant: str = ""
    model_run_id: str = ""
    labels_variant: str = ""
    labels_kind: str = ""


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
    labels_run_id: str | None = None
    """The labels variant a labels reference pins, or ``None`` to resolve one."""
    entries: Scope = field(default_factory=Scope)
    """What this step should compute, already narrowed.

    A step must not re-request its whole scope: it is the scope minus what is
    covered and minus what is quarantined. An unset selector means everything in
    scope, which is what a first run of a cold step gets.
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


def _declaring_class(slug: str) -> type | None:
    """The class that declares feature *slug*'s params, or ``None``.

    A ``def:`` slug names the class declared from its sidecar whatever the scope
    loads. A read of the declared fields leaves the source unimported.
    """
    from mosaic.core.pipeline.definitions.store import is_definition_slug

    if not is_definition_slug(slug):
        return feature_class_for_slug(slug)
    from mosaic.core.pipeline.definitions.scope import declared_class

    sidecar = _scope_sidecar(slug)
    return None if sidecar is None else declared_class(sidecar)


def _params_field_types(kind: str, name: str) -> dict[str, object]:
    """The declared type of each top-level params field of a step.

    What decides how a reference is substituted. Read from the model rather than
    from a label the recipe author wrote, so a reference lands in the shape its
    consumer actually declares -- an ``ArtifactSpec`` field gets a reference with
    a pattern, a ``str`` field gets a bare identifier, and neither is a choice
    the file gets to make.
    """
    if kind == "feature":
        cls = _declaring_class(name)
        params_cls = getattr(cls, "Params", None) if cls is not None else None
    else:
        op_cls = op_class_for_kind(name)
        params_cls = getattr(op_cls, "Params", None) if op_cls is not None else None
    return _field_types_of(params_cls)


def _field_types_of(params_cls: object) -> dict[str, object]:
    """The declared type of each top-level field of the params model *params_cls*."""
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


def _is_labels_field(annotation: object) -> bool:
    """Does this declared type hold a labels source, which names a label kind?"""
    from mosaic.core.pipeline.types.labels import LabelsSource

    candidates = [annotation, *getattr(annotation, "__args__", ())]
    return any(
        isinstance(candidate, type) and issubclass(candidate, LabelsSource)
        for candidate in candidates
    )


def _is_identifier_field(annotation: object) -> bool:
    """Does this declared type hold a bare identifier -- how a model is named?"""
    candidates = [annotation, *getattr(annotation, "__args__", ())]
    return any(candidate is str for candidate in candidates)


type ReferenceSite = Literal["result", "identifier", "labels", "unsupported", "unknown"]
"""What a params field can be substituted with, read off its declared type.

``result`` takes ``{feature, run_id}`` -- a ``Result`` or an ``ArtifactSpec``.
``identifier`` takes the bare upstream identifier, which is how a model is named.
``labels`` takes a label conversion step: the field gets its label kind, and the
step's run is pinned to the labels variant it writes.
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
    return _reference_site_of(types[field_name])


def run_reference_fields(name: str) -> frozenset[str]:
    """The top-level params fields of feature *name* that may name another run.

    A field typed as a ``Result`` or an ``ArtifactSpec`` names a feature run, and
    one typed as a bare identifier names a run or a model. A cleanup reads a run
    whose params are unknown as reading every run when its feature declares one
    (:mod:`~.cleanup`). Empty for a name the registry does not know.
    """
    return frozenset(
        field_name
        for field_name, annotation in _params_field_types("feature", name).items()
        if _reference_site_of(annotation) in ("result", "identifier")
    )


def _reference_site_of(annotation: object) -> ReferenceSite:
    """How a reference would substitute into a field declared as *annotation*."""
    if _is_labels_field(annotation):
        return "labels"
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
    entries: Scope | None = None,
    tracks_run_id: str | None = None,
    pairs: Literal["once"] | None = None,
) -> StepSpec:
    """Substitute every reference in one step from what is already resolved.

    Called in topological order, so every step this one names has been resolved
    before it is reached. Substitution is decided by the **declared field type**,
    never by a label in the file:

    - an ``inputs`` reference becomes ``{feature, run_id}`` naming the upstream's
      *storage* name, which is what a ``Result`` carries;
    - the step-level ``tracks`` reference becomes the upstream's tracks
      **variant**, which is a different identifier from its run id. A step
      reading the tracks without one reads *tracks_run_id*, the request's
      variant, and ``None`` leaves each entry's one variant;
    - a params field declared as a ``Result`` or ``ArtifactSpec`` becomes
      ``{feature, run_id}`` plus the reference's pattern when it gave one;
    - a params field declared as a plain ``str`` becomes the bare identifier,
      which is how a model is named;
    - a params field declared as a labels source becomes ``{"kind": ...}``
      naming the upstream's label kind, with the reference's ``remap`` and
      ``mask`` when it gives them, and the spec's ``labels_run_id`` pins the
      labels variant the upstream writes.

    Only top-level params fields are scanned, matching ``resolve_references`` and
    ``Params.identity_dump()``. The dump asks a nested model for its own dump, and
    no reference is found below the top level, so finding a reference deeper
    would promise a substitution nothing performs.

    A step that a submission binds to a run is resolved in *resolved* as that
    run, so a step reading it reads the pinned run. Its own spec is the one it
    would run, which is what the plan compares the bind's storage with.

    *pairs* is the ``once`` that a plan resolved an aggregate's ``auto`` pair
    rule to (:func:`~.validate.pair_rules`). It is written into the params of a
    step left at ``auto``, and a step that writes its own rule keeps it.

    A ``def:`` feature is written ``def:<name>@<sha256>``, the version it resolves
    to in the definition scope in force, and stores under its name alone.

    Raises:
        FeatureDefinitionsNotAllowed: A ``def:`` feature under a disabled scope.
        DefinitionNotFound: A ``def:`` feature that names no version.
        DefinitionDigestAmbiguous: A ``def:`` feature whose digest prefix more
            than one version starts with.
    """
    step = recipe.step(step_id)

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
            entries=entries if entries is not None else Scope(),
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
    if pairs is not None and params.get(PAIR_RULE_FIELD, "auto") == "auto":
        params[PAIR_RULE_FIELD] = pairs
    labels_run_id: str | None = None
    for name, reference in params_step_refs(step.params).items():
        upstream = resolved.get(reference.step)
        if upstream is None:
            continue
        if _is_labels_field(types.get(name)):
            params[name] = {
                "kind": upstream.labels_kind or "behavior",
                **reference.model_dump(
                    mode="json", include=set(LABEL_CONTROL_FIELDS), exclude_none=True
                ),
            }
            labels_run_id = upstream.labels_variant or None
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

    read_variant: str | None = None
    if step.tracks is not None:
        upstream = resolved.get(step.tracks.step)
        # The variant, never the op run id. They coincide today only because the
        # tracker variant payload is an unwrapped passthrough of the settings.
        read_variant = upstream.tracks_variant if upstream else None
    elif TRACKS_INPUT in step.inputs:
        read_variant = tracks_run_id

    return StepSpec(
        step_id=step_id,
        kind="feature",
        feature=_pinned_feature_slug(step.feature),
        inputs=tuple(inputs),
        params=params,
        tracks_run_id=read_variant,
        labels_run_id=labels_run_id,
        entries=entries if entries is not None else Scope(),
        filter_start_frame=step.run.filter_start_frame,
        filter_end_frame=step.run.filter_end_frame,
        filter_start_time=step.run.filter_start_time,
        filter_end_time=step.run.filter_end_time,
        overlap_frames=step.run.overlap_frames,
    )


def _pinned_feature_slug(slug: str) -> str:
    """*slug* naming one version: a ``def:`` slug with its full digest.

    A built-in slug is returned as it is. A ``def:`` slug resolves in the
    definition scope in force, and the plan records the version it hashed.

    Raises:
        FeatureDefinitionsNotAllowed: The scope is disabled.
        DefinitionNotFound: No version matches *slug*.
        DefinitionDigestAmbiguous: A prefix matches more than one version.
    """
    from mosaic.core.pipeline.definitions.store import is_definition_slug

    if not is_definition_slug(slug):
        return slug
    from mosaic.core.pipeline.definitions.scope import resolve_definition_slug

    name, digest = resolve_definition_slug(slug)
    return f"def:{name}@{digest}"


def toolkit_model_for_ref(ref: str, *, definition: str) -> type[DeclaredModel]:
    """The toolkit model that a definition's sidecar names by ``<module>:<class>``.

    A sidecar records a toolkit model by reference, and planning imports it here so that
    a definition's params are rebuilt from the toolkit's own class. Only a module under
    ``mosaic.`` is imported, and the definitions' ``mosaic_definitions.`` modules are
    excluded. A sidecar can name only the toolkit's code.

    Args:
        ref: ``<module>:<qualified class name>``.
        definition: The name of the definition whose sidecar holds *ref*, which
            each error names.

    Raises:
        DefinitionInvalid: *ref* names a module outside ``mosaic.``, or a class
            that is not a ``DeclaredModel``.
        DefinitionToolkitModelMissing: The module or the class does not exist in
            this installation.
    """
    import importlib

    from mosaic.core.params import DeclaredModel
    from mosaic.core.pipeline.definitions.store import (
        DefinitionFinding,
        DefinitionInvalid,
        DefinitionToolkitModelMissing,
    )

    def invalid(reason: str) -> DefinitionInvalid:
        finding = DefinitionFinding(
            check="params_type", field=None, message=f"the model {ref!r} {reason}"
        )
        return DefinitionInvalid(definition, [finding])

    module_name, separator, qualified_name = ref.partition(":")
    if not separator or not qualified_name:
        raise invalid("is not of the form <module>:<class>")
    if not module_name.startswith("mosaic."):
        raise invalid("is not a toolkit model, whose module is under mosaic.")
    try:
        found: object = importlib.import_module(module_name)
        for part in qualified_name.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError):
        raise DefinitionToolkitModelMissing(definition, ref) from None
    if not (isinstance(found, type) and issubclass(found, DeclaredModel)):
        raise invalid("is not a DeclaredModel")
    return found


def fill_builtin_registries() -> None:
    """Register every built-in feature, op, converter and track schema.

    Importing the feature library registers the features, ``register_ops`` the
    ops, and the two converter calls and the schema module the rest. A source
    that imports a module of the feature library, as a definition may, then
    finds every registry already full and adds nothing to it.
    """
    import importlib

    from mosaic.core.label_converter import ensure_label_converters_registered
    from mosaic.core.track_converter import ensure_track_converters_registered
    from mosaic.tracking import register_ops

    _ = importlib.import_module("mosaic.behavior.feature_library")
    register_ops()
    ensure_label_converters_registered()
    ensure_track_converters_registered()
    _ = importlib.import_module("mosaic.core.schema")


type RegistryName = Literal[
    "FEATURES", "OPS", "TRACK_CONVERTERS", "LABEL_CONVERTERS", "TRACK_SCHEMAS"
]
"""The process-global registries that a definition's source must leave unchanged."""

type RegistrySnapshot = dict[RegistryName, frozenset[str]]
"""The keys each registry holds, a label converter's as ``<format>:<kind>``."""


def registry_snapshot() -> RegistrySnapshot:
    """The keys every registry holds now, to compare with a later snapshot."""
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.label_converter import LABEL_CONVERTERS
    from mosaic.core.pipeline.ops import OPS
    from mosaic.core.schema import TRACK_SCHEMAS
    from mosaic.core.track_converter import TRACK_CONVERTERS

    return {
        "FEATURES": frozenset(FEATURES),
        "OPS": frozenset(OPS),
        "TRACK_CONVERTERS": frozenset(TRACK_CONVERTERS),
        "LABEL_CONVERTERS": frozenset(
            f"{source_format}:{label_kind}"
            for source_format, label_kind in LABEL_CONVERTERS
        ),
        "TRACK_SCHEMAS": frozenset(TRACK_SCHEMAS),
    }
