"""Declaring a feature definition: the checks its source must pass and its sidecar.

:func:`declare_source` checks a source's bytes, imports it with every registry
filled and refuses an import that changes a registry. :func:`declare_class` finds
the class that a definition runs in its imported module, checks it against the
feature protocol, the identity of a run and the params rules, and records the facts
its sidecar keeps. Each failed check is a
:class:`~mosaic.core.pipeline.definitions.store.DefinitionFinding`, and the class
is refused with every finding.

Run as ``python -m mosaic.core.pipeline.definitions.declare <name> <path>``, the
module declares the source at *path* and writes to its standard output the sidecar
JSON, or, exiting with status 1, the findings as ``{"findings": [...]}``. Whatever
the source prints goes to the standard error.

The registries and the toolkit models are reached through ``graph.resolve``, so
this module names no module of the feature library.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import (
    Annotated,
    Final,
    Literal,
    TypeAliasType,
    TypeIs,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import annotated_types
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from mosaic.core.json_object import JSON_OBJECT
from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_object_list, is_object_mapping, is_object_tuple
from mosaic.core.params import (
    Declared,
    DeclaredModel,
    HashExclude,
    NestedParams,
    Params,
    own_declaration,
)
from mosaic.core.pipeline._utils import canonical_json, identity_ready
from mosaic.core.pipeline.column_contract import (
    Provides,
    Requirement,
    declared_provides,
    declared_requires,
)
from mosaic.core.pipeline.definitions.sidecar import (
    SIDECAR_FORMAT,
    SIDECAR_FORMAT_VERSION,
    Attributes,
    Constraints,
    DefinitionSidecar,
    DictType,
    Flags,
    InputItemName,
    InputsRecord,
    ListType,
    LiteralType,
    ModelType,
    ParamRecord,
    PrerequisiteRecord,
    ScalarType,
    ToolkitModelLookup,
    ToolkitModelType,
    TupleType,
    TypeNode,
    UnionType,
    default_inputs,
    prerequisite_extra_findings,
    prerequisite_findings,
    rebuild_inputs_model,
    rebuild_params_model,
    sidecar_json,
)
from mosaic.core.pipeline.definitions.store import (
    DefinitionCheck,
    DefinitionFinding,
    DefinitionInvalid,
    check_definition_name,
    definition_module_name,
)
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.feature_levels import RELATES_LEVELS, EmitsLevel
from mosaic.core.pipeline.graph.compatibility import GpuWhen
from mosaic.core.pipeline.graph.resolve import (
    RegistrySnapshot,
    fill_builtin_registries,
    first_docstring_line,
    registry_snapshot,
    toolkit_model_for_ref,
)
from mosaic.core.pipeline.run import identity_terms
from mosaic.core.pipeline.track_space import feature_track_spaces
from mosaic.core.pipeline.types.inputs import Inputs, TrackInput
from mosaic.core.pipeline.types.labels import LabelsSource
from mosaic.core.pipeline.types.results import Result
from mosaic.core.strict_model import terse
from mosaic.optional_dependency import OptionalModule
from mosaic.version import installed_version

__all__ = [
    "SOURCE_MAX_BYTES",
    "declare_class",
    "declare_source",
    "definition_class",
    "main",
    "registry_findings",
]

SOURCE_MAX_BYTES: Final = 1024 * 1024
"""The most bytes a definition's source holds."""

_REFUSED_EXIT: Final = 1

_FLAGS_IN_INIT: Final = (
    "name",
    "version",
    "parallelizable",
    "scope_dependent",
    "accepts_overlap",
    "consumed_roots",
    "emits",
    "fits",
    "costly_fit",
)
"""The class constants an instance must not hold in its own ``__dict__``."""

_METHODS: Final = ("load_state", "fit", "save_state", "apply")
_BOOLEAN_FLAGS: Final = ("parallelizable", "scope_dependent", "accepts_overlap")
_BOOLEAN_ATTRIBUTES: Final = ("fits", "costly_fit", "stacks_inputs")
_TEXT_ATTRIBUTES: Final = ("report_file", "resource_class")
_INPUTS_DECLARATIONS: Final = frozenset({"_require", "_track_input"})
_INPUT_REQUIRES: Final = ("nonempty", "empty", "any")

_MODEL_MARKER: Final = "$model"
_TOOLKIT_PREFIX: Final = "mosaic."
_ACCEPTED_FORM: Final = "Annotated[list[str] | None, Declared(...)]"

_ALIAS_ATTRIBUTES: Final = (
    "alias",
    "validation_alias",
    "serialization_alias",
    "discriminator",
)
_REFUSED_ATTRIBUTES: Final = (
    "description",
    "title",
    "examples",
    "json_schema_extra",
    "exclude",
    "exclude_if",
    "frozen",
    "validate_default",
    "deprecated",
    "field_title_generator",
    "init",
    "init_var",
    "kw_only",
)
"""``FieldInfo`` attributes that change validation, the dump or the schema."""

_BOUNDS: Final = (
    (annotated_types.Gt, "gt"),
    (annotated_types.Ge, "ge"),
    (annotated_types.Lt, "lt"),
    (annotated_types.Le, "le"),
    (annotated_types.MultipleOf, "multiple_of"),
    (annotated_types.MinLen, "min_length"),
    (annotated_types.MaxLen, "max_length"),
)
"""Each recorded ``annotated_types`` constraint and the attribute that holds it."""

_PATTERN_PROBE: Final[list[object]] = FieldInfo.from_field(pattern=".").metadata
_GENERAL_METADATA: Final = type(_PATTERN_PROBE[0])
"""Pydantic's own metadata class, which holds ``pattern`` among other settings."""

_SCALARS: Final = (
    (bool, ScalarType(kind="bool")),
    (int, ScalarType(kind="int")),
    (float, ScalarType(kind="float")),
    (str, ScalarType(kind="str")),
    (None, ScalarType(kind="null")),
    (type(None), ScalarType(kind="null")),
)

_REFERENCE_TYPES: Final[tuple[type, ...]] = (Result, LabelsSource)
"""The types of a field that names another run, which a definition cannot hold."""

_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_REQUIREMENT: Final[TypeAdapter[Requirement]] = TypeAdapter(Requirement)
_PROVIDES: Final[TypeAdapter[Provides]] = TypeAdapter(Provides)


def declare_source(source: bytes, name: str) -> DefinitionSidecar:
    """Check *source* as the definition *name*, and return its sidecar.

    The source must be UTF-8 of at most :data:`SOURCE_MAX_BYTES`. Every registry is
    filled with the built-ins before the source is imported, from a temporary file,
    under the module name it has when it runs. Neither the import nor the checks
    that build the class may change a registry. The module is removed from
    ``sys.modules`` afterwards. The sidecar's ``defined_on`` is today's UTC date.

    Raises:
        DefinitionNameInvalid: *name* does not match the grammar.
        DefinitionInvalid: The source or its class fails a check, with every
            finding.
    """
    check_definition_name(name)
    if len(source) > SOURCE_MAX_BYTES:
        raise DefinitionInvalid.of_source(
            name,
            f"the source is {len(source)} bytes, more than the {SOURCE_MAX_BYTES} "
            "a definition may hold",
        )
    try:
        _ = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DefinitionInvalid.of_source(
            name, f"the source is not UTF-8: {exc}"
        ) from None
    digest = hashlib.sha256(source).hexdigest()
    defined_on = datetime.now(UTC).date()
    module_name = definition_module_name(name, digest)
    fill_builtin_registries()
    before = registry_snapshot()
    with tempfile.TemporaryDirectory(prefix="mosaic-definition-") as directory:
        path = Path(directory) / f"{module_name.rpartition('.')[2]}.py"
        _ = path.write_bytes(source)
        try:
            module = _import_source(path, module_name, name)
            changed = registry_findings(before, registry_snapshot())
            if changed:
                raise DefinitionInvalid(name, changed)
            try:
                sidecar = declare_class(
                    module,
                    name,
                    digest,
                    defined_on,
                    partial(toolkit_model_for_ref, definition=name),
                )
            except DefinitionInvalid as exc:
                changed = registry_findings(before, registry_snapshot())
                raise DefinitionInvalid(name, [*exc.findings, *changed]) from None
            changed = registry_findings(before, registry_snapshot())
            if changed:
                raise DefinitionInvalid(name, changed)
            return sidecar
        finally:
            _ = sys.modules.pop(module_name, None)


def _import_source(path: Path, module_name: str, name: str) -> types.ModuleType:
    """Import the file at *path* as *module_name*.

    Raises:
        DefinitionInvalid: Importing the source raised, or called ``sys.exit``.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DefinitionInvalid.of_source(name, f"{path} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as exc:
        raise DefinitionInvalid.of_source(
            name, f"importing the source raised {type(exc).__name__}: {exc}"
        ) from None
    return module


def registry_findings(
    before: RegistrySnapshot, after: RegistrySnapshot
) -> list[DefinitionFinding]:
    """A ``registry`` finding for each registry whose keys differ between snapshots.

    Each names the registry as its field and the keys that were added or removed.
    """
    findings: list[DefinitionFinding] = []
    for registry, keys in before.items():
        added = sorted(after[registry] - keys)
        removed = sorted(keys - after[registry])
        if not (added or removed):
            continue
        changes = [
            f"{label} {', '.join(names)}"
            for label, names in (("added", added), ("removed", removed))
            if names
        ]
        findings.append(
            DefinitionFinding(
                "registry",
                registry,
                f"importing the source changed {registry}: {'; '.join(changes)}. A "
                "definition does not call register_feature or another registrar",
            )
        )
    return findings


def main(argv: Sequence[str]) -> int:
    """Declare the source at ``argv[1]`` as the definition ``argv[0]``.

    Writes the sidecar JSON to the standard output and returns 0, or writes
    ``{"findings": [...]}`` and returns 1 when the source is refused. The standard
    output is moved to the standard error while the source runs, so nothing the
    source prints reaches the result.
    """
    name, path = argv
    sys.stdout.flush()
    result = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    _ = os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with result:
        try:
            sidecar = declare_source(Path(path).read_bytes(), name)
        except DefinitionInvalid as exc:
            findings = [finding.as_json() for finding in exc.findings]
            _ = result.write(json.dumps({"findings": findings}).encode("utf-8"))
            return _REFUSED_EXIT
        _ = result.write(sidecar_json(sidecar))
    return 0


def declare_class(
    module: types.ModuleType,
    name: str,
    digest: str,
    defined_on: date,
    toolkit_model: ToolkitModelLookup,
) -> DefinitionSidecar:
    """Check the class that *module* defines for the definition *name*, and record it.

    The module defines exactly one class whose ``name`` is *name*. Its flags,
    attributes, ``Inputs`` and ``Params`` are checked and recorded, each recorded
    default is rebuilt and compared with the source's, and the class is built
    with its default inputs and params.

    Args:
        module: The imported source.
        name: The definition's name.
        digest: The SHA-256 of the source.
        defined_on: The UTC date of the version's first definition.
        toolkit_model: Returns the toolkit model that a ``ref`` names.

    Raises:
        DefinitionInvalid: The class fails a check, with every finding.
    """
    cls = definition_class(module, name)
    findings = _Findings()
    version = _version(cls, findings)
    flags = _flags(cls, findings)
    attributes = _attributes(cls, findings)
    prerequisites = _prerequisites(cls, findings)
    inputs = _inputs(cls, findings)
    params_model, params = _ParamsWalker(findings).params(cls)
    if findings.found or params_model is None or inputs is None:
        raise DefinitionInvalid(name, findings.found)
    flags_record = _record_or_findings(Flags, flags, findings)
    attributes_record = _record_or_findings(Attributes, attributes, findings)
    if flags_record is None or attributes_record is None:
        raise DefinitionInvalid(name, findings.found)

    sidecar = DefinitionSidecar(
        format=SIDECAR_FORMAT,
        format_version=SIDECAR_FORMAT_VERSION,
        name=name,
        digest=digest,
        defined_on=defined_on,
        mosaic_version=installed_version(),
        class_name=cls.__name__,
        version=version,
        summary=first_docstring_line(cls),
        flags=flags_record,
        inputs=inputs,
        params=params,
        attributes=attributes_record,
        prerequisites=prerequisites,
    )
    findings.found.extend(prerequisite_findings(sidecar))
    findings.found.extend(prerequisite_extra_findings(sidecar))
    _rebuild_findings(sidecar, params_model, toolkit_model, findings)
    if not findings.found:
        _construction_findings(cls, sidecar, toolkit_model, findings)
    if findings.found:
        raise DefinitionInvalid(name, findings.found)
    return sidecar


def _record_or_findings[R: BaseModel](
    record: type[R], values: dict[str, object], findings: _Findings
) -> R | None:
    """*values* validated as *record*, or ``None`` with an ``attribute`` finding each.

    A value can pass its own check and still not fit the record, such as a
    ``gpu_when`` value that is not JSON.
    """
    try:
        return record.model_validate(values)
    except ValidationError as exc:
        fields = sorted(
            {str(error["loc"][0]) for error in exc.errors() if error["loc"]}
        )
        for field in fields:
            findings.add(
                "attribute",
                field,
                f"{field} cannot be recorded: {terse(exc)}",
            )
        return None


class _Findings:
    """The findings of one declaration, in the order the checks make them."""

    def __init__(self) -> None:
        self.found: list[DefinitionFinding] = []

    def add(self, check: DefinitionCheck, field: str | None, message: str) -> None:
        self.found.append(DefinitionFinding(check=check, field=field, message=message))


def definition_class(module: types.ModuleType, name: str) -> type:
    """The one class in *module* whose ``name`` is *name*.

    Raises:
        DefinitionInvalid: The module defines none, or more than one.
    """
    found: list[type] = []
    for value in vars(module).values():
        declared: object = getattr(value, "name", None)
        if isinstance(value, type) and declared == name and value not in found:
            found.append(value)
    if len(found) == 1:
        return found[0]
    names = ", ".join(cls.__name__ for cls in found)
    message = (
        f"the source defines no class whose name is {name!r}"
        if not found
        else f"the source defines {len(found)} classes whose name is {name!r}: {names}"
    )
    raise DefinitionInvalid(name, [DefinitionFinding("class", None, message)])


_ABSENT: Final = object()


def _declared(cls: type, attribute: str) -> object:
    """The value of *attribute* on *cls*, or :data:`_ABSENT` when it is not declared."""
    return getattr(cls, attribute, _ABSENT)


def _version(cls: type, findings: _Findings) -> str:
    version = _declared(cls, "version")
    if isinstance(version, str) and version:
        return version
    findings.add("attribute", "version", "version must be a non-empty string")
    return ""


def _is_emits(value: object) -> TypeIs[EmitsLevel]:
    levels: tuple[object, ...] = get_args(EmitsLevel.__value__)
    return value in levels


def _is_text_tuple(value: object) -> TypeIs[tuple[str, ...]]:
    return is_object_tuple(value) and all(isinstance(item, str) for item in value)


def _flags(cls: type, findings: _Findings) -> dict[str, object]:
    """The class constants that :class:`Flags` records."""
    flags: dict[str, object] = {}
    for flag in _BOOLEAN_FLAGS:
        value = _declared(cls, flag)
        if isinstance(value, bool):
            flags[flag] = value
        else:
            findings.add(
                "attribute", flag, f"{flag} must be a bool declared on the class"
            )
    emits = _declared(cls, "emits")
    if _is_emits(emits):
        flags["emits"] = emits
    else:
        levels = ", ".join(get_args(EmitsLevel.__value__))
        findings.add("attribute", "emits", f"emits must be one of {levels}")
    roots = _declared(cls, "consumed_roots")
    if _is_text_tuple(roots):
        flags["consumed_roots"] = roots
    else:
        findings.add(
            "attribute", "consumed_roots", "consumed_roots must be a tuple of strings"
        )
    category = _declared(cls, "category")
    if isinstance(category, str):
        flags["category"] = category
    elif category is not _ABSENT:
        findings.add("attribute", "category", "category must be a string")
    follows_ids = _declared(cls, "follows_ids")
    if isinstance(follows_ids, bool):
        flags["follows_ids"] = follows_ids
    elif follows_ids is not _ABSENT:
        findings.add("attribute", "follows_ids", "follows_ids must be a bool")
    relates = _declared(cls, "relates")
    if relates in RELATES_LEVELS:
        flags["relates"] = relates
    elif relates is not _ABSENT:
        levels = ", ".join(RELATES_LEVELS)
        findings.add("attribute", "relates", f"relates must be one of {levels}")
    for method in _METHODS:
        if not callable(_declared(cls, method)):
            findings.add("attribute", method, f"{method} must be a method of the class")
    if _declared(cls, "__mosaic_definition__") is not _ABSENT:
        findings.add(
            "attribute",
            "__mosaic_definition__",
            "the class must not declare __mosaic_definition__, which the toolkit sets",
        )
    return flags


def _attributes(cls: type, findings: _Findings) -> dict[str, object]:
    """The optional class attributes that :class:`Attributes` records."""
    attributes: dict[str, object] = {}
    for attribute in _BOOLEAN_ATTRIBUTES:
        value = _declared(cls, attribute)
        if isinstance(value, bool):
            attributes[attribute] = value
        elif value is not _ABSENT:
            findings.add("attribute", attribute, f"{attribute} must be a bool")
    for attribute in _TEXT_ATTRIBUTES:
        value = _declared(cls, attribute)
        if isinstance(value, str):
            attributes[attribute] = value
        elif value is not _ABSENT:
            findings.add("attribute", attribute, f"{attribute} must be a string")
    if _declared(cls, "track_spaces") not in (_ABSENT, None):
        try:
            attributes["track_spaces"] = tuple(sorted(feature_track_spaces(cls)))
        except TypeError as exc:
            findings.add("attribute", "track_spaces", str(exc))
    _temporal_reach(cls, attributes, findings)
    gpu_when = _declared(cls, "gpu_when")
    if isinstance(gpu_when, GpuWhen):
        attributes["gpu_when"] = {"field": gpu_when.field, "values": gpu_when.values}
    elif gpu_when is None:
        attributes["gpu_when"] = None
    elif gpu_when is not _ABSENT:
        findings.add("attribute", "gpu_when", "gpu_when must be a GpuWhen")
    pair_rule_column = _declared(cls, "pair_rule_column")
    if pair_rule_column is None or isinstance(pair_rule_column, str):
        attributes["pair_rule_column"] = pair_rule_column
    elif pair_rule_column is not _ABSENT:
        findings.add(
            "attribute", "pair_rule_column", "pair_rule_column must be a string"
        )
    try:
        requires = declared_requires(cls)
    except TypeError as exc:
        findings.add("attribute", "requires", str(exc))
    else:
        if requires is not None:
            attributes["requires"] = tuple(
                JSON_OBJECT.validate_json(_REQUIREMENT.dump_json(requirement))
                for requirement in requires
            )
    try:
        provides = declared_provides(cls)
    except TypeError as exc:
        findings.add("attribute", "provides", str(exc))
    else:
        if provides is not None:
            attributes["provides"] = JSON_OBJECT.validate_json(
                _PROVIDES.dump_json(provides)
            )
    return attributes


def _temporal_reach(
    cls: type, attributes: dict[str, object], findings: _Findings
) -> None:
    """Record a declared ``temporal_reach``, which must not vary with the params."""
    reach = _declared(cls, "temporal_reach")
    if callable(reach):
        findings.add(
            "temporal_reach",
            "temporal_reach",
            "temporal_reach must be an int or None, not a function of the params, "
            "because planning reads it without the class",
        )
    elif reach is None or (isinstance(reach, int) and not isinstance(reach, bool)):
        attributes["temporal_reach"] = reach
    elif reach is not _ABSENT:
        findings.add(
            "temporal_reach", "temporal_reach", "temporal_reach must be an int or None"
        )
    if _declared(cls, "temporal_reach_by_option") is not _ABSENT:
        findings.add(
            "temporal_reach",
            "temporal_reach_by_option",
            "temporal_reach_by_option is not allowed, because planning reads a "
            "step's reach without the class",
        )


def _prerequisites(cls: type, findings: _Findings) -> tuple[PrerequisiteRecord, ...]:
    """Each ``OptionalModule`` of the class's ``prerequisites``."""
    declared = _declared(cls, "prerequisites")
    if declared is _ABSENT:
        return ()
    if not is_object_tuple(declared) or not all(
        isinstance(item, OptionalModule) for item in declared
    ):
        findings.add(
            "attribute",
            "prerequisites",
            "prerequisites must be a tuple of OptionalModule",
        )
        return ()
    return tuple(
        PrerequisiteRecord(module=item.module, extra=item.extra, purpose=item.purpose)
        for item in declared
        if isinstance(item, OptionalModule)
    )


def _inputs(cls: type, findings: _Findings) -> InputsRecord | None:
    """The three facts of the class's ``Inputs``, which must add nothing else."""
    model = _declared(cls, "Inputs")
    if not _is_model_class(model, Inputs):
        findings.add("inputs", None, "Inputs must subclass the toolkit's Inputs")
        return None
    found = len(findings.found)
    declared: set[str] = set(model.__private_attributes__)
    for own in _own_classes(model, Inputs):
        if own.__module__ == Inputs.__module__:
            continue
        annotations: object = vars(own).get("__annotations__", {})
        if is_object_mapping(annotations):
            declared |= {str(key) for key in annotations}
    for key in sorted(declared - _INPUTS_DECLARATIONS):
        findings.add(
            "inputs", key, f"Inputs declares {key}, beyond _require and _track_input"
        )
    _decorator_findings(model, Inputs, None, findings, check="inputs")
    items = _input_items(model, findings)
    require: object = getattr(model, "_require", "nonempty")
    if require not in _INPUT_REQUIRES:
        findings.add("inputs", "_require", f"_require must be one of {_INPUT_REQUIRES}")
    track_input: object = getattr(model, "_track_input", False)
    if not isinstance(track_input, bool):
        findings.add("inputs", "_track_input", "_track_input must be a bool")
    if len(findings.found) > found:
        return None
    return InputsRecord.model_validate(
        {"items": items, "require": require, "track_input": track_input}
    )


def _input_items(
    model: type[BaseModel], findings: _Findings
) -> tuple[InputItemName, ...]:
    """Which of ``tracks`` and a ``Result`` the ``Inputs`` model holds."""
    root = model.model_fields.get("root")
    arguments: tuple[object, ...] = () if root is None else get_args(root.annotation)
    item: object = arguments[0] if arguments else None
    if isinstance(item, TypeVar):
        item = item.__default__ if item.has_default() else item.__bound__
    members: tuple[object, ...] = (
        get_args(item) if get_origin(item) in (Union, types.UnionType) else (item,)
    )
    names: set[InputItemName] = set()
    for member in members:
        if member == TrackInput:
            names.add("tracks")
        elif member is Result:
            names.add("result")
        else:
            findings.add(
                "inputs", None, "the items of Inputs must be TrackInput, Result or both"
            )
            return ()
    return tuple(name for name in ("tracks", "result") if name in names)


def _is_model_class(value: object, base: type[BaseModel]) -> TypeIs[type[BaseModel]]:
    """Whether *value* is *base* or a subclass of it."""
    return isinstance(value, type) and issubclass(value, base)


def _own_classes(model: type[BaseModel], base: type[BaseModel]) -> tuple[type, ...]:
    """The classes of *model*'s MRO that come before *base*."""
    order = model.__mro__
    return order[: order.index(base)]


def _decorator_findings(
    model: type[BaseModel],
    base: type[BaseModel],
    path: str | None,
    findings: _Findings,
    *,
    check: DefinitionCheck | None = None,
) -> None:
    """A finding for each validator, serializer or computed field below *base*.

    Each finding is *check* when given, and otherwise ``params_validator`` or
    ``params_serializer``.
    """
    own = _own_classes(model, base)
    decorators = model.__pydantic_decorators__
    groups: tuple[tuple[Mapping[str, object], DefinitionCheck, str], ...] = (
        (decorators.validators, "params_validator", "a validator"),
        (decorators.field_validators, "params_validator", "a field validator"),
        (decorators.root_validators, "params_validator", "a model validator"),
        (decorators.model_validators, "params_validator", "a model validator"),
        (decorators.field_serializers, "params_serializer", "a field serializer"),
        (decorators.model_serializers, "params_serializer", "a model serializer"),
        (decorators.computed_fields, "params_serializer", "a computed field"),
    )
    for group, kind, what in groups:
        for name, decorator in group.items():
            if not any(name in vars(each) for each in own):
                continue
            info: object = getattr(decorator, "info", None)
            fields: object = getattr(info, "fields", None)
            targets: tuple[str | None, ...]
            if group is decorators.computed_fields:
                targets = (_join(path, name),)
            elif _is_text_tuple(fields):
                targets = tuple(_join(path, field) for field in fields)
            else:
                targets = (path,)
            for target in targets:
                findings.add(
                    check or kind,
                    target,
                    f"{model.__name__} declares {what}, {name}, which the rebuilt "
                    "model would not have",
                )


def _join(path: str | None, name: str) -> str:
    return name if path is None else f"{path}.{name}"


def _is_toolkit_model(model: type) -> TypeIs[type[DeclaredModel]]:
    return issubclass(model, DeclaredModel) and model.__module__.startswith(
        _TOOLKIT_PREFIX
    )


def _ref(model: type) -> str:
    return f"{model.__module__}:{model.__qualname__}"


class _ParamsWalker:
    """Checks a ``Params`` model and every model it holds, and records their fields."""

    def __init__(self, findings: _Findings) -> None:
        self._findings: _Findings = findings
        self._own: dict[type[BaseModel], ModelType] = {}
        self._walking: list[type[BaseModel]] = []

    def params(self, cls: type) -> tuple[type[Params] | None, tuple[ParamRecord, ...]]:
        """The class's ``Params`` model and one record per field."""
        model = _declared(cls, "Params")
        if not (isinstance(model, type) and issubclass(model, Params)):
            self._findings.add(
                "params_base", None, "Params must subclass mosaic.core.params.Params"
            )
            return None, ()
        return model, self._records(model, Params, None)

    def _records(
        self, model: type[BaseModel], base: type[BaseModel], path: str | None
    ) -> tuple[ParamRecord, ...]:
        self._model_findings(model, base, path)
        records: list[ParamRecord] = []
        for name, info in model.model_fields.items():
            record = self._record(name, info, _join(path, name))
            if record is not None:
                records.append(record)
        return tuple(records)

    def _model_findings(
        self, model: type[BaseModel], base: type[BaseModel], path: str | None
    ) -> None:
        """Refuse what changes validation or the dump outside the recorded fields."""
        if model.model_config != base.model_config:
            self._findings.add(
                "params_config",
                path,
                f"the model_config of {model.__name__} differs from that of "
                f"{base.__name__}",
            )
        _decorator_findings(model, base, path, self._findings)
        for own in _own_classes(model, base):
            namespace = vars(own)
            post_init: object = namespace.get("model_post_init")
            if post_init is not None and not _from_pydantic(post_init):
                self._findings.add(
                    "params_post_init",
                    path,
                    f"{own.__name__} defines model_post_init, which can change values "
                    "after validation",
                )
            if "__init__" in namespace:
                self._findings.add(
                    "params_post_init",
                    path,
                    f"{own.__name__} defines __init__, which can change values after "
                    "validation",
                )
            if base is NestedParams and "identity_dump" in namespace:
                self._findings.add(
                    "params_serializer",
                    path,
                    f"{own.__name__} overrides identity_dump",
                )

    def _record(self, name: str, info: FieldInfo, path: str) -> ParamRecord | None:
        found = len(self._findings.found)
        node = self._type(info.annotation, path)
        constraints = self._constraints(info, path)
        self._attribute_findings(info, path)
        if info.is_required():
            self._findings.add("params_default", path, f"{path} has no default")
            return None
        try:
            default: object = info.get_default(call_default_factory=True)
        except (TypeError, ValueError) as exc:
            self._findings.add(
                "params_default", path, f"the default of {path} cannot be built: {exc}"
            )
            return None
        if node is None:
            return None
        encoded = self._encode(node, default, path)
        if len(self._findings.found) > found:
            return None
        declared = own_declaration(info)
        metadata: list[object] = info.metadata
        return ParamRecord(
            name=name,
            type=node,
            constraints=constraints,
            default=encoded,
            default_factory=info.default_factory is not None,
            hash_excluded=any(isinstance(entry, HashExclude) for entry in metadata),
            description=None if declared is None else declared.description,
            unit="" if declared is None else declared.unit,
            unwired="" if declared is None else declared.unwired,
            keypoint=None if declared is None else declared.keypoint,
        )

    def _attribute_findings(self, info: FieldInfo, path: str) -> None:
        """Refuse the ``Field`` arguments that the rebuilt model would not have."""
        groups: tuple[tuple[DefinitionCheck, tuple[str, ...]], ...] = (
            ("params_alias", _ALIAS_ATTRIBUTES),
            ("params_metadata", _REFUSED_ATTRIBUTES),
        )
        for check, attributes in groups:
            set_here = [name for name in attributes if getattr(info, name) is not None]
            if set_here:
                listed = ", ".join(set_here)
                message = f"{path} sets {listed}, which the rebuilt model lacks"
                if "description" in set_here:
                    message += (
                        "; declare a description with Declared(...) in the "
                        "field's Annotated"
                    )
                self._findings.add(check, path, message)

    def _constraints(self, info: FieldInfo, path: str) -> Constraints:
        """The recorded constraints of a field, refusing any other metadata."""
        recorded: dict[str, object] = {}
        metadata: list[object] = info.metadata
        for entry in metadata:
            if isinstance(entry, (Declared, HashExclude)):
                continue
            if self._bound(entry, recorded):
                continue
            if type(entry) is _GENERAL_METADATA:
                settings: dict[str, object] = dict(vars(entry))
                pattern = settings.get("pattern")
                if set(settings) == {"pattern"} and isinstance(pattern, str):
                    recorded["pattern"] = pattern
                    continue
                listed = ", ".join(sorted(settings))
                message = f"{path} sets {listed}, outside the recorded constraints"
            else:
                message = f"{path} carries {entry!r}, outside the recorded constraints"
            self._findings.add("params_metadata", path, message)
        return Constraints.model_validate(recorded)

    @staticmethod
    def _bound(entry: object, recorded: dict[str, object]) -> bool:
        for kind, attribute in _BOUNDS:
            if isinstance(entry, kind):
                recorded[attribute] = getattr(entry, attribute)
                return True
        return False

    def _type(self, annotation: object, path: str) -> TypeNode | None:
        """The type tree of *annotation*, or ``None`` with a finding."""
        if isinstance(annotation, TypeAliasType):
            value: object = annotation.__value__
            return self._type(value, path)
        for scalar, node in _SCALARS:
            if annotation is scalar:
                return node
        origin = get_origin(annotation)
        arguments: tuple[object, ...] = get_args(annotation)
        if origin is Annotated:
            self._findings.add(
                "params_type",
                path,
                f"{path} has Annotated metadata below the top level of its type; "
                f"write it at the top level, as {_ACCEPTED_FORM}",
            )
            return None
        if origin is Literal:
            return self._literal(arguments, path)
        if origin in (Union, types.UnionType):
            return self._members(arguments, path)
        if origin is list and len(arguments) == 1:
            item = self._type(arguments[0], path)
            return None if item is None else ListType(kind="list", item=item)
        if origin is tuple:
            return self._tuple(arguments, path)
        if origin is dict and len(arguments) == 2:
            return self._dict(arguments, path)
        if isinstance(annotation, type) and origin is None:
            return self._class(annotation, path)
        self._findings.add(
            "params_type",
            path,
            f"{path} is typed {annotation!r}, outside the types a definition's "
            "params may hold",
        )
        return None

    def _literal(self, values: tuple[object, ...], path: str) -> TypeNode | None:
        scalars = tuple(
            value
            for value in values
            if value is None or isinstance(value, (bool, int, float, str))
        )
        if len(scalars) != len(values):
            self._findings.add(
                "params_type",
                path,
                f"{path} is a Literal of values that are not scalars",
            )
            return None
        return LiteralType(kind="literal", values=scalars)

    def _members(self, arguments: tuple[object, ...], path: str) -> TypeNode | None:
        members = [self._type(argument, path) for argument in arguments]
        known = tuple(member for member in members if member is not None)
        if len(known) != len(members):
            return None
        return UnionType(kind="union", of=known)

    def _tuple(self, arguments: tuple[object, ...], path: str) -> TypeNode | None:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            item = self._type(arguments[0], path)
            return None if item is None else TupleType(kind="tuple", item=item)
        items = [self._type(argument, path) for argument in arguments]
        known = tuple(item for item in items if item is not None)
        if len(known) != len(items):
            return None
        return TupleType(kind="tuple", items=known)

    def _dict(self, arguments: tuple[object, ...], path: str) -> TypeNode | None:
        key, value = arguments
        if key is not str:
            self._findings.add(
                "params_type", path, f"{path} is a dict whose keys are not str"
            )
            return None
        inner = self._type(value, path)
        return None if inner is None else DictType(kind="dict", value=inner)

    def _class(self, annotation: type[object], path: str) -> TypeNode | None:
        if any(issubclass(annotation, each) for each in _REFERENCE_TYPES):
            self._findings.add(
                "params_reference",
                path,
                f"{path} is typed {annotation.__name__}, a reference to another run, "
                "which a definition cannot hold",
            )
            return None
        if _is_toolkit_model(annotation):
            return ToolkitModelType(kind="toolkit_model", ref=_ref(annotation))
        if issubclass(annotation, DeclaredModel):
            return self._own_model(annotation, path)
        self._findings.add(
            "params_type",
            path,
            f"{path} is typed {annotation.__name__}, outside the types a "
            "definition's params may hold",
        )
        return None

    def _own_model(self, model: type[DeclaredModel], path: str) -> TypeNode | None:
        """Record a model of the definition's own file, field by field."""
        if model in self._walking:
            self._findings.add(
                "params_type", path, f"{model.__name__} holds itself, at {path}"
            )
            return None
        cached = self._own.get(model)
        if cached is not None:
            return cached
        nested = issubclass(model, NestedParams)
        base = NestedParams if nested else DeclaredModel
        self._walking.append(model)
        fields = self._records(model, base, path)
        _ = self._walking.pop()
        node = ModelType(
            kind="model",
            name=model.__name__,
            base="NestedParams" if nested else "DeclaredModel",
            fields=fields,
        )
        self._own[model] = node
        return node

    def _encode(self, node: TypeNode | None, value: object, path: str) -> JsonValue:
        """The JSON of a default *value* at a position of type *node*.

        A toolkit model is ``{"$model": ref, "set": {...}}`` with the fields its
        source set, and a model of the definition's own file is its full dump.
        """
        if isinstance(value, BaseModel):
            model = type(value)
            if _is_toolkit_model(model):
                fields_set = sorted(value.model_fields_set)
                return {
                    _MODEL_MARKER: _ref(model),
                    "set": {
                        key: self._encode(None, getattr(value, key), path)
                        for key in fields_set
                    },
                }
            own = self._own.get(model)
            if own is not None:
                by_name = {record.name: record.type for record in own.fields}
                return {
                    key: self._encode(by_name.get(key), getattr(value, key), path)
                    for key in model.model_fields
                }
        if is_object_mapping(value):
            if _MODEL_MARKER in value and _admits_toolkit_model(node):
                self._findings.add(
                    "model_marker",
                    path,
                    f"the default of {path} holds a plain dict with the key "
                    f"{_MODEL_MARKER!r} where the type also admits a toolkit model",
                )
            inner = _dict_value(node)
            return {
                str(key): self._encode(inner, each, path) for key, each in value.items()
            }
        if is_object_list(value) or is_object_tuple(value):
            return [
                self._encode(_item(node, index), each, path)
                for index, each in enumerate(value)
            ]
        try:
            return _JSON_VALUE.validate_python(identity_ready(value))
        except (TypeError, ValidationError):
            self._findings.add(
                "params_default", path, f"the default of {path} is not JSON: {value!r}"
            )
            return None


def _from_pydantic(function: object) -> bool:
    module: object = getattr(function, "__module__", "")
    return isinstance(module, str) and module.startswith("pydantic")


def _admits_toolkit_model(node: TypeNode | None) -> bool:
    if isinstance(node, ToolkitModelType):
        return True
    if isinstance(node, UnionType):
        return any(isinstance(member, ToolkitModelType) for member in node.of)
    return False


def _dict_value(node: TypeNode | None) -> TypeNode | None:
    """The type of a dict's values at a position of type *node*."""
    if isinstance(node, DictType):
        return node.value
    if isinstance(node, UnionType):
        for member in node.of:
            if isinstance(member, DictType):
                return member.value
    return None


def _item(node: TypeNode | None, index: int) -> TypeNode | None:
    """The type of the item at *index* of a sequence at a position of type *node*."""
    if isinstance(node, ListType):
        return node.item
    if isinstance(node, TupleType):
        if node.items is not None:
            return node.items[index] if index < len(node.items) else None
        return node.item
    if isinstance(node, UnionType):
        for member in node.of:
            if isinstance(member, (ListType, TupleType)):
                return _item(member, index)
    return None


def _dumped(value: object) -> object:
    """*value* with each model dumped, without pydantic's serialization warnings."""
    if isinstance(value, BaseModel):
        return value.model_dump(warnings=False)
    if is_object_mapping(value):
        return {key: _dumped(each) for key, each in value.items()}
    if is_object_list(value) or is_object_tuple(value):
        return [_dumped(each) for each in value]
    return value


def _canonical(value: object) -> str:
    """The canonical JSON of *value*, its models dumped without warnings."""
    return canonical_json(_dumped(value))


def _rebuild_findings(
    sidecar: DefinitionSidecar,
    params_model: type[Params],
    toolkit_model: ToolkitModelLookup,
    findings: _Findings,
) -> None:
    """Refuse each default that its record does not rebuild to."""
    for record in sidecar.params:
        source: object = params_model.model_fields[record.name].get_default(
            call_default_factory=True
        )
        alone = sidecar.model_copy(update={"params": (record,)})
        try:
            rebuilt_model = rebuild_params_model(alone, toolkit_model)
            rebuilt: object = rebuilt_model.model_fields[record.name].get_default(
                call_default_factory=True
            )
        except (ValueError, TypeError, ClassifiedError) as exc:
            findings.add(
                "model_rebuild",
                record.name,
                f"the default of {record.name} does not rebuild from its record: "
                f"{terse(exc)}",
            )
            continue
        if _canonical(rebuilt) != _canonical(source):
            findings.add(
                "model_rebuild",
                record.name,
                f"the default of {record.name} rebuilt from its record differs from "
                "the source's, as a model built with model_construct or changed "
                "after construction does",
            )


def _construction_findings(
    cls: type,
    sidecar: DefinitionSidecar,
    toolkit_model: ToolkitModelLookup,
    findings: _Findings,
) -> None:
    """Build the class with its default inputs and params, and compare its identity."""
    payload = default_inputs(sidecar.inputs)
    inputs_model = _declared(cls, "Inputs")
    constructor: object = cls
    try:
        if not _is_model_class(inputs_model, BaseModel):
            raise TypeError("the class declares no Inputs model")
        if not callable(constructor):
            raise TypeError("the class cannot be called")
        built: object = constructor(inputs_model.model_validate(payload), None)
    except Exception as exc:
        findings.add(
            "construction",
            None,
            f"{cls.__name__}(Inputs({payload!r}), None) raises "
            f"{type(exc).__name__}: {exc}",
        )
        return
    held: object = getattr(built, "__dict__", {})
    for flag in _FLAGS_IN_INIT:
        if is_object_mapping(held) and flag in held:
            findings.add(
                "flag_in_init",
                flag,
                f"__init__ sets {flag} on the instance; declare it on the class, "
                "because planning reads the class",
            )
    params: object = getattr(built, "params", None)
    inputs: object = getattr(built, "inputs", None)
    if not (isinstance(params, Params) and isinstance(inputs, BaseModel)):
        findings.add(
            "construction", None, "the built instance holds no params or inputs model"
        )
        return
    declared_params = rebuild_params_model(sidecar, toolkit_model)()
    declared_inputs = rebuild_inputs_model(sidecar).model_validate(payload)
    if _identity_payload(params, inputs) != _identity_payload(
        declared_params, declared_inputs
    ):
        findings.add(
            "construction",
            None,
            "the identity payload of the built instance differs from the one its "
            "record declares",
        )


def _identity_payload(params: Params, inputs: BaseModel) -> str:
    return _canonical(identity_terms(params, inputs))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
