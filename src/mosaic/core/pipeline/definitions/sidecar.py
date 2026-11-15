"""The sidecar of a feature definition, its reader and the models rebuilt from it.

A sidecar records the facts a definition owns: its class's flags, inputs, params
and attributes. Planning, validation and the catalogs read it without importing the
source, and derive everything else from it on each read: the ``Params`` model
through :func:`rebuild_params_model` and the ``Inputs`` model through
:func:`rebuild_inputs_model`.

A field of a record that the definition does not declare is left unset, and
:func:`sidecar_json` writes only the fields that are set. ``model_fields_set``
therefore tells an attribute declared ``None`` from one not declared at all.

The module imports the standard library, pydantic, ``core.params`` and the store at
import. The ``Inputs`` base is imported inside :func:`rebuild_inputs_model`,
because the ``types`` package loads pandas.
"""

from __future__ import annotations

import copy
import json
import keyword
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from importlib import metadata
from datetime import date
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, TypeAliasType, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_object_tuple
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    DeclaredModel,
    KeypointKind,
    NestedParams,
    Params,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel, RelatesLevel
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_PACKAGE,
    DefinitionCorrupt,
    DefinitionFinding,
    DefinitionInvalid,
    VersionStem,
)
from mosaic.core.spaces import TrackSpace
from mosaic.core.strict_model import StrictModel

__all__ = [
    "SIDECAR_FORMAT",
    "SIDECAR_FORMAT_VERSION",
    "Attributes",
    "Constraints",
    "DefinitionRef",
    "DefinitionSidecar",
    "DictType",
    "Flags",
    "GpuWhenRecord",
    "InputItemName",
    "InputRequireName",
    "InputsRecord",
    "ListType",
    "LiteralType",
    "ModelType",
    "ParamRecord",
    "PrerequisiteRecord",
    "ScalarType",
    "ToolkitModelType",
    "ToolkitModelLookup",
    "TupleType",
    "TypeKind",
    "TypeNode",
    "UnionType",
    "default_inputs",
    "installed_extras",
    "prerequisite_extra_findings",
    "prerequisite_findings",
    "read_sidecar",
    "rebuild_inputs_model",
    "rebuild_params_model",
    "sidecar_json",
]

SIDECAR_FORMAT: Final = "mosaic-feature-definition"
"""The ``format`` every sidecar names."""

SIDECAR_FORMAT_VERSION: Final = 1
"""The newest ``format_version`` this toolkit reads and the one it writes.

It increases when a recorded field is added or changes meaning. A reader fills
each field an earlier version lacks with its default.
"""

_MODEL_MARKER: Final = "$model"
_MODEL_SET: Final = "set"
_UPSTREAM: Final = "upstream__from__tracks"
"""The feature a probe ``Result`` names, for an ``Inputs`` that reads no tracks."""

type TypeKind = Literal[
    "bool",
    "int",
    "float",
    "str",
    "null",
    "literal",
    "list",
    "tuple",
    "dict",
    "union",
    "toolkit_model",
    "model",
]
"""The kinds of node in a field's type tree."""

type ScalarKind = Literal["bool", "int", "float", "str", "null"]
"""The kinds of node that name a scalar type."""

type ScalarValue = bool | int | float | str | None
"""A value a ``Literal`` may hold."""

type InputItemName = Literal["tracks", "result"]
"""What a definition's ``Inputs`` holds: ``tracks``, a ``Result``, or both."""

type InputRequireName = Literal["nonempty", "empty", "any"]
"""Whether a definition's ``Inputs`` must hold items, must hold none, or either."""

type ModelBase = Literal["DeclaredModel", "NestedParams"]
"""The base a model of the definition's own file subclasses."""


@dataclass(frozen=True)
class DefinitionRef:
    """The definition a loaded class, or a run's params, belongs to.

    Attributes:
        name: The definition's bare name, without ``def:``.
        digest: The full SHA-256 of its source.
    """

    name: str
    digest: str

    @property
    def slug(self) -> str:
        """The slug naming this version, ``def:<name>@<digest>``."""
        return f"def:{self.name}@{self.digest}"


class _Record(StrictModel):
    """The frozen base of every record in a sidecar."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ScalarType(_Record):
    """``bool``, ``int``, ``float``, ``str`` or ``None``."""

    kind: ScalarKind


class LiteralType(_Record):
    """A ``Literal`` of scalars."""

    kind: Literal["literal"]
    values: tuple[ScalarValue, ...]


class ListType(_Record):
    """``list[item]``."""

    kind: Literal["list"]
    item: TypeNode


class TupleType(_Record):
    """``tuple[item, ...]`` when ``item`` is set, ``tuple[*items]`` when ``items`` is."""

    kind: Literal["tuple"]
    item: TypeNode | None = None
    items: tuple[TypeNode, ...] | None = None

    @model_validator(mode="after")
    def _one_form(self) -> Self:
        if (self.item is None) == (self.items is None):
            raise ValueError("a tuple sets exactly one of item and items")
        return self


class DictType(_Record):
    """``dict[str, value]``."""

    kind: Literal["dict"]
    value: TypeNode


class UnionType(_Record):
    """A union of its members, in the order the source wrote them."""

    kind: Literal["union"]
    of: tuple[TypeNode, ...]


class ToolkitModelType(_Record):
    """A ``DeclaredModel`` of the toolkit, named ``<module>:<class>``."""

    kind: Literal["toolkit_model"]
    ref: str


class ModelType(_Record):
    """A model of the definition's own file, recorded field by field.

    Attributes:
        name: The class's name, which the rebuilt model takes.
        base: The base the class subclasses.
        fields: One record per field, in field order.
    """

    kind: Literal["model"]
    name: str
    base: ModelBase
    fields: tuple[ParamRecord, ...]


type TypeNode = Annotated[
    ScalarType
    | LiteralType
    | ListType
    | TupleType
    | DictType
    | UnionType
    | ToolkitModelType
    | ModelType,
    Field(discriminator="kind"),
]
"""One node of a field's type tree."""


class Constraints(_Record):
    """The constraints that ``Field`` sets on one field."""

    gt: int | float | None = None
    ge: int | float | None = None
    lt: int | float | None = None
    le: int | float | None = None
    multiple_of: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


class ParamRecord(_Record):
    """One field of a params model, or of a model of the definition's own file.

    Attributes:
        name: The field's name.
        type: The field's type tree.
        constraints: The field's ``Field`` constraints.
        default: The default as JSON. A toolkit model in it is
            ``{"$model": ref, "set": {...}}`` with the fields the source set, and a
            model of the definition's own file is its full dump.
        default_factory: Whether the source gives the default through
            ``default_factory``. ``Params.from_overrides`` merges a partial dict
            override into such a field only. The rebuilt field therefore keeps the form.
        hash_excluded: Whether the field is marked ``HASH_EXCLUDE``.
        description: The ``Declared`` description, ``None`` without a ``Declared``.
        unit: The ``Declared`` unit.
        unwired: The ``Declared`` reason the field reaches nothing.
        keypoint: The ``Declared`` keypoint kind of the field's value.
    """

    name: str
    type: TypeNode
    constraints: Constraints = Field(default_factory=Constraints)
    default: JsonValue
    default_factory: bool = False
    hash_excluded: bool = False
    description: str | None = None
    unit: str = ""
    unwired: str = ""
    keypoint: KeypointKind | None = None


class Flags(_Record):
    """The class constants that planning and the run identity read.

    ``follows_ids`` and ``relates`` are set only when the class declares them.
    """

    parallelizable: bool
    scope_dependent: bool
    accepts_overlap: bool
    emits: EmitsLevel
    category: str = ""
    consumed_roots: tuple[str, ...] = ()
    follows_ids: bool | None = None
    relates: RelatesLevel | None = None


class InputsRecord(_Record):
    """The three facts the ``Inputs`` model is rebuilt from.

    Attributes:
        items: What an input may be: ``tracks``, a ``result``, or both.
        require: The ``_require`` of the class's ``Inputs``.
        track_input: The ``_track_input`` of the class's ``Inputs``.
    """

    items: Annotated[tuple[InputItemName, ...], Field(min_length=1)]
    require: InputRequireName = "nonempty"
    track_input: bool = False


class GpuWhenRecord(_Record):
    """The param whose values make a step GPU work."""

    field: str
    values: tuple[JsonValue, ...]


def _sorted_spaces(spaces: tuple[TrackSpace, ...]) -> tuple[TrackSpace, ...]:
    """Return *spaces* sorted and without repeats: a class declares them as a set."""
    return tuple(sorted(set(spaces)))


class Attributes(_Record):
    """The optional class attributes, each set only when the class declares it.

    ``temporal_reach`` set to ``None`` is a declared reach of ``None``, which reads
    as unknown. ``track_spaces`` is sorted, since a class declares a set whose
    order varies between processes. ``requires`` and ``provides`` hold the JSON
    dumps of the class's column contract.
    """

    fits: bool = False
    costly_fit: bool = False
    stacks_inputs: bool = False
    track_spaces: Annotated[tuple[TrackSpace, ...], AfterValidator(_sorted_spaces)] = ()
    temporal_reach: int | None = None
    report_file: str = ""
    resource_class: str = ""
    gpu_when: GpuWhenRecord | None = None
    pair_rule_column: str | None = None
    requires: tuple[dict[str, JsonValue], ...] | None = None
    provides: dict[str, JsonValue] | None = None


class PrerequisiteRecord(_Record):
    """One ``OptionalModule`` of the class's ``prerequisites``.

    Attributes:
        module: The top-level import name.
        extra: The mosaic extra that installs it.
        purpose: What needs it.
    """

    module: str
    extra: str
    purpose: str


class DefinitionSidecar(_Record):
    """The declaration of one definition version, read without its source.

    Attributes:
        format: Always :data:`SIDECAR_FORMAT`.
        format_version: The format the sidecar was written in.
        name: The definition's name.
        digest: The SHA-256 of the source.
        defined_on: The UTC date of the version's first definition.
        mosaic_version: The toolkit version that wrote the sidecar, read by no check.
        class_name: The class in the source that the definition runs.
        version: The class's ``version``.
        summary: The first line of the class docstring.
        flags: The class constants.
        inputs: The facts of the ``Inputs`` model.
        params: One record per field of the ``Params`` model, in field order.
        attributes: The optional class attributes the class declares.
        prerequisites: The class's ``OptionalModule`` prerequisites.
    """

    format: Literal["mosaic-feature-definition"]
    format_version: int
    name: str
    digest: str
    defined_on: date
    mosaic_version: str
    class_name: str
    version: str
    summary: str
    flags: Flags
    inputs: InputsRecord
    params: tuple[ParamRecord, ...] = ()
    attributes: Attributes = Field(default_factory=Attributes)
    prerequisites: tuple[PrerequisiteRecord, ...] = ()


def sidecar_json(sidecar: DefinitionSidecar) -> bytes:
    """Return *sidecar* as the JSON a sidecar file holds: sorted keys, set fields only."""
    dumped = sidecar.model_dump(mode="json", exclude_unset=True)
    return (json.dumps(dumped, indent=2, sort_keys=True) + "\n").encode("utf-8")


@lru_cache(maxsize=1)
def installed_extras() -> frozenset[str] | None:
    """The extras that the installed ``mosaic-behavior`` distribution declares.

    Read from the distribution's metadata, never from ``pyproject.toml``. ``None``
    when no such distribution is installed, as when the source is run in place,
    and no extra can then be judged.
    """
    try:
        declared = metadata.metadata("mosaic-behavior").get_all("Provides-Extra")
    except metadata.PackageNotFoundError:
        return None
    return frozenset(declared or ())


def prerequisite_findings(sidecar: DefinitionSidecar) -> list[DefinitionFinding]:
    """Return a finding for each prerequisite whose import name is not plain.

    A plain name is one identifier that does not start with ``mosaic_definitions``.
    Checking a dotted name imports its parent package, which would run code that a
    sidecar names.
    """
    return [
        DefinitionFinding(
            check="attribute",
            field="prerequisites",
            message=(
                f"the prerequisite {prerequisite.module!r} is not a single "
                "identifier outside mosaic_definitions"
            ),
        )
        for prerequisite in sidecar.prerequisites
        if not _is_plain_module(prerequisite.module)
    ]


def prerequisite_extra_findings(sidecar: DefinitionSidecar) -> list[DefinitionFinding]:
    """Return a finding for each prerequisite whose extra mosaic does not declare.

    The extra is the one an install hint tells a user to install, and an undeclared
    extra cannot be installed. Only a new definition is checked: a stored sidecar that
    names an extra a later toolkit renamed still reads, and costs only its hint.
    """
    extras = installed_extras()
    if extras is None:
        return []
    return [
        DefinitionFinding(
            check="attribute",
            field="prerequisites",
            message=(
                f"the prerequisite {prerequisite.module!r} names the extra "
                f"{prerequisite.extra!r}, which mosaic does not declare"
            ),
        )
        for prerequisite in sidecar.prerequisites
        if prerequisite.extra not in extras
    ]


def _is_plain_module(module: str) -> bool:
    return (
        module.isidentifier()
        and not keyword.iskeyword(module)
        and not module.startswith(DEFINITIONS_PACKAGE)
    )


class _FormatHeader(BaseModel):
    """The one field a reader checks before it reads the rest of a sidecar."""

    format_version: int


def read_sidecar(path: Path) -> DefinitionSidecar:
    """Read the sidecar at *path*, of any format version up to the current one.

    Raises:
        DefinitionCorrupt: The file is not JSON, is not a sidecar, or names a
            ``format_version`` later than :data:`SIDECAR_FORMAT_VERSION`.
        DefinitionInvalid: A prerequisite's import name is not plain
            (:func:`prerequisite_findings`).
    """
    stem = VersionStem.parse(path.stem)
    corrupt = DefinitionCorrupt(
        path.parent.name, None if stem is None else stem.digest, path
    )
    try:
        text = path.read_bytes()
        header = _FormatHeader.model_validate_json(text)
        if header.format_version > SIDECAR_FORMAT_VERSION:
            raise DefinitionCorrupt(
                corrupt.name,
                corrupt.digest,
                path,
                reason=(
                    f"was written in sidecar format version {header.format_version}, "
                    f"newer than the version {SIDECAR_FORMAT_VERSION} this toolkit "
                    "reads"
                ),
            )
        sidecar = DefinitionSidecar.model_validate_json(text)
    except (OSError, ValidationError):
        raise corrupt from None
    findings = prerequisite_findings(sidecar)
    if findings:
        raise DefinitionInvalid(sidecar.name, findings)
    return sidecar


type ToolkitModelLookup = Callable[[str], type[DeclaredModel]]
"""Return the toolkit model a ``<module>:<class>`` reference names."""


def rebuild_params_model(
    sidecar: DefinitionSidecar, toolkit_model: ToolkitModelLookup
) -> type[Params]:
    """Rebuild the definition's ``Params`` model from its records.

    Each field keeps its type, its constraints, its ``Declared`` description and
    unit, ``HASH_EXCLUDE``, and the form of its default, a value or a factory. A
    model of the definition's own file is rebuilt on its recorded base, and a
    toolkit model is the class *toolkit_model* returns.

    Args:
        sidecar: The definition's sidecar.
        toolkit_model: Returns the toolkit model a ``ref`` names.
    """
    return _Rebuilder(toolkit_model).model("Params", Params, sidecar.params)


class _Rebuilder:
    """Rebuilds type trees into annotations and recorded defaults into values."""

    def __init__(self, toolkit_model: ToolkitModelLookup) -> None:
        self._toolkit_model: ToolkitModelLookup = toolkit_model
        self._own_models: dict[int, type[BaseModel]] = {}

    def model[M: BaseModel](
        self, name: str, base: type[M], records: tuple[ParamRecord, ...]
    ) -> type[M]:
        """Create the model *records* describe, on *base*."""
        annotations: dict[str, object] = {}
        namespace: dict[str, object] = {"__module__": __name__, "__qualname__": name}
        for record in records:
            annotations[record.name] = self.annotation(record.type)
            default = self.decode(record.type, record.default)
            namespace[record.name] = _field_info(record, default)
        namespace["__annotations__"] = annotations
        built = type(name, (base,), namespace)
        if not issubclass(built, base):
            raise TypeError(f"{name} does not subclass {base.__name__}")
        return built

    def annotation(self, node: TypeNode) -> object:
        """Return the annotation *node* describes."""
        match node:
            case ScalarType(kind=kind):
                return _SCALARS[kind]
            case LiteralType(values=values):
                return Literal[values]
            case ListType(item=item):
                return types.GenericAlias(list, (self.annotation(item),))
            case TupleType(items=tuple() as items):
                return types.GenericAlias(
                    tuple, tuple(self.annotation(each) for each in items)
                )
            case TupleType(item=item) if item is not None:
                return types.GenericAlias(tuple, (self.annotation(item), Ellipsis))
            case TupleType():
                raise ValueError("a tuple sets exactly one of item and items")
            case DictType(value=value):
                return types.GenericAlias(dict, (str, self.annotation(value)))
            case UnionType(of=members):
                return Union[tuple(self.annotation(member) for member in members)]
            case ToolkitModelType(ref=ref):
                return self._toolkit_model(ref)
            case ModelType():
                return self._own_model(node)

    def _own_model(self, node: ModelType) -> type[BaseModel]:
        built = self._own_models.get(id(node))
        if built is None:
            base = DeclaredModel if node.base == "DeclaredModel" else NestedParams
            built = self.model(node.name, base, node.fields)
            self._own_models[id(node)] = built
        return built

    def decode(self, node: TypeNode, value: JsonValue) -> object:
        """Return the default *value* records at a position of type *node*.

        A ``$model`` marker becomes a toolkit model only where *node* admits one:
        a ``toolkit_model`` node or a union containing one. Elsewhere it is data.
        """
        match node:
            case ListType(item=item) if isinstance(value, list):
                return [self.decode(item, each) for each in value]
            case TupleType(items=tuple() as items) if isinstance(value, list):
                return tuple(
                    self.decode(kind, each)
                    for kind, each in zip(items, value, strict=False)
                )
            case TupleType(item=item) if item is not None and isinstance(value, list):
                return tuple(self.decode(item, each) for each in value)
            case DictType(value=inner) if isinstance(value, dict):
                return {key: self.decode(inner, each) for key, each in value.items()}
            case ToolkitModelType(ref=ref):
                return self._decode_toolkit(ref, value)
            case ModelType() if isinstance(value, dict):
                model = self._own_models[id(node)]
                by_name = {record.name: record.type for record in node.fields}
                return model.model_validate(
                    {
                        key: self.decode(by_name[key], each) if key in by_name else each
                        for key, each in value.items()
                    }
                )
            case UnionType(of=members):
                return self._decode_union(members, value)
            case _:
                return value

    def _decode_union(self, members: tuple[TypeNode, ...], value: JsonValue) -> object:
        ref = _marker_ref(value)
        for member in members:
            if isinstance(member, ToolkitModelType) and member.ref == ref:
                return self.decode(member, value)
        for member in members:
            if isinstance(value, dict) and isinstance(member, ModelType):
                try:
                    return self.decode(member, value)
                except ValidationError:
                    continue
            if _shape_fits(member, value):
                return self.decode(member, value)
        return value

    def _decode_toolkit(self, ref: str, value: JsonValue) -> object:
        if _marker_ref(value) != ref or not isinstance(value, dict):
            return value
        model = self._toolkit_model(ref)
        recorded = value.get(_MODEL_SET)
        fields = recorded if isinstance(recorded, dict) else {}
        return model.model_validate(
            {
                key: self._decode_in_toolkit(model, key, each)
                for key, each in fields.items()
            }
        )

    def _decode_in_toolkit(
        self, model: type[BaseModel], key: str, value: JsonValue
    ) -> object:
        """Decode a set field of a toolkit model, whose type tree is its annotation.

        A marker anywhere in the value becomes a model when the field's annotation
        names the model the marker refers to.
        """
        info = model.model_fields.get(key)
        named: set[str] = set() if info is None else _model_refs(info.annotation)
        return self._decode_markers(named, value)

    def _decode_markers(self, named: set[str], value: JsonValue) -> object:
        if isinstance(value, list):
            return [self._decode_markers(named, each) for each in value]
        if not isinstance(value, dict):
            return value
        ref = _marker_ref(value)
        if ref is not None and ref in named:
            return self._decode_toolkit(ref, value)
        return {key: self._decode_markers(named, each) for key, each in value.items()}


_SCALARS: Final[Mapping[ScalarKind, object]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "null": type(None),
}


def _field_info(record: ParamRecord, default: object) -> FieldInfo:
    """Return the ``FieldInfo`` of *record*: its default, constraints and markers.

    The default is a factory where the source used one, and the value otherwise.
    """
    constraints = record.constraints
    factory: Callable[[], object] | None = None
    if record.default_factory:
        factory = partial(copy.deepcopy, default)
        default = PydanticUndefined
    info = FieldInfo.from_field(
        default,
        default_factory=factory,
        gt=constraints.gt,
        ge=constraints.ge,
        lt=constraints.lt,
        le=constraints.le,
        multiple_of=constraints.multiple_of,
        min_length=constraints.min_length,
        max_length=constraints.max_length,
        pattern=constraints.pattern,
    )
    if record.description is not None:
        info.metadata.append(
            Declared(
                record.description,
                unit=record.unit,
                unwired=record.unwired,
                keypoint=record.keypoint,
            )
        )
    if record.hash_excluded:
        info.metadata.append(HASH_EXCLUDE)
    return info


def _marker_ref(value: JsonValue) -> str | None:
    """The ``ref`` of ``{"$model": ref, "set": {...}}``, else ``None``."""
    if not isinstance(value, dict) or set(value) != {_MODEL_MARKER, _MODEL_SET}:
        return None
    ref = value[_MODEL_MARKER]
    return ref if isinstance(ref, str) else None


def _shape_fits(node: TypeNode, value: JsonValue) -> bool:
    """Whether a JSON *value* has the shape of a value of *node*."""
    match node:
        case ListType():
            return isinstance(value, list)
        case TupleType(items=tuple() as items):
            return isinstance(value, list) and len(value) == len(items)
        case TupleType():
            return isinstance(value, list)
        case DictType():
            return isinstance(value, dict)
        case _:
            return False


def _model_refs(annotation: object) -> set[str]:
    """The ``<module>:<class>`` of every model that *annotation* names, at any depth."""
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return {f"{annotation.__module__}:{annotation.__qualname__}"}
        return set()
    if isinstance(annotation, TypeAliasType):
        value: object = annotation.__value__
        return _model_refs(value)
    arguments: object = getattr(annotation, "__args__", None)
    found: set[str] = set()
    if is_object_tuple(arguments):
        for argument in arguments:
            found |= _model_refs(argument)
    return found


def default_inputs(record: InputsRecord) -> list[JsonValue]:
    """The inputs a definition is built with: none, ``tracks``, or a probe ``Result``."""
    if record.require != "nonempty":
        return []
    if "tracks" in record.items:
        return ["tracks"]
    return [{"feature": _UPSTREAM}]


def rebuild_inputs_model(sidecar: DefinitionSidecar) -> type[BaseModel]:
    """Rebuild the definition's ``Inputs`` model from its three recorded facts.

    A subclass of ``Inputs[TrackInput]``, ``Inputs[Result]`` or ``Inputs`` of their
    union, with ``_require`` and ``_track_input`` set.
    """
    from mosaic.core.pipeline.types.inputs import Inputs, InputRequire, TrackInput
    from mosaic.core.pipeline.types.results import Result

    record = sidecar.inputs
    items = set(record.items)
    base: type[BaseModel] = (
        Inputs[TrackInput]
        if items == {"tracks"}
        else Inputs[Result]
        if items == {"result"}
        else Inputs[TrackInput | Result]
    )
    namespace: dict[str, object] = {
        "__annotations__": {
            "_require": ClassVar[InputRequire],
            "_track_input": ClassVar[bool],
        },
        "_require": record.require,
        "_track_input": record.track_input,
        "__module__": __name__,
        "__qualname__": "Inputs",
    }
    built = type("Inputs", (base,), namespace)
    if not issubclass(built, BaseModel):
        raise TypeError("the rebuilt Inputs is not a model")
    return built
