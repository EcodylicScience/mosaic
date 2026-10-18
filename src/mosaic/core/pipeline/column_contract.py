"""The columns a feature reads from its merged input, and the columns of its output.

All of a step's inputs are inner-joined into one frame before ``apply`` sees it.
A feature reads columns from the frame by name, often a name stated by one of
its params. Two class attributes declare those names:

- ``requires``: a tuple of requirements, each naming a column that ``apply`` reads.
- ``provides``: one :class:`Provides`, the columns of a non-empty output.

Each is optional, and an absent one means *not declared*. That default is safe.
Every reader of a declaration treats an undeclared feature's output as unknown,
and a requirement on unknown columns is a caution. It is not a refusal.

Pipeline validation reads both from the declaration catalog.
:func:`check_input_columns` checks ``requires`` against the merged frame before
``apply``. One declaration decides in both places, and validation and execution
agree on the columns a feature reads.

A reference names columns by name (:class:`Column`), by the value of a params
field (:class:`ParamColumn`), by a template over the params (:class:`Template`),
or as a family sharing a prefix (:class:`Prefix`). A params field resolves from
the step's params, which contain the field's default when the step does not set
it. A requirement is a reference, a group that any one member meets
(:class:`AnyOf`), the keypoint columns that a ``PoseConfig`` selects
(:class:`Keypoints`), the columns that ``apply`` derives from other columns
(:class:`Derived`), or any numeric payload column (:class:`Payload`). A global
model reads every payload column of its merged input, whatever its name. A
requirement that ``apply`` reads under some params only applies under conditions
on them (:class:`When`). An output names its columns by reference, or names a
family of columns after each payload column of its merged input
(``Provides.payload_families``).
"""

from __future__ import annotations

import string
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.types.data_config import (
    COLUMNS,
    META_COLS,
    NON_PAYLOAD_COLS,
    PoseConfig,
)

if TYPE_CHECKING:
    from mosaic.core.pipeline.types.feature import Feature

__all__ = [
    "AnyOf",
    "Column",
    "ColumnRef",
    "Conditional",
    "Derived",
    "Keypoints",
    "META_COLUMNS_KEPT",
    "META_KEPT",
    "MissingInputColumns",
    "NON_PAYLOAD_KEPT",
    "NamedColumns",
    "ORDER_COLUMN",
    "PAYLOAD_LABEL",
    "POSE_X",
    "POSE_Y",
    "ParamColumn",
    "ParamIs",
    "ParamValues",
    "Payload",
    "Prefix",
    "Provides",
    "Requirement",
    "Template",
    "When",
    "check_input_columns",
    "contract_fields",
    "declared_names",
    "declared_provides",
    "declared_requires",
    "is_set",
    "kept",
    "needs_of",
    "param_values",
    "resolve_names",
]


@dataclass(frozen=True, slots=True)
class Column:
    """One column, by its name."""

    name: str
    kind: Literal["column"] = "column"


@dataclass(frozen=True, slots=True)
class ParamColumn:
    """The column, or columns, that a params field names.

    A string value names one column, and a list of strings names one column per
    item. An unset field (``None`` or empty) names none, unless *fallback* names
    a column instead.

    Attributes:
        field: The params field.
        fallback: What an unset field names: a column name, or a template over
            the step's params such as ``"{column}_{agg}"``.
        prefix: Prepended to every name, as in ``neighbor_<tag>``.
        renamed_by: A mapping-valued params field that each name is looked up
            in, as ``id-tag-columns`` renames its label fields.
        unset: What a ``None`` field without a fallback means: no column
            (``"absent"``), or columns that only the data decides
            (``"unknown"``). Under both, an empty field without a fallback is
            ``"absent"``.
    """

    field: str
    fallback: str = ""
    prefix: str = ""
    renamed_by: str = ""
    unset: Literal["absent", "unknown"] = "absent"
    kind: Literal["param"] = "param"


@dataclass(frozen=True, slots=True)
class Prefix:
    """A family of columns sharing a name prefix, as the pose keypoints do."""

    prefix: str
    kind: Literal["prefix"] = "prefix"


@dataclass(frozen=True, slots=True)
class Template:
    """A column, or a family of columns, whose name is a template over the params.

    ``"{x_prefix}{pose_head_index}"`` names the head keypoint of ``pair-facing``.
    A dotted field reads a nested params model, as ``"{pose.x_prefix}"`` reads a
    ``PoseConfig``. A template whose field is unset or holds no scalar names
    columns that cannot be listed.

    Attributes:
        template: The name, each ``{field}`` filled from the step's params.
        family: Whether the name is the prefix of a family of columns.
    """

    template: str
    family: bool = False
    kind: Literal["template"] = "template"


type ColumnRef = Annotated[
    Column | ParamColumn | Prefix | Template, Field(discriminator="kind")
]
"""One reference to a column or a family of columns."""


@dataclass(frozen=True, slots=True)
class AnyOf:
    """A requirement that any one of several columns meets."""

    of: tuple[ColumnRef, ...]
    kind: Literal["any_of"] = "any_of"


@dataclass(frozen=True, slots=True)
class Derived:
    """Columns that ``apply`` writes from other columns, one per mapping in a params field.

    The field contains a list of mappings, as ``category_specs`` of ``nn-delta-bins``
    does. Each mapping names the column it writes under *name_key* and the column
    it reads under *source_key*. ``apply`` derives the columns in list order and
    skips a mapping that leaves either name unset.

    As an entry of ``requires``, it requires the column that each mapping reads,
    unless an earlier mapping writes it. A column that it writes meets a
    requirement listed after it, because ``apply`` reads such a requirement after
    the derivation. ``apply`` reads a requirement listed before it first, and a
    written column does not meet that one.

    Attributes:
        field: The list-valued params field.
        name_key: The key under which a mapping names the column it writes.
        source_key: The key under which a mapping names the column it reads.
    """

    field: str
    name_key: str
    source_key: str
    kind: Literal["derived"] = "derived"


@dataclass(frozen=True, slots=True)
class Payload:
    """A requirement that any numeric payload column meets.

    A payload column is any column outside ``NON_PAYLOAD_COLS``: the identity
    columns and the bookkeeping ones. ``extract-templates`` samples every
    payload column of its merged input unless its ``columns`` param names some,
    and the global models fitted on the templates read the same columns. A kind
    of column meets the requirement, and a frame without a payload column leaves
    them nothing to read.

    Validation, which reads names alone, counts any column outside
    ``NON_PAYLOAD_COLS``. The check before ``apply`` reads the frame's numeric
    columns.
    """

    kind: Literal["payload"] = "payload"


@dataclass(frozen=True, slots=True)
class Keypoints:
    """Every keypoint coordinate column that a ``PoseConfig`` params field selects.

    The columns are ``<x_prefix><i>`` and ``<y_prefix><i>`` for each index *i*
    in ``pose_indices``, or below ``pose_n`` when ``pose_indices`` is unset
    (:meth:`~mosaic.core.pipeline.types.data_config.PoseConfig.keypoint_columns`).
    Each column is a requirement of its own.

    Attributes:
        field: The params field holding the ``PoseConfig``.
    """

    field: str = "pose"
    kind: Literal["keypoints"] = "keypoints"


@dataclass(frozen=True, slots=True)
class ParamIs:
    """A condition on one params field.

    Attributes:
        field: The params field. A dotted field reads a nested params model.
        equals: The value that the field must hold. ``None`` tests whether the
            field is set (see :func:`is_set`).
        negated: Whether the condition holds in the opposite case: the field
            unset, or holding another value.
    """

    field: str
    equals: str | int | float | None = None
    negated: bool = False


@dataclass(frozen=True, slots=True)
class When:
    """A requirement that ``apply`` reads only while conditions on its params hold.

    ``approach-avoidance`` reads ``ANGLE`` while ``orientation_gate_cos`` is set.
    A condition whose field cannot be read leaves the requirement unknown.

    Attributes:
        requirement: The requirement that applies.
        conditions: The conditions, every one of which must hold.
    """

    requirement: Annotated[
        Column | ParamColumn | Prefix | Template | AnyOf | Keypoints,
        Field(discriminator="kind"),
    ]
    conditions: tuple[ParamIs, ...]
    kind: Literal["when"] = "when"


type Requirement = Annotated[
    Column
    | ParamColumn
    | Prefix
    | Template
    | AnyOf
    | Keypoints
    | Derived
    | Payload
    | When,
    Field(discriminator="kind"),
]
"""One entry of a feature's ``requires``."""


@dataclass(frozen=True, slots=True)
class Conditional:
    """A column that the output contains only under a condition.

    Attributes:
        column: The column, or columns.
        when_param: A params field that must be set (see :func:`is_set`).
        when_present: A column that the merged input must contain. When it and
            *column* both name several columns, they pair by position: the
            i-th column appears when the i-th input column is present.

    With both conditions named, the column appears when both are true. With
    neither, the data decides, and the column is possible whatever the recipe
    says.
    """

    column: ColumnRef
    when_param: str = ""
    when_present: ColumnRef | None = None


@dataclass(frozen=True, slots=True)
class Provides:
    """The columns of a feature's per-entry output.

    An empty declaration (:attr:`is_empty`) means the feature writes only a
    sample, and its table for each entry is empty.

    Attributes:
        passthrough: Whether the output copies every column of its merged input.
        adds: Columns it always writes.
        conditional: Columns it writes under a condition.
        payload_families: Whether the output writes a family of columns per
            payload column of its merged input, each name beginning with that
            column's name. ``temporal-stack`` writes ``speed__t+05`` and
            ``speed__pool_mean`` from ``speed``. Params and data decide the rest
            of each name. A family may include the input column.
    """

    passthrough: bool = False
    adds: tuple[ColumnRef, ...] = ()
    conditional: tuple[Conditional, ...] = ()
    payload_families: bool = False

    @property
    def is_empty(self) -> bool:
        """Whether the declared output is empty, with no passthrough, column or family.

        A feature that writes only a sample declares this. Its table for each
        entry is empty, and a step cannot read it through ``inputs``.
        """
        return self == Provides()


def kept(*names: str) -> tuple[Conditional, ...]:
    """Columns that the output copies from the merged input when it contains them."""
    return tuple(Conditional(Column(name), when_present=Column(name)) for name in names)


META_KEPT: Final[tuple[Conditional, ...]] = kept(*sorted(COLUMNS.meta_set()))
"""The identity columns, copied when present, as a feature joining them back does."""

META_COLUMNS_KEPT: Final[tuple[Conditional, ...]] = kept(*sorted(META_COLS))
"""The identity columns and the pair columns, copied when present.

A feature that rebuilds its output from ``META_COLS`` keeps these, as
``temporal-stack`` does.
"""

NON_PAYLOAD_KEPT: Final[tuple[Conditional, ...]] = kept(*sorted(NON_PAYLOAD_COLS))
"""The identity and bookkeeping columns, copied when present.

A global model's ``apply`` keeps these beside the columns that it writes, as
``global-kmeans`` keeps every column except the payload that it assigns clusters
from.
"""

ORDER_COLUMN: Final = AnyOf((Column(COLUMNS.frame_col), Column(COLUMNS.time_col)))
"""What ``resolve_order_col`` orders rows by: the frame, or else the time."""

_POSE_DEFAULTS: Final = PoseConfig()

POSE_X: Final = Prefix(_POSE_DEFAULTS.x_prefix)
"""The keypoint x columns under ``PoseConfig``'s default prefix, ``poseX*``.

A feature without a ``PoseConfig`` field reads its keypoints under this prefix.
"""

POSE_Y: Final = Prefix(_POSE_DEFAULTS.y_prefix)
"""The keypoint y columns under ``PoseConfig``'s default prefix, ``poseY*``."""


# --- reading a declaration -------------------------------------------------------

_REQUIREMENTS: Final[TypeAdapter[tuple[Requirement, ...]]] = TypeAdapter(
    tuple[Requirement, ...]
)


def declared_requires(owner: object) -> tuple[Requirement, ...] | None:
    """The ``requires`` that *owner* declares, or ``None`` when it declares none.

    Raises:
        TypeError: The attribute is present and is not a tuple of requirements.
    """
    declared: object = getattr(owner, "requires", None)
    if declared is None:
        return None
    try:
        # Strict mode passes only a tuple of the declared classes, unchanged.
        return _REQUIREMENTS.validate_python(declared, strict=True)
    except ValidationError as exc:
        name = getattr(owner, "__name__", type(owner).__name__)
        msg = (
            f"{name}.requires must be a tuple of column requirements, got {declared!r}"
        )
        raise TypeError(msg) from exc


def declared_provides(owner: object) -> Provides | None:
    """The ``provides`` that *owner* declares, or ``None`` when it declares none.

    Raises:
        TypeError: The attribute is present and is not a :class:`Provides`.
    """
    declared: object = getattr(owner, "provides", None)
    if declared is None or isinstance(declared, Provides):
        return declared
    name = getattr(owner, "__name__", type(owner).__name__)
    msg = f"{name}.provides must be a Provides, got {declared!r}"
    raise TypeError(msg)


def _top_field(path: str) -> str:
    """The params field that a dotted path into a nested model starts at."""
    return path.split(".", 1)[0]


def _template_fields(template: str) -> list[str]:
    """The params fields that a template reads, each nested path by its first field."""
    return [
        _top_field(field_name)
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    ]


def _reference_fields(ref: ColumnRef) -> list[str]:
    """The params fields that resolving *ref* reads."""
    if isinstance(ref, Template):
        return _template_fields(ref.template)
    if not isinstance(ref, ParamColumn):
        return []
    found = [ref.field, *_template_fields(ref.fallback)]
    if ref.renamed_by:
        found.append(ref.renamed_by)
    return found


def _members(requirement: Requirement) -> tuple[ColumnRef, ...]:
    """The references of one requirement: an any-of group's members, or itself.

    A conditional requirement contributes the references of the requirement it
    applies. A derivation and a keypoint selection name their columns in a
    params value instead of a reference, and a payload requirement names none.
    """
    if isinstance(requirement, When):
        return _members(requirement.requirement)
    if isinstance(requirement, AnyOf):
        return requirement.of
    if isinstance(requirement, Derived | Payload | Keypoints):
        return ()
    return (requirement,)


def _value_fields(requirement: Requirement) -> list[str]:
    """The params fields that a requirement reads outside its references."""
    if isinstance(requirement, When):
        return [
            *(_top_field(condition.field) for condition in requirement.conditions),
            *_value_fields(requirement.requirement),
        ]
    if isinstance(requirement, Derived | Keypoints):
        return [requirement.field]
    return []


def _references(
    requires: Sequence[Requirement] | None, provides: Provides | None
) -> Iterator[ColumnRef]:
    """Every column reference in the two declarations, conditions included."""
    for requirement in requires or ():
        yield from _members(requirement)
    if provides is None:
        return
    yield from provides.adds
    for conditional in provides.conditional:
        yield conditional.column
        if conditional.when_present is not None:
            yield conditional.when_present


def contract_fields(
    requires: Sequence[Requirement] | None, provides: Provides | None
) -> frozenset[str]:
    """Every params field that the two declarations read."""
    found = [
        field_name
        for ref in _references(requires, provides)
        for field_name in _reference_fields(ref)
    ]
    found.extend(
        field_name
        for requirement in requires or ()
        for field_name in _value_fields(requirement)
    )
    if provides is not None:
        found.extend(
            conditional.when_param
            for conditional in provides.conditional
            if conditional.when_param
        )
    return frozenset(found)


def declared_names(
    requires: Sequence[Requirement] | None, provides: Provides | None
) -> frozenset[str]:
    """The columns that the two declarations name literally, as a :class:`Column`."""
    return frozenset(
        ref.name for ref in _references(requires, provides) if isinstance(ref, Column)
    )


# --- resolving references ---------------------------------------------------------

PAYLOAD_LABEL: Final = "a numeric payload column"
"""How a message names what a :class:`Payload` requirement needs."""

type ParamValues = Mapping[str, JsonValue]
"""A step's params as JSON, every field that a declaration reads included."""

_PARAM_VALUES: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(
    dict[str, JsonValue]
)


def param_values(params: BaseModel, fields: Collection[str]) -> dict[str, JsonValue]:
    """The values of *fields* in a built params model, as JSON.

    A field the model does not declare is left out, which a reference reading it
    resolves as unknown.
    """
    dumped: object = params.model_dump(mode="json", include=set(fields))
    return _PARAM_VALUES.validate_python(dumped)


def is_set(value: JsonValue) -> bool:
    """Whether a params value counts as set: not ``None``, ``False`` or empty."""
    if value is None or value is False:
        return False
    if isinstance(value, str | list | dict):
        return bool(value)
    return True


@dataclass(frozen=True, slots=True)
class NamedColumns:
    """What a reference, or a requirement, names for one step's params.

    Attributes:
        columns: Column names. A requirement is met by any one of them.
        prefixes: Prefixes of column families, met by any column they begin.
        params: The params fields whose values were read.
        unknown: Whether a params value could not be read as column names. The
            reference then names columns that cannot be listed. A requirement
            listed after a derivation that cannot be read is unknown too, because
            the derivation may write its column.
        payload: Whether any payload column meets it, a column outside
            ``NON_PAYLOAD_COLS``. :class:`Payload` resolves to this.
    """

    columns: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    unknown: bool = False
    payload: bool = False

    @property
    def label(self) -> str:
        """The names, quoted and joined with ``or``, as a message prints them."""
        names = [repr(name) for name in self.columns]
        names.extend(repr(f"{prefix}*") for prefix in self.prefixes)
        if self.payload:
            names.append(PAYLOAD_LABEL)
        if names:
            return " or ".join(names)
        fields = ", ".join(self.params)
        return f"the column named by {fields}"

    def met_by(self, present: Collection[str]) -> bool:
        """Whether a frame with the columns *present* meets this requirement."""
        if any(name in present for name in self.columns):
            return True
        if self.payload and any(column not in NON_PAYLOAD_COLS for column in present):
            return True
        return any(
            column.startswith(prefix) for prefix in self.prefixes for column in present
        )


def _listed(value: JsonValue) -> tuple[str, ...] | None:
    """The column names a params value lists, ``()`` when unset, ``None`` when unreadable."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        names = tuple(item for item in value if isinstance(item, str))
        return names if len(names) == len(value) else None
    return None


def _lookup(values: ParamValues, path: str) -> tuple[bool, JsonValue]:
    """The value at a dotted *path* into *values*, and whether the path is there."""
    first, *rest = path.split(".")
    if first not in values:
        return False, None
    value = values[first]
    for part in rest:
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _render(template: str, values: ParamValues) -> str | None:
    """*template* with each ``{field}`` filled from *values*, or ``None`` when one is unset."""
    parts: list[str] = []
    for literal, field_name, _, _ in string.Formatter().parse(template):
        parts.append(literal)
        if not field_name:
            continue
        _, value = _lookup(values, field_name)
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            return None
        if value == "":
            return None
        parts.append(str(value))
    return "".join(parts)


def _renames(ref: ParamColumn, values: ParamValues) -> dict[str, str] | None:
    """The renaming that *ref* applies, ``None`` when its field cannot be read."""
    if not ref.renamed_by:
        return {}
    mapping = values.get(ref.renamed_by)
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        return None
    renames = {key: value for key, value in mapping.items() if isinstance(value, str)}
    return renames if len(renames) == len(mapping) else None


def resolve_names(ref: ColumnRef, values: ParamValues) -> NamedColumns:
    """The columns that *ref* names under one step's params."""
    if isinstance(ref, Column):
        return NamedColumns(columns=(ref.name,))
    if isinstance(ref, Prefix):
        return NamedColumns(prefixes=(ref.prefix,))
    read = tuple(_reference_fields(ref))
    if isinstance(ref, Template):
        rendered = _render(ref.template, values)
        if rendered is None:
            return NamedColumns(params=read, unknown=True)
        if ref.family:
            return NamedColumns(prefixes=(rendered,), params=read)
        return NamedColumns(columns=(rendered,), params=read)
    unknown = NamedColumns(params=read, unknown=True)
    if ref.field not in values:
        return unknown
    value = values[ref.field]
    listed = _listed(value)
    if listed is None:
        return unknown
    if not listed:
        if ref.fallback:
            rendered = _render(ref.fallback, values)
            if rendered is None:
                return unknown
            listed = (rendered,)
        elif ref.unset == "unknown" and value is None:
            return unknown
        else:
            return NamedColumns(params=read)
    renames = _renames(ref, values)
    if renames is None:
        return unknown
    names = tuple(f"{ref.prefix}{renames.get(name, name)}" for name in listed)
    return NamedColumns(columns=names, params=read)


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    """*items* with repeats removed, in first-seen order."""
    return tuple(dict.fromkeys(items))


def _derivations(
    derived: Derived, values: ParamValues
) -> tuple[tuple[str, str], ...] | None:
    """The ``(written, read)`` column pairs that *derived* lists, in list order.

    ``None`` when its params value cannot be read as mappings of column names.
    """
    if derived.field not in values:
        return None
    value = values[derived.field]
    if value is None:
        return ()
    if not isinstance(value, list):
        return None
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        written = item.get(derived.name_key)
        read = item.get(derived.source_key)
        if not written or not read:
            continue
        if not isinstance(written, str) or not isinstance(read, str):
            return None
        pairs.append((written, read))
    return tuple(pairs)


def _resolved(requirement: AnyOf | ColumnRef, values: ParamValues) -> NamedColumns:
    """One reference or any-of group, resolved against one step's params."""
    resolved = [resolve_names(member, values) for member in _members(requirement)]
    return NamedColumns(
        columns=_dedupe(name for one in resolved for name in one.columns),
        prefixes=_dedupe(prefix for one in resolved for prefix in one.prefixes),
        params=_dedupe(field for one in resolved for field in one.params),
        unknown=any(one.unknown for one in resolved),
    )


def _pose_config(value: JsonValue) -> PoseConfig | None:
    """*value* read as a ``PoseConfig``, ``None`` when it is not one."""
    if not isinstance(value, dict):
        return None
    try:
        return PoseConfig.model_validate(value)
    except ValidationError:
        return None


def _keypoint_needs(requirement: Keypoints, values: ParamValues) -> list[NamedColumns]:
    """One requirement per keypoint column that *requirement* selects.

    One unknown requirement when its field does not hold a ``PoseConfig``.
    """
    params = (requirement.field,)
    _, value = _lookup(values, requirement.field)
    config = _pose_config(value)
    if config is None:
        return [NamedColumns(params=params, unknown=True)]
    xs, ys = config.keypoint_columns()
    return [
        NamedColumns(columns=(name,), params=params)
        for pair in zip(xs, ys, strict=True)
        for name in pair
    ]


def _holds(condition: ParamIs, values: ParamValues) -> bool | None:
    """Whether *condition* holds under *values*, ``None`` when its field is not there."""
    present, value = _lookup(values, condition.field)
    if not present:
        return None
    holds = is_set(value) if condition.equals is None else value == condition.equals
    return holds != condition.negated


def needs_of(
    requires: Sequence[Requirement], values: ParamValues
) -> tuple[NamedColumns, ...]:
    """Each requirement in *requires*, resolved against one step's params.

    A requirement is left out when its params field is unset, as an unset
    ``subgroup_col`` is, or when a condition on it does not hold. A requirement
    is also left out when a derivation listed before it writes a column that
    meets it. A derivation contributes one requirement per column that it reads
    from the input, and a keypoint selection one per column that it selects. A
    payload requirement resolves to one that any payload column meets.
    """
    found: list[NamedColumns] = []
    written: set[str] = set()
    derivation_unknown = False
    for requirement in requires:
        if isinstance(requirement, Derived):
            pairs = _derivations(requirement, values)
            if pairs is None:
                derivation_unknown = True
                found.append(NamedColumns(params=(requirement.field,), unknown=True))
                continue
            read: list[str] = []
            for name, source in pairs:
                if source not in written:
                    read.append(source)
                written.add(name)
            found.extend(
                NamedColumns(columns=(source,), params=(requirement.field,))
                for source in _dedupe(read)
            )
            continue
        unknown = derivation_unknown
        applied: Column | ParamColumn | Prefix | Template | AnyOf | Keypoints | Payload
        if isinstance(requirement, When):
            outcomes = [
                _holds(condition, values) for condition in requirement.conditions
            ]
            if any(outcome is False for outcome in outcomes):
                continue
            unknown = unknown or None in outcomes
            applied = requirement.requirement
        else:
            applied = requirement
        if isinstance(applied, Keypoints):
            needs = _keypoint_needs(applied, values)
        elif isinstance(applied, Payload):
            needs = [NamedColumns(payload=True)]
        else:
            needs = [_resolved(applied, values)]
        for need in needs:
            if need.met_by(written):
                continue
            if unknown:
                need = replace(need, unknown=True)
            if need.columns or need.prefixes or need.unknown or need.payload:
                found.append(need)
    return tuple(found)


# --- the check before apply ---------------------------------------------------------


class MissingInputColumns(ValueError):
    """A feature's merged input lacks a column that its declaration requires.

    Raised per entry before ``apply``, and ``run_feature`` records it as that
    entry's failure. Every feature's message has one form, naming the missing
    columns and the merged inputs.

    Attributes:
        feature: The feature's name.
        missing: The requirements that the frame does not meet.
        inputs: The step's inputs, as ``tracks`` or the producing feature.
    """

    def __init__(
        self, feature: str, missing: Sequence[NamedColumns], inputs: Sequence[str]
    ) -> None:
        self.feature: str = feature
        self.missing: tuple[NamedColumns, ...] = tuple(missing)
        self.inputs: tuple[str, ...] = tuple(inputs)
        listed = "; ".join(need.label for need in self.missing)
        merged = ", ".join(self.inputs) or "no inputs"
        message = (
            f"{feature} cannot read its input. The frame merged from {merged} "
            f"lacks {listed}"
        )
        super().__init__(message)


def check_input_columns(
    feature: Feature,
    columns: Iterable[object],
    *,
    numeric: Iterable[object] | None = None,
) -> None:
    """Raise unless *columns* meet every requirement that *feature* declares.

    A feature without a ``requires`` declaration passes. So does a requirement
    whose params value cannot be read as column names.

    Args:
        feature: The feature about to apply.
        columns: The columns of its merged frame.
        numeric: The numeric columns among *columns*, which alone meet a
            :class:`Payload` requirement. ``None`` reads every column as
            numeric.

    Raises:
        MissingInputColumns: Naming every requirement the frame does not meet.
    """
    requires = declared_requires(type(feature))
    if requires is None:
        return
    values = param_values(feature.params, contract_fields(requires, None))
    present = frozenset(str(column) for column in columns)
    numbers = present if numeric is None else frozenset(str(name) for name in numeric)
    missing = [
        need
        for need in needs_of(requires, values)
        if not need.unknown and not need.met_by(numbers if need.payload else present)
    ]
    if missing:
        inputs = [
            item if isinstance(item, str) else item.feature
            for item in feature.inputs.root
        ]
        raise MissingInputColumns(feature.name, missing, inputs)
