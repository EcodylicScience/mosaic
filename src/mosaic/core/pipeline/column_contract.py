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
field (:class:`ParamColumn`), or as a family sharing a prefix (:class:`Prefix`).
A params field resolves from the step's params, which contain the field's default
when the step does not set it. A requirement is a reference, a group that any one
member meets (:class:`AnyOf`), or the columns that ``apply`` derives from other
columns (:class:`Derived`).
"""

from __future__ import annotations

import string
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.types.data_config import COLUMNS

if TYPE_CHECKING:
    from mosaic.core.pipeline.types.feature import Feature

__all__ = [
    "AnyOf",
    "Column",
    "ColumnRef",
    "Conditional",
    "Derived",
    "META_KEPT",
    "MissingInputColumns",
    "NamedColumns",
    "ORDER_COLUMN",
    "ParamColumn",
    "ParamValues",
    "Prefix",
    "Provides",
    "Requirement",
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


type ColumnRef = Annotated[Column | ParamColumn | Prefix, Field(discriminator="kind")]
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


type Requirement = Annotated[
    Column | ParamColumn | Prefix | AnyOf | Derived, Field(discriminator="kind")
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
    """The columns of a feature's non-empty output.

    Attributes:
        passthrough: Whether the output copies every column of its merged input.
        adds: Columns it always writes.
        conditional: Columns it writes under a condition.
    """

    passthrough: bool = False
    adds: tuple[ColumnRef, ...] = ()
    conditional: tuple[Conditional, ...] = ()


def kept(*names: str) -> tuple[Conditional, ...]:
    """Columns that the output copies from the merged input when it contains them."""
    return tuple(Conditional(Column(name), when_present=Column(name)) for name in names)


META_KEPT: Final[tuple[Conditional, ...]] = kept(*sorted(COLUMNS.meta_set()))
"""The identity columns, copied when present, as a feature joining them back does."""

ORDER_COLUMN: Final = AnyOf((Column(COLUMNS.frame_col), Column(COLUMNS.time_col)))
"""What ``resolve_order_col`` orders rows by: the frame, or else the time."""


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


def _template_fields(template: str) -> list[str]:
    """The params fields that a fallback template reads."""
    return [
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    ]


def _reference_fields(ref: ColumnRef) -> list[str]:
    """The params fields that resolving *ref* reads."""
    if not isinstance(ref, ParamColumn):
        return []
    found = [ref.field, *_template_fields(ref.fallback)]
    if ref.renamed_by:
        found.append(ref.renamed_by)
    return found


def _members(requirement: Requirement) -> tuple[ColumnRef, ...]:
    """The references of one requirement: an any-of group's members, or itself.

    A derivation names its columns in a params value instead of a reference.
    """
    if isinstance(requirement, AnyOf):
        return requirement.of
    if isinstance(requirement, Derived):
        return ()
    return (requirement,)


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
        requirement.field
        for requirement in requires or ()
        if isinstance(requirement, Derived)
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
    """

    columns: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    unknown: bool = False

    @property
    def label(self) -> str:
        """The names, quoted and joined with ``or``, as a message prints them."""
        names = [repr(name) for name in self.columns]
        names.extend(repr(f"{prefix}*") for prefix in self.prefixes)
        if names:
            return " or ".join(names)
        fields = ", ".join(self.params)
        return f"the column named by {fields}"

    def met_by(self, present: Collection[str]) -> bool:
        """Whether a frame with the columns *present* meets this requirement."""
        if any(name in present for name in self.columns):
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


def _render(template: str, values: ParamValues) -> str | None:
    """*template* with each ``{field}`` filled from *values*, or ``None`` when one is unset."""
    parts: list[str] = []
    for literal, field_name, _, _ in string.Formatter().parse(template):
        parts.append(literal)
        if not field_name:
            continue
        value = values.get(field_name)
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


def needs_of(
    requires: Sequence[Requirement], values: ParamValues
) -> tuple[NamedColumns, ...]:
    """Each requirement in *requires*, resolved against one step's params.

    A requirement is left out when its params field is unset, as an unset
    ``subgroup_col`` is. A requirement is also left out when a derivation listed
    before it writes a column that meets it. A derivation contributes one
    requirement per column that it reads from the input.
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
        need = _resolved(requirement, values)
        if need.met_by(written):
            continue
        if derivation_unknown:
            need = replace(need, unknown=True)
        if need.columns or need.prefixes or need.unknown:
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


def check_input_columns(feature: Feature, columns: Iterable[object]) -> None:
    """Raise unless *columns* meet every requirement that *feature* declares.

    A feature without a ``requires`` declaration passes. So does a requirement
    whose params value cannot be read as column names.

    Raises:
        MissingInputColumns: Naming every requirement the frame does not meet.
    """
    requires = declared_requires(type(feature))
    if requires is None:
        return
    values = param_values(feature.params, contract_fields(requires, None))
    present = frozenset(str(column) for column in columns)
    missing = [
        need
        for need in needs_of(requires, values)
        if not need.unknown and not need.met_by(present)
    ]
    if missing:
        inputs = [
            item if isinstance(item, str) else item.feature
            for item in feature.inputs.root
        ]
        raise MissingInputColumns(feature.name, missing, inputs)
