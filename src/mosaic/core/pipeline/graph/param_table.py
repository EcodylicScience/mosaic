"""A step type's params, read off its JSON Schema as rows a reader can scan.

A params model's JSON Schema answers what a client may send, and it is long: the
largest dumps to tens of kilobytes, most of it ``title`` keywords that repeat the
field names and class docstrings copied into ``$defs``. A person choosing a value,
or a model writing a recipe, needs each field's path, type, default, constraints
and meaning. :func:`param_rows` gives one row per field, nested models flattened
to dotted paths, and :func:`compact_params_schema` gives the schema without what
repeats.

The docs generator (``scripts/gen_docs_reference.py``) renders its parameter
tables with the same helpers, passing :data:`MARKDOWN` where this package passes
:data:`PLAIN`, so the reference pages and the catalog describe a type in one
vocabulary.

Standard library only, at module scope. The MCP tool contract imports
:func:`strip_schema_keywords`, and its import budget excludes pandas and the op
registry, which :mod:`~mosaic.core.pipeline.graph.wire` loads.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mosaic.core.json_value import JsonValue

if TYPE_CHECKING:
    from .wire import ParamRow, ReferenceKind

__all__ = [
    "CONSTRAINT_LABELS",
    "MARKDOWN",
    "MAX_PARAM_DEPTH",
    "PLAIN",
    "Markup",
    "compact_params_schema",
    "constraints_text",
    "description_text",
    "is_model",
    "model_name",
    "param_rows",
    "strip_schema_keywords",
    "type_text",
]


@dataclass(frozen=True, slots=True)
class Markup:
    """How a rendered type or description marks code and emphasis.

    Attributes:
        code: Wraps a name or a literal value.
        strong: Wraps a label that must stand out.
    """

    code: Callable[[str], str]
    strong: Callable[[str], str]


def _unchanged(text: str) -> str:
    return text


MARKDOWN: Final = Markup(
    code=lambda text: f"`{text}`", strong=lambda text: f"**{text}**"
)
"""Markdown markup, as the generated reference pages print it."""

PLAIN: Final = Markup(code=_unchanged, strong=_unchanged)
"""No markup, as a JSON answer carries the text."""

MAX_PARAM_DEPTH: Final = 3
"""The most path components of a row :func:`param_rows` gives a nested field."""

# Ordered so a page reads constraint-tightest first, and closed so an unknown
# keyword is dropped rather than rendered as a bare JSON fragment.
CONSTRAINT_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("minimum", ">="),
    ("exclusiveMinimum", ">"),
    ("maximum", "<="),
    ("exclusiveMaximum", "<"),
    ("minLength", "min length"),
    ("maxLength", "max length"),
    ("minItems", "min items"),
    ("maxItems", "max items"),
    ("pattern", "matches"),
)


def model_name(raw: str) -> str:
    """A `$defs` key, tidied without being renamed.

    pydantic spells a parameterized generic by flattening its arguments into the
    key -- `ArtifactSpec_ParquetLoadSpec_DataFrame_` -- and the trailing
    separator reads as a typo. Only that is trimmed, and the same function is
    used for the type column and for the collapsed block's heading, so the two
    always name the same thing.
    """
    return raw.rstrip("_")


def is_model(spec: Mapping[str, JsonValue]) -> bool:
    """Return whether a `$defs` entry is a model with fields.

    pydantic also writes a `$defs` entry for a named type alias: a `Literal`
    alias, a discriminated union of models, or the recursive `JsonValue`. Such
    an entry does not declare `properties`. It is a type, and the type column writes
    it out wherever a field uses it.
    """
    return isinstance(spec.get("properties"), dict)


def type_text(
    spec: Mapping[str, JsonValue],
    aliases: Mapping[str, Mapping[str, JsonValue]],
    expanding: frozenset[str] = frozenset(),
    *,
    markup: Markup = PLAIN,
) -> str:
    """One JSON-Schema property reduced to a readable type expression.

    `anyOf` is what pydantic emits for `X | None`, which most optional fields
    are; echoing the full two-branch union would drown the table. `oneOf` is
    what a tagged union emits, and its branches are the models a client picks
    between. A `$ref` names a nested model. A `$ref` to a type alias in
    *aliases* is replaced by the type that the alias names. *expanding* names the
    aliases already being written out, and a recursive alias met inside itself
    is written as its name. *markup* wraps each name and literal.
    """
    code = markup.code
    ref = spec.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        alias = aliases.get(name)
        if alias is not None and name not in expanding:
            return type_text(alias, aliases, expanding | {name}, markup=markup)
        return code(model_name(name))
    for key in ("anyOf", "oneOf"):
        variants = spec.get(key)
        if isinstance(variants, list):
            # `dict.fromkeys` rather than `set`, to keep the declared branch
            # order: a set would reorder `str | None` between runs and flap
            # `--check`.
            parts = [
                type_text(v, aliases, expanding, markup=markup)
                for v in variants
                if isinstance(v, dict)
            ]
            return " | ".join(dict.fromkeys(parts))
    if "const" in spec:
        return code(json.dumps(spec["const"]))
    choices = spec.get("enum")
    if isinstance(choices, list):
        return " | ".join(code(json.dumps(c)) for c in choices)
    fixed = spec.get("prefixItems")
    if isinstance(fixed, list):
        inner = ", ".join(
            type_text(f, aliases, expanding, markup=markup)
            for f in fixed
            if isinstance(f, dict)
        )
        return f"tuple of ({inner})" if inner else code("tuple")
    kind = spec.get("type")
    if kind == "array":
        items = spec.get("items")
        inner = (
            type_text(items, aliases, expanding, markup=markup)
            if isinstance(items, dict)
            else "any"
        )
        # `list of A | None` reads as a list or None. A union of item types is
        # parenthesized, to read as a list of either.
        if " | " in inner:
            inner = f"({inner})"
        return f"list of {inner}"
    if kind == "object":
        return code("object")
    if kind == "null":
        return code("None")
    if isinstance(kind, str):
        return code(kind)
    return "any"


def constraints_text(spec: Mapping[str, JsonValue], *, markup: Markup = PLAIN) -> str:
    """The validation keywords of one property, as ``>= 1, max length 8``."""
    return ", ".join(
        f"{label} {markup.code(str(spec[key]))}"
        for key, label in CONSTRAINT_LABELS
        if key in spec
    )


def description_text(spec: Mapping[str, JsonValue], *, markup: Markup = PLAIN) -> str:
    """A field's declared prose, its unit, and why nothing reads it.

    Every key comes from `Declared`, which is mosaic's own text rather than
    anything pydantic spells, so none moves when a pydantic release changes
    how it renders a union. A field declared `NEEDS_DESCRIPTION` renders an empty
    cell, which is what it has to say.

    `x-mosaic-unwired` is rendered because a parameter table is what a reader
    consults before setting a parameter. Printing the prose alone would describe
    a control that changes nothing and say nothing about it.
    """
    description = str(spec.get("description", "") or "").strip()
    unit = str(spec.get("x-mosaic-unit", "") or "").strip()
    unwired = str(spec.get("x-mosaic-unwired", "") or "").strip()
    if description and unit:
        text = f"{description} [{unit}]"
    else:
        text = description or (f"[{unit}]" if unit else "")
    if not unwired:
        return text
    note = f"{markup.strong('Unwired:')} {unwired}."
    return f"{text} {note}" if text else note


_SCHEMA_MAPS: Final = frozenset({"properties", "patternProperties", "$defs"})
"""The keywords whose value maps names to schemas. The names are not keywords."""

_DATA_KEYWORDS: Final = frozenset({"default", "const", "enum", "examples"})
"""The keywords whose value is data, which is copied as it is."""


def strip_schema_keywords(
    schema: Mapping[str, JsonValue], dropped: frozenset[str]
) -> dict[str, JsonValue]:
    """Return *schema* without the keywords in *dropped*, in every subschema.

    A name in a ``properties``, ``patternProperties`` or ``$defs`` map is a field
    or model name and is kept, so a property named ``title`` survives dropping
    the ``title`` keyword. A keyword whose value is data (``default``,
    ``const``, ``enum``, ``examples``) is copied unread.
    """
    kept: dict[str, JsonValue] = {}
    for keyword, value in schema.items():
        if keyword in dropped:
            continue
        if keyword in _DATA_KEYWORDS:
            kept[keyword] = value
        elif keyword in _SCHEMA_MAPS and isinstance(value, dict):
            kept[keyword] = {
                name: _subschema_without(subschema, dropped)
                for name, subschema in value.items()
            }
        else:
            kept[keyword] = _subschema_without(value, dropped)
    return kept


def _subschema_without(value: JsonValue, dropped: frozenset[str]) -> JsonValue:
    """Apply :func:`strip_schema_keywords` to each schema that *value* holds."""
    if isinstance(value, dict):
        return strip_schema_keywords(value, dropped)
    if isinstance(value, list):
        return [_subschema_without(item, dropped) for item in value]
    return value


def compact_params_schema(schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a params JSON Schema without what repeats or documents classes.

    Every ``title`` keyword goes, which repeats a field or model name. So does
    the ``description`` of the schema itself and of each ``$defs`` entry, which
    pydantic copies from a class docstring written for the code's readers.
    Each property keeps its description, which says what the field means.
    """
    compact = strip_schema_keywords(schema, frozenset({"title"}))
    compact.pop("description", None)
    nested = compact.get("$defs")
    if isinstance(nested, dict):
        compact["$defs"] = {
            name: (
                {key: value for key, value in spec.items() if key != "description"}
                if isinstance(spec, dict)
                else spec
            )
            for name, spec in nested.items()
        }
    return compact


def _required_names(schema: Mapping[str, JsonValue]) -> frozenset[str]:
    names = schema.get("required")
    if not isinstance(names, list):
        return frozenset()
    return frozenset(name for name in names if isinstance(name, str))


def _nested_model(
    spec: Mapping[str, JsonValue], models: Mapping[str, Mapping[str, JsonValue]]
) -> Mapping[str, JsonValue] | None:
    """The model a property holds, read through ``X | None``, or ``None``.

    An array, a map or a union of models is not a nested model: its rows would
    hold one value per item or per branch, which a dotted path cannot name.
    """
    ref = spec.get("$ref")
    if isinstance(ref, str):
        return models.get(ref.rsplit("/", 1)[-1])
    variants = spec.get("anyOf")
    if not isinstance(variants, list):
        return None
    branches = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
    if len(branches) != 1 or len(branches) == len(variants):
        return None
    return _nested_model(branches[0], models)


def _result_type(
    spec: Mapping[str, JsonValue], models: Mapping[str, Mapping[str, JsonValue]]
) -> str:
    """How a recipe writes a run reference into a field that holds one."""
    target = _nested_model(spec, models)
    properties = target.get("properties") if target is not None else None
    takes_pattern = isinstance(properties, dict) and "pattern" in properties
    if takes_pattern:
        return '{"step": "<id>", "pattern"?: "<glob>"}'
    return '{"step": "<id>"}'


def param_rows(
    schema: Mapping[str, JsonValue],
    reference_fields: Mapping[str, ReferenceKind] | None = None,
) -> tuple[ParamRow, ...]:
    """One row per params field of *schema*, nested models as dotted paths.

    A field holding a model, or ``None`` and a model, is listed and followed by
    a row per field of that model, ``fit.perplexity``, to at most
    :data:`MAX_PARAM_DEPTH` components. A list, a map or a union of models is
    one row, typed by its model's name, and the schema describes it.

    Args:
        schema: A params model's JSON Schema.
        reference_fields: The top-level fields that accept a step reference.
            A ``result`` field is typed as the reference a recipe writes and is
            not expanded. An ``identifier`` field keeps its own type.

    Returns:
        The rows, in declaration order, each nested row after its parent.
    """
    from .wire import ParamRow

    references = reference_fields or {}
    nested = schema.get("$defs")
    defs: dict[str, Mapping[str, JsonValue]] = {
        name: spec
        for name, spec in (nested.items() if isinstance(nested, dict) else ())
        if isinstance(spec, dict)
    }
    models = {name: spec for name, spec in defs.items() if is_model(spec)}
    aliases = {name: spec for name, spec in defs.items() if not is_model(spec)}
    rows: list[ParamRow] = []

    def walk(model: Mapping[str, JsonValue], prefix: str, depth: int) -> None:
        properties = model.get("properties")
        if not isinstance(properties, dict):
            return
        required = _required_names(model)
        for name, raw in properties.items():
            spec: Mapping[str, JsonValue] = raw if isinstance(raw, dict) else {}
            path = f"{prefix}{name}"
            is_result = depth == 1 and references.get(name) == "result"
            rows.append(
                ParamRow(
                    path=path,
                    type=(
                        _result_type(spec, models)
                        if is_result
                        else type_text(spec, aliases)
                    ),
                    required=name in required,
                    constructed=name not in required and "default" not in spec,
                    default=spec.get("default"),
                    constraints=constraints_text(spec),
                    description=description_text(spec),
                    reference="result" if is_result else None,
                )
            )
            inner = None if is_result else _nested_model(spec, models)
            if inner is not None and depth < MAX_PARAM_DEPTH:
                walk(inner, f"{path}.", depth + 1)

    walk(schema, "", 1)
    return tuple(rows)
