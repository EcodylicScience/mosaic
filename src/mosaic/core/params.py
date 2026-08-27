"""Provides the vocabulary every parameter model in mosaic declares itself with.

Imports pydantic, the standard library, and StrictModel.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from inspect import cleandoc
from typing import Annotated, Final, Self, TypeAlias

from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from mosaic.core.strict_model import StrictModel

Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
"""The 0..1 bound, named for the parameters that reuse it.

The name does not claim coverage. Four fields are annotated with it, and more
0..1 parameters -- probabilities among them -- spell ``Field(ge=0.0, le=1.0)``
inline instead.

A named ``Annotated`` alias states the bound once, and states the constraint
alone. A description declared here would land inside the ``anyOf`` branch of
every optional field that reuses it, leaving the property itself undescribed, so
each field declares its prose beside the field.
"""


NEEDS_DESCRIPTION: Final = ""
"""Declares a field whose prose is not written yet.

``Declared(NEEDS_DESCRIPTION)`` publishes the description key, so the field is
declared and a client draws a control for it, and publishes it empty, so
``tests/test_params_declaration.py`` counts the field against a ceiling that only
comes down. The two states are separable at the schema: no key at all means no
``Declared``, an empty one means the prose is owed.

Written as a name rather than a bare ``""`` so that deferring is a deliberate act
a reader can grep for.
"""


@dataclass(frozen=True, slots=True)
class Declared:
    """Declares one field's prose and publishes it into that field's schema.

    Goes in the field's ``Annotated``, beside pydantic's own ``Field``::

        track_max_speed: Annotated[
            float | None,
            Field(gt=0.0),
            Declared("the maximum plausible speed", unit="cm/s"),
        ] = None

    ``Field`` in that bracket carries constraints. A ``description=`` on it
    overrides this one silently: pydantic applies ``FieldInfo``'s description
    after the metadata hooks, whichever order the two are written in.
    ``tests/test_params_declaration.py`` asserts ``FieldInfo.description is
    None`` on every field to catch that spelling.

    ``description`` is the first positional parameter and has no default. A
    field that declares a ``Declared`` without an argument fails to type-check
    where it is written; a field whose prose is not written yet passes
    :data:`NEEDS_DESCRIPTION`, which declares it and records the debt.

    Prose given as a value is stored in the class. Python discards an attribute
    docstring at run time, and ``use_attribute_docstrings=True`` recovers one by
    reading the class's source lines, which makes the description a property of
    the source file instead of the class.

    ``mosaic features describe`` and ``mosaic tracking describe`` publish the
    whole schema, description included. ``scripts/gen_docs_reference.py`` reads
    the schema as well, and its parameter table renders name, type, default,
    constraints and the description, with the unit appended in brackets.

    Attributes:
        description: What the parameter means, stated without its unit.
        unit: The quantity's unit, published as ``x-mosaic-unit``. Empty
            publishes no unit key.
        unwired: Why the field reaches nothing, published as
            ``x-mosaic-unwired``. Empty publishes no key. For a field that is
            declared and validated but that no code path reads, so a client
            can refuse to offer a control that changes nothing. The reason
            rather than a flag, because what is missing differs per field.
            Removing such a field is the maintainer's call, so the record sits
            beside it; a ``strict`` xfail test names the wiring it lacks and
            fails the day someone supplies it.
    """

    description: str
    unit: str = ""
    unwired: str = ""

    def __get_pydantic_json_schema__(
        self, source: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Add ``description``, plus ``x-mosaic-unit`` and ``x-mosaic-unwired``.

        Every key sits at the field level, outside the ``anyOf`` an optional
        field renders. A client that unwraps only the non-null branch reads them
        from the property itself.
        """
        schema = handler(source)
        schema["description"] = self.description
        if self.unit:
            schema["x-mosaic-unit"] = self.unit
        if self.unwired:
            schema["x-mosaic-unwired"] = self.unwired
        return schema


class HashExclude:
    """Marker for ``Annotated[T, HASH_EXCLUDE]`` Params fields omitted from the
    run_id hash. Use for any field that does not determine the output: a
    throughput knob (batch sizes, worker counts) that changes runtime only, a
    selector whose selection is hashed by identity instead, or a permission
    whose effect depends on the machine rather than on the value. Folding one in
    moves the identity without the output moving, which is a cache miss costing
    a recompute for nothing. The field still appears in model_dump(),
    params.json, and propagates to workers -- only the run identity hash ignores
    it.
    """

    __slots__ = ()

    def __get_pydantic_json_schema__(
        self, source: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Add ``x-mosaic-hash-exclude`` to the field's schema.

        The hook lives on the marker because a bare ``Annotated`` marker reaches
        ``FieldInfo.metadata`` but not the core schema, whose per-field metadata
        reads ``{}``. Every ``GenerateJsonSchema`` method receives a core schema,
        and a generator subclass therefore cannot read this marker.
        """
        schema = handler(source)
        schema["x-mosaic-hash-exclude"] = True
        return schema


HASH_EXCLUDE = HashExclude()


class Params(StrictModel):
    """Base for every parameter model in mosaic.

    Provides from_overrides() constructor for user-config dicts.
    Subclasses declare feature-specific fields.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Render the fields' declared prose into an ``Attributes:`` section.

        ``help(SomeParams)`` already reports every description, inside the
        ``__init__`` signature pydantic synthesizes -- one line per field
        carrying ``<HashExclude object at 0x...>``, a repr of every ``Declared``
        and pydantic's own ``FieldInfo`` for any aliased constraint. The prose is
        in there and nobody can read it. This is the same prose as a list.

        ``__pydantic_init_subclass__`` rather than a decorator: pydantic
        guarantees ``model_fields`` is complete here, and no model can forget to
        opt in. A decorator would also read as registry membership, which is what
        one means everywhere else in this codebase.

        Written onto the subclass, and read from ``cls.__dict__`` rather than
        ``cls.__doc__``, so a class that declares no docstring of its own neither
        inherits its parent's prose nor has this section appended to it. Prose is
        never invented for such a class: it gets the section alone.

        Fields declared :data:`NEEDS_DESCRIPTION` are left out. An entry with no
        description states nothing, and the ceiling in
        ``tests/test_params_declaration.py`` is what tracks them.
        """
        super().__pydantic_init_subclass__(**kwargs)
        own = cls.__dict__.get("__doc__")
        prose = cleandoc(own) if isinstance(own, str) and own.strip() else ""
        # Kept before ``__doc__`` is rewritten below, because that is the only
        # moment the model's own prose exists apart from the generated section.
        # ``__get_pydantic_json_schema__`` publishes this rather than
        # ``__doc__``, which pydantic would otherwise copy whole.
        cls.__mosaic_prose__ = prose
        described = [
            (name, declared)
            for name, info in cls.model_fields.items()
            for declared in info.metadata
            if isinstance(declared, Declared) and declared.description
        ]
        if not described:
            return
        lines = ["Attributes:"]
        for name, declared in described:
            unit = f" [{declared.unit}]" if declared.unit else ""
            body = f"{name}: {declared.description}{unit}"
            lines += textwrap.wrap(
                body, width=79, initial_indent="    ", subsequent_indent="        "
            )
        head = f"{prose}\n\n" if prose else ""
        cls.__doc__ = head + "\n".join(lines)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, source: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Publish the model's own prose as the schema description, alone.

        pydantic copies ``__doc__`` into the schema verbatim, and
        ``__pydantic_init_subclass__`` appends a rendered ``Attributes:`` section
        to it. Left alone, every field's description is published twice: once
        under ``properties`` where a client reads it, and once inside a
        multi-kilobyte newline-laden string at the top. The generated section is
        for ``help()`` in a REPL and a schema client has the structured form
        already.

        A model with no prose of its own publishes no description rather than
        its parent's: the same rule ``__doc__`` follows above.
        """
        schema = handler(source)
        prose = cls.__dict__.get("__mosaic_prose__")
        if prose:
            schema["description"] = prose
        else:
            _ = schema.pop("description", None)
        return schema

    def identity_dump(self) -> dict[str, object]:
        """model_dump() minus HASH_EXCLUDE-marked fields -- the run_id hash
        input.

        Fields tagged ``Annotated[T, HASH_EXCLUDE]`` are throughput-only knobs
        that don't affect output, so they're stripped here to keep them out of
        run identity. Top-level fields only (throughput knobs are top-level);
        nested config models are not recursed into.
        """
        dumped = self.model_dump()
        for name, info in type(self).model_fields.items():
            if any(isinstance(m, HashExclude) for m in info.metadata):
                dumped.pop(name, None)
        return dumped

    @classmethod
    def from_overrides(cls, overrides: dict[str, object] | None = None) -> Self:
        """Construct from user dict; missing keys get field defaults.

        For BaseModel-typed fields with default_factory, partial dict overrides
        are merged on top of the default before validation (1-level deep merge).
        This replaces the per-feature deep-merge hacks in global_ward.
        """
        if not overrides:
            return cls()
        merged = dict(overrides)
        for key, value in list(merged.items()):
            if not isinstance(value, dict):
                continue
            field_info = cls.model_fields.get(key)
            if field_info is None or field_info.default_factory is None:
                continue
            default_obj: object = field_info.get_default(call_default_factory=True)
            if isinstance(default_obj, BaseModel):
                merged[key] = {**default_obj.model_dump(), **value}
        return cls.model_validate(merged)
