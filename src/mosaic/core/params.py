"""Provides the vocabulary a mosaic model declares its fields with.

:class:`Declared` and :class:`HashExclude` are that vocabulary, and
:class:`DeclaredModel` is the base that reads it -- taken by parameter models
through :class:`Params` and by the configuration models a parameter field
points at, which are not parameter models themselves.

Imports pydantic, the standard library, and StrictModel.
"""

from __future__ import annotations

import textwrap
import types
from dataclasses import dataclass
from inspect import cleandoc
from typing import (
    Annotated,
    Final,
    Literal,
    Self,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.fields import FieldInfo
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


KeypointKind: TypeAlias = Literal["name", "index", "either"]
"""What a keypoint field's value refers to keypoints by.

``name`` for a value that names keypoints, ``index`` for one that indexes them,
``either`` for one that accepts an index or a name.
"""


NEEDS_DESCRIPTION: Final = ""
"""Declares a field whose prose is not written yet.

``Declared(NEEDS_DESCRIPTION)`` publishes the description key, so the field is
declared and a client draws a control for it, and publishes it empty, so
``tests/meta/test_params_declaration.py`` counts the field against a ceiling
that only comes down. The two states are separable at the schema: no key at all
means no ``Declared``, an empty one means the prose is owed.

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
    ``tests/meta/test_params_declaration.py`` asserts ``FieldInfo.description is
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
        keypoint: What the field's value refers to keypoints by, published as
            ``x-mosaic-keypoint``: ``name`` for a value that names keypoints,
            ``index`` for one that indexes them, ``either`` for one that
            accepts an index or a name. ``None`` publishes no key. A client
            offers the dataset's keypoints for such a field. The kind is
            published only and does not reach the run identity.
    """

    description: str
    unit: str = ""
    unwired: str = ""
    keypoint: KeypointKind | None = None

    def __get_pydantic_json_schema__(
        self, source: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Add ``description``, plus the ``x-mosaic-`` key of each fact set.

        Every key sits at the field level, outside the ``anyOf`` an optional
        field renders. A client that unwraps only the non-null branch reads them
        from the property itself.
        """
        schema = handler(source)
        self.publish_into(schema)
        return schema

    def publish_into(self, schema: JsonSchemaValue) -> None:
        """Write the description and each fact set into one field's *schema*."""
        schema["description"] = self.description
        if self.unit:
            schema["x-mosaic-unit"] = self.unit
        if self.unwired:
            schema["x-mosaic-unwired"] = self.unwired
        if self.keypoint is not None:
            schema["x-mosaic-keypoint"] = self.keypoint


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


def declared_field_names(model: type[BaseModel]) -> frozenset[str]:
    """Return every field name that *model* or a pydantic base of it declares.

    A subclass that removed fields from its ``model_fields`` still names them
    here, through the model it narrows. A rule over field names reads this set,
    so that a narrowed subclass reserves the names its base model reserves.
    """
    return frozenset(
        name
        for base in model.__mro__
        if issubclass(base, BaseModel)
        for name in base.model_fields
    )


def has_fields(model: BaseModel | type[BaseModel], *names: str) -> bool:
    """Return whether *model*, or the class of the instance *model*, declares every one of *names*.

    A validator reading another field returns without a verdict when this is
    false. A control plane exposes a subclass without the fields it sets
    itself, and the full model's validation of the completed payload gives the
    verdict.
    """
    fields = (model if isinstance(model, type) else type(model)).model_fields
    return all(name in fields for name in names)


def own_declaration(info: FieldInfo) -> Declared | None:
    """The :class:`Declared` in a field's own ``Annotated``, if it carries one.

    The one place the metadata is searched. Its two readers ask different
    questions of the same answer: :func:`_declared_for` wants the declaration,
    and the schema pass below wants only whether there is one.
    """
    for entry in info.metadata:
        if isinstance(entry, Declared):
            return entry
    return None


def _declared_for(model: type[BaseModel], field: str) -> Declared | None:
    """The :class:`Declared` for *field*, from the nearest class holding one.

    A subclass that re-annotates an inherited field replaces that field's whole
    ``Annotated``, so ``TemplatesRef.load`` carries no ``Declared`` where
    ``ArtifactSpec.load`` declares one. Pydantic keeps a separate
    ``model_fields`` per class, so the declaration is still reachable up the
    method resolution order, and a subclass narrowing a type or pinning a
    default is not asked to restate prose that has not changed.

    A class re-annotating a field *with* a ``Declared`` shadows its ancestors',
    because its own is found first.
    """
    for base in model.__mro__:
        if not issubclass(base, BaseModel):
            continue
        info = base.model_fields.get(field)
        if info is None:
            continue
        declared = own_declaration(info)
        if declared is not None:
            return declared
    return None


class DeclaredModel(StrictModel):
    """Base for any model whose fields carry :class:`Declared`.

    Two hooks, and between them they keep a class's own prose and its fields'
    declarations from being published as one string.
    ``__pydantic_init_subclass__`` renders the declarations into an
    ``Attributes:`` section appended to the subclass's docstring, so ``help()``
    reports them; ``__get_pydantic_json_schema__`` publishes the prose alone as
    the schema description, so a client reads each field's description once, on
    the property that carries it.

    A base of :class:`Params` rather than merged into it because the nested
    configuration models a ``Params`` field points at need the same two hooks
    and are not parameter models: a load spec, a pose column naming scheme and
    a fitting-algorithm configuration are composed into parameters rather than
    being them. The duplication those two hooks prevent is sharpest at that
    depth. A field typed as a model renders as a bare ``$ref``, pydantic writes
    a sibling ``description`` only where the field declares one of its own, and
    a client that merges sibling over target therefore reads the
    class docstring whole.

    Not on :class:`StrictModel`, which is a leaf declaring the pydantic base
    and nothing beyond it. The rendering reads ``Declared``, which lives here.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Render the fields' declared prose into an ``Attributes:`` section.

        ``help()`` on a declaring model already reports every description,
        inside the ``__init__`` signature pydantic synthesizes -- one line per
        field carrying ``<HashExclude object at 0x...>``, a repr of every
        ``Declared`` and pydantic's own ``FieldInfo`` for any aliased
        constraint. The prose is in there and nobody can read it. This is the
        same prose as a list.

        ``__pydantic_init_subclass__`` rather than a decorator: pydantic
        guarantees ``model_fields`` is complete here, and every subclass is
        covered without listing one. A decorator would also read as registry
        membership, which is what one means everywhere else in this codebase.

        Opting in is picking this base, and that is a thing a model can get
        wrong -- a model declaring its fields on top of ``StrictModel``
        publishes each description on its property, because ``Declared`` writes
        those itself, and gets no section at all. The guard named
        ``test_every_model_carrying_a_declared_subclasses_declared_model`` is
        what refuses it.

        Written onto the subclass, and read from ``cls.__dict__`` rather than
        ``cls.__doc__``, so a class that declares no docstring of its own neither
        inherits its parent's prose nor has this section appended to it. Prose
        is never invented for such a class: it gets the section alone.

        Fields declared :data:`NEEDS_DESCRIPTION` are left out. An entry with
        no description states nothing, and the ceiling in
        ``tests/meta/test_params_declaration.py`` is what tracks them.
        """
        super().__pydantic_init_subclass__(**kwargs)
        own = cls.__dict__.get("__doc__")
        prose = cleandoc(own) if isinstance(own, str) and own.strip() else ""
        # Kept before ``__doc__`` is rewritten below, because that is the only
        # moment the model's own prose exists apart from the generated section.
        # ``__get_pydantic_json_schema__`` publishes this rather than
        # ``__doc__``, which pydantic would otherwise copy whole.
        cls.__mosaic_prose__ = prose
        described: list[tuple[str, Declared]] = []
        for name in cls.model_fields:
            declared = _declared_for(cls, name)
            if declared is not None and declared.description:
                described.append((name, declared))
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
        """Publish the model's own prose, and describe its inherited fields.

        pydantic copies ``__doc__`` into the schema verbatim, and
        ``__pydantic_init_subclass__`` appends a rendered ``Attributes:`` section
        to it. Left alone, every field's description is published twice: once
        under ``properties`` where a client reads it, and once inside a
        multi-kilobyte newline-laden string at the top. The generated section
        is for ``help()`` in a REPL, and a schema client already holds the
        structured form of it.

        A model with no prose of its own publishes no description rather than
        its parent's: the same rule ``__doc__`` follows above.

        The per-field pass writes what :func:`declared_for` finds for a field
        whose own metadata carries none. It has to happen here because
        ``Declared.__get_pydantic_json_schema__`` is the hook that publishes a
        description, and that hook never runs for a field carrying no
        ``Declared``. A property already holding a description is left alone.

        The pass is skipped when ``handler`` returns a bare ``$ref`` rather
        than the expanded object, which is a shape this writes nothing into
        rather than guesses at.

        One asymmetry to know about: ``handle_ref_overrides`` has already run
        by the time this writes, so a description written here survives even
        where it equals the referenced schema's, while a declared one in that
        position is deleted. Nothing in the tree is in that position, and no
        check guards against it.
        """
        schema = handler(source)
        prose = cls.__dict__.get("__mosaic_prose__")
        if prose:
            schema["description"] = prose
        else:
            _ = schema.pop("description", None)
        if "properties" not in schema:
            return schema
        properties: dict[str, JsonSchemaValue] = schema["properties"]
        for name, info in cls.model_fields.items():
            if own_declaration(info) is not None:
                continue
            declared = _declared_for(cls, name)
            if declared is None:
                continue
            field_schema = properties.get(name)
            if field_schema is None or "description" in field_schema:
                continue
            declared.publish_into(field_schema)
        return schema


def _sole_model_type(annotation: object) -> type[BaseModel] | None:
    """The one model type a field annotation admits, or ``None``.

    Strips ``Annotated`` wrappers at any depth and drops ``None`` from a union.
    Returns the remaining member when exactly one remains and it is a
    ``BaseModel`` subclass. A union of several models, or of a model and a
    non-model type such as a mapping, returns ``None``.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in (Union, types.UnionType):
        members = [
            member for member in get_args(annotation) if member is not type(None)
        ]
        if len(members) != 1:
            return None
        return _sole_model_type(members[0])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


class NestedParams(DeclaredModel):
    """Base for a model a :class:`Params` field holds that names its own identity.

    :meth:`Params.identity_dump` dumps such a field's value through the value's
    :meth:`identity_dump`, so a nested model can leave a field out of a run's
    identity, as a labels source leaves out a ``remap`` it does not set.
    """

    def identity_dump(self) -> dict[str, object]:
        """The model's part of the run_id hash input: its whole ``model_dump()``."""
        return self.model_dump()


class Params(DeclaredModel):
    """Base for every parameter model in mosaic.

    Provides from_overrides() constructor for user-config dicts.
    Subclasses declare feature-specific fields.
    """

    def identity_dump(self) -> dict[str, object]:
        """model_dump() minus HASH_EXCLUDE-marked fields -- the run_id hash
        input.

        Fields tagged ``Annotated[T, HASH_EXCLUDE]`` are throughput-only knobs
        that don't affect output, so they're stripped here to keep them out of
        run identity. Top-level fields only (throughput knobs are top-level).
        A top-level field whose value is a :class:`NestedParams` dumps as that
        value's own ``identity_dump()``, one level deep. Every other nested
        model dumps whole.
        """
        dumped = self.model_dump()
        for name, info in type(self).model_fields.items():
            if any(isinstance(m, HashExclude) for m in info.metadata):
                dumped.pop(name, None)
                continue
            value: object = getattr(self, name, None)
            if isinstance(value, NestedParams):
                dumped[name] = value.identity_dump()
        return dumped

    @classmethod
    def from_overrides(cls, overrides: dict[str, object] | None = None) -> Self:
        """Construct from user dict; missing keys get field defaults.

        A dict override of a field typed as one model, with or without ``None``,
        and with a factory default, is merged on top of that default before
        validation (1-level deep merge). A dict override of any other field is
        validated as given: a field admitting several model types, or a model
        beside a non-model member such as a mapping, has no single default to
        merge into. A partial dict for such a field names its tag, for example
        the ``kind`` of a discriminated union.
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
            model_type = _sole_model_type(field_info.annotation)
            if model_type is None:
                continue
            default_obj: object = field_info.get_default(call_default_factory=True)
            if isinstance(default_obj, model_type):
                merged[key] = {**default_obj.model_dump(), **value}
        return cls.model_validate(merged)
