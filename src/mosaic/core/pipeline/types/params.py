from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Generic, Self

from pydantic import BaseModel, Field, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from typing_extensions import TypeVar

from mosaic.core.entry import Entry
from mosaic.core.pipeline._loaders import StrictModel
from mosaic.core.pipeline.types.artifacts import JoblibArtifact, TemplatesRef

# ``JsonValue`` used to live here. It moved to ``mosaic.core.json_value`` -- a
# module with no imports at all -- because the dataset manifest needs the same
# type and importing this one drags the loader and artifact machinery, and
# through them pandas, into a manifest read. The package ``__init__`` re-exports
# it from its new home, so every existing import path is unchanged.


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
    field that declares a ``Declared`` without prose fails to type-check where
    it is written.

    Prose given as a value is stored in the class. Python discards an attribute
    docstring at run time, and ``use_attribute_docstrings=True`` recovers one by
    reading the class's source lines, which makes the description a property of
    the source file instead of the class.

    ``mosaic features describe`` and ``mosaic tracking describe`` publish the
    whole schema and carry the description to a reader.
    ``scripts/gen_docs_reference.py`` reads the schema as well, but its
    parameter table holds name, type, default and constraints alone and drops
    the description.

    Attributes:
        description: What the parameter means, stated without its unit.
        unit: The quantity's unit, published as ``x-mosaic-unit``. Empty
            publishes no unit key.
    """

    description: str
    unit: str = ""

    def __get_pydantic_json_schema__(
        self, source: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Add ``description``, and ``x-mosaic-unit`` where a unit is declared.

        Both keys sit at the field level, outside the ``anyOf`` an optional
        field renders. A client that unwraps only the non-null branch reads them
        from the property itself.
        """
        schema = handler(source)
        schema["description"] = self.description
        if self.unit:
            schema["x-mosaic-unit"] = self.unit
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
    """Base for all feature parameter models.

    Provides from_overrides() constructor for user-config dicts.
    Subclasses declare feature-specific fields.
    """

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
            default_obj: object = field_info.get_default(call_default_factory=True)  # pyright: ignore[reportAny]
            if isinstance(default_obj, BaseModel):
                merged[key] = {**default_obj.model_dump(), **value}
        return cls.model_validate(merged)


_ENTRIES_DESCRIPTION = (
    "Which (group, sequence) entries the run covers. Unset covers every entry "
    "the media index holds."
)

_OVERWRITE_DESCRIPTION = (
    "Recompute an entry whose output is already on disk, instead of keeping "
    "what is there and reporting it done."
)


class OpParams(Params):
    """Base for every registered op's parameters: what to run over, and re-runs.

    Both fields are ``HASH_EXCLUDE``. A selector names which entries a run
    covers rather than what comes out of any one of them, and an op whose
    coverage does change its output hashes the resolved entry identities in
    ``plan_identity`` instead -- which is also what keeps a sequence rename from
    moving a run identifier.

    One selector, and it enumerates. A ``groups`` field beside a ``sequences``
    one expresses a cross product, so it cannot name group A sequence 1 together
    with group B sequence 2 without also naming A/2 and B/1. ``mosaic track``
    takes the pair as flags and ``mosaic run --kind`` takes it inside
    ``--params``; both enumerate it through
    :meth:`~mosaic.core.dataset.Dataset.expand_media_scope` before an op's model
    validates.

    An op that refuses an unscoped run declares its own required ``entries`` on
    :class:`Params` instead of inheriting this one, because a subclass can
    neither drop an inherited default nor narrow ``list[Entry] | None`` to
    ``list[Entry]`` -- both are override errors under a strict type checker.
    :class:`~mosaic.core.pipeline.transcode.TranscodeParams` is the one such op.

    Attributes:
        entries: The ``(group, sequence)`` pairs to run over, or ``None`` for
            every indexed entry.
        overwrite: Whether an entry already on disk is recomputed.
    """

    entries: Annotated[
        list[Entry] | None, HASH_EXCLUDE, Declared(_ENTRIES_DESCRIPTION)
    ] = None
    overwrite: Annotated[bool, HASH_EXCLUDE, Declared(_OVERWRITE_DESCRIPTION)] = False


M = TypeVar("M", bound=JoblibArtifact[object], default=JoblibArtifact[object])
T = TypeVar("T", bound=TemplatesRef, default=TemplatesRef)


class GlobalModelParams(Params, Generic[M, T]):
    """Base params for global features that fit on a templates artifact
    or load a pre-fitted model.

    Type parameter M is the model artifact type (must extend JoblibArtifact), and
    T the templates artifact type (must extend TemplatesRef). Exactly one of
    `templates` or `model` must be provided.

    **Both are pinned types rather than the bare aliases, and that is the whole
    defence of the artifact edge.** An ``ArtifactSpec`` with no pattern derives
    ``*.<load kind>``, and a producer's run root holds one per-entry output parquet
    per sequence beside its named artifacts -- so a generic ``ParquetArtifact`` here
    resolved whichever file sorted first, which is a per-entry table, and nothing
    downstream could tell. Naming the type puts the filename in the declaration,
    where it costs nothing at run time and cannot be forgotten by a recipe author.

    Neither field carries a default_factory. ``from_overrides`` merges a partial
    dict onto such a default, which would splice the *base* type's load spec into a
    payload destined for a narrowed one; with a plain ``None`` the payload validates
    straight against T and the pinned class defaults supply pattern and load.

    Attributes:
        templates: Templates artifact to fit from. Mutually exclusive with model.
        model: Pre-fitted model artifact. Mutually exclusive with templates.
    """

    templates: T | None = None
    model: M | None = None

    @model_validator(mode="after")
    def _exclusive_source(self) -> Self:
        """Exactly one source, counted by what was *given* rather than named.

        Presence in ``model_fields_set`` alone is not the question, because this
        model's own ``model_dump`` writes both keys -- the one that was not
        provided as an explicit ``null``. Reading that dump back therefore set
        both, and every params file any global model feature ever wrote was
        rejected on reload. ``reconcile`` rebuilds a run's feature from exactly
        that file, so it could not confirm a single such run and reported them
        all unresolvable; the reader is `run_feature`'s own output.

        ``model_fields_set`` still has to be consulted for that reason alone:
        neither field carries a non-``None`` default any more, but a reloaded dump
        names both keys, and only the values tell which one was given.
        """
        has_templates = (
            "templates" in self.model_fields_set and self.templates is not None
        )
        has_model = "model" in self.model_fields_set and self.model is not None
        if has_templates == has_model:
            msg = "Exactly one of 'templates' or 'model' must be provided"
            raise ValueError(msg)
        if not has_templates:
            self.templates = None
        if not has_model:
            self.model = None
        return self
