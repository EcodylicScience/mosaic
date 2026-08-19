from __future__ import annotations

from typing import Generic, Self

from pydantic import BaseModel, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import StrictModel
from mosaic.core.pipeline.types.artifacts import JoblibArtifact, TemplatesRef

# ``JsonValue`` used to live here. It moved to ``mosaic.core.json_value`` -- a
# module with no imports at all -- because the dataset manifest needs the same
# type and importing this one drags the loader and artifact machinery, and
# through them pandas, into a manifest read. The package ``__init__`` re-exports
# it from its new home, so every existing import path is unchanged.


class _HashExclude:
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


HASH_EXCLUDE = _HashExclude()


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
            if any(isinstance(m, _HashExclude) for m in info.metadata):
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
