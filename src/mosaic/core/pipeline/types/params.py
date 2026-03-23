from __future__ import annotations

from typing import Generic, Self

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import ParquetLoadSpec, StrictModel
from mosaic.core.pipeline.types.artifacts import (
    ArtifactSpec,
    JoblibArtifact,
    ParquetArtifact,
)


class Params(StrictModel):
    """Base for all feature parameter models.

    Provides from_overrides() constructor for user-config dicts.
    Subclasses declare feature-specific fields.
    """

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


class GlobalModelParams(Params, Generic[M]):
    """Base params for global features that fit on a templates artifact
    or load a pre-fitted model.

    Type parameter M is the model artifact type (must extend JoblibArtifact).
    Exactly one of `templates` or `model` must be provided.

    Both fields use default_factory so that from_overrides() merges
    partial dicts correctly. The _exclusive_source validator checks
    model_fields_set and nulls out the field that was not provided.

    Attributes:
        templates: Templates artifact to fit from. Mutually exclusive with model.
        model: Pre-fitted model artifact. Mutually exclusive with templates.
    """

    templates: ParquetArtifact | None = Field(
        default_factory=lambda: ArtifactSpec(feature="", load=ParquetLoadSpec())
    )
    model: M | None = None

    @model_validator(mode="after")
    def _exclusive_source(self) -> Self:
        has_templates = "templates" in self.model_fields_set
        has_model = "model" in self.model_fields_set
        if has_templates == has_model:
            msg = "Exactly one of 'templates' or 'model' must be provided"
            raise ValueError(msg)
        if not has_templates:
            self.templates = None
        if not has_model:
            self.model = None
        return self
