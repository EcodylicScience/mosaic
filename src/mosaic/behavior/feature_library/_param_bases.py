from __future__ import annotations

from collections.abc import KeysView
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class DictModel(BaseModel):
    """BaseModel with dict-like access for backward compatibility.

    Provides __getitem__, get, __contains__, and keys() so that existing
    code using dict-style access (spec["key"], spec.get("key")) works
    transparently with typed models.
    """

    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, key: str) -> object:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: object = None) -> object:
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self.__class__.model_fields

    def keys(self) -> KeysView[str]:
        """Support {**params} dict spread and dict(params) conversion."""
        return self.__class__.model_fields.keys()


class FeatureParams(DictModel):
    """Base for all feature parameter models.

    Provides from_overrides() constructor for user-config dicts.

    Attributes:
        id_col: Animal/subject identifier column. Default "id".
        seq_col: Sequence identifier column. Default "sequence".
        group_col: Group/session identifier column. Default "group".
        order_pref: Column names to use for temporal ordering, tried in order.
            Default ("frame", "time").
    """

    id_col: str = "id"
    seq_col: str = "sequence"
    group_col: str = "group"
    order_pref: tuple[str, ...] = ("frame", "time")

    @classmethod
    def from_overrides(cls, overrides: dict[str, object] | None = None) -> Self:
        """Construct from user dict; missing keys get field defaults.

        For BaseModel-typed fields with default_factory, partial dict overrides
        are merged on top of the default before validation (1-level deep merge).
        This replaces the per-feature deep-merge hacks in global_ward/ward_assign.
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
            default_obj = field_info.get_default(call_default_factory=True)
            if isinstance(default_obj, BaseModel):
                merged[key] = {**default_obj.model_dump(), **value}
        return cls.model_validate(merged)


# --- Spec models ---


class NpzLoadSpec(DictModel):
    """Load spec for numpy .npz archives.

    Attributes:
        kind: Discriminator literal "npz".
        key: Array key to extract from the .npz file. Required.
        transpose: Transpose the loaded array. Default False.
    """

    kind: Literal["npz"] = "npz"
    key: str
    transpose: bool = False


class ParquetLoadSpec(DictModel):
    """Load spec for parquet files.

    Attributes:
        kind: Discriminator literal "parquet".
        transpose: Transpose the loaded array. Default False.
        columns: Explicit column list. None uses numeric_only filter.
        drop_columns: Columns to drop before loading.
        numeric_only: Keep only numeric columns. Default True.
        frame_column: Column to extract as frame indices.
    """

    kind: Literal["parquet"] = "parquet"
    transpose: bool = False
    columns: list[str] | None = None
    drop_columns: list[str] | None = None
    numeric_only: bool = True
    frame_column: str | None = None


LoadSpec = Annotated[
    NpzLoadSpec | ParquetLoadSpec, Field(discriminator="kind")
]


class FeatureRef(DictModel):
    """Reference to another feature's output on disk.

    Attributes:
        feature: Feature name. Required.
        run_id: Specific run ID, or None for latest.
        pattern: Glob pattern for files in the run root.
    """

    feature: str
    run_id: str | None = None
    pattern: str = "*.npz"


class ArtifactSpec(FeatureRef):
    """Feature artifact reference with load specification.

    Attributes:
        load: How to load the matched files. Required.
    """

    load: LoadSpec


class InputSpec(ArtifactSpec):
    """Input specification for multi-input features.

    Attributes:
        name: Display/prefix name for the input. None uses feature name.
    """

    name: str | None = None


# --- Mixins ---


class PositionColumnsMixin(BaseModel):
    """Mixin for features that use spatial position columns.

    Attributes:
        x_col: X-coordinate column name. Default "X".
        y_col: Y-coordinate column name. Default "Y".
        angle_col: Body-orientation angle column name. Default "ANGLE".
    """

    x_col: str = "X"
    y_col: str = "Y"
    angle_col: str = "ANGLE"


class InterpolationMixin(BaseModel):
    """Mixin for features that interpolate missing pose/position data.

    Attributes:
        linear_interp_limit: Max consecutive NaN frames to fill via linear
            interpolation. Default 10, must be >= 1.
        edge_fill_limit: Max frames to forward/backward fill at sequence edges.
            Default 3, must be >= 0.
        max_missing_fraction: Rows with a higher fraction of NaN columns are
            dropped entirely. Default 0.10, range [0, 1].
    """

    linear_interp_limit: int = Field(default=10, ge=1)
    edge_fill_limit: int = Field(default=3, ge=0)
    max_missing_fraction: float = Field(default=0.10, ge=0.0, le=1.0)


class SamplingMixin(BaseModel):
    """Mixin for features that depend on frame rate or temporal smoothing.

    Attributes:
        fps_default: Fallback frames-per-second when the data does not carry an
            fps column. Default 30.0, must be > 0.
        smooth_win: Moving-average window size applied to pose coordinates
            before feature computation. 0 disables smoothing. Default 0.
    """

    fps_default: float = Field(default=30.0, gt=0)
    smooth_win: int = Field(default=0, ge=0)
