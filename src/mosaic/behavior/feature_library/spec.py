from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, Self

if TYPE_CHECKING:
    import pandas as pd

from pydantic import Field, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline.types import (
    ArtifactSpec,
    BodyScaleResult,
    DictModel,
    Feature,
    FeatureLabelsSource,
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    InputsLike,
    JoblibArtifact,
    JoblibLoadSpec,
    LabelsSource,
    LabelsSourceSpec,
    LoadSpec,
    NNResult,
    NpzArtifact,
    NpzLoadSpec,
    OutputType,
    Params,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
    ResultColumn,
    TrackInput,
)

__all__ = [
    "ArtifactSpec",
    "BodyScaleResult",
    "DictModel",
    "Feature",
    "FeatureLabelsSource",
    "GroundTruthLabelsSource",
    "InputRequire",
    "Inputs",
    "InputsLike",
    "JoblibArtifact",
    "JoblibLoadSpec",
    "LabelsSource",
    "LabelsSourceSpec",
    "LoadSpec",
    "NNResult",
    "NpzArtifact",
    "NpzLoadSpec",
    "OutputType",
    "Params",
    "ParquetArtifact",
    "ParquetLoadSpec",
    "Result",
    "ResultColumn",
    "TrackInput",
    "Columns",
    "COLUMNS",
    "resolve_order_col",
    "InterpolationConfig",
    "SamplingConfig",
    "PoseConfig",
    "PoolConfig",
    "GlobalModelParams",
    "FEATURES",
    "register_feature",
]


class Columns(DictModel):
    """Dataset column name conventions.

    Single source of truth for all standard column names. Override by
    monkey-patching COLUMNS before importing features:
        from mosaic.behavior.feature_library import params
        params.COLUMNS = params.Columns(id_col="animal")

    Attributes:
        id_col: Animal/subject identifier column. Default "id".
        seq_col: Sequence identifier column. Default "sequence".
        group_col: Group/session identifier column. Default "group".
        frame_col: Frame number column name. Default "frame".
        time_col: Timestamp column name. Default "time".
        order_by: Preferred temporal ordering column. Default "frames".
        x_col: X-coordinate column name. Default "X".
        y_col: Y-coordinate column name. Default "Y".
        orientation_col: Body-orientation angle column name. Default "ANGLE".
    """

    id_col: str = "id"
    seq_col: str = "sequence"
    group_col: str = "group"
    frame_col: str = "frame"
    time_col: str = "time"
    order_by: Literal["frames", "time"] = "frames"
    x_col: str = "X"
    y_col: str = "Y"
    orientation_col: str = "ANGLE"

    def meta_set(self) -> set[str]:
        """The five metadata column names as a set.

        Useful for set intersection (passthrough) or set difference (exclusion)
        against ``df.columns``.  Spatial columns (x, y, orientation) are
        intentionally excluded — they are data, not metadata.
        """
        return {
            self.id_col,
            self.seq_col,
            self.group_col,
            self.frame_col,
            self.time_col,
        }


COLUMNS = Columns()


def resolve_order_col(df: pd.DataFrame) -> str:
    """Pick the best ordering column present in *df*.

    Uses COLUMNS.order_by preference, then falls back to the other option.
    Raises ValueError when neither column exists.
    """
    if COLUMNS.order_by == "frames":
        first, second = COLUMNS.frame_col, COLUMNS.time_col
    else:
        first, second = COLUMNS.time_col, COLUMNS.frame_col
    if first in df.columns:
        return first
    if second in df.columns:
        return second
    raise ValueError(
        f"Need '{COLUMNS.frame_col}' or '{COLUMNS.time_col}' column to order rows."
    )


class InterpolationConfig(DictModel):
    """Interpolation parameters for missing pose/position data.

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


class SamplingConfig(DictModel):
    """Frame rate and temporal smoothing parameters.

    Attributes:
        fps_default: Fallback frames-per-second when the data does not carry an
            fps column. Default 30.0, must be > 0.
        smooth_win: Moving-average window size applied to pose coordinates
            before feature computation. 0 disables smoothing. Default 0.
    """

    fps_default: float = Field(default=30.0, gt=0)
    smooth_win: int = Field(default=0, ge=0)


class PoseConfig(DictModel):
    """Pose keypoint column naming and selection.

    Attributes:
        pose_n: Total number of pose keypoints in the data. Default 7.
        pose_indices: Subset of keypoint indices to use. None uses all.
        x_prefix: Column name prefix for X coordinates. Default "poseX".
        y_prefix: Column name prefix for Y coordinates. Default "poseY".
    """

    pose_n: int = 7
    pose_indices: list[int] | None = None
    x_prefix: str = "poseX"
    y_prefix: str = "poseY"


class PoolConfig(DictModel):
    """Candidate pool configuration for template extraction.

    Controls how per-entry contributions to the candidate pool are
    allocated before the final template selection step.

    Attributes:
        size: Candidate pool size. For "random" strategy, defaults to
            n_templates (pool == output). For "farthest_first", should
            be larger (e.g. n_templates * 3).
        allocation: How per-entry quotas are computed.
            "reservoir": weighted reservoir sampling, single pass.
            "exact": two-pass -- first counts rows, second samples
            with exact proportional quotas.
            Default "reservoir".
        max_entry_fraction: Cap per entry as fraction of pool size.
            None means no cap (purely proportional). At runtime,
            effective cap is max(max_entry_fraction, 1 / n_entries)
            so the pool can always be filled completely. Default None.
    """

    size: int | None = None
    allocation: Literal["reservoir", "exact"] = "reservoir"
    max_entry_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


_M = TypeVar("_M", bound=JoblibArtifact[object], default=JoblibArtifact[object])


class GlobalModelParams(Params, Generic[_M]):
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
    model: _M | None = None

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


FEATURES: dict[str, type[Feature]] = {}

_F = TypeVar("_F", bound=Feature)


def register_feature(cls: type[_F]) -> type[_F]:
    FEATURES[cls.__name__] = cls
    return cls
