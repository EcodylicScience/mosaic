from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

from pydantic import Field
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
    JoblibLoadSpec,
    LabelsSource,
    LabelsSourceSpec,
    LoadSpec,
    NNResult,
    NpzLoadSpec,
    OutputType,
    Params,
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
    "JoblibLoadSpec",
    "LabelsSource",
    "LabelsSourceSpec",
    "LoadSpec",
    "NNResult",
    "NpzLoadSpec",
    "OutputType",
    "Params",
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


FEATURES: dict[str, type[Feature]] = {}

_F = TypeVar("_F", bound=Feature)


def register_feature(cls: type[_F]) -> type[_F]:
    FEATURES[cls.__name__] = cls
    return cls
