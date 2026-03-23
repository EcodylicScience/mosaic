from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import model_validator

from mosaic.core.pipeline._loaders import StrictModel

if TYPE_CHECKING:
    import pandas as pd


class Columns(StrictModel):
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
        intentionally excluded -- they are data, not metadata.
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


class PoseConfig(StrictModel):
    """Pose keypoint column naming and selection.

    Attributes:
        pose_n: Total number of pose keypoints in the data. Default 7.
        pose_indices: Subset of keypoint indices to use. None uses all.
        x_prefix: Column name prefix for X coordinates. Default "poseX".
        y_prefix: Column name prefix for Y coordinates. Default "poseY".
        confidence_prefix: Column prefix for confidence scores. Default "poseP".
        keypoint_names: Human-readable names for each keypoint. Default None
            (auto-generated as ["kp0", "kp1", ...] by features that need names).
    """

    pose_n: int = 7
    pose_indices: list[int] | None = None
    x_prefix: str = "poseX"
    y_prefix: str = "poseY"
    confidence_prefix: str = "poseP"
    keypoint_names: list[str] | None = None

    @model_validator(mode="after")
    def _check_keypoint_names_length(self) -> Self:
        if self.keypoint_names is not None and len(self.keypoint_names) != self.pose_n:
            msg = (
                f"len(keypoint_names)={len(self.keypoint_names)} "
                f"does not match pose_n={self.pose_n}"
            )
            raise ValueError(msg)
        return self
