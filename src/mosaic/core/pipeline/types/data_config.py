from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import model_validator

from mosaic.core.strict_model import StrictModel

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


# What identifies a pair row. A pair feature emits one row per *ordered* pair per
# frame -- ``id1`` the focal, ``id2`` the other -- and ``perspective`` says which
# ordering, so ``(frame, id1, id2)`` is not a key and ``(frame, id1, id2,
# perspective)`` is.
#
# ``perspective`` belongs here rather than among the measurements because leaving it
# out made it neither a join key nor carried metadata, and the two halves of that
# compounded: a merge of two pair inputs renamed the second copy to
# ``perspective__1``, which then read as a numeric feature and was fitted as data,
# while the surviving plain column was dropped by any feature rebuilding its output
# from the metadata set.
PAIR_COLS: frozenset[str] = frozenset({"id1", "id2", "perspective"})

# The columns that name *who* a row is about. What ``alignment_verdict`` asks for
# when two inputs sit at different entity levels: sharing ``frame`` alone is a
# cartesian product, sharing an id is an alignment. ``perspective`` is deliberately
# absent -- it separates two rows of one pair, it does not say who they are about, so
# a stray one on an individual-level frame must not unlock a cross-level join.
ID_COLS: frozenset[str] = frozenset({COLUMNS.id_col, "id1", "id2"})

# Join keys for a multi-input merge, and the metadata a feature passes through.
# They differ by exactly ``{group, sequence}``: those are constant within an entry,
# so joining on them narrows nothing, and a blank group spelled ``""`` on one side
# and ``NaN`` on the other would empty the join instead.
ALIGN_COLS: frozenset[str] = frozenset(
    (COLUMNS.meta_set() - {COLUMNS.group_col, COLUMNS.seq_col}) | PAIR_COLS
)
META_COLS: frozenset[str] = frozenset(COLUMNS.meta_set() | PAIR_COLS)
