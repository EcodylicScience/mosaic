from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import model_validator

from mosaic.core.keypoint_refs import resolve_keypoint_refs
from mosaic.core.params import Declared, DeclaredModel
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


_POSE_N_DESCRIPTION = (
    "Total number of pose keypoints per individual, before any subset "
    "selection by pose_indices."
)

_POSE_INDICES_DESCRIPTION = (
    "Zero-based keypoint indices to use, as positions into the full "
    "keypoint set. Unset uses every keypoint counted by pose_n."
)

_X_PREFIX_DESCRIPTION = (
    "Column name prefix for a keypoint's X coordinate, followed by its index."
)

_Y_PREFIX_DESCRIPTION = (
    "Column name prefix for a keypoint's Y coordinate, followed by its index."
)

_CONFIDENCE_PREFIX_DESCRIPTION = (
    "Column name prefix for a keypoint's confidence score, followed by its index."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "Human-readable name for each keypoint, ordered to match the keypoint "
    "columns. Its length must equal pose_n when set. Unset, a feature that "
    "needs names generates them (kp0, kp1, ...)."
)


POSE_CONFIG_DESCRIPTION = "Pose keypoint configuration: indices and column prefixes."
"""What a field holding a :class:`PoseConfig` is described as, written once.

Six features carry such a field on their nested ``Params`` -- ``KpmsFeature``,
``PairEgocentricFeatures``, ``PairPoseDistancePCA``, ``TrackSubsample``,
``EgocentricCrop`` and ``InteractionCropPipeline`` -- and every one of them
reaches this sentence rather than spelling its own.

It must not restate ``PoseConfig``'s docstring summary, and the reason is
mechanical rather than stylistic: pydantic's ``handle_ref_overrides`` deletes a
sibling key beside a ``$ref`` whose value equals the referenced schema's, so a
field description character-identical to the object description is dropped from
the property outright. Tidying the two into agreement would silently take the
description off every property that refs the type.
"""


class PoseConfig(DeclaredModel):
    """Pose keypoint column naming and selection."""

    pose_n: Annotated[int, Declared(_POSE_N_DESCRIPTION)] = 7
    pose_indices: Annotated[list[int] | None, Declared(_POSE_INDICES_DESCRIPTION)] = (
        None
    )
    x_prefix: Annotated[str, Declared(_X_PREFIX_DESCRIPTION)] = "poseX"
    y_prefix: Annotated[str, Declared(_Y_PREFIX_DESCRIPTION)] = "poseY"
    confidence_prefix: Annotated[str, Declared(_CONFIDENCE_PREFIX_DESCRIPTION)] = (
        "poseP"
    )
    keypoint_names: Annotated[
        list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
    ] = None

    def selected_indices(self) -> list[int]:
        """The keypoint indices to read: ``pose_indices``, or every index below ``pose_n``."""
        if self.pose_indices is None:
            return list(range(self.pose_n))
        return list(self.pose_indices)

    def keypoint_columns(self) -> tuple[list[str], list[str]]:
        """The x and the y column of each selected keypoint, in index order."""
        indices = self.selected_indices()
        return (
            [f"{self.x_prefix}{index}" for index in indices],
            [f"{self.y_prefix}{index}" for index in indices],
        )

    def names(self, n: int) -> list[str]:
        """Name the keypoints: ``keypoint_names`` when set, else ``kp0`` to ``kp{n-1}``.

        Args:
            n: How many keypoints to name when ``keypoint_names`` is unset.
        """
        if self.keypoint_names is not None:
            return list(self.keypoint_names)
        return [f"kp{index}" for index in range(n)]

    def resolve_keypoints(self, refs: Sequence[int | str], *, who: str) -> list[int]:
        """Resolve keypoint references to positions among the selected keypoints.

        An integer is a keypoint index, as ``pose_indices`` lists them, and a
        string is one of ``keypoint_names``. An index is checked against the
        selection alone, because ``pose_indices`` may select keypoints at or past
        ``pose_n``.

        Args:
            refs: The keypoint references, in the order wanted.
            who: The caller and its parameter, for the message.

        Returns:
            The position of each reference in :meth:`selected_indices`, in the
            order of *refs*.

        Raises:
            ValueError: On a name that
                :func:`~mosaic.core.keypoint_refs.resolve_keypoint_refs` refuses
                against ``keypoint_names``, on a negative index, and on a
                reference outside the selected keypoints.
        """
        names = [ref for ref in refs if isinstance(ref, str)]
        by_name = dict(
            zip(
                names,
                resolve_keypoint_refs(names, self.keypoint_names, self.pose_n, who=who),
            )
        )
        indices = [by_name[ref] if isinstance(ref, str) else ref for ref in refs]
        negative = [index for index in indices if index < 0]
        if negative:
            msg = f"{who}: {negative} are not keypoint indices, which start at 0"
            raise ValueError(msg)
        selected = self.selected_indices()
        positions: dict[int, int] = {}
        for position, index in enumerate(selected):
            _ = positions.setdefault(index, position)
        outside = [ref for ref, index in zip(refs, indices) if index not in positions]
        if outside:
            remedy = (
                "pose.pose_indices must include them"
                if self.pose_indices is not None
                else "raise pose.pose_n past them or list them in pose.pose_indices"
            )
            msg = (
                f"{who}: keypoints {outside} are not among the selected keypoints "
                f"{selected}; {remedy}"
            )
            raise ValueError(msg)
        return [positions[index] for index in indices]

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

BOOKKEEPING_COLS: frozenset[str] = frozenset(
    {
        "attention_target_id",
        "entity_level",
        "fps",
        "identity_margin",
        "identity_score",
        "identity_source_id",
        "label",
        "nn_id",
        "split",
    }
)
"""Columns that are neither identity nor measurement.

A sampling rate, a ground-truth label, a train/test assignment and a level
marker record facts about a run and say nothing about whom a row describes. A
frame's payload excludes them as it excludes the identity columns, and a feature
rebuilding its output drops them.

``nn_id`` (``nearest-neighbor``) and ``attention_target_id``
(``attention-target``) record another individual's track id. Trackers number
individuals per recording, and a fit that reads the id as a number puts ids 3
and 4 close together and ids 0 and 9 far apart. Neither column is identity. The
row describes the focal individual, and a frame without a neighbor or a target
leaves the id NaN, so neither can be a join key.

These two columns are dropped like ``fps`` and ``label``, and ``meta_columns``
leaves them out. A feature that rebuilds its output from ``meta_columns``
(``temporal-stack``, ``xgboost``, ``lightning-action``) writes neither. A global
model's ``apply`` keeps every column outside its fitted set, so
``global-tsne``, ``global-kmeans`` and ``global-ward`` keep them unfitted beside
``fps`` and ``label``. Readers of the neighbor (``nn-delta-response``,
``social-motion-summary``, the ``pair_filter`` lookup) read ``nn_id`` by name
from ``nearest-neighbor``'s output.

``identity_score``, ``identity_margin`` and ``identity_source_id``
(``assign-identities``) record how a tracklet was assigned its identity and the
track id it had before. They describe the assignment and not the individual.

The identity names are in ``META_COLS`` and absent here. The payload and the
metadata passthrough both read them there. Two lists once disagreed, and
``perspective`` was excluded from the payload by one and dropped from the
metadata by the other.
"""

NON_PAYLOAD_COLS: frozenset[str] = META_COLS | BOOKKEEPING_COLS
"""Every column that is not a measurement. The other columns of a frame are its
payload, which a global model reads as one vector per row."""
