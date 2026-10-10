"""Heading from pose keypoints, as a chosen method rather than a default.

Four converters used to compute this inline and write it into the track table as
``ANGLE``, which made it look like something the tracker had reported. It is not.
A pose model returns keypoints; a direction is an inference over them, and the
two available inferences are not equally good:

- ``two_point`` orders a pair of landmarks along the body, so the direction is
  determined. This is a heading.
- ``pca`` takes the first principal component of every keypoint, which is an
  **axis**. Its sign is arbitrary, so consecutive frames of a clean track can
  come back pi apart, and anything differencing successive angles reads those
  flips as real turns.

Inside a converter the fallback between them was silent, unversioned and
unrecorded: a table's ``ANGLE`` gave no indication of which rule produced it, or
that a rule had been applied at all. Here the method is a parameter, so it enters
the run identifier, and two runs under different methods are two addressable
results rather than one column that quietly changed meaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, final

import numpy as np
import pandas as pd

from mosaic.core.individuals import iter_individuals
from mosaic.core.kinematics import angle_from_pca, angle_from_two_points
from mosaic.core.pose_columns import pose_column_pairs
from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    ORDER_COLUMN,
    POSE_X,
    POSE_Y,
    Column,
    ParamColumn,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    InputStream,
    TrackInputs,
    resolve_order_col,
)
from mosaic.core.params import Declared, Params

from .registry import register_feature

HeadingMethod = Literal["two_point", "pca"]

_METHOD_DESCRIPTION = (
    "Which method derives the heading. two_point points from rear_idx "
    "toward front_idx, a determined direction. pca takes the first "
    "principal component of every keypoint, an axis whose sign is "
    "arbitrary."
)

_FRONT_IDX_DESCRIPTION = (
    "The keypoint index of the forward landmark, such as the head, snout "
    "or rostrum. Used when method is two_point."
)

_REAR_IDX_DESCRIPTION = (
    "The keypoint index of the rearward landmark, such as the tail base "
    "or abdomen. Used when method is two_point."
)

_OUTPUT_COL_DESCRIPTION = (
    "The column the heading is written to. Defaults to the library's own "
    "orientation column, the name a downstream feature reading ANGLE "
    "already expects."
)


@final
@register_feature
class HeadingFeature:
    """Per-frame body heading derived from pose keypoints.

    Outputs one row per input row: ``frame``, ``id``, the heading column, plus
    the metadata columns. The heading is in radians about the +x axis, measured
    in image coordinates where ``y`` increases downward -- so a positive angle
    turns clockwise on screen, matching the frame the keypoints themselves are in.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.heading.HeadingFeature.Params`.
    """

    category = "per-frame"
    name = "heading"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        ORDER_COLUMN,
        POSE_X,
        POSE_Y,
    )
    provides: ClassVar[Provides] = Provides(
        adds=(ParamColumn("output_col"),), conditional=META_KEPT
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        method: Annotated[HeadingMethod, Declared(_METHOD_DESCRIPTION)] = "two_point"
        front_idx: Annotated[int, Declared(_FRONT_IDX_DESCRIPTION)] = 0
        rear_idx: Annotated[int, Declared(_REAR_IDX_DESCRIPTION)] = 1
        output_col: Annotated[str, Declared(_OUTPUT_COL_DESCRIPTION)] = (
            C.orientation_col
        )

    def __init__(
        self,
        inputs: HeadingFeature.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        order_col = resolve_order_col(df)
        pose_pairs = pose_column_pairs(list(df.columns))
        # The declaration asks for a column of each prefix. A keypoint is a
        # poseX/poseY pair of one suffix, which a prefix cannot express.
        if not pose_pairs:
            raise ValueError(
                f"{self.name} needs pose keypoint columns (poseX*/poseY*) and this "
                "table carries none. A heading is derived from keypoints; a table "
                "with only a centroid has no body axis to read."
            )

        # Sorting by id first yields the individuals in id order.
        df = df.sort_values(C.id_col, kind="stable")
        parts: list[pd.DataFrame] = []
        for _, rows in iter_individuals(df, order_col):
            xy = np.stack(
                [
                    np.stack(
                        [
                            rows[x_col].to_numpy(dtype=float),
                            rows[y_col].to_numpy(dtype=float),
                        ],
                        axis=-1,
                    )
                    for x_col, y_col in pose_pairs
                ],
                axis=1,
            )  # (T, L, 2)
            parts.append(self._with_heading(rows, self._heading(xy, len(pose_pairs))))
        # A row without an id belongs to no individual and keeps a NaN heading.
        unidentified = df[df[C.id_col].isna()].sort_values(order_col, kind="stable")
        if not unidentified.empty:
            angle = np.full(len(unidentified), np.nan)
            parts.append(self._with_heading(unidentified, angle))
        return pd.concat(parts, ignore_index=True)

    def _with_heading(self, rows: pd.DataFrame, angle: np.ndarray) -> pd.DataFrame:
        """Return *angle* as the heading of *rows*, beside their identity columns."""
        out = pd.DataFrame({self.params.output_col: angle})
        meta = C.meta_set() & set(rows.columns)
        return out.join(rows[sorted(meta)].reset_index(drop=True))

    def _heading(self, xy: np.ndarray, n_keypoints: int) -> np.ndarray:
        """One individual's heading, by the chosen method.

        No fallback between the two. The converters silently dropped from a
        two-point heading to a principal axis whenever an index was out of
        range, which meant a table could hold both kinds of number under one
        name. A method that cannot be applied is a configuration error, and
        saying so is more useful than substituting the one whose sign is
        arbitrary.
        """
        p = self.params
        if p.method == "pca":
            return angle_from_pca(xy)

        for label, index in (("front_idx", p.front_idx), ("rear_idx", p.rear_idx)):
            if not 0 <= index < n_keypoints:
                raise ValueError(
                    f"{self.name}: {label}={index} is out of range for a table with "
                    f"{n_keypoints} keypoints. Set it to a valid index, or pass "
                    "method='pca' to use the principal axis instead -- noting that "
                    "its sign is arbitrary."
                )
        if p.front_idx == p.rear_idx:
            raise ValueError(
                f"{self.name}: front_idx and rear_idx are both {p.front_idx}, which "
                "names one point rather than an axis."
            )
        return angle_from_two_points(xy[:, p.front_idx, :], xy[:, p.rear_idx, :])
