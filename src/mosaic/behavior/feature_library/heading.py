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
from typing import Literal, final

import numpy as np
import pandas as pd

from mosaic.core.kinematics import angle_from_pca, angle_from_two_points
from mosaic.core.pipeline.loading import pose_column_pairs
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    Params,
    TrackInputs,
    resolve_order_col,
)

from .helpers import ensure_columns
from .registry import register_feature

HeadingMethod = Literal["two_point", "pca"]


@final
@register_feature
class HeadingFeature:
    """Per-frame body heading derived from pose keypoints.

    Outputs one row per input row: ``frame``, ``id``, the heading column, plus
    the metadata columns. The heading is in radians about the +x axis, measured
    in image coordinates where ``y`` increases downward -- so a positive angle
    turns clockwise on screen, matching the frame the keypoints themselves are in.

    Params:
        method: ``"two_point"`` uses ``front_idx`` and ``rear_idx``; ``"pca"``
            uses every keypoint and returns an axis whose sign is arbitrary.
        front_idx: Keypoint index of the forward landmark (head, snout, rostrum).
        rear_idx: Keypoint index of the rearward landmark (tail base, abdomen).
        output_col: Name to write the heading under. Defaults to the library's
            orientation column, so a downstream feature reading ``ANGLE`` finds
            it without being reconfigured.
    """

    category = "per-frame"
    name = "heading"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        method: HeadingMethod = "two_point"
        front_idx: int = 0
        rear_idx: int = 1
        output_col: str = C.orientation_col

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

        p = self.params
        order_col = resolve_order_col(df)
        ensure_columns(df, [C.id_col])
        pose_pairs = pose_column_pairs(list(df.columns))
        if not pose_pairs:
            raise ValueError(
                f"{self.name} needs pose keypoint columns (poseX*/poseY*) and this "
                "table carries none. A heading is derived from keypoints; a table "
                "with only a centroid has no body axis to read."
            )

        df = df.sort_values([C.id_col, order_col]).reset_index(drop=True)
        angle = np.full(len(df), np.nan, dtype=float)

        for _, sub in df.groupby(C.id_col, sort=False):
            idx = sub.index.to_numpy()
            xy = np.stack(
                [
                    np.stack(
                        [
                            sub[x_col].to_numpy(dtype=float),
                            sub[y_col].to_numpy(dtype=float),
                        ],
                        axis=-1,
                    )
                    for x_col, y_col in pose_pairs
                ],
                axis=1,
            )  # (T, L, 2)
            angle[idx] = self._heading(xy, len(pose_pairs))

        out = pd.DataFrame({p.output_col: angle}, index=df.index)
        meta = C.meta_set() & set(df.columns)
        return out.join(df[sorted(meta)])

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
