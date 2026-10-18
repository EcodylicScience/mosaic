"""
BodyScaleFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import ClassVar, final

import numpy as np
import pandas as pd

from mosaic.core.pose_columns import pose_column_pairs
from mosaic.core.pipeline.column_contract import (
    POSE_X,
    POSE_Y,
    Column,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    InputStream,
    TrackInputs,
)
from mosaic.core.params import Params

from .helpers import (
    ensure_columns,
    finite_keypoints,
    one_row_per_frame_and_id,
    pairwise_median_distance,
)
from .registry import register_feature


def _expect_single(df: pd.DataFrame, col: str) -> str:
    """Return the single unique value of *col*, or '' if absent."""
    if col not in df.columns:
        return ""
    vals = df[col].unique()
    if len(vals) != 1:
        msg = f"Expected exactly 1 unique {col}, got {len(vals)}: {vals.tolist()}"
        raise ValueError(msg)
    return str(vals[0])


@final
@register_feature
class BodyScaleFeature:
    """
    Per-frame body scale: median intra-animal pose distance.

    Outputs per sequence parquet with columns: frame, id, scale, sequence, group.
    Intended to be averaged later (per sequence or dataset) to derive a single
    normalization constant for downstream orientation features.
    """

    category = "per-frame"
    name = "body-scale"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # measures one entry, and refuses a multi-entry frame
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        POSE_X,
        POSE_Y,
    )
    # The group and sequence are written blank when the input lacks them.
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.frame_col),
            Column(C.id_col),
            Column("scale"),
            Column(C.seq_col),
            Column(C.group_col),
        )
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: BodyScaleFeature.Inputs = Inputs(("tracks",)),
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
        ensure_columns(df, [C.frame_col, C.id_col])
        pose_pairs = pose_column_pairs(df.columns)
        if not pose_pairs:
            raise ValueError(
                f"{self.name} needs pose keypoint columns (poseX*/poseY*) and this "
                "table carries none. A body scale is the spread among an "
                "individual's keypoints; a table with only a centroid has one "
                "point per individual and no spread to measure."
            )
        group = _expect_single(df, C.group_col)
        sequence = _expect_single(df, C.seq_col)
        # trex_v1 guarantees one row per (frame, id); a repeat is ignored
        read = [C.frame_col, C.id_col, *chain.from_iterable(pose_pairs)]
        individuals = one_row_per_frame_and_id(df[read])
        x, y = finite_keypoints(individuals, pose_pairs)
        measurable = np.isfinite(x).sum(axis=1) >= 2
        if not measurable.any():
            return pd.DataFrame()
        result = individuals.loc[measurable, [C.frame_col, C.id_col]].reset_index(
            drop=True
        )
        result["scale"] = pairwise_median_distance(x, y)[measurable]
        result[C.seq_col] = sequence
        result[C.group_col] = group
        return result
