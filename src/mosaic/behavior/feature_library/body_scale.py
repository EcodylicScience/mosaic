"""
BodyScaleFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .helpers import _pose_column_pairs, ensure_columns
from .spec import COLUMNS as C
from .spec import Inputs, Params, TrackInput, register_feature


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

    name = "body-scale"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
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

    def load_state(self, run_root: Path, artifact_paths: dict[str, Path]) -> bool:
        return True

    def fit(self, inputs: Iterator[tuple[str, pd.DataFrame]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        ensure_columns(df, [C.frame_col, C.id_col])
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = _expect_single(df, C.group_col)
        sequence = _expect_single(df, C.seq_col)
        rows: list[dict[str, int | str | float]] = []
        x_cols = [x for x, _ in pose_pairs]
        y_cols = [y for _, y in pose_pairs]
        # trex_v1 guarantees one row per (frame, id); groupby is defensive
        for _, sub in df.groupby([C.frame_col, C.id_col], sort=True):
            row = sub.iloc[0]
            xs = row[x_cols].to_numpy(dtype=float)
            ys = row[y_cols].to_numpy(dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() < 2:
                continue
            pts = np.column_stack((xs[valid], ys[valid]))
            # pdist returns only upper-triangle pairwise distances (no full NxN matrix)
            dists = pdist(pts)
            rows.append(
                {
                    C.frame_col: row[C.frame_col],
                    C.id_col: row[C.id_col],
                    "scale": float(np.median(dists)),
                    C.seq_col: sequence,
                    C.group_col: group,
                }
            )
        return pd.DataFrame(rows)
