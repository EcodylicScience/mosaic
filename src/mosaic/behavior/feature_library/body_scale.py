"""
BodyScaleFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature

from .helpers import _pose_column_pairs
from .params import Inputs, OutputType, Params, TrackInput


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
    output_type: OutputType = "per_frame"

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
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = False
        self._ds = None
        self._scope_filter: dict[str, object] = {}

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]):
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def save_model(self, path: Path) -> None:
        raise NotImplementedError

    def load_model(self, path: Path) -> None:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "frame" not in df.columns or "id" not in df.columns:
            return pd.DataFrame()
        pose_pairs = _pose_column_pairs(df.columns)
        if not pose_pairs:
            return pd.DataFrame()
        group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else ""
        sequence = (
            str(df["sequence"].iloc[0]) if "sequence" in df.columns and len(df) else ""
        )
        rows = []
        for frame_val, g in df.groupby("frame", sort=True):
            for id_val, sub in g.groupby("id"):
                pts = []
                for x_col, y_col in pose_pairs:
                    x = sub.iloc[0].get(x_col)
                    y = sub.iloc[0].get(y_col)
                    if (
                        x is None
                        or y is None
                        or not np.isfinite(x)
                        or not np.isfinite(y)
                    ):
                        continue
                    pts.append((float(x), float(y)))
                if len(pts) < 2:
                    continue
                arr = np.asarray(pts, dtype=float)
                dists = np.sqrt(((arr[:, None, :] - arr[None, :, :]) ** 2).sum(axis=2))
                dists = dists[np.triu_indices_from(dists, k=1)]
                if dists.size == 0:
                    continue
                med = float(np.median(dists))
                rows.append(
                    {
                        "frame": int(frame_val),
                        "id": id_val,
                        "scale": med,
                        "sequence": sequence,
                        "group": group,
                    }
                )
        return pd.DataFrame(rows)
