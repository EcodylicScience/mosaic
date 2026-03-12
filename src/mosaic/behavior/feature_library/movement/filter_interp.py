"""Movement-based confidence filtering and interpolation feature."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, final

import pandas as pd

from mosaic.core.dataset import register_feature

from ..params import COLUMNS, Inputs, OutputType, Params, TrackInput
from .convert import _ensure_movement, from_movement_dataset, to_movement_dataset


@final
@register_feature
class MovementFilterInterpolate:
    """Filter low-confidence points and interpolate gaps using ``movement``.

    Wraps ``movement.filtering.filter_by_confidence`` and
    ``movement.filtering.interpolate_over_time``.

    When no confidence columns (poseP0..N) are present, the confidence
    filter is skipped and only interpolation of existing NaN gaps is
    performed.

    The output is a full track DataFrame with cleaned positions replacing
    the originals, so downstream features can chain off the result.
    """

    name = "movement-filter-interpolate"
    version = "0.1"
    parallelizable = True
    output_type: OutputType = "per_frame"

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        confidence_threshold: float = 0.6
        interpolation_method: str = "linear"
        max_gap: int | None = None
        include_centroid: bool = True
        fps: float | None = None
        keypoint_names: list[str] | None = None

    def __init__(
        self,
        inputs: MovementFilterInterpolate.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ds = None
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path: Path) -> None:
        return

    def load_model(self, path: Path) -> None:
        return

    # ----------------------- Core logic --------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        _ensure_movement()
        from movement.filtering import (
            filter_by_confidence,
            interpolate_over_time,
        )

        p = self.params
        ds, meta = to_movement_dataset(
            df,
            fps=p.fps,
            keypoint_names=p.keypoint_names,
            include_centroid=p.include_centroid,
        )

        position = ds["position"]
        confidence = ds["confidence"]

        # Step 1: confidence filter (skip if no real confidence data)
        if meta["has_confidence"]:
            position = filter_by_confidence(
                position, confidence, threshold=p.confidence_threshold
            )
        else:
            warnings.warn(
                "No poseP columns found — skipping confidence filter, "
                "only interpolating existing NaN gaps.",
                stacklevel=2,
            )

        # Step 2: interpolate NaN gaps
        kwargs = {"method": p.interpolation_method}
        if p.max_gap is not None:
            kwargs["max_gap"] = p.max_gap
        position = interpolate_over_time(position, **kwargs)

        ds["position"] = position
        return from_movement_dataset(
            ds, df, meta, update_confidence=meta["has_confidence"]
        )
