"""Movement-based confidence filtering and interpolation feature."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import final

import pandas as pd

from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
)

from ..registry import register_feature
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

    category = "per-frame"
    name = "movement-filter-interpolate"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

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

    # --- State (stateless feature) ---

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

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
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
        kwargs: dict = {"method": p.interpolation_method}
        if p.max_gap is not None:
            kwargs["max_gap"] = p.max_gap
        position = interpolate_over_time(position, **kwargs)

        ds["position"] = position
        return from_movement_dataset(
            ds, df, meta, update_confidence=meta["has_confidence"]
        )
