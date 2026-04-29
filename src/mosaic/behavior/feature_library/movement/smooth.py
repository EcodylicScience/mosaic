"""Movement-based trajectory smoothing feature."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, final

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
class MovementSmooth:
    """Smooth trajectory positions using the ``movement`` library.

    Wraps ``movement.filtering.rolling_filter`` and
    ``movement.filtering.savgol_filter`` to smooth X/Y centroid and/or
    poseX/poseY keypoint positions.

    The output is a full track DataFrame with smoothed positions replacing
    the originals, so downstream features can chain off the result via
    ``Inputs((Result(feature="movement-smooth"),))``.
    """

    category = "per-frame"
    name = "movement-smooth"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        method: Literal["rolling", "savgol"] = "savgol"
        window: int = 5
        # rolling-specific
        statistic: Literal["median", "mean", "max", "min"] = "median"
        min_periods: int | None = None
        # savgol-specific
        polyorder: int = 2
        savgol_mode: str = "nearest"
        # shared
        include_centroid: bool = True
        fps: float | None = None
        keypoint_names: list[str] | None = None

    def __init__(
        self,
        inputs: MovementSmooth.Inputs = Inputs(("tracks",)),
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
        from movement.filtering import rolling_filter, savgol_filter

        p = self.params
        ds, meta = to_movement_dataset(
            df,
            fps=p.fps,
            keypoint_names=p.keypoint_names,
            include_centroid=p.include_centroid,
        )

        position = ds["position"]

        if p.method == "rolling":
            kwargs = {"window": p.window, "statistic": p.statistic}
            if p.min_periods is not None:
                kwargs["min_periods"] = p.min_periods
            smoothed = rolling_filter(position, **kwargs)
        elif p.method == "savgol":
            smoothed = savgol_filter(
                position, window=p.window, polyorder=p.polyorder, mode=p.savgol_mode
            )
        else:
            raise ValueError(f"Unknown smoothing method: {p.method!r}")

        ds["position"] = smoothed
        return from_movement_dataset(ds, df, meta)
