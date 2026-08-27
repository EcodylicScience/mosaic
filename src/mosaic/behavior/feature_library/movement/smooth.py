"""Movement-based trajectory smoothing feature."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, final

import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    InputStream,
    TrackInputs,
)
from mosaic.core.params import Declared, Params

from ..registry import register_feature
from .convert import _ensure_movement, from_movement_dataset, to_movement_dataset

_METHOD_DESCRIPTION = (
    "The movement library filter used to smooth the trajectory. savgol "
    "applies a Savitzky-Golay filter, and rolling applies a rolling-window "
    "statistic."
)

_WINDOW_DESCRIPTION = (
    "The size of the smoothing window, as the number of observations "
    "covered by each window."
)

_STATISTIC_DESCRIPTION = (
    "Which statistic the rolling filter computes over each window. Read "
    "only when method is rolling."
)

_MIN_PERIODS_DESCRIPTION = (
    "Minimum observations required within a window for the rolling filter "
    "to produce a value there rather than a gap. Unset, it equals window. "
    "Read only when method is rolling."
)

_POLYORDER_DESCRIPTION = (
    "The order of the polynomial fit within each smoothing window. Must "
    "be less than window. Read only when method is savgol."
)

_SAVGOL_MODE_DESCRIPTION = (
    "The edge-padding mode for the Savitzky-Golay filter, forwarded to "
    "scipy.signal.savgol_filter. Read only when method is savgol. Known "
    "values are mirror, constant, nearest, wrap and interp. Any other "
    "value raises. interp also raises on a window with a NaN at its "
    "edge, under scipy 1.17 and later."
)

_INCLUDE_CENTROID_DESCRIPTION = (
    "Include the X/Y body center as an additional keypoint named "
    "centroid, alongside any pose keypoints. False on a table with no "
    "pose columns raises."
)

_FPS_DESCRIPTION = (
    "Frame rate used to build the time axis. Unset, the time axis uses "
    "frame numbers instead."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "Names for the pose keypoints, one per poseX/poseY column pair. "
    "Unset, defaults to keypoint_0, keypoint_1, and so on. A count that "
    "does not match the number of pose column pairs raises."
)


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

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.movement.smooth.MovementSmooth.Params`.
    """

    category = "per-frame"
    name = "movement-smooth"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        method: Annotated[
            Literal["rolling", "savgol"], Declared(_METHOD_DESCRIPTION)
        ] = "savgol"
        window: Annotated[int, Declared(_WINDOW_DESCRIPTION, unit="frames")] = 5
        # rolling-specific
        statistic: Annotated[
            Literal["median", "mean", "max", "min"], Declared(_STATISTIC_DESCRIPTION)
        ] = "median"
        min_periods: Annotated[
            int | None, Declared(_MIN_PERIODS_DESCRIPTION, unit="frames")
        ] = None
        # savgol-specific
        polyorder: Annotated[int, Declared(_POLYORDER_DESCRIPTION)] = 2
        savgol_mode: Annotated[
            str,
            Field(examples=["mirror", "constant", "nearest", "wrap", "interp"]),
            Declared(_SAVGOL_MODE_DESCRIPTION),
        ] = "nearest"
        # shared
        include_centroid: Annotated[bool, Declared(_INCLUDE_CENTROID_DESCRIPTION)] = (
            True
        )
        fps: Annotated[float | None, Declared(_FPS_DESCRIPTION, unit="fps")] = None
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
        ] = None

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
