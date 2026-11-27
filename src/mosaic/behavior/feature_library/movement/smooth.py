"""Movement-based trajectory smoothing feature."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, final, override

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.feature_levels import RelatesLevel
from mosaic.core.pipeline.column_contract import Provides, Requirement
from mosaic.core.pipeline.types import DependencyLookup, InputStream, TrackInputs
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.params import Declared, Params, has_fields
from mosaic.optional_dependency import Prerequisite

from ..registry import register_feature
from .convert import (
    MOVEMENT,
    MOVEMENT_PROVIDES,
    MOVEMENT_REQUIRES,
    from_movement_dataset,
    movement_arrays,
    to_movement_dataset,
    write_positions,
)

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

_ROLLING_EDGES_DESCRIPTION = (
    "How the rolling mean treats the ends of a series. min_periods is the "
    "movement library's rolling filter, whose gaps follow the min_periods "
    "param. zero_padded averages a window that runs past an end as if the "
    "missing samples were 0, so a series keeps its length and its values "
    "near an end are pulled toward 0. It needs method rolling, statistic mean "
    "and an odd window. It refuses min_periods, an fps, and a polyorder or "
    "savgol_mode off its default, none of which it reads. It computes "
    "positions in float64 without the movement library, where min_periods "
    "and the savgol method round them to float32 first. A NaN sample makes "
    "NaN every output whose window covers it, and a series shorter than "
    "window is NaN throughout."
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

    With ``rolling_edges="zero_padded"`` the rolling mean is
    ``np.convolve(x, np.ones(window) / window, mode="same")`` of each id's
    series of each coordinate, which treats the samples beyond an end as 0. It
    computes in float64 and does not import ``movement``. The ``rolling`` and
    ``savgol`` paths round positions to float32 before the filter. The feature
    declares the ``movement`` prerequisite whatever its params, so a zero-padded
    step is planned and run only where ``movement`` is installed.

    The output is a full track DataFrame with smoothed positions replacing
    the originals, so downstream features can chain off the ``Result`` the
    run returned, or its storage name:
    ``Inputs((Result(feature="movement-smooth__from__tracks"),))``.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.movement.smooth.MovementSmooth.Params`.
    """

    category = "per-frame"
    name = "movement-smooth"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    prerequisites: tuple[Prerequisite, ...] = (MOVEMENT,)
    emits: EmitsLevel = "as-input"
    follows_ids: bool = True
    requires: ClassVar[tuple[Requirement, ...]] = MOVEMENT_REQUIRES
    provides: ClassVar[Provides] = MOVEMENT_PROVIDES

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
        rolling_edges: Annotated[
            Literal["min_periods", "zero_padded"],
            Declared(_ROLLING_EDGES_DESCRIPTION),
        ] = "min_periods"
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
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION, keypoint="name")
        ] = None

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, less ``rolling_edges`` while it is ``min_periods``.

            A run that leaves it at its default writes the output of the version
            before it existed, so it keeps its identifier.
            """
            dumped = super().identity_dump()
            if self.rolling_edges == "min_periods":
                _ = dumped.pop("rolling_edges", None)
            return dumped

        @model_validator(mode="after")
        def _zero_padded_is_a_rolling_mean(self) -> Self:
            """Refuse zero-padded edges for any but an odd-window rolling mean."""
            if not has_fields(
                self,
                "rolling_edges",
                "method",
                "statistic",
                "min_periods",
                "window",
                "fps",
                "polyorder",
                "savgol_mode",
            ):
                return self
            if self.rolling_edges != "zero_padded":
                return self
            if self.method != "rolling":
                message = (
                    "rolling_edges zero_padded is a rolling mean, and method is "
                    f"{self.method!r}"
                )
                raise ValueError(message)
            if self.statistic != "mean":
                message = (
                    "rolling_edges zero_padded is a rolling mean, and statistic is "
                    f"{self.statistic!r}: set statistic to mean"
                )
                raise ValueError(message)
            if self.min_periods is not None:
                message = (
                    "min_periods leaves a gap at the edges and zero_padded fills "
                    "them, so min_periods is refused with rolling_edges zero_padded"
                )
                raise ValueError(message)
            if self.fps is not None:
                message = (
                    "rolling_edges zero_padded builds no time axis, so fps is "
                    "refused with it"
                )
                raise ValueError(message)
            fields = type(self).model_fields
            for name in ("polyorder", "savgol_mode"):
                if getattr(self, name) != fields[name].default:
                    message = (
                        f"{name} is read only when method is savgol, and "
                        "rolling_edges zero_padded needs method rolling"
                    )
                    raise ValueError(message)
            if self.window < 1 or self.window % 2 == 0:
                message = (
                    "rolling_edges zero_padded needs window to be a positive odd "
                    f"number, and window is {self.window}: an even window centers "
                    "one sample earlier than an odd one"
                )
                raise ValueError(message)
            return self

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

        p = self.params
        if p.rolling_edges == "zero_padded":
            return self._zero_padded_mean(df)

        _ = MOVEMENT.load()
        from movement.filtering import rolling_filter, savgol_filter

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

    def _zero_padded_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth each id's series of each keypoint coordinate in float64."""
        p = self.params
        position, _, meta = movement_arrays(df, p.keypoint_names, p.include_centroid)
        smoothed = np.full_like(position, np.nan)
        if position.shape[0] >= p.window:
            kernel = np.ones(p.window) / p.window
            _, n_axes, n_keypoints, n_individuals = position.shape
            for axis in range(n_axes):
                for keypoint in range(n_keypoints):
                    for individual in range(n_individuals):
                        smoothed[:, axis, keypoint, individual] = np.convolve(
                            position[:, axis, keypoint, individual],
                            kernel,
                            mode="same",
                        )
        return write_positions(df, smoothed, meta)
