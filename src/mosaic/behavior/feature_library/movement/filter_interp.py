"""Movement-based confidence filtering and interpolation feature."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, final

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

_CONFIDENCE_THRESHOLD_DESCRIPTION = (
    "Confidence cutoff below which a point is set to NaN. Skipped when "
    "the input has no confidence columns."
)

_INTERPOLATION_METHOD_DESCRIPTION = (
    "Method used to interpolate over NaN gaps in position data. Known "
    "values are linear, nearest, zero, slinear, quadratic, cubic, "
    "quintic, polynomial, barycentric, krogh, pchip, spline, akima and "
    "makima."
)

_MAX_GAP_DESCRIPTION = (
    "The longest gap that is interpolated. Unset, gaps of any length are interpolated."
)

_INCLUDE_CENTROID_DESCRIPTION = (
    "Include the body center (X, Y) as an additional keypoint named "
    "centroid, alongside the pose keypoints."
)

_FPS_DESCRIPTION = (
    "Frame rate used for the movement dataset's time dimension. Unset, "
    "the time dimension uses frame numbers instead of seconds."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "Names for the pose keypoints, one per poseX/poseY column pair in "
    "column order. Unset, they are named keypoint_0, keypoint_1, and so "
    "on."
)


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
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        confidence_threshold: Annotated[
            float, Declared(_CONFIDENCE_THRESHOLD_DESCRIPTION)
        ] = 0.6
        interpolation_method: Annotated[
            str,
            Field(
                examples=[
                    "linear",
                    "nearest",
                    "zero",
                    "slinear",
                    "quadratic",
                    "cubic",
                    "quintic",
                    "polynomial",
                    "barycentric",
                    "krogh",
                    "pchip",
                    "spline",
                    "akima",
                    "makima",
                ]
            ),
            Declared(_INTERPOLATION_METHOD_DESCRIPTION),
        ] = "linear"
        max_gap: Annotated[
            int | None, Declared(_MAX_GAP_DESCRIPTION, unit="frames")
        ] = None
        include_centroid: Annotated[bool, Declared(_INCLUDE_CENTROID_DESCRIPTION)] = (
            True
        )
        fps: Annotated[float | None, Declared(_FPS_DESCRIPTION, unit="fps")] = None
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
        ] = None

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
