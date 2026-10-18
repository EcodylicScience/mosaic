"""
OrientationRelativeFeature feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.column_contract import (
    POSE_X,
    POSE_Y,
    Column,
    Conditional,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pose_columns import pose_column_pairs
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    BodyScaleResult,
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.params import Declared, Params

from .helpers import (
    block_rows,
    ensure_columns,
    entry_identity,
    finite_keypoints,
    one_row_per_frame_and_id,
    pairwise_median_distance,
    row_max,
    row_median,
    row_quantile,
    sort_nan_last,
    wrap_angle,
)
from .registry import register_feature

_SCALE_DESCRIPTION = (
    "Reference to the upstream body-scale run whose per-sequence mean "
    "scale normalizes distances into scale-free units. Where the "
    "dependency does not resolve, each row falls back to the focal "
    "individual's median pairwise keypoint distance in that frame."
)

_NEAREST_K_DESCRIPTION = "Number of nearest pose-point distances to emit per pair."

_QUANTILES_DESCRIPTION = (
    "Quantiles of the scale-normalized distance from each of the other "
    "individual's keypoints to the focal individual's centroid. "
    "Rotating into the focal's body frame changes each point's "
    "direction but not this distance, which depends only on scale."
)


@final
@register_feature
class OrientationRelativeFeature:
    """
    Orientation-aware relative features between animal pairs, order-agnostic to pose points.

    For each frame and ordered pair (id1 -> id2):
      - Express the other in the focal's body frame (using heading angle and
        global scale).
      - Emit signed centroid deltas, heading difference, quantiles over the
        other's points in the focal's frame, and nearest-k distances.

    One row per ordered pair per frame: ``id1`` is the focal, ``id2`` the other,
    and ``perspective`` says which ordering, so the key is
    ``(frame, id1, id2, perspective)``.
    """

    category = "per-frame"
    name = "orientation-rel"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        Column(C.orientation_col),
        POSE_X,
        POSE_Y,
    )
    # One dist_q<percent> column per quantile, and near_0 to near_<nearest_k - 1>.
    # group and sequence are written whether or not the input has them, blank
    # when it does not.
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.frame_col),
            Column("id1"),
            Column("id2"),
            Column("perspective"),
            Column(C.seq_col),
            Column(C.group_col),
            Column("dx"),
            Column("dy"),
            Column("dtheta"),
            Column("dist_min"),
            Column("dist_median"),
            Column("dist_max"),
            Column("x_min"),
            Column("x_median"),
            Column("x_max"),
            Column("y_min"),
            Column("y_median"),
            Column("y_max"),
            Prefix("near_"),
        ),
        conditional=(Conditional(Prefix("dist_q"), when_param="quantiles"),),
    )

    # A table of positions joined with heading's ANGLE, as on mosaic_v1 tracks
    # that do not record it. The requirements name the columns read.
    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        """Orientation-relative feature parameters."""

        scale: Annotated[BodyScaleResult, Declared(_SCALE_DESCRIPTION)] = Field(
            default_factory=BodyScaleResult
        )
        nearest_k: Annotated[int, Declared(_NEAREST_K_DESCRIPTION)] = Field(
            default=3, ge=1
        )
        quantiles: Annotated[list[float], Declared(_QUANTILES_DESCRIPTION)] = Field(
            default=[0.25, 0.5, 0.75]
        )

    def __init__(
        self,
        inputs: OrientationRelativeFeature.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._scale_lookup: dict[tuple[str, str], Path] | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._scale_lookup = dependency_lookups.get("scale")
        return True

    def _scale_for(self, group: str, sequence: str) -> float | None:
        """Look up mean body scale for the given sequence from the dependency lookup."""
        if self._scale_lookup is None:
            return None

        path = self._scale_lookup.get((group, sequence))
        if path is None:
            return None
        df_scale = pd.read_parquet(path)
        mean_scale = float(df_scale["scale"].dropna().mean())
        if np.isfinite(mean_scale) and mean_scale > 0:
            return mean_scale
        return None

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        ensure_columns(df, [C.frame_col, C.id_col, C.orientation_col])
        pose_pairs = pose_column_pairs(df.columns)
        if not pose_pairs:
            raise ValueError(
                f"{self.name} needs pose keypoint columns (poseX*/poseY*) and this "
                "table carries none. It places each keypoint in a neighbour's "
                "frame; a table with only a centroid has no keypoints to place."
            )
        group, sequence = entry_identity(df)
        global_scale = self._scale_for(group, sequence)

        rows = one_row_per_frame_and_id(df)
        x, y = finite_keypoints(rows, pose_pairs)
        measurable = np.isfinite(x).sum(axis=1) >= 2
        rows, x, y = rows[measurable], x[measurable], y[measurable]
        if global_scale is not None and np.isfinite(global_scale) and global_scale > 0:
            scale = np.full(len(rows), float(global_scale))
        else:
            scale = pairwise_median_distance(x, y)
        individuals = _Individuals.of(
            x, y, rows[C.orientation_col].to_numpy(dtype=float), scale
        )

        frames = rows[C.frame_col].to_numpy().astype(np.int64)
        focal, other = _ordered_pairs(frames)
        # A focal with no usable scale measures no one
        usable = np.isfinite(scale[focal]) & (scale[focal] > 0)
        focal, other = focal[usable], other[usable]
        if not focal.size:
            return pd.DataFrame()

        step = block_rows(x.shape[1])
        blocks = [
            individuals.pair_columns(
                focal[lo : lo + step],
                other[lo : lo + step],
                self.params.quantiles,
                self.params.nearest_k,
            )
            for lo in range(0, focal.size, step)
        ]
        ids = rows[C.id_col]
        columns: dict[str, object] = {
            C.frame_col: frames[focal],
            "id1": ids.iloc[focal].reset_index(drop=True),
            "id2": ids.iloc[other].reset_index(drop=True),
            # The focal-first ordering is perspective 0; its mirror, the same
            # pair computed the other way round, is 1. A frame's rows run in id
            # order, so the order of the rows is the order of the ids.
            "perspective": (other < focal).astype(np.int64),
            C.seq_col: sequence,
            C.group_col: group,
        }
        for name in blocks[0]:
            columns[name] = np.concatenate([block[name] for block in blocks])
        return pd.DataFrame(columns)


def _ordered_pairs(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return every ordered pair of rows that share a frame, as two arrays of row numbers.

    Args:
        frames: The frame of each row, in non-decreasing order.

    Returns:
        ``(focal, other)``, which list the pairs by focal row and then by other
        row. A frame with one row has none.
    """
    n_rows = frames.size
    starts = np.flatnonzero(np.r_[True, frames[1:] != frames[:-1]])
    lengths = np.diff(np.r_[starts, n_rows])
    first = np.repeat(starts, lengths)
    n_others = np.repeat(lengths, lengths) - 1
    focal = np.repeat(np.arange(n_rows), n_others)
    # The k-th row of the focal's frame, leaving the focal's own out
    k = np.arange(focal.size) - np.repeat(np.cumsum(n_others) - n_others, n_others)
    return focal, first[focal] + k + (k >= focal - first[focal])


@dataclass(frozen=True, eq=False)
class _Individuals:
    """The individuals of one entry, a row of each array apiece.

    Attributes:
        x: ``(n, n_keypoints)`` x of each keypoint, NaN where it is missing.
        y: The same for y.
        centre: ``(n, 2)`` mean of the keypoints present.
        heading: ``(n,)`` heading in radians.
        scale: ``(n,)`` the length that this individual's distances divide by.
        cos: ``(n,)`` cosine of the rotation that turns the heading onto +x.
        sin: The sine of that rotation.
    """

    x: np.ndarray
    y: np.ndarray
    centre: np.ndarray
    heading: np.ndarray
    scale: np.ndarray
    cos: np.ndarray
    sin: np.ndarray

    @classmethod
    def of(
        cls, x: np.ndarray, y: np.ndarray, heading: np.ndarray, scale: np.ndarray
    ) -> _Individuals:
        """Return the individuals with these keypoints, headings and scales."""
        n_present = np.isfinite(x).sum(axis=1)
        centre = np.column_stack([np.nansum(x, axis=1), np.nansum(y, axis=1)])
        return cls(
            x,
            y,
            centre / n_present[:, None],
            heading,
            scale,
            np.cos(-heading),
            np.sin(-heading),
        )

    def pair_columns(
        self,
        focal: np.ndarray,
        other: np.ndarray,
        quantiles: list[float],
        nearest_k: int,
    ) -> dict[str, np.ndarray]:
        """Return the output columns of the pairs ``(focal[i], other[i])``.

        Each of the other's keypoints is placed in the focal's body frame:
        centred on the focal's centroid, turned by its heading and divided by its
        scale.
        """
        cos, sin, scale = self.cos[focal], self.sin[focal], self.scale[focal]
        dx = self.x[other] - self.centre[focal, :1]
        dy = self.y[other] - self.centre[focal, 1:]
        rel_x = (cos[:, None] * dx - sin[:, None] * dy) / scale[:, None]
        rel_y = (sin[:, None] * dx + cos[:, None] * dy) / scale[:, None]
        between_x, between_y = (self.centre[other] - self.centre[focal]).T
        ordered, count = sort_nan_last(np.sqrt(rel_x**2 + rel_y**2))

        columns = {
            "dx": (cos * between_x - sin * between_y) / scale,
            "dy": (sin * between_x + cos * between_y) / scale,
            "dtheta": wrap_angle(self.heading[other] - self.heading[focal]),
            "dist_min": ordered[:, 0],
            "dist_median": row_median(ordered, count),
            "dist_max": row_max(ordered, count),
        }
        for q in quantiles:
            columns[f"dist_q{int(q * 100):02d}"] = row_quantile(ordered, count, q)
        for axis, rel in (("x", rel_x), ("y", rel_y)):
            along, along_count = sort_nan_last(rel)
            columns[f"{axis}_min"] = along[:, 0]
            columns[f"{axis}_median"] = row_median(along, along_count)
            columns[f"{axis}_max"] = row_max(along, along_count)
        for i in range(nearest_k):
            # Past the keypoints there is nothing to be nearest
            if i < ordered.shape[1]:
                nearest = np.where(np.isfinite(ordered[:, i]), ordered[:, i], np.nan)
            else:
                nearest = np.full(len(focal), np.nan)
            columns[f"near_{i}"] = nearest
        return columns
