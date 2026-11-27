"""Body length per individual, the unit a developmental comparison is made in.

A fish that grows from 4 mm to 12 mm between two recordings swims three times as
far in a second for the same gait, and stands three times as far from a
neighbour at the same social distance. Comparing ages in pixels or centimetres
compares sizes. Comparing them in body lengths compares behaviour.

``body-scale`` measures something else: the median distance between an
individual's keypoints, a spread whose relation to length depends on how the
keypoints are placed along the body. This feature measures a length, along a
chain of keypoints named in order from the front to the back, or reads one a
tracker already measured.

The per-frame length is noisy: a missing tail point, a bent body, a keypoint
swap. The reported ``body_length`` is therefore a statistic of per-frame lengths,
by default the median of an individual's over the sequence, repeated on each of
its rows, beside the per-frame ``body_length_frame`` it summarizes. The
statistic, the rows it pools, and whether it applies to the chain or to each of
its segments are options, and a world-frame table is measured in 3D.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, final, override

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.feature_levels import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    POSE_X,
    POSE_Y,
    Column,
    ParamColumn,
    ParamIs,
    Prefix,
    Provides,
    Requirement,
    When,
    kept,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.params import Declared, Params, has_fields
from mosaic.core.spaces import TrackSpace
from mosaic.core.keypoint_refs import check_keypoint_refs, resolve_keypoint_refs
from mosaic.core.pose_columns import pose_column_pairs

from .helpers import ensure_columns, finite_keypoints, one_row_per_frame_and_id
from .registry import register_feature

_SOURCE_DESCRIPTION = (
    "Where the length comes from. keypoints sums the distances between "
    "consecutive keypoints of the chain that keypoints names, a polyline "
    "that follows a bent body. column reads a per-row length a tracker "
    "already measured, from the column that column names."
)

_KEYPOINTS_DESCRIPTION = (
    "The chain of keypoints the length runs along, from the front of the "
    "body to the back, each by index or by name. Unset, every keypoint in "
    "the table's order. A frame where any keypoint of the chain is missing "
    "has no length. Read when source is keypoints."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "Names for the table's keypoints, in index order, so that keypoints can "
    "name them. Must list one name per keypoint the table carries."
)

_COLUMN_DESCRIPTION = (
    "The column holding a length already measured per row, read when source is column."
)

_STATISTIC_DESCRIPTION = (
    "The statistic of the lengths that body_length reports. median is "
    "np.median, mean is np.mean, and percentile is np.percentile with linear "
    "interpolation at the value of percentile."
)

_PERCENTILE_DESCRIPTION = (
    "The percentile, from 0 to 100, that statistic computes. Required with "
    "statistic percentile and refused with any other."
)

_PER_DESCRIPTION = (
    "The rows one reported value pools. id pools each individual's rows. "
    "sequence pools every row of the entry, so every individual records the "
    "same value."
)

_DIMENSIONS_DESCRIPTION = (
    "The dimensions each segment is measured in. 2 reads poseX and poseY and "
    "refuses a table with a Z axis. 3 reads poseX, poseY and poseZ of every "
    "keypoint of the chain and refuses a table without them, as a world-frame "
    "table (mosaic_world_v1) carries them. Needs source: keypoints."
)

_SEGMENT_STATISTIC_DESCRIPTION = (
    "Apply statistic to each segment of the chain over the rows where that "
    "segment is finite, and report the sum of those statistics in chain order, "
    "so a row missing one keypoint still counts for the segments it has. "
    "body_length_frame and the rows written stay those with a finite, positive "
    "full-chain length. Needs source: keypoints."
)

_NAME = "body-length"

_OPTION_DEFAULTS: dict[str, object] = {
    "statistic": "median",
    "percentile": None,
    "per": "id",
    "dimensions": 2,
    "segment_statistic": False,
}


@final
@register_feature
class BodyLength:
    """Body length of each individual, per frame and as a pooled statistic.

    Outputs one row per ``(frame, id)`` with a measurable length:
    ``body_length_frame``, that frame's length, and ``body_length``, by default
    the median of the individual's per-frame lengths across the sequence, in the
    units of the input positions. ``scale-to-bl`` reads ``body_length`` to
    convert other tables into body lengths.

    A table in the world frame (``mosaic_world_v1``) is measured with
    ``dimensions: 3``, along the 3D chain.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.body_length.BodyLength.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = False  # a median over one entry, which overlap would bias
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(
        {"image_px", "plane_cm", "world_m"}
    )
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        When(POSE_X, (ParamIs("source", equals="keypoints"),)),
        When(POSE_Y, (ParamIs("source", equals="keypoints"),)),
        When(
            Prefix("poseZ"),
            (ParamIs("source", equals="keypoints"), ParamIs("dimensions", equals=3)),
        ),
        When(ParamColumn("column"), (ParamIs("source", equals="column"),)),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.frame_col),
            Column(C.id_col),
            Column("body_length_frame"),
            Column("body_length"),
        ),
        conditional=kept(C.group_col, C.seq_col, C.time_col),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        source: Annotated[
            Literal["keypoints", "column"], Declared(_SOURCE_DESCRIPTION)
        ] = "keypoints"
        keypoints: Annotated[
            list[int | str] | None, Declared(_KEYPOINTS_DESCRIPTION, keypoint="either")
        ] = None
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION, keypoint="name")
        ] = None
        column: Annotated[str, Declared(_COLUMN_DESCRIPTION)] = Field(
            default="body_length", min_length=1
        )
        statistic: Annotated[
            Literal["median", "mean", "percentile"], Declared(_STATISTIC_DESCRIPTION)
        ] = "median"
        percentile: Annotated[float | None, Declared(_PERCENTILE_DESCRIPTION)] = Field(
            default=None, ge=0, le=100
        )
        per: Annotated[Literal["id", "sequence"], Declared(_PER_DESCRIPTION)] = "id"
        dimensions: Annotated[Literal[2, 3], Declared(_DIMENSIONS_DESCRIPTION)] = 2
        segment_statistic: Annotated[bool, Declared(_SEGMENT_STATISTIC_DESCRIPTION)] = (
            False
        )

        @model_validator(mode="after")
        def _check_percentile(self) -> Self:
            """Refuse a percentile without its statistic and the reverse."""
            if not has_fields(self, "statistic", "percentile"):
                return self
            if self.statistic == "percentile" and self.percentile is None:
                raise ValueError(
                    f"{_NAME}: statistic='percentile' needs percentile, a number "
                    "from 0 to 100."
                )
            if self.statistic != "percentile" and self.percentile is not None:
                raise ValueError(
                    f"{_NAME}: percentile is read only with "
                    f"statistic='percentile', and statistic is {self.statistic!r}."
                )
            return self

        @model_validator(mode="after")
        def _check_keypoint_options(self) -> Self:
            """Refuse a keypoint option that a column source would never read."""
            if not has_fields(self, "source", "dimensions", "segment_statistic"):
                return self
            if self.source != "column":
                return self
            if self.dimensions != 2:
                raise ValueError(
                    f"{_NAME}: dimensions={self.dimensions} reads keypoints, and "
                    "source='column' reads a measured length. Use "
                    "source='keypoints' or leave dimensions at 2."
                )
            if self.segment_statistic:
                raise ValueError(
                    f"{_NAME}: segment_statistic reads keypoints, and "
                    "source='column' reads a measured length. Use "
                    "source='keypoints' or leave segment_statistic false."
                )
            return self

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, less each option while it holds its default.

            A run that leaves them unset writes the output of the version before
            they existed, so it keeps its identifier.
            """
            dumped = super().identity_dump()
            for name, default in _OPTION_DEFAULTS.items():
                if name in dumped and dumped[name] == default:
                    del dumped[name]
            return dumped

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Refuse at construction a chain that no table could measure."""
            refs = self.keypoints
            if refs is None:
                return self
            if len(refs) < 2:
                raise ValueError(
                    f"{_NAME}: keypoints={refs} names fewer than two keypoints, "
                    "and a length runs between at least two."
                )
            check_keypoint_refs(refs, self.keypoint_names, _NAME)
            return self

    def __init__(
        self,
        inputs: BodyLength.Inputs = Inputs(("tracks",)),
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
        rows = one_row_per_frame_and_id(df)
        segments: np.ndarray | None = None
        if self.params.source == "column":
            lengths = self._from_column(rows)
        else:
            segments = self._segments(rows)
            lengths = _chain_sum(segments)

        measured = np.isfinite(lengths) & (lengths > 0)
        if not measured.any():
            return pd.DataFrame()
        keep = [
            c
            for c in (C.frame_col, C.time_col, C.group_col, C.seq_col, C.id_col)
            if c in rows.columns
        ]
        out = rows.loc[measured, keep].reset_index(drop=True)
        out["body_length_frame"] = lengths[measured]
        out["body_length"] = self._pooled(rows, lengths, measured, segments)[measured]
        return out

    def _pooled(
        self,
        rows: pd.DataFrame,
        lengths: np.ndarray,
        measured: np.ndarray,
        segments: np.ndarray | None,
    ) -> np.ndarray:
        """The reported length of each row of *rows*, NaN for a unit with no length."""
        pooled = np.full(len(rows), np.nan)
        for unit in self._units(rows):
            if not measured[unit].any():
                continue
            if self.params.segment_statistic and segments is not None:
                value = 0.0
                for column in segments[unit].T:
                    value += self._statistic(column[np.isfinite(column)])
            else:
                value = self._statistic(lengths[unit[measured[unit]]])
            pooled[unit] = value
        return pooled

    def _units(self, rows: pd.DataFrame) -> list[np.ndarray]:
        """The row positions one reported value pools, as ``per`` says."""
        if self.params.per == "sequence":
            return [np.arange(len(rows))]
        by_id = rows.groupby(C.id_col, sort=False).indices
        return [np.asarray(positions) for positions in by_id.values()]

    def _statistic(self, values: np.ndarray) -> float:
        """The configured statistic of *values*."""
        percentile = self.params.percentile
        if percentile is not None:
            return float(np.percentile(values, percentile, method="linear"))
        if self.params.statistic == "mean":
            return float(np.mean(values))
        return float(np.median(values))

    def _from_column(self, rows: pd.DataFrame) -> np.ndarray:
        column = self.params.column
        if column not in rows.columns:
            raise ValueError(
                f"{self.name}: source='column' reads {column!r}, and this table "
                "does not carry it."
            )
        return np.asarray(pd.to_numeric(rows[column], errors="coerce"), dtype=float)

    def _segments(self, rows: pd.DataFrame) -> np.ndarray:
        """The ``(rows, keypoints - 1)`` lengths between consecutive chain keypoints.

        A segment is NaN where either of its keypoints is missing.
        """
        p = self.params
        pose_pairs = pose_column_pairs(rows.columns)
        if len(pose_pairs) < 2:
            raise ValueError(
                f"{self.name} needs at least two pose keypoints (poseX*/poseY*) "
                f"and this table carries {len(pose_pairs)}. A table with only a "
                "centroid has no length to measure; pass source='column' when "
                "the tracker reports one."
            )
        n_keypoints = len(pose_pairs)
        if p.keypoint_names is not None and len(p.keypoint_names) != n_keypoints:
            raise ValueError(
                f"{self.name}: keypoint_names has {len(p.keypoint_names)} names "
                f"and the table carries {n_keypoints} keypoints; a name is matched "
                "to a keypoint by position, so the two must agree."
            )
        chain = (
            list(range(n_keypoints))
            if p.keypoints is None
            else resolve_keypoint_refs(
                p.keypoints, p.keypoint_names, n_keypoints, who=self.name
            )
        )
        chain_pairs = [pose_pairs[k] for k in chain]
        if p.dimensions == 3:
            return self._segments_3d(rows, chain_pairs)
        self._refuse_a_height_axis(rows)
        x, y = finite_keypoints(rows, chain_pairs)
        # One missing point breaks the chain: a length summed over the points
        # that remain would read a fish with a lost tail as a shorter fish.
        return np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))

    def _refuse_a_height_axis(self, rows: pd.DataFrame) -> None:
        """Refuse a table with a height axis under ``dimensions: 2``."""
        carries = [
            str(c) for c in rows.columns if str(c) == "Z" or str(c).startswith("poseZ")
        ]
        if carries:
            raise ValueError(
                f"{self.name}: this table carries a height axis ({carries[0]!r}), a "
                "world-frame table, and dimensions is 2, which would measure the "
                "chain projected onto one plane. Set dimensions: 3 to measure it in "
                "3D."
            )

    def _segments_3d(
        self, rows: pd.DataFrame, chain_pairs: list[tuple[str, str]]
    ) -> np.ndarray:
        """The 3D segment lengths of *chain_pairs*, the norm of each step."""
        if "Z" not in rows.columns:
            raise ValueError(
                f"{self.name}: dimensions is 3 and this table has no Z column. "
                "A planar table has no height; 3D lengths need a world-frame "
                "table (mosaic_world_v1), and a planar table needs dimensions: 2."
            )
        z_columns = [
            f"poseZ{x_column.removeprefix('poseX')}" for x_column, _ in chain_pairs
        ]
        absent = [column for column in z_columns if column not in rows.columns]
        if absent:
            raise ValueError(
                f"{self.name}: dimensions is 3 and the chain's keypoints lack "
                f"{', '.join(absent)}; every keypoint of the chain needs its "
                "poseZ column."
            )
        x, y = finite_keypoints(rows, chain_pairs)
        z = rows[z_columns].to_numpy(dtype=float)
        z = np.where(np.isfinite(z), z, np.nan)
        steps = np.stack(
            [np.diff(x, axis=1), np.diff(y, axis=1), np.diff(z, axis=1)], axis=-1
        )
        return np.linalg.norm(steps, axis=-1)


def _chain_sum(segments: np.ndarray) -> np.ndarray:
    """Sum each row's segments, NaN where any segment is missing."""
    return np.where(
        np.isfinite(segments).all(axis=1), np.nansum(segments, axis=1), np.nan
    )
