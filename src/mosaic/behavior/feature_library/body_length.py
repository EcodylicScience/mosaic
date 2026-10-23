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
swap. The reported ``body_length`` is therefore the median of an individual's
per-frame lengths over the sequence, repeated on each of its rows, beside the
per-frame ``body_length_frame`` it summarizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.column_contract import (
    POSE_X,
    POSE_Y,
    Column,
    ParamColumn,
    ParamIs,
    Provides,
    Requirement,
    When,
    kept,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.params import Declared, Params
from mosaic.core.pose_columns import pose_column_pairs, resolve_keypoint_refs

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

_NAME = "body-length"


@final
@register_feature
class BodyLength:
    """Body length of each individual, per frame and as a per-sequence median.

    Outputs one row per ``(frame, id)`` with a measurable length:
    ``body_length_frame``, that frame's length, and ``body_length``, the median
    of the individual's per-frame lengths across the sequence, in the units of
    the input positions. ``scale-to-bl`` reads ``body_length`` to convert other
    tables into body lengths.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.body_length.BodyLength.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # a median over one entry, which overlap would bias
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        When(POSE_X, (ParamIs("source", equals="keypoints"),)),
        When(POSE_Y, (ParamIs("source", equals="keypoints"),)),
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
            list[int | str] | None, Declared(_KEYPOINTS_DESCRIPTION)
        ] = None
        keypoint_names: Annotated[
            list[str] | None, Declared(_KEYPOINT_NAMES_DESCRIPTION)
        ] = None
        column: Annotated[str, Declared(_COLUMN_DESCRIPTION)] = Field(
            default="body_length", min_length=1
        )

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
            negative = [ref for ref in refs if isinstance(ref, int) and ref < 0]
            if negative:
                raise ValueError(
                    f"{_NAME}: keypoint indices {negative} are negative; a keypoint "
                    "is referred to by its position from 0, or by name."
                )
            names = self.keypoint_names
            if names is not None:
                _ = resolve_keypoint_refs(refs, names, len(names), who=_NAME)
            else:
                named = [ref for ref in refs if isinstance(ref, str)]
                _ = resolve_keypoint_refs(named, None, 0, who=_NAME)
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
        if self.params.source == "column":
            lengths = self._from_column(rows)
        else:
            lengths = self._from_keypoints(rows)

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
        out["body_length"] = out.groupby(C.id_col, sort=False)[
            "body_length_frame"
        ].transform("median")
        return out

    def _from_column(self, rows: pd.DataFrame) -> np.ndarray:
        column = self.params.column
        if column not in rows.columns:
            raise ValueError(
                f"{self.name}: source='column' reads {column!r}, and this table "
                "does not carry it."
            )
        return np.asarray(pd.to_numeric(rows[column], errors="coerce"), dtype=float)

    def _from_keypoints(self, rows: pd.DataFrame) -> np.ndarray:
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
        x, y = finite_keypoints(rows, [pose_pairs[k] for k in chain])
        # One missing point breaks the chain: a length summed over the points
        # that remain would read a fish with a lost tail as a shorter fish.
        segments = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
        return _chain_sum(segments)


def _chain_sum(segments: np.ndarray) -> np.ndarray:
    """Sum each row's segments, NaN where any segment is missing."""
    return np.where(
        np.isfinite(segments).all(axis=1), np.nansum(segments, axis=1), np.nan
    )
