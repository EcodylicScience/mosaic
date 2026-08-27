"""
FrameAggregate -- generic per-frame across-ids summary feature.

Consumes any feature output (or raw tracks) and reduces multi-id data to one
row per frame by applying an aggregation (mean/median/min/max/std/sum/count)
to a chosen column. Optional pre-aggregation transforms cover the common
cases: ``transform="abs"`` for magnitudes, ``threshold=X`` with
``agg="mean"`` for a fraction-of-ids-in-the-frame-exceeding-threshold summary.

Composes naturally with pair-aware features: pointing FrameAggregate at
``PairPositionFeatures`` with ``column="AB_dist", agg="mean"`` yields mean
pairwise distance per frame. Perspective duplication (A->B and B->A both
emitted) does not affect mean/median/min/max/std; for sum/count, pass
``filter_expr="perspective == 0"`` to dedupe explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, final

import pandas as pd

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.params import Declared, Params

from .helpers import ensure_columns
from .registry import register_feature


_AGG_MODES = ("mean", "median", "min", "max", "std", "sum", "count")

_COLUMN_DESCRIPTION = "The name of the column to aggregate."

_AGG_DESCRIPTION = (
    "The aggregation applied across ids within each frame. Known values "
    "are mean, median, min, max, std, sum and count. Without threshold, "
    "NaN values are skipped for every mode, and count counts non-null "
    "values."
)

_OUTPUT_COLUMN_DESCRIPTION = (
    "The name of the result column. Unset, it is column followed by agg, "
    "joined with an underscore."
)

_FILTER_EXPR_DESCRIPTION = (
    "A pandas DataFrame.query expression applied before aggregation, for "
    "example to dedupe pair-perspective input or drop flagged frames."
)

_THRESHOLD_DESCRIPTION = (
    "Aggregate the boolean column > threshold instead of the raw column. "
    "Unset, the raw column is aggregated. With agg set to mean, the "
    "result is the fraction of ids in the frame exceeding the threshold."
)

_TRANSFORM_DESCRIPTION = (
    "A transform applied to the column before aggregation. Known values "
    "are none and abs. abs aggregates the column's absolute value, useful "
    "for a magnitude such as angular velocity."
)


@final
@register_feature
class FrameAggregate:
    """
    Per-sequence feature reducing multi-id data to one row per frame.

    Output columns (one row per frame):
      - frame
      - time (if present in input)
      - <output_column>: the aggregated value
      - group, sequence (if present in input)

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.frame_aggregate.FrameAggregate.Params`.

    Notes
    -----
    Pair-perspective dedup is **not** needed for mean/median/min/max/std --
    duplicate values per pair yield the same scalar. For sum/count, dedup
    explicitly with ``filter_expr="perspective == 0"``.

    This feature does not filter ``bad_frame`` automatically. Either consume
    upstream output that has already filtered them, or pass
    ``filter_expr="~bad_frame"`` (when the column is present).
    """

    category = "summary"
    name = "frame-aggregate"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        column: Annotated[str, Declared(_COLUMN_DESCRIPTION)]
        agg: Annotated[
            Literal["mean", "median", "min", "max", "std", "sum", "count"],
            Declared(_AGG_DESCRIPTION),
        ] = "mean"
        output_column: Annotated[str | None, Declared(_OUTPUT_COLUMN_DESCRIPTION)] = (
            None
        )
        filter_expr: Annotated[str | None, Declared(_FILTER_EXPR_DESCRIPTION)] = None
        threshold: Annotated[float | None, Declared(_THRESHOLD_DESCRIPTION)] = None
        transform: Annotated[
            Literal["none", "abs"], Declared(_TRANSFORM_DESCRIPTION)
        ] = "none"

    def __init__(
        self,
        inputs: FrameAggregate.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State protocol (stateless) ---

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
        if df.empty:
            return pd.DataFrame()

        p = self.params
        order_col = resolve_order_col(df)
        ensure_columns(df, [p.column])

        if p.filter_expr:
            df = df.query(p.filter_expr)
            if df.empty:
                return pd.DataFrame()

        s = df[p.column]
        if p.transform == "abs":
            s = s.abs()
        if p.threshold is not None:
            s = (s > p.threshold).astype(float)

        out_col = p.output_column or f"{p.column}_{p.agg}"

        work = pd.DataFrame({order_col: df[order_col].to_numpy(), "_v": s.to_numpy()})
        has_time = "time" in df.columns
        if has_time:
            work["time"] = df["time"].to_numpy()

        named = {out_col: pd.NamedAgg(column="_v", aggfunc=p.agg)}
        if has_time:
            named["time"] = pd.NamedAgg(column="time", aggfunc="first")
        result = work.groupby(order_col, sort=True).agg(**named).reset_index()

        # Reorder columns: frame, time (if present), value, then any metadata
        cols = [order_col]
        if has_time:
            cols.append("time")
        cols.append(out_col)
        result = result[cols]

        # Identity per frame, not from row 0. Constant per sequence, but with
        # overlap the input spans the neighbouring sequences and row 0 is the
        # previous one's; a frame belongs to exactly one sequence, so the map is
        # exact either way.
        for meta_col in (C.seq_col, C.group_col):
            if meta_col in df.columns:
                result[meta_col] = result[order_col].map(
                    df.groupby(order_col)[meta_col].first()
                )

        return result
