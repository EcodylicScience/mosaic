"""
FrameAggregate -- generic per-frame across-ids summary feature.

Consumes any feature output (or raw tracks) and reduces multi-id data to one
row per frame by applying an aggregation (mean/median/min/max/std/sum/count)
to a chosen column. Optional pre-aggregation transforms cover the common
cases: ``transform="abs"`` for magnitudes, ``threshold=X`` for boolean
"fraction-of-frames-with-condition" summaries.

Composes naturally with pair-aware features: pointing FrameAggregate at
``PairPositionFeatures`` with ``column="AB_dist", agg="mean"`` yields mean
pairwise distance per frame. Perspective duplication (A->B and B->A both
emitted) does not affect mean/median/min/max/std; for sum/count, pass
``filter_expr="perspective == 0"`` to dedupe explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, final

import pandas as pd

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
    resolve_order_col,
)

from .helpers import ensure_columns
from .registry import register_feature


_AGG_MODES = ("mean", "median", "min", "max", "std", "sum", "count")


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

    Params
    ------
    column : str
        Name of the column to aggregate.
    agg : {"mean","median","min","max","std","sum","count"}
        Aggregation applied across ids within each frame. Default "mean".
        Pandas semantics: NaN is skipped for all modes; "count" counts
        non-null values.
    output_column : str | None
        Name of the result column. Default ``f"{column}_{agg}"``.
    filter_expr : str | None
        Optional ``pd.DataFrame.query`` filter applied before aggregation.
        Example: ``"perspective == 0"`` to dedupe pair-perspective inputs
        before sum/count, or ``"~bad_frame"`` to drop flagged frames.
    threshold : float | None
        If set, aggregate the boolean ``(column > threshold)`` instead of
        the raw column. With ``agg="mean"`` this gives a per-frame
        fraction-of-ids-exceeding-threshold (e.g. group_size > 1).
    transform : {"none","abs"}
        If "abs", aggregate ``column.abs()`` instead of ``column``.
        Useful for magnitudes (e.g. |angular velocity|). Default "none".

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

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        column: str
        agg: Literal["mean", "median", "min", "max", "std", "sum", "count"] = "mean"
        output_column: str | None = None
        filter_expr: str | None = None
        threshold: float | None = None
        transform: Literal["none", "abs"] = "none"

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

        # Attach sequence/group metadata (constant per sequence)
        if C.seq_col in df.columns:
            result[C.seq_col] = df[C.seq_col].iloc[0]
        if C.group_col in df.columns:
            result[C.group_col] = df[C.group_col].iloc[0]

        return result
