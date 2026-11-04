"""
FrameAggregate -- generic per-frame across-ids summary feature.

Consumes any feature output (or raw tracks) and reduces multi-id data to one
row per frame by applying an aggregation (mean/median/min/max/std/sum/count)
to a chosen column. Optional pre-aggregation transforms cover the common
cases: ``transform="abs"`` for magnitudes, ``threshold=X`` with
``agg="mean"`` for a fraction-of-ids-in-the-frame-exceeding-threshold summary.

Composes naturally with pair-aware features: pointing FrameAggregate at
``PairPositionFeatures`` with ``column="AB_dist", agg="mean"`` yields mean
pairwise distance per frame. A pair table holds each pair once per
perspective (A->B and B->A), and ``pairs`` says which rows sum and count read.
``once`` keeps the ``perspective == 0`` rows, so each pair counts once, and
``each_perspective`` reads every row. ``auto``, the default, is resolved by a
recipe's plan: ``once`` when the column is declared symmetric over mirrored
rows and ``filter_expr`` is unset, every row otherwise. A run outside a plan
(``mosaic run --feature``) reads every row under ``auto``; set ``pairs="once"``
there, or ``filter_expr="perspective == 0"``, to count each pair once.
mean/median/min/max/std read every row under any rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import pandas as pd

from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    ParamColumn,
    Provides,
    Requirement,
    kept,
)
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

from .aggregate_common import PairAggregateParams, reduce_column
from .registry import register_feature


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
    A pair table holds each pair once per perspective. For sum/count,
    ``pairs="once"`` keeps the ``perspective == 0`` rows when the input has a
    ``perspective`` column, so each pair counts once, and
    ``pairs="each_perspective"`` reads every row. Under ``auto`` a recipe's
    plan writes ``once`` for a column that the features behind it declare
    symmetric over mirrored rows, with no ``filter_expr``, and every other
    ``auto`` reads every row. A run outside a plan (``mosaic run --feature``)
    reads every row under ``auto``; set ``pairs="once"``, or
    ``filter_expr="perspective == 0"``, to count each pair once there. For
    mean/median/min/max/std every row is read under any rule.

    This feature does not filter ``bad_frame`` automatically. Either consume
    upstream output that has already filtered them, or pass
    ``filter_expr="~bad_frame"`` (when the column is present).
    """

    category = "summary"
    name = "frame-aggregate"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"
    follows_ids: bool = False
    requires: ClassVar[tuple[Requirement, ...]] = (ParamColumn("column"), ORDER_COLUMN)
    provides: ClassVar[Provides] = Provides(
        adds=(ParamColumn("output_column", fallback="{column}_{agg}"),),
        conditional=kept(C.frame_col, C.time_col, C.seq_col, C.group_col),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    pair_rule_column: ClassVar[str] = "column"

    class Params(PairAggregateParams):
        pass

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

        order_col = resolve_order_col(df)
        # A table without frames is ordered by its time, which is then the group
        # key and is not aggregated beside it.
        has_time = "time" in df.columns and order_col != "time"
        result = reduce_column(
            df,
            self.params,
            keys=[order_col],
            first=["time"] if has_time else [],
            pairs=self.params.pair_read(),
        )
        if result.empty:
            return pd.DataFrame()

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
