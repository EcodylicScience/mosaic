"""
SequenceAggregate -- one summary row per entry.

Consumes any feature output (or raw tracks) and reduces every row of an entry
to one value by applying an aggregation (mean/median/min/max/std/sum/count) to
a chosen column, over all its frames and all its ids. The optional
pre-aggregation steps of ``frame-aggregate`` apply here too: ``transform="abs"``
for magnitudes, and ``threshold=X`` with ``agg="mean"`` for the fraction of rows
that exceed it.

The class docstring states which rows of a pair table sum and count read.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import pandas as pd

from mosaic.core.pipeline.column_contract import (
    Column,
    ParamColumn,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.pipeline.types.feature import RelatesLevel

from .aggregate_common import PairAggregateParams, reduce_column
from .registry import register_feature


@final
@register_feature
class SequenceAggregate:
    """
    Per-sequence feature reducing every row of an entry to one row.

    Output columns (one row per entry):
      - group, sequence: the entry, copied from the frame
      - <output_column>: the aggregated value

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.sequence_aggregate.SequenceAggregate.Params`.

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
    name = "sequence-aggregate"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # One row per entry, which a neighbor's rows would change.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"
    follows_ids: bool = False
    requires: ClassVar[tuple[Requirement, ...]] = (
        ParamColumn("column"),
        Column(C.group_col),
        Column(C.seq_col),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.group_col),
            Column(C.seq_col),
            ParamColumn("output_column", fallback="{column}_{agg}"),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    pair_rule_column: ClassVar[str] = "column"

    class Params(PairAggregateParams):
        pass

    def __init__(
        self,
        inputs: SequenceAggregate.Inputs = Inputs(("tracks",)),
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
        return reduce_column(
            df,
            self.params,
            keys=[C.group_col, C.seq_col],
            pairs=self.params.pair_read(),
        )
