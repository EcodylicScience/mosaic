"""
FrameAggregate -- generic per-frame across-ids summary feature.

Consumes any feature output (or raw tracks) and reduces multi-id data to one row per
frame by applying each chosen aggregation (mean/median/min/max/std/sum/count) to each
chosen column. Optional pre-aggregation transforms cover the
common cases: ``transform="abs"`` for magnitudes, ``threshold=X`` with
``"mean"`` in ``aggs`` for a fraction-of-ids-in-the-frame-exceeding-threshold summary.
An input with an ``arena`` column gives one row per frame and arena, and
``subgroup_col`` one row per frame and value of that column, or of each column when it
lists several.

Composes naturally with pair-aware features: pointing FrameAggregate at
``PairPositionFeatures`` with ``columns=["AB_dist"], aggs=["mean"]`` yields
mean pairwise distance per frame. A pair table holds each pair once per
perspective (A->B and B->A), and ``pairs`` says which rows sum and count read.
``once`` keeps the ``perspective == 0`` rows, so each pair counts once, and
``each_perspective`` reads every row. ``auto``, the default, is resolved by a recipe's
plan: ``once`` when the column is declared symmetric over mirrored rows, neither
``filter_expr`` nor ``subsets`` is set and every ``value_bins`` column is symmetric too,
every row otherwise. A run outside a plan (``mosaic run --feature``) reads every row
under ``auto``; set
``pairs="once"`` there, or ``filter_expr="perspective == 0"``, to count each pair
once.
mean/median/min/max/std read every row under any rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self, final

import pandas as pd
from pydantic import model_validator

from mosaic.core.pipeline.feature_levels import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Conditional,
    ParamColumn,
    ParamIs,
    Provides,
    Requirement,
    When,
    kept,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)

from mosaic.core.params import has_fields
from mosaic.core.schema import ARENA_COLUMN
from mosaic.core.spaces import TRACK_SPACES, TrackSpace

from .aggregate_common import (
    BIN_SUFFIX,
    SUBSET,
    PairAggregateParams,
    reduce_column,
    split_keys,
)
from .registry import register_feature


@final
@register_feature
class FrameAggregate:
    """
    Per-sequence feature reducing multi-id data to one row per frame.

    Output columns (one row per frame, arena, subgroup, subset and value bin):
      - frame
      - arena (if present in input)
      - <subgroup_col> (when set)
      - subset: the subset's name, when ``subsets`` is set; a subset that keeps
        no row of a frame has no row for it
      - <column>_bin: the 0-based bin of each ``value_bins`` column, a nullable
        integer; ``dense`` writes every bin of each frame
      - time (if present in input)
      - <keep_columns>: each kept column, with its one value per row
      - <column>_<agg>: the aggregated value, for each column and aggregation
      - group, sequence (if present in input)

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.frame_aggregate.FrameAggregate.Params`.

    Notes
    -----
    A pair table holds each pair once per perspective. For sum/count,
    ``pairs="once"`` keeps the ``perspective == 0`` rows when the input has a
    ``perspective`` column, so each pair counts once, and
    ``pairs="each_perspective"`` reads every row. Under ``auto`` a recipe's
    plan writes ``once`` for a column that the features behind it declare symmetric over
    mirrored rows, with no ``filter_expr``, no ``subsets`` and no ``value_bins`` column
    that is not symmetric too, and every other ``auto`` reads every row. A run outside a
    plan (``mosaic run --feature``) reads every row under ``auto``; set
    ``pairs="once"``, or ``filter_expr="perspective == 0"``, to count each pair
    once there. For mean/median/min/max/std every row is read under any rule.

    This feature does not filter ``bad_frame`` automatically. Either consume upstream
    output that has already filtered them, or pass
    ``filter_expr="~bad_frame"`` (when the column is present).
    """

    category = "summary"
    name = "frame-aggregate"
    version = "0.3"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"
    follows_ids: bool = False
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = (
        ParamColumn("columns"),
        ParamColumn("subgroup_col"),
        ParamColumn("keep_columns"),
        ParamColumn("value_bins", key="column"),
        ORDER_COLUMN,
        When(ParamColumn("order_col"), (ParamIs("aggs", equals="first_pass"),)),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            ParamColumn("subgroup_col"),
            ParamColumn("keep_columns"),
            ParamColumn("value_bins", key="column", suffix=BIN_SUFFIX),
            ParamColumn(
                "columns",
                suffixes="aggs",
                families=("share",),
                members="categories",
                open_families=("quantile",),
            ),
        ),
        conditional=(
            *kept(C.frame_col, C.time_col, C.seq_col, C.group_col, ARENA_COLUMN),
            Conditional(Column(SUBSET), when_param="subsets"),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    pair_rule_column: ClassVar[str] = "columns"

    class Params(PairAggregateParams):
        """The shared parameters, with every aggregation but ``duration`` and ``share``."""

        @model_validator(mode="after")
        def _refuse_the_aggregations_of_seconds(self) -> Self:
            """Refuse ``duration`` and ``share``, which a frame's rows do not give.

            A frame's group holds one row of each individual, and their seconds
            would sum each individual's step to its next row, outside the frame.
            """
            if not has_fields(self, "aggs"):
                return self
            for agg in ("duration", "share"):
                if agg in self.aggs:
                    raise ValueError(
                        f"frame-aggregate does not take the {agg} aggregation: a "
                        "frame's group would sum each individual's seconds to its "
                        "next frame, which lie outside the frame. Use "
                        "sequence-aggregate or individual-aggregate."
                    )
            return self

        @model_validator(mode="after")
        def _refuse_time_bins(self) -> Self:
            """Refuse ``bin_s``: a frame's rows are one frame each."""
            if not has_fields(self, "bin_s"):
                return self
            if self.bin_s is not None:
                raise ValueError(
                    "frame-aggregate does not take bin_s: its rows are already "
                    "one frame each. Use sequence-aggregate or individual-aggregate."
                )
            return self

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
            keys=[order_col, *split_keys(df, self.params)],
            who=self.name,
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
