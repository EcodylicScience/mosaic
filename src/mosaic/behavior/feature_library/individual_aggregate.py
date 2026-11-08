"""
IndividualAggregate -- one summary row per individual of an entry.

Consumes any feature output (or raw tracks) and reduces the rows of each
individual to one row by applying each chosen aggregation
(mean/median/min/max/std/sum/count) to each chosen column, over all its frames.
The optional pre-aggregation steps of ``frame-aggregate`` apply here too:
``transform="abs"`` for magnitudes, and ``threshold=X`` with ``"mean"`` in
``aggs`` for the fraction of an individual's rows that exceed it. An input with
an ``arena`` column keys each row by its arena too, and ``subgroup_col`` splits
an individual's rows by that column's value.

Individual input is grouped by ``id``, and pair input by the focal ``id1``,
which the output names ``id``. The class docstring states why pair input reads
every perspective row.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import pandas as pd

from mosaic.core.pipeline.column_contract import (
    AnyOf,
    Column,
    ParamColumn,
    Provides,
    Requirement,
    kept,
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

from mosaic.core.schema import ARENA_COLUMN

from .aggregate_common import AggregateParams, reduce_column, split_keys
from .registry import register_feature

_FOCAL_ID = "id1"


@final
@register_feature
class IndividualAggregate:
    """
    Per-sequence feature reducing the rows of each individual to one row.

    Output columns (one row per individual and subgroup):
      - group, sequence: the entry, copied from the frame
      - id: the individual, or the focal ``id1`` of pair input
      - arena: when the input has it
      - <subgroup_col>: when set
      - <column>_<agg>: the aggregated value, for each column and aggregation

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.individual_aggregate.IndividualAggregate.Params`.

    Notes
    -----
    Pair input is reduced over every perspective row grouped by ``id1``. Each
    ordered pair row is one focal's view of the pair, so a directional column
    such as ``A_speed`` belongs to its focal, and ``perspective == 0`` would
    drop every row whose focal is the higher id. Every aggregation reads all of
    a focal's rows; the params have no ``pairs`` field.

    This feature does not filter ``bad_frame`` automatically. Either consume
    upstream output that has already filtered them, or pass
    ``filter_expr="~bad_frame"`` (when the column is present).
    """

    category = "summary"
    name = "individual-aggregate"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # One row per individual of the entry, which a neighbor's rows would change.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    requires: ClassVar[tuple[Requirement, ...]] = (
        ParamColumn("columns"),
        ParamColumn("subgroup_col"),
        AnyOf((Column(C.id_col), Column(_FOCAL_ID))),
        Column(C.group_col),
        Column(C.seq_col),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.group_col),
            Column(C.seq_col),
            Column(C.id_col),
            ParamColumn("subgroup_col"),
            ParamColumn("columns", suffixes="aggs"),
        ),
        conditional=kept(ARENA_COLUMN),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(AggregateParams):
        pass

    def __init__(
        self,
        inputs: IndividualAggregate.Inputs = Inputs(("tracks",)),
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
        # The filter reads the input's own names, so pair input keeps id1 until
        # the rows are reduced.
        focal = C.id_col if C.id_col in df.columns else _FOCAL_ID
        result = reduce_column(
            df,
            self.params,
            keys=[C.group_col, C.seq_col, focal, *split_keys(df, self.params)],
            pairs="each_perspective",
        )
        return result.rename(columns={focal: C.id_col})
