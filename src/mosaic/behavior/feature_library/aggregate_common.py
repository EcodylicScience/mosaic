"""
Shared reduction of the aggregate features.

``frame-aggregate``, ``sequence-aggregate`` and ``individual-aggregate`` reduce
columns of a feature output or of raw tracks with the same steps: filter,
transform, threshold, then a group-by. :func:`reduce_column` holds those steps
and :class:`AggregateParams` the parameters they read, so the three features
differ only in the keys they group by and in whether they take a ``pairs``
rule (:class:`PairAggregateParams`). Each feature also groups by the input's
``arena`` column when it has one, and by ``subgroup_col`` when it is set
(:func:`split_keys`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    PAIR_COUNTING_AGGS,
    PairRead,
    PairRule,
)
from mosaic.core.schema import ARENA_COLUMN

type Aggregation = Literal["mean", "median", "min", "max", "std", "sum", "count"]

_COLUMNS_DESCRIPTION = (
    "The names of the columns to aggregate. Each column is reduced by each "
    "aggregation in aggs, and the result column is named column followed by "
    "agg, joined with an underscore."
)

_AGGS_DESCRIPTION = (
    "The aggregations applied across the rows that the feature reduces "
    "together: those of one frame, one entry or one id, as the feature "
    "groups them. Known values "
    "are mean, median, min, max, std, sum and count. Without threshold, "
    "NaN values are skipped for every mode, and count counts non-null "
    "values."
)

_SUBGROUP_COL_DESCRIPTION = (
    "A column of the input that splits the rows reduced together into "
    "subgroups, one result row per value, such as a tag column or the event "
    "of ffgroups. Rows whose value is null belong to no subgroup and are "
    "dropped. Unset, the rows are not split."
)
_FILTER_EXPR_DESCRIPTION = (
    "A pandas DataFrame.query expression applied before aggregation, for "
    "example to drop flagged frames."
)

_THRESHOLD_DESCRIPTION = (
    "Aggregate the boolean column > threshold instead of each raw column. "
    "Unset, the raw columns are aggregated. With mean in aggs, the "
    "result is the fraction of the rows reduced together that exceed the "
    "threshold."
)

_TRANSFORM_DESCRIPTION = (
    "A transform applied to each column before aggregation. Known values "
    "are none and abs. abs aggregates the column's absolute value, useful "
    "for a magnitude such as angular velocity."
)

_PAIRS_DESCRIPTION = (
    "Which rows of a pair table sum and count read, when the input has a "
    "perspective column; the other aggregations read every row. once reads "
    "perspective 0, each pair once, and each_perspective reads every "
    "ordered pair row. auto is resolved by a recipe's plan: once when every "
    "column is declared symmetric over mirrored rows and filter_expr is "
    "unset, every row otherwise. A run outside a plan (mosaic run --feature) has "
    "no input chain to resolve and reads every row under auto, which never "
    "halves a value without a declaration. Set once there to count each "
    "pair once."
)

_PERSPECTIVE_COLUMN = "perspective"


class AggregateParams(Params):
    """The parameters of the reduction that every aggregate shares.

    Each of ``columns`` is reduced by each of ``aggs``. Before it,
    ``filter_expr`` drops rows, ``transform`` takes the absolute value and
    ``threshold`` turns each column into whether it exceeds a value.
    ``subgroup_col`` splits the rows reduced together.
    :class:`PairAggregateParams` adds the ``pairs`` rule for the aggregates
    whose groups hold both rows of a pair.
    """

    columns: Annotated[list[str], Declared(_COLUMNS_DESCRIPTION)] = Field(min_length=1)
    aggs: Annotated[list[Aggregation], Declared(_AGGS_DESCRIPTION)] = Field(
        default_factory=lambda: ["mean"], min_length=1
    )
    filter_expr: Annotated[str | None, Declared(_FILTER_EXPR_DESCRIPTION)] = None
    threshold: Annotated[float | None, Declared(_THRESHOLD_DESCRIPTION)] = None
    transform: Annotated[Literal["none", "abs"], Declared(_TRANSFORM_DESCRIPTION)] = (
        "none"
    )
    subgroup_col: Annotated[str | None, Declared(_SUBGROUP_COL_DESCRIPTION)] = None

    @field_validator("columns", "aggs", check_fields=False)
    @classmethod
    def _no_repeats(cls, value: list[str]) -> list[str]:
        """Refuse a name listed twice, which would name one result column twice."""
        repeated = sorted({item for item in value if value.count(item) > 1})
        if repeated:
            raise ValueError(f"lists {', '.join(repeated)} more than once")
        return value


class PairAggregateParams(AggregateParams):
    """The shared parameters, and the rows of a pair table a sum or count reads.

    ``frame-aggregate`` and ``sequence-aggregate`` group both rows of a pair
    together, so a ``sum`` or ``count`` over a column with one value on the two
    counts each pair twice. ``pairs`` says whether to read one of them.
    """

    pairs: Annotated[PairRule, Declared(_PAIRS_DESCRIPTION)] = "auto"

    def pair_read(self) -> PairRead:
        """The rows ``reduce_column`` reads: ``once``, or every ordered pair row.

        ``auto`` reads every row here, since only a recipe's plan resolves it to
        ``once``.
        """
        return "once" if self.pairs == "once" else "each_perspective"


def split_keys(df: pd.DataFrame, params: AggregateParams) -> list[str]:
    """The keys that split the rows a feature reduces together.

    ``arena`` when the input has the column, so the animals of different arenas
    are never pooled, then ``subgroup_col`` when it is set.
    """
    keys = [ARENA_COLUMN] if ARENA_COLUMN in df.columns else []
    if params.subgroup_col and params.subgroup_col not in keys:
        keys.append(params.subgroup_col)
    return keys


def _plain_values(values: pd.Series) -> np.ndarray:
    """The values of one column as an array a group-by aggregates.

    A pandas extension column of numbers or booleans (a nullable ``boolean`` or
    ``Int64``) becomes floats with NaN for a null, which every aggregation skips.
    As an object array of numbers and ``NA`` it would sum to a mix of types and
    raise in ``std``. Any other column is its own numpy array.
    """
    dtype = values.dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and (
        pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    ):
        return values.to_numpy(dtype=float, na_value=np.nan)
    return values.to_numpy()


def reduce_column(
    df: pd.DataFrame,
    params: AggregateParams,
    keys: Sequence[str],
    *,
    first: Sequence[str] = (),
    pairs: PairRead,
) -> pd.DataFrame:
    """Reduce each of ``params.columns`` of ``df`` to one row per distinct ``keys``.

    Applies ``filter_expr``, ``transform`` and ``threshold``, then groups by
    ``keys`` and aggregates each column with each of ``aggs``. The result holds
    ``keys``, then each column of ``first`` (its first value in the group), then
    ``<column>_<agg>`` for each column and, within it, each aggregation, sorted
    by ``keys``. A row whose key is null belongs to no group. The result is
    empty when no row survives the filter.

    A pair table holds each pair once per perspective. Under ``pairs="once"``
    and with a ``perspective`` column present, ``sum`` and ``count`` read the
    ``perspective == 0`` rows, so each pair counts once, while the other
    aggregations of the same call read every row. Under ``each_perspective``
    every aggregation reads every ordered pair row, as a caller that groups pair
    rows by a focal id needs.

    Args:
        df: The rows to reduce.
        params: The columns, aggregations and pre-aggregation steps.
        keys: The columns to group by.
        first: Columns that take their first value within each group.
        pairs: The rows of a pair table that ``sum`` and ``count`` read.

    Returns:
        The reduced table, empty when no row remains.
    """
    if params.filter_expr:
        df = df.query(params.filter_expr)
    if df.empty:
        return pd.DataFrame()

    work = pd.DataFrame({key: df[key].to_numpy() for key in (*keys, *first)})
    for index, column in enumerate(params.columns):
        values = df[column]
        if params.transform == "abs":
            values = values.abs()
        if params.threshold is not None:
            values = (values > params.threshold).astype(float)
        work[f"_v{index}"] = _plain_values(values)

    once = pairs == "once" and _PERSPECTIVE_COLUMN in df.columns
    halved = work[(df[_PERSPECTIVE_COLUMN] == 0).to_numpy()] if once else work
    group_keys = list(keys)
    grouped = work.groupby(group_keys, sort=True)
    reduced = (
        grouped[list(first)].first()
        if first
        else pd.DataFrame(index=grouped.size().index)
    )
    for index, column in enumerate(params.columns):
        for agg in params.aggs:
            counting = agg in PAIR_COUNTING_AGGS
            source = halved if once and counting else work
            result = source.groupby(group_keys, sort=True)[f"_v{index}"].agg(agg)
            # A group whose rows are all of the other perspective sums no row.
            reduced[f"{column}_{agg}"] = (
                result.reindex(reduced.index, fill_value=0)
                if counting
                else result.reindex(reduced.index)
            )
    if reduced.empty:
        return pd.DataFrame()
    return reduced.reset_index()


def blank_reduction(
    df: pd.DataFrame, params: AggregateParams, keys: Sequence[str]
) -> pd.DataFrame:
    """One row per distinct ``keys`` of ``df`` that reduces no values.

    For an entry whose ``filter_expr`` keeps no row. It has the columns that
    :func:`reduce_column` gives an entry that keeps rows: each of the ``sum``
    and ``count`` aggregations is 0, as for a group none of whose rows it reads,
    and every other aggregation is NaN. A row whose key is null belongs to no
    group.

    Args:
        df: The rows before the filter, which hold the keys.
        params: The columns and aggregations of the result.
        keys: The columns to group by.

    Returns:
        The blank rows, sorted by ``keys``.
    """
    groups = df[list(keys)].dropna().drop_duplicates().sort_values(list(keys))
    blank = groups.reset_index(drop=True)
    for column in params.columns:
        for agg in params.aggs:
            blank[f"{column}_{agg}"] = 0.0 if agg in PAIR_COUNTING_AGGS else np.nan
    return blank
