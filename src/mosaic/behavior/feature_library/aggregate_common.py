"""
Shared reduction of the aggregate features.

``frame-aggregate``, ``sequence-aggregate`` and ``individual-aggregate`` reduce
one column of a feature output or of raw tracks with the same steps: filter,
transform, threshold, then a group-by. :func:`reduce_column` holds those steps
and :class:`AggregateParams` the parameters they read, so the three features
differ only in the keys they group by and in whether they take a ``pairs``
rule (:class:`PairAggregateParams`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import pandas as pd

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    PAIR_COUNTING_AGGS,
    PairRead,
    PairRule,
)

_COLUMN_DESCRIPTION = "The name of the column to aggregate."

_AGG_DESCRIPTION = (
    "The aggregation applied across the rows that the feature reduces "
    "together: those of one frame, one entry or one id, as the feature "
    "groups them. Known values "
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
    "example to drop flagged frames."
)

_THRESHOLD_DESCRIPTION = (
    "Aggregate the boolean column > threshold instead of the raw column. "
    "Unset, the raw column is aggregated. With agg set to mean, the "
    "result is the fraction of the rows reduced together that exceed the "
    "threshold."
)

_TRANSFORM_DESCRIPTION = (
    "A transform applied to the column before aggregation. Known values "
    "are none and abs. abs aggregates the column's absolute value, useful "
    "for a magnitude such as angular velocity."
)

_PAIRS_DESCRIPTION = (
    "Which rows of a pair table sum and count read, when the input has a "
    "perspective column; the other aggregations read every row. once reads "
    "perspective 0, each pair once, and each_perspective reads every "
    "ordered pair row. auto is resolved by a recipe's plan: once for a "
    "column declared symmetric over mirrored rows with no filter_expr, "
    "every row otherwise. A run outside a plan (mosaic run --feature) has "
    "no input chain to resolve and reads every row under auto, which never "
    "halves a value without a declaration. Set once there to count each "
    "pair once."
)

_PERSPECTIVE_COLUMN = "perspective"


class AggregateParams(Params):
    """The parameters of the reduction that every aggregate shares.

    ``column`` is the column reduced and ``agg`` the aggregation. Before it,
    ``filter_expr`` drops rows, ``transform`` takes the absolute value and
    ``threshold`` turns the column into whether it exceeds a value.
    ``output_column`` names the result. :class:`PairAggregateParams` adds the
    ``pairs`` rule for the aggregates whose groups hold both rows of a pair.
    """

    column: Annotated[str, Declared(_COLUMN_DESCRIPTION)]
    agg: Annotated[
        Literal["mean", "median", "min", "max", "std", "sum", "count"],
        Declared(_AGG_DESCRIPTION),
    ] = "mean"
    output_column: Annotated[str | None, Declared(_OUTPUT_COLUMN_DESCRIPTION)] = None
    filter_expr: Annotated[str | None, Declared(_FILTER_EXPR_DESCRIPTION)] = None
    threshold: Annotated[float | None, Declared(_THRESHOLD_DESCRIPTION)] = None
    transform: Annotated[Literal["none", "abs"], Declared(_TRANSFORM_DESCRIPTION)] = (
        "none"
    )


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


def output_column_name(params: AggregateParams) -> str:
    """The result column: ``output_column``, else ``column`` and ``agg`` joined."""
    return params.output_column or f"{params.column}_{params.agg}"


def reduce_column(
    df: pd.DataFrame,
    params: AggregateParams,
    keys: Sequence[str],
    *,
    first: Sequence[str] = (),
    pairs: PairRead,
) -> pd.DataFrame:
    """Reduce ``params.column`` of ``df`` to one row per distinct ``keys``.

    Applies ``filter_expr``, ``transform`` and ``threshold``, then groups by
    ``keys`` and aggregates with ``agg``. The result holds ``keys``, then each
    column of ``first`` (its first value in the group), then the output column,
    sorted by ``keys``. It is empty when no row survives the filter.

    A pair table holds each pair once per perspective. Under ``pairs="once"``
    and with a ``perspective`` column present, ``sum`` and ``count`` keep the
    ``perspective == 0`` rows, so each pair counts once. Under
    ``each_perspective`` they read every ordered pair row, as a caller that
    groups pair rows by a focal id needs. The other aggregations read every
    row under either.

    Args:
        df: The rows to reduce.
        params: The column, aggregation and pre-aggregation steps.
        keys: The columns to group by.
        first: Columns that take their first value within each group.
        pairs: The rows of a pair table that ``sum`` and ``count`` read.

    Returns:
        The reduced table, empty when no row remains.
    """
    if params.filter_expr:
        df = df.query(params.filter_expr)
    if (
        pairs == "once"
        and params.agg in PAIR_COUNTING_AGGS
        and _PERSPECTIVE_COLUMN in df.columns
    ):
        df = df[df[_PERSPECTIVE_COLUMN] == 0]
    if df.empty:
        return pd.DataFrame()

    values = df[params.column]
    if params.transform == "abs":
        values = values.abs()
    if params.threshold is not None:
        values = (values > params.threshold).astype(float)

    work = pd.DataFrame({key: df[key].to_numpy() for key in (*keys, *first)})
    work["_v"] = values.to_numpy()

    out_col = output_column_name(params)
    named = {name: pd.NamedAgg(column=name, aggfunc="first") for name in first}
    named[out_col] = pd.NamedAgg(column="_v", aggfunc=params.agg)
    return work.groupby(list(keys), sort=True).agg(**named).reset_index()
