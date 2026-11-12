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

``keep_columns`` names columns of the input that the output carries beside the
reduction keys: a label such as a strain, which every row of a reduced group
shares. :func:`reduce_column` refuses a group whose rows disagree on one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal, Self, override

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.params import Declared, Params, has_fields
from mosaic.core.pipeline.column_contract import (
    PAIR_COUNTING_AGGS,
    PairRead,
    PairRule,
)
from mosaic.core.pipeline.types import COLUMNS as C
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
_KEEP_COLUMNS_DESCRIPTION = (
    "Columns of the input to write to the output, after the reduction keys, "
    "with one value per output row, such as a strain label. Each must take "
    "one value over the rows reduced together, null counting as a value, and "
    "the input must have it. A group whose rows disagree, or an input without "
    "the column, fails the entry. The names the reduction writes itself "
    "(group, sequence, frame, time, id, id1, id2, perspective and arena), a "
    "name in columns and subgroup_col are refused."
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
_FOCAL_COLUMNS = ("id1", "id2")
_MAX_LISTED_VALUES = 5


class AggregateParams(Params):
    """The parameters of the reduction that every aggregate shares.

    Each of ``columns`` is reduced by each of ``aggs``. Before it,
    ``filter_expr`` drops rows, ``transform`` takes the absolute value and
    ``threshold`` turns each column into whether it exceeds a value.
    ``subgroup_col`` splits the rows reduced together. ``keep_columns`` are
    written to the output beside the keys.
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
    keep_columns: Annotated[list[str], Declared(_KEEP_COLUMNS_DESCRIPTION)] = Field(
        default_factory=list
    )

    @field_validator("columns", "aggs", "keep_columns", check_fields=False)
    @classmethod
    def _no_repeats(cls, value: list[str]) -> list[str]:
        """Refuse a name listed twice, which would name one result column twice."""
        repeated = sorted({item for item in value if value.count(item) > 1})
        if repeated:
            raise ValueError(f"lists {', '.join(repeated)} more than once")
        return value

    @model_validator(mode="after")
    def _refuse_a_kept_name_the_output_has_already(self) -> Self:
        """Refuse a kept name that is aggregated, splits the rows or is a reduction key.

        A subclass without the three fields leaves the verdict to the full model.
        """
        if not has_fields(self, "keep_columns", "columns", "subgroup_col"):
            return self
        owned = _reduction_columns()
        for name in self.keep_columns:
            if name in self.columns:
                raise ValueError(f"keep_columns lists {name}, which is also in columns")
            if name == self.subgroup_col:
                raise ValueError(f"keep_columns lists {name}, which is subgroup_col")
            if name in owned:
                raise ValueError(
                    f"keep_columns lists {name}, which the reduction writes itself"
                )
        return self

    @override
    def identity_dump(self) -> dict[str, object]:
        """The hashed params, less ``keep_columns`` when it is empty.

        A run that keeps no column writes the output it wrote before the field
        existed, so leaving the field out keeps every such run's identifier. A
        kept column adds an output column and enters the hash as any field does.
        """
        dumped = super().identity_dump()
        if not self.keep_columns:
            del dumped["keep_columns"]
        return dumped


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


def _reduction_columns() -> frozenset[str]:
    """The names a reduction writes itself, which a kept column cannot take."""
    return frozenset(
        {*C.meta_set(), *_FOCAL_COLUMNS, _PERSPECTIVE_COLUMN, ARENA_COLUMN}
    )


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


def _is_null(value: object) -> bool:
    """Whether a cell is a null: ``None``, ``NaN``, ``NA`` or ``NaT``."""
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or value is pd.NA or value is pd.NaT


def _first_varying_group(
    df: pd.DataFrame, keys: Sequence[str], column: str
) -> dict[str, object] | None:
    """The keys of the first group of *keys* whose rows take two values of *column*.

    Null counts as a value. A row whose key is null belongs to no group, as in
    the reduction. ``None`` when every group holds one value.
    """
    counts = df.groupby(list(keys), sort=True)[column].nunique(dropna=False)
    varying = counts[counts > 1]
    if varying.empty:
        return None
    first = varying.index.to_frame(index=False).astype(object).iloc[0]
    group: dict[str, object] = {}
    for key in keys:
        part: object = first[key]
        group[key] = part
    return group


def _refuse_a_varying_column(
    df: pd.DataFrame, keys: Sequence[str], columns: Sequence[str]
) -> None:
    """Refuse a column of *columns* that takes two values in one group of *keys*.

    Raises:
        ValueError: A group's rows hold more than one value of a column. The
            message names the column, the group's keys and up to five of its
            values.
    """
    for column in columns:
        group = _first_varying_group(df, keys, column)
        if group is None:
            continue
        in_group = np.logical_and.reduce(
            [df[key].to_numpy() == part for key, part in group.items()]
        )
        texts = sorted(
            {
                "null" if _is_null(value) else repr(value)
                for value in df[column][in_group].tolist()
            }
        )
        listed = ", ".join(texts[:_MAX_LISTED_VALUES])
        more = ", ..." if len(texts) > _MAX_LISTED_VALUES else ""
        where = ", ".join(f"{key}={part!r}" for key, part in group.items())
        raise ValueError(
            f"keep_columns column {column!r} takes {len(texts)} values within "
            f"{where}: {listed}{more}"
        )


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
    each of ``params.keep_columns`` (its one value in the group), then
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

    Raises:
        ValueError: A kept column is not in ``df``, or takes more than one value
            in a group of the rows left after ``filter_expr``.
    """
    kept = tuple(params.keep_columns)
    for column in kept:
        if column not in df.columns:
            raise ValueError(f"keep_columns column {column!r} is not in the input")
    if params.filter_expr:
        df = df.query(params.filter_expr)
    if df.empty:
        return pd.DataFrame()
    _refuse_a_varying_column(df, keys, kept)

    carried = (*first, *kept)
    work = pd.DataFrame({key: df[key].to_numpy() for key in (*keys, *carried)})
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
        grouped[list(carried)].first()
        if carried
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
