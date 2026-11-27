"""
Shared reduction of the aggregate features.

``frame-aggregate``, ``sequence-aggregate`` and ``individual-aggregate`` reduce
columns of a feature output or of raw tracks with the same steps: filter,
transform, threshold, then a group-by. :func:`reduce_column` holds those steps
and :class:`AggregateParams` the parameters they read, so the three features
differ only in the keys they group by and in whether they take a ``pairs``
rule (:class:`PairAggregateParams`). Each feature also groups by the input's
``arena`` column when it has one, and by each column of ``subgroup_col`` when it
is set (:func:`split_keys`).

``keep_columns`` names columns of the input that the output carries beside the
reduction keys: a label such as a strain, which every row of a reduced group
shares. :func:`reduce_column` refuses a group whose rows disagree on one, and
:func:`blank_reduction` reads it from an emptied entry's rows before the filter.

``duration`` weighs each row by the seconds it stands for, read from the input's
``frame`` and ``time`` columns (:func:`~mosaic.core.timebase.row_seconds`), so
the sum is a time and not a count of rows at a frame rate that varies. ``share``
weighs the rows the same way: it writes, for each value of a text, categorical,
boolean or integer column, the fraction of the group's seconds that the column
holds it.

``bin_s`` splits the reduction by time: each row goes to the bin
``floor(time / bin_s)``, anchored at time 0 of the recording's axis, and
``bin_start`` and ``bin_end`` join the reduction keys (:func:`bin_columns`). A
row's seconds, for ``duration`` and ``share``, divide among the bins it spans.

``first_pass`` is the smallest value of ``order_col`` among the rows whose value
passes the ``threshold`` comparison, NaN when none does. It reads every row
under any ``pairs`` rule and skips a row whose ``order_col`` is null.

``subsets`` names filters that each reduce the rows they keep, after
``filter_expr``, and ``subset`` joins the reduction keys after the caller's
keys (:data:`SUBSET`). A subset that keeps no row has no row in
:func:`reduce_column`, and :func:`blank_rows` writes the blank row that
``sequence-aggregate`` gives it.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal, Self, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from mosaic.core.params import Declared, Params, has_fields
from mosaic.core.pipeline.column_contract import (
    PAIR_COUNTING_AGGS,
    PairRead,
    PairRule,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.schema import ARENA_COLUMN
from mosaic.core.timebase import row_seconds

type Aggregation = Literal[
    "mean",
    "median",
    "min",
    "max",
    "std",
    "sum",
    "count",
    "duration",
    "share",
    "first_pass",
]

type ThresholdOp = Literal[">", ">=", "<", "<="]


@dataclass(frozen=True)
class AggregationRule:
    """How the groups of an aggregation combine across entries.

    Attributes:
        combines: ``"partials"`` when a pool gets each group's value from
            per-key partials of its entries, ``"values"`` when it needs every
            value of the group.
    """

    combines: Literal["partials", "values"]


AGGREGATIONS: Mapping[Aggregation, AggregationRule] = MappingProxyType(
    {
        "mean": AggregationRule("partials"),
        "median": AggregationRule("values"),
        "min": AggregationRule("partials"),
        "max": AggregationRule("partials"),
        "std": AggregationRule("partials"),
        "sum": AggregationRule("partials"),
        "count": AggregationRule("partials"),
        "duration": AggregationRule("partials"),
        "share": AggregationRule("partials"),
        "first_pass": AggregationRule("partials"),
    }
)
"""Every member of :data:`Aggregation` and how its groups combine across entries."""

_COLUMNS_DESCRIPTION = (
    "The names of the columns to aggregate. Each column is reduced by each "
    "aggregation in aggs, and the result column is named column followed by "
    "agg, joined with an underscore. The share aggregation writes one column "
    "per value instead, named `<column>_share_<value>`."
)

_AGGS_DESCRIPTION = (
    "The aggregations applied across the rows that the feature reduces "
    "together: those of one frame, one entry or one id, as the feature "
    "groups them. Known values "
    "are mean, median, min, max, std, sum, count, duration, share and "
    "first_pass. A null "
    "value (NaN, None or NA) is skipped for every mode, and count counts "
    "non-null values. With threshold, a null value stays null: it passes no "
    "comparison, so mean skips it, count does not count it and duration does "
    "not count its seconds. A value that should count, such as a latency capped at "
    "its window or a zero, is filled in the feature that produces it, as "
    "event-response does with recovery_time_s. "
    "duration is the seconds the rows that count last: each row lasts until the "
    "next frame of its individual, read from the time column, and a row counts "
    "when its value is not null, or, with threshold, when it passes the "
    "comparison threshold_op, so a boolean column with threshold 0.5 gives the "
    "time it holds. share reads a column of text, categories, booleans or "
    "integers and writes a column `<column>_share_<value>` for each value, true "
    "or false for a boolean and the decimal integer for a number: the seconds "
    "of the rows with that value over the seconds of the rows whose value is "
    "not null, so a group's shares sum to 1. A group whose rows are all null gets NaN, and "
    "a value the group lacks gets 0. A value must consist of letters, digits, "
    "underscores and hyphens. share is refused with threshold and with "
    "transform abs. duration and share need a time column, and "
    "frame-aggregate takes neither. first_pass is the smallest order_col "
    "value among the rows whose value passes the comparison threshold_op "
    "against threshold, after transform, or NaN when no row does: a null "
    "value passes nothing, infinity passes a comparison it satisfies, and a "
    "row with a null order_col value is skipped. It reads every row under any "
    "pairs rule, and is refused without threshold."
)

_CATEGORIES_DESCRIPTION = (
    "The values the share aggregation writes a column for, as each value's "
    "text: a string as it is, true or false for a boolean, an integer in "
    "decimal. Each gives the column `<column>_share_<value>`, in the listed order, "
    "and an entry where one does not occur gets 0 for it. A value outside the "
    "list fails the entry. Unset, share writes one column per value in the "
    "entry, in sorted order, and entries may differ in columns. Set it to read "
    "the run as one table. Refused without share in aggs."
)

_SUBGROUP_COL_DESCRIPTION = (
    "A column of the input, or a list of them, that splits the rows reduced "
    "together into subgroups, one result row per value, or per combination "
    "of values for a list, such as a tag column or the event of ffgroups. "
    "The output carries each column in the order listed, after arena. Rows "
    "whose value is null in any of them belong to no subgroup and are "
    "dropped. A list names a column once, and a list of one column is that "
    "column. Unset, the rows are not split."
)
_KEEP_COLUMNS_DESCRIPTION = (
    "Columns of the input to write to the output, after the reduction keys, "
    "with one value per output row, such as a strain label. Each must take "
    "one value over the rows reduced together, null counting as a value, and "
    "the input must have it. A group whose rows disagree, or an input without "
    "the column, fails the entry. The names the reduction writes itself "
    "(group, sequence, frame, time, id, id1, id2, perspective and arena), a "
    "name in columns and any subgroup_col are refused, and with bin_s set, so "
    "are bin_start and bin_end, and with subsets set, so is subset."
)

_BIN_S_DESCRIPTION = (
    "The length in seconds of the time bins. Unset, an entry is reduced "
    "whole. Set, each row goes to the bin floor(time / bin_s), anchored at "
    "time 0 of the recording's axis, so the bins of two entries or two "
    "divisions of one recording line up. A time counts as on a bin edge "
    "when its distance from it, in bins, is at most a billionth of the "
    "edge's index, or a billionth of a bin if that is more. The output "
    "gains bin_start and bin_end, in seconds, after the keys, and has one "
    "row per reduced group and bin. A bin that no row starts in is not "
    "written. duration and share weigh a row by the seconds of the interval "
    "from its time to the next frame of its individual: a row that crosses "
    "a bin edge gives each bin the seconds it overlaps, and seconds that "
    "fall in a bin where the group has no row starting are dropped. The "
    "other aggregations read each row in the bin it starts in. A row "
    "without a time is in no bin. Needs a time column, so give the step an "
    "input with one, such as tracks. Refused by frame-aggregate, whose rows "
    "are one frame each. keep_columns and subgroup_col may not be bin_start "
    "or bin_end."
)

_ORDER_COL_DESCRIPTION = (
    "The column that first_pass orders the rows of a group by: its value is "
    "the smallest value of this column among the rows that pass the "
    "threshold, such as the time of the first row above it. The column must "
    "be numeric, and a row where it is null is skipped. Refused unless "
    "first_pass is in aggs, except for the default time, and first_pass "
    "needs a threshold."
)

_FILTER_EXPR_DESCRIPTION = (
    "A pandas DataFrame.query expression applied before aggregation, for "
    "example to drop flagged frames."
)

_THRESHOLD_DESCRIPTION = (
    "Aggregate whether each value passes the comparison threshold_op "
    "instead of each raw column. Unset, the raw columns are aggregated. With "
    "mean in aggs, the result is the fraction of the rows reduced together "
    "that pass. A null value passes no comparison and stays null, so it is "
    "left out of that fraction."
)

_THRESHOLD_OP_DESCRIPTION = (
    "The comparison of each value with threshold: > (greater than), >= "
    "(at least), < (less than) or <= (at most). Refused unless threshold is "
    "set, except for the default >."
)

_TRANSFORM_DESCRIPTION = (
    "A transform applied to each column before aggregation. Known values "
    "are none and abs. abs aggregates the column's absolute value, useful "
    "for a magnitude such as angular velocity."
)

_PAIRS_DESCRIPTION = (
    "Which rows of a pair table sum, count and duration read, when the input has a "
    "perspective column; the other aggregations read every row. once reads "
    "perspective 0, each pair once, and each_perspective reads every "
    "ordered pair row. auto is resolved by a recipe's plan: once when every "
    "column is declared symmetric over mirrored rows and neither filter_expr "
    "nor subsets is set, every row otherwise. A run outside a plan "
    "(mosaic run --feature) has no input chain to resolve and reads every row "
    "under auto, which never halves a value without a declaration. Set once "
    "there to count each pair once."
)

_SUBSETS_DESCRIPTION = (
    "Named filters, each in the syntax of filter_expr and applied after it, "
    "that each reduce the rows they keep. The output has a row for each group "
    "and subset, and a text column subset, after the subgroup columns, holds "
    "the subset's name. Subsets may overlap, so a row that two keep counts in "
    "both. A name is made of letters, digits, underscores and hyphens, and the "
    "order of the mapping does not matter. duration and share weigh a row by "
    "the seconds computed before filter_expr and the subsets. A subset that "
    "keeps no row of an entry is written as a blank row by sequence-aggregate, "
    "so that every entry has a row for each subset, and has no row in "
    "frame-aggregate and individual-aggregate. subset may not be a kept or "
    "subgroup column, and an input with a subset column is refused. Unset, the "
    "rows are not split."
)

_ORDER_COLUMN = "_order"

_TIME_SENTENCE = (
    "duration and share weigh each row by its seconds, read from the time column: "
    "give the step an input with a time column, such as tracks."
)

_BIN_TIME_SENTENCE = "give the step an input with a time column, such as tracks."

BIN_START = "bin_start"
BIN_END = "bin_end"
SUBSET = "subset"

_SAFE_VALUE = re.compile(r"[A-Za-z0-9_-]+")
_PERSPECTIVE_COLUMN = "perspective"
_FOCAL_COLUMNS = ("id1", "id2")
_MAX_LISTED_VALUES = 5


class AggregateParams(Params):
    """The parameters of the reduction that every aggregate shares.

    Each of ``columns`` is reduced by each of ``aggs``. Before it,
    ``filter_expr`` drops rows, ``transform`` takes the absolute value and
    ``threshold`` turns each column into whether it passes the comparison
    ``threshold_op`` against a value, with a null staying null.
    ``subgroup_col``, one column or a list of them, splits the rows reduced
    together, and ``bin_s`` splits them again by time bin. ``keep_columns`` are
    written to the output beside the keys. ``categories`` fixes the columns
    that ``share`` writes. ``first_pass`` gives the smallest ``order_col`` value
    among the rows that pass ``threshold``.
    :class:`PairAggregateParams` adds the ``pairs`` rule for the aggregates
    whose groups hold both rows of a pair.
    """

    columns: Annotated[list[str], Declared(_COLUMNS_DESCRIPTION)] = Field(min_length=1)
    aggs: Annotated[list[Aggregation], Declared(_AGGS_DESCRIPTION)] = Field(
        default_factory=lambda: ["mean"], min_length=1
    )
    filter_expr: Annotated[str | None, Declared(_FILTER_EXPR_DESCRIPTION)] = None
    threshold: Annotated[float | None, Declared(_THRESHOLD_DESCRIPTION)] = None
    threshold_op: Annotated[ThresholdOp, Declared(_THRESHOLD_OP_DESCRIPTION)] = ">"
    transform: Annotated[Literal["none", "abs"], Declared(_TRANSFORM_DESCRIPTION)] = (
        "none"
    )
    subgroup_col: Annotated[
        str | list[str] | None, Declared(_SUBGROUP_COL_DESCRIPTION)
    ] = None
    bin_s: Annotated[
        float | None, Declared(_BIN_S_DESCRIPTION, unit="s"), Field(gt=0)
    ] = None
    keep_columns: Annotated[list[str], Declared(_KEEP_COLUMNS_DESCRIPTION)] = Field(
        default_factory=list
    )
    categories: Annotated[list[str] | None, Declared(_CATEGORIES_DESCRIPTION)] = Field(
        default=None, min_length=1
    )
    order_col: Annotated[str, Declared(_ORDER_COL_DESCRIPTION)] = C.time_col
    subsets: Annotated[dict[str, str] | None, Declared(_SUBSETS_DESCRIPTION)] = Field(
        default=None, min_length=1
    )

    @field_validator(
        "columns", "aggs", "keep_columns", "categories", check_fields=False
    )
    @classmethod
    def _no_repeats(cls, value: list[str] | None) -> list[str] | None:
        """Refuse a name listed twice, which would name one result column twice.

        An unset ``categories`` passes.
        """
        if value is None:
            return value
        return _refuse_repeats(value)

    @field_validator("subgroup_col", check_fields=False)
    @classmethod
    def _one_column_or_a_list_of_distinct_columns(
        cls, value: str | list[str] | None
    ) -> str | list[str] | None:
        """Refuse an empty list or a repeated name, and read one item as a string."""
        if not isinstance(value, list):
            return value
        if not value:
            raise ValueError("subgroup_col lists no column")
        value = _refuse_repeats(value)
        return value[0] if len(value) == 1 else value

    @field_validator("subsets", check_fields=False)
    @classmethod
    def _safe_names_and_filters(
        cls, value: dict[str, str] | None
    ) -> dict[str, str] | None:
        """Refuse a subset name of unsafe characters, or a blank filter."""
        for name, expression in (value or {}).items():
            if not _SAFE_VALUE.fullmatch(name):
                raise ValueError(
                    f"subsets lists {name!r}, which is not made of letters, "
                    "digits, underscores and hyphens"
                )
            if not expression.strip():
                raise ValueError(f"subsets lists no filter for {name!r}")
        return value

    def subgroup_columns(self) -> list[str]:
        """The subgroup columns in output order: none, one or several."""
        if not self.subgroup_col:
            return []
        if isinstance(self.subgroup_col, str):
            return [self.subgroup_col]
        return list(self.subgroup_col)

    @field_validator("categories", check_fields=False)
    @classmethod
    def _safe_categories(cls, value: list[str] | None) -> list[str] | None:
        """Refuse a category that is not a safe part of a column name."""
        for item in value or ():
            if not _SAFE_VALUE.fullmatch(item):
                raise ValueError(
                    f"categories lists {item!r}, which is not made of letters, "
                    "digits, underscores and hyphens"
                )
        return value

    @model_validator(mode="after")
    def _refuse_what_share_cannot_read(self) -> Self:
        """Refuse ``share`` with a step that changes values, and ``categories`` alone.

        A share counts the values of the column as they are, and ``threshold``
        and ``transform: "abs"`` turn them into others. ``categories`` names the
        columns of ``share``, so it needs ``share`` in ``aggs``.
        """
        if not has_fields(self, "aggs", "threshold", "transform", "categories"):
            return self
        shares = "share" in self.aggs
        if shares and self.threshold is not None:
            raise ValueError(
                "share counts values as they are and cannot be combined with threshold"
            )
        if shares and self.transform == "abs":
            raise ValueError(
                "share counts values as they are and cannot be combined with "
                "transform 'abs'"
            )
        if self.categories is not None and not shares:
            raise ValueError("categories is set but aggs has no share")
        return self

    @model_validator(mode="after")
    def _refuse_a_threshold_comparison_without_a_threshold(self) -> Self:
        """Refuse a ``threshold_op`` other than ``>`` when ``threshold`` is unset."""
        if not has_fields(self, "threshold", "threshold_op"):
            return self
        if self.threshold is None and self.threshold_op != ">":
            raise ValueError(
                f"threshold_op is {self.threshold_op!r} but threshold is not set"
            )
        return self

    @model_validator(mode="after")
    def _refuse_a_first_pass_without_what_it_reads(self) -> Self:
        """Refuse ``first_pass`` without ``threshold``, and ``order_col`` without it.

        ``first_pass`` is the order value of the first row that passes the
        comparison, so it needs ``threshold``. ``order_col`` names the column
        that orders those rows, so a value other than the default needs
        ``first_pass`` in ``aggs``.
        """
        if not has_fields(self, "aggs", "threshold", "order_col"):
            return self
        passes = "first_pass" in self.aggs
        if passes and self.threshold is None:
            raise ValueError("first_pass needs a threshold, which is not set")
        if self.order_col != C.time_col and not passes:
            raise ValueError(
                f"order_col is {self.order_col!r} but aggs has no first_pass"
            )
        return self

    @model_validator(mode="after")
    def _refuse_a_kept_name_the_output_has_already(self) -> Self:
        """Refuse a kept name that is aggregated, splits the rows or is a reduction key.

        With ``bin_s`` set, ``bin_start`` and ``bin_end`` are reduction keys and
        neither a kept column nor a subgroup column may take them, and with
        ``subsets`` set ``subset`` is one as well. A subclass without the five
        fields leaves the verdict to the full model.
        """
        if not has_fields(
            self, "keep_columns", "columns", "subgroup_col", "bin_s", "subsets"
        ):
            return self
        owned = _reduction_columns()
        subgroups = self.subgroup_columns()
        written_by: dict[str, str] = {}
        if self.bin_s is not None:
            written_by.update({BIN_START: "bin_s", BIN_END: "bin_s"})
        if self.subsets is not None:
            written_by[SUBSET] = "subsets"
        owned = owned | set(written_by)
        for name in subgroups:
            if name in written_by:
                raise ValueError(
                    f"subgroup_col is {name}, which {written_by[name]} writes itself"
                )
        for name in self.keep_columns:
            if name in self.columns:
                raise ValueError(f"keep_columns lists {name}, which is also in columns")
            if name in subgroups:
                raise ValueError(f"keep_columns lists {name}, which is subgroup_col")
            if name in owned:
                raise ValueError(
                    f"keep_columns lists {name}, which the reduction writes itself"
                )
        return self

    @override
    def identity_dump(self) -> dict[str, object]:
        """The hashed params, less each field that leaves the output as it was.

        ``keep_columns`` empty, ``categories`` unset, ``bin_s`` unset,
        ``subsets`` unset, ``threshold_op`` ``>`` and ``order_col`` ``time``
        write the output a run wrote before the fields existed, so leaving them
        out keeps every such run's identifier. A kept column or a category adds
        output columns and enters the hash as any field does.

        A set ``threshold`` adds ``threshold_missing``: a null value stays null
        under the comparison, where a NaN of a float column once counted as
        below it. A thresholded run's output differs on an input with missing
        values, so its identifier moves, and a run without a threshold keeps
        its own.
        """
        dumped = super().identity_dump()
        if not self.keep_columns:
            del dumped["keep_columns"]
        if self.categories is None:
            del dumped["categories"]
        if self.bin_s is None:
            del dumped["bin_s"]
        if self.threshold_op == ">":
            del dumped["threshold_op"]
        if self.order_col == C.time_col:
            del dumped["order_col"]
        if self.subsets is None:
            del dumped["subsets"]
        if self.threshold is not None:
            dumped["threshold_missing"] = "skip"
        return dumped


class PairAggregateParams(AggregateParams):
    """The shared parameters, and the pair-table rows that sum, count and duration read.

    ``frame-aggregate`` and ``sequence-aggregate`` group both rows of a pair
    together, so a ``sum``, ``count`` or ``duration`` over a column with one
    value on the two counts each pair twice. ``pairs`` says whether to read one
    of them. ``share`` and the other aggregations read every row.
    """

    pairs: Annotated[PairRule, Declared(_PAIRS_DESCRIPTION)] = "auto"

    def pair_read(self) -> PairRead:
        """The rows ``reduce_column`` reads: ``once``, or every ordered pair row.

        ``auto`` reads every row here, since only a recipe's plan resolves it to
        ``once``.
        """
        return "once" if self.pairs == "once" else "each_perspective"


def _refuse_repeats(value: list[str]) -> list[str]:
    """Refuse a name listed twice, which would name one result column twice."""
    repeated = sorted({item for item in value if value.count(item) > 1})
    if repeated:
        raise ValueError(f"lists {', '.join(repeated)} more than once")
    return value


def _reduction_columns() -> frozenset[str]:
    """The names a reduction writes itself, which a kept column cannot take."""
    return frozenset(
        {*C.meta_set(), *_FOCAL_COLUMNS, _PERSPECTIVE_COLUMN, ARENA_COLUMN}
    )


def split_keys(df: pd.DataFrame, params: AggregateParams) -> list[str]:
    """The keys that split the rows a feature reduces together.

    ``arena`` when the input has the column, so the animals of different arenas
    are never pooled, then each subgroup column in the order listed.
    """
    keys = [ARENA_COLUMN] if ARENA_COLUMN in df.columns else []
    keys.extend(name for name in params.subgroup_columns() if name not in keys)
    return keys


_COMPARISONS: dict[ThresholdOp, Callable[[pd.Series, float], pd.Series]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def _passes_threshold(
    values: pd.Series, threshold: float, op: ThresholdOp
) -> pd.Series:
    """Whether each value passes ``op`` against *threshold*, as 0.0 or 1.0.

    A missing value (NaN, ``None`` or ``NA``) stays NaN whatever the dtype of
    the column, so ``mean`` skips it, ``count`` does not count it and ``duration``
    does not count its seconds.
    """
    compared = _COMPARISONS[op](values, threshold)
    return compared.astype(float).where(values.notna())


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


def _series_keys(df: pd.DataFrame) -> list[str]:
    """The columns that split *df* into series of rows with one frame axis each.

    ``arena`` when the input has it, then ``id`` when it has that, and otherwise
    each of ``id1``, ``id2`` and ``perspective`` it has. A table with none is one
    series. An aggregate reads one entry, so ``group`` and ``sequence`` are constant.
    """
    keys = [ARENA_COLUMN] if ARENA_COLUMN in df.columns else []
    if C.id_col in df.columns:
        return [*keys, C.id_col]
    pair = (*_FOCAL_COLUMNS, _PERSPECTIVE_COLUMN)
    return [*keys, *(column for column in pair if column in df.columns)]


def _seconds_by_row(df: pd.DataFrame, who: str) -> NDArray[np.float64]:
    """The seconds each row of *df* stands for, by position.

    Raises:
        ValueError: The table's time axis cannot be measured. The message is the
            one :func:`~mosaic.core.timebase.row_seconds` raises, followed by what
            to give the step.
    """
    try:
        return row_seconds(df, _series_keys(df), who)
    except ValueError as error:
        raise ValueError(f"{error} {_TIME_SENTENCE}") from error


@dataclass(frozen=True, eq=False)
class _ShareTexts:
    """What ``share`` reads of one column.

    Attributes:
        texts: The text of each row's value, by position, ``None`` for a null.
        readable: Whether each row belongs to a group, by position.
        numeric: Whether the values are integers, which sort by number and not
            by their text.
    """

    texts: NDArray[np.object_]
    readable: NDArray[np.bool_]
    numeric: bool


def _texts_of_values(
    values: pd.Series, column: str
) -> tuple[NDArray[np.object_], bool]:
    """The text of each value ``share`` accepts, and whether the values are numbers.

    A string is its own text, a boolean is ``true`` or ``false``, and an integer
    its decimal form. A categorical column reads the text of its categories.

    Args:
        values: The column.
        column: Its name, for the message of a column ``share`` refuses.

    Returns:
        The texts by position, ``None`` for a null, and whether the values are
        integers, which sort by number and not by their text.

    Raises:
        ValueError: The column is of an unaccepted dtype, or an ``object`` column
            whose values are not all strings.
    """
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        names, numeric = _texts_of_values(pd.Series(dtype.categories), column)
        lookup = np.empty(names.size + 1, dtype=object)
        lookup[:-1] = names
        lookup[-1] = None
        return lookup[values.cat.codes.to_numpy()], numeric
    numeric = False
    if pd.api.types.is_bool_dtype(dtype):
        texts = values.astype("string").str.lower()
    elif pd.api.types.is_integer_dtype(dtype):
        texts = values.astype("string")
        numeric = True
    elif isinstance(dtype, pd.StringDtype):
        texts = values
    elif pd.api.types.is_object_dtype(dtype):
        if pd.api.types.infer_dtype(values, skipna=True) not in {"string", "empty"}:
            kinds = sorted(
                {type(value).__name__ for value in values if not _is_null(value)}
            )
            raise ValueError(
                f"share reads text, categorical, boolean or integer columns, and "
                f"column {column!r} is an object column of {', '.join(kinds)}"
            )
        texts = values
    else:
        raise ValueError(
            f"share reads text, categorical, boolean or integer columns, and "
            f"column {column!r} has dtype {dtype}"
        )
    result = texts.to_numpy(dtype=object, copy=True)
    result[texts.isna().to_numpy()] = None
    return result, numeric


def _share_texts(
    values: pd.Series, column: str, readable: NDArray[np.bool_]
) -> _ShareTexts:
    """The texts of one column that ``share`` reads.

    Raises:
        ValueError: The column is of an unaccepted kind.
    """
    texts, numeric = _texts_of_values(values, column)
    return _ShareTexts(texts=texts, readable=readable, numeric=numeric)


def _share_names(
    share: _ShareTexts, column: str, categories: Sequence[str] | None
) -> list[str]:
    """The values that ``share`` writes a column for, from one column's texts.

    The values are those of the rows that belong to a group: ``categories`` in
    their order when set, and otherwise each value that occurs, integers by
    number and other values by code point.

    Raises:
        ValueError: A value is not made of letters, digits, underscores and
            hyphens, or is not among ``categories``.
    """
    texts = share.texts
    read = {str(text) for text in texts[share.readable & ~pd.isna(texts)]}
    unsafe = sorted(text for text in read if not _SAFE_VALUE.fullmatch(text))
    if unsafe:
        raise ValueError(
            f"share cannot name a column for the value {unsafe[0]!r} of column "
            f"{column!r}: recode it upstream to letters, digits, underscores "
            "and hyphens"
        )
    if categories is None:
        return sorted(read, key=int) if share.numeric else sorted(read)
    outside = sorted(read.difference(categories))
    if outside:
        raise ValueError(
            f"column {column!r} has the value {outside[0]!r}, which is not "
            f"among the categories {list(categories)}"
        )
    return list(categories)


def _refuse_a_repeated_output_name(
    params: AggregateParams,
    keys: Sequence[str],
    carried: Sequence[str],
    share_names: Mapping[str, Sequence[str]],
) -> None:
    """Refuse an output column name that two of the entry's outputs would take.

    Raises:
        ValueError: Two sources write one name. The message names the name and
            both sources.
    """
    sources: dict[str, str] = {}

    def claim(name: str, source: str) -> None:
        if name in sources:
            raise ValueError(
                f"the column {name!r} would be written by both {sources[name]} "
                f"and {source}"
            )
        sources[name] = source

    for name in (*keys, *carried):
        claim(name, f"the column {name!r} of the input")
    for column in params.columns:
        for agg in params.aggs:
            if agg == "share":
                for value in share_names[column]:
                    claim(
                        f"{column}_share_{value}",
                        f"share of column {column!r} for the value {value!r}",
                    )
            else:
                claim(f"{column}_{agg}", f"{agg} of column {column!r}")


def _share_fractions(
    work: pd.DataFrame,
    keys: Sequence[str],
    share: _ShareTexts,
    names: Sequence[str],
    seconds: NDArray[np.float64],
    groups: pd.Index,
) -> pd.DataFrame:
    """Each group's fraction of seconds in each value of one column.

    Args:
        work: The rows reduced, holding the keys.
        keys: The columns that group the rows.
        share: The column's texts.
        names: The values that get a column, in the order of the columns.
        seconds: The seconds of each row of *work*, by position.
        groups: The groups of the result, in order.

    Returns:
        One column per value of *names*, ``float64`` with one row per group: the
        seconds of the rows with that value over the seconds of the rows whose
        value is not null, 0 for a value a group lacks and NaN for a group with
        no value.
    """
    present = share.readable & ~pd.isna(share.texts)
    rows = work.loc[present, list(keys)].assign(
        _text=share.texts[present], _seconds=seconds[present]
    )
    if rows.empty:
        return pd.DataFrame(np.nan, index=groups, columns=list(names))
    total = rows.groupby(list(keys), sort=True)["_seconds"].sum().reindex(groups)
    by_value = (
        rows.groupby([*keys, "_text"], sort=True)["_seconds"]
        .sum()
        .unstack("_text")
        .reindex(index=groups, columns=list(names))
        .fillna(0.0)
    )
    return by_value.div(total, axis=0)


_EDGE_TOLERANCE = 1e-9
"""How near a whole number of bins a time counts as on the edge.

The distance is relative to the edge's position in bins, and at least this
many bins.
"""


def _in_bins(times: NDArray[np.float64], bin_s: float) -> NDArray[np.float64]:
    """Each time in bins, snapped to a nearby whole number.

    A time that is ``frame / rate`` in floating point can fall an ulp under the
    edge it names: ``0.3 / 0.1`` is ``2.9999999999999996``. A position within
    ``_EDGE_TOLERANCE`` times the larger of 1 and the whole number is on it.
    """
    with np.errstate(invalid="ignore"):
        quotient = times / bin_s
        whole = np.rint(quotient)
        near = np.abs(quotient - whole) <= _EDGE_TOLERANCE * np.maximum(
            1.0, np.abs(whole)
        )
    return np.where(near, whole, quotient)


def _bin_index(times: NDArray[np.float64], bin_s: float) -> NDArray[np.float64]:
    """The bin ``floor(time / bin_s)`` of each time, NaN if not finite.

    A time within the tolerance of an edge is on the edge, so it starts the
    later bin (:func:`_in_bins`).
    """
    with np.errstate(invalid="ignore"):
        index = np.floor(_in_bins(times, bin_s))
    return np.where(np.isfinite(index), index, np.nan)


def _bin_edges(
    index: NDArray[np.float64], bin_s: float
) -> dict[str, NDArray[np.float64]]:
    """The ``bin_start`` and ``bin_end`` seconds of each bin index."""
    return {BIN_START: index * bin_s, BIN_END: (index + 1) * bin_s}


@dataclass(frozen=True)
class _BinnedSeconds:
    """The seconds of rows divided among the bins their intervals cover.

    Attributes:
        rows: The position of the row that each piece comes from.
        seconds: The seconds of each piece, the overlap of the row's interval
            with its bin.
        index: The bin of each piece, NaN for a row with no bin.
    """

    rows: NDArray[np.intp]
    seconds: NDArray[np.float64]
    index: NDArray[np.float64]


def _spread_over_bins(
    times: NDArray[np.float64],
    seconds: NDArray[np.float64],
    first: NDArray[np.float64],
    bin_s: float,
) -> _BinnedSeconds:
    """Divide each row's seconds among the bins of ``[time, time + seconds)``.

    Args:
        times: The time of each row.
        seconds: The seconds each row lasts, by position.
        first: The bin each row starts in (:func:`_bin_index`), NaN for none.
        bin_s: The length of a bin in seconds.

    Returns:
        One piece per row and bin it covers, each crediting the bin with the
        overlap. A row with no bin is one piece with no bin.
    """
    ends = times + seconds
    finite = np.isfinite(first)
    with np.errstate(invalid="ignore"):
        covered = np.ceil(_in_bins(ends, bin_s)) - 1
        last = np.where(finite, np.maximum(first, covered), first)
        spans = np.where(finite, last - first, 0).astype(np.intp) + 1
    rows = np.repeat(np.arange(times.size, dtype=np.intp), spans)
    offset = np.arange(rows.size) - np.repeat(np.cumsum(spans) - spans, spans)
    index = first[rows] + offset
    with np.errstate(invalid="ignore"):
        overlap = np.minimum(ends[rows], (index + 1) * bin_s) - np.maximum(
            times[rows], index * bin_s
        )
    credited = np.where(np.isfinite(index), np.maximum(overlap, 0.0), seconds[rows])
    return _BinnedSeconds(rows=rows, seconds=credited, index=index)


def _order_values(df: pd.DataFrame, order_col: str) -> NDArray[np.float64]:
    """The values of the ``order_col`` column as floats, NaN for a null.

    Raises:
        ValueError: The column is missing, is boolean or is not numeric.
    """
    if order_col not in df.columns:
        raise ValueError(
            f"order_col {order_col!r} is not in the input, which has "
            f"{sorted(map(str, df.columns))}"
        )
    values = df[order_col]
    if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(
        values.dtype
    ):
        raise ValueError(
            f"order_col {order_col!r} must be numeric, and it has dtype {values.dtype}"
        )
    return _plain_values(values).astype(np.float64)


def _split_into_subsets(df: pd.DataFrame, subsets: Mapping[str, str]) -> pd.DataFrame:
    """The rows each subset's filter keeps, stacked, with its name in ``subset``.

    The subsets stand in the order of their names, which the sorted reduction
    does not depend on. A subset that keeps no row adds none, and a row that
    several keep appears once for each, with its index.
    """
    parts: list[pd.DataFrame] = []
    for name, expression in sorted(subsets.items()):
        rows = df.query(expression)
        if not rows.empty:
            parts.append(rows.assign(**{SUBSET: name}))
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts)


@dataclass(frozen=True, eq=False)
class PreparedRows:
    """The rows of one table after the filters, ``transform`` and ``threshold``.

    The filters are ``filter_expr`` and then each of ``subsets``.

    :func:`prepare_rows` builds it and :func:`reduce_prepared` groups it, so a
    pool can prepare each entry and reduce the union (:func:`combine_prepared`).

    Attributes:
        group_keys: The columns that group the rows: the caller's keys, then
            ``subset`` when ``subsets`` is set, then ``bin_start`` and
            ``bin_end`` when ``bin_s`` is set.
        carried: The columns that take a value within each group: the ``first``
            columns, then ``keep_columns``.
        rows: The kept rows, by position, with the group keys, the carried
            columns, ``perspective`` when the input has it, as ``_v<i>``,
            the values of each of ``params.columns`` after ``transform`` and
            ``threshold`` and, with ``first_pass``, ``_order``, the values of
            ``params.order_col``. Empty when no row survives the filter.
        pieces: The position in ``rows`` of the row each seconds piece comes
            from. A row is one piece unless ``bin_s`` divides its seconds.
        piece_seconds: The seconds of each piece, ``None`` when no aggregation
            reads them.
        piece_keys: The group keys of each piece, with ``_s<i>``, the seconds
            of the piece that ``duration`` counts for each column.
        shares: The texts of each column, for ``share``. Empty without it.
    """

    group_keys: tuple[str, ...]
    carried: tuple[str, ...]
    rows: pd.DataFrame
    pieces: NDArray[np.intp]
    piece_seconds: NDArray[np.float64] | None
    piece_keys: pd.DataFrame
    shares: Mapping[str, _ShareTexts]


def prepare_rows(
    df: pd.DataFrame,
    params: AggregateParams,
    keys: Sequence[str],
    *,
    who: str,
    first: Sequence[str] = (),
) -> PreparedRows:
    """Apply the pre-aggregation steps of ``params`` to ``df``.

    Applies ``filter_expr``, then each of ``subsets`` to the rows it left (a row
    that two subsets keep is a row of each), ``transform`` and ``threshold``,
    finds each row's bin and divides its seconds among the bins it spans.
    :func:`reduce_prepared` reduces the result. The rules are those of
    :func:`reduce_column`.

    Args:
        df: The rows to reduce.
        params: The columns, aggregations and pre-aggregation steps.
        keys: The columns to group by.
        who: The feature's name, which opens the message of a time axis that
            ``duration`` cannot read.
        first: Columns that take their first value within each group.

    Returns:
        The prepared rows, with no rows when none survives the filter.

    Raises:
        ValueError: As :func:`reduce_column`, for what it reads from ``df``:
            a ``subset`` column present when ``subsets`` is set, a kept or
            subgroup column missing, a kept column varying within a
            group, ``bin_s`` without ``time``, ``duration`` or ``share``
            without a ``time`` that advances, ``share`` of a column of an
            unaccepted kind, or ``first_pass`` with an ``order_col`` that is
            missing or not numeric.
    """
    kept = tuple(params.keep_columns)
    for column in kept:
        if column not in df.columns:
            raise ValueError(f"keep_columns column {column!r} is not in the input")
    for column in params.subgroup_columns():
        if column not in df.columns:
            raise ValueError(f"subgroup_col column {column!r} is not in the input")
    if params.subsets is not None and SUBSET in df.columns:
        raise ValueError(
            f"subsets writes the column {SUBSET!r}, which the input already has"
        )
    bin_s = params.bin_s
    if bin_s is not None and C.time_col not in df.columns:
        raise ValueError(
            f"{who} bins rows by the {C.time_col!r} column, which the table "
            f"lacks: it has {sorted(map(str, df.columns))}. {_BIN_TIME_SENTENCE}"
        )
    seconds: NDArray[np.float64] | None = None
    if "duration" in params.aggs or "share" in params.aggs:
        df = df.reset_index(drop=True)
        seconds = _seconds_by_row(df, who)
    if params.filter_expr:
        df = df.query(params.filter_expr)
    if params.subsets is not None and not df.empty:
        df = _split_into_subsets(df, params.subsets)
    carried = (*first, *kept)
    edge_names = (BIN_START, BIN_END) if bin_s is not None else ()
    row_keys = (*keys, SUBSET) if params.subsets is not None else tuple(keys)
    group_keys = (*row_keys, *edge_names)
    if df.empty:
        return PreparedRows(
            group_keys=group_keys,
            carried=carried,
            rows=pd.DataFrame(),
            pieces=np.empty(0, dtype=np.intp),
            piece_seconds=None,
            piece_keys=pd.DataFrame(),
            shares={},
        )
    times = np.empty(0, dtype=np.float64)
    first_bin = np.empty(0, dtype=np.float64)
    edges: dict[str, NDArray[np.float64]] = {}
    if bin_s is not None:
        times = df[C.time_col].to_numpy(dtype=np.float64)
        first_bin = _bin_index(times, bin_s)
        edges = _bin_edges(first_bin, bin_s)
    _refuse_a_varying_column(df.assign(**edges) if edges else df, group_keys, kept)

    work = pd.DataFrame({key: df[key].to_numpy() for key in (*row_keys, *carried)})
    for name, edge in edges.items():
        work[name] = edge
    if _PERSPECTIVE_COLUMN in df.columns and _PERSPECTIVE_COLUMN not in work.columns:
        work[_PERSPECTIVE_COLUMN] = df[_PERSPECTIVE_COLUMN].to_numpy()
    kept_seconds = seconds[df.index.to_numpy()] if seconds is not None else None
    # Each piece of a row's seconds, by the row it comes from and the bin it
    # falls in. Without bins a row is one piece.
    pieces = np.arange(len(work), dtype=np.intp)
    piece_seconds = kept_seconds
    piece_keys = work[list(group_keys)].reset_index(drop=True)
    if kept_seconds is not None and bin_s is not None:
        spread = _spread_over_bins(times, kept_seconds, first_bin, bin_s)
        pieces = spread.rows
        piece_seconds = spread.seconds
        piece_keys = work[list(row_keys)].iloc[pieces].reset_index(drop=True)
        for name, edge in _bin_edges(spread.index, bin_s).items():
            piece_keys[name] = edge
    shares: dict[str, _ShareTexts] = {}
    if piece_seconds is not None and "share" in params.aggs:
        readable = work[list(group_keys)].notna().all(axis=1).to_numpy()
        shares = {
            column: _share_texts(df[column], column, readable)
            for column in params.columns
        }
    if "first_pass" in params.aggs:
        work[_ORDER_COLUMN] = _order_values(df, params.order_col)
    for index, column in enumerate(params.columns):
        values = df[column]
        if params.transform == "abs":
            values = values.abs()
        if params.threshold is not None:
            values = _passes_threshold(values, params.threshold, params.threshold_op)
        work[f"_v{index}"] = _plain_values(values)
        if piece_seconds is not None and "duration" in params.aggs:
            counted = (
                work[f"_v{index}"].to_numpy() == 1.0
                if params.threshold is not None
                else values.notna().to_numpy()
            )
            piece_keys[f"_s{index}"] = np.where(counted[pieces], piece_seconds, 0.0)
    return PreparedRows(
        group_keys=group_keys,
        carried=carried,
        rows=work,
        pieces=pieces,
        piece_seconds=piece_seconds,
        piece_keys=piece_keys,
        shares=shares,
    )


def combine_prepared(parts: Sequence[PreparedRows]) -> PreparedRows:
    """The union of several tables' prepared rows, to reduce as one.

    Concatenates the rows, the seconds pieces (each part's ``pieces`` offset by
    the rows before it) and the ``share`` texts, in the order of *parts*. A part
    with no rows adds nothing.

    Args:
        parts: The preparations of the tables, made with the same params.

    Returns:
        The prepared rows of every part.

    Raises:
        ValueError: *parts* is empty, a part's keys, carried columns or
            columns differ from the first's, or a ``share`` column holds
            integers in some parts and other values in others.
    """
    if not parts:
        raise ValueError("cannot combine no parts: give at least one")
    head = parts[0]
    for part in parts[1:]:
        if part.group_keys != head.group_keys:
            raise ValueError(
                "cannot combine rows prepared with different keys: "
                f"{list(head.group_keys)} and {list(part.group_keys)}"
            )
        if part.carried != head.carried:
            raise ValueError(
                "cannot combine rows prepared with different carried columns: "
                f"{list(head.carried)} and {list(part.carried)}"
            )
    live = [part for part in parts if not part.rows.empty]
    if not live:
        return head
    lead = live[0]
    for part in live[1:]:
        if (
            list(part.rows.columns) != list(lead.rows.columns)
            or list(part.piece_keys.columns) != list(lead.piece_keys.columns)
            or list(part.shares) != list(lead.shares)
            or (part.piece_seconds is None) != (lead.piece_seconds is None)
        ):
            raise ValueError(
                "cannot combine rows prepared with different columns: "
                f"{list(lead.rows.columns)} and {list(part.rows.columns)}"
            )
        for column, share in lead.shares.items():
            if part.shares[column].numeric != share.numeric:
                raise ValueError(
                    f"cannot combine rows prepared with integer values of "
                    f"column {column!r} in some parts and other values in others"
                )
    offsets: list[int] = []
    total = 0
    for part in live:
        offsets.append(total)
        total += len(part.rows)
    seconds = [part.piece_seconds for part in live if part.piece_seconds is not None]
    return PreparedRows(
        group_keys=head.group_keys,
        carried=head.carried,
        rows=pd.concat([part.rows for part in live], ignore_index=True),
        pieces=np.concatenate(
            [part.pieces + offset for part, offset in zip(live, offsets, strict=True)]
        ),
        piece_seconds=np.concatenate(seconds) if seconds else None,
        piece_keys=pd.concat([part.piece_keys for part in live], ignore_index=True),
        shares={
            column: _ShareTexts(
                texts=np.concatenate([part.shares[column].texts for part in live]),
                readable=np.concatenate(
                    [part.shares[column].readable for part in live]
                ),
                numeric=lead.shares[column].numeric,
            )
            for column in lead.shares
        },
    )


def reduce_prepared(
    prepared: PreparedRows, params: AggregateParams, *, pairs: PairRead
) -> pd.DataFrame:
    """Group prepared rows and aggregate each column with each of ``aggs``.

    Derives the ``share`` column names from the prepared texts and
    ``params.categories``, refuses an output name that two sources write, and
    reduces as :func:`reduce_column` describes.

    Args:
        prepared: The rows from :func:`prepare_rows` or :func:`combine_prepared`.
        params: The params the rows were prepared with.
        pairs: The rows of a pair table that ``sum``, ``count`` and ``duration``
            read.

    Returns:
        The reduced table, empty when no row remains.

    Raises:
        ValueError: A ``share`` value cannot name a column or is not among
            ``categories``, or an output name is written twice.
    """
    work = prepared.rows
    if work.empty:
        return pd.DataFrame()
    group_keys = list(prepared.group_keys)
    carried = prepared.carried
    pieces = prepared.pieces
    piece_seconds = prepared.piece_seconds
    piece_keys = prepared.piece_keys
    reduced = _carried_by_group(work, group_keys, carried)
    fractions: dict[str, pd.DataFrame] = {}
    if piece_seconds is not None and "share" in params.aggs:
        names = {
            column: _share_names(prepared.shares[column], column, params.categories)
            for column in params.columns
        }
        _refuse_a_repeated_output_name(params, group_keys, carried, names)
        fractions = {
            column: _share_fractions(
                piece_keys,
                group_keys,
                _ShareTexts(
                    texts=share.texts[pieces],
                    readable=share.readable[pieces],
                    numeric=share.numeric,
                ),
                names[column],
                piece_seconds,
                reduced.index,
            )
            for column, share in prepared.shares.items()
        }
    once = pairs == "once" and _PERSPECTIVE_COLUMN in work.columns
    perspective_zero = (work[_PERSPECTIVE_COLUMN] == 0).to_numpy() if once else None
    halved = work[perspective_zero] if perspective_zero is not None else work
    halved_pieces = (
        piece_keys[perspective_zero[pieces]]
        if perspective_zero is not None
        else piece_keys
    )
    for index, column in enumerate(params.columns):
        for agg in params.aggs:
            if agg == "share":
                for value, fraction in fractions[column].items():
                    reduced[f"{column}_share_{value}"] = fraction
                continue
            counting = agg in PAIR_COUNTING_AGGS
            if agg == "duration":
                source = halved_pieces if once else piece_keys
                result = source.groupby(group_keys, sort=True)[f"_s{index}"].sum()
            elif agg == "first_pass":
                passing = work[f"_v{index}"].to_numpy() == 1.0
                result = (
                    work[group_keys]
                    .assign(_first=work[_ORDER_COLUMN].where(passing))
                    .groupby(group_keys, sort=True)["_first"]
                    .min()
                )
            else:
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


def reduce_column(
    df: pd.DataFrame,
    params: AggregateParams,
    keys: Sequence[str],
    *,
    who: str,
    first: Sequence[str] = (),
    pairs: PairRead,
) -> pd.DataFrame:
    """Reduce each of ``params.columns`` of ``df`` to one row per distinct ``keys``.

    Applies ``filter_expr``, each of ``subsets``, ``transform`` and
    ``threshold``, then groups by ``keys`` and aggregates each column with each
    of ``aggs``. The result holds ``keys``, then ``subset`` when
    ``params.subsets`` is set, then ``bin_start`` and ``bin_end`` when
    ``params.bin_s`` is set, then each column of ``first`` (its first value in
    the group), then each of ``params.keep_columns`` (its one value in the
    group), then
    ``<column>_<agg>`` for each column and, within it, each aggregation, sorted
    by ``keys``. A row whose key is null belongs to no group. The result is
    empty when no row survives the filter.

    ``subset`` is the name of the subset whose filter keeps the rows of the
    group, after the subgroup columns among ``keys`` and before the bins. Each
    subset filters the rows left by ``filter_expr``, and a row that several
    subsets keep is a row of each, so the rows of two subsets overlap and the
    group of one does not read the group of another. A subset that keeps no row
    has no group, and the result is empty when none does.

    A pair table holds each pair once per perspective. Under ``pairs="once"``
    and with a ``perspective`` column present, ``sum``, ``count`` and
    ``duration`` read the ``perspective == 0`` rows, so each pair counts once,
    while the other aggregations of the same call read every row. Under
    ``each_perspective`` every aggregation reads every ordered pair row, as a
    caller that groups pair rows by a focal id needs.

    ``duration`` is the seconds of the rows that count: a value that is not
    null, or with ``threshold`` one that passes the ``threshold_op`` comparison
    after ``transform``. A row lasts until the next frame of its series, per
    :func:`~mosaic.core.timebase.row_seconds`, where a series is an arena's
    individual, or a pair row's ``id1``, ``id2`` and ``perspective``. The
    seconds are those of the rows before ``filter_expr``, ``subsets`` and
    ``keys`` split them, so a row that the filters keep lasts as it did without
    them.
    ``duration`` is read from the rows ``pairs`` selects, as ``sum`` and
    ``count`` are, is 0 for a group none of whose rows it reads, and is
    ``float64``.

    ``first_pass`` is the smallest ``params.order_col`` value among the group's
    rows whose value passes the ``threshold`` comparison after ``transform``
    (a null passes nothing, infinity passes a comparison it satisfies), skipping
    a row whose ``order_col`` is null. It is NaN for a group where no row
    passes, ``float64``, and reads every row under any ``pairs`` rule.

    ``share`` writes, for each value of a text, categorical, boolean or integer
    column, ``<column>_share_<value>``: the
    seconds of a group's rows whose value is it, over the seconds of the rows
    whose value is not null. It weighs rows as ``duration`` does, reads the
    column as it is after ``filter_expr`` and every row under any ``pairs``
    rule, and is ``float64``. A null row is in neither sum, a group of null
    rows only is NaN in every share column, and a value the group lacks is 0.
    Each value's text is the string, ``true`` or ``false``, or the decimal
    integer. The values are ``params.categories`` in their order when set, and
    otherwise those of the rows that belong to a group: integers by number,
    other values by code point. The columns of a ``share`` stand where it is in
    ``aggs``, within each column. Only a call with ``duration`` or ``share``
    reads ``time``.

    With ``params.bin_s`` the groups split by bin as well, and the result has a
    row per group and bin in which a row of the group starts. A row starts in
    the bin ``floor(time / bin_s)``, and a row with no finite ``time`` belongs
    to no group. Every aggregation but ``duration`` and ``share`` reads a row
    in the bin it starts in. ``duration`` and ``share`` credit each bin with the
    overlap of the row's interval, from its time for its seconds, and drop what
    falls in a bin of the group where no row starts.

    Args:
        df: The rows to reduce.
        params: The columns, aggregations and pre-aggregation steps.
        keys: The columns to group by.
        who: The feature's name, which opens the message of a time axis that
            ``duration`` cannot read.
        first: Columns that take their first value within each group.
        pairs: The rows of a pair table that ``sum``, ``count`` and ``duration``
            read.

    Returns:
        The reduced table, empty when no row remains.

    Raises:
        ValueError: A kept column is not in ``df``, or takes more than one value
            in a group of the rows left after ``filter_expr``, or ``bin_s`` is
            set and ``time`` is missing, or ``duration``
            or ``share`` is asked of an input whose ``time`` is missing or does
            not advance, or ``share`` is asked of a column of an unaccepted
            kind, of a value that cannot name a column or that is not among
            ``categories``, or of a name that another output has, or
            ``first_pass`` is asked of an ``order_col`` that is missing or not
            numeric.
    """
    return reduce_prepared(
        prepare_rows(df, params, keys, who=who, first=first), params, pairs=pairs
    )


def blank_reduction(
    df: pd.DataFrame,
    params: AggregateParams,
    keys: Sequence[str],
    *,
    subsets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per distinct ``keys`` of ``df`` that reduces no values.

    For an entry whose ``filter_expr`` keeps no row, or whose ``subsets`` leave
    some subset no row. It has the columns that
    :func:`reduce_column` gives an entry that keeps rows: each of
    ``params.keep_columns`` holds its one value in the group's rows before the
    filter, each of the ``sum``, ``count`` and ``duration`` aggregations is 0
    (seconds for ``duration``), as for a group none of whose rows it reads, and
    every other aggregation, ``first_pass`` among them, is NaN. ``share`` is
    NaN in the column of each of ``params.categories``, in their order, and
    writes no column when that is unset, since the entry has no value to name
    one. A row whose key is null
    belongs to no group. With ``params.bin_s`` the row of each group has NaN
    ``bin_start`` and ``bin_end`` after the keys: no row of the entry starts in
    a bin, and the row keeps the entry, its kept labels and the columns that an
    entry with rows writes. With *subsets*, each group has one such row for each
    name, in the ``subset`` column that follows the keys.

    Args:
        df: The rows before the filter, which hold the keys and the kept columns.
        params: The columns, aggregations and kept columns of the result.
        keys: The columns to group by.
        subsets: The names of the subsets that keep no row, or ``None`` for a
            result without a ``subset`` column.

    Returns:
        The blank rows, sorted by ``keys`` and then ``subset``.

    Raises:
        ValueError: A kept column takes more than one value in a group of ``df``.
    """
    kept = tuple(params.keep_columns)
    _refuse_a_varying_column(df, keys, kept)
    work = pd.DataFrame({key: df[key].to_numpy() for key in (*keys, *kept)})
    blank = _carried_by_group(work, keys, kept).reset_index()
    edges_at = len(keys)
    if subsets is not None:
        named: list[pd.DataFrame] = []
        for name in sorted(subsets):
            part = blank.copy()
            part.insert(len(keys), SUBSET, name)
            named.append(part)
        blank = pd.concat(named, ignore_index=True).sort_values(
            [*keys, SUBSET], kind="stable", ignore_index=True
        )
        edges_at += 1
    if params.bin_s is not None:
        blank.insert(edges_at, BIN_START, np.nan)
        blank.insert(edges_at + 1, BIN_END, np.nan)
    for column in params.columns:
        for agg in params.aggs:
            if agg == "share":
                for category in params.categories or ():
                    blank[f"{column}_share_{category}"] = np.nan
            else:
                blank[f"{column}_{agg}"] = 0.0 if agg in PAIR_COUNTING_AGGS else np.nan
    return blank


def blank_rows(
    df: pd.DataFrame,
    params: AggregateParams,
    keys: Sequence[str],
    reduced: pd.DataFrame,
) -> pd.DataFrame:
    """*reduced*, with a blank row for what it lacks, so every entry has the same rows.

    Without ``subsets``, an entry that reduced to nothing is its blank rows, one
    for each group of ``keys`` in *df*. With them, each subset that *reduced*
    has no row of gets those blank rows under its name, and the rows stay sorted
    by ``keys`` and ``subset``. A blank value that is a whole number (the 0 of
    a ``count`` or ``sum``) takes the integer type of the rows that hold values,
    so an entry's counts do not turn to floats.

    Args:
        df: The rows before the filter, which hold the keys and the kept columns.
        params: The params the rows were reduced with.
        keys: The columns the rows were grouped by.
        reduced: The reduction of *df*, empty when no row survived the filters.

    Returns:
        The table that the entry writes.

    Raises:
        ValueError: A kept column takes more than one value in a group of ``df``.
    """
    if params.subsets is None:
        return reduced if not reduced.empty else blank_reduction(df, params, keys)
    written: set[str] = set()
    if not reduced.empty:
        written = {str(name) for name in reduced[SUBSET].unique()}
    empty = [name for name in params.subsets if name not in written]
    if not empty:
        return reduced
    blank = blank_reduction(df, params, keys, subsets=empty)
    if reduced.empty:
        return blank
    for column in blank.columns.intersection(reduced.columns):
        dtype = reduced[column].dtype
        if pd.api.types.is_integer_dtype(dtype) and blank[column].notna().all():
            blank[column] = blank[column].astype(dtype)
    return pd.concat([reduced, blank], ignore_index=True).sort_values(
        [*keys, SUBSET], kind="stable", ignore_index=True
    )


def _carried_by_group(
    work: pd.DataFrame, keys: Sequence[str], carried: Sequence[str]
) -> pd.DataFrame:
    """The first value of each of *carried* per group of *keys*, indexed by the keys.

    A row whose key is null belongs to no group. Without *carried*, the frame
    holds only the index of the groups.
    """
    grouped = work.groupby(list(keys), sort=True)
    if carried:
        return grouped[list(carried)].first()
    return pd.DataFrame(index=grouped.size().index)
