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
shares. :func:`reduce_column` refuses a group whose rows disagree on one, and
:func:`blank_reduction` reads it from an emptied entry's rows before the filter.

``duration`` weighs each row by the seconds it stands for, read from the input's
``frame`` and ``time`` columns (:func:`~mosaic.core.timebase.row_seconds`), so
the sum is a time and not a count of rows at a frame rate that varies. ``share``
weighs the rows the same way: it writes, for each value of a text, categorical,
boolean or integer column, the fraction of the group's seconds that the column
holds it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
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
    "mean", "median", "min", "max", "std", "sum", "count", "duration", "share"
]

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
    "are mean, median, min, max, std, sum, count, duration and share. Without "
    "threshold, NaN values are skipped for every mode, and count counts "
    "non-null values. duration is the seconds the rows that count last: each "
    "row lasts until the next frame of its individual, read from the time "
    "column, and a row counts when its value is not null, or, with threshold, "
    "when it exceeds it, so a boolean column with threshold 0.5 gives the time "
    "it holds. share reads a column of text, categories, booleans or integers "
    "and writes a column `<column>_share_<value>` for each value, true or false for "
    "a boolean and the decimal integer for a number: the seconds of the rows "
    "with that value over the seconds of the rows whose value is not null, so "
    "a group's shares sum to 1. A group whose rows are all null gets NaN, and "
    "a value the group lacks gets 0. A value must consist of letters, digits, "
    "underscores and hyphens. share is refused with threshold and with "
    "transform abs. duration and share need a time column, and "
    "frame-aggregate takes neither."
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
    "Which rows of a pair table sum, count and duration read, when the input has a "
    "perspective column; the other aggregations read every row. once reads "
    "perspective 0, each pair once, and each_perspective reads every "
    "ordered pair row. auto is resolved by a recipe's plan: once when every "
    "column is declared symmetric over mirrored rows and filter_expr is "
    "unset, every row otherwise. A run outside a plan (mosaic run --feature) has "
    "no input chain to resolve and reads every row under auto, which never "
    "halves a value without a declaration. Set once there to count each "
    "pair once."
)

_TIME_SENTENCE = (
    "duration and share weigh each row by its seconds, read from the time column: "
    "give the step an input with a time column, such as tracks."
)

_SAFE_VALUE = re.compile(r"[A-Za-z0-9_-]+")
_PERSPECTIVE_COLUMN = "perspective"
_FOCAL_COLUMNS = ("id1", "id2")
_MAX_LISTED_VALUES = 5


class AggregateParams(Params):
    """The parameters of the reduction that every aggregate shares.

    Each of ``columns`` is reduced by each of ``aggs``. Before it,
    ``filter_expr`` drops rows, ``transform`` takes the absolute value and
    ``threshold`` turns each column into whether it exceeds a value.
    ``subgroup_col`` splits the rows reduced together. ``keep_columns`` are
    written to the output beside the keys. ``categories`` fixes the columns that
    ``share`` writes.
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
    categories: Annotated[list[str] | None, Declared(_CATEGORIES_DESCRIPTION)] = Field(
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
        repeated = sorted({item for item in value if value.count(item) > 1})
        if repeated:
            raise ValueError(f"lists {', '.join(repeated)} more than once")
        return value

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
        """The hashed params, less an empty ``keep_columns`` or unset ``categories``.

        A run that keeps no column and sets no category writes the output it
        wrote before the fields existed, so leaving them out keeps every such
        run's identifier. A kept column or a category adds output columns and
        enters the hash as any field does.
        """
        dumped = super().identity_dump()
        if not self.keep_columns:
            del dumped["keep_columns"]
        if self.categories is None:
            del dumped["categories"]
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


@dataclass(frozen=True)
class _ShareColumn:
    """What ``share`` reads of one column, and the columns it writes for it.

    Attributes:
        texts: The text of each row's value, by position, ``None`` for a null.
        readable: Whether each row belongs to a group, by position.
        names: The values that get a column, in the order of the columns.
    """

    texts: NDArray[np.object_]
    readable: NDArray[np.bool_]
    names: list[str]


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


def _share_column(
    values: pd.Series,
    column: str,
    readable: NDArray[np.bool_],
    categories: Sequence[str] | None,
) -> _ShareColumn:
    """The texts of one column and the values that ``share`` writes a column for.

    The values are those of the rows that belong to a group: ``categories`` in
    their order when set, and otherwise each value that occurs, integers by
    number and other values by code point.

    Raises:
        ValueError: The column is of an unaccepted kind, a value is not made of
            letters, digits, underscores and hyphens, or a value is not among
            ``categories``.
    """
    texts, numeric = _texts_of_values(values, column)
    read = {str(text) for text in texts[readable & ~pd.isna(texts)]}
    unsafe = sorted(text for text in read if not _SAFE_VALUE.fullmatch(text))
    if unsafe:
        raise ValueError(
            f"share cannot name a column for the value {unsafe[0]!r} of column "
            f"{column!r}: recode it upstream to letters, digits, underscores "
            "and hyphens"
        )
    if categories is None:
        names = sorted(read, key=int) if numeric else sorted(read)
    else:
        outside = sorted(read.difference(categories))
        if outside:
            raise ValueError(
                f"column {column!r} has the value {outside[0]!r}, which is not "
                f"among the categories {list(categories)}"
            )
        names = list(categories)
    return _ShareColumn(texts=texts, readable=readable, names=names)


def _refuse_a_repeated_output_name(
    params: AggregateParams,
    keys: Sequence[str],
    carried: Sequence[str],
    shares: dict[str, _ShareColumn],
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
                for value in shares[column].names:
                    claim(
                        f"{column}_share_{value}",
                        f"share of column {column!r} for the value {value!r}",
                    )
            else:
                claim(f"{column}_{agg}", f"{agg} of column {column!r}")


def _share_fractions(
    work: pd.DataFrame,
    keys: Sequence[str],
    share: _ShareColumn,
    seconds: NDArray[np.float64],
    groups: pd.Index,
) -> pd.DataFrame:
    """Each group's fraction of seconds in each value of one column.

    Args:
        work: The rows reduced, holding the keys.
        keys: The columns that group the rows.
        share: The column's texts and the values that get a column.
        seconds: The seconds of each row of *work*, by position.
        groups: The groups of the result, in order.

    Returns:
        One column per value of *share*, ``float64`` with one row per group: the
        seconds of the rows with that value over the seconds of the rows whose
        value is not null, 0 for a value a group lacks and NaN for a group with
        no value.
    """
    present = share.readable & ~pd.isna(share.texts)
    rows = work.loc[present, list(keys)].assign(
        _text=share.texts[present], _seconds=seconds[present]
    )
    if rows.empty:
        return pd.DataFrame(np.nan, index=groups, columns=share.names)
    total = rows.groupby(list(keys), sort=True)["_seconds"].sum().reindex(groups)
    by_value = (
        rows.groupby([*keys, "_text"], sort=True)["_seconds"]
        .sum()
        .unstack("_text")
        .reindex(index=groups, columns=share.names)
        .fillna(0.0)
    )
    return by_value.div(total, axis=0)


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

    Applies ``filter_expr``, ``transform`` and ``threshold``, then groups by
    ``keys`` and aggregates each column with each of ``aggs``. The result holds
    ``keys``, then each column of ``first`` (its first value in the group), then
    each of ``params.keep_columns`` (its one value in the group), then
    ``<column>_<agg>`` for each column and, within it, each aggregation, sorted
    by ``keys``. A row whose key is null belongs to no group. The result is
    empty when no row survives the filter.

    A pair table holds each pair once per perspective. Under ``pairs="once"``
    and with a ``perspective`` column present, ``sum``, ``count`` and
    ``duration`` read the ``perspective == 0`` rows, so each pair counts once,
    while the other aggregations of the same call read every row. Under
    ``each_perspective`` every aggregation reads every ordered pair row, as a
    caller that groups pair rows by a focal id needs.

    ``duration`` is the seconds of the rows that count: a value that is not
    null, or with ``threshold`` one above it after ``transform``. A row lasts
    until the next frame of its series, per
    :func:`~mosaic.core.timebase.row_seconds`, where a series is an arena's
    individual, or a pair row's ``id1``, ``id2`` and ``perspective``. The
    seconds are those of the rows before ``filter_expr`` and before ``keys``
    split them, so a row that the filter keeps lasts as it did without it.
    ``duration`` is read from the rows ``pairs`` selects, as ``sum`` and
    ``count`` are, is 0 for a group none of whose rows it reads, and is
    ``float64``.

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
            in a group of the rows left after ``filter_expr``, or ``duration``
            or ``share`` is asked of an input whose ``time`` is missing or does
            not advance, or ``share`` is asked of a column of an unaccepted
            kind, of a value that cannot name a column or that is not among
            ``categories``, or of a name that another output has.
    """
    kept = tuple(params.keep_columns)
    for column in kept:
        if column not in df.columns:
            raise ValueError(f"keep_columns column {column!r} is not in the input")
    seconds: NDArray[np.float64] | None = None
    if "duration" in params.aggs or "share" in params.aggs:
        df = df.reset_index(drop=True)
        seconds = _seconds_by_row(df, who)
    if params.filter_expr:
        df = df.query(params.filter_expr)
    if df.empty:
        return pd.DataFrame()
    _refuse_a_varying_column(df, keys, kept)

    carried = (*first, *kept)
    work = pd.DataFrame({key: df[key].to_numpy() for key in (*keys, *carried)})
    kept_seconds = seconds[df.index.to_numpy()] if seconds is not None else None
    reduced = _carried_by_group(work, keys, carried)
    fractions: dict[str, pd.DataFrame] = {}
    if kept_seconds is not None and "share" in params.aggs:
        readable = work[list(keys)].notna().all(axis=1).to_numpy()
        shares = {
            column: _share_column(df[column], column, readable, params.categories)
            for column in params.columns
        }
        _refuse_a_repeated_output_name(params, keys, carried, shares)
        fractions = {
            column: _share_fractions(work, keys, share, kept_seconds, reduced.index)
            for column, share in shares.items()
        }
    for index, column in enumerate(params.columns):
        values = df[column]
        if params.transform == "abs":
            values = values.abs()
        if params.threshold is not None:
            values = (values > params.threshold).astype(float)
        work[f"_v{index}"] = _plain_values(values)
        if kept_seconds is not None and "duration" in params.aggs:
            counted = (
                work[f"_v{index}"].to_numpy() == 1.0
                if params.threshold is not None
                else values.notna().to_numpy()
            )
            work[f"_s{index}"] = np.where(counted, kept_seconds, 0.0)

    once = pairs == "once" and _PERSPECTIVE_COLUMN in df.columns
    halved = work[(df[_PERSPECTIVE_COLUMN] == 0).to_numpy()] if once else work
    group_keys = list(keys)
    for index, column in enumerate(params.columns):
        for agg in params.aggs:
            if agg == "share":
                for value, fraction in fractions[column].items():
                    reduced[f"{column}_share_{value}"] = fraction
                continue
            counting = agg in PAIR_COUNTING_AGGS
            source = halved if once and counting else work
            if agg == "duration":
                result = source.groupby(group_keys, sort=True)[f"_s{index}"].sum()
            else:
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
    :func:`reduce_column` gives an entry that keeps rows: each of
    ``params.keep_columns`` holds its one value in the group's rows before the
    filter, each of the ``sum``, ``count`` and ``duration`` aggregations is 0
    (seconds for ``duration``), as for a group none of whose rows it reads, and
    every other aggregation is NaN. ``share`` is NaN in the column of each of
    ``params.categories``, in their order, and writes no column when that is
    unset, since the entry has no value to name one. A row whose key is null
    belongs to no group.

    Args:
        df: The rows before the filter, which hold the keys and the kept columns.
        params: The columns, aggregations and kept columns of the result.
        keys: The columns to group by.

    Returns:
        The blank rows, sorted by ``keys``.

    Raises:
        ValueError: A kept column takes more than one value in a group of ``df``.
    """
    kept = tuple(params.keep_columns)
    _refuse_a_varying_column(df, keys, kept)
    work = pd.DataFrame({key: df[key].to_numpy() for key in (*keys, *kept)})
    blank = _carried_by_group(work, keys, kept).reset_index()
    for column in params.columns:
        for agg in params.aggs:
            if agg == "share":
                for category in params.categories or ():
                    blank[f"{column}_share_{category}"] = np.nan
            else:
                blank[f"{column}_{agg}"] = 0.0 if agg in PAIR_COUNTING_AGGS else np.nan
    return blank


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
