"""Tests for the reduction that the aggregate features share."""

from __future__ import annotations

from typing import get_args

import operator
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import stats

from mosaic.behavior.feature_library.aggregate_common import (
    AGGREGATIONS,
    MAX_VALUE_BIN_EDGES,
    Aggregation,
    AggregateParams,
    QuantileMethod,
    ThresholdOp,
    ValueBins,
    blank_reduction,
    blank_rows,
    combine_prepared,
    distinct_quantiles,
    prepare_rows,
    quantile_text,
    quantiles_or_nan,
    reduce_column,
    reduce_prepared,
    repeated_items,
    split_keys,
)

KEYS = ["group", "sequence"]
WHO = "test-aggregate"


def _entry(
    name: str, first_id: int, offset_s: float, values: np.ndarray
) -> pd.DataFrame:
    """Two individuals over 20 frames at 10 Hz, one entry's table.

    Args:
        name: The entry's ``group``.
        first_id: The id of the first individual, so entries share no series.
        offset_s: Seconds added to every time, which moves the rows against the
            bin edges.
        values: The values of the two individuals, 40 in all.
    """
    frames = np.tile(np.arange(20), 2)
    states = np.where(values > 0.5, "up", "down")
    return pd.DataFrame(
        {
            "group": name,
            "sequence": "S",
            "id": np.repeat([first_id, first_id + 1], 20),
            "frame": frames,
            "time": frames / 10 + offset_s,
            "value": values,
            "state": states,
        }
    )


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two entries whose rows cross bin edges at different places."""
    generator = np.random.default_rng(7)
    first = _entry("A", 0, 0.0, generator.random(40))
    second = _entry("B", 2, 0.02, generator.random(40))
    # A value that only the second entry holds.
    second.loc[[3, 7, 25], "state"] = "flat"
    return first, second


def _numeric_params(bin_s: float | None) -> AggregateParams:
    return AggregateParams(
        columns=["value"], aggs=["sum", "count", "duration", "mean"], bin_s=bin_s
    )


def test_the_table_lists_the_members_of_the_alias_and_no_other() -> None:
    assert set(AGGREGATIONS) == set(get_args(Aggregation.__value__))


def test_the_table_assigns_each_aggregation_its_combination() -> None:
    partials = {
        name for name, rule in AGGREGATIONS.items() if rule.combines == "partials"
    }
    values = {name for name, rule in AGGREGATIONS.items() if rule.combines == "values"}
    assert partials == {
        "count",
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "duration",
        "share",
        "first_pass",
        "circmean",
        "circstd",
    }
    assert values == {"median", "quantile"}


@pytest.mark.parametrize("keys", [KEYS, ["sequence"]], ids=["apart", "pooled"])
@pytest.mark.parametrize("bin_s", [None, 0.55])
@pytest.mark.parametrize(
    ("column", "aggs"),
    [
        ("value", ["sum", "count", "duration"]),
        ("state", ["count", "duration", "share"]),
    ],
)
def test_entries_prepared_apart_then_combined_equal_the_inline_reduction(
    keys: list[str], bin_s: float | None, column: str, aggs: list[Aggregation]
) -> None:
    first, second = _tables()
    params = AggregateParams(columns=[column], aggs=aggs, bin_s=bin_s)
    inline = reduce_column(
        pd.concat([first, second], ignore_index=True),
        params,
        keys,
        who=WHO,
        pairs="each_perspective",
    )
    combined = combine_prepared(
        [prepare_rows(table, params, keys, who=WHO) for table in (first, second)]
    )
    pooled = reduce_prepared(combined, params, pairs="each_perspective")
    pd.testing.assert_frame_equal(pooled, inline, check_exact=True)
    assert pooled[f"{column}_duration"].gt(0).all()
    if column == "state":
        # The value only the second part holds is named, and 0 where a group has
        # none of it.
        flat = pooled["state_share_flat"]
        assert flat.gt(0).any()
        if keys == KEYS:
            assert flat[pooled["group"] == "A"].eq(0).all()
    if bin_s is not None:
        assert pooled["bin_start"].nunique() > 1


def _dyadic_entry(name: str, offset_s: float) -> pd.DataFrame:
    """One individual over 8 frames at 8 Hz, whose times are exact in binary."""
    frames = np.arange(8)
    return pd.DataFrame(
        {
            "group": name,
            "sequence": "S",
            "id": 0,
            "frame": frames,
            "time": frames / 8 + offset_s,
            "value": 1.0,
        }
    )


def test_a_row_crossing_a_bin_edge_credits_both_bins_after_combining() -> None:
    # Every time and length is a multiple of 1 / 32 s and the bins are 5 / 16 s
    # long, so each sum is exact in any order.
    first = _dyadic_entry("A", 0.0)
    second = _dyadic_entry("B", 1 / 32)
    params = AggregateParams(
        columns=["value"], aggs=["count", "duration"], bin_s=5 / 16
    )
    parts = [prepare_rows(t, params, KEYS, who=WHO) for t in (first, second)]
    pooled = reduce_prepared(combine_prepared(parts), params, pairs="each_perspective")
    # Entry A: frame 2 spans 0.25 to 0.375 s, over the edge at 0.3125 s, and it
    # starts in the first bin. The last frame's tail past 0.9375 s falls in a bin
    # where no row of the group starts, so it is dropped.
    entry_a = pooled[pooled["group"] == "A"]
    np.testing.assert_array_equal(entry_a["bin_start"], [0.0, 5 / 16, 10 / 16])
    np.testing.assert_array_equal(entry_a["value_duration"], [5 / 16] * 3)
    np.testing.assert_array_equal(entry_a["value_count"], [3, 2, 3])
    # Entry B: frame 2 spans 0.28125 to 0.40625 s and starts in the first bin,
    # so 1 / 32 s of it is credited there and 3 / 32 s to the second.
    entry_b = pooled[pooled["group"] == "B"]
    np.testing.assert_array_equal(entry_b["value_duration"], [9 / 32, 10 / 32, 10 / 32])
    np.testing.assert_array_equal(entry_b["value_count"], [3, 2, 3])
    # Pooled into one group, the two entries' credits add.
    pool = [prepare_rows(t, params, ["sequence"], who=WHO) for t in (first, second)]
    both = reduce_prepared(combine_prepared(pool), params, pairs="each_perspective")
    np.testing.assert_array_equal(
        both["value_duration"], [5 / 16 + 9 / 32, 5 / 16 + 10 / 32, 5 / 16 + 10 / 32]
    )


def test_a_part_with_other_keys_is_refused() -> None:
    first, second = _tables()
    params = _numeric_params(None)
    parts = [
        prepare_rows(first, params, KEYS, who=WHO),
        prepare_rows(second, params, ["group"], who=WHO),
    ]
    with pytest.raises(ValueError, match="keys"):
        _ = combine_prepared(parts)


def test_a_part_with_other_carried_columns_is_refused() -> None:
    first, second = _tables()
    params = _numeric_params(None)
    parts = [
        prepare_rows(first, params, KEYS, who=WHO, first=["time"]),
        prepare_rows(second, params, KEYS, who=WHO),
    ]
    with pytest.raises(ValueError, match="carried"):
        _ = combine_prepared(parts)


def test_a_part_that_the_filter_emptied_adds_nothing() -> None:
    first, second = _tables()
    params = AggregateParams(
        columns=["value"],
        aggs=["sum", "duration"],
        filter_expr="group == 'A'",
    )
    parts = [prepare_rows(table, params, KEYS, who=WHO) for table in (first, second)]
    pooled = reduce_prepared(combine_prepared(parts), params, pairs="each_perspective")
    alone = reduce_column(first, params, KEYS, who=WHO, pairs="each_perspective")
    pd.testing.assert_frame_equal(pooled, alone, check_exact=True)


def test_no_part_is_refused() -> None:
    with pytest.raises(ValueError, match="no parts"):
        _ = combine_prepared([])


def test_parts_whose_share_column_differs_in_kind_are_refused() -> None:
    first, second = _tables()
    first["state"] = np.where(first["value"] > 0.5, 1, 2)
    params = AggregateParams(columns=["state"], aggs=["share"])
    parts = [prepare_rows(table, params, KEYS, who=WHO) for table in (first, second)]
    with pytest.raises(ValueError, match="state"):
        _ = combine_prepared(parts)


# --- first_pass -----------------------------------------------------------------------

_COMPARES: dict[ThresholdOp, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def _first_pass_params(
    op: ThresholdOp = ">", threshold: float = 1.0, **fields: object
) -> AggregateParams:
    return AggregateParams.model_validate(
        {
            "columns": ["value"],
            "aggs": ["first_pass"],
            "threshold": threshold,
            "threshold_op": op,
            **fields,
        }
    )


def _first_crossing(
    order: list[float], values: list[float], op: ThresholdOp, threshold: float
) -> float:
    """The first order value, in order, whose value passes: the inline reference.

    A null order value is skipped and a NaN value passes nothing.
    """
    rows = sorted((o, v) for o, v in zip(order, values, strict=True) if not np.isnan(o))
    for o, v in rows:
        if not np.isnan(v) and _COMPARES[op](v, threshold):
            return o
    return float("nan")


def _timed(values: list[float], order: list[float], group: str = "A") -> pd.DataFrame:
    """One individual's rows, with the order column ``stim`` beside the time."""
    count = len(values)
    return pd.DataFrame(
        {
            "group": group,
            "sequence": "S",
            "id": 0,
            "frame": np.arange(count),
            "time": np.arange(count) / 10,
            "stim": order,
            "value": values,
        }
    )


def _first_pass(
    table: pd.DataFrame, params: AggregateParams, keys: list[str] = KEYS
) -> pd.DataFrame:
    return reduce_column(table, params, keys, who=WHO, pairs="each_perspective")


def test_an_infinite_value_passes_and_a_nan_does_not() -> None:
    table = _timed([np.nan, np.inf, 0.5, 2.0], [1.0, 2.0, 3.0, 4.0])

    out = _first_pass(table, _first_pass_params(order_col="stim"))

    np.testing.assert_array_equal(out["value_first_pass"], [2.0])


def test_a_negative_infinity_passes_a_less_than_comparison() -> None:
    table = _timed([np.nan, -np.inf, 5.0], [1.0, 2.0, 3.0])

    out = _first_pass(table, _first_pass_params("<", order_col="stim"))

    np.testing.assert_array_equal(out["value_first_pass"], [2.0])


def test_a_group_where_no_row_passes_is_nan_in_float64() -> None:
    table = pd.concat(
        [_timed([0.0, 0.5], [1.0, 2.0], "A"), _timed([3.0, 0.5], [7.0, 8.0], "B")]
    )

    out = _first_pass(table, _first_pass_params(order_col="stim"))

    assert out["value_first_pass"].dtype == np.float64
    np.testing.assert_array_equal(out["value_first_pass"], [np.nan, 7.0])


def test_rows_in_any_order_give_the_smallest_order_value_that_passes() -> None:
    table = _timed([5.0, 5.0, 0.0, 5.0], [4.0, 2.0, 1.0, 2.0])

    out = _first_pass(table, _first_pass_params(order_col="stim"))

    # The two passing rows tie at 2.0, and the row of order 1.0 fails.
    np.testing.assert_array_equal(out["value_first_pass"], [2.0])


def test_a_row_with_a_null_order_value_is_skipped() -> None:
    table = _timed([5.0, 5.0, 5.0], [np.nan, 3.0, 2.0])

    out = _first_pass(table, _first_pass_params(order_col="stim"))

    np.testing.assert_array_equal(out["value_first_pass"], [2.0])
    only_null = _timed([5.0], [np.nan])
    alone = _first_pass(only_null, _first_pass_params(order_col="stim"))
    np.testing.assert_array_equal(alone["value_first_pass"], [np.nan])


def test_a_nullable_integer_order_column_is_read_with_nulls_skipped() -> None:
    table = _timed([5.0, 5.0, 5.0], [0, 4, 3])
    table["stim"] = pd.array([None, 4, 3], dtype="Int64")

    out = _first_pass(table, _first_pass_params(order_col="stim"))

    np.testing.assert_array_equal(out["value_first_pass"], [3.0])


@pytest.mark.parametrize("op", [">", ">=", "<", "<="])
def test_each_comparison_matches_the_inline_loop(op: ThresholdOp) -> None:
    # Values equal to the threshold separate a strict comparison from a loose one.
    values = [2.0, 3.0, np.nan, 3.0, 4.0, 1.0, np.inf, -np.inf]
    order = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    table = _timed(values, order)

    out = _first_pass(table, _first_pass_params(op, 3.0, order_col="stim"))

    expected = _first_crossing(order, values, op, 3.0)
    np.testing.assert_array_equal(out["value_first_pass"], [expected])


def test_a_transform_applies_before_the_comparison() -> None:
    table = _timed([-5.0, 0.5, 2.0], [1.0, 2.0, 3.0])
    params = _first_pass_params(order_col="stim", transform="abs")

    out = _first_pass(table, params)

    np.testing.assert_array_equal(out["value_first_pass"], [1.0])


def test_the_filter_drops_rows_before_the_first_pass() -> None:
    table = _timed([5.0, 5.0, 5.0], [-0.5, 1.0, 2.0])
    params = _first_pass_params(order_col="stim", filter_expr="stim != -0.5")

    out = _first_pass(table, params)

    np.testing.assert_array_equal(out["value_first_pass"], [1.0])


@pytest.mark.parametrize("order_col", ["time", "stim", "frame"])
@pytest.mark.parametrize("op", [">", "<="])
def test_the_value_equals_a_loop_over_rows_sorted_by_the_order_column(
    order_col: str, op: ThresholdOp
) -> None:
    generator = np.random.default_rng(11)
    tables: list[pd.DataFrame] = []
    for name in "ABCD":
        count = 30
        values = generator.normal(size=count)
        values[generator.random(count) < 0.1] = np.nan
        order = generator.permutation(count).astype(float)
        order[generator.random(count) < 0.1] = np.nan
        tables.append(_timed(values.tolist(), order.tolist(), name))
    table = pd.concat(tables, ignore_index=True)
    table["time"] = table.groupby("group").cumcount() / 10
    params = _first_pass_params(op, 0.3, order_col=order_col)

    out = _first_pass(table, params)

    expected: list[float] = []
    for name in "ABCD":
        rows = table[table["group"] == name]
        expected.append(
            _first_crossing(rows[order_col].tolist(), rows["value"].tolist(), op, 0.3)
        )
    assert out["group"].tolist() == list("ABCD")
    np.testing.assert_array_equal(out["value_first_pass"], expected)


def test_each_bin_takes_the_first_pass_of_its_own_rows() -> None:
    generator = np.random.default_rng(5)
    values = generator.normal(size=40)
    order = generator.permutation(40).astype(float)
    table = _timed(values.tolist(), order.tolist())
    params = _first_pass_params(">", 0.2, order_col="stim", bin_s=1.0)

    out = _first_pass(table, params)

    expected = [
        _first_crossing(
            order[start : start + 10].tolist(),
            values[start : start + 10].tolist(),
            ">",
            0.2,
        )
        for start in range(0, 40, 10)
    ]
    np.testing.assert_array_equal(out["bin_start"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out["value_first_pass"], expected)


def test_every_row_is_read_under_any_pairs_rule() -> None:
    table = pd.DataFrame(
        {
            "group": "A",
            "sequence": "S",
            "id1": [0, 0, 0, 0],
            "id2": [1, 1, 1, 1],
            "perspective": [0, 1, 0, 1],
            "frame": [0, 0, 1, 1],
            "time": [0.0, 0.0, 0.1, 0.1],
            "stim": [4.0, 1.0, 5.0, 2.0],
            "value": [0.0, 9.0, 0.0, 9.0],
        }
    )
    params = _first_pass_params(order_col="stim")

    for pairs in ("once", "each_perspective"):
        out = reduce_column(table, params, KEYS, who=WHO, pairs=pairs)
        np.testing.assert_array_equal(out["value_first_pass"], [1.0])


def test_a_pool_of_entries_takes_the_first_pass_of_the_union() -> None:
    first = _timed([5.0, 0.0], [9.0, 8.0], "A")
    second = _timed([0.0, 5.0], [1.0, 3.0], "B")
    params = _first_pass_params(order_col="stim")
    pool = ["sequence"]

    parts = [prepare_rows(table, params, pool, who=WHO) for table in (first, second)]
    pooled = reduce_prepared(combine_prepared(parts), params, pairs="each_perspective")

    inline = _first_pass(pd.concat([first, second], ignore_index=True), params, pool)
    pd.testing.assert_frame_equal(pooled, inline, check_exact=True)
    np.testing.assert_array_equal(pooled["value_first_pass"], [3.0])


def test_a_blank_reduction_writes_nan() -> None:
    table = _timed([5.0], [1.0])

    out = blank_reduction(table, _first_pass_params(order_col="stim"), KEYS)

    assert out["value_first_pass"].dtype == np.float64
    np.testing.assert_array_equal(out["value_first_pass"], [np.nan])


def test_a_missing_order_column_is_refused_by_name() -> None:
    table = _timed([5.0], [1.0]).drop(columns=["stim"])

    with pytest.raises(ValueError, match="order_col.*stim"):
        _ = _first_pass(table, _first_pass_params(order_col="stim"))


def test_a_text_order_column_is_refused_by_name() -> None:
    table = _timed([5.0, 5.0], [1.0, 2.0])
    table["stim"] = ["a", "b"]

    with pytest.raises(ValueError, match="order_col.*stim"):
        _ = _first_pass(table, _first_pass_params(order_col="stim"))


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_a_boolean_order_column_is_refused_by_name(dtype: str) -> None:
    table = _timed([5.0, 5.0], [1.0, 2.0])
    table["stim"] = pd.Series([True, False], dtype=dtype)

    with pytest.raises(ValueError, match="order_col.*stim"):
        _ = _first_pass(table, _first_pass_params(order_col="stim"))


def test_first_pass_without_a_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="first_pass.*threshold"):
        _ = AggregateParams(columns=["value"], aggs=["first_pass"])


def test_an_order_column_other_than_time_without_first_pass_is_refused() -> None:
    with pytest.raises(ValueError, match="order_col.*first_pass"):
        _ = AggregateParams(columns=["value"], order_col="stim")


def test_the_default_order_column_without_first_pass_is_accepted() -> None:
    params = AggregateParams(columns=["value"], order_col="time")

    assert params.order_col == "time"


def test_the_order_column_leaves_the_identity_dump_while_it_is_time() -> None:
    plain = _first_pass_params()
    named = _first_pass_params(order_col="time")
    moved = _first_pass_params(order_col="stim")

    assert "order_col" not in plain.identity_dump()
    assert named.identity_dump() == plain.identity_dump()
    assert moved.identity_dump()["order_col"] == "stim"


def _two_subgroups() -> pd.DataFrame:
    """Eight rows over two subgroup columns, a null in each, and an arena."""
    return pd.DataFrame(
        {
            "group": "A",
            "sequence": "S",
            "arena": ["a", "a", "a", "a", "a", "a", "b", "b"],
            "tag": ["x", "x", "y", "y", "x", None, "x", "x"],
            "phase": [1.0, 2.0, 1.0, 2.0, np.nan, 1.0, 1.0, 1.0],
            "value": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        }
    )


def test_split_keys_appends_each_subgroup_column_after_the_arena_in_order() -> None:
    table = _two_subgroups()

    both = AggregateParams(columns=["value"], subgroup_col=["tag", "phase"])
    swapped = AggregateParams(columns=["value"], subgroup_col=["phase", "tag"])
    one = AggregateParams(columns=["value"], subgroup_col="tag")
    none = AggregateParams(columns=["value"])

    assert split_keys(table, both) == ["arena", "tag", "phase"]
    assert split_keys(table, swapped) == ["arena", "phase", "tag"]
    assert split_keys(table, one) == ["arena", "tag"]
    assert split_keys(table, none) == ["arena"]
    assert split_keys(table.drop(columns=["arena"]), both) == ["tag", "phase"]


def test_two_subgroup_columns_give_one_row_per_pair_of_values() -> None:
    table = _two_subgroups()
    params = AggregateParams(
        columns=["value"], aggs=["sum", "count"], subgroup_col=["tag", "phase"]
    )

    out = reduce_column(
        table,
        params,
        [*KEYS, *split_keys(table, params)],
        who=WHO,
        pairs="each_perspective",
    )

    # The row with a null tag and the row with a null phase are in no group.
    assert out[["arena", "tag", "phase"]].values.tolist() == [
        ["a", "x", 1.0],
        ["a", "x", 2.0],
        ["a", "y", 1.0],
        ["a", "y", 2.0],
        ["b", "x", 1.0],
    ]
    np.testing.assert_array_equal(out["value_sum"], [1.0, 2.0, 4.0, 8.0, 192.0])
    np.testing.assert_array_equal(out["value_count"], [1, 1, 1, 1, 2])


def test_a_null_in_either_subgroup_column_drops_the_row() -> None:
    table = _two_subgroups().iloc[[4, 5]]
    params = AggregateParams(
        columns=["value"], aggs=["count"], subgroup_col=["tag", "phase"]
    )

    out = reduce_column(
        table,
        params,
        [*KEYS, *split_keys(table, params)],
        who=WHO,
        pairs="each_perspective",
    )

    assert len(out) == 0


@pytest.mark.parametrize(
    "subgroup_col", ["absent", ["tag", "absent"], ["absent", "tag"]]
)
def test_a_subgroup_column_the_input_lacks_is_refused_by_name(
    subgroup_col: str | list[str],
) -> None:
    table = _two_subgroups()
    params = AggregateParams(columns=["value"], subgroup_col=subgroup_col)
    keys = [*KEYS, *params.subgroup_columns()]

    with pytest.raises(ValueError, match="subgroup_col column 'absent'"):
        _ = reduce_column(table, params, keys, who=WHO, pairs="each_perspective")


def test_a_one_item_list_validates_to_the_string() -> None:
    listed = AggregateParams(columns=["value"], subgroup_col=["tag"])

    assert listed.subgroup_col == "tag"


def test_a_repeated_subgroup_column_is_refused() -> None:
    with pytest.raises(ValueError, match="tag more than once"):
        _ = AggregateParams(columns=["value"], subgroup_col=["tag", "phase", "tag"])


def test_an_empty_subgroup_list_is_refused() -> None:
    with pytest.raises(ValueError, match="subgroup_col.*no column"):
        _ = AggregateParams(columns=["value"], subgroup_col=[])


def test_a_kept_name_that_is_any_subgroup_column_is_refused() -> None:
    with pytest.raises(ValueError, match="phase.*subgroup_col"):
        _ = AggregateParams(
            columns=["value"], subgroup_col=["tag", "phase"], keep_columns=["phase"]
        )


@pytest.mark.parametrize("name", ["bin_start", "bin_end"])
def test_a_bin_name_in_the_subgroup_list_is_refused_with_bin_s(name: str) -> None:
    with pytest.raises(ValueError, match=f"subgroup_col.*{name}"):
        _ = AggregateParams(columns=["value"], bin_s=60, subgroup_col=["tag", name])
    allowed = AggregateParams(columns=["value"], subgroup_col=["tag", name])
    assert allowed.subgroup_col == ["tag", name]


def test_the_subgroup_column_leaves_the_identity_dump_as_it_was() -> None:
    unset = AggregateParams(columns=["value"])
    string = AggregateParams(columns=["value"], subgroup_col="tag")
    one = AggregateParams(columns=["value"], subgroup_col=["tag"])
    two = AggregateParams(columns=["value"], subgroup_col=["tag", "phase"])
    swapped = AggregateParams(columns=["value"], subgroup_col=["phase", "tag"])

    assert one.identity_dump() == string.identity_dump()
    assert string.identity_dump()["subgroup_col"] == "tag"
    assert unset.identity_dump()["subgroup_col"] is None
    assert two.identity_dump()["subgroup_col"] == ["tag", "phase"]
    assert swapped.identity_dump() != two.identity_dump()


# --- subsets --------------------------------------------------------------------------


def _dyadic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two entries of two individuals whose values and times are exact in binary.

    Every value is a multiple of 1 / 64 and every time a multiple of 1 / 8, so a
    sum or a duration is the same in any order of the rows. A group-by sum of
    other floats can differ in the last place with the groups around it.
    """
    generator = np.random.default_rng(5)
    tables: list[pd.DataFrame] = []
    for name, first_id in (("A", 0), ("B", 2)):
        frames = np.tile(np.arange(20), 2)
        values = generator.integers(0, 64, size=40) / 64.0
        tables.append(
            pd.DataFrame(
                {
                    "group": name,
                    "sequence": "S",
                    "id": np.repeat([first_id, first_id + 1], 20),
                    "frame": frames,
                    "time": frames / 8,
                    "value": values,
                    "state": np.where(values > 0.5, "up", "down"),
                }
            )
        )
    return tables[0], tables[1]


def _by_separate_runs(
    table: pd.DataFrame,
    params: AggregateParams,
    keys: list[str],
    subsets: dict[str, str],
) -> pd.DataFrame:
    """One filtered run per subset, joined with the subset's name after the keys."""
    parts: list[pd.DataFrame] = []
    for name, expression in subsets.items():
        narrowed = params.model_copy(
            update={
                "subsets": None,
                "filter_expr": (
                    expression
                    if params.filter_expr is None
                    else f"({params.filter_expr}) and ({expression})"
                ),
            }
        )
        part = reduce_column(table, narrowed, keys, who=WHO, pairs="each_perspective")
        if not part.empty:
            part.insert(len(keys), "subset", name)
            parts.append(part)
    joined = pd.concat(parts, ignore_index=True)
    return joined.sort_values([*keys, "subset"], kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("bin_s", [None, 0.625])
@pytest.mark.parametrize("filter_expr", [None, "frame % 3 != 1"])
def test_overlapping_subsets_equal_separate_filtered_runs(
    bin_s: float | None, filter_expr: str | None
) -> None:
    first, second = _dyadic_tables()
    table = pd.concat([first, second], ignore_index=True)
    subsets = {
        "low": "value < 0.6",
        "high": "value > 0.4",
        "all": "value >= 0",
        "flat": "state == 'down'",
    }
    params = AggregateParams(
        columns=["value"],
        aggs=["sum", "count", "mean", "median", "duration"],
        filter_expr=filter_expr,
        bin_s=bin_s,
        subsets=subsets,
    )

    together = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")

    pd.testing.assert_frame_equal(
        together, _by_separate_runs(table, params, KEYS, subsets), check_exact=True
    )
    assert together["subset"].nunique() == len(subsets)


@pytest.mark.parametrize("bin_s", [None, 0.625])
def test_the_std_of_overlapping_subsets_matches_separate_runs_closely(
    bin_s: float | None,
) -> None:
    # Pandas computes std with a running method, so the rows of other groups
    # may move the last places.
    first, second = _tables()
    table = pd.concat([first, second], ignore_index=True)
    subsets = {"low": "value < 0.6", "high": "value > 0.4", "all": "value >= 0"}
    params = AggregateParams(
        columns=["value"], aggs=["std", "count"], bin_s=bin_s, subsets=subsets
    )

    together = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")
    separate = _by_separate_runs(table, params, KEYS, subsets)

    np.testing.assert_allclose(
        together["value_std"].to_numpy(), separate["value_std"].to_numpy(), rtol=1e-12
    )
    pd.testing.assert_frame_equal(
        together.drop(columns="value_std"),
        separate.drop(columns="value_std"),
        check_exact=True,
    )
    assert together["value_std"].notna().any()


def test_six_overlapping_speed_windows_equal_six_filtered_runs() -> None:
    generator = np.random.default_rng(11)
    first, second = _dyadic_tables()
    table = pd.concat([first, second], ignore_index=True)
    table["speed"] = generator.integers(0, 128, size=len(table)) / 64.0
    subsets = {
        f"window-{index}": f"speed >= {low} and speed < {low + 0.75}"
        for index, low in enumerate((0.0, 0.25, 0.5, 0.75, 1.0, 1.25))
    }
    params = AggregateParams(
        columns=["speed"],
        aggs=["mean", "min", "max", "count", "duration"],
        subsets=subsets,
    )
    keys = [*KEYS, "id"]

    together = reduce_column(table, params, keys, who=WHO, pairs="each_perspective")

    pd.testing.assert_frame_equal(
        together, _by_separate_runs(table, params, keys, subsets), check_exact=True
    )
    assert together["subset"].nunique() == 6


def test_duration_and_share_weigh_rows_by_their_seconds_before_both_filters() -> None:
    first, _ = _tables()
    params = AggregateParams(
        columns=["state"],
        aggs=["duration", "share"],
        filter_expr="frame % 2 == 0",
        subsets={"even-up": "state == 'up'", "even": "frame >= 0"},
    )

    out = reduce_column(first, params, KEYS, who=WHO, pairs="each_perspective")

    even = out[out["subset"] == "even"]
    # Ten kept frames of each of two individuals, each lasting the 0.1 s to its
    # next frame, not the 0.2 s to its next kept one.
    assert even["state_duration"].tolist() == [pytest.approx(2.0)]
    assert even["state_share_up"].iloc[0] + even["state_share_down"].iloc[
        0
    ] == pytest.approx(1.0)


def test_the_rows_do_not_depend_on_the_order_of_the_mapping() -> None:
    first, _ = _tables()
    forward = {"low": "value < 0.6", "high": "value > 0.4", "mid": "value > 0.3"}
    reverse = dict(reversed(forward.items()))

    def run(subsets: dict[str, str]) -> pd.DataFrame:
        params = AggregateParams(columns=["value"], aggs=["sum"], subsets=subsets)
        return reduce_column(first, params, KEYS, who=WHO, pairs="each_perspective")

    pd.testing.assert_frame_equal(run(forward), run(reverse), check_exact=True)
    assert run(forward)["subset"].tolist() == ["high", "low", "mid"]


def test_the_subset_key_follows_the_other_keys_and_precedes_the_bins() -> None:
    first, second = _tables()
    table = pd.concat([first, second], ignore_index=True)
    params = AggregateParams(
        columns=["value"],
        aggs=["count"],
        bin_s=0.55,
        subsets={"b": "value > 0.5", "a": "value <= 0.5"},
    )
    keys = [*KEYS, "state"]

    out = reduce_column(table, params, keys, who=WHO, pairs="each_perspective")

    names = ["group", "sequence", "state", "subset", "bin_start", "bin_end"]
    assert list(out.columns)[:6] == names
    pd.testing.assert_frame_equal(
        out, out.sort_values(names, kind="stable").reset_index(drop=True)
    )


def test_a_subset_that_keeps_no_row_has_no_row_in_the_reduction() -> None:
    first, _ = _tables()
    params = AggregateParams(
        columns=["value"],
        aggs=["sum"],
        subsets={"some": "value > 0.5", "none": "value > 100"},
    )

    out = reduce_column(first, params, KEYS, who=WHO, pairs="each_perspective")

    assert out["subset"].tolist() == ["some"]
    emptied = params.model_copy(update={"subsets": {"none": "value > 100"}})
    assert reduce_column(first, emptied, KEYS, who=WHO, pairs="each_perspective").empty


def test_a_kept_column_is_checked_within_each_subset() -> None:
    first, _ = _tables()
    first["label"] = np.where(first["value"] > 0.5, "p", "q")
    params = AggregateParams(
        columns=["value"],
        aggs=["count"],
        keep_columns=["label"],
        subsets={"p-only": "label == 'p'", "both": "value >= 0"},
    )

    with pytest.raises(ValueError, match="label"):
        _ = reduce_column(first, params, KEYS, who=WHO, pairs="each_perspective")
    only = params.model_copy(update={"subsets": {"p-only": "label == 'p'"}})
    out = reduce_column(first, only, KEYS, who=WHO, pairs="each_perspective")
    assert out["label"].tolist() == ["p"]


def test_entries_prepared_with_subsets_then_combined_equal_the_inline_reduction() -> (
    None
):
    first, second = _dyadic_tables()
    params = AggregateParams(
        columns=["state"],
        aggs=["count", "duration", "share"],
        bin_s=0.625,
        subsets={"low": "value < 0.6", "high": "value > 0.4"},
    )
    inline = reduce_column(
        pd.concat([first, second], ignore_index=True),
        params,
        ["sequence"],
        who=WHO,
        pairs="each_perspective",
    )
    parts = [prepare_rows(t, params, ["sequence"], who=WHO) for t in (first, second)]
    pooled = reduce_prepared(combine_prepared(parts), params, pairs="each_perspective")

    pd.testing.assert_frame_equal(pooled, inline, check_exact=True)
    assert parts[0].group_keys == ("sequence", "subset", "bin_start", "bin_end")
    assert set(pooled["subset"]) == {"low", "high"}


def test_a_part_prepared_without_subsets_cannot_join_one_prepared_with_them() -> None:
    first, second = _tables()
    plain = AggregateParams(columns=["value"])
    split = AggregateParams(columns=["value"], subsets={"all": "value >= 0"})

    with pytest.raises(ValueError, match="keys"):
        _ = combine_prepared(
            [
                prepare_rows(first, plain, KEYS, who=WHO),
                prepare_rows(second, split, KEYS, who=WHO),
            ]
        )


def test_an_input_with_a_subset_column_is_refused_when_subsets_is_set() -> None:
    first, _ = _tables()
    first["subset"] = "x"
    params = AggregateParams(columns=["value"], subsets={"all": "value >= 0"})

    with pytest.raises(ValueError, match="'subset'.*input"):
        _ = prepare_rows(first, params, KEYS, who=WHO)


def test_an_input_with_a_subset_column_is_read_as_any_column_without_subsets() -> None:
    first, _ = _tables()
    first["subset"] = np.where(first["value"] > 0.5, "x", "y")
    params = AggregateParams(columns=["value"], aggs=["count"])

    out = reduce_column(
        first, params, [*KEYS, "subset"], who=WHO, pairs="each_perspective"
    )

    assert out["subset"].tolist() == ["x", "y"]


@pytest.mark.parametrize("name", ["", "a b", "a.b", "a/b", "é"])
def test_a_subset_name_outside_letters_digits_underscores_hyphens_is_refused(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="letters, digits, underscores and hyphens"):
        _ = AggregateParams(columns=["value"], subsets={name: "value > 0"})


def test_a_subset_name_of_letters_digits_underscores_and_hyphens_is_accepted() -> None:
    params = AggregateParams(
        columns=["value"], subsets={"Fast_1-a": "value > 0", "9": "value < 0"}
    )

    assert params.subsets == {"Fast_1-a": "value > 0", "9": "value < 0"}


def test_no_subsets_listed_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _ = AggregateParams(columns=["value"], subsets={})


@pytest.mark.parametrize("expression", ["", "   "])
def test_a_subset_without_a_filter_is_refused(expression: str) -> None:
    with pytest.raises(ValueError, match="no filter"):
        _ = AggregateParams(columns=["value"], subsets={"all": expression})


def test_subset_is_refused_as_a_kept_or_subgroup_column_when_subsets_is_set() -> None:
    subsets = {"all": "value >= 0"}
    with pytest.raises(ValueError, match="keep_columns lists subset.*writes itself"):
        _ = AggregateParams(columns=["value"], subsets=subsets, keep_columns=["subset"])
    with pytest.raises(ValueError, match="subgroup_col.*subset"):
        _ = AggregateParams(columns=["value"], subsets=subsets, subgroup_col="subset")
    with pytest.raises(ValueError, match="subgroup_col.*subset"):
        _ = AggregateParams(
            columns=["value"], subsets=subsets, subgroup_col=["tag", "subset"]
        )


def test_subset_stays_an_ordinary_column_name_without_subsets() -> None:
    params = AggregateParams(
        columns=["value"], subgroup_col="subset", keep_columns=["phase"]
    )

    assert params.subgroup_col == "subset"


def test_subsets_leave_the_identity_dump_while_unset_and_enter_it_when_set() -> None:
    unset = AggregateParams(columns=["value"])
    named_none = AggregateParams(columns=["value"], subsets=None)
    one = AggregateParams(columns=["value"], subsets={"a": "value > 0"})
    other = AggregateParams(columns=["value"], subsets={"a": "value > 1"})

    assert "subsets" not in unset.identity_dump()
    assert named_none.identity_dump() == unset.identity_dump()
    assert one.identity_dump()["subsets"] == {"a": "value > 0"}
    assert one.identity_dump() != other.identity_dump()


# --- value bins -----------------------------------------------------------------------

_PI_EDGES = {"linspace": ["-pi", "pi", 9]}
_PI_ARRAY = np.linspace(-np.pi, np.pi, 9)
_STEP_EDGES = {"arange": [0.0, 3.5, 0.5]}
_STEP_ARRAY = np.arange(0.0, 3.5, 0.5)


def _values_on_the_edges(
    edges: NDArray[np.float64], generator: np.random.Generator, size: int = 400
) -> NDArray[np.float64]:
    """Random values beyond both ends, every edge twice, NaN and both infinities."""
    values = generator.uniform(edges[0] - 1.0, edges[-1] + 1.0, size)
    special = np.concatenate([edges, edges, [np.nan, np.nan, np.inf, -np.inf]])
    values[generator.choice(size, size=special.size, replace=False)] = special
    return values


def _scatter(**edges: NDArray[np.float64]) -> pd.DataFrame:
    """One entry whose column of each name takes values on and beyond its edges."""
    generator = np.random.default_rng(19)
    columns = {
        name: _values_on_the_edges(array, generator) for name, array in edges.items()
    }
    size = len(next(iter(columns.values())))
    return pd.DataFrame(
        {
            "group": "G",
            "sequence": "S",
            "id": 0,
            "frame": np.arange(size),
            "time": np.arange(size) / 8,
            "v": np.arange(size, dtype=float),
            **columns,
        }
    )


def _floats(table: pd.DataFrame, column: str) -> NDArray[np.float64]:
    """The values of one column of *table* as a float array."""
    return np.asarray(table[column], dtype=np.float64)


def _bins(column: str, edges: object, closed_last: bool = True) -> ValueBins:
    return ValueBins.model_validate(
        {"column": column, "edges": edges, "closed_last": closed_last}
    )


def _reduce(table: pd.DataFrame, params: AggregateParams) -> pd.DataFrame:
    return reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        pytest.param(_PI_EDGES, _PI_ARRAY, id="linspace"),
        pytest.param(_STEP_EDGES, _STEP_ARRAY, id="arange"),
    ],
)
def test_counts_equal_np_histogram_on_values_on_every_edge(
    edges: object, expected: NDArray[np.float64]
) -> None:
    table = _scatter(x=expected)
    counts, _ = np.histogram(_floats(table, "x"), bins=expected)
    params = AggregateParams(
        columns=["v"], aggs=["count"], value_bins=[_bins("x", edges)]
    )

    out = _reduce(table, params)

    assert out["x_bin"].tolist() == np.flatnonzero(counts).tolist()
    np.testing.assert_array_equal(out["v_count"].to_numpy(), counts[counts > 0])


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        pytest.param(_PI_EDGES, _PI_ARRAY, id="linspace"),
        pytest.param(_STEP_EDGES, _STEP_ARRAY, id="arange"),
    ],
)
def test_dense_counts_are_the_whole_np_histogram(
    edges: object, expected: NDArray[np.float64]
) -> None:
    table = _scatter(x=expected)
    x = _floats(table, "x")
    in_the_fourth = (x >= expected[3]) & (x < expected[4])
    table = table.iloc[np.flatnonzero(~in_the_fourth)].reset_index(drop=True)
    counts, _ = np.histogram(_floats(table, "x"), bins=expected)
    params = AggregateParams(
        columns=["v"],
        aggs=["count", "sum"],
        value_bins=[_bins("x", edges)],
        dense=True,
    )

    out = _reduce(table, params)

    assert out["x_bin"].tolist() == list(range(len(counts)))
    np.testing.assert_array_equal(out["v_count"].to_numpy(), counts)
    assert (out["v_sum"][counts == 0] == 0.0).all()
    assert counts[3] == 0


@pytest.mark.parametrize("dense", [False, True])
def test_two_binned_columns_count_as_np_histogram2d(dense: bool) -> None:
    table = _scatter(x=_PI_ARRAY, y=_STEP_ARRAY)
    counts, _, _ = np.histogram2d(
        _floats(table, "x"),
        _floats(table, "y"),
        bins=[_PI_ARRAY, _STEP_ARRAY],
    )
    params = AggregateParams(
        columns=["v"],
        aggs=["count"],
        value_bins=[_bins("x", _PI_EDGES), _bins("y", _STEP_EDGES)],
        dense=dense,
    )

    out = _reduce(table, params)

    assert list(out.columns)[:4] == ["group", "sequence", "x_bin", "y_bin"]
    where = np.nonzero(np.ones_like(counts) if dense else counts)
    assert out["x_bin"].tolist() == where[0].tolist()
    assert out["y_bin"].tolist() == where[1].tolist()
    np.testing.assert_array_equal(out["v_count"].to_numpy(), counts[where])


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        pytest.param(_PI_EDGES, _PI_ARRAY, id="linspace"),
        pytest.param(_STEP_EDGES, _STEP_ARRAY, id="arange"),
    ],
)
def test_half_open_counts_equal_np_digitize(
    edges: object, expected: NDArray[np.float64]
) -> None:
    table = _scatter(x=expected)
    digits = np.digitize(_floats(table, "x"), expected)
    counts = np.bincount(digits, minlength=len(expected) + 1)[1 : len(expected)]
    params = AggregateParams(
        columns=["v"],
        aggs=["count"],
        value_bins=[_bins("x", edges, closed_last=False)],
        dense=True,
    )
    closed = params.model_copy(update={"value_bins": [_bins("x", edges)]})

    out = _reduce(table, params)
    closing = _reduce(table, closed)

    np.testing.assert_array_equal(out["v_count"].to_numpy(), counts)
    on_last = int(np.count_nonzero(_floats(table, "x") == expected[-1]))
    assert on_last == 2
    assert closing["v_count"].iloc[-1] == counts[-1] + on_last


def test_each_generator_is_the_array_that_numpy_returns() -> None:
    linspace = _bins("x", {"linspace": ["-pi", "pi", 20]})
    arange = _bins("x", {"arange": ["-pi", "pi", 0.3]})
    scaled = _bins("x", {"arange": [0, "pi", 0.25]})
    listed = _bins("x", ["-pi", 0.0, "pi"])

    np.testing.assert_array_equal(linspace.edge_array(), np.linspace(-np.pi, np.pi, 20))
    np.testing.assert_array_equal(arange.edge_array(), np.arange(-np.pi, np.pi, 0.3))
    np.testing.assert_array_equal(scaled.edge_array(), np.arange(0, np.pi, 0.25))
    np.testing.assert_array_equal(listed.edge_array(), np.array([-np.pi, 0.0, np.pi]))
    assert not np.array_equal(linspace.edge_array(), np.linspace(-1, 1, 20) * np.pi)


def test_a_binned_column_is_also_reduced_from_the_same_rows() -> None:
    generator = np.random.default_rng(3)
    x = generator.integers(-8, 72, size=200) / 8.0
    table = pd.DataFrame(
        {"group": "G", "sequence": "S", "id": 0, "frame": np.arange(200), "x": x}
    )
    edges = [0.0, 2.0, 4.0, 6.0]
    params = AggregateParams(
        columns=["x"],
        aggs=["count", "sum", "mean", "min", "max"],
        value_bins=[_bins("x", edges, closed_last=False)],
    )

    out = _reduce(table, params)

    digits = np.digitize(x, edges)
    for k, count, total, mean, low, high in zip(
        out["x_bin"].tolist(),
        out["x_count"].tolist(),
        out["x_sum"].tolist(),
        out["x_mean"].tolist(),
        out["x_min"].tolist(),
        out["x_max"].tolist(),
        strict=True,
    ):
        inside = x[digits == k + 1]
        assert count == inside.size
        assert total == np.sum(inside)
        assert mean == np.mean(inside)
        assert (low, high) == (inside.min(), inside.max())
    assert out["x_bin"].tolist() == [0, 1, 2]


def test_a_row_outside_every_bin_belongs_to_no_group() -> None:
    table = _scatter(x=_STEP_ARRAY, y=_PI_ARRAY)
    params = AggregateParams(
        columns=["v"],
        aggs=["count"],
        value_bins=[_bins("x", _STEP_EDGES), _bins("y", _PI_EDGES)],
    )
    x = _floats(table, "x")
    y = _floats(table, "y")
    inside_x = (x >= _STEP_ARRAY[0]) & (x <= _STEP_ARRAY[-1])
    inside_y = (y >= _PI_ARRAY[0]) & (y <= _PI_ARRAY[-1])

    out = _reduce(table, params)

    assert out["v_count"].sum() == np.sum(inside_x & inside_y)


def _dense_tables() -> pd.DataFrame:
    first, second = _dyadic_tables()
    return pd.concat([first, second], ignore_index=True)


_VALUE_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("bin_s", [None, 0.625])
def test_dense_cells_are_the_sparse_rows_on_the_full_grid_of_each_present_group(
    bin_s: float | None,
) -> None:
    table = _dense_tables()
    sparse_params = AggregateParams(
        columns=["value"],
        aggs=["count", "sum", "duration", "mean", "median"],
        value_bins=[_bins("value", _VALUE_EDGES)],
        filter_expr="group == 'A' or value < 0.5",
        bin_s=bin_s,
    )
    dense_params = sparse_params.model_copy(update={"dense": True})

    sparse = _reduce(table, sparse_params)
    dense = _reduce(table, dense_params)

    other = ["group", "sequence"] + (["bin_start", "bin_end"] if bin_s else [])
    present = sparse[other].drop_duplicates()
    grid = (
        present.merge(pd.DataFrame({"value_bin": range(4)}), how="cross")
        .astype({"value_bin": "Int64"})
        .sort_values([*other[:2], "value_bin", *other[2:]], kind="stable")
    )
    names = ["group", "sequence", "value_bin", *other[2:]]
    expected = grid.merge(sparse, on=[*other, "value_bin"], how="left")
    for column in ("value_count", "value_sum", "value_duration"):
        expected[column] = expected[column].fillna(0).astype(sparse[column].dtype)
    expected = expected[list(dense.columns)].sort_values(names, kind="stable")

    pd.testing.assert_frame_equal(
        dense, expected.reset_index(drop=True), check_exact=True
    )
    assert dense["value_bin"].dtype == pd.Int64Dtype()
    empty = dense[dense["value_count"] == 0]
    assert not empty.empty
    assert empty["value_mean"].isna().all()
    assert empty["value_median"].isna().all()
    assert (empty["value_sum"] == 0.0).all()
    assert (empty["value_duration"] == 0.0).all()


def test_a_dense_group_with_no_row_in_any_bin_is_not_written() -> None:
    table = _dense_tables()
    params = AggregateParams(
        columns=["value"],
        aggs=["count"],
        value_bins=[_bins("value", [0.5, 1.0])],
        filter_expr="group == 'A' or value < 0.5",
        dense=True,
    )

    out = _reduce(table, params)

    assert set(out["group"]) == {"A"}


def test_a_kept_column_that_differs_between_bins_is_refused_with_dense() -> None:
    first, _ = _dyadic_tables()
    first["label"] = np.where(first["value"] >= 0.5, "p", "q")
    params = AggregateParams(
        columns=["value"],
        aggs=["count"],
        keep_columns=["label"],
        value_bins=[_bins("value", [0.0, 0.5, 1.0])],
    )

    assert _reduce(first, params)["label"].tolist() == ["q", "p"]
    with pytest.raises(ValueError, match="label"):
        _ = _reduce(first, params.model_copy(update={"dense": True}))


def test_dense_cells_of_a_kept_column_take_the_value_of_their_group() -> None:
    first, _ = _dyadic_tables()
    first["label"] = "same"
    params = AggregateParams(
        columns=["value"],
        aggs=["count"],
        keep_columns=["label"],
        value_bins=[_bins("value", [0.0, 0.5, 2.0, 3.0])],
        dense=True,
    )

    out = _reduce(first, params)

    assert out["value_count"].iloc[-1] == 0
    assert out["label"].tolist() == ["same"] * 3


def _blank_params(**overrides: object) -> AggregateParams:
    options: dict[str, object] = {
        "columns": ["value"],
        "aggs": ["count", "mean", "duration"],
        "value_bins": [_bins("value", _VALUE_EDGES)],
    }
    return AggregateParams.model_validate({**options, **overrides})


def test_a_blank_row_has_a_null_bin_of_the_dtype_of_a_binned_row() -> None:
    table = _dense_tables()
    params = _blank_params()

    blank = blank_reduction(table, params, KEYS)
    sparse = _reduce(table, params)

    assert list(blank.columns)[:3] == ["group", "sequence", "value_bin"]
    assert blank["value_bin"].isna().all()
    assert blank["value_bin"].dtype == sparse["value_bin"].dtype == pd.Int64Dtype()
    assert blank["group"].tolist() == ["A", "B"]


def test_the_dense_blank_rows_are_the_full_grid_of_each_group() -> None:
    table = _dense_tables()
    params = _blank_params(dense=True, bin_s=0.625)

    blank = blank_reduction(table, params, KEYS)

    assert blank["group"].tolist() == ["A"] * 4 + ["B"] * 4
    assert blank["value_bin"].tolist() == [0, 1, 2, 3] * 2
    assert blank["value_bin"].dtype == pd.Int64Dtype()
    assert blank["bin_start"].isna().all()
    assert (blank["value_count"] == 0).all()
    assert (blank["value_duration"] == 0.0).all()
    assert blank["value_mean"].isna().all()
    assert list(blank.columns)[:6] == [
        "group",
        "sequence",
        "value_bin",
        "bin_start",
        "bin_end",
        "value_count",
    ]


def test_an_entry_whose_bins_hold_no_row_is_its_blank_rows() -> None:
    table = _dense_tables()
    params = _blank_params(dense=True, filter_expr="value > 100")

    reduced = _reduce(table, params)
    out = blank_rows(table, params, KEYS, reduced)

    assert reduced.empty
    assert out["value_bin"].tolist() == [0, 1, 2, 3] * 2


def test_a_subset_that_keeps_no_row_is_the_full_grid_under_dense() -> None:
    table = _dense_tables()
    params = _blank_params(
        dense=True, subsets={"some": "value > 0.5", "none": "value > 100"}
    )

    reduced = _reduce(table, params)
    out = blank_rows(table, params, KEYS, reduced)

    names = ["group", "sequence", "subset", "value_bin"]
    pd.testing.assert_frame_equal(
        out[names], out[names].sort_values(names, kind="stable").reset_index(drop=True)
    )
    none = out[out["subset"] == "none"]
    assert none["value_bin"].tolist() == [0, 1, 2, 3] * 2
    assert (none["value_count"] == 0).all()
    assert out["value_bin"].dtype == pd.Int64Dtype()
    assert out["value_count"].dtype == reduced["value_count"].dtype


def test_the_bin_keys_follow_the_subset_and_precede_the_time_bins() -> None:
    table = _dense_tables()
    params = _blank_params(
        bin_s=0.625,
        subsets={"b": "value > 0.5", "a": "value <= 0.5"},
        value_bins=[_bins("value", _VALUE_EDGES), _bins("frame", [0, 10, 20])],
    )

    out = _reduce(table, params)

    names = ["group", "sequence", "subset", "value_bin", "frame_bin"]
    names += ["bin_start", "bin_end"]
    assert list(out.columns)[:7] == names
    pd.testing.assert_frame_equal(
        out, out.sort_values(names, kind="stable").reset_index(drop=True)
    )


def test_prepared_value_bins_entries_combine_to_the_inline_reduction() -> None:
    first, second = _dyadic_tables()
    for dense in (False, True):
        params = _blank_params(
            columns=["state"],
            aggs=["count", "duration", "share"],
            bin_s=0.625,
            dense=dense,
            value_bins=[_bins("value", _VALUE_EDGES)],
            subsets={"low": "value < 0.6", "high": "value > 0.4"},
        )
        inline = reduce_column(
            pd.concat([first, second], ignore_index=True),
            params,
            ["sequence"],
            who=WHO,
            pairs="each_perspective",
        )
        parts = [
            prepare_rows(t, params, ["sequence"], who=WHO) for t in (first, second)
        ]
        pooled = reduce_prepared(
            combine_prepared(parts), params, pairs="each_perspective"
        )

        pd.testing.assert_frame_equal(pooled, inline, check_exact=True)
        assert parts[0].group_keys == (
            "sequence",
            "subset",
            "value_bin",
            "bin_start",
            "bin_end",
        )


def test_a_part_prepared_without_value_bins_cannot_join_one_prepared_with_them() -> (
    None
):
    first, second = _tables()
    plain = AggregateParams(columns=["value"])
    binned = AggregateParams(columns=["value"], value_bins=[_bins("value", [0, 1])])

    with pytest.raises(ValueError, match="keys"):
        _ = combine_prepared(
            [
                prepare_rows(first, plain, KEYS, who=WHO),
                prepare_rows(second, binned, KEYS, who=WHO),
            ]
        )


def test_duration_and_share_weigh_rows_by_their_seconds_before_the_bins_drop_rows() -> (
    None
):
    table = _dense_tables()
    params = AggregateParams(
        columns=["state"],
        aggs=["duration", "share"],
        value_bins=[_bins("value", [0.0, 0.5], closed_last=False)],
    )
    narrowed = AggregateParams(
        columns=["state"], aggs=["duration", "share"], filter_expr="value < 0.5"
    )

    out = _reduce(table, params)
    expected = _reduce(table, narrowed)

    np.testing.assert_array_equal(
        out["state_duration"].to_numpy(), expected["state_duration"].to_numpy()
    )
    np.testing.assert_array_equal(
        out["state_share_down"].to_numpy(), expected["state_share_down"].to_numpy()
    )


def test_a_value_bins_column_the_input_lacks_is_refused_by_name() -> None:
    table = _dense_tables()
    params = AggregateParams(columns=["value"], value_bins=[_bins("nope", [0, 1])])

    with pytest.raises(
        ValueError, match="value_bins column 'nope' is not in the input"
    ):
        _ = _reduce(table, params)


@pytest.mark.parametrize("column", ["state", "flag"])
def test_a_text_or_boolean_value_bins_column_is_refused_by_name(column: str) -> None:
    table = _dense_tables()
    table["flag"] = table["value"] > 0.5
    params = AggregateParams(columns=["value"], value_bins=[_bins(column, [0, 1])])

    with pytest.raises(
        ValueError, match=f"value_bins column '{column}' must be numeric"
    ):
        _ = _reduce(table, params)


def test_a_nullable_integer_value_bins_column_drops_its_nulls() -> None:
    table = _dense_tables().iloc[:6].copy()
    table["n"] = pd.array([0, 1, None, 2, 1, 5], dtype="Int64")
    params = AggregateParams(
        columns=["value"],
        aggs=["count"],
        value_bins=[_bins("n", [0, 1, 2, 3])],
        dense=True,
    )

    out = _reduce(table, params)

    assert out["value_count"].tolist() == [1, 2, 1]


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([1.0], "at least two"),
        ([], "at least two"),
        ([0.0, float("nan"), 2.0], "finite"),
        ([0.0, float("inf")], "finite"),
        ([0.0, 2.0, 1.0], "strictly increase"),
        ([0.0, 1.0, 1.0], "strictly increase"),
        ({"linspace": [0, 1, 1]}, "at least 2"),
        ({"linspace": [0, 1, 2.5]}, "whole number"),
        ({"linspace": [0, 1]}, "three"),
        ({"linspace": [0, "pi", "pi"]}, "whole number"),
        ({"linspace": [1, 0, 4]}, "strictly increase"),
        ({"linspace": [0, float("inf"), 4]}, "finite"),
        ({"arange": [0, 1, 0]}, "step"),
        ({"arange": [0, 1, -0.25]}, "at least two"),
        ({"arange": [1, 0, 0.25]}, "at least two"),
        ({"arange": [0, 0.5, 1]}, "at least two"),
        ({"arange": [0, 1]}, "three"),
        ({"logspace": [0, 1, 4]}, "linspace"),
        ({"linspace": [0, 1, 4], "arange": [0, 1, 0.25]}, "one"),
        ({}, "one"),
        (["tau", 1.0], "pi"),
        ([True, 2.0], "number"),
        (["1.5", 2.0], "number"),
    ],
)
def test_edges_that_are_not_finite_increasing_numbers_are_refused(
    edges: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = ValueBins.model_validate({"column": "x", "edges": edges})


@pytest.mark.parametrize(
    "edges",
    [
        {"linspace": [0, 1, 1e12]},
        {"linspace": [0, 1, MAX_VALUE_BIN_EDGES + 1]},
        {"arange": [0, 100, 1e-9]},
        {"arange": [0, MAX_VALUE_BIN_EDGES + 1, 1]},
        {"arange": [0, 1e308, 1e-308]},
        {"arange": [-1e308, 1e308, 1e-300]},
        list(range(MAX_VALUE_BIN_EDGES + 1)),
    ],
)
def test_edges_beyond_the_bound_are_refused_before_numpy_builds_them(
    edges: object,
) -> None:
    with pytest.raises(ValueError, match=f"at most {MAX_VALUE_BIN_EDGES}"):
        _ = ValueBins.model_validate({"column": "x", "edges": edges})


def test_an_arange_just_past_the_bound_names_the_edges_numpy_would_build() -> None:
    stop = MAX_VALUE_BIN_EDGES + 0.000001
    built = len(np.arange(0, stop, 1))

    with pytest.raises(ValueError, match=f"gives {built} edges"):
        _ = ValueBins.model_validate({"column": "x", "edges": {"arange": [0, stop, 1]}})


@pytest.mark.parametrize(
    "edges",
    [
        {"linspace": [0, 1, MAX_VALUE_BIN_EDGES]},
        {"arange": [0, MAX_VALUE_BIN_EDGES, 1]},
        list(range(MAX_VALUE_BIN_EDGES)),
    ],
)
def test_the_bound_itself_is_accepted(edges: object) -> None:
    built = _bins("x", edges).edge_array()

    assert built.size == MAX_VALUE_BIN_EDGES


def test_a_value_bins_entry_with_an_unknown_key_is_refused() -> None:
    with pytest.raises(ValueError, match="labels"):
        _ = ValueBins.model_validate({"column": "x", "edges": [0, 1], "labels": "a"})


def test_a_column_binned_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="x.*more than once"):
        _ = AggregateParams(
            columns=["v"], value_bins=[_bins("x", [0, 1]), _bins("x", [0, 2])]
        )


def test_an_empty_value_bins_list_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _ = AggregateParams(columns=["v"], value_bins=[])


def test_dense_without_value_bins_is_refused() -> None:
    with pytest.raises(ValueError, match="dense.*value_bins"):
        _ = AggregateParams(columns=["v"], dense=True)
    assert AggregateParams(columns=["v"], dense=False).dense is False


def test_a_bin_name_is_refused_as_a_kept_or_subgroup_column_while_binned() -> None:
    bins = [_bins("x", [0, 1])]
    with pytest.raises(ValueError, match="keep_columns lists x_bin.*writes itself"):
        _ = AggregateParams(columns=["v"], value_bins=bins, keep_columns=["x_bin"])
    with pytest.raises(ValueError, match="subgroup_col is x_bin.*value_bins"):
        _ = AggregateParams(columns=["v"], value_bins=bins, subgroup_col="x_bin")
    with pytest.raises(ValueError, match="subgroup_col is x_bin.*value_bins"):
        _ = AggregateParams(
            columns=["v"], value_bins=bins, subgroup_col=["tag", "x_bin"]
        )


def test_a_bin_name_stays_an_ordinary_column_name_when_its_column_is_not_binned() -> (
    None
):
    bins = [_bins("x", [0, 1])]
    other = AggregateParams(
        columns=["v"], value_bins=bins, keep_columns=["y_bin"], subgroup_col="z_bin"
    )
    unset = AggregateParams(columns=["v"], keep_columns=["x_bin"], subgroup_col="y_bin")

    assert other.keep_columns == ["y_bin"]
    assert unset.keep_columns == ["x_bin"]


def test_value_bins_and_dense_leave_the_identity_dump_while_unset() -> None:
    unset = AggregateParams(columns=["v"])
    named_none = AggregateParams(columns=["v"], value_bins=None, dense=False)
    one = AggregateParams(columns=["v"], value_bins=[_bins("x", [0, 1])])
    dense = AggregateParams(columns=["v"], value_bins=[_bins("x", [0, 1])], dense=True)
    reopened = AggregateParams(columns=["v"], value_bins=[_bins("x", [0, 2])])
    closed = AggregateParams(
        columns=["v"], value_bins=[_bins("x", [0, 1], closed_last=False)]
    )
    ordered = AggregateParams(
        columns=["v"], value_bins=[_bins("x", [0, 1]), _bins("y", [0, 1])]
    )
    swapped = AggregateParams(
        columns=["v"], value_bins=[_bins("y", [0, 1]), _bins("x", [0, 1])]
    )

    for name in ("value_bins", "dense"):
        assert name not in unset.identity_dump()
    assert named_none.identity_dump() == unset.identity_dump()
    assert one.identity_dump()["value_bins"] == [
        {"column": "x", "edges": [0, 1], "closed_last": True}
    ]
    assert "dense" not in one.identity_dump()
    assert dense.identity_dump()["dense"] is True
    assert len({str(p.identity_dump()) for p in (one, dense, reopened, closed)}) == 4
    assert ordered.identity_dump() != swapped.identity_dump()


# --- circmean and circstd -------------------------------------------------------------


def _angles(count: int = 60, seed: int = 11) -> pd.DataFrame:
    """Two sequences' rows in an interleaved order, with random angles in radians."""
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "group": "G",
            "sequence": np.where(generator.random(count) < 0.5, "A", "B"),
            "id": 0,
            "frame": np.arange(count),
            "time": np.arange(count) / 10,
            "angle": generator.uniform(-np.pi, np.pi, count),
        }
    )


def _circular_params(**fields: object) -> AggregateParams:
    return AggregateParams.model_validate(
        {"columns": ["angle"], "aggs": ["circmean", "circstd"], **fields}
    )


def test_circmean_and_circstd_equal_the_scipy_calls_on_the_rows_in_input_order() -> (
    None
):
    table = _angles()

    out = reduce_column(
        table, _circular_params(), KEYS, who=WHO, pairs="each_perspective"
    )

    mean: list[float] = []
    spread: list[float] = []
    for sequence in ("A", "B"):
        values = table.loc[table["sequence"] == sequence, "angle"].to_numpy()
        mean.append(float(stats.circmean(values, high=np.pi, low=-np.pi)))
        spread.append(float(stats.circstd(values)))
    assert out["sequence"].tolist() == ["A", "B"]
    np.testing.assert_array_equal(out["angle_circmean"], mean)
    np.testing.assert_array_equal(out["angle_circstd"], spread)
    assert out["angle_circmean"].dtype == np.float64
    assert out["angle_circstd"].dtype == np.float64


def test_circular_statistics_read_the_rows_in_the_order_the_table_holds_them() -> None:
    table = _angles(count=200, seed=5)
    shuffled = table.sample(frac=1.0, random_state=3)
    params = _circular_params()

    out = reduce_column(shuffled, params, ["group"], who=WHO, pairs="each_perspective")

    values = shuffled["angle"].to_numpy()
    np.testing.assert_array_equal(
        out["angle_circmean"], [stats.circmean(values, high=np.pi, low=-np.pi)]
    )
    np.testing.assert_array_equal(out["angle_circstd"], [stats.circstd(values)])


def test_a_null_angle_is_skipped_and_an_all_null_group_is_nan() -> None:
    table = _angles(count=30, seed=2)
    table.loc[table["sequence"] == "B", "angle"] = np.nan
    table.loc[table.index[table["sequence"] == "A"][:3], "angle"] = np.nan

    out = reduce_column(
        table, _circular_params(), KEYS, who=WHO, pairs="each_perspective"
    )

    kept = table.loc[table["sequence"] == "A", "angle"].dropna().to_numpy()
    np.testing.assert_array_equal(
        out["angle_circmean"],
        [stats.circmean(kept, high=np.pi, low=-np.pi), np.nan],
    )
    np.testing.assert_array_equal(out["angle_circstd"], [stats.circstd(kept), np.nan])


def test_an_empty_group_does_not_call_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> float:
        raise AssertionError("scipy was called for an empty group")

    monkeypatch.setattr(stats, "circmean", refuse)
    monkeypatch.setattr(stats, "circstd", refuse)
    table = _angles(count=6)
    table["angle"] = pd.array([None] * 6, dtype="Float64")

    out = reduce_column(
        table, _circular_params(), ["group"], who=WHO, pairs="each_perspective"
    )

    np.testing.assert_array_equal(out["angle_circmean"], [np.nan])
    np.testing.assert_array_equal(out["angle_circstd"], [np.nan])


def test_a_nullable_column_gives_the_same_statistics_as_its_float_values() -> None:
    table = _angles(count=24, seed=9)
    nullable = table.assign(angle=table["angle"].astype("Float64"))
    params = _circular_params()

    plain = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")
    out = reduce_column(nullable, params, KEYS, who=WHO, pairs="each_perspective")

    pd.testing.assert_frame_equal(out, plain, check_exact=True)


def test_circmean_and_circstd_read_every_row_under_any_pairs_rule() -> None:
    generator = np.random.default_rng(4)
    angles = generator.uniform(-np.pi, np.pi, 8)
    table = pd.DataFrame(
        {
            "group": "G",
            "sequence": "S",
            "id1": 0,
            "id2": 1,
            "perspective": np.tile([0, 1], 4),
            "frame": np.repeat(np.arange(4), 2),
            "time": np.repeat(np.arange(4), 2) / 10,
            "angle": angles,
        }
    )
    params = _circular_params()

    for pairs in ("once", "each_perspective"):
        out = reduce_column(table, params, KEYS, who=WHO, pairs=pairs)
        np.testing.assert_array_equal(
            out["angle_circmean"],
            [stats.circmean(angles, high=np.pi, low=-np.pi)],
        )
        np.testing.assert_array_equal(out["angle_circstd"], [stats.circstd(angles)])


def test_a_pool_of_entries_reduces_the_rows_of_its_parts_in_their_order() -> None:
    first = _angles(count=20, seed=1).assign(group="A")
    second = _angles(count=25, seed=2).assign(group="B")
    params = _circular_params()
    pool = ["sequence"]

    parts = [prepare_rows(table, params, pool, who=WHO) for table in (first, second)]
    pooled = reduce_prepared(combine_prepared(parts), params, pairs="each_perspective")

    inline = reduce_column(
        pd.concat([first, second], ignore_index=True),
        params,
        pool,
        who=WHO,
        pairs="each_perspective",
    )
    pd.testing.assert_frame_equal(pooled, inline, check_exact=True)


def test_each_value_bin_takes_the_statistics_of_its_own_rows() -> None:
    table = _angles(count=40, seed=8)
    params = _circular_params(value_bins=[{"column": "time", "edges": [0.0, 2.0, 4.0]}])

    out = reduce_column(table, params, ["group"], who=WHO, pairs="each_perspective")

    low = table.loc[table["time"] < 2.0, "angle"].to_numpy()
    high = table.loc[
        (table["time"] >= 2.0) & (table["time"] <= 4.0), "angle"
    ].to_numpy()
    np.testing.assert_array_equal(
        out["angle_circmean"],
        [stats.circmean(part, high=np.pi, low=-np.pi) for part in (low, high)],
    )
    np.testing.assert_array_equal(
        out["angle_circstd"], [stats.circstd(part) for part in (low, high)]
    )


def test_overlapping_subsets_each_take_the_statistics_of_their_own_rows() -> None:
    table = _angles(count=50, seed=13)
    params = _circular_params(subsets={"early": "time < 3.0", "late": "time >= 2.0"})

    out = reduce_column(table, params, ["group"], who=WHO, pairs="each_perspective")

    early = table.loc[table["time"] < 3.0, "angle"].to_numpy()
    late = table.loc[table["time"] >= 2.0, "angle"].to_numpy()
    overlap = table.loc[(table["time"] >= 2.0) & (table["time"] < 3.0)]
    assert not overlap.empty
    assert out["subset"].tolist() == ["early", "late"]
    np.testing.assert_array_equal(
        out["angle_circmean"],
        [stats.circmean(part, high=np.pi, low=-np.pi) for part in (early, late)],
    )
    np.testing.assert_array_equal(
        out["angle_circstd"], [stats.circstd(part) for part in (early, late)]
    )


def test_circular_statistics_are_refused_with_a_threshold() -> None:
    for agg in ("circmean", "circstd"):
        with pytest.raises(ValueError, match=f"{agg}.*threshold"):
            _ = AggregateParams(columns=["angle"], aggs=[agg], threshold=0.5)


def test_circular_statistics_are_refused_with_the_absolute_transform() -> None:
    for agg in ("circmean", "circstd"):
        with pytest.raises(ValueError, match=f"{agg}.*transform"):
            _ = AggregateParams(columns=["angle"], aggs=[agg], transform="abs")


def test_other_aggregations_keep_a_threshold_and_a_transform() -> None:
    _ = AggregateParams(columns=["angle"], aggs=["mean"], threshold=0.5)
    _ = AggregateParams(columns=["angle"], aggs=["mean"], transform="abs")


def test_a_blank_reduction_writes_nan_for_each_circular_statistic() -> None:
    out = blank_reduction(_angles(count=4), _circular_params(), KEYS)

    for name in ("angle_circmean", "angle_circstd"):
        assert out[name].dtype == np.float64
        assert out[name].isna().all()


# --- quantiles ------------------------------------------------------------------------

METHODS: tuple[QuantileMethod, ...] = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)
QUANTILES = [0.0, 0.07, 0.1, 0.5, 0.995, 1.0]


def _samples(count: int = 61, seed: int = 21) -> pd.DataFrame:
    """Two sequences' rows in an interleaved order, with random values."""
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "group": "G",
            "sequence": np.where(generator.random(count) < 0.5, "A", "B"),
            "id": 0,
            "frame": np.arange(count),
            "time": np.arange(count) / 10,
            "value": generator.normal(size=count),
        }
    )


def _quantile_params(**fields: object) -> AggregateParams:
    return AggregateParams.model_validate(
        {"columns": ["value"], "aggs": ["quantile"], "quantiles": QUANTILES, **fields}
    )


def _name(q: float) -> str:
    return f"value_quantile_{quantile_text(q)}"


def test_the_methods_are_the_names_numpy_accepts() -> None:
    assert set(get_args(QuantileMethod.__value__)) == set(METHODS)
    for method in METHODS:
        _ = np.quantile(np.arange(5.0), 0.3, method=method)


@pytest.mark.parametrize(
    ("q", "text"),
    [
        (0.5, "50"),
        (1.0, "100"),
        (0.1, "10"),
        (0.995, "99_5"),
        (0.07, "7"),
        (0.95, "95"),
        (0.0, "0"),
        (-0.0, "0"),
        (0.25, "25"),
        (0.001, "0_1"),
        (1e-05, "0_001"),
    ],
)
def test_the_text_of_a_quantile_is_its_percent_without_trailing_zeros(
    q: float, text: str
) -> None:
    assert quantile_text(q) == text


def test_distinct_quantiles_reads_negative_zero_as_zero_and_names_the_field() -> None:
    assert distinct_quantiles([-0.0, 0.5], "f") == [0.0, 0.5]
    with pytest.raises(ValueError, match="^f lists 0.5 and 0.5"):
        distinct_quantiles([0.5, 0.5], "f")
    with pytest.raises(ValueError, match="^f lists 0.0 and 0.0"):
        distinct_quantiles([-0.0, 0.0], "f")


def test_quantiles_or_nan_gives_nan_for_no_values_and_numpys_otherwise() -> None:
    none = quantiles_or_nan(np.array([], dtype=np.float64), [0.1, 0.9], "linear")
    values = np.array([4.0, 1.0, 3.0])
    found = quantiles_or_nan(values, [0.1, 0.9], "higher")

    assert np.isnan(none).all()
    assert none.shape == (2,)
    np.testing.assert_array_equal(
        found, np.quantile(values, [0.1, 0.9], method="higher")
    )


@pytest.mark.parametrize("method", METHODS)
def test_each_quantile_equals_np_quantile_of_its_group_with_the_method(
    method: QuantileMethod,
) -> None:
    table = _samples()
    params = _quantile_params(quantile_method=method)

    out = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")

    assert out["sequence"].tolist() == ["A", "B"]
    assert list(out.columns) == [*KEYS, *(_name(q) for q in QUANTILES)]
    for q in QUANTILES:
        expected = [
            np.quantile(
                table.loc[table["sequence"] == sequence, "value"].to_numpy(),
                q,
                method=method,
            )
            for sequence in ("A", "B")
        ]
        np.testing.assert_array_equal(out[_name(q)], expected)
        assert out[_name(q)].dtype == np.float64


def test_an_unset_method_is_numpys_linear_method() -> None:
    table = _samples()

    out = reduce_column(
        table, _quantile_params(), ["group"], who=WHO, pairs="each_perspective"
    )

    values = table["value"].to_numpy()
    for q in QUANTILES:
        np.testing.assert_array_equal(out[_name(q)], [np.quantile(values, q)])


def test_quantile_columns_stand_at_the_place_of_the_aggregation() -> None:
    params = _quantile_params(aggs=["mean", "quantile", "max"], quantiles=[0.95, 0.5])

    out = reduce_column(_samples(), params, KEYS, who=WHO, pairs="each_perspective")

    assert list(out.columns) == [
        *KEYS,
        "value_mean",
        "value_quantile_95",
        "value_quantile_50",
        "value_max",
    ]


def test_a_null_value_is_skipped_and_an_all_null_group_is_nan() -> None:
    table = _samples(count=30, seed=2)
    table.loc[table["sequence"] == "B", "value"] = np.nan
    table.loc[table.index[table["sequence"] == "A"][:3], "value"] = np.nan

    out = reduce_column(
        table, _quantile_params(), KEYS, who=WHO, pairs="each_perspective"
    )

    kept = table.loc[table["sequence"] == "A", "value"].dropna().to_numpy()
    for q in QUANTILES:
        np.testing.assert_array_equal(out[_name(q)], [np.quantile(kept, q), np.nan])


def test_an_empty_group_does_not_call_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> float:
        raise AssertionError("np.quantile was called for an empty group")

    monkeypatch.setattr(np, "quantile", refuse)
    table = _samples(count=6)
    table["value"] = pd.array([None] * 6, dtype="Float64")

    out = reduce_column(
        table, _quantile_params(), ["group"], who=WHO, pairs="each_perspective"
    )

    for q in QUANTILES:
        np.testing.assert_array_equal(out[_name(q)], [np.nan])


def test_a_nullable_column_gives_the_same_quantiles_as_its_float_values() -> None:
    table = _samples(count=24, seed=9)
    nullable = table.assign(value=table["value"].astype("Float64"))
    params = _quantile_params(quantile_method="hazen")

    plain = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")
    out = reduce_column(nullable, params, KEYS, who=WHO, pairs="each_perspective")

    pd.testing.assert_frame_equal(out, plain, check_exact=True)


def test_an_integer_column_is_read_as_float64() -> None:
    table = _samples(count=20, seed=4)
    table["value"] = (table["value"] * 10).astype("int64")

    out = reduce_column(
        table, _quantile_params(), ["group"], who=WHO, pairs="each_perspective"
    )

    values = table["value"].to_numpy(dtype=np.float64)
    for q in QUANTILES:
        np.testing.assert_array_equal(out[_name(q)], [np.quantile(values, q)])


def test_quantiles_read_every_row_under_any_pairs_rule() -> None:
    generator = np.random.default_rng(4)
    values = generator.normal(size=8)
    table = pd.DataFrame(
        {
            "group": "G",
            "sequence": "S",
            "id1": 0,
            "id2": 1,
            "perspective": np.tile([0, 1], 4),
            "frame": np.repeat(np.arange(4), 2),
            "time": np.repeat(np.arange(4), 2) / 10,
            "value": values,
        }
    )
    params = _quantile_params(aggs=["quantile", "median", "count"])

    for pairs in ("once", "each_perspective"):
        out = reduce_column(table, params, KEYS, who=WHO, pairs=pairs)
        for q in QUANTILES:
            np.testing.assert_array_equal(out[_name(q)], [np.quantile(values, q)])
        np.testing.assert_array_equal(out["value_median"], [np.median(values)])
        assert out["value_count"].tolist() == [4 if pairs == "once" else 8]


def test_a_pool_of_entries_takes_the_quantiles_of_the_union() -> None:
    first = _samples(count=20, seed=1).assign(group="A")
    second = _samples(count=25, seed=2).assign(group="B")
    params = _quantile_params()
    pool = ["sequence"]

    parts = [prepare_rows(table, params, pool, who=WHO) for table in (first, second)]
    pooled = reduce_prepared(combine_prepared(parts), params, pairs="each_perspective")

    union = pd.concat([first, second], ignore_index=True)
    inline = reduce_column(union, params, pool, who=WHO, pairs="each_perspective")
    pd.testing.assert_frame_equal(pooled, inline, check_exact=True)
    for q in QUANTILES:
        expected = [
            np.quantile(union.loc[union["sequence"] == sequence, "value"], q)
            for sequence in ("A", "B")
        ]
        np.testing.assert_array_equal(pooled[_name(q)], expected)


def test_each_value_bin_takes_the_quantiles_of_its_own_rows() -> None:
    table = _samples(count=40, seed=8)
    params = _quantile_params(value_bins=[{"column": "time", "edges": [0.0, 2.0, 4.0]}])

    out = reduce_column(table, params, ["group"], who=WHO, pairs="each_perspective")

    low = table.loc[table["time"] < 2.0, "value"].to_numpy()
    high = table.loc[
        (table["time"] >= 2.0) & (table["time"] <= 4.0), "value"
    ].to_numpy()
    for q in QUANTILES:
        np.testing.assert_array_equal(
            out[_name(q)], [np.quantile(part, q) for part in (low, high)]
        )


def test_overlapping_subsets_each_take_the_quantiles_of_their_own_rows() -> None:
    table = _samples(count=50, seed=13)
    params = _quantile_params(subsets={"early": "time < 3.0", "late": "time >= 2.0"})

    out = reduce_column(table, params, ["group"], who=WHO, pairs="each_perspective")

    early = table.loc[table["time"] < 3.0, "value"].to_numpy()
    late = table.loc[table["time"] >= 2.0, "value"].to_numpy()
    assert out["subset"].tolist() == ["early", "late"]
    for q in QUANTILES:
        np.testing.assert_array_equal(
            out[_name(q)], [np.quantile(part, q) for part in (early, late)]
        )


def test_a_threshold_gives_the_quantiles_of_the_passing_indicator() -> None:
    table = _samples(count=30, seed=6)
    params = _quantile_params(threshold=0.0, quantiles=[0.25, 0.75])

    out = reduce_column(table, params, ["group"], who=WHO, pairs="each_perspective")

    passing = (table["value"].to_numpy() > 0.0).astype(np.float64)
    for q in (0.25, 0.75):
        np.testing.assert_array_equal(out[_name(q)], [np.quantile(passing, q)])


def test_a_blank_reduction_writes_nan_for_each_quantile() -> None:
    params = _quantile_params(aggs=["quantile", "count"], quantiles=[0.5, 0.995])

    out = blank_reduction(_samples(count=4), params, KEYS)

    assert list(out.columns) == [
        *KEYS,
        "value_quantile_50",
        "value_quantile_99_5",
        "value_count",
    ]
    for name in ("value_quantile_50", "value_quantile_99_5"):
        assert out[name].dtype == np.float64
        assert out[name].isna().all()
    assert (out["value_count"] == 0.0).all()


def test_quantiles_without_the_aggregation_are_refused() -> None:
    with pytest.raises(ValueError, match="quantiles.*aggs has no quantile"):
        _ = AggregateParams(columns=["value"], aggs=["mean"], quantiles=[0.5])


def test_the_aggregation_without_quantiles_is_refused() -> None:
    with pytest.raises(ValueError, match="quantile.*quantiles is not set"):
        _ = AggregateParams(columns=["value"], aggs=["quantile"])


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_a_quantile_outside_zero_to_one_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="quantiles"):
        _ = _quantile_params(quantiles=[0.5, bad])


def test_an_empty_list_of_quantiles_is_refused() -> None:
    with pytest.raises(ValueError, match="quantiles"):
        _ = _quantile_params(quantiles=[])


@pytest.mark.parametrize("repeated", [[0.5, 0.5], [0.0, -0.0], [0.1, 0.9, 0.1]])
def test_two_quantiles_with_one_text_are_refused(repeated: list[float]) -> None:
    with pytest.raises(ValueError, match="quantiles.*same column name"):
        _ = _quantile_params(quantiles=repeated)


def test_a_method_other_than_linear_without_the_aggregation_is_refused() -> None:
    with pytest.raises(ValueError, match="quantile_method.*aggs has no quantile"):
        _ = AggregateParams(columns=["value"], aggs=["mean"], quantile_method="hazen")


def test_the_linear_method_without_the_aggregation_is_the_default() -> None:
    params = AggregateParams(columns=["value"], aggs=["mean"], quantile_method="linear")

    assert (
        params.identity_dump()
        == AggregateParams(columns=["value"], aggs=["mean"]).identity_dump()
    )


def test_negative_zero_and_zero_are_one_quantile_in_the_identity_dump() -> None:
    negative = _quantile_params(quantiles=[-0.0, 0.5])
    positive = _quantile_params(quantiles=[0.0, 0.5])

    assert repr(negative.identity_dump()["quantiles"]) == "[0.0, 0.5]"
    assert negative.identity_dump() == positive.identity_dump()
    assert repr(negative.quantiles) == "[0.0, 0.5]"


def test_a_group_of_one_or_two_values_takes_the_scalar_quantiles() -> None:
    table = pd.DataFrame(
        {
            "group": "G",
            "sequence": ["A"] + ["B"] * 2 + ["C"] * 3,
            "value": [4.0, 1.0, 9.0, 2.0, 2.0, 5.0],
        }
    )
    for method in METHODS:
        params = _quantile_params(quantile_method=method)

        out = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")

        for q in QUANTILES:
            expected = [
                np.quantile(
                    table.loc[table["sequence"] == sequence, "value"].to_numpy(),
                    q,
                    method=method,
                )
                for sequence in ("A", "B", "C")
            ]
            np.testing.assert_array_equal(out[_name(q)], expected)


@pytest.mark.parametrize(
    ("kept", "aggs", "extra"),
    [
        ("x_mean", ["mean"], {}),
        ("x_count", ["mean", "count"], {}),
        ("x_median", ["median"], {}),
        ("x_quantile_50", ["quantile"], {"quantiles": [0.5]}),
        ("x_quantile_99_5", ["quantile"], {"quantiles": [0.1, 0.995]}),
        ("x_share_up", ["share"], {"categories": ["up", "down"]}),
    ],
)
def test_a_kept_column_named_like_an_aggregate_output_is_refused_by_the_params(
    kept: str, aggs: list[str], extra: dict[str, object]
) -> None:
    fields: dict[str, object] = {
        "columns": ["x"],
        "aggs": aggs,
        "keep_columns": [kept],
        **extra,
    }

    with pytest.raises(ValueError, match=f"keep_columns lists {kept}, which"):
        _ = AggregateParams.model_validate(fields)


def test_a_subgroup_column_named_like_an_aggregate_output_is_refused() -> None:
    with pytest.raises(ValueError, match="subgroup_col lists x_mean, which"):
        _ = AggregateParams(columns=["x"], aggs=["mean"], subgroup_col="x_mean")
    with pytest.raises(ValueError, match="subgroup_col lists x_mean, which"):
        _ = AggregateParams(
            columns=["x"], aggs=["mean"], subgroup_col=["tag", "x_mean"]
        )


def test_an_entry_that_keeps_no_row_cannot_overwrite_a_kept_column_either() -> None:
    """The refusal is the params', so no blank row is written over a kept column."""
    with pytest.raises(ValueError, match="keep_columns lists x_mean"):
        _ = AggregateParams(
            columns=["x"],
            aggs=["mean"],
            keep_columns=["x_mean"],
            filter_expr="x > 100",
        )


def test_names_that_no_output_takes_may_be_kept() -> None:
    params = AggregateParams(
        columns=["x"],
        aggs=["quantile", "count"],
        quantiles=[0.5],
        keep_columns=["x_quantile_90", "x_mean", "x_share_up", "y_count"],
    )

    assert params.keep_columns == ["x_quantile_90", "x_mean", "x_share_up", "y_count"]


def test_a_kept_column_named_like_a_share_column_of_the_data_is_refused() -> None:
    table = pd.DataFrame(
        {
            "group": "G",
            "sequence": ["A", "A", "B"],
            "time": [0.0, 0.1, 0.0],
            "frame": [0, 1, 0],
            "x": ["up", "down", "up"],
            "x_share_up": ["k", "k", "j"],
        }
    )
    params = AggregateParams(columns=["x"], aggs=["share"], keep_columns=["x_share_up"])

    with pytest.raises(ValueError, match="the column 'x_share_up' would be written by"):
        _ = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="quantile_method"):
        _ = _quantile_params(quantile_method="pandas")


def test_a_quantile_column_that_a_share_column_also_names_is_refused() -> None:
    table = _samples(count=8).assign(x="y_quantile_50", x_share_y="a")
    params = AggregateParams(
        columns=["x", "x_share_y"],
        aggs=["share", "quantile"],
        quantiles=[0.5],
    )

    with pytest.raises(ValueError, match="x_share_y_quantile_50"):
        _ = reduce_column(table, params, KEYS, who=WHO, pairs="each_perspective")


def test_the_quantile_options_leave_the_identity_dump_while_unset_or_linear() -> None:
    plain = AggregateParams(columns=["value"], aggs=["quantile"], quantiles=[0.5])
    linear = AggregateParams(
        columns=["value"],
        aggs=["quantile"],
        quantiles=[0.5],
        quantile_method="linear",
    )
    hazen = AggregateParams(
        columns=["value"],
        aggs=["quantile"],
        quantiles=[0.5],
        quantile_method="hazen",
    )
    unset = AggregateParams(columns=["value"], aggs=["mean"])

    assert "quantiles" not in unset.identity_dump()
    assert "quantile_method" not in unset.identity_dump()
    assert plain.identity_dump()["quantiles"] == [0.5]
    assert "quantile_method" not in plain.identity_dump()
    assert linear.identity_dump() == plain.identity_dump()
    assert hazen.identity_dump()["quantile_method"] == "hazen"


def test_repeated_items_lists_each_repeat_once_in_the_order_it_is_reached() -> None:
    assert repeated_items(["b", "a", "b", "a", "a", "c"]) == ["b", "a"]


def test_repeated_items_of_distinct_items_is_empty() -> None:
    assert repeated_items(iter([3, 1, 2])) == []
