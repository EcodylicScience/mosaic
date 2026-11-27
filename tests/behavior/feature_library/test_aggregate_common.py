"""Tests for the reduction that the aggregate features share."""

from __future__ import annotations

from typing import get_args

import operator
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.aggregate_common import (
    AGGREGATIONS,
    Aggregation,
    AggregateParams,
    ThresholdOp,
    blank_reduction,
    combine_prepared,
    prepare_rows,
    reduce_column,
    reduce_prepared,
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
    }
    assert values == {"median"}


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
def test_a_subset_name_of_other_characters_than_letters_digits_underscores_hyphens_is_refused(
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
