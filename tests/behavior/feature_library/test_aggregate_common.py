"""Tests for the reduction that the aggregate features share."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.aggregate_common import (
    AGGREGATIONS,
    Aggregation,
    AggregateParams,
    combine_prepared,
    prepare_rows,
    reduce_column,
    reduce_prepared,
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
