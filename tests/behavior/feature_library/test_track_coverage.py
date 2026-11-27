"""Tests for the TrackCoverage feature: per-entry counts of frames and gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.aggregate_common import QuantileMethod
from mosaic.behavior.feature_library.track_coverage import TrackCoverage

NAN = float("nan")

type Values = dict[str, float]
"""The columns of one row, by name."""

type Spec = dict[int, dict[int, Values]]
"""The rows of one table: frame, then id, then the row's values."""

_COLUMNS = ("X", "Y", "poseX0", "poseY0")


def _values(x: float, y: float, px: float, py: float) -> Values:
    return dict(zip(_COLUMNS, (x, y, px, py), strict=True))


_ALL_FINITE = _values(1.0, 1.0, 1.0, 1.0)
_ALL_NAN = _values(NAN, NAN, NAN, NAN)

_SPEC: Spec = {
    0: {1: _ALL_FINITE, 2: _values(NAN, 1.0, 1.0, 1.0)},
    1: {1: _ALL_NAN, 2: _ALL_NAN},
    2: {1: _values(NAN, NAN, 1.0, 1.0), 2: _ALL_NAN},
    5: {1: _ALL_FINITE, 2: _ALL_FINITE},
    6: {1: _values(1.0, 1.0, NAN, NAN)},
    8: {1: _ALL_FINITE, 2: _ALL_FINITE},
    9: {1: _ALL_NAN, 2: _ALL_NAN},
    10: {1: _ALL_NAN, 2: _ALL_NAN},
    11: {1: _ALL_NAN},
}
"""Twelve frames, 0 to 11.

Frames 3, 4 and 7 have no row, and frame 6 has no row of id 2.

By rule ``xy`` (X and Y, all) the frames where no id passes are 1 to 4, 7 and 9
to 11, so the gaps are 4, 1 and 3.
"""


def _table(spec: Spec, **extra: float | str) -> pd.DataFrame:
    """The rows of *spec* as a tracks table of one entry."""
    rows = [
        {"group": "G", "sequence": "S1", "frame": frame, "id": identity, **values}
        for frame, by_id in spec.items()
        for identity, values in by_id.items()
    ]
    table = pd.DataFrame(rows)
    for name, value in extra.items():
        table[name] = value
    return table


def _passes(values: Values, columns: tuple[str, ...], mode: str) -> bool:
    finite = [not np.isnan(values[name]) for name in columns]
    return all(finite) if mode == "all" else any(finite)


def _runs(flags: list[bool]) -> list[int]:
    """The lengths of the runs of true flags, found by walking the list."""
    lengths: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _expected_rule(
    spec: Spec,
    span: range,
    ids: set[int],
    columns: tuple[str, ...],
    mode: str,
    quantiles: dict[str, float],
    method: QuantileMethod,
    thresholds: list[int],
) -> dict[str, object]:
    """The values of one rule's columns, counted frame by frame from *spec*."""
    flags_all: list[bool] = []
    flags_none: list[bool] = []
    for frame in span:
        passing = {
            identity
            for identity, values in spec.get(frame, {}).items()
            if _passes(values, columns, mode)
        }
        flags_all.append(passing == ids)
        flags_none.append(not passing)
    gaps = np.array(_runs(flags_none), dtype=np.int64)
    out: dict[str, object] = {
        "frames_all": sum(flags_all),
        "frames_none": sum(flags_none),
        "n_gaps": int(gaps.size),
        "gap_mean": float(np.mean(gaps)) if gaps.size else NAN,
        "gap_max": int(gaps.max()) if gaps.size else 0,
    }
    for text, q in quantiles.items():
        out[f"gap_quantile_{text}"] = (
            float(np.quantile(gaps.astype(np.float64), q, method=method))
            if gaps.size
            else NAN
        )
    for t in thresholds:
        out[f"gaps_over_{t}"] = int((gaps > t).sum())
    return out


def _same(actual: object, expected: object) -> bool:
    """Whether two scalars are equal bit for bit, NaN equal to NaN."""
    if isinstance(expected, float) and np.isnan(expected):
        return isinstance(actual, float) and bool(np.isnan(actual))
    return actual == expected


def _assert_row(row: pd.Series, expected: dict[str, object]) -> None:
    for name, value in expected.items():
        assert _same(row[name], value), (name, row[name], value)


def test_every_column_counts_the_frames_of_the_span_for_each_rule() -> None:
    rules = [
        {"name": "xy", "columns": ["X", "Y"], "mode": "all"},
        {"name": "pose", "columns": ["poseX0", "poseY0"], "mode": "any"},
        {"name": "either", "columns": ["X", "Y"], "mode": "any"},
    ]
    out = TrackCoverage(params={"rules": rules}).apply(_table(_SPEC))

    assert len(out) == 1
    row = out.iloc[0]
    assert row["group"] == "G"
    assert row["sequence"] == "S1"
    assert row["n_ids"] == 2
    assert row["n_frames"] == 12
    assert row["n_frames_with_rows"] == 9
    for rule in rules:
        columns = tuple(rule["columns"])
        expected = _expected_rule(
            _SPEC, range(12), {1, 2}, columns, str(rule["mode"]), {}, "linear", []
        )
        _assert_row(row, {f"{rule['name']}_{k}": v for k, v in expected.items()})
    assert (row["xy_frames_all"], row["xy_frames_none"]) == (2, 8)
    assert (row["xy_n_gaps"], row["xy_gap_max"], row["xy_gap_mean"]) == (3, 4, 8 / 3)


def test_the_default_rule_is_x_and_y_over_all_ids() -> None:
    out = TrackCoverage().apply(_table(_SPEC))

    expected = _expected_rule(
        _SPEC, range(12), {1, 2}, ("X", "Y"), "all", {}, "linear", []
    )
    assert list(out.columns)[:6] == [
        "group",
        "sequence",
        "n_ids",
        "n_frames",
        "n_frames_with_rows",
        "xy_frames_all",
    ]
    _assert_row(out.iloc[0], {f"xy_{k}": v for k, v in expected.items()})


def test_the_columns_come_in_the_documented_order_with_their_dtypes() -> None:
    params: dict[str, object] = {
        "rules": [
            {"name": "a", "columns": ["X", "Y"], "mode": "all"},
            {"name": "b", "columns": ["poseX0"], "mode": "any"},
        ],
        "gap_quantiles": [0.5, 0.95],
        "gap_thresholds": [1, 3],
    }
    out = TrackCoverage(params=params).apply(_table(_SPEC, time=0.1))

    def rule(name: str) -> list[str]:
        return [
            f"{name}_frames_all",
            f"{name}_frames_none",
            f"{name}_n_gaps",
            f"{name}_gap_mean",
            f"{name}_gap_max",
            f"{name}_gap_quantile_50",
            f"{name}_gap_quantile_95",
            f"{name}_gaps_over_1",
            f"{name}_gaps_over_3",
        ]

    assert list(out.columns) == [
        "group",
        "sequence",
        "n_ids",
        "n_frames",
        "n_frames_with_rows",
        *rule("a"),
        *rule("b"),
        "time_step_median",
        "time_step_max",
    ]
    for name in ("n_ids", "n_frames", "n_frames_with_rows"):
        assert out[name].dtype == np.int64
    for name in ("a", "b"):
        for suffix in ("frames_all", "frames_none", "n_gaps", "gap_max"):
            assert out[f"{name}_{suffix}"].dtype == np.int64
        for suffix in ("gap_mean", "gap_quantile_50", "gap_quantile_95"):
            assert out[f"{name}_{suffix}"].dtype == np.float64
        for suffix in ("gaps_over_1", "gaps_over_3"):
            assert out[f"{name}_{suffix}"].dtype == np.int64
    assert out["time_step_median"].dtype == np.float64
    assert out["time_step_max"].dtype == np.float64


def test_an_entry_without_gaps_writes_zero_counts_and_nan_statistics() -> None:
    spec: Spec = {f: {1: _ALL_FINITE, 2: _ALL_FINITE} for f in range(4)}
    params: dict[str, object] = {"gap_quantiles": [0.5], "gap_thresholds": [0, 5]}

    out = TrackCoverage(params=params).apply(_table(spec))

    row = out.iloc[0]
    assert row["xy_frames_all"] == 4
    assert row["xy_frames_none"] == 0
    assert row["xy_n_gaps"] == 0
    assert np.isnan(row["xy_gap_mean"])
    assert row["xy_gap_max"] == 0
    assert np.isnan(row["xy_gap_quantile_50"])
    assert row["xy_gaps_over_0"] == 0
    assert row["xy_gaps_over_5"] == 0
    assert out["xy_gap_max"].dtype == np.int64
    assert out["xy_gap_mean"].dtype == np.float64


def test_quantile_columns_are_named_by_the_quantile_text() -> None:
    quantiles = {"0": 0.0, "50": 0.5, "99_5": 0.995, "100": 1.0}
    params: dict[str, object] = {"gap_quantiles": list(quantiles.values())}

    out = TrackCoverage(params=params).apply(_table(_SPEC))

    assert [c for c in out.columns if "quantile" in c] == [
        f"xy_gap_quantile_{text}" for text in quantiles
    ]
    expected = _expected_rule(
        _SPEC, range(12), {1, 2}, ("X", "Y"), "all", quantiles, "linear", []
    )
    _assert_row(out.iloc[0], {f"xy_{k}": v for k, v in expected.items()})
    assert out.iloc[0]["xy_gap_quantile_100"] == 4.0
    assert out.iloc[0]["xy_gap_quantile_0"] == 1.0


def test_the_quantile_method_reaches_numpy() -> None:
    quantiles = {"50": 0.5, "90": 0.9}
    params: dict[str, object] = {
        "gap_quantiles": list(quantiles.values()),
        "quantile_method": "higher",
    }

    out = TrackCoverage(params=params).apply(_table(_SPEC))

    expected = _expected_rule(
        _SPEC, range(12), {1, 2}, ("X", "Y"), "all", quantiles, "higher", []
    )
    _assert_row(out.iloc[0], {f"xy_{k}": v for k, v in expected.items()})
    linear = _expected_rule(
        _SPEC, range(12), {1, 2}, ("X", "Y"), "all", quantiles, "linear", []
    )
    assert linear["gap_quantile_90"] != expected["gap_quantile_90"]


def test_a_threshold_counts_the_gaps_strictly_longer_than_it() -> None:
    thresholds = [0, 1, 2, 3, 4, 10]

    out = TrackCoverage(params={"gap_thresholds": thresholds}).apply(_table(_SPEC))

    row = out.iloc[0]
    assert [row[f"xy_gaps_over_{t}"] for t in thresholds] == [3, 2, 2, 1, 0, 0]
    expected = _expected_rule(
        _SPEC, range(12), {1, 2}, ("X", "Y"), "all", {}, "linear", thresholds
    )
    _assert_row(row, {f"xy_{k}": v for k, v in expected.items()})


def test_a_gap_at_either_edge_of_the_span_counts() -> None:
    spec: Spec = {
        0: {1: _ALL_NAN},
        1: {1: _ALL_FINITE},
        2: {1: _ALL_FINITE},
        3: {1: _ALL_NAN},
        4: {1: _ALL_NAN},
    }

    row = TrackCoverage().apply(_table(spec)).iloc[0]

    assert (row["xy_n_gaps"], row["xy_gap_max"], row["xy_gap_mean"]) == (2, 2, 1.5)


def test_a_frame_without_a_row_has_no_id_passing_and_extends_a_gap() -> None:
    spec: Spec = {
        0: {1: _ALL_FINITE},
        3: {1: _ALL_FINITE},
        4: {1: _ALL_NAN},
        5: {1: _ALL_FINITE},
    }

    row = TrackCoverage().apply(_table(spec)).iloc[0]

    assert row["n_frames"] == 6
    assert row["n_frames_with_rows"] == 4
    assert row["xy_frames_all"] == 3
    assert row["xy_frames_none"] == 3
    assert (row["xy_n_gaps"], row["xy_gap_max"]) == (2, 2)


def test_an_id_with_no_row_in_a_frame_is_absent_there() -> None:
    spec: Spec = {0: {1: _ALL_FINITE, 2: _ALL_FINITE}, 1: {1: _ALL_FINITE}}

    row = TrackCoverage().apply(_table(spec)).iloc[0]

    assert row["n_ids"] == 2
    assert row["xy_frames_all"] == 1
    assert row["xy_frames_none"] == 0


def test_two_rows_of_one_id_in_a_frame_count_the_id_once() -> None:
    spec: Spec = {f: {1: _ALL_FINITE, 2: _ALL_FINITE} for f in (0, 1)}
    table = _table(spec)
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    row = TrackCoverage().apply(table).iloc[0]

    assert row["n_frames_with_rows"] == 2
    assert row["xy_frames_all"] == 2
    assert row["xy_frames_none"] == 0


def test_a_row_without_an_id_belongs_to_no_id() -> None:
    table = _table({0: {1: _ALL_FINITE}, 1: {1: _ALL_FINITE}})
    table["id"] = pd.array([1, None], dtype="Int64")

    row = TrackCoverage().apply(table).iloc[0]

    assert row["n_ids"] == 1
    assert row["n_frames_with_rows"] == 2
    assert row["xy_frames_all"] == 1
    assert row["xy_frames_none"] == 1


def test_a_missing_value_in_a_nullable_column_reads_as_absent() -> None:
    table = _table(_SPEC)
    table["X"] = table["X"].astype("Float64").mask(table["X"].isna(), pd.NA)

    out = TrackCoverage().apply(table)

    expected = _expected_rule(
        _SPEC, range(12), {1, 2}, ("X", "Y"), "all", {}, "linear", []
    )
    _assert_row(out.iloc[0], {f"xy_{k}": v for k, v in expected.items()})


def test_the_rows_of_each_arena_are_counted_apart_over_one_span() -> None:
    spec_a: Spec = {f: {1: _ALL_FINITE} for f in (0, 1, 2, 3)}
    spec_b: Spec = {2: {2: _ALL_FINITE, 3: _ALL_NAN}, 5: {2: _ALL_FINITE}}
    table = pd.concat(
        [_table(spec_b, arena="B"), _table(spec_a, arena="A")], ignore_index=True
    )

    out = TrackCoverage(params={"gap_thresholds": [1]}).apply(table)

    assert out["arena"].tolist() == ["A", "B"]
    assert list(out.columns)[:6] == [
        "group",
        "sequence",
        "arena",
        "n_ids",
        "n_frames",
        "n_frames_with_rows",
    ]
    assert out["n_frames"].tolist() == [6, 6]
    for arena, spec, ids, frames_with_rows in (
        ("A", spec_a, {1}, 4),
        ("B", spec_b, {2, 3}, 2),
    ):
        row = out[out["arena"] == arena].iloc[0]
        assert row["n_ids"] == len(ids)
        assert row["n_frames_with_rows"] == frames_with_rows
        expected = _expected_rule(
            spec, range(6), ids, ("X", "Y"), "all", {}, "linear", [1]
        )
        _assert_row(row, {f"xy_{k}": v for k, v in expected.items()})
    assert out[out["arena"] == "A"].iloc[0]["xy_gap_max"] == 2


def test_a_row_without_an_arena_is_refused_with_its_count() -> None:
    table = pd.concat(
        [_table({0: {1: _ALL_FINITE}}, arena="A"), _table({1: {1: _ALL_FINITE}})],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="track-coverage: 1 of 2 rows have no arena"):
        TrackCoverage().apply(table)


def test_an_arena_column_without_a_value_is_refused_with_its_count() -> None:
    table = _table(_SPEC)
    table["arena"] = pd.array([None] * len(table), dtype="object")

    with pytest.raises(
        ValueError, match=rf"track-coverage.*{len(table)} rows.*no arena"
    ):
        TrackCoverage().apply(table)


def test_a_table_without_an_arena_column_gives_one_row() -> None:
    assert len(TrackCoverage().apply(_table(_SPEC))) == 1
    assert "arena" not in TrackCoverage().apply(_table(_SPEC)).columns


def test_the_time_steps_are_the_differences_between_frames_that_have_rows() -> None:
    times = {0: 0.0, 1: 0.1, 2: 0.25, 5: 1.0, 6: 1.1, 8: 1.5, 9: 1.7, 10: 1.8, 11: 2.0}
    table = _table(_SPEC)
    table["time"] = table["frame"].map(times)
    table.loc[table.index[table["frame"] == 0][1], "time"] = 99.0

    row = TrackCoverage().apply(table).iloc[0]

    steps = np.diff(np.array([times[f] for f in sorted(times)], dtype=np.float64))
    assert row["time_step_median"] == np.median(steps)
    assert row["time_step_max"] == np.max(steps)
    assert row["time_step_max"] == 0.75


def test_the_time_steps_are_nan_with_fewer_than_two_frames_with_rows() -> None:
    table = _table({3: {1: _ALL_FINITE, 2: _ALL_FINITE}}, time=0.5)

    out = TrackCoverage().apply(table)

    assert np.isnan(out.iloc[0]["time_step_median"])
    assert np.isnan(out.iloc[0]["time_step_max"])
    assert out["time_step_median"].dtype == np.float64
    assert out.iloc[0]["n_frames"] == 1


def test_the_time_steps_are_per_arena() -> None:
    first = _table({0: {1: _ALL_FINITE}, 1: {1: _ALL_FINITE}}, arena="A")
    second = _table({0: {2: _ALL_FINITE}, 4: {2: _ALL_FINITE}}, arena="B")
    table = pd.concat([first, second], ignore_index=True)
    table["time"] = table["frame"] * 0.5

    out = TrackCoverage().apply(table)

    assert out["time_step_max"].tolist() == [0.5, 2.0]
    assert out["time_step_median"].tolist() == [0.5, 2.0]


def test_a_table_without_time_has_no_time_step_columns() -> None:
    assert not any(
        "time_step" in c for c in TrackCoverage().apply(_table(_SPEC)).columns
    )


def test_an_empty_table_gives_an_empty_table() -> None:
    assert TrackCoverage().apply(pd.DataFrame()).empty


def test_a_rule_column_the_table_lacks_is_refused_by_name() -> None:
    params: dict[str, object] = {
        "rules": [{"name": "pose", "columns": ["poseX9"], "mode": "any"}]
    }

    with pytest.raises(ValueError, match="track-coverage.*poseX9"):
        TrackCoverage(params=params).apply(_table(_SPEC))


def test_the_params_default_to_one_rule_over_x_and_y() -> None:
    params = TrackCoverage.Params()

    assert [rule.model_dump() for rule in params.rules] == [
        {"name": "xy", "columns": ["X", "Y"], "mode": "all"}
    ]
    assert params.gap_quantiles == []
    assert params.quantile_method == "linear"
    assert params.gap_thresholds == []


_REFUSED: list[tuple[dict[str, object], str]] = [
    (
        {
            "rules": [
                {"name": "a", "columns": ["X"], "mode": "all"},
                {"name": "a", "columns": ["Y"], "mode": "all"},
            ]
        },
        "listed twice",
    ),
    ({"rules": []}, "at least 1"),
    ({"rules": [{"name": "a b", "columns": ["X"], "mode": "all"}]}, "pattern"),
    ({"rules": [{"name": "", "columns": ["X"], "mode": "all"}]}, "pattern"),
    ({"rules": [{"name": "a", "columns": [], "mode": "all"}]}, "at least 1"),
    ({"rules": [{"name": "a", "columns": ["X"], "mode": "most"}]}, "mode"),
    ({"gap_quantiles": [1.5]}, "less than or equal to 1"),
    ({"gap_quantiles": [-0.1]}, "greater than or equal to 0"),
    ({"gap_quantiles": [0.5, 0.5]}, "same column name"),
    ({"gap_thresholds": [-1]}, "greater than or equal to 0"),
    ({"gap_thresholds": [2, 2]}, "more than once"),
    ({"quantile_method": "hazen"}, "gap_quantiles"),
    ({"quantile_method": "nope", "gap_quantiles": [0.5]}, "quantile_method"),
]


@pytest.mark.parametrize(("overrides", "message"), _REFUSED)
def test_the_params_refuse_what_would_name_a_column_twice_or_mean_nothing(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        TrackCoverage.Params.model_validate(overrides)


def test_a_method_with_quantiles_is_accepted() -> None:
    params = TrackCoverage.Params(gap_quantiles=[0.5], quantile_method="hazen")

    assert params.quantile_method == "hazen"


def test_negative_zero_and_zero_are_one_quantile() -> None:
    with pytest.raises(ValidationError, match="same column name"):
        TrackCoverage.Params(gap_quantiles=[-0.0, 0.0])
