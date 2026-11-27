"""Tests for the derive-columns feature: row arithmetic over a table's columns.

Each expected value is the reference formula written inline on the same arrays
and is compared bit for bit, with NaN positions equal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.derive_columns import DeriveColumns


def _table() -> pd.DataFrame:
    """Four rows of two individuals, with a NaN position and an out-of-order frame."""
    return pd.DataFrame(
        {
            "group": ["g", "g", "g", "g"],
            "sequence": ["s", "s", "s", "s"],
            "frame": [3, 1, 2, 0],
            "id": [1, 0, 1, 0],
            "speed": [1.5, 0.0, 7.25, np.nan],
            "pos_x": [50.0, 60.0, np.nan, 10.0],
            "pos_y": [50.0, 50.0, 20.0, 90.0],
            "x": [0.1, 0.25, -3.0, 1e-3],
        }
    )


def _derive(columns: list[dict[str, str]], df: pd.DataFrame) -> pd.DataFrame:
    return DeriveColumns(params={"columns": columns}).apply(df)


def _assert_bits(actual: pd.Series, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual.to_numpy(), expected)
    assert actual.to_numpy().dtype == expected.dtype


def test_a_scaled_speed_equals_the_formula_bit_for_bit() -> None:
    df = _table()
    out = _derive([{"name": "speed_m", "expr": "speed * 30 / 100"}], df)

    expected = (df["speed"].to_numpy() * np.float64(30)) / np.float64(100)
    _assert_bits(out["speed_m"], expected)


def test_a_disc_membership_with_a_null_rule_equals_the_where_form() -> None:
    df = _table()
    expression = (
        "nan if isnull(pos_x) | isnull(pos_y) else "
        "(0.5 if (pos_x - 50)**2 + (pos_y - 50)**2 < 25**2 else 0.0)"
    )
    out = _derive([{"name": "inside", "expr": expression}], df)

    px = df["pos_x"].to_numpy()
    py = df["pos_y"].to_numpy()
    distance = (px - 50.0) * (px - 50.0) + (py - 50.0) * (py - 50.0)
    expected = np.where(
        np.isnan(px) | np.isnan(py),
        np.nan,
        np.where(distance < 25.0 * 25.0, 0.5, 0.0),
    )
    _assert_bits(out["inside"], expected)


def test_a_percentage_equals_the_formula_bit_for_bit() -> None:
    df = _table()
    out = _derive([{"name": "x_pct", "expr": "x * 100"}], df)

    _assert_bits(out["x_pct"], df["x"].to_numpy() * np.float64(100))


def test_a_later_entry_reads_an_earlier_one() -> None:
    df = _table()
    out = _derive(
        [
            {"name": "scaled", "expr": "speed * 30"},
            {"name": "halved", "expr": "scaled / 2"},
        ],
        df,
    )

    scaled = df["speed"].to_numpy() * np.float64(30)
    _assert_bits(out["scaled"], scaled)
    _assert_bits(out["halved"], scaled / np.float64(2))


def test_the_output_keeps_the_input_rows_and_columns_then_the_derived_ones() -> None:
    df = _table()
    out = _derive(
        [
            {"name": "b_col", "expr": "x * 2"},
            {"name": "a_col", "expr": "x > 0.2"},
            {"name": "c_col", "expr": "frame // 2"},
        ],
        df,
    )

    assert list(out.columns) == [*df.columns, "b_col", "a_col", "c_col"]
    pd.testing.assert_frame_equal(out[list(df.columns)], df)
    assert out.index.equals(df.index)
    assert out["frame"].tolist() == [3, 1, 2, 0]


def test_a_derived_column_has_the_dtype_numpy_gives() -> None:
    df = _table()
    out = _derive(
        [
            {"name": "scaled", "expr": "x * 2"},
            {"name": "flag", "expr": "x > 0.2"},
        ],
        df,
    )

    assert out["scaled"].dtype == np.float64
    assert out["flag"].dtype == np.bool_


def test_a_name_read_before_a_later_entry_writes_it_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"later_col.*later"):
        DeriveColumns(
            params={
                "columns": [
                    {"name": "first", "expr": "later_col * 2"},
                    {"name": "later_col", "expr": "x * 2"},
                ]
            }
        )


@pytest.mark.parametrize("name", ["pi", "nan", "inf"])
def test_a_constant_name_is_refused(name: str) -> None:
    with pytest.raises(ValidationError, match="constant"):
        DeriveColumns(params={"columns": [{"name": name, "expr": "x * 2"}]})


def test_a_name_listed_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="listed twice"):
        DeriveColumns(
            params={
                "columns": [
                    {"name": "dup", "expr": "x * 2"},
                    {"name": "dup", "expr": "x * 3"},
                ]
            }
        )


@pytest.mark.parametrize(
    "name",
    ["group", "sequence", "frame", "time", "id", "id1", "id2", "perspective", "arena"],
)
def test_a_metadata_name_is_refused(name: str) -> None:
    with pytest.raises(ValidationError, match="metadata"):
        DeriveColumns(params={"columns": [{"name": name, "expr": "x * 2"}]})


@pytest.mark.parametrize("name", ["1speed", "speed m", "", "speed-m"])
def test_a_name_outside_the_identifier_pattern_is_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        DeriveColumns(params={"columns": [{"name": name, "expr": "x * 2"}]})


def test_an_expression_that_does_not_parse_is_refused_at_validation() -> None:
    with pytest.raises(ValidationError, match="x @ x"):
        DeriveColumns(params={"columns": [{"name": "bad", "expr": "x @ x"}]})


def test_no_columns_is_refused() -> None:
    with pytest.raises(ValidationError):
        DeriveColumns(params={"columns": []})


def test_an_unknown_key_of_an_entry_is_refused() -> None:
    with pytest.raises(ValidationError):
        DeriveColumns(params={"columns": [{"name": "a", "expr": "x", "dtype": "f"}]})


def test_a_name_the_input_already_has_is_refused_at_apply() -> None:
    with pytest.raises(ValueError, match=r"already has the column 'speed'"):
        _derive([{"name": "speed", "expr": "x * 2"}], _table())


def test_a_missing_column_fails_naming_the_expression_and_the_column() -> None:
    with pytest.raises(
        ValueError, match=r"nowhere \* 2.*nowhere|nowhere.*nowhere \* 2"
    ):
        _derive([{"name": "out", "expr": "nowhere * 2"}], _table())


def test_an_entry_reading_its_own_name_fails_at_apply() -> None:
    with pytest.raises(ValueError, match=r"reads the column 'out', which the table"):
        _derive([{"name": "out", "expr": "out * 2"}], _table())


def test_an_empty_table_gives_an_empty_table() -> None:
    out = _derive([{"name": "out", "expr": "x * 2"}], pd.DataFrame())

    assert out.empty


def test_the_columns_enter_the_identity_in_list_order() -> None:
    first = DeriveColumns(
        params={
            "columns": [
                {"name": "a_col", "expr": "x * 2"},
                {"name": "b_col", "expr": "x * 3"},
            ]
        }
    )
    swapped = DeriveColumns(
        params={
            "columns": [
                {"name": "b_col", "expr": "x * 3"},
                {"name": "a_col", "expr": "x * 2"},
            ]
        }
    )

    assert first.params.identity_dump() != swapped.params.identity_dump()
