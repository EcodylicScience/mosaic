"""The row-expression evaluator: its accepted nodes, its refusals and its dtype rules.

Every expected value is the numpy call the expression names, written inline on
synthetic arrays that hold NaN, infinities, zeros and negative values, and
compared bit for bit.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from mosaic.core.expressions import (
    ExpressionError,
    ParsedExpression,
    evaluate,
    names_read,
    parse_expression,
)

if TYPE_CHECKING:
    from mosaic.core.expressions import ColumnValues

NAN = float("nan")
INF = float("inf")

A = np.array([NAN, INF, -INF, 0.0, -0.0, -3.5, 2.0, 7.25, 0.5, -1.0])
B = np.array([1.0, NAN, 2.0, 0.0, -2.0, -1.5, 0.0, INF, 0.5, -INF])
C = np.array([2.0, 1.0, NAN, 0.0, 3.0, -1.5, 5.0, 1.0, 0.25, -0.5])
UNIT = np.array([NAN, 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, INF, -INF, 0.25])
N_ROWS = len(A)
COLUMNS = {"a": A, "b": B, "c": C, "u": UNIT}


def run(
    text: str, columns: Mapping[str, ColumnValues] | None = None, n_rows: int = N_ROWS
) -> NDArray[np.generic]:
    return evaluate(
        parse_expression(text), COLUMNS if columns is None else columns, n_rows
    )


def same(actual: NDArray[np.generic], expected: NDArray[np.generic]) -> None:
    """Bit for bit: dtype, values with NaN positions, and the sign of zeros."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype.kind == "f":
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def reference(call: Callable[[], NDArray[np.generic]]) -> NDArray[np.generic]:
    with np.errstate(all="ignore"):
        return call()


BINARY_NODES: list[tuple[str, Callable[[], NDArray[np.generic]]]] = [
    ("a + b", lambda: np.add(A, B)),
    ("a - b", lambda: np.subtract(A, B)),
    ("a * b", lambda: np.multiply(A, B)),
    ("a / b", lambda: np.true_divide(A, B)),
    ("a // b", lambda: np.floor_divide(A, B)),
    ("a % b", lambda: np.remainder(A, B)),
    ("a ** b", lambda: np.power(A, B)),
    ("-a", lambda: np.negative(A)),
    ("+a", lambda: np.positive(A)),
    ("a < b", lambda: np.less(A, B)),
    ("a <= b", lambda: np.less_equal(A, B)),
    ("a > b", lambda: np.greater(A, B)),
    ("a >= b", lambda: np.greater_equal(A, B)),
    ("a == b", lambda: np.equal(A, B)),
    ("a != b", lambda: np.not_equal(A, B)),
    ("(a > 0) & (b > 0)", lambda: np.logical_and(A > 0.0, B > 0.0)),
    ("(a > 0) | (b > 0)", lambda: np.logical_or(A > 0.0, B > 0.0)),
    ("~(a > 0)", lambda: np.logical_not(A > 0.0)),
    ("a if a > b else b", lambda: np.where(A > B, A, B)),
    ("a + 1", lambda: np.add(A, np.float64(1.0))),
    (
        "2 * a - 0.5",
        lambda: np.subtract(np.multiply(np.float64(2.0), A), np.float64(0.5)),
    ),
    ("pi * a", lambda: np.multiply(np.pi, A)),
    ("nan + a", lambda: np.add(np.nan, A)),
    ("inf - a", lambda: np.subtract(np.inf, A)),
    ("abs(a)", lambda: np.absolute(A)),
    ("sqrt(a)", lambda: np.sqrt(A)),
    ("exp(a)", lambda: np.exp(A)),
    ("log(a)", lambda: np.log(A)),
    ("floor(a)", lambda: np.floor(A)),
    ("ceil(a)", lambda: np.ceil(A)),
    ("sin(a)", lambda: np.sin(A)),
    ("cos(a)", lambda: np.cos(A)),
    ("deg2rad(a)", lambda: np.deg2rad(A)),
    ("rad2deg(a)", lambda: np.rad2deg(A)),
    ("arccos(u)", lambda: np.arccos(UNIT)),
    ("arctan2(a, b)", lambda: np.arctan2(A, B)),
    ("minimum(a, b)", lambda: np.minimum(A, B)),
    ("maximum(a, b)", lambda: np.maximum(A, B)),
    ("isnull(a)", lambda: np.isnan(A)),
    ("notnull(a)", lambda: np.logical_not(np.isnan(A))),
    ("fillna(a, -9)", lambda: np.where(np.isnan(A), np.float64(-9.0), A)),
    ("fillna(a, b)", lambda: np.where(np.isnan(A), B, A)),
]


@pytest.mark.parametrize(
    ("text", "expected"), BINARY_NODES, ids=[text for text, _ in BINARY_NODES]
)
def test_an_accepted_node_equals_the_numpy_call_it_names(
    text: str, expected: Callable[[], NDArray[np.generic]]
) -> None:
    same(run(text), reference(expected))


def test_operations_run_in_the_order_of_the_syntax_tree() -> None:
    speed = np.array([1.0, 3.0, 7.0, 11.0, 0.1, 123.456, 0.3, 77.7])
    got = run("speed * 30 / 100", {"speed": speed}, len(speed))
    same(got, (speed * np.float64(30.0)) / np.float64(100.0))
    assert not np.array_equal(got, speed * np.float64(30.0 / 100.0))


def test_a_division_by_zero_gives_what_numpy_computes_without_a_warning() -> None:
    zero = np.zeros(3)
    column = np.array([1.0, -1.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run("x / z", {"x": column, "z": zero}, 3)
        run("x // z + x % z + log(z) + sqrt(-x)", {"x": column, "z": zero}, 3)
    np.testing.assert_array_equal(got, [INF, -INF, NAN])


def test_the_floating_point_state_is_restored_after_an_evaluation() -> None:
    before = np.geterr()
    run("a / b")
    assert np.geterr() == before


def test_a_literal_is_float64() -> None:
    for text, value in (("600 / 60", 10.0), ("25 ** 2", 625.0), ("1", 1.0)):
        got = run(text, {}, 4)
        assert got.dtype == np.float64
        same(got, np.full(4, value))


def test_an_integer_column_times_a_literal_is_float64() -> None:
    steps = np.array([1, 2, 3], dtype=np.int64)
    got = run("steps * 100", {"steps": steps}, 3)
    same(got, np.array([100.0, 200.0, 300.0]))


def test_an_integer_column_keeps_its_dtype_through_integer_arithmetic() -> None:
    left = np.array([1, 2, 3], dtype=np.int64)
    right = np.array([4, 5, 6], dtype=np.int64)
    same(run("l + r", {"l": left, "r": right}, 3), np.add(left, right))
    same(run("l / r", {"l": left, "r": right}, 3), np.true_divide(left, right))


def test_a_boolean_literal_is_a_numpy_bool() -> None:
    got = run("True", {}, 3)
    same(got, np.array([True, True, True]))
    same(run("False", {}, 2), np.array([False, False]))


def test_a_square_equals_the_product_bit_for_bit() -> None:
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(scale=40.0, size=5000), [NAN, INF, -INF, 0.0, -0.0]])
    same(run("x ** 2", {"x": x}, len(x)), x * x)


def test_modulo_and_floor_division_are_floored_on_negative_operands() -> None:
    left = np.array([-7.0, 7.0, -7.0, 7.0, -0.5, 0.5])
    right = np.array([2.0, -2.0, -2.0, 2.0, 0.25, -0.25])
    columns = {"l": left, "r": right}
    same(
        run("l % r", columns, 6),
        np.array(
            [-7.0 % 2.0, 7.0 % -2.0, -7.0 % -2.0, 7.0 % 2.0, -0.5 % 0.25, 0.5 % -0.25]
        ),
    )
    same(
        run("l // r", columns, 6),
        np.array(
            [
                -7.0 // 2.0,
                7.0 // -2.0,
                -7.0 // -2.0,
                7.0 // 2.0,
                -0.5 // 0.25,
                0.5 // -0.25,
            ]
        ),
    )
    assert run("l // r", columns, 6).tolist() == [-4.0, -4.0, 3.0, 3.0, -2.0, -2.0]
    assert run("l % r", columns, 6).tolist() == [1.0, -1.0, -1.0, 1.0, 0.0, 0.0]


def test_integer_modulo_and_floor_division_are_floored() -> None:
    left = np.array([-7, 7, -7, 7], dtype=np.int64)
    right = np.array([2, -2, -2, 2], dtype=np.int64)
    columns = {"l": left, "r": right}
    assert run("l % r", columns, 4).tolist() == [1, -1, -1, 1]
    assert run("l // r", columns, 4).tolist() == [-4, -4, 3, 3]


def test_a_chained_comparison_joins_its_pairs_left_to_right() -> None:
    x = np.array([-1.0, 0.0, 1.0, 5.0, 6.0, NAN])
    same(
        run("0 < x <= 5", {"x": x}, 6),
        np.array([False, False, True, True, False, False]),
    )
    low, mid, high = (
        np.array([1.0, 1.0, 3.0]),
        np.array([2.0, 0.0, 4.0]),
        np.array([3.0, 5.0, 3.0]),
    )
    expected = np.logical_and(np.logical_and(low < mid, mid < high), high >= low)
    same(run("l < m < h >= l", {"l": low, "m": mid, "h": high}, 3), expected)


def test_a_comparison_with_nan_is_false_and_not_equal_is_true() -> None:
    x = np.array([NAN, 1.0, 2.0])
    columns = {"x": x}
    assert run("x == x", columns, 3).tolist() == [False, True, True]
    assert run("x != x", columns, 3).tolist() == [True, False, False]
    assert run("x < 1", columns, 3).tolist() == [False, False, False]
    assert run("x >= 1", columns, 3).tolist() == [False, True, True]
    assert run("x != nan", columns, 3).tolist() == [True, True, True]


def test_isnull_is_false_and_notnull_true_on_integers_and_booleans() -> None:
    count = np.array([0, 1, 2], dtype=np.int64)
    flag = np.array([True, False, True])
    columns = {"n": count, "f": flag}
    assert run("isnull(n)", columns, 3).tolist() == [False, False, False]
    assert run("notnull(n)", columns, 3).tolist() == [True, True, True]
    assert run("isnull(f)", columns, 3).tolist() == [False, False, False]
    assert run("notnull(f)", columns, 3).tolist() == [True, True, True]
    assert run("fillna(n, 9)", columns, 3).tolist() == [0.0, 1.0, 2.0]


def test_a_nullable_integer_column_reads_as_float64_with_nan_for_na() -> None:
    column = pd.array([1, None, 3], dtype="Int64")
    got = run("x + 1", {"x": column}, 3)
    same(got, np.array([2.0, NAN, 4.0]))
    assert run("isnull(x)", {"x": column}, 3).tolist() == [False, True, False]
    series = pd.Series([1.5, None, 2.5], dtype="Float64")
    same(run("x * 2", {"x": series}, 3), np.array([3.0, NAN, 5.0]))


def test_a_nullable_boolean_column_without_na_reads_as_a_numpy_boolean() -> None:
    column = pd.array([True, False, True], dtype="boolean")
    same(run("~f", {"f": column}, 3), np.array([False, True, False]))
    same(
        run("f & (x > 0)", {"f": column, "x": np.array([1.0, 1.0, -1.0])}, 3),
        np.array([True, False, False]),
    )


def test_a_nullable_boolean_column_with_na_is_refused_naming_the_column() -> None:
    column = pd.array([True, None, False], dtype="boolean")
    with pytest.raises(ExpressionError, match="'flag'"):
        run("flag & flag", {"flag": column}, 3)


@pytest.mark.parametrize(
    "column",
    [
        np.array(["a", "b"]),
        np.array(["a", "b"], dtype=object),
        pd.Series(["a", "b"], dtype="string"),
        pd.Series(["a", "b"], dtype="category"),
        pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"])),
    ],
    ids=["unicode", "object", "string", "category", "datetime"],
)
def test_a_column_of_another_dtype_is_refused_naming_the_column(
    column: ColumnValues,
) -> None:
    with pytest.raises(ExpressionError, match="'label'"):
        run("label", {"label": column}, 2)


def test_every_column_is_converted_before_any_operation() -> None:
    with pytest.raises(ExpressionError, match="'label'"):
        run("a + label", {"a": A, "label": np.array(["x"] * N_ROWS)})


def test_a_column_the_table_lacks_is_refused_naming_the_expression_and_the_column() -> (
    None
):
    with pytest.raises(ExpressionError, match="(?s)'a \\+ missing'.*'missing'"):
        run("a + missing")


def test_a_scalar_result_is_broadcast_to_one_value_per_row() -> None:
    same(run("3 + 4", {}, 5), np.full(5, 7.0))
    same(run("nan", {}, 3), np.full(3, NAN))
    same(run("1 < 2", {}, 3), np.full(3, True))
    same(run("pi", {}, 2), np.full(2, np.pi))


def test_a_column_may_not_be_named_for_a_constant() -> None:
    for name in ("pi", "nan", "inf"):
        assert names_read(parse_expression(f"{name} + x")) == ("x",)
    same(run("pi + x", {"pi": np.zeros(2), "x": np.zeros(2)}, 2), np.full(2, np.pi))


@pytest.mark.parametrize(
    "text",
    ["a & b", "a | b", "~a", "1 if a else 2", "a & (b > 0)", "(a > 0) | b"],
)
def test_a_boolean_operator_refuses_a_float_operand_at_evaluation(text: str) -> None:
    parsed = parse_expression(text)
    with pytest.raises(ExpressionError, match="boolean"):
        evaluate(parsed, COLUMNS, N_ROWS)


def test_a_boolean_operator_refuses_an_integer_operand_at_evaluation() -> None:
    parsed = parse_expression("n & n")
    with pytest.raises(ExpressionError, match="boolean"):
        evaluate(parsed, {"n": np.array([1, 0])}, 2)


def test_a_condition_may_be_a_boolean_column() -> None:
    flag = np.array([True, False, True])
    same(run("1 if f else 2", {"f": flag}, 3), np.array([1.0, 2.0, 1.0]))


REFUSED = [
    ("a and b", "and"),
    ("a or b", "or"),
    ("not a", "not"),
    ("a in b", "in"),
    ("a not in b", "not in"),
    ("a is b", "is"),
    ("a is not b", "is not"),
    ("a.real", "attribute"),
    ("a[0]", "subscript"),
    ("a[1:2]", "subscript"),
    ("'text'", "string"),
    ("b'text'", "bytes"),
    ("a == 'x'", "string"),
    ("sqrt(a, out=b)", "keyword"),
    ("minimum(*a)", "starred"),
    ("unknown(a)", "unknown"),
    ("a.sum()", "attribute"),
    ("sqrt(a)(b)", "call"),
    ("sqrt()", "sqrt"),
    ("sqrt(a, b)", "sqrt"),
    ("arctan2(a)", "arctan2"),
    ("fillna(a)", "fillna"),
    ("isnull(a, b)", "isnull"),
    ("a @ b", "@"),
    ("a << b", "<<"),
    ("a >> b", ">>"),
    ("a ^ b", "^"),
    ("lambda: a", "lambda"),
    ("[x for x in a]", "comprehension"),
    ("{x for x in a}", "comprehension"),
    ("(x for x in a)", "comprehension"),
    ("{x: 1 for x in a}", "comprehension"),
    ("(y := a)", "assignment expression"),
    ("f'{a}'", "f-string"),
    ("[a, b]", "list"),
    ("(a, b)", "tuple"),
    ("{1: a}", "dict"),
    ("None", "None"),
    ("1j", "complex"),
    ("...", "Ellipsis"),
    ("a if b else c if d else 'x'", "string"),
]


@pytest.mark.parametrize(("text", "node"), REFUSED, ids=[text for text, _ in REFUSED])
def test_a_refused_node_names_the_expression_and_the_node(text: str, node: str) -> None:
    with pytest.raises(ExpressionError) as refused:
        parse_expression(text)
    assert repr(text) in str(refused.value)
    assert node in str(refused.value)
    assert isinstance(refused.value, ValueError)


def test_the_first_refused_node_in_the_source_is_named() -> None:
    with pytest.raises(ExpressionError) as refused:
        parse_expression("a ^ b and c")
    assert "^" in str(refused.value)


def test_a_syntax_error_is_an_expression_error_naming_the_expression() -> None:
    for text in ("a +", "", "a b", "(a", "a = 1", " a"):
        with pytest.raises(ExpressionError) as refused:
            parse_expression(text)
        assert repr(text) in str(refused.value)


def test_names_read_lists_columns_in_first_read_order_without_repeats() -> None:
    parsed = parse_expression("b + a * b - sqrt(c) + pi + nan + inf + abs(d) - a")
    assert names_read(parsed) == ("b", "a", "c", "d")


def test_names_read_follows_the_source_for_a_conditional_and_a_chain() -> None:
    assert names_read(parse_expression("x if c else y")) == ("x", "c", "y")
    assert names_read(parse_expression("lo < mid <= hi")) == ("lo", "mid", "hi")
    assert names_read(parse_expression("fillna(p, q) + isnull(r)")) == ("p", "q", "r")


def test_names_read_of_constants_alone_is_empty() -> None:
    assert names_read(parse_expression("pi * 2 + inf - nan")) == ()


def test_a_column_may_share_a_function_name_when_it_is_not_called() -> None:
    parsed = parse_expression("floor + floor(x)")
    assert names_read(parsed) == ("floor", "x")
    same(
        run(
            "floor + floor(x)",
            {"floor": np.array([1.0, 2.0]), "x": np.array([0.5, 1.5])},
            2,
        ),
        np.array([1.0, 3.0]),
    )


def test_a_parsed_expression_keeps_its_text_and_compares_by_it() -> None:
    first = parse_expression("a + 1")
    second = parse_expression("a + 1")
    assert isinstance(first, ParsedExpression)
    assert first.text == "a + 1"
    assert first == second
    assert first != parse_expression("a + 2")
    field_name = "text"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(first, field_name, "b")


def test_a_column_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ExpressionError, match="'a'"):
        run("a", {"a": np.zeros(3)}, 4)


def test_a_literal_beyond_float64_is_an_expression_error_at_evaluation() -> None:
    with pytest.raises(ExpressionError, match="could not be evaluated"):
        run("1" + "0" * 400)


def test_an_operation_numpy_refuses_is_an_expression_error() -> None:
    with pytest.raises(ExpressionError, match="could not be evaluated"):
        run("-f", {"f": np.array([True, False])}, 2)
