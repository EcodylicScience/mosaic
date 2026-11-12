"""How a ``where`` value of text is read as the values a group key could hold."""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.summary_arguments import KeyValue, readings, read_where


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("A", ["A"], id="text"),
        pytest.param("1", ["1", 1], id="whole number"),
        pytest.param("-12", ["-12", -12], id="signed whole number"),
        pytest.param("1.5", ["1.5", 1.5], id="decimal"),
        pytest.param("1e3", ["1e3", 1000.0], id="exponent"),
        pytest.param("true", ["true", True], id="true"),
        pytest.param("false", ["false", False], id="false"),
        pytest.param("True", ["True"], id="capitalized boolean stays text"),
        pytest.param("nan", ["nan"], id="nan stays text"),
        pytest.param("inf", ["inf"], id="infinity stays text"),
        pytest.param("", [""], id="empty"),
    ],
)
def test_text_is_read_as_each_value_it_could_hold(
    text: str, expected: list[KeyValue]
) -> None:
    assert readings(text) == expected
    assert [type(value) for value in readings(text)] == [
        type(value) for value in expected
    ]


def test_read_where_reads_a_string_and_keeps_a_number_and_a_boolean() -> None:
    where: dict[str, list[KeyValue]] = {
        "n_inds": ["1"],
        "strain": ["A", 7],
        "fed": [True, 0.5],
    }

    read = read_where(where)

    assert read == {"n_inds": ["1", 1], "strain": ["A", 7], "fed": [True, 0.5]}
    assert where["n_inds"] == ["1"]
