"""``summarize_run`` describes one run's rows by group, from the table they come in.

Each run here is written cell by cell (``labeled_run``), so a key's Arrow type in
each entry's file is the one the test chose: a label null for a whole entry is
Arrow ``null``, an integer label with a null is ``double``. The cases hold the
summary to numpy's statistics over the finite cells, to one type per key across
entries, to the counts of what it leaves out, to scipy's tests between groups and
Holm's adjustment, to the refusals of a per-frame run, of keys that disagree and of
too many pairs, and to one result whatever the order of the index.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from scipy import stats

from mosaic.core import pipeline as pipeline_package
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline import run_matrix as run_matrix_module
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph.param_table import strip_schema_keywords
from mosaic.core.pipeline.preview import ArtifactPreview
from mosaic.core.pipeline.run_matrix import RunRead, RunReadInvalid
from mosaic.core.pipeline.run_summary import (
    MAX_PAIRWISE_GROUPS,
    PreviewOutput,
    RunSummary,
    RunSummaryWire,
    SummaryConflict,
    SummaryInvalid,
    run_summary_wire,
    summarize_run,
)
from mosaic.core.pipeline.summary_arguments import KeyValue
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from tests.helpers import labeled_run, rewrite_index

type Summarize = Callable[[Dataset, Result[str]], RunSummary]
"""One call of ``summarize_run`` on a run."""

BIG = 2**53
"""The least integer magnitude that float64 cannot tell from its neighbor."""


def table(**columns: object) -> pd.DataFrame:
    """One entry's rows, from its columns."""
    return pd.DataFrame(dict(columns))


def run_of(
    tmp_path: Path, tables: Mapping[Entry, pd.DataFrame]
) -> tuple[Dataset, Result[str]]:
    return labeled_run(tmp_path, tables)


def records(frame: pd.DataFrame, *columns: str) -> list[tuple[object, ...]]:
    """The rows of *frame* over *columns*, as plain Python values."""
    return [tuple(row) for row in frame[list(columns)].itertuples(index=False)]


def statistics_of(summary: RunSummary, column: str) -> pd.DataFrame:
    rows = summary.statistics[summary.statistics["column"] == column]
    return rows.reset_index(drop=True)


# --- statistics -------------------------------------------------------------


def three_strains(tmp_path: Path) -> tuple[Dataset, Result[str]]:
    """Strain A over two entries, B with one finite value, C with none."""
    return run_of(
        tmp_path,
        {
            ("g", "a"): table(
                strain=["A", "A", "A"],
                value=[1.0, 2.0, 4.0],
                other=[5.0, 6.0, 7.0],
            ),
            ("g", "b"): table(
                strain=["A", "A"], value=[np.nan, 7.0], other=[8.0, np.nan]
            ),
            ("g", "c"): table(
                strain=["B", "B", "B"],
                value=[3.0, np.inf, -np.inf],
                other=[1.0, 2.0, 3.0],
            ),
            ("g", "d"): table(
                strain=["C", "C"], value=[np.nan, np.nan], other=[1.0, 1.0]
            ),
        },
    )


def expected_statistics(cells: Sequence[float]) -> dict[str, float]:
    """numpy's statistics over the finite cells of one group's column."""
    everything = np.asarray(cells, dtype=np.float64)
    finite = everything[np.isfinite(everything)]
    n = len(finite)
    sd = float(np.std(finite, ddof=1)) if n >= 2 else np.nan
    return {
        "n": n,
        "n_missing": len(everything) - n,
        "mean": float(np.mean(finite)) if n else np.nan,
        "sd": sd,
        "sem": sd / np.sqrt(n) if n >= 2 else np.nan,
        "median": float(np.median(finite)) if n else np.nan,
    }


def test_statistics_are_numpys_over_the_finite_cells_of_each_group(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    summary = summarize_run(ds, result, by=["strain"])

    assert summary.by == ("strain",)
    assert summary.columns == ("value", "other")
    assert list(summary.statistics.columns) == [
        "strain",
        "column",
        "n",
        "n_missing",
        "mean",
        "sd",
        "sem",
        "median",
    ]
    assert records(summary.statistics, "strain", "column") == [
        ("A", "value"),
        ("A", "other"),
        ("B", "value"),
        ("B", "other"),
        ("C", "value"),
        ("C", "other"),
    ]
    cells = {
        ("A", "value"): [1.0, 2.0, 4.0, np.nan, 7.0],
        ("A", "other"): [5.0, 6.0, 7.0, 8.0, np.nan],
        ("B", "value"): [3.0, np.inf, -np.inf],
        ("B", "other"): [1.0, 2.0, 3.0],
        ("C", "value"): [np.nan, np.nan],
        ("C", "other"): [1.0, 1.0],
    }
    for row in summary.statistics.itertuples(index=False):
        expected = expected_statistics(cells[(str(row.strain), str(row.column))])
        actual = {name: getattr(row, name) for name in expected}
        assert int(actual["n"]) == expected["n"]
        assert int(actual["n_missing"]) == expected["n_missing"]
        for name in ("mean", "sd", "sem", "median"):
            np.testing.assert_allclose(actual[name], expected[name], equal_nan=True)


def test_a_group_with_one_finite_cell_has_no_spread_and_one_with_none_no_statistics(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    value = statistics_of(summarize_run(ds, result, by=["strain"]), "value")

    one, none = value.iloc[1], value.iloc[2]
    assert (one["strain"], one["n"], one["n_missing"]) == ("B", 1, 2)
    assert one["mean"] == one["median"] == 3.0
    assert np.isnan(one["sd"]) and np.isnan(one["sem"])
    assert (none["strain"], none["n"], none["n_missing"]) == ("C", 0, 2)
    assert np.isnan([none[name] for name in ("mean", "sd", "sem", "median")]).all()


def test_groups_count_rows_and_entries(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)

    summary = summarize_run(ds, result, by=["strain"])

    assert list(summary.groups.columns) == ["strain", "n_rows", "n_entries"]
    assert records(summary.groups, "strain", "n_rows", "n_entries") == [
        ("A", 5, 2),
        ("B", 3, 1),
        ("C", 2, 1),
    ]
    assert summary.entries == (("g", "a"), ("g", "b"), ("g", "c"), ("g", "d"))
    assert summary.missing == ()
    assert summary.finished
    assert summary.rows_without_key == 0
    assert summary.entries_without_rows == ()
    assert summary.max_rows_per_entry == 3


def test_the_tests_frames_are_empty_with_their_columns_when_no_test_is_asked(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    summary = summarize_run(ds, result, by=["strain"])

    assert summary.kruskal.empty
    assert list(summary.kruskal.columns) == [
        "column",
        "k",
        "n",
        "statistic",
        "df",
        "p",
        "reason",
    ]
    assert summary.mann_whitney.empty
    assert list(summary.mann_whitney.columns) == [
        "column",
        "strain_a",
        "strain_b",
        "n_a",
        "n_b",
        "statistic",
        "p",
        "p_holm",
        "comparisons",
        "reason",
    ]


def test_columns_chooses_the_values_summarized_in_order(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)

    summary = summarize_run(ds, result, by=["strain"], columns=["other", "value"])

    assert summary.columns == ("other", "value")
    assert records(summary.statistics, "strain", "column")[:2] == [
        ("A", "other"),
        ("A", "value"),
    ]


# --- key types across entries -----------------------------------------------


def test_an_entry_whose_key_is_all_null_is_left_out_and_the_others_grouped(
    tmp_path: Path,
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(strain=[None, None], value=[1.0, 2.0]),
            ("g", "b"): table(strain=["B", "B"], value=[3.0, 4.0]),
        },
    )

    summary = summarize_run(ds, result, by=["strain"])

    assert summary.rows_without_key == 2
    assert records(summary.groups, "strain", "n_rows", "n_entries") == [("B", 2, 1)]


def test_an_integer_key_stored_as_double_and_as_int64_is_one_integer_key(
    tmp_path: Path,
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(
                batch=np.array([1.0, np.nan, 2.0]), value=[1.0, 2.0, 3.0]
            ),
            ("g", "b"): table(batch=np.array([1, 3], dtype=np.int64), value=[4.0, 5.0]),
        },
    )

    summary = summarize_run(ds, result, by=["batch"])

    assert records(summary.groups, "batch", "n_rows", "n_entries") == [
        (1, 2, 2),
        (2, 1, 1),
        (3, 1, 1),
    ]
    assert [type(value) for value in summary.groups["batch"].tolist()] == [int] * 3
    assert summary.rows_without_key == 1


def test_numeric_keys_sort_as_numbers(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(
                batch=np.array([10, 9, 2], dtype=np.int64), value=[1.0, 2.0, 3.0]
            ),
            ("g", "b"): table(
                batch=np.array([-1, 10], dtype=np.int64), value=[4.0, 5.0]
            ),
        },
    )

    summary = summarize_run(ds, result, by=["batch"])

    assert summary.groups["batch"].tolist() == [-1, 2, 9, 10]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (["x", "x"], np.array([1, 1], dtype=np.int64)),
        ([True, True], np.array([1.5, 1.5])),
        (["x", "x"], [True, False]),
    ],
    ids=["text-and-number", "boolean-and-number", "text-and-boolean"],
)
def test_a_key_that_is_text_boolean_or_number_by_entry_is_a_conflict(
    tmp_path: Path, first: Sequence[object], second: Sequence[object]
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(label=first, value=[1.0, 2.0]),
            ("g", "b"): table(label=second, value=[3.0, 4.0]),
        },
    )

    with pytest.raises(SummaryConflict) as raised:
        _ = summarize_run(ds, result, by=["label"])

    assert isinstance(raised.value, ValueError)
    assert raised.value.error_kind == "conflict"
    assert "label" in str(raised.value)


@pytest.mark.parametrize("big", [BIG, -BIG], ids=["positive", "negative"])
def test_an_integer_of_two_to_the_53_beside_floats_is_a_conflict(
    tmp_path: Path, big: int
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(batch=np.array([0.5]), value=[1.0]),
            ("g", "b"): table(batch=np.array([big], dtype=np.int64), value=[2.0]),
        },
    )

    with pytest.raises(SummaryConflict, match="batch"):
        _ = summarize_run(ds, result, by=["batch"])


def test_an_integer_below_two_to_the_53_beside_floats_is_merged(
    tmp_path: Path,
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(batch=np.array([float(BIG - 1)]), value=[1.0]),
            ("g", "b"): table(batch=np.array([BIG - 1], dtype=np.int64), value=[2.0]),
        },
    )

    summary = summarize_run(ds, result, by=["batch"])

    assert records(summary.groups, "batch", "n_rows") == [(BIG - 1, 2)]


def test_a_timestamp_key_is_invalid(tmp_path: Path) -> None:
    stamps = pd.to_datetime(["2026-01-01", "2026-01-02"])
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(when=stamps, value=[1.0, 2.0]),
            ("g", "b"): table(when=stamps, value=[3.0, 4.0]),
        },
    )

    with pytest.raises(SummaryInvalid, match="when") as raised:
        _ = summarize_run(ds, result, by=["when"])

    assert isinstance(raised.value, ValueError)
    assert raised.value.error_kind == "invalid"


# --- keys -------------------------------------------------------------------


def labeled_individuals(tmp_path: Path) -> tuple[Dataset, Result[str]]:
    """Two individuals per entry, with a strain per entry and a sex per individual."""
    return run_of(
        tmp_path,
        {
            ("g1", "a"): table(
                id=np.array([0, 1], dtype=np.int64),
                strain=["A", "A"],
                sex=["F", "M"],
                value=[1.0, 2.0],
            ),
            ("g1", "b"): table(
                id=np.array([0, 1], dtype=np.int64),
                strain=["B", "B"],
                sex=["F", "F"],
                value=[3.0, 4.0],
            ),
            ("g2", "a"): table(
                id=np.array([0, 1], dtype=np.int64),
                strain=["A", "A"],
                sex=["M", "M"],
                value=[5.0, 6.0],
            ),
        },
    )


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        (["strain"], [("A", 4, 2), ("B", 2, 1)]),
        (["group"], [("g1", 4, 2), ("g2", 2, 1)]),
        (["sequence"], [("a", 4, 2), ("b", 2, 1)]),
        (["id"], [(0, 3, 3), (1, 3, 3)]),
    ],
    ids=["kept-text", "group", "sequence", "id"],
)
def test_a_key_groups_the_rows_by_its_values(
    tmp_path: Path, by: list[str], expected: list[tuple[object, ...]]
) -> None:
    ds, result = labeled_individuals(tmp_path)

    summary = summarize_run(ds, result, by=by)

    assert records(summary.groups, by[0], "n_rows", "n_entries") == expected
    assert summary.max_rows_per_entry == 2


def test_two_keys_group_by_their_combinations_that_hold_rows(tmp_path: Path) -> None:
    ds, result = labeled_individuals(tmp_path)

    summary = summarize_run(ds, result, by=["strain", "sex"])

    assert records(summary.groups, "strain", "sex", "n_rows", "n_entries") == [
        ("A", "F", 1, 1),
        ("A", "M", 3, 2),
        ("B", "F", 2, 1),
    ]
    assert records(summary.statistics, "strain", "sex", "column", "mean") == [
        ("A", "F", "value", 1.0),
        ("A", "M", "value", 13.0 / 3.0),
        ("B", "F", "value", 3.5),
    ]


def test_keys_go_to_the_reader_as_named_once_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = labeled_individuals(tmp_path)
    planned = run_matrix_module.plan_run_read
    asked: list[tuple[str, ...]] = []

    def recording(
        ds: Dataset,
        result: Result[str],
        *,
        scope: Scope | None = None,
        perspective: int | None = None,
        columns: str | Sequence[str] | None = None,
        keys: Sequence[str] = (),
        allow_partial: bool = False,
    ) -> RunRead:
        asked.append(tuple(keys))
        return planned(
            ds,
            result,
            scope=scope,
            perspective=perspective,
            columns=columns,
            keys=keys,
            allow_partial=allow_partial,
        )

    monkeypatch.setattr(run_matrix_module, "plan_run_read", recording)

    summary = summarize_run(
        ds, result, by=["id", "group", "strain", "id"], where={"sex": ["F"], "id": [0]}
    )

    assert asked == [("id", "group", "strain", "sex")]
    assert records(summary.groups, "id", "group", "strain", "n_rows") == [
        (0, "g1", "A", 1),
        (0, "g1", "B", 1),
    ]


def test_an_identity_key_that_the_run_does_not_write_takes_the_readers_refusal(
    tmp_path: Path,
) -> None:
    ds, result = labeled_individuals(tmp_path)

    with pytest.raises(RunReadInvalid, match="id1"):
        _ = summarize_run(ds, result, by=["id1"])


def test_integer_and_boolean_keys_with_nulls_are_typed_back(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(
                batch=pd.array([2, None, 1], dtype="Int64"),
                treated=[True, None, False],
                value=[1.0, 2.0, 3.0],
            ),
            ("g", "b"): table(
                batch=pd.array([1, 2, None], dtype="Int64"),
                treated=[None, True, True],
                value=[4.0, 5.0, 6.0],
            ),
        },
    )

    batch = summarize_run(ds, result, by=["batch"])
    treated = summarize_run(ds, result, by=["treated"])
    both = summarize_run(ds, result, by=["batch", "treated"])

    assert records(batch.groups, "batch", "n_rows") == [(1, 2), (2, 2)]
    assert [type(value) for value in batch.groups["batch"].tolist()] == [int, int]
    assert batch.rows_without_key == 2
    assert records(treated.groups, "treated", "n_rows") == [(False, 1), (True, 3)]
    assert [type(value) for value in treated.groups["treated"].tolist()] == [
        bool,
        bool,
    ]
    assert treated.rows_without_key == 2
    assert records(both.groups, "batch", "treated", "n_rows") == [
        (1, False, 1),
        (2, True, 2),
    ]
    assert both.rows_without_key == 3


def test_no_keys_give_one_group_of_every_row(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)

    summary = summarize_run(ds, result)

    assert summary.by == ()
    assert list(summary.groups.columns) == ["n_rows", "n_entries"]
    assert records(summary.groups, "n_rows", "n_entries") == [(10, 4)]
    assert records(summary.statistics, "column", "n", "n_missing") == [
        ("value", 5, 5),
        ("other", 9, 1),
    ]


# --- where ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"strain": ["A"]}, [("A", "F", 1), ("A", "M", 3)]),
        ({"strain": ["A"], "sex": ["M"]}, [("A", "M", 3)]),
        ({"strain": ["A", "Z"], "sex": ["M", "Q"]}, [("A", "M", 3)]),
        ({"strain": ["Z"]}, []),
    ],
    ids=["one-key", "two-keys", "values-without-rows", "no-group"],
)
def test_where_keeps_the_rows_whose_key_values_are_listed(
    tmp_path: Path,
    where: dict[str, Sequence[KeyValue]],
    expected: list[tuple[object, ...]],
) -> None:
    ds, result = labeled_individuals(tmp_path)

    summary = summarize_run(ds, result, by=["strain", "sex"], where=where)

    assert records(summary.groups, "strain", "sex", "n_rows") == expected


def test_a_where_key_need_not_be_a_group_key(tmp_path: Path) -> None:
    ds, result = labeled_individuals(tmp_path)

    summary = summarize_run(ds, result, by=["strain"], where={"sex": ["F"]})

    assert records(summary.groups, "strain", "n_rows", "n_entries") == [
        ("A", 1, 1),
        ("B", 2, 1),
    ]
    assert "sex" not in summary.groups.columns


def test_the_rows_per_entry_are_counted_after_where(tmp_path: Path) -> None:
    ds, result = labeled_individuals(tmp_path)

    summary = summarize_run(ds, result, by=["strain"], where={"id": [0]})

    assert summary.max_rows_per_entry == 1
    assert records(summary.groups, "strain", "n_rows", "n_entries") == [
        ("A", 2, 2),
        ("B", 1, 1),
    ]


def test_where_matches_typed_values(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(
                batch=np.array([1.0, np.nan]), treated=[True, False], value=[1.0, 2.0]
            ),
            ("g", "b"): table(
                batch=np.array([1, 2], dtype=np.int64),
                treated=[True, True],
                value=[3.0, 4.0],
            ),
        },
    )

    by_number = summarize_run(ds, result, by=["batch"], where={"batch": [1.0]})
    by_boolean = summarize_run(ds, result, where={"batch": [True]})
    by_flag = summarize_run(ds, result, by=["batch"], where={"treated": [True]})

    assert records(by_number.groups, "batch", "n_rows") == [(1, 2)]
    assert by_boolean.groups.empty
    assert records(by_flag.groups, "batch", "n_rows") == [(1, 2), (2, 1)]


def test_the_counts_of_what_is_left_out_come_before_where(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(strain=["A", None, "A"], value=[1.0, 2.0, 3.0]),
            ("g", "b"): table(strain=["B", "B"], value=[4.0, 5.0]),
            ("g", "c"): table(
                strain=pd.Series([], dtype=object), value=pd.Series([], dtype=float)
            ),
        },
    )

    summary = summarize_run(ds, result, by=["strain"], where={"strain": ["B"]})

    assert summary.rows_without_key == 1
    assert summary.entries_without_rows == (("g", "c"),)
    assert records(summary.groups, "strain", "n_rows") == [("B", 2)]


# --- entries and rows -------------------------------------------------------


def test_an_entry_whose_output_holds_no_row_is_listed(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(value=[1.0, 2.0]),
            ("g", "b"): table(value=pd.Series([], dtype=float)),
        },
    )

    summary = summarize_run(ds, result)

    assert summary.entries == (("g", "a"), ("g", "b"))
    assert summary.entries_without_rows == (("g", "b"),)
    assert records(summary.groups, "n_rows", "n_entries") == [(2, 1)]


def test_several_rows_per_entry_are_counted_beside_the_entries(
    tmp_path: Path,
) -> None:
    ds, result = labeled_individuals(tmp_path)

    summary = summarize_run(ds, result, by=["strain"])

    assert summary.max_rows_per_entry == 2
    assert records(summary.groups, "strain", "n_rows", "n_entries") == [
        ("A", 4, 2),
        ("B", 2, 1),
    ]


# --- the unit of analysis ---------------------------------------------------


@pytest.mark.parametrize("column", ["frame", "time"])
def test_a_per_frame_run_is_refused_before_a_value_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, column: str
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(**{column: np.arange(3)}, value=[1.0, 2.0, 3.0]),
            ("g", "b"): table(**{column: np.arange(3)}, value=[4.0, 5.0, 6.0]),
        },
    )

    def unread(*_: object, **__: object) -> object:
        raise AssertionError("read_run was called")

    monkeypatch.setattr(run_matrix_module, "read_run", unread)

    with pytest.raises(SummaryInvalid) as raised:
        _ = summarize_run(ds, result)

    message = str(raised.value)
    assert isinstance(raised.value, ValueError)
    assert raised.value.error_kind == "invalid"
    assert result.feature in message
    assert str(result.run_id) in message
    assert "sequence-aggregate" in message
    assert "individual-aggregate" in message
    assert "keep_columns" in message


# --- refusals ---------------------------------------------------------------


RESERVED = ("column", "n", "n_rows", "n_entries", "n_missing", "mean", "sd", "sem")
"""Names of the summary's own columns, which no key may take, besides ``median``."""


def by_a_value_column(columns: str | list[str]) -> Summarize:
    return lambda ds, result: summarize_run(ds, result, by=["value"], columns=columns)


def by_a_reserved_name(name: str) -> Summarize:
    return lambda ds, result: summarize_run(ds, result, by=[name])


def where_without_values(ds: Dataset, result: Result[str]) -> RunSummary:
    return summarize_run(ds, result, where={"strain": []})


def where_on_a_value_column(ds: Dataset, result: Result[str]) -> RunSummary:
    return summarize_run(
        ds, result, by=["strain"], columns=["value"], where={"value": [1.0]}
    )


def an_unknown_test(ds: Dataset, result: Result[str]) -> RunSummary:
    untyped: Callable[..., RunSummary] = summarize_run
    return untyped(ds, result, by=["strain"], tests=["ttest"])


@pytest.mark.parametrize(
    "call",
    [
        by_a_value_column(["value"]),
        by_a_value_column("value"),
        where_without_values,
        where_on_a_value_column,
        an_unknown_test,
        *[by_a_reserved_name(name) for name in (*RESERVED, "median")],
    ],
    ids=[
        "by-in-columns",
        "by-is-the-column",
        "where-empty",
        "where-in-columns",
        "unknown-test",
        *(f"by-{name}" for name in (*RESERVED, "median")),
    ],
)
def test_arguments_that_state_no_summary_are_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: Summarize
) -> None:
    ds, result = three_strains(tmp_path)

    def unplanned(*_: object, **__: object) -> object:
        raise AssertionError("plan_run_read was called")

    monkeypatch.setattr(run_matrix_module, "plan_run_read", unplanned)

    with pytest.raises(SummaryInvalid) as raised:
        _ = call(ds, result)

    assert isinstance(raised.value, ValueError)
    assert isinstance(raised.value, ClassifiedError)
    assert raised.value.error_kind == "invalid"
    detail = raised.value.error_detail()
    assert detail is not None
    assert (detail["feature"], detail["run_id"]) == (result.feature, result.run_id)


def test_a_where_key_among_the_columns_is_named_in_the_refusal(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    with pytest.raises(SummaryInvalid) as raised:
        _ = where_on_a_value_column(ds, result)

    message = str(raised.value)
    assert "where={'value': [1.0]} names ['value']" in message
    assert "by=" not in message
    assert raised.value.columns == ("value",)


def test_the_readers_refusals_pass_through(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)

    with pytest.raises(RunReadInvalid):
        _ = summarize_run(ds, result, scope=Scope(groups=["nowhere"]))
    with pytest.raises(RunReadInvalid, match="absent"):
        _ = summarize_run(ds, result, by=["absent"])


# --- tests between groups ---------------------------------------------------


def three_samples(tmp_path: Path) -> tuple[Dataset, Result[str]]:
    """Three strains whose values differ, with NaN and infinite cells among them."""
    return run_of(
        tmp_path,
        {
            ("g", "a"): table(
                strain=["A", "A", "A", "A"],
                value=[1.0, 2.5, np.nan, 3.0],
                other=[4.0, 4.0, 5.0, 9.0],
            ),
            ("g", "b"): table(
                strain=["B", "B", "B", "B"],
                value=[4.0, np.inf, 6.0, 5.5],
                other=[1.0, 2.0, 4.0, 4.0],
            ),
            ("g", "c"): table(
                strain=["C", "C", "C"],
                value=[2.0, 7.0, 8.0],
                other=[3.0, -np.inf, 6.0],
            ),
        },
    )


THREE_SAMPLES = {
    ("A", "value"): [1.0, 2.5, 3.0],
    ("B", "value"): [4.0, 6.0, 5.5],
    ("C", "value"): [2.0, 7.0, 8.0],
    ("A", "other"): [4.0, 4.0, 5.0, 9.0],
    ("B", "other"): [1.0, 2.0, 4.0, 4.0],
    ("C", "other"): [3.0, 6.0],
}
"""The finite cells of each strain and column of :func:`three_samples`."""

PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))


def test_kruskal_wallis_is_scipys_over_the_finite_cells(tmp_path: Path) -> None:
    ds, result = three_samples(tmp_path)

    summary = summarize_run(ds, result, by=["strain"], tests=["kruskal"])

    assert records(summary.kruskal, "column", "k", "n", "df") == [
        ("value", 3, 9, 2.0),
        ("other", 3, 10, 2.0),
    ]
    for row in summary.kruskal.itertuples(index=False):
        column = str(row.column)
        expected = stats.kruskal(*(THREE_SAMPLES[(s, column)] for s in "ABC"))
        assert row.statistic == pytest.approx(expected.statistic)
        assert row.p == pytest.approx(expected.pvalue)
        assert row.reason is None
    assert summary.mann_whitney.empty
    assert list(summary.mann_whitney.columns[:4]) == [
        "column",
        "strain_a",
        "strain_b",
        "n_a",
    ]
    for side in ("strain_a", "strain_b"):
        assert summary.mann_whitney[side].dtype == summary.groups["strain"].dtype


def test_mann_whitney_is_scipys_for_every_pair_with_u_of_the_first_group(
    tmp_path: Path,
) -> None:
    ds, result = three_samples(tmp_path)

    summary = summarize_run(ds, result, by=["strain"], tests=["mannwhitney"])

    assert records(summary.mann_whitney, "column", "strain_a", "strain_b") == [
        (column, a, b) for column in ("value", "other") for a, b in PAIRS
    ]
    for row in summary.mann_whitney.itertuples(index=False):
        column, a, b = str(row.column), str(row.strain_a), str(row.strain_b)
        first, second = THREE_SAMPLES[(a, column)], THREE_SAMPLES[(b, column)]
        expected = stats.mannwhitneyu(
            first, second, alternative="two-sided", method="auto"
        )
        assert (row.n_a, row.n_b) == (len(first), len(second))
        assert row.statistic == pytest.approx(expected.statistic)
        assert row.p == pytest.approx(expected.pvalue)
        assert row.comparisons == 3
        assert row.reason is None
    assert summary.kruskal.empty


@dataclass(frozen=True)
class Answer:
    """What a stand-in for ``mannwhitneyu`` returns."""

    statistic: float
    pvalue: float


def test_holm_adjusts_the_pairs_of_each_column_on_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = three_samples(tmp_path)
    answers = iter([0.01, 0.04, 0.03, 0.6, 0.2, 0.9])

    def mannwhitneyu(*_: object, **__: object) -> Answer:
        return Answer(statistic=1.0, pvalue=next(answers))

    monkeypatch.setattr(stats, "mannwhitneyu", mannwhitneyu)

    summary = summarize_run(ds, result, by=["strain"], tests=["mannwhitney"])

    assert records(summary.mann_whitney, "column", "p", "p_holm", "comparisons") == [
        ("value", 0.01, pytest.approx(0.03), 3),
        ("value", 0.04, pytest.approx(0.06), 3),
        ("value", 0.03, pytest.approx(0.06), 3),
        ("other", 0.6, pytest.approx(1.0), 3),
        ("other", 0.2, pytest.approx(0.6), 3),
        ("other", 0.9, pytest.approx(1.0), 3),
    ]


def scipy_unused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every call of scipy's two tests fail the test."""

    def called(*_: object, **__: object) -> object:
        raise AssertionError("scipy was called")

    monkeypatch.setattr(stats, "kruskal", called)
    monkeypatch.setattr(stats, "mannwhitneyu", called)


def test_values_that_are_all_equal_have_no_variation_and_call_no_scipy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(strain=["A", "A"], value=[2.0, 2.0]),
            ("g", "b"): table(strain=["B", "B", "B"], value=[2.0, np.nan, 2.0]),
        },
    )
    scipy_unused(monkeypatch)

    summary = summarize_run(ds, result, by=["strain"], tests=["kruskal", "mannwhitney"])

    assert records(summary.kruskal, "column", "k", "n", "df", "reason") == [
        ("value", 2, 4, 1.0, "no_variation")
    ]
    assert np.isnan(summary.kruskal[["statistic", "p"]].to_numpy()).all()
    assert records(
        summary.mann_whitney,
        "strain_a",
        "strain_b",
        "n_a",
        "n_b",
        "comparisons",
        "reason",
    ) == [("A", "B", 2, 2, 0, "no_variation")]
    assert np.isnan(summary.mann_whitney[["statistic", "p", "p_holm"]].to_numpy()).all()


def test_a_group_without_finite_cells_has_no_values_and_enters_no_kruskal(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    summary = summarize_run(
        ds, result, by=["strain"], columns=["value"], tests=["kruskal", "mannwhitney"]
    )

    expected = stats.kruskal([1.0, 2.0, 4.0, 7.0], [3.0])
    assert records(summary.kruskal, "column", "k", "n", "df", "reason") == [
        ("value", 2, 5, 1.0, None)
    ]
    assert summary.kruskal["statistic"].iloc[0] == pytest.approx(expected.statistic)
    assert records(
        summary.mann_whitney,
        "strain_a",
        "strain_b",
        "n_a",
        "n_b",
        "comparisons",
        "reason",
    ) == [
        ("A", "B", 4, 1, 1, None),
        ("A", "C", 4, 0, 1, "no_values"),
        ("B", "C", 1, 0, 1, "no_values"),
    ]
    tested = summary.mann_whitney.iloc[0]
    assert tested["p_holm"] == pytest.approx(tested["p"])
    assert np.isnan(
        summary.mann_whitney[["statistic", "p", "p_holm"]].iloc[1:].to_numpy()
    ).all()


def test_one_group_is_too_few_for_kruskal_wallis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = three_strains(tmp_path)
    scipy_unused(monkeypatch)

    summary = summarize_run(ds, result, tests=["kruskal", "mannwhitney"])

    assert records(summary.kruskal, "column", "k", "n", "reason") == [
        ("value", 1, 5, "too_few_groups"),
        ("other", 1, 9, "too_few_groups"),
    ]
    assert np.isnan(summary.kruskal[["statistic", "df", "p"]].to_numpy()).all()
    assert summary.mann_whitney.empty
    assert list(summary.mann_whitney.columns) == [
        "column",
        "n_a",
        "n_b",
        "statistic",
        "p",
        "p_holm",
        "comparisons",
        "reason",
    ]


def many_strains(tmp_path: Path, count: int) -> tuple[Dataset, Result[str]]:
    """One entry holding *count* strains of two rows each."""
    strains = [f"s{index:02d}" for index in range(count) for _ in range(2)]
    values = [float(index) for index in range(2 * count)]
    return run_of(tmp_path, {("g", "a"): table(strain=strains, value=values)})


def test_mann_whitney_over_more_than_twelve_groups_is_refused_before_any_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = many_strains(tmp_path, MAX_PAIRWISE_GROUPS + 1)
    scipy_unused(monkeypatch)

    with pytest.raises(SummaryInvalid) as raised:
        _ = summarize_run(ds, result, by=["strain"], tests=["kruskal", "mannwhitney"])

    message = str(raised.value)
    assert isinstance(raised.value, ValueError)
    assert raised.value.error_kind == "invalid"
    for word in ("13 groups", "12", "where", "scope", "fewer keys", "kruskal"):
        assert word in message


def test_twelve_groups_are_compared_pairwise_and_more_by_kruskal_alone(
    tmp_path: Path,
) -> None:
    ds, result = many_strains(tmp_path / "twelve", MAX_PAIRWISE_GROUPS)
    wide, wide_result = many_strains(tmp_path / "thirteen", MAX_PAIRWISE_GROUPS + 1)

    pairwise = summarize_run(ds, result, by=["strain"], tests=["mannwhitney"])
    kruskal = summarize_run(wide, wide_result, by=["strain"], tests=["kruskal"])

    assert len(pairwise.mann_whitney) == 66
    assert set(pairwise.mann_whitney["comparisons"]) == {66}
    assert records(kruskal.kruskal, "k", "n") == [(13, 26)]


# --- determinism ------------------------------------------------------------


def assert_same_summary(left: RunSummary, right: RunSummary) -> None:
    for name in (
        "feature",
        "run_id",
        "by",
        "columns",
        "entries",
        "missing",
        "finished",
        "entries_without_rows",
        "rows_without_key",
        "max_rows_per_entry",
    ):
        assert getattr(left, name) == getattr(right, name), name
    for name in ("groups", "statistics", "kruskal", "mann_whitney"):
        pd.testing.assert_frame_equal(getattr(left, name), getattr(right, name))


def test_the_summary_does_not_depend_on_the_order_of_the_index(
    tmp_path: Path,
) -> None:
    ds, result = labeled_individuals(tmp_path)
    before = summarize_run(ds, result, by=["sex", "strain"])

    rewrite_index(
        ds,
        result,
        lambda frame: frame.sample(frac=1.0, random_state=7).reset_index(drop=True),
    )
    after = summarize_run(ds, result, by=["sex", "strain"])

    assert_same_summary(before, after)


# --- wire -------------------------------------------------------------------

REF = "feature/prov-labeled/0.1-0123456789"


def wire_of(summary: RunSummary) -> RunSummaryWire:
    return run_summary_wire(summary, REF)


def dumped(model: BaseModel) -> dict[str, JsonValue]:
    """*model* as the command line and the server print it."""
    document: dict[str, JsonValue] = model.model_dump(mode="json")
    return document


def test_the_wire_names_the_run_and_counts_what_it_read(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)

    wire = wire_of(summarize_run(ds, result, by=["strain"]))

    assert wire.ref == REF
    assert wire.by == ("strain",)
    assert wire.columns == ("value", "other")
    assert wire.n_entries == 4
    assert wire.missing == ()
    assert wire.finished is True
    assert wire.entries_without_rows == ()
    assert wire.rows_without_key == 0
    assert wire.max_rows_per_entry == 3
    assert [dumped(group) for group in wire.groups] == [
        {"key": {"strain": "A"}, "n_rows": 5, "n_entries": 2},
        {"key": {"strain": "B"}, "n_rows": 3, "n_entries": 1},
        {"key": {"strain": "C"}, "n_rows": 2, "n_entries": 1},
    ]
    assert wire.kruskal == () and wire.mann_whitney == ()


def test_the_wire_dumps_every_non_finite_float_as_null_in_strict_json(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)
    wire = wire_of(
        summarize_run(ds, result, by=["strain"], tests=["kruskal", "mannwhitney"])
    )

    json.dumps(dumped(wire), allow_nan=False)
    by_group = {(row.key["strain"], row.column): dumped(row) for row in wire.statistics}
    assert by_group["C", "value"] == {
        "column": "value",
        "key": {"strain": "C"},
        "n": 0,
        "n_missing": 2,
        "mean": None,
        "sd": None,
        "sem": None,
        "median": None,
    }
    one = by_group["B", "value"]
    assert (one["n"], one["mean"], one["sd"], one["sem"]) == (1, 3.0, None, None)
    empty_pair = next(
        dumped(row)
        for row in wire.mann_whitney
        if row.column == "value" and row.b == {"strain": "C"}
    )
    assert empty_pair["a"] == {"strain": "A"}
    assert (empty_pair["statistic"], empty_pair["p"], empty_pair["p_holm"]) == (
        None,
        None,
        None,
    )
    assert empty_pair["reason"] == "no_values"
    assert empty_pair["comparisons"] == 1


def test_the_wire_keeps_the_degrees_of_freedom_an_integer_and_null_without_groups(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    tested = wire_of(summarize_run(ds, result, by=["strain"], tests=["kruskal"]))
    alone = wire_of(
        summarize_run(
            ds,
            result,
            by=["strain"],
            where={"strain": ["A"]},
            tests=["kruskal"],
        )
    )

    other = next(row for row in tested.kruskal if row.column == "other")
    assert (other.k, other.n, other.df, other.reason) == (3, 9, 2, None)
    assert isinstance(other.df, int)
    assert other.statistic is not None and other.p is not None
    assert [dumped(row) for row in alone.kruskal] == [
        {
            "column": column,
            "k": 1,
            "n": 4,
            "statistic": None,
            "df": None,
            "p": None,
            "reason": "too_few_groups",
        }
        for column in ("value", "other")
    ]


def test_the_wire_lists_entries_as_group_and_sequence_pairs(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(strain=["A"], value=[1.0]),
            ("g", "c"): table(
                strain=pd.Series([], dtype="object"),
                value=pd.Series([], dtype="float64"),
            ),
        },
    )

    wire = wire_of(summarize_run(ds, result, by=["strain"]))

    assert wire.n_entries == 2
    assert wire.entries_without_rows == (("g", "c"),)
    assert dumped(wire)["entries_without_rows"] == [["g", "c"]]


def test_the_wire_keeps_each_key_in_its_type(tmp_path: Path) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(
                strain=["A", "A"],
                n_inds=np.array([1, 1], dtype=np.int64),
                fed=[True, True],
                size=[0.5, 0.5],
                value=[1.0, 2.0],
            ),
            ("g", "b"): table(
                strain=["B", "B"],
                n_inds=np.array([2, 2], dtype=np.int64),
                fed=[False, False],
                size=[1.5, 1.5],
                value=[3.0, 4.0],
            ),
        },
    )

    wire = wire_of(
        summarize_run(
            ds,
            result,
            by=["strain", "n_inds", "fed", "size"],
            tests=["mannwhitney"],
        )
    )

    key = wire.groups[0].key
    assert key == {"strain": "A", "n_inds": 1, "fed": True, "size": 0.5}
    assert {name: type(value) for name, value in key.items()} == {
        "strain": str,
        "n_inds": int,
        "fed": bool,
        "size": float,
    }
    (pair,) = wire.mann_whitney
    assert pair.a == key
    assert pair.b == {"strain": "B", "n_inds": 2, "fed": False, "size": 1.5}


def test_the_wire_regroups_the_pair_columns_of_keys_that_end_in_a_side(
    tmp_path: Path,
) -> None:
    ds, result = run_of(
        tmp_path,
        {
            ("g", "a"): table(x=["p", "p"], x_a=["q", "q"], value=[1.0, 2.0]),
            ("g", "b"): table(x=["p", "p"], x_a=["r", "r"], value=[3.0, 4.0]),
        },
    )

    wire = wire_of(summarize_run(ds, result, by=["x", "x_a"], tests=["mannwhitney"]))

    (pair,) = wire.mann_whitney
    assert pair.a == {"x": "p", "x_a": "q"}
    assert pair.b == {"x": "p", "x_a": "r"}


def test_the_wire_validates_from_its_own_dump(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)
    wire = wire_of(
        summarize_run(ds, result, by=["strain"], tests=["kruskal", "mannwhitney"])
    )

    assert RunSummaryWire.model_validate(dumped(wire)) == wire


def test_a_summary_without_keys_has_one_group_with_an_empty_key(
    tmp_path: Path,
) -> None:
    ds, result = three_strains(tmp_path)

    wire = wire_of(summarize_run(ds, result, tests=["mannwhitney"]))

    assert [dumped(group) for group in wire.groups] == [
        {"key": {}, "n_rows": 10, "n_entries": 4}
    ]
    assert wire.mann_whitney == ()


# --- the output of preview_output -------------------------------------------

_OUTPUT_DROPPED = frozenset({"title", "description", "default"})
"""The keywords an MCP output schema leaves out."""


def stripped_output_schema() -> dict[str, JsonValue]:
    schema: dict[str, JsonValue] = PreviewOutput.model_json_schema(mode="serialization")
    return strip_schema_keywords(schema, _OUTPUT_DROPPED)


def branches_accepting(schema: Mapping[str, JsonValue], value: JsonValue) -> list[int]:
    """The ``anyOf`` branches whose required and permitted properties *value* meets.

    Both members are strict models, whose schemas list every field as required and
    forbid others, so a result is told apart by these two keywords alone.
    """
    definitions = schema["$defs"]
    branches = schema["anyOf"]
    assert isinstance(definitions, dict) and isinstance(branches, list)
    assert isinstance(value, dict)
    accepted: list[int] = []
    for position, branch in enumerate(branches):
        assert isinstance(branch, dict)
        reference = branch["$ref"]
        assert isinstance(reference, str)
        definition = definitions[reference.removeprefix("#/$defs/")]
        assert isinstance(definition, dict)
        required = definition["required"]
        properties = definition["properties"]
        assert isinstance(required, list) and isinstance(properties, dict)
        assert definition["additionalProperties"] is False
        if set(required) <= set(value) and set(value) <= set(properties):
            accepted.append(position)
    return accepted


def an_artifact_preview() -> ArtifactPreview:
    return ArtifactPreview(
        ref="feature/prov-labeled/0.1-0123456789",
        group="g",
        sequence="a",
        path="/dataset/features/a.parquet",
        columns=(),
        n_rows=0,
        frame_min=None,
        frame_max=None,
        read_columns=(),
        rows=(),
        has_more=False,
    )


def test_the_output_schema_is_an_object_of_two_branches() -> None:
    schema = PreviewOutput.model_json_schema(mode="serialization")

    assert schema["type"] == "object"
    assert [branch["$ref"].rpartition("/")[2] for branch in schema["anyOf"]] == [
        "ArtifactPreview",
        "RunSummaryWire",
    ]


def test_each_member_validates_against_its_own_branch_only(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)
    schema = stripped_output_schema()

    preview = dumped(an_artifact_preview())
    summary = dumped(wire_of(summarize_run(ds, result, by=["strain"])))

    assert branches_accepting(schema, preview) == [0]
    assert branches_accepting(schema, summary) == [1]


def test_the_output_model_validates_each_member_as_itself(tmp_path: Path) -> None:
    ds, result = three_strains(tmp_path)
    wire = wire_of(summarize_run(ds, result, by=["strain"]))
    preview = an_artifact_preview()

    assert PreviewOutput.model_validate(dumped(preview)).root == preview
    assert PreviewOutput.model_validate(dumped(wire)).root == wire


def test_the_package_exports_the_summary_lazily() -> None:
    assert pipeline_package.summarize_run is summarize_run
    assert pipeline_package.RunSummary is RunSummary
    assert pipeline_package.RunSummaryWire is RunSummaryWire
