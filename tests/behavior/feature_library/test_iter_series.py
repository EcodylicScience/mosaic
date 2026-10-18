"""One series of rows at a time: an individual, an ordered pair, or the whole table.

A model that reads rows in order must not read one animal's rows after another's.
``iter_series`` splits a table where its identity columns say to, and
``label_series`` writes the labels back with those columns so that they join to the
rows they describe.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import (
    iter_series,
    label_individuals,
    label_series,
    refuse_repeated_series_rows,
    series_columns,
)
from mosaic.core.individuals import iter_individuals


def _pairs() -> pd.DataFrame:
    """Two frames of the pair (4, 9), the rows of the perspectives interleaved.

    The rows come in the order a merge of two inputs can produce: by frame, and
    the perspective 1 row of a frame before the perspective 0 row of the next.
    """
    return pd.DataFrame(
        {
            "frame": [2, 2, 1, 1],
            "perspective": [1, 0, 1, 0],
            "id1": [9, 4, 9, 4],
            "id2": [4, 9, 4, 9],
            "value": [92, 42, 91, 41],
        }
    )


def _individuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [2, 1, 2, 1],
            "id": [7, 7, 3, 3],
            "value": [72, 71, 32, 31],
        }
    )


# --- Which columns key a series ---


def test_an_id_column_keys_the_series() -> None:
    assert series_columns(_individuals()) == ["id"]


def test_the_id_wins_over_the_pair_columns() -> None:
    assert series_columns(_individuals().assign(id1=0, id2=1, perspective=0)) == ["id"]


def test_the_pair_columns_key_a_table_without_an_id() -> None:
    assert series_columns(_pairs()) == ["id1", "id2", "perspective"]


def test_a_table_keeps_only_the_pair_columns_it_has() -> None:
    """A pair-level table may carry the perspective without the ids."""
    assert series_columns(_pairs().drop(columns=["id1", "id2"])) == ["perspective"]


def test_a_table_without_identity_is_one_series() -> None:
    assert series_columns(_pairs()[["frame", "value"]]) == []


# --- Iterating ---


def test_pair_series_come_sorted_by_their_key_with_frame_ordered_rows() -> None:
    series = dict(iter_series(_pairs()))

    assert list(series) == [(4, 9, 0), (9, 4, 1)]
    assert series[(4, 9, 0)]["value"].tolist() == [41, 42]
    assert series[(9, 4, 1)]["value"].tolist() == [91, 92]
    assert series[(4, 9, 0)].index.tolist() == [0, 1]


def test_the_row_order_of_a_pair_table_decides_nothing() -> None:
    rows = _pairs()
    shuffled = rows.iloc[[3, 0, 2, 1]]

    for (key, expected), (shuffled_key, actual) in zip(
        iter_series(rows), iter_series(shuffled), strict=True
    ):
        assert shuffled_key == key
        pd.testing.assert_frame_equal(actual, expected)


def test_an_id_table_is_split_exactly_as_its_individuals_are() -> None:
    """kpms and vame read individuals, and a series of ids is the same split."""
    rows = _individuals()

    series = list(iter_series(rows))
    individuals = list(iter_individuals(rows))

    assert [key for key, _ in series] == [(7,), (3,)]
    for (_, from_series), (_, from_individuals) in zip(
        series, individuals, strict=True
    ):
        pd.testing.assert_frame_equal(from_series, from_individuals)


def test_a_table_without_identity_is_yielded_whole_in_frame_order() -> None:
    [(key, rows)] = list(iter_series(_pairs()[["frame", "value"]]))

    assert key == ()
    assert rows["frame"].tolist() == [1, 1, 2, 2]
    assert rows.index.tolist() == [0, 1, 2, 3]


def test_another_column_can_order_the_rows() -> None:
    rows = _pairs().assign(time=[0.1, 0.2, 0.3, 0.4])

    series = dict(iter_series(rows, order_col="time"))

    assert series[(4, 9, 0)]["value"].tolist() == [42, 41]


def test_a_row_missing_part_of_its_pair_key_belongs_to_no_series() -> None:
    rows = _pairs().astype({"id2": float})
    rows.loc[0, "id2"] = np.nan

    series = dict(iter_series(rows))

    assert list(series) == [(4.0, 9.0, 0), (9.0, 4.0, 1)]
    assert sum(len(found) for found in series.values()) == 3


def test_the_table_is_left_as_it_was() -> None:
    rows = _pairs()
    before = rows.copy()

    _ = list(iter_series(rows))

    pd.testing.assert_frame_equal(rows, before)


# --- Labeling ---


def test_each_pair_series_is_labeled_and_carries_its_key() -> None:
    asked: list[object] = []

    def label(key: object, rows: pd.DataFrame) -> npt.NDArray[np.int64]:
        asked.append(key)
        return rows["value"].to_numpy() * 10

    labeled = label_series(_pairs(), label)

    assert asked == [(4, 9, 0), (9, 4, 1)]
    expected = pd.DataFrame(
        {
            "frame": [1, 2, 1, 2],
            "syllable": [410, 420, 910, 920],
            "id1": [4, 4, 9, 9],
            "id2": [9, 9, 4, 4],
            "perspective": [0, 0, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(labeled, expected)


def test_the_labels_join_back_to_the_rows_they_describe() -> None:
    rows = _pairs()

    labeled = label_series(rows, lambda key, series: series["value"].to_numpy())

    joined = rows.merge(
        labeled,
        on=["frame", "id1", "id2", "perspective"],
        how="outer",
        validate="one_to_one",
    )
    assert len(joined) == len(rows)
    assert (joined["syllable"] == joined["value"]).all()


def test_the_key_columns_keep_their_dtype() -> None:
    rows = _pairs().astype({"id1": "int32", "id2": "int32", "perspective": "int8"})

    labeled = label_series(rows, lambda key, series: np.zeros(len(series), dtype=int))

    assert labeled["id1"].dtype == np.int32
    assert labeled["perspective"].dtype == np.int8


def test_an_id_table_is_labeled_as_label_individuals_labels_it() -> None:
    rows = _individuals()

    def label(key: object, series: pd.DataFrame) -> npt.NDArray[np.int64]:
        return series["value"].to_numpy() * 10

    pd.testing.assert_frame_equal(
        label_series(rows, label),
        label_individuals(rows, label),
    )


def test_a_table_without_identity_is_labeled_with_its_frame_alone() -> None:
    rows = pd.DataFrame({"frame": [1, 0], "value": [1, 0]})

    labeled = label_series(
        rows, lambda key, series: series["value"].to_numpy(), column="state"
    )

    pd.testing.assert_frame_equal(
        labeled, pd.DataFrame({"frame": [0, 1], "state": [0, 1]})
    )


def test_a_table_with_no_series_is_labeled_with_no_rows() -> None:
    empty = _pairs().iloc[:0]

    labeled = label_series(empty, lambda key, series: np.zeros(len(series), dtype=int))

    assert labeled.empty
    assert list(labeled.columns) == ["frame", "syllable", "id1", "id2", "perspective"]


# --- Repeated rows ---


def test_the_perspectives_of_one_frame_are_not_repeated_rows() -> None:
    refuse_repeated_series_rows(_pairs(), "feature")
    refuse_repeated_series_rows(_individuals(), "feature")


def test_a_repeated_frame_of_a_pair_series_is_named_with_its_key() -> None:
    rows = pd.concat([_pairs(), _pairs().iloc[[1]]], ignore_index=True)

    with pytest.raises(
        ValueError,
        match=(
            r"^feature: frame 2 of id1 4, id2 9, perspective 0 "
            r"appears in more than one row$"
        ),
    ):
        refuse_repeated_series_rows(rows, "feature")


@pytest.mark.parametrize(
    ("rows", "where"),
    [
        (
            pd.DataFrame({"frame": [1, 2, 2, 2], "id": ["a", "a", "b", "a"]}),
            "frame 2 of id a",
        ),
        (pd.DataFrame({"frame": [3, 1, 3]}), "frame 3"),
    ],
    ids=["id", "none"],
)
def test_a_repeated_frame_is_named_as_the_individuals_rule_names_it(
    rows: pd.DataFrame, where: str
) -> None:
    with pytest.raises(
        ValueError, match=rf"^feature: {where} appears in more than one row$"
    ):
        refuse_repeated_series_rows(rows, "feature")


def test_a_row_in_no_series_repeats_nothing() -> None:
    """Two rows missing their key are read in no series, so no frame repeats."""
    rows = pd.DataFrame(
        {
            "frame": [1, 1, 2],
            "id1": [np.nan, np.nan, 4.0],
            "id2": [9, 9, 9],
            "perspective": [0, 0, 0],
        }
    )

    refuse_repeated_series_rows(rows, "feature")
