"""One individual's rows at a time, in the order a feature reads them."""

from __future__ import annotations

import pandas as pd

from mosaic.behavior.feature_library.helpers import iter_individuals


def _rows() -> pd.DataFrame:
    """Frame-major rows of ids 7, 2 and 5, with frames and times out of order."""
    return pd.DataFrame(
        {
            "frame": [3, 3, 1, 1, 2, 2, 1],
            "time": [0.0, 0.1, 0.5, 0.6, 0.2, 0.3, 0.4],
            "id": [7, 2, 7, 2, 7, 2, 5],
            "value": [73, 23, 71, 21, 72, 22, 51],
        },
        index=[10, 11, 12, 13, 14, 15, 16],
    )


def test_individuals_come_in_the_order_they_first_appear() -> None:
    """Sorting the ids would reorder a feature's output and every key built from it."""
    assert [individual for individual, _ in iter_individuals(_rows())] == [7, 2, 5]


def test_each_individual_is_ordered_by_frame_with_a_fresh_index() -> None:
    by_id = dict(iter_individuals(_rows()))

    assert by_id[7]["frame"].tolist() == [1, 2, 3]
    assert by_id[7]["value"].tolist() == [71, 72, 73]
    assert by_id[2]["value"].tolist() == [21, 22, 23]
    assert by_id[7].index.tolist() == [0, 1, 2]


def test_another_column_can_order_the_rows() -> None:
    by_id = dict(iter_individuals(_rows(), order_col="time"))

    assert by_id[7]["time"].tolist() == [0.0, 0.2, 0.5]
    assert by_id[2]["time"].tolist() == [0.1, 0.3, 0.6]


def test_a_table_without_ids_is_one_individual() -> None:
    rows = _rows().drop(columns="id")

    [(individual, ordered)] = list(iter_individuals(rows))

    assert individual is None
    assert ordered["frame"].tolist() == [1, 1, 1, 2, 2, 3, 3]
    assert ordered.index.tolist() == list(range(7))


def test_the_input_table_is_left_as_it_was() -> None:
    rows = _rows()
    before = rows.copy()

    _ = list(iter_individuals(rows))

    pd.testing.assert_frame_equal(rows, before)


def test_a_row_without_an_id_belongs_to_no_individual() -> None:
    rows = _rows()
    rows["id"] = rows["id"].astype(float)
    rows.loc[16, "id"] = float("nan")

    by_id = dict(iter_individuals(rows))

    assert list(by_id) == [7.0, 2.0]
    assert sum(len(individual_rows) for individual_rows in by_id.values()) == 6


def test_rows_at_one_frame_keep_their_order() -> None:
    """A stable sort gives one row order on every platform."""
    rows = pd.DataFrame(
        {"frame": [2, 1] * 20 + [1], "id": [4] * 41, "value": list(range(41))}
    )

    [(_, ordered)] = list(iter_individuals(rows))

    assert ordered["value"].tolist() == [*range(1, 41, 2), 40, *range(0, 40, 2)]
