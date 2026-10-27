"""One individual's rows at a time, in frame order.

The features and the resampling of a tracks table read one individual at a time.
Each refuses an individual with two rows of one frame.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd
import pytest

from mosaic.core.individuals import (
    id_arenas,
    individual_pairs,
    iter_individuals,
    refuse_repeated_rows,
    same_arena_mask,
)


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


# --- Repeated rows ---


def test_rows_of_distinct_frames_per_id_are_accepted() -> None:
    refuse_repeated_rows(_rows(), "feature")


def test_a_repeated_frame_is_named_with_its_id() -> None:
    rows = pd.DataFrame({"frame": [1, 2, 2, 2], "id": ["a", "a", "b", "a"]})

    with pytest.raises(
        ValueError, match=r"^feature: frame 2 of id a appears in more than one row$"
    ):
        refuse_repeated_rows(rows, "feature")


def test_a_repeated_frame_of_a_table_without_ids_is_named() -> None:
    """A table without an id column is one individual."""
    with pytest.raises(
        ValueError, match=r"^feature: frame 3 appears in more than one row$"
    ):
        refuse_repeated_rows(_rows().drop(columns="id"), "feature")


def test_another_column_can_key_the_rows() -> None:
    rows = _rows()
    rows.loc[[10, 12], "time"] = 0.0

    refuse_repeated_rows(rows, "feature")
    with pytest.raises(ValueError, match=r"^feature: time 0.0 of id 7 appears"):
        refuse_repeated_rows(rows, "feature", order_col="time")


def test_a_refusal_gives_the_remedy_it_is_handed() -> None:
    rows = pd.DataFrame({"frame": [0, 0], "id": [1, 1]})

    with pytest.raises(
        ValueError,
        match=r"^table: frame 0 of id 1 appears in more than one row\. Renumber it\.$",
    ):
        refuse_repeated_rows(rows, "table", remedy="Renumber it.")


def test_without_arenas_every_two_individuals_pair() -> None:
    ids = [3, 1, 2]

    assert individual_pairs(ids, None) == [(3, 1), (3, 2), (1, 2)]
    assert len(individual_pairs(ids, None, ordered=True)) == 6


def test_with_arenas_only_individuals_of_one_arena_pair() -> None:
    arenas: dict[Hashable, str] = {1: "A1", 2: "A1", 3: "B1", 4: "B1"}

    assert individual_pairs([1, 2, 3, 4, 5], arenas) == [(1, 2), (3, 4)]
    assert individual_pairs([1, 2, 3], arenas, ordered=True) == [(1, 2), (2, 1)]
    assert same_arena_mask([1, 1, 5], [2, 3, 5], arenas).tolist() == [
        True,
        False,
        False,
    ]


def test_each_id_s_arena_is_read_from_the_table() -> None:
    table = pd.DataFrame({"id": [0, 1, 0, 2], "arena": ["A1", "B1", "A1", None]})

    assert id_arenas(table) == {0: "A1", 1: "B1"}
    assert id_arenas(table.drop(columns="arena")) is None


def test_an_id_in_two_arenas_is_refused() -> None:
    table = pd.DataFrame({"id": [0, 0], "arena": ["A1", "B1"]})

    with pytest.raises(ValueError, match="id 0 appears in two arenas"):
        _ = id_arenas(table)
