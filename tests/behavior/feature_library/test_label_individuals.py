"""One label per row of each individual, in the order that a feature reads them."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.behavior.feature_library.helpers import label_individuals


def test_each_individual_is_labeled_in_frame_order() -> None:
    df = pd.DataFrame(
        {"frame": [2, 1, 2, 1], "id": [7, 7, 3, 3], "value": [72, 71, 32, 31]}
    )
    asked: list[object] = []

    def label(individual: object, rows: pd.DataFrame) -> npt.NDArray[np.int64]:
        asked.append(individual)
        return rows["value"].to_numpy() * 10

    labeled = label_individuals(df, label)

    assert asked == [7, 3]
    expected = pd.DataFrame(
        {
            "frame": [1, 2, 1, 2],
            "syllable": [710, 720, 310, 320],
            "id": [7, 7, 3, 3],
        }
    )
    pd.testing.assert_frame_equal(labeled, expected)


def test_a_table_without_an_id_column_is_one_individual_without_an_id() -> None:
    df = pd.DataFrame({"frame": [1, 0], "value": [1, 0]})

    labeled = label_individuals(
        df, lambda individual, rows: rows["value"].to_numpy(), column="state"
    )

    expected = pd.DataFrame({"frame": [0, 1], "state": [0, 1]})
    pd.testing.assert_frame_equal(labeled, expected)
