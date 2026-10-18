"""A feature that reads rows as one sequence refuses a pair table.

Two rows per frame, one per perspective, would be one sequence of interleaved
animals. The refusal needs no model, so it is checked here without one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import refuse_pair_rows


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "id1": [4, 9, 4, 9],
            "id2": [9, 4, 9, 4],
            "perspective": [0, 1, 0, 1],
            "value": [0.0, 1.0, 2.0, 3.0],
        }
    )


def test_a_pair_table_is_refused_naming_the_feature() -> None:
    with pytest.raises(
        ValueError,
        match=r"^\[feature\] the input is a pair table \(id1, id2, perspective\)",
    ):
        refuse_pair_rows(_pairs(), "feature")


def test_a_table_with_the_perspective_alone_is_a_pair_table() -> None:
    """Its rows are interleaved by perspective as a pair's are."""
    with pytest.raises(ValueError, match=r"pair table \(perspective\)"):
        refuse_pair_rows(_pairs().drop(columns=["id1", "id2"]), "feature")


def test_an_individual_table_and_a_table_without_identity_are_accepted() -> None:
    individuals = pd.DataFrame({"frame": [0, 1], "id": [7, 7], "value": [0.0, 1.0]})

    refuse_pair_rows(individuals, "feature")
    refuse_pair_rows(individuals.drop(columns="id"), "feature")
