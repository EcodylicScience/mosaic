"""``nearest-neighbor`` over frames where an individual's position is missing.

A tracker leaves a row with NaN coordinates for an individual it lost in a frame.
That individual is nobody's neighbor there, and every other individual still has
one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor


def _frame(x: list[float], y: list[float]) -> pd.DataFrame:
    """One frame of individuals 0..n-1 at (x, y), each heading along +x."""
    n = len(x)
    return pd.DataFrame(
        {
            "frame": [0] * n,
            "time": [0.0] * n,
            "id": list(range(n)),
            "X": x,
            "Y": y,
            "ANGLE": [0.0] * n,
        }
    )


def _by_id(out: pd.DataFrame) -> pd.DataFrame:
    return out.set_index("id").sort_index()


def test_a_missing_individual_is_nobodys_neighbor() -> None:
    """Individual 2 has no position; 0 and 1 are still each other's neighbor."""
    out = _by_id(
        NearestNeighbor().apply(_frame([0.0, 3.0, np.nan], [0.0, 4.0, np.nan]))
    )

    assert out.loc[0, "nn_id"] == 1
    assert out.loc[1, "nn_id"] == 0
    assert out.loc[0, "nn_dist"] == 5.0
    assert out.loc[1, "nn_dist"] == 5.0
    assert out.loc[0, "nn_delta_x"] == 3.0


def test_an_individual_without_a_position_has_no_neighbor() -> None:
    out = _by_id(
        NearestNeighbor().apply(_frame([0.0, 3.0, np.nan], [0.0, 4.0, np.nan]))
    )

    missing = out.loc[[2], ["nn_id", "nn_dist", "nn_delta_x_ego"]]
    assert bool(missing.isna().to_numpy().all())


def test_an_individual_whose_others_are_all_missing_has_no_neighbor() -> None:
    out = _by_id(
        NearestNeighbor().apply(_frame([0.0, np.nan, np.nan], [0.0, np.nan, np.nan]))
    )

    assert out["nn_dist"].isna().all()
    assert out["nn_id"].isna().all()


def test_a_complete_frame_is_unchanged() -> None:
    """The closest of the others, as before, in the focal individual's frame."""
    out = _by_id(NearestNeighbor().apply(_frame([0.0, 1.0, 10.0], [0.0, 0.0, 0.0])))

    assert out["nn_id"].tolist() == [1, 0, 1]
    assert out["nn_dist"].tolist() == [1.0, 1.0, 9.0]
    assert out.loc[0, "nn_delta_x_ego"] == 1.0
