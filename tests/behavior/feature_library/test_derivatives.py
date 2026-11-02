"""The keypoint position read that the keypoint features share."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library.derivatives import keypoint_positions


def test_positions_follow_the_row_order_and_a_non_finite_value_is_nan() -> None:
    df = pd.DataFrame(
        {
            "poseX0": [1.0, np.inf, 3.0],
            "poseY0": [4.0, 5.0, -np.inf],
            "poseX1": [7.0, 8.0, 9.0],
        }
    )
    order = np.array([2, 0, 1], dtype=np.int64)

    positions = keypoint_positions(df, ("poseX0", "poseY0"), order)

    assert positions.dtype == np.float64
    np.testing.assert_array_equal(positions, [[3.0, np.nan], [1.0, 4.0], [np.nan, 5.0]])
    assert np.isinf(df["poseX0"].iloc[1])
