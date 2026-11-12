"""trajectory-smooth's bridge check and interpolation marks.

Every track is one individual at one row per second, so the speed threshold in
units per second reads as units per row, and a bridge speed is a distance over a
row count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library.helpers import feature_columns
from mosaic.behavior.feature_library.trajectory_smooth import TrajectorySmooth
from mosaic.core.pipeline.column_contract import Column


def _track(x: list[float], y: list[float] | None = None) -> pd.DataFrame:
    n = len(x)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "time": np.arange(n, dtype=float),
            "id": np.zeros(n, dtype=np.int64),
            "sequence": "s",
            "group": "g",
            "X": np.asarray(x, dtype=float),
            "Y": np.zeros(n) if y is None else np.asarray(y, dtype=float),
        }
    )


def _smooth(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    defaults: dict[str, object] = {"speed_threshold": 5.0, "expand_frames": 0}
    return TrajectorySmooth(params={**defaults, **params}).apply(df)


NAN = float("nan")


def test_a_gap_bridged_too_fast_is_unbridgeable() -> None:
    # Observed at 0..3 at x=0, missing 4..6, observed again at 7.. at x=100:
    # 100 units over 4 frames is 25 per frame, above the threshold of 5.
    out = _smooth(_track([0, 0, 0, 0, NAN, NAN, NAN, 100, 100, 100]))
    assert out["unbridgeable"].tolist() == [False] * 4 + [True] * 3 + [False] * 3
    assert out["interpolated"].tolist() == [False] * 4 + [True] * 3 + [False] * 3
    assert np.isfinite(out["X"]).all()


def test_a_gap_bridged_slowly_is_not_unbridgeable() -> None:
    # 8 units over 4 frames is 2 per frame.
    out = _smooth(_track([0, 0, 0, 0, NAN, NAN, NAN, 8, 8, 8]))
    assert not out["unbridgeable"].any()
    assert out["interpolated"].sum() == 3
    np.testing.assert_allclose(out["X"].iloc[4:7], [2.0, 4.0, 6.0])


def test_a_persistent_jump_in_continuous_data_is_unbridgeable() -> None:
    # No missing position: the speed gate flags frame 5, expansion widens it to
    # 3..7, and the bridge from frame 2 to frame 8 spans 100 units in 6 frames.
    x = [0.0] * 5 + [100.0] * 5
    out = _smooth(_track(x), expand_frames=2)
    flagged = [False] * 3 + [True] * 5 + [False] * 2
    assert out["bad_frame"].tolist() == flagged
    assert out["unbridgeable"].tolist() == flagged
    assert out["interpolated"].tolist() == flagged


def test_interp_run_frames_and_long_interp_on_a_known_gap() -> None:
    x = [0.0] * 3 + [NAN] * 4 + [0.0] * 3 + [NAN] * 3 + [0.0] * 2
    out = _smooth(_track(x), long_run_frames=3)
    assert out["interp_run_frames"].tolist() == (
        [0] * 3 + [4] * 4 + [0] * 3 + [3] * 3 + [0] * 2
    )
    # A run of exactly long_run_frames is not long.
    assert out["long_interp"].tolist() == [False] * 3 + [True] * 4 + [False] * 8


def test_edge_fills_are_interpolated_but_not_unbridgeable() -> None:
    out = _smooth(_track([NAN, NAN, 0, 0, 0, 1000, 1000, NAN]))
    # The speed gate flags frame 5, the one run with an observed row on both sides.
    assert out["interpolated"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    assert out["interp_run_frames"].tolist() == [2, 2, 0, 0, 0, 1, 0, 1]
    assert out["unbridgeable"].tolist() == [False] * 5 + [True] + [False] * 2


def test_without_centroid_interpolation_the_marks_are_empty() -> None:
    out = _smooth(_track([0, 0, NAN, NAN, 100, 100]), interpolate_centroid=False)
    assert not out["interpolated"].any()
    assert not out["unbridgeable"].any()
    assert not out["long_interp"].any()
    assert (out["interp_run_frames"] == 0).all()
    assert out["X"].isna().sum() == 2


def test_without_a_speed_threshold_no_run_is_unbridgeable() -> None:
    out = _smooth(_track([0, 0, NAN, NAN, 100, 100]), speed_threshold=None)
    assert out["interpolated"].tolist() == [False, False, True, True, False, False]
    assert not out["unbridgeable"].any()


def test_the_bridge_uses_both_axes() -> None:
    # 3-4-5: the bridge spans 50 units over 2 frames, 25 per frame.
    out = _smooth(_track([0, NAN, 30], [0, NAN, 40]), speed_threshold=24.0)
    assert out["unbridgeable"].tolist() == [False, True, False]
    out = _smooth(_track([0, NAN, 30], [0, NAN, 40]), speed_threshold=26.0)
    assert not out["unbridgeable"].any()


def test_the_marks_are_declared_and_the_run_length_is_not_payload() -> None:
    added = {
        column.name
        for column in TrajectorySmooth.provides.adds
        if isinstance(column, Column)
    }
    assert {"interpolated", "interp_run_frames", "unbridgeable", "long_interp"} <= added
    out = _smooth(_track([0, NAN, 1]))
    assert feature_columns(out) == ["X", "Y"]
