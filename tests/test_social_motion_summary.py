"""Unit tests for SocialMotionSummary against hand-computed values.

A small 2-fish, 5-frame scenario (fps=1) is built so every emitted per-fish
scalar can be checked by hand. Fish 0 has a linearly increasing speed
[1,2,3,4,5]; Fish 1 is constant at 2. The two are always neighbours
(group_size=2, shared group_membership).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.social_motion_summary import SocialMotionSummary


def _make_merged_df() -> pd.DataFrame:
    """The merged (nn + speed + ffgroups) frame the pipeline would hand to apply()."""
    frames = [0, 1, 2, 3, 4]
    rows = []
    speed0 = [1.0, 2.0, 3.0, 4.0, 5.0]
    speed1 = [2.0, 2.0, 2.0, 2.0, 2.0]
    for k, f in enumerate(frames):
        # Fish 0: neighbour (fish 1) directly ahead, aligned heading
        rows.append(
            {
                "frame": f,
                "id": 0,
                "sequence": "seq1",
                "group": "g",
                "speed": speed0[k],
                "nn_id": 1.0,
                "nn_delta_angle": 0.0,          # cos = 1  -> nn_align = 1
                "nn_delta_x_ego": 1.0,          # ahead    -> frac_nn_ahead = 1
                "nn_delta_y_ego": 0.0,
                "group_membership": 0,
                "group_size": 2,
            }
        )
        # Fish 1: neighbour (fish 0) directly behind, anti-aligned heading
        rows.append(
            {
                "frame": f,
                "id": 1,
                "sequence": "seq1",
                "group": "g",
                "speed": speed1[k],
                "nn_id": 0.0,
                "nn_delta_angle": math.pi,      # cos = -1 -> nn_align = -1
                "nn_delta_x_ego": -1.0,         # behind   -> frac_nn_ahead = 0
                "nn_delta_y_ego": 0.0,
                "group_membership": 0,
                "group_size": 2,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def result() -> pd.DataFrame:
    feat = SocialMotionSummary(params={"fps": 1.0})
    return feat.apply(_make_merged_df()).set_index("id")


def test_one_row_per_fish(result: pd.DataFrame) -> None:
    assert sorted(result.index.tolist()) == [0, 1]
    assert result.loc[0, "sequence"] == "seq1"
    assert result.loc[0, "group"] == "g"


def test_nn_alignment(result: pd.DataFrame) -> None:
    assert result.loc[0, "nn_align"] == pytest.approx(1.0)
    assert result.loc[1, "nn_align"] == pytest.approx(-1.0)


def test_neighbor_bearing(result: pd.DataFrame) -> None:
    assert result.loc[0, "frac_nn_ahead"] == pytest.approx(1.0)
    assert result.loc[1, "frac_nn_ahead"] == pytest.approx(0.0)
    assert result.loc[0, "nn_bearing_x"] == pytest.approx(1.0)
    assert result.loc[1, "nn_bearing_x"] == pytest.approx(-1.0)
    assert result.loc[0, "nn_bearing_y"] == pytest.approx(0.0)


def test_speed_match_nn(result: pd.DataFrame) -> None:
    # |own - neighbour speed| averaged: [1,0,1,2,3] / 5 = 1.4 for both fish
    assert result.loc[0, "speed_match_nn"] == pytest.approx(1.4)
    assert result.loc[1, "speed_match_nn"] == pytest.approx(1.4)


def test_speed_match_group(result: pd.DataFrame) -> None:
    # |speed - group-mean| averaged: [0.5,0,0.5,1,1.5] / 5 = 0.7 for both fish
    assert result.loc[0, "speed_match_group"] == pytest.approx(0.7)
    assert result.loc[1, "speed_match_group"] == pytest.approx(0.7)


def test_speed_dispersion(result: pd.DataFrame) -> None:
    # Fish 0 speed [1..5]: std=sqrt(2), mean=3 -> cv; IQR=2, median=3 -> rcv
    assert result.loc[0, "speed_cv"] == pytest.approx(math.sqrt(2) / 3)
    assert result.loc[0, "speed_rcv"] == pytest.approx(2.0 / 3.0)
    # Fish 1 constant -> zero dispersion
    assert result.loc[1, "speed_cv"] == pytest.approx(0.0)
    assert result.loc[1, "speed_rcv"] == pytest.approx(0.0)


def test_accel_jerk(result: pd.DataFrame) -> None:
    # Fish 0: accel = [1,1,1,1] -> mean/med 1, zero dispersion; jerk all 0
    assert result.loc[0, "accel_mean"] == pytest.approx(1.0)
    assert result.loc[0, "accel_med"] == pytest.approx(1.0)
    assert result.loc[0, "accel_cv"] == pytest.approx(0.0)
    assert result.loc[0, "jerk_mean"] == pytest.approx(0.0)
    assert np.isnan(result.loc[0, "jerk_cv"])  # std/mean with mean 0 -> nan
    # Fish 1: constant speed -> zero acceleration magnitude
    assert result.loc[1, "accel_mean"] == pytest.approx(0.0)


def test_group_free_operation() -> None:
    """Without group_membership, speed_match_group is NaN but nn metrics still work."""
    df = _make_merged_df().drop(columns=["group_membership"])
    res = SocialMotionSummary(params={"fps": 1.0}).apply(df).set_index("id")
    assert np.isnan(res.loc[0, "speed_match_group"])
    assert res.loc[0, "speed_match_nn"] == pytest.approx(1.4)
    assert res.loc[0, "nn_align"] == pytest.approx(1.0)


def test_burst_coast_optional() -> None:
    """Burst-coast columns appear only when the flag is set."""
    df = _make_merged_df()
    off = SocialMotionSummary(params={"fps": 1.0}).apply(df)
    assert "kick_rate" not in off.columns
    on = SocialMotionSummary(
        params={"fps": 1.0, "compute_burst_coast": True}
    ).apply(df)
    assert "kick_rate" in on.columns
    assert "burst_coast_ratio" in on.columns


def test_empty_dataframe() -> None:
    assert SocialMotionSummary().apply(pd.DataFrame()).empty
