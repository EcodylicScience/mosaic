"""Features read their rates from ``time`` on a table that dropped frames.

Every table here is the shared fixture of ``tests.helpers.dropped_frames``: rows
0.0189 s apart with gaps of up to 1.45 s, and individuals that move at a constant
velocity. A feature that counts frames, or divides by a constant rate, reads the
gaps as slow stretches or as jumps. One that differences on ``time`` reports the
constant velocity at every row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance
from mosaic.behavior.feature_library.collective_motion_metrics import (
    CollectiveMotionMetrics,
)
from mosaic.behavior.feature_library.neighbor_delta_response import (
    NeighborDeltaResponse,
)
from mosaic.behavior.feature_library.individual_wavelet import IndividualWavelet
from mosaic.behavior.feature_library.keypoint_kinematics import KeypointKinematics
from mosaic.behavior.feature_library.label_statistics import (
    LabelStatistics,
    ResultColumn,
)
from mosaic.behavior.feature_library.nn_delta_response import NearestNeighborDelta
from mosaic.behavior.feature_library.pair_wavelet import PairWavelet
from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.behavior.feature_library.social_motion_summary import SocialMotionSummary
from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
from scipy.ndimage import gaussian_filter1d
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.behavior.feature_library.trajectory_smooth import TrajectorySmooth

from mosaic.core.pipeline.types import Result

from tests.helpers import (
    ANGULAR_VELOCITY,
    GAP_SECONDS,
    SPEED,
    VELOCITY,
    dropped_frame_tracks,
    mean_rate,
)

AFTER_A_GAP = sorted(GAP_SECONDS)


# --- speed-angvel ----------------------------------------------------------


def test_speed_angvel_reports_the_constant_velocity_across_every_gap() -> None:
    out = SpeedAngvel().apply(dropped_frame_tracks())
    rest = out[out["frame"] != 0]
    np.testing.assert_allclose(rest["speed"], SPEED)
    np.testing.assert_allclose(rest["vx"], VELOCITY[0])
    np.testing.assert_allclose(rest["vy"], VELOCITY[1])
    np.testing.assert_allclose(rest["angvel"], ANGULAR_VELOCITY)


def test_speed_angvel_steps_of_several_rows_divide_by_their_seconds() -> None:
    out = SpeedAngvel(params={"step_size": 4}).apply(dropped_frame_tracks(ids=(0,)))
    np.testing.assert_allclose(out["speed_step"].iloc[4:], SPEED)
    np.testing.assert_allclose(out["angvel_step"].iloc[4:], ANGULAR_VELOCITY)


def test_speed_angvel_refuses_an_fps_by_name() -> None:
    with pytest.raises(ValidationError, match="fps"):
        _ = SpeedAngvel(params={"fps": 30.0})


def test_speed_angvel_refuses_a_table_without_time() -> None:
    table = dropped_frame_tracks(ids=(0,)).drop(columns="time")
    with pytest.raises(ValueError, match="time"):
        _ = SpeedAngvel().apply(table)


# --- collective-motion-metrics --------------------------------------------


def test_collective_motion_reports_the_group_speed_across_every_gap() -> None:
    out = CollectiveMotionMetrics().apply(dropped_frame_tracks(ids=(0, 1, 2, 3)))
    speed = out.set_index("frame")["centroid_speed"]
    np.testing.assert_allclose(speed.iloc[1:], SPEED)
    for row in AFTER_A_GAP:
        assert speed.loc[row] == pytest.approx(SPEED)


def test_collective_motion_max_frame_gap_still_counts_frames() -> None:
    table = dropped_frame_tracks(ids=(0, 1, 2, 3))
    table = table[table["frame"] != 90]
    out = CollectiveMotionMetrics(params={"max_frame_gap": 1}).apply(table)
    speed = out.set_index("frame")["centroid_speed"]
    assert np.isnan(speed.loc[91])
    assert speed.loc[92] == pytest.approx(SPEED)


def test_collective_motion_refuses_an_fps_by_name() -> None:
    with pytest.raises(ValidationError, match="fps"):
        _ = CollectiveMotionMetrics(params={"fps": 30.0})


# --- trajectory-smooth -----------------------------------------------------


def test_trajectory_smooth_allows_the_move_a_gap_allows() -> None:
    """A speed threshold of 6 per second passes a 7.25 unit move over 1.45 s."""
    table = dropped_frame_tracks(ids=(0,))
    out = TrajectorySmooth(params={"speed_threshold": 6.0, "expand_frames": 0}).apply(
        table
    )
    assert not out["bad_frame"].any()
    np.testing.assert_allclose(out["X"], table["X"])


def test_trajectory_smooth_still_flags_a_jump_in_one_step() -> None:
    table = dropped_frame_tracks(ids=(0,))
    table.loc[60:, "X"] += 50.0
    out = TrajectorySmooth(params={"speed_threshold": 6.0, "expand_frames": 0}).apply(
        table
    )
    assert out["bad_frame"].to_numpy().nonzero()[0].tolist() == [60]


def test_trajectory_smooth_refuses_an_fps_by_name() -> None:
    with pytest.raises(ValidationError, match="fps"):
        _ = TrajectorySmooth(params={"fps": 30.0})


def test_trajectory_smooth_refuses_a_speed_threshold_without_time() -> None:
    table = dropped_frame_tracks(ids=(0,)).drop(columns="time")
    with pytest.raises(ValueError, match="time"):
        _ = TrajectorySmooth(params={"speed_threshold": 6.0}).apply(table)


# --- pair-position and pair-egocentric -------------------------------------


def test_pair_position_velocities_come_from_the_time_column() -> None:
    out = PairPositionFeatures().apply(dropped_frame_tracks())
    np.testing.assert_allclose(out["A_speed"], SPEED)
    np.testing.assert_allclose(out["B_speed"], SPEED)
    np.testing.assert_allclose(out["A_ang_speed"], ANGULAR_VELOCITY)


def test_pair_position_keeps_time_for_what_reads_the_pairs() -> None:
    table = dropped_frame_tracks()
    out = PairPositionFeatures().apply(table)
    expected = table[table["id"] == 0].set_index("frame")["time"]
    first = out[(out["id1"] == 0) & (out["perspective"] == 0)].set_index("frame")
    np.testing.assert_allclose(first["time"], expected.loc[first.index])


def test_pair_position_refuses_an_fps_in_its_sampling_settings() -> None:
    with pytest.raises(ValidationError, match="fps_default"):
        _ = PairPositionFeatures(params={"sampling": {"fps_default": 30.0}})


def _with_keypoints(table: pd.DataFrame) -> pd.DataFrame:
    """Two keypoints a unit either side of the center, along the heading."""
    along = np.cos(table["ANGLE"]), np.sin(table["ANGLE"])
    keypoints = {
        "poseX0": table["X"] + along[0],
        "poseY0": table["Y"] + along[1],
        "poseX1": table["X"] - along[0],
        "poseY1": table["Y"] - along[1],
    }
    return table.assign(**keypoints)


def test_pair_egocentric_velocities_come_from_the_time_column() -> None:
    feature = PairEgocentricFeatures(
        params={
            "pose": {"pose_n": 2},
            "neck_idx": 0,
            "tail_base_idx": 1,
        }
    )
    out = feature.apply(_with_keypoints(dropped_frame_tracks()))
    np.testing.assert_allclose(out["A_speed"], SPEED)
    np.testing.assert_allclose(out["A_ang_speed"], ANGULAR_VELOCITY)


# --- approach-avoidance ---------------------------------------------------


def _approaching(table: pd.DataFrame) -> pd.DataFrame:
    """Two individuals closing on each other along x at 5 units per second."""
    moved = table.copy()
    moved["Y"] = 0.0
    moved["X"] = np.where(
        moved["id"] == 0,
        SPEED * moved["time"],
        1000.0 - SPEED * moved["time"],
    )
    return moved


def _events(table: pd.DataFrame, **params: object) -> pd.DataFrame:
    feature = ApproachAvoidance(
        params={
            "velocity_units": "per_second",
            "distance_threshold": 2000.0,
            "approacher_velocity_threshold": 4.9,
            "avoider_velocity_threshold": 0.0,
            "cos_approacher_threshold": 0.5,
            "cos_avoider_threshold": -1.0,
            "orientation_gate_cos": None,
            "min_event_length": 1,
            "min_event_count": 1,
            **params,
        }
    )
    return feature.apply(table)


def test_approach_avoidance_speeds_in_units_per_second_divide_by_the_step() -> None:
    out = _events(_approaching(dropped_frame_tracks()))
    flagged = out[(out["id1"] == 0) & (out["id2"] == 1)].sort_values("frame")
    # Row 0 has no previous row. Every other step moves 5 units a second,
    # including the three that cross a gap.
    assert flagged["aa_event"].iloc[1:].eq(1).all()


def test_approach_avoidance_window_sec_warns_that_the_steps_are_uneven(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        _ = _events(_approaching(dropped_frame_tracks()), smooth_window_sec=0.5)
    messages = [record.getMessage() for record in caplog.records]
    assert any("[approach-avoidance]" in m and "resample-tracks" in m for m in messages)


def test_approach_avoidance_refuses_an_fps_in_its_sampling_settings() -> None:
    with pytest.raises(ValidationError, match="sampling"):
        _ = ApproachAvoidance(params={"sampling": {"fps_default": 30.0}})


# --- nn-delta-response and neighbor-delta-response ------------------------


def _with_neighbors(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(
        speed=SPEED,
        nn_id=1.0 - table["id"].astype(float),
        nn_delta_x_ego=1.0,
        nn_delta_y_ego=0.0,
    )


def test_nn_delta_response_turn_rate_is_radians_per_second() -> None:
    feature = NearestNeighborDelta(params={"speed_col": "speed", "diff_numframes": 4})
    out = feature.apply(_with_neighbors(dropped_frame_tracks()))
    np.testing.assert_allclose(out["dangle"], ANGULAR_VELOCITY)


def test_nn_delta_response_refuses_an_fps_in_its_sampling_settings() -> None:
    with pytest.raises(ValidationError, match="sampling"):
        _ = NearestNeighborDelta(params={"sampling": {"fps_default": 30.0}})


def test_neighbor_delta_response_turn_rate_is_radians_per_second() -> None:
    feature = NeighborDeltaResponse(params={"speed_col": "speed", "diff_numframes": 4})
    out = feature.apply(_with_neighbors(dropped_frame_tracks()))
    np.testing.assert_allclose(out["dangle"], ANGULAR_VELOCITY)


# --- social-motion-summary --------------------------------------------------


def _ramp(table: pd.DataFrame) -> pd.DataFrame:
    """Speed that rises 2 per second, so every acceleration is 2 and jerk is 0."""
    return table.assign(speed=2.0 * table["time"])


def test_social_motion_summary_accelerations_come_from_the_time_column() -> None:
    out = SocialMotionSummary().apply(_ramp(dropped_frame_tracks()))
    np.testing.assert_allclose(out["accel_mean"], 2.0)
    np.testing.assert_allclose(out["jerk_mean"], 0.0, atol=1e-9)


def test_social_motion_summary_refuses_an_fps_by_name() -> None:
    with pytest.raises(ValidationError, match="fps"):
        _ = SocialMotionSummary(params={"fps": 30.0})


def test_social_motion_summary_counts_kicks_per_second_of_elapsed_time() -> None:
    table = dropped_frame_tracks(ids=(0,))
    rows = np.arange(len(table))
    table["speed"] = np.abs((rows % 20) - 10).astype(float)
    out = SocialMotionSummary(params={"compute_burst_coast": True}).apply(table)
    times = table["time"].to_numpy(dtype=np.float64)
    speeds = table["speed"].to_numpy(dtype=np.float64)
    rising = np.diff(speeds) / np.diff(times) > 0
    peaks = int(np.count_nonzero(rising[:-1] & ~rising[1:]))
    kick_rate = float(out["kick_rate"].to_numpy(dtype=np.float64)[0])
    assert kick_rate == pytest.approx(peaks / float(times[-1] - times[0]))


# --- wavelets --------------------------------------------------------------


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_individual_wavelet_measures_its_rate_from_time_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = dropped_frame_tracks().assign(speed=1.0)
    feature = IndividualWavelet(
        IndividualWavelet.Inputs((Result(feature="speed-angvel"),)),
        params={"cols": ["speed"], "f_min": 2.0, "f_max": 5.0, "n_freq": 4},
    )
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        out = feature.apply(table)
    assert out.attrs["fps"] == pytest.approx(mean_rate())
    [message] = _warnings(caplog)
    assert "[individual-wavelet]" in message
    assert "1.45 s" in message
    assert "resample-tracks" in message


def test_pair_wavelet_measures_its_rate_from_the_pair_features_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pairs = PairPositionFeatures().apply(dropped_frame_tracks())
    feature = PairWavelet(
        PairWavelet.Inputs((Result(feature="pair-position"),)),
        params={"cols": ["A_speed"], "f_min": 2.0, "f_max": 5.0, "n_freq": 4},
    )
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        out = feature.apply(pairs)
    assert out.attrs["fps"] == pytest.approx(mean_rate())
    assert any("[pair-wavelet]" in message for message in _warnings(caplog))


# --- label-statistics ------------------------------------------------------


def test_label_statistics_bouts_last_the_sum_of_their_rows_steps() -> None:
    table = dropped_frame_tracks(ids=(0,))
    table["state"] = np.where(table["frame"] < 100, 0, 1)
    feature = LabelStatistics(
        params={"labels": ResultColumn(column="state"), "states": [0, 1]}
    )
    _ = feature.load_state(Path(), {}, {})

    row = feature.apply(table).iloc[0]

    times = table["time"].to_numpy()
    last_step = float(times[-1] - times[-2])
    assert row["bout_mean_s__0"] == pytest.approx(times[100] - times[0], rel=1e-5)
    assert row["bout_mean_s__1"] == pytest.approx(
        times[-1] - times[100] + last_step, rel=1e-5
    )
    present_minutes = (times[-1] - times[0] + last_step) / 60.0
    assert row["rate_per_min__0"] == pytest.approx(1.0 / present_minutes, rel=1e-5)


# --- uniform-grid methods --------------------------------------------------


def test_keypoint_kinematics_warns_that_the_steps_are_uneven(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = _with_keypoints(dropped_frame_tracks(ids=(0,)))
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        _ = KeypointKinematics().apply(table)
    [message] = _warnings(caplog)
    assert "[keypoint-kinematics]" in message
    assert "1.45 s" in message
    assert "resample-tracks" in message


def test_keypoint_kinematics_with_an_explicit_rate_does_not_measure_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = _with_keypoints(dropped_frame_tracks(ids=(0,)))
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        _ = KeypointKinematics(params={"fps": 53.0}).apply(table)
    assert _warnings(caplog) == []


def test_temporal_stack_sizes_its_pooling_window_at_the_measured_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = dropped_frame_tracks(ids=(0,))
    table["value"] = np.sin(np.arange(len(table)) / 3.0)
    win_sec = 1.2
    feature = TemporalStackingFeature(
        TemporalStackingFeature.Inputs((Result(feature="x"),)),
        params={
            "use_temporal_stack": False,
            "add_pool": True,
            "pool_stats": ("mean",),
            "sigma_pool": 0.0,
            "win_sec": win_sec,
            "columns": ["value"],
        },
    )
    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        out = feature.apply(table)

    frames = win_sec * mean_rate()
    sigma = max(1.0, frames / 6.0)
    truncate = max(1.0, max(1, round(frames)) / (2.0 * sigma))
    expected = gaussian_filter1d(
        table["value"].to_numpy(dtype=np.float32),
        sigma=sigma,
        mode="nearest",
        truncate=truncate,
    )
    np.testing.assert_allclose(out["value__pool_mean"], expected, rtol=1e-5)
    assert any("[temporal-stack]" in message for message in _warnings(caplog))


def _pooling_feature(**overrides: object) -> TemporalStackingFeature:
    return TemporalStackingFeature(
        TemporalStackingFeature.Inputs((Result(feature="x"),)),
        params={
            "use_temporal_stack": False,
            "add_pool": True,
            "pool_stats": ("mean",),
            "sigma_pool": 0.0,
            "win_sec": 1.2,
            "columns": ["value"],
            **overrides,
        },
    )


def test_temporal_stack_takes_an_explicit_fps_without_measuring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = dropped_frame_tracks(ids=(0,)).drop(columns="time")
    table["value"] = np.sin(np.arange(len(table)) / 3.0)

    with caplog.at_level(logging.WARNING, logger="mosaic.core.timebase"):
        out = _pooling_feature(fps=40.0).apply(table)

    frames = 1.2 * 40.0
    sigma = max(1.0, frames / 6.0)
    truncate = max(1.0, round(frames) / (2.0 * sigma))
    expected = gaussian_filter1d(
        table["value"].to_numpy(dtype=np.float32),
        sigma=sigma,
        mode="nearest",
        truncate=truncate,
    )
    np.testing.assert_allclose(out["value__pool_mean"], expected, rtol=1e-5)
    assert _warnings(caplog) == []


def test_temporal_stack_without_fps_or_time_is_refused_by_name() -> None:
    table = dropped_frame_tracks(ids=(0,)).drop(columns="time")
    table["value"] = 1.0
    with pytest.raises(ValueError, match="temporal-stack.*'time'"):
        _ = _pooling_feature().apply(table)


def test_temporal_stack_declares_a_reach_when_fps_is_given() -> None:
    params = TemporalStackingFeature.Params(
        sigma_pool=10, win_sec=2, fps=30, use_temporal_stack=False
    )
    assert TemporalStackingFeature.temporal_reach(params) == 30
    unmeasured = TemporalStackingFeature.Params(sigma_pool=10, win_sec=2)
    assert TemporalStackingFeature.temporal_reach(unmeasured) is None


def test_temporal_stack_refuses_a_non_positive_fps() -> None:
    with pytest.raises(ValidationError, match="fps"):
        _ = TemporalStackingFeature.Params(fps=0.0)
