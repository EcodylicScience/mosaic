"""A short track or a one-frame pair fragment leaves its own rows unsmoothed or NaN.

One id with too few rows for the smoothing window, or one pair that shares a single
frame, must not fail the recording: the rows it cannot measure are NaN or raw, and
every other id and pair is written as usual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from mosaic.behavior.feature_library.helpers import gradient_along, unwrap_diff
from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel

FPS = 30.0
KEYPOINTS = 3
NECK = 1
TAIL = 2


def _track(identity: int, frames: list[int], *, slope: float = 1.0) -> pd.DataFrame:
    """One id's rows, moving along x at *slope* units per frame."""
    index = np.asarray(frames, dtype=float)
    return pd.DataFrame(
        {
            "frame": frames,
            "time": index / FPS,
            "id": identity,
            "X": slope * index + 0.3 * np.sin(index),
            "Y": 0.5 * index,
            "ANGLE": 0.05 * index,
            "sequence": "s",
            "group": "g",
            "fps": FPS,
        }
    )


def _speed(table: pd.DataFrame, window: int) -> pd.DataFrame:
    out = SpeedAngvel(params={"smooth_window": window}).apply(table)
    return out.sort_values(["id", "frame"]).reset_index(drop=True)


def _smoothed(speed: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay of *speed* with its NaN rows filled by zero and masked after."""
    result = savgol_filter(np.nan_to_num(speed), window, 1)
    result[np.isnan(speed)] = np.nan
    return result


def test_an_id_shorter_than_the_window_is_smoothed_with_the_window_that_fits() -> None:
    table = pd.concat(
        [_track(0, list(range(40))), _track(1, list(range(4)))], ignore_index=True
    )

    out = _speed(table, 7)

    long, short = (out[out["id"] == i] for i in (0, 1))
    np.testing.assert_allclose(
        long["speed_smooth"].to_numpy(), _smoothed(long["speed"].to_numpy(), 7)
    )
    np.testing.assert_allclose(
        short["speed_smooth"].to_numpy(), _smoothed(short["speed"].to_numpy(), 3)
    )


def test_an_id_under_three_rows_is_left_unsmoothed() -> None:
    table = pd.concat(
        [_track(0, list(range(40))), _track(1, [0, 1])], ignore_index=True
    )

    out = _speed(table, 7)

    short = out[out["id"] == 1]
    assert len(short) == 2
    np.testing.assert_array_equal(
        short["speed_smooth"].to_numpy(), short["speed"].to_numpy()
    )
    assert out[out["id"] == 0]["speed_smooth"].notna().sum() > 30


def test_a_window_of_one_still_raises_on_a_long_id() -> None:
    with pytest.raises(ValueError, match="polyorder"):
        _ = _speed(_track(0, list(range(40))), 1)


def test_a_window_that_fits_is_never_replaced() -> None:
    out = _speed(_track(0, list(range(40))), 2)

    np.testing.assert_allclose(
        out["speed_smooth"].to_numpy(), _smoothed(out["speed"].to_numpy(), 2)
    )


def test_a_one_row_id_is_written() -> None:
    table = pd.concat([_track(0, list(range(40))), _track(1, [5])], ignore_index=True)

    out = _speed(table, 7)

    assert (out["id"] == 1).sum() == 1


def test_the_gradient_of_a_series_under_two_samples_is_nan() -> None:
    assert np.isnan(gradient_along(np.array([3.0]))).all()
    assert gradient_along(np.array([])).size == 0


def test_the_gradient_over_sample_times_is_the_derivative() -> None:
    time = np.array([0.0, 1.0, 3.0, 6.0])

    np.testing.assert_allclose(gradient_along(2.0 * time, time), 2.0)


def test_the_gradient_without_times_is_per_sample() -> None:
    np.testing.assert_allclose(gradient_along(np.array([0.0, 2.0, 4.0])), 2.0)


def test_the_angular_velocity_of_one_angle_is_nan() -> None:
    assert np.isnan(unwrap_diff(np.array([0.4]), FPS)).all()


def _pairs() -> pd.DataFrame:
    """Ids 0 and 1 share 12 frames, and id 2 shares exactly one frame with id 0."""
    return pd.concat(
        [
            _track(0, list(range(12))),
            _track(1, list(range(12)), slope=2.0),
            _track(2, [11], slope=0.5),
        ],
        ignore_index=True,
    )


def test_pair_position_writes_a_one_frame_fragment_beside_a_long_pair() -> None:
    out = PairPositionFeatures().apply(_pairs())

    fragment = out[(out["id1"] == 0) & (out["id2"] == 2)]
    assert len(fragment) == 1
    assert np.isnan(fragment["A_speed"]).all()
    long = out[(out["id1"] == 0) & (out["id2"] == 1)]
    assert len(long) == 12
    assert np.isfinite(long["A_speed"]).all()


def _keypoint_pairs() -> pd.DataFrame:
    """Ids 0 and 1 share 12 frames, and id 2 shares exactly one frame with id 0."""
    rows: list[dict[str, object]] = []
    spans = {0: range(12), 1: range(12), 2: [11]}
    for identity, frames in spans.items():
        for frame in frames:
            x, y = float(frame) + identity, 5.0 * identity
            row: dict[str, object] = {
                "frame": frame,
                "id": identity,
                "group": "g",
                "sequence": "s",
            }
            for keypoint in range(KEYPOINTS):
                row[f"poseX{keypoint}"] = x + (keypoint == NECK)
                row[f"poseY{keypoint}"] = y
            rows.append(row)
    return pd.DataFrame(rows)


def test_pair_egocentric_writes_a_one_frame_fragment_beside_a_long_pair() -> None:
    feature = PairEgocentricFeatures(
        params={
            "pose": {"pose_n": KEYPOINTS},
            "neck_idx": NECK,
            "tail_base_idx": TAIL,
        }
    )

    out = feature.apply(_keypoint_pairs())

    fragment = out[(out["id1"] == 0) & (out["id2"] == 2)]
    assert len(fragment) == 1
    assert np.isnan(fragment["A_speed"]).all()
    long = out[(out["id1"] == 0) & (out["id2"] == 1)]
    assert len(long) == 12
    assert np.isfinite(long["A_speed"]).all()
