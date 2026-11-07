"""The keypoints ``pair-egocentric`` reads its heading and its centers from.

``neck_idx``, ``tail_base_idx`` and an integer ``center_mode`` name keypoints by
their column number, and ``pose.pose_indices`` may select keypoints at or past
``pose.pose_n``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures

N_FRAMES = 6
N_KEYPOINTS = 12
NECK = 9
TAIL = 11


def _tracks() -> pd.DataFrame:
    """Two individuals whose neck lies one unit along +x from the tail base.

    Every other keypoint lies along +y from the body's origin. A heading read from
    any other pair of keypoints points elsewhere.
    """
    rows: list[dict[str, object]] = []
    for individual in (0, 1):
        for frame in range(N_FRAMES):
            x, y = 10.0 * individual + frame, 5.0 * individual
            row: dict[str, object] = {
                "frame": frame,
                "id": individual,
                "group": "g",
                "sequence": "s",
            }
            for k in range(N_KEYPOINTS):
                row[f"poseX{k}"] = x
                row[f"poseY{k}"] = y + k
            row[f"poseX{NECK}"], row[f"poseY{NECK}"] = x + 1.0, y
            row[f"poseX{TAIL}"], row[f"poseY{TAIL}"] = x, y
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_heading_comes_from_keypoints_selected_past_pose_n() -> None:
    feature = PairEgocentricFeatures(
        params={
            "pose": {"pose_indices": [8, NECK, 10, TAIL]},
            "neck_idx": NECK,
            "tail_base_idx": TAIL,
        }
    )

    out = feature.apply(_tracks())

    assert len(out) == 2 * N_FRAMES
    assert np.allclose(out["A_heading_cos"], 1.0)
    assert np.allclose(out["A_heading_sin"], 0.0)


def test_a_neck_outside_the_selection_is_refused_before_any_pair_is_built() -> None:
    feature = PairEgocentricFeatures(
        params={"pose": {"pose_indices": [8, 10, TAIL]}, "neck_idx": NECK}
    )

    with pytest.raises(ValueError, match=r"neck_idx: keypoints \[9\] are not among"):
        _ = feature.apply(_tracks().iloc[:0])


def _pair(
    keypoints: dict[int, tuple[tuple[float, float], tuple[float, float]]],
) -> pd.DataFrame:
    """Two static individuals, keypoint ``k`` of each at ``keypoints[k]``."""
    rows: list[dict[str, object]] = []
    for frame in range(N_FRAMES):
        for individual in (0, 1):
            row: dict[str, object] = {
                "frame": frame,
                "id": individual,
                "group": "g",
                "sequence": "s",
            }
            for k, points in keypoints.items():
                row[f"poseX{k}"], row[f"poseY{k}"] = points[individual]
            rows.append(row)
    return pd.DataFrame(rows)


# Keypoint 0 of the two individuals lies 10 apart; their keypoint means lie at
# y = 3 and y = 1, sqrt(10^2 + 2^2) apart.
_THREE_KEYPOINTS = {
    0: ((0.0, 0.0), (10.0, 0.0)),
    1: ((0.0, 1.0), (10.0, 1.0)),
    2: ((0.0, 8.0), (10.0, 2.0)),
}


def _separations(params: dict[str, object], tracks: pd.DataFrame) -> np.ndarray:
    out = PairEgocentricFeatures(params=params).apply(tracks)
    return np.asarray(out["AB_dist"], dtype=float)


def test_center_mode_centers_on_one_keypoint() -> None:
    base: dict[str, object] = {
        "neck_idx": 1,
        "tail_base_idx": 0,
        "pose": {"pose_n": 3},
    }
    tracks = _pair(_THREE_KEYPOINTS)

    on_keypoint = _separations({**base, "center_mode": 0}, tracks)
    on_mean = _separations({**base, "center_mode": "mean"}, tracks)

    assert np.allclose(on_keypoint, 10.0)
    assert np.allclose(on_mean, np.sqrt(104.0))


def test_center_mode_reads_an_index_as_neck_idx_does() -> None:
    tracks = _pair(
        {
            1: ((0.0, 0.0), (10.0, 0.0)),
            3: ((0.0, 5.0), (7.0, 5.0)),
            5: ((0.0, 9.0), (10.0, 9.0)),
        }
    )
    params: dict[str, object] = {
        "pose": {"pose_indices": [1, 3, 5]},
        "neck_idx": 3,
        "tail_base_idx": 1,
        "center_mode": 3,
    }

    assert np.allclose(_separations(params, tracks), 7.0)


def test_center_mode_refuses_a_name() -> None:
    with pytest.raises(ValidationError):
        _ = PairEgocentricFeatures(params={"center_mode": "nose"})


def test_the_default_center_mode_dumps_as_before() -> None:
    dumped = PairEgocentricFeatures.Params().identity_dump()

    assert dumped["center_mode"] == "mean"
