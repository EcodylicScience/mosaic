"""The keypoints ``pair-egocentric`` reads its heading from.

``neck_idx`` and ``tail_base_idx`` name keypoints by their column number, and
``pose.pose_indices`` may select keypoints at or past ``pose.pose_n``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
