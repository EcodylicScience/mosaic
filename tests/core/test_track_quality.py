"""``instance_body_lengths``: one body length per instance from its keypoints."""

from __future__ import annotations

import numpy as np

from mosaic.core.track_quality import instance_body_lengths


def test_head_to_tail_length_when_both_are_finite() -> None:
    xy = np.array([[[0.0, 0.0], [1.0, 5.0], [3.0, 4.0]]])
    assert instance_body_lengths(xy, 0, 2).tolist() == [5.0]


def test_tight_box_diagonal_when_head_or_tail_is_missing() -> None:
    xy = np.array([[[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]])
    assert instance_body_lengths(xy, None, None).tolist() == [5.0]
    assert instance_body_lengths(xy, 0, None).tolist() == [5.0]


def test_tight_box_diagonal_when_head_or_tail_is_not_finite() -> None:
    xy = np.array([[[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]]])
    assert instance_body_lengths(xy, 0, 1).tolist() == [5.0]


def test_an_out_of_range_index_falls_back_to_the_diagonal() -> None:
    xy = np.array([[[0.0, 0.0], [3.0, 4.0]]])
    assert instance_body_lengths(xy, 0, 7).tolist() == [5.0]


def test_nan_when_no_keypoint_is_finite() -> None:
    xy = np.full((1, 3, 2), np.nan)
    assert np.isnan(instance_body_lengths(xy, 0, 2)).all()


def test_one_length_per_instance() -> None:
    xy = np.array(
        [
            [[0.0, 0.0], [3.0, 4.0]],
            [[np.nan, np.nan], [np.nan, 1.0]],
            [[0.0, 0.0], [6.0, 8.0]],
        ]
    )
    lengths = instance_body_lengths(xy, 0, 1)
    assert lengths[0] == 5.0
    assert np.isnan(lengths[1])
    assert lengths[2] == 10.0
