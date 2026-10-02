"""Heading is an inference over keypoints, and which inference is now visible.

Four converters computed this inline and wrote it into the track table as
``ANGLE``, so it read as something the tracker had measured. It is not: a pose
model returns keypoints, and turning those into a direction is a choice.

The choice is not free. ``two_point`` orders a pair of landmarks and yields a
determined heading; ``pca`` yields an **axis**, whose sign is arbitrary. The
converters fell back from the first to the second silently -- whenever an index
was out of range -- so one column could hold both kinds of number with nothing to
say which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.heading import HeadingFeature
from mosaic.core.kinematics import angle_from_pca, angle_from_two_points


def _frame(front: np.ndarray, rear: np.ndarray, *, n_ids: int = 1) -> pd.DataFrame:
    """A track table with two keypoints per row: index 0 front, index 1 rear."""
    n = len(front)
    return pd.DataFrame(
        {
            "frame": np.tile(np.arange(n, dtype=np.int64), n_ids),
            "time": np.tile(np.arange(n, dtype=float) / 30.0, n_ids),
            "id": np.repeat(np.arange(n_ids, dtype=np.int64), n),
            "group": [""] * (n * n_ids),
            "sequence": ["seq"] * (n * n_ids),
            "poseX0": np.tile(front[:, 0], n_ids),
            "poseY0": np.tile(front[:, 1], n_ids),
            "poseX1": np.tile(rear[:, 0], n_ids),
            "poseY1": np.tile(rear[:, 1], n_ids),
        }
    )


def test_two_point_heading_points_from_rear_to_front() -> None:
    """An animal at the origin facing +x has heading 0."""
    front = np.array([[1.0, 0.0], [2.0, 0.0]])
    rear = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = HeadingFeature().apply(_frame(front, rear))
    assert out["ANGLE"].to_numpy() == pytest.approx([0.0, 0.0])


def test_the_heading_is_measured_in_image_coordinates() -> None:
    """y increases downward, so a positive angle turns clockwise on screen.

    Flipping this would put the heading and the keypoints it came from in
    different conventions, which is worse than either convention alone.
    """
    front = np.array([[0.0, 1.0]])
    rear = np.array([[0.0, 0.0]])
    out = HeadingFeature().apply(_frame(front, rear))
    assert out["ANGLE"].to_numpy() == pytest.approx([np.pi / 2])


def test_reversing_the_landmarks_reverses_the_heading() -> None:
    """The pair is ordered, which is exactly what makes this a heading."""
    front = np.array([[1.0, 0.0]])
    rear = np.array([[0.0, 0.0]])
    forward = HeadingFeature().apply(_frame(front, rear))
    backward = HeadingFeature(params={"front_idx": 1, "rear_idx": 0}).apply(
        _frame(front, rear)
    )
    assert abs(float(forward["ANGLE"].iloc[0]) - float(backward["ANGLE"].iloc[0])) == (
        pytest.approx(np.pi)
    )


def test_each_individual_gets_its_own_heading() -> None:
    front = np.array([[1.0, 0.0], [2.0, 0.0]])
    rear = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = HeadingFeature().apply(_frame(front, rear, n_ids=3))
    assert len(out) == 6
    assert set(out["id"]) == {0, 1, 2}
    assert out["ANGLE"].to_numpy() == pytest.approx(np.zeros(6))


def test_an_out_of_range_landmark_refuses_instead_of_falling_back() -> None:
    """The converters silently substituted the principal axis here.

    That put two kinds of number under one column name, with nothing recording
    which rule had run. A configuration error is more useful said out loud.
    """
    front = np.array([[1.0, 0.0]])
    rear = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="out of range"):
        _ = HeadingFeature(params={"front_idx": 0, "rear_idx": 7}).apply(
            _frame(front, rear)
        )


def test_naming_one_point_twice_refuses() -> None:
    front = np.array([[1.0, 0.0]])
    rear = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="one point"):
        _ = HeadingFeature(params={"front_idx": 1, "rear_idx": 1}).apply(
            _frame(front, rear)
        )


def test_a_table_without_keypoints_refuses() -> None:
    """A centroid has no body axis to read."""
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "time": [0.0, 0.04],
            "id": [0, 0],
            "group": ["", ""],
            "sequence": ["seq", "seq"],
            "X": [1.0, 2.0],
            "Y": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="pose keypoint columns"):
        _ = HeadingFeature().apply(df)


def test_the_pca_method_returns_an_axis_whose_sign_is_arbitrary() -> None:
    """Documented, tested, and the reason two_point is the default.

    The same body, mirrored through its own centre, is the same *axis* and the
    opposite *heading*. PCA cannot tell them apart -- which is why anything
    differencing successive angles can read a flip as a real turn.
    """
    forward = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    reversed_body = forward[:, ::-1, :]

    assert angle_from_pca(forward) == pytest.approx(angle_from_pca(reversed_body))
    # The two-point form, given the same two ends, does distinguish them.
    assert angle_from_two_points(forward[:, 2, :], forward[:, 0, :]) != pytest.approx(
        angle_from_two_points(forward[:, 0, :], forward[:, 2, :])
    )


def test_the_output_column_is_configurable() -> None:
    """So a caller can keep a converter-era ANGLE and compare against it."""
    front = np.array([[1.0, 0.0]])
    rear = np.array([[0.0, 0.0]])
    out = HeadingFeature(params={"output_col": "heading_rad"}).apply(
        _frame(front, rear)
    )
    assert "heading_rad" in out.columns
    assert "ANGLE" not in out.columns


def test_the_method_is_part_of_the_run_identity() -> None:
    """Two methods are two addressable results, not one column that changed."""
    two_point = HeadingFeature(params={"method": "two_point"})
    pca = HeadingFeature(params={"method": "pca"})
    assert two_point.params.identity_dump() != pca.params.identity_dump()
