"""The keypoints a ``PoseConfig`` selects, their columns, their names and references to them.

A keypoint index here is the number in the column name, as ``pose_indices`` lists
them, so ``3`` is ``poseX3``/``poseY3``. A feature reads the selected keypoints into
an array in ``pose_indices`` order, and a reference resolves to a position in that
array.
"""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.types import PoseConfig


def test_every_keypoint_is_selected_without_pose_indices() -> None:
    assert PoseConfig(pose_n=3).selected_indices() == [0, 1, 2]


def test_pose_indices_select_in_their_own_order() -> None:
    assert PoseConfig(pose_n=7, pose_indices=[5, 1]).selected_indices() == [5, 1]


def test_the_columns_of_the_selected_keypoints_carry_the_prefixes() -> None:
    pose = PoseConfig(pose_n=7, pose_indices=[5, 1], x_prefix="kx", y_prefix="ky")

    assert pose.keypoint_columns() == (["kx5", "kx1"], ["ky5", "ky1"])


def test_unnamed_keypoints_are_named_by_count() -> None:
    assert PoseConfig(pose_n=2).names(4) == ["kp0", "kp1", "kp2", "kp3"]


def test_declared_names_are_used_whatever_the_count() -> None:
    """The count only sizes the generated names; declared ones are the pose's."""
    pose = PoseConfig(pose_n=2, keypoint_names=["nose", "tail"])

    assert pose.names(5) == ["nose", "tail"]


def test_an_index_resolves_to_itself_when_every_keypoint_is_selected() -> None:
    assert PoseConfig(pose_n=7).resolve_keypoints([6, 3], who="t") == [6, 3]


def test_a_reference_resolves_to_its_position_among_the_selected() -> None:
    pose = PoseConfig(
        pose_n=4, pose_indices=[3, 1], keypoint_names=["nose", "neck", "hip", "tail"]
    )

    assert pose.resolve_keypoints(["neck", 3], who="t") == [1, 0]


def test_a_reference_outside_the_selection_is_refused_by_name() -> None:
    """Reading it would need a column the feature does not load."""
    pose = PoseConfig(pose_n=7, pose_indices=[1, 4])

    with pytest.raises(
        ValueError,
        match=r"pair-egocentric neck_idx.*\[2\].*\[1, 4\]; pose.pose_indices must",
    ):
        _ = pose.resolve_keypoints([1, 2], who="pair-egocentric neck_idx")


def test_an_index_past_pose_n_is_refused_when_every_keypoint_is_selected() -> None:
    """Without ``pose_indices``, raising ``pose_n`` selects the index too."""
    with pytest.raises(
        ValueError, match=r"\[3\] are not among .*\[0, 1, 2\]; raise pose.pose_n"
    ):
        _ = PoseConfig(pose_n=3).resolve_keypoints([3], who="t")


def test_a_negative_index_is_refused_as_no_index() -> None:
    """The refusal does not offer a remedy, because a selection cannot contain -1."""
    with pytest.raises(ValueError, match=r"\[-1\] are not keypoint indices"):
        _ = PoseConfig(pose_n=3).resolve_keypoints([-1], who="t")


def test_a_selected_index_past_pose_n_resolves() -> None:
    """``pose_indices`` may select past ``pose_n``, and the columns follow them."""
    pose = PoseConfig(pose_indices=[8, 9, 10, 11])

    assert pose.resolve_keypoints([9, 11], who="t") == [1, 3]


def test_a_name_without_declared_names_is_refused() -> None:
    """``kp0`` names a keypoint for a library, and is not a reference a recipe makes."""
    with pytest.raises(ValueError, match="no keypoint names"):
        _ = PoseConfig(pose_n=3).resolve_keypoints(["kp0"], who="t")
