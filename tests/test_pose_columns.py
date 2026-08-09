"""Keypoint order is keypoint identity, so the sort has to be numeric.

``pose_column_pairs`` returns a list, and every consumer indexes into it
positionally: ``heading`` reads ``xy[:, front_idx, :]`` and ``xy[:, rear_idx, :]``
to build a two-point angle, and the overlay draws skeleton lines between
positions. "The Nth keypoint" is therefore a claim about this list's order.

Sorted lexicographically, that claim silently breaks from ten keypoints on --
``poseX10`` orders between ``poseX1`` and ``poseX2``, so a 21-point midline is
measured and drawn scrambled with nothing in the output saying so, and
``front_idx=0`` stops meaning the landmark the config names. Nothing raises,
which is why it needs pinning here rather than being left to a downstream
assertion that would only ever fire on a small rig.
"""

from __future__ import annotations

from mosaic.core.pipeline.loading import pose_column_pairs


def _numbered(count: int) -> list[str]:
    return [f"pose{axis}{i}" for i in range(count) for axis in ("X", "Y")]


def test_keypoints_past_nine_keep_their_numeric_position() -> None:
    """The regression: a lexicographic sort puts poseX10 at index 1."""
    pairs = pose_column_pairs(_numbered(12))

    assert pairs == [(f"poseX{i}", f"poseY{i}") for i in range(12)]
    assert pairs[10] == ("poseX10", "poseY10")


def test_the_order_does_not_depend_on_the_order_columns_arrive_in() -> None:
    """A frame's column order is an accident of the converter, not a schema."""
    shuffled = ["poseY2", "poseX11", "poseY11", "poseX2", "poseX0", "poseY0"]

    assert pose_column_pairs(shuffled) == [
        ("poseX0", "poseY0"),
        ("poseX2", "poseY2"),
        ("poseX11", "poseY11"),
    ]


def test_an_x_without_its_y_is_skipped_rather_than_half_reported() -> None:
    """A half-pair has no point to name, and would shift every later index."""
    columns = ["poseX0", "poseY0", "poseX1", "poseX2", "poseY2"]

    assert pose_column_pairs(columns) == [
        ("poseX0", "poseY0"),
        ("poseX2", "poseY2"),
    ]


def test_named_keypoints_are_ordered_deterministically_after_numeric_ones() -> None:
    """A non-numeric suffix must not raise, and must not reorder run to run.

    Numeric suffixes are the convention every in-tree converter emits; a named
    set is tolerated on read, and lands after them in lexicographic order so two
    reads of one file agree.
    """
    columns = ["poseXhead", "poseYhead", "poseX0", "poseY0", "poseXtail", "poseYtail"]

    assert pose_column_pairs(columns) == [
        ("poseX0", "poseY0"),
        ("poseXhead", "poseYhead"),
        ("poseXtail", "poseYtail"),
    ]


def test_no_pose_columns_is_an_empty_list_not_an_error() -> None:
    """Centroid-only tracks are ordinary, and every caller branches on falsy."""
    assert pose_column_pairs(["frame", "id", "x", "y"]) == []
