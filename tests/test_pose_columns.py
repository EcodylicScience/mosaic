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

import warnings

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pose_columns import (
    configured_pose_pairs,
    frame_keypoint_centroid,
    keypoint_centroid,
    pose_column_pairs,
)


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


# --- the body centre derived from those columns ---------------------------
#
# `keypoint_centroid` is the one answer to what `X`/`Y` mean for a producer that
# localizes landmarks and reports no centre of its own. It replaced seven
# hand-rolled copies that had drifted apart -- one averaged with a plain `mean`
# where a single missing landmark poisoned the row, one paired its X and Y
# column lists independently, and four let numpy's all-NaN warning through. Each
# of those is pinned below, because none of them raised.


def test_a_missing_keypoint_is_left_out_rather_than_poisoning_the_row() -> None:
    """The CalMS21 defect: a plain `mean` makes one NaN landmark NaN the centre."""
    x = np.array([[1.0, 3.0], [1.0, np.nan]])
    y = np.array([[2.0, 6.0], [4.0, np.nan]])

    cx, cy = keypoint_centroid(x, y)

    assert cx.tolist() == [2.0, 1.0]
    assert cy.tolist() == [4.0, 4.0]


def test_a_row_with_no_detected_keypoint_is_nan_and_warns_about_nothing() -> None:
    """Absent is a legitimate state, and its warning says nothing the NaN does not."""
    x = np.array([[np.nan, np.nan], [1.0, 3.0]])
    y = np.array([[np.nan, np.nan], [2.0, 6.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cx, cy = keypoint_centroid(x, y)

    assert np.isnan(cx[0]) and np.isnan(cy[0])
    assert (cx[1], cy[1]) == (2.0, 4.0)


def test_a_table_with_no_keypoints_at_all_is_nan_throughout() -> None:
    """A centroid-only producer never reaches here, but an empty stack must not raise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cx, cy = keypoint_centroid(np.empty((3, 0)), np.empty((3, 0)))

    assert cx.shape == (3,) and np.isnan(cx).all()
    assert cy.shape == (3,) and np.isnan(cy).all()


def test_mismatched_keypoint_arrays_are_refused() -> None:
    """Two shapes cannot be averaged pairwise, and guessing which to trim is worse."""
    with pytest.raises(ValueError, match="same shape"):
        _ = keypoint_centroid(np.zeros((2, 3)), np.zeros((2, 4)))


def test_the_frame_form_averages_every_pair_it_finds() -> None:
    frame = pd.DataFrame(
        {"poseX0": [1.0], "poseY0": [2.0], "poseX1": [5.0], "poseY1": [8.0]}
    )

    cx, cy = frame_keypoint_centroid(frame)

    assert cx.tolist() == [3.0]
    assert cy.tolist() == [5.0]


def test_the_frame_form_honours_an_explicit_pair_list() -> None:
    """A caller whose keypoint set is declared must not widen to what a table holds."""
    frame = pd.DataFrame(
        {"poseX0": [1.0], "poseY0": [2.0], "poseX1": [5.0], "poseY1": [8.0]}
    )

    cx, cy = frame_keypoint_centroid(frame, [("poseX0", "poseY0")])

    assert cx.tolist() == [1.0]
    assert cy.tolist() == [2.0]


def test_a_frame_with_no_pose_columns_is_nan_rather_than_an_error() -> None:
    cx, cy = frame_keypoint_centroid(pd.DataFrame({"frame": [0, 1]}))

    assert np.isnan(cx).all() and cx.shape == (2,)
    assert np.isnan(cy).all() and cy.shape == (2,)


def test_half_a_configured_pair_contributes_no_keypoint() -> None:
    """The interaction-crop defect: filtering X and Y independently misaligns them.

    With `poseY1` absent, an x list of three and a y list of two used to be
    stacked side by side, averaging keypoint 2's x against keypoint 2's y at one
    index and keypoint 1's y at another. The pair is dropped whole instead.
    """
    columns = ["poseX0", "poseY0", "poseX1", "poseX2", "poseY2"]

    pairs = configured_pose_pairs(columns, x_prefix="poseX", y_prefix="poseY", count=7)

    assert pairs == [("poseX0", "poseY0"), ("poseX2", "poseY2")]


def test_a_configured_set_does_not_widen_to_what_the_table_carries() -> None:
    """`pose_n` is a declaration; a table with more keypoints does not change it."""
    columns = _numbered(6)

    assert configured_pose_pairs(
        columns, x_prefix="poseX", y_prefix="poseY", count=2
    ) == [("poseX0", "poseY0"), ("poseX1", "poseY1")]
