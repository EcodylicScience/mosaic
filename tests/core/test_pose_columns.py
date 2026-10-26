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
    KeypointArrays,
    configured_pose_pairs,
    frame_keypoint_centroid,
    keypoint_arrays,
    keypoint_centroid,
    keypoint_suffixes,
    pose_column_pairs,
    pose_column_triplets,
    reindex_contiguous,
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


# --- keypoint arrays ---------------------------------------------------------


def test_suffixes_follow_keypoint_order_under_any_prefix() -> None:
    columns = ["kx10", "ky10", "kx2", "ky2", "kxnose", "kx0", "frame"]

    assert keypoint_suffixes(columns, "kx") == ["0", "2", "10", "nose"]


def test_suffixes_need_only_the_x_column() -> None:
    """The caller names every axis it reads, and a missing one raises there."""
    assert keypoint_suffixes(["poseX0", "poseX1", "poseY0"], "poseX") == ["0", "1"]


def test_a_table_without_the_prefix_has_no_suffixes() -> None:
    assert keypoint_suffixes(["frame", "X", "Y"], "poseX") == []


def _pose_rows(**columns: list[float]) -> pd.DataFrame:
    """Three frames, 7 to 9, with the given pose columns."""
    return pd.DataFrame({"frame": [7, 8, 9], **columns})


def _arrays(rows: pd.DataFrame, suffixes: list[str]) -> KeypointArrays:
    return keypoint_arrays(
        rows,
        x_prefix="poseX",
        y_prefix="poseY",
        confidence_prefix="poseP",
        suffixes=suffixes,
    )


def test_arrays_stack_the_requested_keypoints_in_the_order_given() -> None:
    rows = _pose_rows(
        poseX0=[0.0, 1.0, 2.0],
        poseY0=[10.0, 11.0, 12.0],
        poseX2=[20.0, 21.0, 22.0],
        poseY2=[30.0, 31.0, 32.0],
    )

    arrays = _arrays(rows, ["2", "0"])

    assert arrays.frames.dtype == np.int64
    assert arrays.frames.tolist() == [7, 8, 9]
    assert arrays.position.dtype == np.float32
    assert arrays.position.shape == (3, 2, 2)
    assert arrays.position[:, 0, :].tolist() == [[20, 30], [21, 31], [22, 32]]
    assert arrays.position[:, 1, :].tolist() == [[0, 10], [1, 11], [2, 12]]


def test_confidence_is_read_when_every_keypoint_has_its_column() -> None:
    rows = _pose_rows(
        poseX0=[0.0, 1.0, 2.0],
        poseY0=[0.0, 1.0, 2.0],
        poseP0=[0.5, 0.25, 0.125],
        poseX1=[0.0, 1.0, 2.0],
        poseY1=[0.0, 1.0, 2.0],
        poseP1=[1.0, 0.75, 0.0],
    )

    confidence = _arrays(rows, ["0", "1"]).confidence

    assert confidence.dtype == np.float32
    assert confidence.tolist() == [[0.5, 1.0], [0.25, 0.75], [0.125, 0.0]]


def test_confidence_is_one_when_a_keypoint_lacks_its_column() -> None:
    """Half the scores would leave the other half at an invented value."""
    rows = _pose_rows(
        poseX0=[0.0, 1.0, 2.0],
        poseY0=[0.0, 1.0, 2.0],
        poseP0=[0.5, 0.25, 0.125],
        poseX1=[0.0, 1.0, 2.0],
        poseY1=[0.0, 1.0, 2.0],
    )

    assert _arrays(rows, ["0", "1"]).confidence.tolist() == [[1.0, 1.0]] * 3


def test_a_non_finite_position_is_nan_with_confidence_zero() -> None:
    """One infinite coordinate makes the whole point missing, scored or not."""
    rows = _pose_rows(
        poseX0=[np.inf, np.nan, 2.0],
        poseY0=[5.0, 6.0, -np.inf],
        poseP0=[0.9, 0.9, 0.9],
        poseX1=[1.0, 1.0, 1.0],
        poseY1=[1.0, 1.0, 1.0],
        poseP1=[0.9, 0.9, 0.9],
    )

    arrays = _arrays(rows, ["0", "1"])

    assert np.isnan(arrays.position[:, 0, :]).all()
    assert arrays.confidence[:, 0].tolist() == [0.0, 0.0, 0.0]
    assert arrays.position[:, 1, :].tolist() == [[1.0, 1.0]] * 3
    assert np.allclose(arrays.confidence[:, 1], 0.9)


def test_a_nan_score_at_a_finite_position_is_kept() -> None:
    """A missing score is the producer's statement, and the caller decides on it."""
    rows = _pose_rows(
        poseX0=[0.0, 1.0, 2.0],
        poseY0=[0.0, 1.0, 2.0],
        poseP0=[0.5, np.nan, 0.5],
    )

    confidence = _arrays(rows, ["0"]).confidence

    assert np.isnan(confidence[1, 0])
    assert confidence[[0, 2], 0].tolist() == [0.5, 0.5]


def test_a_requested_keypoint_without_its_y_column_is_refused() -> None:
    rows = _pose_rows(poseX0=[0.0, 1.0, 2.0], poseY0=[0.0, 1.0, 2.0], poseX1=[0.0] * 3)

    with pytest.raises(KeyError, match="poseY1"):
        _ = _arrays(rows, ["0", "1"])


def test_reading_scores_leaves_the_table_unchanged() -> None:
    rows = _pose_rows(poseX0=[np.inf, 1.0, 2.0], poseY0=[0.0, 1.0, 2.0])
    rows["poseP0"] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    before = rows.copy()

    _ = _arrays(rows, ["0"])

    pd.testing.assert_frame_equal(rows, before)


def test_reindexing_fills_missing_frames_as_unseen() -> None:
    rows = pd.DataFrame(
        {"frame": [4, 2, 5], "poseX0": [4.0, 2.0, 5.0], "poseY0": [-4.0, -2.0, -5.0]}
    )

    arrays = reindex_contiguous(_arrays(rows, ["0"]))

    assert arrays.frames.tolist() == [2, 3, 4, 5]
    assert arrays.position[[0, 2, 3], 0, 0].tolist() == [2.0, 4.0, 5.0]
    assert np.isnan(arrays.position[1]).all()
    assert arrays.confidence[:, 0].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_reindexing_refuses_a_repeated_frame() -> None:
    """A contiguous frame axis has one row per frame."""
    rows = pd.DataFrame({"frame": [3, 3], "poseX0": [0.0, 1.0], "poseY0": [0.0, 1.0]})

    with pytest.raises(ValueError, match="frame 3"):
        _ = reindex_contiguous(_arrays(rows, ["0"]))


def test_reindexing_no_rows_returns_no_rows() -> None:
    rows = pd.DataFrame(
        {
            "frame": np.array([], dtype=np.int64),
            "poseX0": np.array([], dtype=float),
            "poseY0": np.array([], dtype=float),
        }
    )

    arrays = reindex_contiguous(_arrays(rows, ["0"]))

    assert arrays.frames.shape == (0,)
    assert arrays.position.shape == (0, 1, 2)
    assert arrays.confidence.shape == (0, 1)


# -- 3D keypoints ---------------------------------------------------------------


def test_triplets_are_ordered_numerically_like_pairs() -> None:
    columns = [f"pose{axis}{i}" for i in (10, 2, 1) for axis in ("X", "Y", "Z")]
    assert pose_column_triplets(columns) == [
        ("poseX1", "poseY1", "poseZ1"),
        ("poseX2", "poseY2", "poseZ2"),
        ("poseX10", "poseY10", "poseZ10"),
    ]


def test_a_keypoint_without_its_z_is_not_a_triplet() -> None:
    columns = ["poseX0", "poseY0", "poseZ0", "poseX1", "poseY1"]
    assert pose_column_triplets(columns) == [("poseX0", "poseY0", "poseZ0")]


def test_a_2d_table_has_no_triplets() -> None:
    assert pose_column_triplets(_numbered(3)) == []
