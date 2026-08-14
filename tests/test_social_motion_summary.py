"""Unit tests for SocialMotionSummary against hand-computed values.

Two scenarios, both fps=1 so a frame is a second and every emitted scalar can be
checked by hand.

``_make_merged_df`` -- 2 fish, 5 frames, always neighbours (group_size=2, shared
group_membership). Fish 0's speed increases linearly [1,2,3,4,5]; fish 1 is
constant at 2. This is the unsplit scenario and its literals are pinned below;
do not edit it.

``_make_split_df`` -- 2 fish, 8 frames, whose group_size drops to 1 for frames
3-4. Exercises ``subgroup_col``: the frames where the fish are apart carry the
two largest accelerations in the series, so an interval leaking across the
boundary is visible in the numbers rather than only in principle.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.social_motion_summary import SocialMotionSummary


def _make_merged_df() -> pd.DataFrame:
    """The merged (nn + speed + ffgroups) frame the pipeline would hand to apply()."""
    frames = [0, 1, 2, 3, 4]
    rows = []
    speed0 = [1.0, 2.0, 3.0, 4.0, 5.0]
    speed1 = [2.0, 2.0, 2.0, 2.0, 2.0]
    for k, f in enumerate(frames):
        # Fish 0: neighbour (fish 1) directly ahead, aligned heading
        rows.append(
            {
                "frame": f,
                "id": 0,
                "sequence": "seq1",
                "group": "g",
                "speed": speed0[k],
                "nn_id": 1.0,
                "nn_delta_angle": 0.0,  # cos = 1  -> nn_align = 1
                "nn_delta_x_ego": 1.0,  # ahead    -> frac_nn_ahead = 1
                "nn_delta_y_ego": 0.0,
                "group_membership": 0,
                "group_size": 2,
            }
        )
        # Fish 1: neighbour (fish 0) directly behind, anti-aligned heading
        rows.append(
            {
                "frame": f,
                "id": 1,
                "sequence": "seq1",
                "group": "g",
                "speed": speed1[k],
                "nn_id": 0.0,
                "nn_delta_angle": math.pi,  # cos = -1 -> nn_align = -1
                "nn_delta_x_ego": -1.0,  # behind   -> frac_nn_ahead = 0
                "nn_delta_y_ego": 0.0,
                "group_membership": 0,
                "group_size": 2,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def result() -> pd.DataFrame:
    feat = SocialMotionSummary(params={"fps": 1.0})
    return feat.apply(_make_merged_df()).set_index("id")


def test_one_row_per_fish(result: pd.DataFrame) -> None:
    assert sorted(result.index.tolist()) == [0, 1]
    assert result.loc[0, "sequence"] == "seq1"
    assert result.loc[0, "group"] == "g"


def test_nn_alignment(result: pd.DataFrame) -> None:
    assert result.loc[0, "nn_align"] == pytest.approx(1.0)
    assert result.loc[1, "nn_align"] == pytest.approx(-1.0)


def test_neighbor_bearing(result: pd.DataFrame) -> None:
    assert result.loc[0, "frac_nn_ahead"] == pytest.approx(1.0)
    assert result.loc[1, "frac_nn_ahead"] == pytest.approx(0.0)
    assert result.loc[0, "nn_bearing_x"] == pytest.approx(1.0)
    assert result.loc[1, "nn_bearing_x"] == pytest.approx(-1.0)
    assert result.loc[0, "nn_bearing_y"] == pytest.approx(0.0)


def test_speed_match_nn(result: pd.DataFrame) -> None:
    # |own - neighbour speed| averaged: [1,0,1,2,3] / 5 = 1.4 for both fish
    assert result.loc[0, "speed_match_nn"] == pytest.approx(1.4)
    assert result.loc[1, "speed_match_nn"] == pytest.approx(1.4)


def test_speed_match_group(result: pd.DataFrame) -> None:
    # |speed - group-mean| averaged: [0.5,0,0.5,1,1.5] / 5 = 0.7 for both fish
    assert result.loc[0, "speed_match_group"] == pytest.approx(0.7)
    assert result.loc[1, "speed_match_group"] == pytest.approx(0.7)


def test_speed_dispersion(result: pd.DataFrame) -> None:
    # Fish 0 speed [1..5]: std=sqrt(2), mean=3 -> cv; IQR=2, median=3 -> rcv
    assert result.loc[0, "speed_cv"] == pytest.approx(math.sqrt(2) / 3)
    assert result.loc[0, "speed_rcv"] == pytest.approx(2.0 / 3.0)
    # Fish 1 constant -> zero dispersion
    assert result.loc[1, "speed_cv"] == pytest.approx(0.0)
    assert result.loc[1, "speed_rcv"] == pytest.approx(0.0)


def test_accel_jerk(result: pd.DataFrame) -> None:
    # Fish 0: accel = [1,1,1,1] -> mean/med 1, zero dispersion; jerk all 0
    assert result.loc[0, "accel_mean"] == pytest.approx(1.0)
    assert result.loc[0, "accel_med"] == pytest.approx(1.0)
    assert result.loc[0, "accel_cv"] == pytest.approx(0.0)
    assert result.loc[0, "jerk_mean"] == pytest.approx(0.0)
    assert np.isnan(result.loc[0, "jerk_cv"])  # std/mean with mean 0 -> nan
    # Fish 1: constant speed -> zero acceleration magnitude
    assert result.loc[1, "accel_mean"] == pytest.approx(0.0)


def test_group_free_operation() -> None:
    """Without group_membership, speed_match_group is NaN but nn metrics still work."""
    df = _make_merged_df().drop(columns=["group_membership"])
    res = SocialMotionSummary(params={"fps": 1.0}).apply(df).set_index("id")
    assert np.isnan(res.loc[0, "speed_match_group"])
    assert res.loc[0, "speed_match_nn"] == pytest.approx(1.4)
    assert res.loc[0, "nn_align"] == pytest.approx(1.0)


def test_burst_coast_optional() -> None:
    """Burst-coast columns appear only when the flag is set."""
    df = _make_merged_df()
    off = SocialMotionSummary(params={"fps": 1.0}).apply(df)
    assert "kick_rate" not in off.columns
    on = SocialMotionSummary(params={"fps": 1.0, "compute_burst_coast": True}).apply(df)
    assert "kick_rate" in on.columns
    assert "burst_coast_ratio" in on.columns


def test_empty_dataframe() -> None:
    assert SocialMotionSummary().apply(pd.DataFrame()).empty


# --- subgroup_col: one row per (id, group_size) ---------------------------------


def _make_split_df() -> pd.DataFrame:
    """2 fish over 8 frames whose group_size drops to 1 for frames 3-4.

    Fish 0's speed is [1,3,2, 100,200, 4,6,5]. The two intervals that straddle
    the group-size change (2 -> 100 and 200 -> 4) are by far the largest
    accelerations in the series, so a split that let either leak into a subgroup
    would be obvious in the assertions rather than merely wrong.

    ``nn_id`` stays finite throughout, *including* while the fish are apart, so a
    NaN social metric on the group_size == 1 row proves it was
    ``social_min_group_size`` that masked it and not the absence of a neighbour.
    """
    group_sizes = [2, 2, 2, 1, 1, 2, 2, 2]
    speed0 = [1.0, 3.0, 2.0, 100.0, 200.0, 4.0, 6.0, 5.0]
    rows = []
    for f, (gs, s0) in enumerate(zip(group_sizes, speed0, strict=True)):
        # together they share one component; apart, each is its own
        membership = (0, 0) if gs == 2 else (0, 1)
        for fish, (speed, nn, angle, ego_x) in enumerate(
            ((s0, 1.0, 0.0, 1.0), (2.0, 0.0, math.pi, -1.0))
        ):
            rows.append(
                {
                    "frame": f,
                    "id": fish,
                    "sequence": "seq1",
                    "group": "g",
                    "speed": speed,
                    "nn_id": nn,
                    "nn_delta_angle": angle,
                    "nn_delta_x_ego": ego_x,
                    "nn_delta_y_ego": 0.0,
                    "group_membership": membership[fish],
                    "group_size": gs,
                }
            )
    return pd.DataFrame(rows)


def _split(**overrides: object) -> pd.DataFrame:
    params: dict[str, object] = {"fps": 1.0, "subgroup_col": "group_size"}
    params.update(overrides)
    return SocialMotionSummary(params=params).apply(_make_split_df())


def test_subgroup_off_matches_a_single_valued_split() -> None:
    """Splitting on a column with one value reproduces the unsplit row exactly.

    Not approximately: the subgroup mask is all-True, so every reduction sees the
    same array and returns the same bits. This is the invariant that lets the
    split and unsplit paths stay one code path.
    """
    df = _make_merged_df()  # group_size == 2 on every frame
    off = SocialMotionSummary(params={"fps": 1.0}).apply(df)
    on = SocialMotionSummary(params={"fps": 1.0, "subgroup_col": "group_size"}).apply(
        df
    )
    counts = ["group_size", "n_frames", "n_social_frames", "n_accel", "n_jerk"]
    pd.testing.assert_frame_equal(on.drop(columns=counts), off, check_exact=True)


def test_flag_off_columns_are_unchanged() -> None:
    """The unsplit schema, pinned in order.

    ``pd.DataFrame(rows)`` takes its columns from the first dict's insertion
    order, so refactoring the metric computation is exactly what silently
    reorders a parquet's columns.
    """
    out = SocialMotionSummary(params={"fps": 1.0}).apply(_make_merged_df())
    assert out.columns.tolist() == [
        "id",
        "nn_align",
        "frac_nn_ahead",
        "nn_bearing_x",
        "nn_bearing_y",
        "speed_match_nn",
        "speed_match_group",
        "speed_cv",
        "speed_rcv",
        "accel_mean",
        "accel_med",
        "accel_cv",
        "accel_rcv",
        "jerk_mean",
        "jerk_med",
        "jerk_cv",
        "jerk_rcv",
        "sequence",
        "group",
    ]


def test_one_row_per_fish_per_subgroup() -> None:
    out = _split()
    assert list(zip(out["id"], out["group_size"], strict=True)) == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]
    assert (out["sequence"] == "seq1").all()
    assert (out["group"] == "g").all()
    # taken off the original column, so it still merges against ffgroups-metrics
    assert out["group_size"].dtype.kind == "i"


def test_derivatives_are_not_computed_across_a_subgroup_boundary() -> None:
    """The point of the change: an interval counts only where both ends do.

    Fish 0's accelerations are [2,-1,98,100,-196,2,-1]. The 98 and the -196
    straddle the group-size change and belong to neither subgroup; the 100 lies
    wholly inside the separated stretch and belongs to it alone.
    """
    out = _split().set_index(["id", "group_size"])
    assert out.loc[(0, 2), "accel_mean"] == pytest.approx(1.5)  # |[2,-1,2,-1]|
    assert out.loc[(0, 1), "accel_mean"] == pytest.approx(100.0)  # |[100]|
    # jerk needs three consecutive in-subgroup samples; the lone stretch has none
    assert out.loc[(0, 2), "jerk_mean"] == pytest.approx(3.0)
    assert np.isnan(out.loc[(0, 1), "jerk_mean"])
    # and neither is the pooled number, which averages in the two straddlers
    pooled = SocialMotionSummary(params={"fps": 1.0}).apply(_make_split_df())
    assert pooled.set_index("id").loc[0, "accel_mean"] == pytest.approx(400.0 / 7.0)


def test_a_lone_subgroup_carries_no_social_metrics() -> None:
    """group_size == 1 has no neighbour to align with, and says how it knows."""
    out = _split().set_index(["id", "group_size"])
    for column in (
        "nn_align",
        "frac_nn_ahead",
        "nn_bearing_x",
        "nn_bearing_y",
        "speed_match_nn",
        "speed_match_group",
    ):
        assert np.isnan(out.loc[(0, 1), column]), column
    # the row exists and reports how much data it declined to use
    assert out.loc[(0, 1), "n_frames"] == 2
    assert out.loc[(0, 1), "n_social_frames"] == 0


def test_counts_report_the_sample_size() -> None:
    out = _split().set_index(["id", "group_size"])
    assert out.loc[(0, 2), ["n_frames", "n_social_frames"]].tolist() == [6, 6]
    # 6 in-subgroup samples span 4 intervals, not 5: the stretch is split in two
    assert out.loc[(0, 2), ["n_accel", "n_jerk"]].tolist() == [4, 2]
    assert out.loc[(0, 1), ["n_accel", "n_jerk"]].tolist() == [1, 0]


def test_counts_appear_only_under_the_split() -> None:
    """Unsplit output keeps its schema; the counts are meaningless there anyway."""
    counts = ["n_frames", "n_social_frames", "n_accel", "n_jerk"]
    off = SocialMotionSummary(params={"fps": 1.0}).apply(_make_split_df())
    on = _split()
    assert [c for c in counts if c in off.columns] == []
    assert [c for c in counts if c in on.columns] == counts


def test_subgroup_col_collision_is_refused_at_construction() -> None:
    """A colliding name would give the assembled frame two identical labels."""
    for bad in ("sequence", "speed_cv", "id", "n_frames"):
        with pytest.raises(ValueError, match="collides with a metadata or emitted"):
            SocialMotionSummary(params={"subgroup_col": bad})


def test_missing_subgroup_column_raises() -> None:
    df = _make_split_df().drop(columns=["group_size"])
    with pytest.raises(ValueError, match="Missing required columns"):
        SocialMotionSummary(params={"fps": 1.0, "subgroup_col": "group_size"}).apply(df)


def test_nan_subgroup_rows_are_dropped() -> None:
    """A NaN key could not be matched by any downstream merge, so it forms no row."""
    df = _make_split_df()
    df.loc[(df["frame"] == 0), "group_size"] = np.nan
    out = SocialMotionSummary(params={"fps": 1.0, "subgroup_col": "group_size"}).apply(
        df
    )
    assert not out["group_size"].isna().any()
    assert sorted(out["group_size"].unique().tolist()) == [1.0, 2.0]
    # frame 0 is counted in no subgroup: fish 0 keeps 5 of its 6 grouped frames
    assert out.set_index(["id", "group_size"]).loc[(0, 2.0), "n_frames"] == 5


def test_burst_coast_under_a_split() -> None:
    """The rate's denominator is in-subgroup time, not the track's whole span."""
    out = _split(compute_burst_coast=True).set_index(["id", "group_size"])
    # 2 peaks over the 4 retained intervals = 4 s; the boundary peak counts for
    # neither subgroup
    assert out.loc[(0, 2), "kick_rate"] == pytest.approx(0.5)
    assert out.loc[(0, 2), "burst_coast_ratio"] == pytest.approx(0.5)
    pooled = SocialMotionSummary(
        params={"fps": 1.0, "compute_burst_coast": True}
    ).apply(_make_split_df())
    # pooled counts 3 peaks over the full 7 s span
    assert pooled.set_index("id").loc[0, "kick_rate"] == pytest.approx(3.0 / 7.0)


def test_a_lone_fish_deviates_zero_from_its_own_group_mean() -> None:
    """Documented wart, pinned so it is not filed as a bug.

    At social_min_group_size=1 a solitary fish is its own group, so it deviates
    from its own mean by exactly zero -- which reads as a perfect social match.
    """
    out = _split(social_min_group_size=1).set_index(["id", "group_size"])
    assert out.loc[(0, 1), "speed_match_group"] == pytest.approx(0.0)
