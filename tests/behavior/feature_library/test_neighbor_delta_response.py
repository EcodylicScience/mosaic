"""Neighbor maps over every neighbor, the shuffled control, and time windows.

``neighbor-delta-response`` emits one row per focal, neighbor and frame, and
``nn-delta-bins`` maps those rows: the occupancy of neighbor positions, and the
focal's turning and speeding against them, within a window of time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.neighbor_delta_response import (
    NeighborDeltaResponse,
    _rank_rows,
)
from mosaic.behavior.feature_library.nn_delta_bins import NearestNeighborDeltaBins
from mosaic.behavior.feature_library.nn_delta_response import NearestNeighborDelta


def _school(n_frames: int = 30, fps: float = 10.0) -> pd.DataFrame:
    """Three fish swimming along +x in a fixed formation.

    Fish 0 leads at the origin of the formation, fish 1 sits 2 units behind it
    and fish 2 sits 5 units to its left, so every ego offset is known.
    """
    offsets = {0: (0.0, 0.0), 1: (-2.0, 0.0), 2: (0.0, 5.0)}
    rows: list[dict[str, object]] = []
    for fid, (ox, oy) in offsets.items():
        for frame in range(n_frames):
            rows.append(
                {
                    "frame": frame,
                    "time": frame / fps,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": frame * 1.0 + ox,
                    "Y": oy,
                    "ANGLE": 0.0,
                    "speed": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _respond(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    return NeighborDeltaResponse(params={"speed_col": "speed", **params}).apply(df)


# --- neighbor-delta-response ---------------------------------------------------


def test_one_row_per_focal_neighbor_and_frame() -> None:
    out = _respond(_school(n_frames=30), diff_numframes=4)
    # 26 frames with a future sample, three focals, two neighbors each.
    assert len(out) == 26 * 3 * 2
    assert set(zip(out["id1"], out["id2"])) == {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }


def test_neighbors_sit_where_the_formation_puts_them() -> None:
    out = _respond(_school())
    lead = out[out["id1"] == 0].groupby("id2")[["neighbor_x", "neighbor_y"]].first()
    # Heading along +x, so ego x is ahead and ego y is to the left.
    assert lead.loc[1].tolist() == pytest.approx([-2.0, 0.0])
    assert lead.loc[2].tolist() == pytest.approx([0.0, 5.0])


def test_rank_and_distance() -> None:
    out = _respond(_school())
    lead = (
        out[out["id1"] == 0].groupby("id2")[["neighbor_dist", "neighbor_rank"]].first()
    )
    assert lead.loc[1].tolist() == pytest.approx([2.0, 1])
    assert lead.loc[2].tolist() == pytest.approx([5.0, 2])


def test_perspective_follows_the_pair_convention() -> None:
    out = _respond(_school())
    assert ((out["perspective"] == 0) == (out["id1"] < out["id2"])).all()


def test_rank_one_is_the_nearest_neighbor_pairing() -> None:
    """With ``max_neighbor_rank=1`` the response is nn-delta-response's."""
    df = _school()
    out = _respond(df, max_neighbor_rank=1).sort_values(["id1", "frame"])

    nearest = {0: 1, 1: 0, 2: 0}
    merged = df.assign(
        nn_id=df["id"].map(nearest).astype(float),
    )
    # The nearest neighbor's world offset, as nearest-neighbor writes it.
    pos = df.set_index(["frame", "id"])[["X", "Y"]]
    nn_pos = pos.reindex(list(zip(merged["frame"], merged["nn_id"].astype(int))))
    merged["nn_delta_x"] = nn_pos["X"].to_numpy() - merged["X"].to_numpy()
    merged["nn_delta_y"] = nn_pos["Y"].to_numpy() - merged["Y"].to_numpy()
    reference = (
        NearestNeighborDelta(params={"speed_col": "speed"})
        .apply(merged)
        .sort_values(["id", "frame"])
    )
    assert out["id2"].tolist() == reference["nn_id"].astype(int).tolist()
    for col in ("neighbor_x", "neighbor_y", "dx", "dy", "dt", "dangle", "dspeed"):
        assert out[col].to_numpy() == pytest.approx(reference[col].to_numpy()), col


def test_a_turning_focal_reports_its_turn() -> None:
    df = _school(fps=10.0)
    turning = df["id"] == 0
    df.loc[turning, "ANGLE"] = 0.1 * df.loc[turning, "frame"]
    out = _respond(df, diff_numframes=2, sampling={"fps_default": 10.0})
    # 0.1 rad per frame at 10 fps.
    assert out.loc[out["id1"] == 0, "dangle"].to_numpy() == pytest.approx(1.0)


def test_an_untracked_neighbor_frame_is_left_out() -> None:
    df = _school()
    df = df[~((df["id"] == 2) & (df["frame"] == 3))]
    out = _respond(df)
    assert not ((out["id2"] == 2) & (out["frame"] == 3)).any()
    assert ((out["id2"] == 1) & (out["frame"] == 3)).any()


def test_focal_flags_pass_through_for_both_roles() -> None:
    df = _school()
    df["Focal_fish"] = df["id"] == 0
    out = _respond(df)
    row = out[(out["id1"] == 1) & (out["id2"] == 0)].iloc[0]
    assert not row["Focal_fish"]
    assert row["neighbor_focal"]


def test_a_single_individual_has_no_neighbors() -> None:
    df = _school()
    assert _respond(df[df["id"] == 0]).empty


# --- the shuffled control -------------------------------------------------------


def test_the_shuffle_breaks_the_formation_but_not_the_focal() -> None:
    df = _school(n_frames=200)
    # Neighbors that wander, so a shifted trajectory sits somewhere else.
    wobble = np.sin(df["frame"].to_numpy() / 7.0) * 3.0 * (df["id"] > 0)
    df["Y"] = df["Y"] + wobble
    real = _respond(df)
    shuffled = _respond(df, time_shuffle=True, shuffle_min_shift=20)
    key = ["frame", "id1", "id2"]
    both = real.merge(shuffled, on=key, suffixes=("", "_shuf"))
    assert len(both) > 0
    for col in ("dx", "dy", "dangle", "dspeed"):
        assert both[col].to_numpy() == pytest.approx(both[f"{col}_shuf"].to_numpy())
    assert not np.allclose(both["neighbor_x"], both["neighbor_x_shuf"])


def test_the_shuffle_is_deterministic_per_seed() -> None:
    df = _school(n_frames=200)
    df["Y"] = df["Y"] + np.cos(df["frame"] / 5.0) * (df["id"] + 1)
    a = _respond(df, time_shuffle=True, shuffle_min_shift=20)
    b = _respond(df, time_shuffle=True, shuffle_min_shift=20)
    c = _respond(df, time_shuffle=True, shuffle_min_shift=20, shuffle_seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a["neighbor_y"].equals(c["neighbor_y"])


def test_a_sequence_too_short_to_shuffle_refuses() -> None:
    with pytest.raises(ValueError, match="shuffle_min_shift"):
        _respond(_school(n_frames=30), time_shuffle=True, shuffle_min_shift=15)


def test_rank_rows_leaves_missing_distances_unranked() -> None:
    dist = np.array([[3.0, np.nan, 1.0], [np.nan, np.nan, np.nan]])
    assert _rank_rows(dist).tolist() == [[2, 0, 1], [0, 0, 0]]


# --- nn-delta-bins over every neighbor ----------------------------------------


def _bins(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    return NearestNeighborDeltaBins(
        params={"focal_category_col": None, "binmax": 6.0, "nbins": 13, **params}
    ).apply(df)


def test_occupancy_counts_every_neighbor_position() -> None:
    responses = _respond(_school())
    out = _bins(responses)
    occ = out[(out["dim"] == "2d") & (out["metric"] == "occupancy")]
    assert occ["count"].sum() == len(responses)
    assert occ["value"].sum() == pytest.approx(1.0)
    # Two neighbor positions per focal; the formation puts them in few bins.
    assert (occ["count"] > 0).sum() <= 6


def test_unsplit_bins_need_no_category_column() -> None:
    out = _bins(_respond(_school()))
    assert not out.empty
    assert out["center_category"].isna().all()


def test_the_time_window_bins_only_its_rows() -> None:
    responses = _respond(_school(n_frames=30, fps=10.0))
    out = _bins(responses, start_time=1.0, end_time=2.0)
    occ = out[(out["dim"] == "2d") & (out["metric"] == "occupancy")]
    in_window = responses[(responses["time"] >= 1.0) & (responses["time"] < 2.0)]
    assert occ["count"].sum() == len(in_window) == 10 * 6


def test_a_time_window_without_a_time_column_refuses() -> None:
    responses = _respond(_school()).drop(columns=["time"])
    with pytest.raises(ValueError, match="time"):
        _bins(responses, start_time=1.0)


def test_category_split_still_works_on_every_neighbor() -> None:
    df = _school()
    df["Focal_fish"] = df["id"] == 0
    out = NearestNeighborDeltaBins(params={"binmax": 6.0, "nbins": 13}).apply(
        _respond(df)
    )
    assert set(out["center_category"].dropna().unique()) == {0, 1}
