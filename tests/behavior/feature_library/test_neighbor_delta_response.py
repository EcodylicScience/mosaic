"""Neighbor maps over every neighbor, the shuffled control, and time windows.

``neighbor-delta-response`` emits one row per focal, neighbor and frame, and
``nn-delta-bins`` maps those rows: the occupancy of neighbor positions, and the
focal's turning and speeding against them, within a window of time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.neighbor_delta_response import (
    NeighborDeltaResponse,
    _rank_rows,
)
from mosaic.behavior.feature_library.nn_delta_bins import NearestNeighborDeltaBins
from mosaic.behavior.feature_library.nn_delta_response import NearestNeighborDelta
from mosaic.core.pipeline.column_contract import (
    MissingInputColumns,
    check_input_columns,
    needs_of,
    param_values,
)


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
    out = _respond(df, diff_numframes=2)
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


# --- vector projections ---------------------------------------------------------

_DELTA_COLUMNS = ["dx", "dy", "dt", "dangle", "dspeed"]
_PROJECTION_COLUMNS = ["response_par", "response_perp"]
_PARTNER_COLUMNS = ["partner_par", "partner_perp", "partner_norm"]


def _two_fish(n_frames: int = 12) -> pd.DataFrame:
    """Two fish on arbitrary positions, headings and vectors.

    ``vx``/``vy`` are a reference vector, ``ax``/``ay`` the focal's response and
    ``xs``/``ys`` another pair of position columns.
    """
    rng = np.random.default_rng(7)
    frames = np.arange(n_frames)
    tables: list[pd.DataFrame] = []
    for fid in (0, 1):
        tables.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "time": frames / 10.0,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": rng.normal(size=n_frames) * 10.0,
                    "Y": rng.normal(size=n_frames) * 10.0,
                    "xs": rng.normal(size=n_frames) * 10.0,
                    "ys": rng.normal(size=n_frames) * 10.0,
                    "ANGLE": rng.uniform(-np.pi, np.pi, size=n_frames),
                    "speed": 1.0,
                    "vx": rng.normal(size=n_frames),
                    "vy": rng.normal(size=n_frames),
                    "ax": rng.normal(size=n_frames),
                    "ay": rng.normal(size=n_frames),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _fish(df: pd.DataFrame, fid: int) -> pd.DataFrame:
    return df[df["id"] == fid].reset_index(drop=True)


def _focal_rows(out: pd.DataFrame, fid: int) -> pd.DataFrame:
    return out[out["id1"] == fid].sort_values("frame").reset_index(drop=True)


def test_reference_vectors_project_offsets_response_and_partner() -> None:
    df = _two_fish()
    out = _respond(
        df,
        delta_response=False,
        x_col="xs",
        y_col="ys",
        reference_cols=["vx", "vy"],
        response_cols=["ax", "ay"],
        partner_cols=["vx", "vy"],
    )
    focal, partner = _fish(df, 0), _fish(df, 1)
    vx, vy = focal["vx"].to_numpy(), focal["vy"].to_numpy()
    nrm = np.sqrt(vx**2 + vy**2)
    ux, uy, px, py = vx / nrm, vy / nrm, -vy / nrm, vx / nrm
    x = partner["xs"].to_numpy() - focal["xs"].to_numpy()
    y = partner["ys"].to_numpy() - focal["ys"].to_numpy()
    ax, ay = focal["ax"].to_numpy(), focal["ay"].to_numpy()
    vx2, vy2 = partner["vx"].to_numpy(), partner["vy"].to_numpy()

    got = _focal_rows(out, 0)

    assert len(got) == len(focal)
    np.testing.assert_array_equal(got["neighbor_x"], x * ux + y * uy)
    np.testing.assert_array_equal(got["neighbor_y"], x * px + y * py)
    np.testing.assert_array_equal(got["neighbor_dist"], np.hypot(x, y))
    np.testing.assert_array_equal(got["response_par"], ax * ux + ay * uy)
    np.testing.assert_array_equal(got["response_perp"], ax * px + ay * py)
    np.testing.assert_array_equal(got["partner_par"], vx2 * ux + vy2 * uy)
    np.testing.assert_array_equal(got["partner_perp"], vx2 * px + vy2 * py)
    np.testing.assert_array_equal(got["partner_norm"], np.sqrt(vx2**2 + vy2**2))
    assert got["partner_norm"].dtype == np.float64


def test_a_response_and_partner_project_on_the_orientation_without_a_reference() -> (
    None
):
    df = _two_fish()
    out = _respond(
        df,
        delta_response=False,
        response_cols=["ax", "ay"],
        partner_cols=["vx", "vy"],
    )
    focal, partner = _fish(df, 0), _fish(df, 1)
    angle = focal["ANGLE"].to_numpy()
    ux, uy = np.cos(angle), np.sin(angle)
    px, py = -uy, ux
    x = partner["X"].to_numpy() - focal["X"].to_numpy()
    y = partner["Y"].to_numpy() - focal["Y"].to_numpy()
    ax, ay = focal["ax"].to_numpy(), focal["ay"].to_numpy()
    vx2, vy2 = partner["vx"].to_numpy(), partner["vy"].to_numpy()

    got = _focal_rows(out, 0)

    np.testing.assert_array_equal(
        got["neighbor_x"], x * np.cos(angle) + y * np.sin(angle)
    )
    np.testing.assert_array_equal(
        got["neighbor_y"], -x * np.sin(angle) + y * np.cos(angle)
    )
    np.testing.assert_array_equal(got["response_par"], ax * ux + ay * uy)
    np.testing.assert_array_equal(got["response_perp"], ax * px + ay * py)
    np.testing.assert_array_equal(got["partner_par"], vx2 * ux + vy2 * uy)
    np.testing.assert_array_equal(got["partner_perp"], vx2 * px + vy2 * py)
    np.testing.assert_array_equal(got["partner_norm"], np.sqrt(vx2**2 + vy2**2))


def test_the_partner_is_read_from_its_own_row_for_the_other_perspective() -> None:
    df = _two_fish()
    out = _respond(
        df, delta_response=False, reference_cols=["vx", "vy"], partner_cols=["ax", "ay"]
    )
    focal, partner = _fish(df, 1), _fish(df, 0)
    vx, vy = focal["vx"].to_numpy(), focal["vy"].to_numpy()
    nrm = np.sqrt(vx**2 + vy**2)
    ux, uy = vx / nrm, vy / nrm
    vx2, vy2 = partner["ax"].to_numpy(), partner["ay"].to_numpy()

    got = _focal_rows(out, 1)

    assert (got["perspective"] == 1).all()
    np.testing.assert_array_equal(got["partner_par"], vx2 * ux + vy2 * uy)


def test_a_missing_partner_value_stays_missing_and_keeps_the_row() -> None:
    df = _two_fish()
    df.loc[(df["id"] == 1) & (df["frame"] == 4), "vx"] = np.nan
    out = _respond(
        df, delta_response=False, reference_cols=["vx", "vy"], partner_cols=["vx", "vy"]
    )

    got = _focal_rows(out, 0)

    assert len(got) == 12
    assert got["partner_par"].isna().tolist() == [f == 4 for f in range(12)]
    assert got["partner_norm"].isna().tolist() == [f == 4 for f in range(12)]
    assert got["neighbor_x"].notna().all()


def test_a_reference_leaves_the_response_of_the_windowed_delta_as_it_is() -> None:
    df = _two_fish()
    plain = _respond(df)
    referenced = _respond(df, reference_cols=["vx", "vy"])

    assert len(plain) == len(referenced) == 8 * 2
    for column in _DELTA_COLUMNS:
        np.testing.assert_array_equal(referenced[column], plain[column])
    assert not np.array_equal(referenced["neighbor_x"], plain["neighbor_x"])


def test_other_position_columns_place_the_offsets_and_the_windowed_delta() -> None:
    df = _two_fish()
    out = _respond(df, x_col="xs", y_col="ys")
    focal, partner = _fish(df, 0), _fish(df, 1)
    angle = focal["ANGLE"].to_numpy()[:8]
    x = (partner["xs"].to_numpy() - focal["xs"].to_numpy())[:8]
    y = (partner["ys"].to_numpy() - focal["ys"].to_numpy())[:8]

    got = _focal_rows(out, 0)

    np.testing.assert_array_equal(
        got["neighbor_x"], x * np.cos(angle) + y * np.sin(angle)
    )
    np.testing.assert_array_equal(
        got["dx"], focal["xs"].to_numpy()[4:] - focal["xs"].to_numpy()[:8]
    )
    np.testing.assert_array_equal(
        got["dy"], focal["ys"].to_numpy()[4:] - focal["ys"].to_numpy()[:8]
    )


def test_without_the_windowed_delta_every_focal_row_counts_and_no_delta_is_written() -> (
    None
):
    df = _two_fish()
    out = _respond(df, delta_response=False)

    assert len(out) == 12 * 2
    assert set(out.columns) == {
        "frame",
        "time",
        "group",
        "sequence",
        "id1",
        "id2",
        "perspective",
        "neighbor_x",
        "neighbor_y",
        "neighbor_dist",
        "neighbor_rank",
    }


def test_without_the_windowed_delta_the_speed_and_time_columns_are_not_read() -> None:
    df = _two_fish().drop(columns=["speed", "time"])

    out = NeighborDeltaResponse(params={"delta_response": False}).apply(df)

    assert len(out) == 12 * 2
    assert "time" not in out.columns


def test_without_the_windowed_delta_a_response_column_adds_its_projections() -> None:
    df = _two_fish()
    out = _respond(df, delta_response=False, response_cols=["ax", "ay"])

    assert set(out.columns) >= set(_PROJECTION_COLUMNS)
    assert not set(out.columns) & set(_DELTA_COLUMNS + _PARTNER_COLUMNS)


def test_a_focal_row_without_a_finite_response_is_left_out() -> None:
    df = _two_fish()
    df.loc[(df["id"] == 0) & (df["frame"] == 5), "ay"] = np.nan
    df.loc[(df["id"] == 1) & (df["frame"] == 2), "ax"] = np.inf

    without = _respond(df, delta_response=False)
    with_response = _respond(df, delta_response=False, response_cols=["ax", "ay"])

    assert len(without) == 12 * 2
    assert _focal_rows(with_response, 0)["frame"].tolist() == [
        f for f in range(12) if f != 5
    ]
    assert _focal_rows(with_response, 1)["frame"].tolist() == [
        f for f in range(12) if f != 2
    ]


def test_a_response_column_also_filters_rows_under_the_windowed_delta() -> None:
    df = _two_fish()
    df.loc[(df["id"] == 0) & (df["frame"] == 5), "ax"] = np.nan

    out = _respond(df, response_cols=["ax", "ay"])

    assert _focal_rows(out, 0)["frame"].tolist() == [0, 1, 2, 3, 4, 6, 7]
    assert _focal_rows(out, 1)["frame"].tolist() == list(range(8))


def test_a_zero_reference_drops_its_rows() -> None:
    df = _two_fish()
    zero = (df["id"] == 0) & df["frame"].isin([3, 4])
    df.loc[zero, ["vx", "vy"]] = 0.0

    out = _respond(df, delta_response=False, reference_cols=["vx", "vy"])

    assert _focal_rows(out, 0)["frame"].tolist() == [
        f for f in range(12) if f not in (3, 4)
    ]
    assert _focal_rows(out, 1)["frame"].tolist() == list(range(12))


def test_a_missing_reference_drops_its_rows() -> None:
    df = _two_fish()
    df.loc[(df["id"] == 1) & (df["frame"] == 6), "vy"] = np.nan

    out = _respond(df, delta_response=False, reference_cols=["vx", "vy"])

    assert 6 not in _focal_rows(out, 1)["frame"].tolist()
    assert 6 in _focal_rows(out, 0)["frame"].tolist()


def test_the_shuffle_rolls_the_partner_vectors_with_its_positions() -> None:
    """The partner's ``PX`` is its ``X``, so a rolled ``PX`` is the rolled position."""
    n_frames = 200
    frames = np.arange(n_frames, dtype=float)
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "frame": np.arange(n_frames),
                    "time": frames / 10.0,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": frames * 3.0 + 5.0 * fid + np.sin(frames / 7.0).round() * fid,
                    "Y": 2.0 * fid + np.cos(frames / 5.0).round() * fid,
                    "vx": 1.0,
                    "vy": 0.0,
                }
            )
            for fid in (0, 1)
        ],
        ignore_index=True,
    )
    df["PX"] = df["X"]
    df["PY"] = df["Y"]
    params = {
        "delta_response": False,
        "reference_cols": ["vx", "vy"],
        "partner_cols": ["PX", "PY"],
    }

    plain = _focal_rows(_respond(df, **params), 0)
    shuffled = _focal_rows(
        _respond(df, **params, time_shuffle=True, shuffle_min_shift=20), 0
    )

    focal = _fish(df, 0)
    # The reference is (1, 0), so a neighbor offset is its world offset and a partner
    # vector is its own components. Every value is a whole number, so sums are exact.
    for out in (plain, shuffled):
        np.testing.assert_array_equal(
            out["partner_par"], focal["X"].to_numpy() + out["neighbor_x"].to_numpy()
        )
        np.testing.assert_array_equal(
            out["partner_perp"], focal["Y"].to_numpy() + out["neighbor_y"].to_numpy()
        )
    assert not np.array_equal(plain["partner_par"], shuffled["partner_par"])


def test_the_shuffle_leaves_the_focal_response_where_it_is() -> None:
    df = _two_fish(n_frames=200)
    params = {
        "delta_response": False,
        "reference_cols": ["vx", "vy"],
        "response_cols": ["ax", "ay"],
    }
    plain = _focal_rows(_respond(df, **params), 0)
    shuffled = _focal_rows(
        _respond(df, **params, time_shuffle=True, shuffle_min_shift=20), 0
    )

    for column in _PROJECTION_COLUMNS:
        np.testing.assert_array_equal(shuffled[column], plain[column])


def _with_neighbor_values() -> pd.DataFrame:
    """Two fish with a float column ``h`` and an integer column ``n`` of their own."""
    df = _two_fish()
    rng = np.random.default_rng(11)
    df["h"] = rng.normal(size=len(df)) * 90.0
    df["n"] = rng.integers(0, 50, size=len(df))
    return df


def test_each_row_of_a_pair_records_the_other_rows_neighbor_columns() -> None:
    df = _with_neighbor_values()

    out = _respond(df, delta_response=False, neighbor_columns=["h", "n"])

    for focal_id, partner_id in ((0, 1), (1, 0)):
        focal, partner = _fish(df, focal_id), _fish(df, partner_id)
        got = _focal_rows(out, focal_id)
        assert len(got) == len(focal)
        np.testing.assert_array_equal(got["h"], focal["h"].to_numpy())
        np.testing.assert_array_equal(got["neighbor_h"], partner["h"].to_numpy())
        np.testing.assert_array_equal(got["n"], focal["n"].to_numpy(dtype=float))
        np.testing.assert_array_equal(
            got["neighbor_n"], partner["n"].to_numpy(dtype=float)
        )
        for column in ("h", "n", "neighbor_h", "neighbor_n"):
            assert got[column].dtype == np.float64


def test_neighbor_columns_follow_the_rows_the_windowed_delta_keeps() -> None:
    df = _with_neighbor_values()

    out = _respond(df, neighbor_columns=["h"])

    focal, partner = _fish(df, 0), _fish(df, 1)
    got = _focal_rows(out, 0)
    kept = got["frame"].to_numpy()
    assert 0 < len(kept) < len(focal)
    np.testing.assert_array_equal(got["h"], focal["h"].to_numpy()[kept])
    np.testing.assert_array_equal(got["neighbor_h"], partner["h"].to_numpy()[kept])


def test_a_missing_neighbor_value_is_nan_and_keeps_the_row() -> None:
    df = _with_neighbor_values()
    df.loc[(df["id"] == 1) & (df["frame"] == 5), "h"] = np.nan

    out = _respond(df, delta_response=False, neighbor_columns=["h"])

    got = _focal_rows(out, 0)
    expected = _fish(df, 1)["h"].to_numpy()
    assert np.isnan(expected[5])
    np.testing.assert_array_equal(got["neighbor_h"], expected)
    np.testing.assert_array_equal(got["h"], _fish(df, 0)["h"].to_numpy())


def test_a_neighbor_without_a_row_in_a_frame_leaves_the_pair_out() -> None:
    df = _with_neighbor_values()
    df = df[~((df["id"] == 1) & (df["frame"] == 5))]

    out = _respond(df, delta_response=False, neighbor_columns=["h"])

    got = _focal_rows(out, 0)
    assert 5 not in got["frame"].tolist()
    assert not got["neighbor_h"].isna().any()


def test_neighbor_columns_add_nothing_while_empty() -> None:
    df = _with_neighbor_values()

    plain = _respond(df, delta_response=False)
    empty = _respond(df, delta_response=False, neighbor_columns=[])

    pd.testing.assert_frame_equal(empty, plain)
    assert "h" not in plain.columns


def test_the_shuffle_rolls_the_neighbor_columns_with_the_positions() -> None:
    """The ``PX`` column is the position ``X``, so the rolled ``PX`` is the rolled position."""
    n_frames = 200
    frames = np.arange(n_frames, dtype=float)
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "frame": np.arange(n_frames),
                    "time": frames / 10.0,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": frames * 3.0 + 5.0 * fid + np.sin(frames / 7.0).round() * fid,
                    "Y": 2.0 * fid + np.cos(frames / 5.0).round() * fid,
                    "vx": 1.0,
                    "vy": 0.0,
                }
            )
            for fid in (0, 1)
        ],
        ignore_index=True,
    )
    df["PX"] = df["X"]
    params = {
        "delta_response": False,
        "reference_cols": ["vx", "vy"],
        "neighbor_columns": ["PX"],
    }

    plain = _focal_rows(_respond(df, **params), 0)
    shuffled = _focal_rows(
        _respond(df, **params, time_shuffle=True, shuffle_min_shift=20), 0
    )

    focal = _fish(df, 0)
    # The reference is (1, 0), so neighbor_x is the partner's X less the focal's, and
    # every value is a whole number, so the sums are exact.
    for out in (plain, shuffled):
        np.testing.assert_array_equal(out["PX"], focal["PX"].to_numpy())
        np.testing.assert_array_equal(
            out["neighbor_PX"], focal["X"].to_numpy() + out["neighbor_x"].to_numpy()
        )
    assert not np.array_equal(plain["neighbor_PX"], shuffled["neighbor_PX"])


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"neighbor_columns": ["label"]}, id="neighbor_columns"),
        pytest.param({"partner_cols": ["label", "vy"]}, id="partner_cols"),
        pytest.param({"response_cols": ["ax", "label"]}, id="response_cols"),
        pytest.param({"reference_cols": ["label", "vy"]}, id="reference_cols"),
    ],
)
def test_a_column_that_is_not_numeric_is_refused_by_name(
    params: dict[str, object],
) -> None:
    df = _two_fish()
    df["label"] = "a"

    with pytest.raises(ValueError, match=r"neighbor-delta-response.*'label'"):
        _ = _respond(df, delta_response=False, **params)


# --- the parameters -----------------------------------------------------------------


@pytest.mark.parametrize("field", ["reference_cols", "response_cols", "partner_cols"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (["vx"], "two columns"),
        ([], "two columns"),
        (["vx", "vy", "ax"], "two columns"),
        (["vx", "vx"], "distinct"),
    ],
)
def test_a_vector_list_is_two_distinct_columns(
    field: str, value: list[str], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = NeighborDeltaResponse(params={field: value})


@pytest.mark.parametrize(
    "override",
    [
        {"diff_numframes": 5},
        {"wrap_angle": False},
        {"divide_dangle_by_frames": False},
        {"scale_dangle_by_fps": False},
    ],
)
def test_a_delta_option_is_refused_without_the_windowed_delta(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="delta_response"):
        _ = NeighborDeltaResponse(params={"delta_response": False, **override})


def test_the_default_delta_options_are_accepted_without_the_windowed_delta() -> None:
    feature = NeighborDeltaResponse(
        params={
            "delta_response": False,
            "diff_numframes": 4,
            "wrap_angle": True,
            "divide_dangle_by_frames": True,
            "scale_dangle_by_fps": True,
        }
    )
    assert not feature.params.delta_response


@pytest.mark.parametrize(
    "name",
    [
        "frame",
        "time",
        "group",
        "sequence",
        "id1",
        "id2",
        "perspective",
        "x",
        "y",
        "dist",
        "rank",
        "focal",
        "neighbor_x",
        "dx",
        "dspeed",
        "response_par",
        "partner_norm",
        "Focal_fish",
    ],
)
def test_a_neighbor_column_that_collides_with_an_output_is_refused(name: str) -> None:
    with pytest.raises(ValidationError, match="already writes"):
        _ = NeighborDeltaResponse(params={"neighbor_columns": [name]})


@pytest.mark.parametrize("names", [["h", "neighbor_h"], ["neighbor_h", "h"]])
def test_two_neighbor_columns_that_write_one_name_are_refused(
    names: list[str],
) -> None:
    with pytest.raises(ValidationError, match="already writes"):
        _ = NeighborDeltaResponse(params={"neighbor_columns": names})


def test_a_neighbor_column_named_like_the_flag_column_is_refused() -> None:
    with pytest.raises(ValidationError, match="already writes"):
        _ = NeighborDeltaResponse(
            params={"focal_col": "flag", "neighbor_columns": ["flag"]}
        )


def test_a_neighbor_column_listed_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="twice"):
        _ = NeighborDeltaResponse(params={"neighbor_columns": ["h", "h"]})


def test_neighbor_columns_that_do_not_collide_are_accepted() -> None:
    feature = NeighborDeltaResponse(
        params={"neighbor_columns": ["heading_deg", "X", "neighbor"]}
    )
    assert feature.params.neighbor_columns == ["heading_deg", "X", "neighbor"]


def test_the_new_params_leave_the_identity_while_at_their_defaults() -> None:
    base = NeighborDeltaResponse().params.identity_dump()
    at_default = NeighborDeltaResponse(
        params={
            "x_col": "X",
            "y_col": "Y",
            "delta_response": True,
            "reference_cols": None,
            "response_cols": None,
            "partner_cols": None,
            "neighbor_columns": [],
        }
    ).params.identity_dump()

    assert at_default == base
    assert not {
        "x_col",
        "y_col",
        "delta_response",
        "reference_cols",
        "response_cols",
        "partner_cols",
        "neighbor_columns",
    } & set(base)


# --- the requirements ---------------------------------------------------------------


def _needed(**params: object) -> set[str]:
    feature = NeighborDeltaResponse(params=params)
    values = param_values(feature.params, type(feature.params).model_fields)
    return {
        column for need in needs_of(feature.requires, values) for column in need.columns
    }


_BASE_NEEDS = {"frame", "id", "X", "Y"}


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(
            {}, _BASE_NEEDS | {"ANGLE", "time", "SPEED", "speed"}, id="defaults"
        ),
        pytest.param(
            {"delta_response": False}, _BASE_NEEDS | {"ANGLE"}, id="no-delta-no-ref"
        ),
        pytest.param(
            {"delta_response": False, "reference_cols": ["vx", "vy"]},
            _BASE_NEEDS | {"vx", "vy"},
            id="no-delta-with-ref",
        ),
        pytest.param(
            {"reference_cols": ["vx", "vy"]},
            _BASE_NEEDS | {"vx", "vy", "ANGLE", "time", "SPEED", "speed"},
            id="delta-with-ref",
        ),
        pytest.param(
            {"delta_response": False, "x_col": "xs", "y_col": "ys"},
            {"frame", "id", "xs", "ys", "ANGLE"},
            id="own-positions",
        ),
        pytest.param(
            {"delta_response": False, "response_cols": ["ax", "ay"]},
            _BASE_NEEDS | {"ANGLE", "ax", "ay"},
            id="response",
        ),
        pytest.param(
            {"delta_response": False, "partner_cols": ["pvx", "pvy"]},
            _BASE_NEEDS | {"ANGLE", "pvx", "pvy"},
            id="partner",
        ),
        pytest.param(
            {"delta_response": False, "neighbor_columns": ["h", "n"]},
            _BASE_NEEDS | {"ANGLE", "h", "n"},
            id="neighbor-columns",
        ),
        pytest.param(
            {"delta_response": False, "neighbor_columns": []},
            _BASE_NEEDS | {"ANGLE"},
            id="no-neighbor-columns",
        ),
        pytest.param(
            {"speed_col": "v", "delta_response": True},
            _BASE_NEEDS | {"ANGLE", "time", "v", "speed"},
            id="named-speed",
        ),
    ],
)
def test_each_requirement_follows_its_condition(
    params: dict[str, object], expected: set[str]
) -> None:
    assert _needed(**params) == expected


def test_a_vector_list_needs_every_column_it_names() -> None:
    feature = NeighborDeltaResponse(
        params={"delta_response": False, "partner_cols": ["pvx", "pvy"]}
    )
    columns = ["frame", "id", "X", "Y", "ANGLE", "pvx"]

    with pytest.raises(MissingInputColumns, match="lacks 'pvy'"):
        check_input_columns(feature, columns)
    check_input_columns(feature, [*columns, "pvy"])


def test_a_neighbor_column_list_needs_every_column_it_names() -> None:
    feature = NeighborDeltaResponse(
        params={"delta_response": False, "neighbor_columns": ["h", "n"]}
    )
    columns = ["frame", "id", "X", "Y", "ANGLE", "h"]

    with pytest.raises(MissingInputColumns, match="lacks 'n'"):
        check_input_columns(feature, columns)
    check_input_columns(feature, [*columns, "n"])


def test_the_heading_is_one_requirement_when_two_conditions_name_it() -> None:
    feature = NeighborDeltaResponse()
    values = param_values(feature.params, type(feature.params).model_fields)

    named = [need.columns for need in needs_of(feature.requires, values)]

    assert named.count(("ANGLE",)) == 1
