"""Unit tests for LocalOrderMetrics against hand-computed values.

The canonical scenario is five individuals in a plus sign: id0 at the origin and
ids 1-4 at unit distance along +x, +y, -x, -y, all heading +x, with a disc radius
of 1.5. The group center is the origin, so distances are [0, 1, 1, 1, 1] and each
individual's own radial rotation term is [nan, 0, -1, 0, +1] -- every disc mean
below follows from those five numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.local_order_metrics import LocalOrderMetrics

PLUS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
]

# Four on the unit circle with counter-clockwise tangential headings.
RING: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
RING_ANGLES: list[float] = [np.pi / 2, np.pi, -np.pi / 2, 0.0]


def _frame(
    positions: list[tuple[float, float]],
    angles: list[float] | None,
    frame: int = 0,
    first_id: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k, (px, py) in enumerate(positions):
        row: dict[str, object] = {
            "frame": frame,
            "time": float(frame) * 0.02,
            "id": first_id + k,
            "X": px,
            "Y": py,
            "sequence": "S1",
            "group": "G",
        }
        if angles is not None:
            row["ANGLE"] = angles[k]
        rows.append(row)
    return pd.DataFrame(rows)


def _apply(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    base: dict[str, object] = {"radius": 1.5}
    base.update(params)
    out = LocalOrderMetrics(params=base).apply(df)
    return out if out.empty else out.sort_values("id")


def _canonical() -> pd.DataFrame:
    return _apply(_frame(PLUS, [0.0] * 5))


# --- disc membership and local order ----------------------------------------


def test_disc_membership() -> None:
    """id1's disc excludes id3 at distance 2 but includes id2 and id4 at sqrt(2)."""
    out = _canonical()
    np.testing.assert_array_equal(out["n_local_neighbors"], [4, 3, 3, 3, 3])
    np.testing.assert_array_equal(out["n_local_headings"], [4, 3, 3, 3, 3])
    assert (out["local_radius"] == 1.5).all()
    assert (out["local_heading_source"] == "orientation").all()


def test_local_polarization_of_an_aligned_group() -> None:
    out = _canonical()
    np.testing.assert_allclose(out["local_polarization"], 1.0)
    np.testing.assert_allclose(out["local_heading_x"], 1.0)
    np.testing.assert_allclose(out["local_heading_y"], 0.0, atol=1e-15)


def test_local_heading_magnitude_equals_local_polarization() -> None:
    out = _canonical()
    np.testing.assert_allclose(
        np.hypot(out["local_heading_x"], out["local_heading_y"]),
        out["local_polarization"],
    )


def test_radial_rotation_self_and_local_rotation() -> None:
    out = _canonical()
    rr = out["radial_rotation_self"].to_numpy()
    assert np.isnan(rr[0]), "the individual on the centroid has no radial direction"
    np.testing.assert_allclose(rr[1:], [0.0, -1.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(
        out["local_rotation"], [0.0, 0.0, -1.0 / 3.0, 0.0, 1.0 / 3.0], atol=1e-15
    )


def test_distance_to_group_center() -> None:
    np.testing.assert_allclose(
        _canonical()["dist_to_group_center"], [0.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_radius_actually_excludes() -> None:
    """Pushing id1 past the radius drops it from id0's disc."""
    moved = PLUS.copy()
    moved[1] = (1.6, 0.0)
    out = _apply(_frame(moved, [0.0] * 5))
    assert out["n_local_neighbors"].iloc[0] == 3


# --- the four states, locally -----------------------------------------------


def test_perfect_local_milling() -> None:
    out = _apply(_frame(RING, RING_ANGLES), radius=3.0)
    np.testing.assert_allclose(out["local_polarization"], 0.0, atol=1e-15)
    np.testing.assert_allclose(out["radial_rotation_self"], 1.0)
    np.testing.assert_allclose(out["local_rotation"], 1.0)
    np.testing.assert_allclose(out["group_outer_radius"], 1.0)
    np.testing.assert_array_equal(out["shell_index"], [5, 5, 5, 5])
    np.testing.assert_allclose(out["shell_radial_rotation"], 1.0)


def test_local_swarm() -> None:
    out = _apply(_frame(RING, [0.0, np.pi, 0.0, np.pi]), radius=10.0)
    np.testing.assert_allclose(out["local_polarization"], 0.0, atol=1e-15)
    np.testing.assert_allclose(out["local_rotation"], 0.0, atol=1e-15)


def test_local_transition() -> None:
    out = _apply(_frame(RING, [np.pi, 0.0, 0.0, 0.0]), radius=10.0)
    np.testing.assert_allclose(out["local_polarization"], 0.5)


# --- shells -----------------------------------------------------------------


def test_shells_of_the_canonical_scenario() -> None:
    """R_out = median([0,1,1,1,1]) = 1, width = 1/6, so the ring clips to shell 5."""
    out = _canonical()
    np.testing.assert_allclose(out["group_outer_radius"], 1.0)
    np.testing.assert_array_equal(out["shell_index"], [0, 5, 5, 5, 5])
    np.testing.assert_array_equal(out["shell_n"], [1, 4, 4, 4, 4])
    shell_rr = out["shell_radial_rotation"].to_numpy()
    assert np.isnan(shell_rr[0]), "shell 0 holds only the centroid individual"
    np.testing.assert_allclose(shell_rr[1:], 0.0, atol=1e-15)


def test_shell_index_never_reaches_n_shells() -> None:
    out = _apply(_frame(PLUS, [0.0] * 5), n_shells=4)
    assert out["shell_index"].max() == 3


def test_shell_table_is_reconstructible_from_the_broadcast() -> None:
    """The broadcast form is lossless; the per-shell table is one call away."""
    out = _canonical()
    per_shell = out.drop_duplicates(["frame", "shell_index"])[
        ["frame", "shell_index", "shell_n", "shell_radial_rotation"]
    ]
    assert len(per_shell) == 2
    assert sorted(per_shell["shell_index"]) == [0, 5]


@pytest.mark.parametrize(
    ("positions", "expected_outer", "expected_shells"),
    [
        ([(0.0, 0.0)], 0.0, [-1]),
        ([(0.0, 0.0), (2.0, 0.0)], 1.0, [5, 5]),
        ([(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)], 1.0, [5, 0, 5]),
    ],
    ids=["one", "two", "three"],
)
def test_outer_radius_for_small_groups(
    positions: list[tuple[float, float]],
    expected_outer: float,
    expected_shells: list[int],
) -> None:
    """With fewer than n_peripheral individuals, k falls back to all of them."""
    out = _apply(_frame(positions, [0.0] * len(positions)), radius=10.0)
    np.testing.assert_allclose(out["group_outer_radius"], expected_outer)
    np.testing.assert_array_equal(out["shell_index"], expected_shells)


def test_outer_radius_uses_the_five_most_peripheral() -> None:
    positions = [(float(k), 0.0) for k in range(7)]
    out = _apply(_frame(positions, [0.0] * 7), radius=10.0)
    center = 3.0
    distances = np.abs(np.arange(7, dtype=float) - center)
    expected = float(np.median(np.sort(distances)[-5:]))
    np.testing.assert_allclose(out["group_outer_radius"], expected)


def test_all_coincident_has_no_shells() -> None:
    out = _apply(_frame([(1.0, 1.0)] * 3, [0.0] * 3), radius=10.0)
    np.testing.assert_allclose(out["group_outer_radius"], 0.0)
    np.testing.assert_array_equal(out["shell_index"], [-1, -1, -1])
    np.testing.assert_array_equal(out["shell_n"], [-1, -1, -1])


# --- gating and degenerate rows ---------------------------------------------


def test_min_neighbors_gates_on_headings_not_positions() -> None:
    """Three position-valid neighbors with no usable heading is not a measurement.

    Without this the focal's own heading would be the whole sample and
    local_polarization would read 1.0.
    """
    df = _frame(PLUS[:4], [0.0, np.nan, np.nan, np.nan])
    out = _apply(df, radius=10.0)
    assert out["n_local_neighbors"].iloc[0] == 3
    assert out["n_local_headings"].iloc[0] == 0
    assert np.isnan(out["local_polarization"].iloc[0])

    ungated = _apply(df, radius=10.0, min_neighbors=0)
    assert ungated["local_polarization"].iloc[0] == pytest.approx(1.0)


def test_isolated_focal_is_nan_by_default() -> None:
    out = _apply(_frame([(0.0, 0.0), (50.0, 0.0)], [0.0, 0.0]))
    assert out["n_local_neighbors"].iloc[0] == 0
    assert np.isnan(out["local_polarization"].iloc[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_focal_is_sentinelled_and_contaminates_nothing(
    bad: float,
) -> None:
    intruder = PLUS + [(bad, 5.0)]
    out = _apply(_frame(intruder, [0.0] * 6))
    ghost = out[out["id"] == 5].iloc[0]
    assert ghost["n_local_neighbors"] == -1
    assert ghost["shell_index"] == -1
    assert ghost["shell_n"] == -1
    for col in (
        "local_polarization",
        "local_rotation",
        "dist_to_group_center",
        "radial_rotation_self",
    ):
        assert np.isnan(ghost[col]), col
    # The other five see exactly what they saw without it.
    clean = _canonical()
    for col in ("n_local_neighbors", "local_rotation", "dist_to_group_center"):
        np.testing.assert_allclose(
            out[out["id"] < 5][col].to_numpy(dtype=float),
            clean[col].to_numpy(dtype=float),
        )


# --- headings ---------------------------------------------------------------


def test_velocity_fallback_when_angle_is_absent() -> None:
    first = _frame(PLUS, None, frame=0)
    moved = [(px + 1.0, py) for px, py in PLUS]
    second = _frame(moved, None, frame=1)
    out = _apply(pd.concat([first, second], ignore_index=True))
    assert (out["local_heading_source"] == "velocity").all()
    early = out[out["frame"] == 0]
    late = out[out["frame"] == 1]
    assert early["heading_x"].isna().all()
    assert early["local_polarization"].isna().all()
    np.testing.assert_allclose(late["heading_x"], 1.0)
    np.testing.assert_allclose(late["local_polarization"], 1.0)


def test_all_nan_angle_column_resolves_to_velocity() -> None:
    df = _frame(PLUS, [np.nan] * 5)
    assert (_apply(df)["local_heading_source"] == "velocity").all()


def test_headings_are_computed_before_exclusion() -> None:
    """Excluding a row must not silently change its successor's heading.

    id0 moves +x each frame. Excluding frame 1 leaves frame 2's heading as the
    step from frame 1, not the doubled step from frame 0.
    """
    rows: list[dict[str, object]] = []
    for frame in range(3):
        rows.append(
            {
                "frame": frame,
                "id": 0,
                "X": float(frame),
                "Y": 0.0,
                "sequence": "S1",
                "group": "G",
                "bad_frame": frame == 1,
            }
        )
        rows.append(
            {
                "frame": frame,
                "id": 1,
                "X": float(frame),
                "Y": 0.5,
                "sequence": "S1",
                "group": "G",
                "bad_frame": False,
            }
        )
    out = _apply(
        pd.DataFrame(rows), radius=10.0, exclude_cols=["bad_frame"], min_neighbors=0
    )
    assert len(out) == 5, "one row dropped, four kept plus the unexcluded partner"
    kept = out[(out["id"] == 0) & (out["frame"] == 2)].iloc[0]
    assert kept["heading_x"] == pytest.approx(1.0)


# --- input contract ---------------------------------------------------------


def test_pair_shaped_input_is_refused() -> None:
    df = _frame(PLUS, [0.0] * 5)
    df["perspective"] = 0
    with pytest.raises(ValueError, match="pair-shaped output"):
        _apply(df)


def test_duplicated_frame_id_is_refused() -> None:
    df = pd.concat([_frame(PLUS, [0.0] * 5)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicates"):
        _apply(df)


def test_radius_is_required() -> None:
    with pytest.raises(ValueError, match="radius"):
        LocalOrderMetrics(params={})


def test_body_scale_units_require_a_reference() -> None:
    with pytest.raises(ValueError, match="requires 'body_scale'"):
        LocalOrderMetrics(params={"radius": 3.0, "radius_units": "body_scale"})


def test_subgroup_col_collision_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="collides with a metadata or emitted"):
        LocalOrderMetrics(params={"radius": 1.0, "subgroup_col": "group"})
    with pytest.raises(ValueError, match="collides with a metadata or emitted"):
        LocalOrderMetrics(params={"radius": 1.0, "subgroup_col": "local_rotation"})


def test_missing_subgroup_column_raises() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        _apply(_frame(PLUS, [0.0] * 5), subgroup_col="event")


# --- subgroups --------------------------------------------------------------


def test_subgroups_confine_the_disc() -> None:
    """Two interleaved subgroups: without the partition every disc would mix them."""
    left = _frame(RING, RING_ANGLES, first_id=0)
    left["event"] = 0
    right = _frame([(px + 0.2, py) for px, py in RING], [0.0] * 4, first_id=4)
    right["event"] = 1
    df = pd.concat([left, right], ignore_index=True)

    merged = _apply(df, radius=10.0)
    split = _apply(df, radius=10.0, subgroup_col="event")
    assert merged["local_polarization"].iloc[0] != pytest.approx(
        split["local_polarization"].iloc[0]
    )
    np.testing.assert_allclose(split[split["event"] == 1]["local_polarization"], 1.0)
    np.testing.assert_allclose(
        split[split["event"] == 0]["local_polarization"], 0.0, atol=1e-15
    )


def test_negative_subgroup_rows_are_sentinelled() -> None:
    df = _frame(PLUS, [0.0] * 5)
    df["event"] = [0, 0, 0, 0, -1]
    out = _apply(df, subgroup_col="event", radius=10.0)
    ghost = out[out["event"] == -1].iloc[0]
    assert ghost["n_local_neighbors"] == -1
    assert np.isnan(ghost["local_polarization"])
    assert (out[out["event"] == 0]["n_local_neighbors"] == 3).all()


# --- output contract --------------------------------------------------------


def test_metadata_is_carried_through() -> None:
    out = _canonical()
    for col in ("frame", "time", "id", "sequence", "group"):
        assert col in out.columns, col
    assert (out["sequence"] == "S1").all()
    assert (out["group"] == "G").all()


def test_dtypes() -> None:
    out = _canonical()
    assert out["shell_index"].dtype == np.int16
    assert out["shell_n"].dtype == np.int32
    assert out["n_local_neighbors"].dtype == np.int32
    assert out["local_polarization"].dtype == np.float64


def test_empty_input() -> None:
    assert LocalOrderMetrics(params={"radius": 1.0}).apply(pd.DataFrame()).empty


def test_filter_that_removes_everything() -> None:
    assert _apply(_frame(PLUS, [0.0] * 5), filter_expr="id > 99").empty


def test_row_order_does_not_change_the_result() -> None:
    df = pd.concat(
        [_frame(PLUS, [0.0] * 5, frame=0), _frame(PLUS, [0.0] * 5, frame=1)],
        ignore_index=True,
    )
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    left = _apply(df).sort_values(["frame", "id"]).reset_index(drop=True)
    right = _apply(shuffled).sort_values(["frame", "id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_disc_sums_match_an_independent_oracle() -> None:
    """A deliberately different formulation of the same disc means."""
    df = _frame(PLUS, [0.3, 1.1, -0.7, 2.4, 0.0])
    out = _apply(df, radius=1.5)
    pos = np.array(PLUS)
    ang = np.array([0.3, 1.1, -0.7, 2.4, 0.0])
    expected: list[float] = []
    for i in range(5):
        members = [
            j
            for j in range(5)
            if float(np.hypot(pos[j][0] - pos[i][0], pos[j][1] - pos[i][1])) <= 1.5
        ]
        vx = float(np.mean([np.cos(ang[j]) for j in members]))
        vy = float(np.mean([np.sin(ang[j]) for j in members]))
        expected.append(float(np.hypot(vx, vy)))
    np.testing.assert_allclose(out["local_polarization"], expected)
