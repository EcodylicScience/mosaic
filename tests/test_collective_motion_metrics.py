"""Unit tests for CollectiveMotionMetrics against hand-computed values.

The canonical scenario is four individuals at the corners of a 2x2 square,
ids 0-3 at (0,0), (2,0), (2,2), (0,2). The centroid is (1,1) and every radial
distance is sqrt(2), so each unit radial vector is (+-0.7071, +-0.7071) and every
order parameter below can be evaluated by hand.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.collective_motion_metrics import (
    CollectiveMotionMetrics,
)

CORNERS: list[tuple[float, float]] = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

# Tangential headings for counter-clockwise rotation about (1,1): each is the
# radial direction turned a quarter turn anticlockwise.
MILLING_ANGLES: list[float] = [-np.pi / 4, np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]


def _square(
    angles: list[float] | None,
    frame: int = 0,
    offset: tuple[float, float] = (0.0, 0.0),
    seq: str = "S1",
    group: str = "G",
) -> pd.DataFrame:
    """One frame of the canonical square; ``angles=None`` omits the ANGLE column."""
    rows: list[dict[str, object]] = []
    for i, (px, py) in enumerate(CORNERS):
        row: dict[str, object] = {
            "frame": frame,
            "time": float(frame) * 0.02,
            "id": i,
            "X": px + offset[0],
            "Y": py + offset[1],
            "sequence": seq,
            "group": group,
        }
        if angles is not None:
            row["ANGLE"] = angles[i]
        rows.append(row)
    return pd.DataFrame(rows)


def _apply(df: pd.DataFrame, **params: object) -> pd.DataFrame:
    base: dict[str, object] = {"fps": 1.0}
    base.update(params)
    return CollectiveMotionMetrics(params=base).apply(df)


def _cell(out: pd.DataFrame, key: int, column: str) -> float:
    """One float cell of a frame-indexed result."""
    return float(out[column].loc[key])


# --- the four collective states ---------------------------------------------


def test_perfect_polarization() -> None:
    out = _apply(_square([0.0, 0.0, 0.0, 0.0]))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["polarization"] == pytest.approx(1.0)
    assert row["rotation"] == pytest.approx(0.0, abs=1e-12)
    assert row["rotation_signed"] == pytest.approx(0.0, abs=1e-12)
    assert row["mean_heading"] == pytest.approx(0.0)
    assert row["state"] == "Polarized"
    assert row["n_ids"] == 4
    assert row["n_ids_heading"] == 4
    assert row["n_at_centroid"] == 0
    assert row["heading_source"] == "orientation"
    assert row["sd_major"] == pytest.approx(1.0)
    assert row["sd_minor"] == pytest.approx(1.0)
    assert row["elongation"] == pytest.approx(1.0)
    assert row["area"] == pytest.approx(4.0)
    assert row["density"] == pytest.approx(1.0)
    assert row["centroid_x"] == pytest.approx(1.0)
    assert row["centroid_y"] == pytest.approx(1.0)


def test_perfect_milling_signed_rotation_is_positive_for_counter_clockwise() -> None:
    """The sign pins the chirality convention: cross(r_hat, u) > 0 is anticlockwise."""
    out = _apply(_square(MILLING_ANGLES)).iloc[0]
    assert out["polarization"] == pytest.approx(0.0, abs=1e-12)
    assert out["rotation"] == pytest.approx(1.0)
    assert out["rotation_signed"] == pytest.approx(1.0)
    assert out["state"] == "Milling"
    assert out["area"] == pytest.approx(4.0)


def test_clockwise_milling_flips_only_the_signed_column() -> None:
    out = _apply(_square([a + np.pi for a in MILLING_ANGLES])).iloc[0]
    assert out["rotation"] == pytest.approx(1.0)
    assert out["rotation_signed"] == pytest.approx(-1.0)
    assert out["state"] == "Milling"


def test_swarm() -> None:
    """Opposed pairs, not a cyclic sequence.

    The pairing is load-bearing: ``[0, pi/2, pi, 3pi/2]`` over these corners is
    a half-formed mill with ``rotation_signed`` near +0.707, and would classify
    as Milling rather than Swarm.
    """
    out = _apply(_square([0.0, np.pi, 0.0, np.pi])).iloc[0]
    assert out["polarization"] == pytest.approx(0.0, abs=1e-12)
    assert out["rotation_signed"] == pytest.approx(0.0, abs=1e-12)
    assert out["state"] == "Swarm"


def test_transitional() -> None:
    """One of four reversed: O_p = 2/4, |O_r| = 1/(2*sqrt(2))."""
    out = _apply(_square([np.pi, 0.0, 0.0, 0.0])).iloc[0]
    assert out["polarization"] == pytest.approx(0.5)
    assert out["rotation_signed"] == pytest.approx(-1.0 / (2.0 * np.sqrt(2.0)))
    assert out["rotation"] == pytest.approx(1.0 / (2.0 * np.sqrt(2.0)))
    assert out["state"] == "Transitional"


# --- degenerate geometry ----------------------------------------------------


def test_collinear_group() -> None:
    """Three in a row: the middle one sits on the centroid, so it has no radial
    direction; the shape collapses onto one axis."""
    df = pd.DataFrame(
        [
            {
                "frame": 0,
                "id": i,
                "X": float(i),
                "Y": 0.0,
                "ANGLE": 0.0,
                "sequence": "S1",
                "group": "G",
            }
            for i in range(3)
        ]
    )
    out = _apply(df).iloc[0]
    assert out["centroid_x"] == pytest.approx(1.0)
    assert out["n_ids"] == 3
    assert out["n_at_centroid"] == 1
    assert out["polarization"] == pytest.approx(1.0)
    assert out["rotation_signed"] == pytest.approx(0.0, abs=1e-12)
    assert out["sd_major"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert out["sd_minor"] == pytest.approx(0.0)
    assert out["elongation"] == np.inf
    assert out["principal_axis_angle"] == pytest.approx(0.0)
    assert np.isnan(out["area"])
    assert np.isnan(out["density"])
    assert out["state"] == "Polarized"


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_position_is_dropped_entirely(bad: float) -> None:
    """Scrubbing is joint: the intruder's finite Y must not reach the centroid.

    A per-column nanmean would admit Y=5.0 and move ``centroid_y`` from 1.0 to
    1.8; an untreated ``inf`` would make it infinite outright.
    """
    clean = _square([0.0, 0.0, 0.0, 0.0])
    intruder = pd.DataFrame(
        [
            {
                "frame": 0,
                "time": 0.0,
                "id": 4,
                "X": bad,
                "Y": 5.0,
                "ANGLE": 0.0,
                "sequence": "S1",
                "group": "G",
            }
        ]
    )
    dirty = pd.concat([clean, intruder], ignore_index=True)
    out = _apply(dirty)
    assert out.iloc[0]["n_ids"] == 4
    assert out.iloc[0]["centroid_y"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(out, _apply(clean), check_exact=True)


def test_partial_nan_heading_is_row_masked() -> None:
    """A row with a NaN angle leaves the polarization divisor, it does not
    contribute a shortened vector to the numerator."""
    df = _square([0.0, 0.0, 0.0, np.nan])
    out = _apply(df, min_individuals=3).iloc[0]
    assert out["n_ids"] == 4
    assert out["n_ids_heading"] == 3
    assert out["polarization"] == pytest.approx(1.0)


# --- the min_individuals gate -----------------------------------------------


def test_below_min_individuals_emits_a_row_of_nans() -> None:
    """Never drop a row: a gap in the frame axis breaks every time-series consumer."""
    df = pd.DataFrame(
        [
            {
                "frame": 0,
                "id": 0,
                "X": 0.0,
                "Y": 0.0,
                "ANGLE": 0.0,
                "sequence": "S1",
                "group": "G",
            }
        ]
    )
    out = _apply(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_ids"] == 1
    for col in ("polarization", "rotation", "rotation_signed", "group_angvel"):
        assert np.isnan(row[col]), col
    for col in ("sd_major", "sd_minor", "elongation", "area", "density"):
        assert np.isnan(row[col]), col
    assert row["state"] == "Undefined"
    # The centroid is exactly what it claims even for one individual.
    assert row["centroid_x"] == pytest.approx(0.0)
    assert row["centroid_y"] == pytest.approx(0.0)


def test_single_individual_rotation_is_nan_not_zero() -> None:
    """O_p is 1 by construction for N=1. Substituting a zero radial vector would
    give (1.0, 0.0) -- Polarized with maximum confidence."""
    df = pd.DataFrame(
        [
            {
                "frame": 0,
                "id": 0,
                "X": 0.0,
                "Y": 0.0,
                "ANGLE": 0.0,
                "sequence": "S1",
                "group": "G",
            }
        ]
    )
    out = _apply(df, min_individuals=1).iloc[0]
    assert out["polarization"] == pytest.approx(1.0)
    assert np.isnan(out["rotation"])
    assert out["n_at_centroid"] == 1


# --- headings ---------------------------------------------------------------


def test_velocity_fallback_when_angle_is_absent() -> None:
    df = pd.concat(
        [_square(None, frame=0), _square(None, frame=1, offset=(1.0, 0.0))],
        ignore_index=True,
    )
    out = _apply(df).set_index("frame")
    assert (out["heading_source"] == "velocity").all()
    assert out.loc[0, "n_ids_heading"] == 0
    assert np.isnan(_cell(out, 0, "polarization"))
    assert out.loc[0, "state"] == "Undefined"
    assert out.loc[1, "polarization"] == pytest.approx(1.0)
    assert out.loc[1, "mean_heading"] == pytest.approx(0.0)
    assert np.isnan(_cell(out, 0, "centroid_speed"))
    assert out.loc[1, "centroid_speed"] == pytest.approx(1.0)
    assert out.loc[1, "n_centroid_common"] == 4


def test_all_nan_angle_column_falls_back_to_velocity() -> None:
    df = pd.concat(
        [
            _square([np.nan] * 4, frame=0),
            _square([np.nan] * 4, frame=1, offset=(1.0, 0.0)),
        ],
        ignore_index=True,
    )
    out = _apply(df).set_index("frame")
    assert (out["heading_source"] == "velocity").all()
    assert out.loc[1, "polarization"] == pytest.approx(1.0)


def test_orientation_source_refuses_to_fall_back() -> None:
    with pytest.raises(ValueError, match="no usable orientation column"):
        _apply(_square(None), heading_source="orientation")


# --- centroid kinematics ----------------------------------------------------


def test_stationary_group_has_no_centroid_heading() -> None:
    """arctan2(0, 0) is silently 0.0 -- due east. A mill would read as traveling."""
    df = pd.concat(
        [_square([0.0] * 4, frame=0), _square([0.0] * 4, frame=1)], ignore_index=True
    )
    out = _apply(df).set_index("frame")
    assert out.loc[1, "centroid_speed"] == pytest.approx(0.0)
    assert np.isnan(_cell(out, 1, "centroid_heading"))
    assert np.isfinite(_cell(out, 1, "sd_major"))
    assert np.isfinite(_cell(out, 1, "area"))


def test_frame_gap_uses_the_true_elapsed_interval() -> None:
    gapped = pd.concat(
        [_square([0.0] * 4, frame=0), _square([0.0] * 4, frame=2, offset=(2.0, 0.0))],
        ignore_index=True,
    )
    out = _apply(gapped).set_index("frame")
    assert out.loc[2, "centroid_speed"] == pytest.approx(1.0)

    capped = _apply(gapped, max_frame_gap=1).set_index("frame")
    assert np.isnan(_cell(capped, 2, "centroid_speed"))


def test_centroid_velocity_uses_the_common_id_set() -> None:
    """A track that merely appears must not look like the whole group lurched.

    Differencing the all-ids centroid would report 20 px/frame here with nobody
    having moved.
    """
    later = pd.concat(
        [
            _square([0.0] * 4, frame=1),
            pd.DataFrame(
                [
                    {
                        "frame": 1,
                        "time": 0.02,
                        "id": 4,
                        "X": 100.0,
                        "Y": 100.0,
                        "ANGLE": 0.0,
                        "sequence": "S1",
                        "group": "G",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df = pd.concat([_square([0.0] * 4, frame=0), later], ignore_index=True)
    out = _apply(df).set_index("frame")
    assert out.loc[1, "n_ids"] == 5
    assert out.loc[1, "n_centroid_common"] == 4
    assert out.loc[1, "centroid_speed"] == pytest.approx(0.0)


def test_group_angular_velocity_of_a_rigid_mill() -> None:
    """Four individuals rotating a tenth of a radian about their centroid."""
    theta = 0.1
    frames: list[pd.DataFrame] = []
    for frame in (0, 1):
        rot = theta * frame
        rows: list[dict[str, object]] = []
        for i, (px, py) in enumerate(CORNERS):
            dx, dy = px - 1.0, py - 1.0
            rows.append(
                {
                    "frame": frame,
                    "id": i,
                    "X": 1.0 + dx * np.cos(rot) - dy * np.sin(rot),
                    "Y": 1.0 + dx * np.sin(rot) + dy * np.cos(rot),
                    "ANGLE": 0.0,
                    "sequence": "S1",
                    "group": "G",
                }
            )
        frames.append(pd.DataFrame(rows))
    out = _apply(pd.concat(frames, ignore_index=True)).set_index("frame")
    # A chord over one frame slightly undershoots the arc, hence the tolerance.
    assert out.loc[1, "group_angvel"] == pytest.approx(theta, rel=0.02)
    assert out.loc[1, "centroid_speed"] == pytest.approx(0.0, abs=1e-12)


# --- subgroups --------------------------------------------------------------


def _two_subgroups(frame: int = 0) -> pd.DataFrame:
    polar = _square([0.0] * 4, frame=frame)
    polar["event"] = 0
    mill = _square(MILLING_ANGLES, frame=frame, offset=(10.0, 10.0))
    mill["id"] = mill["id"] + 4
    mill["event"] = 1
    return pd.concat([polar, mill], ignore_index=True)


def test_subgroup_mode_describes_each_subgroup_separately() -> None:
    out = _apply(_two_subgroups(), subgroup_col="event").set_index("event")
    assert len(out) == 2
    assert out.loc[0, "state"] == "Polarized"
    assert out.loc[1, "state"] == "Milling"
    assert out.loc[0, "centroid_x"] == pytest.approx(1.0)
    assert out.loc[1, "centroid_x"] == pytest.approx(11.0)
    assert (out["n_ids"] == 4).all()


def test_subgroup_switch_contributes_to_neither_centroid_velocity() -> None:
    """An individual crossing subgroups has two positions measured against two
    origins; that difference is not a velocity of either subgroup."""
    first = _two_subgroups(frame=0)
    second = _two_subgroups(frame=1)
    second.loc[second["id"] == 3, "event"] = 1
    out = _apply(
        pd.concat([first, second], ignore_index=True), subgroup_col="event"
    ).set_index(["frame", "event"])
    assert out.loc[(1, 0), "n_ids"] == 3
    assert out.loc[(1, 0), "n_centroid_common"] == 3
    assert out.loc[(1, 1), "n_ids"] == 5
    assert out.loc[(1, 1), "n_centroid_common"] == 4


def test_filter_expr_drops_the_non_event_pseudo_group() -> None:
    df = _two_subgroups()
    df.loc[df["id"] == 0, "event"] = -1
    out = _apply(df, subgroup_col="event", filter_expr="event >= 0")
    assert sorted(out["event"].tolist()) == [0, 1]


def test_subgroup_col_colliding_with_an_output_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="collides with a metadata or emitted"):
        CollectiveMotionMetrics(params={"subgroup_col": "group"})
    with pytest.raises(ValueError, match="collides with a metadata or emitted"):
        CollectiveMotionMetrics(params={"subgroup_col": "polarization"})


# --- area methods -----------------------------------------------------------


def test_alpha_shape_area() -> None:
    out = _apply(_square([0.0] * 4), area_method="alpha_shape", alpha=10.0).iloc[0]
    assert out["area"] == pytest.approx(4.0)
    assert out["alpha_n_triangles"] == 2


def test_alpha_shape_admitting_nothing_gives_no_density() -> None:
    """Never +inf: an unmeasurable area must read as absent, not as zero."""
    out = _apply(_square([0.0] * 4), area_method="alpha_shape", alpha=1e-6).iloc[0]
    assert np.isnan(out["area"])
    assert np.isnan(out["density"])
    assert out["alpha_n_triangles"] == 0


def test_alpha_shape_without_alpha_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="requires 'alpha'"):
        CollectiveMotionMetrics(params={"area_method": "alpha_shape"})


def test_area_method_none_skips_both_columns() -> None:
    out = _apply(_square([0.0] * 4), area_method="none").iloc[0]
    assert np.isnan(out["area"])
    assert np.isnan(out["density"])


# --- output contract --------------------------------------------------------


def test_metadata_and_composition_names() -> None:
    out = _apply(_square([0.0] * 4))
    assert (out["sequence"] == "S1").all()
    assert (out["group"] == "G").all()
    assert "time" in out.columns
    # ffgroups-metrics reads this exact name, and the overlay reads "state" as
    # a label column.
    assert "centroid_heading" in out.columns
    assert "state" in out.columns


def test_mean_speed_only_when_requested() -> None:
    df = _square([0.0] * 4)
    df["speed"] = [1.0, 2.0, 3.0, 4.0]
    assert "mean_speed" not in _apply(df).columns
    assert _apply(df, speed_col="speed").iloc[0]["mean_speed"] == pytest.approx(2.5)


def test_empty_input() -> None:
    assert _apply(pd.DataFrame()).empty


def test_filter_that_removes_everything() -> None:
    assert _apply(_square([0.0] * 4), filter_expr="id > 99").empty


def test_exclude_cols() -> None:
    df = _square([0.0] * 4)
    df["bad_frame"] = [False, False, False, True]
    assert _apply(df, exclude_cols=["bad_frame"]).iloc[0]["n_ids"] == 3


def test_row_order_does_not_change_the_result() -> None:
    df = pd.concat(
        [_square(MILLING_ANGLES, frame=0), _square(MILLING_ANGLES, frame=1)],
        ignore_index=True,
    )
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    pd.testing.assert_frame_equal(_apply(df), _apply(shuffled), check_exact=True)
    assert _apply(df)["frame"].is_monotonic_increasing


def test_float32_input_matches_float64() -> None:
    df64 = _square(MILLING_ANGLES)
    df32 = df64.copy()
    for col in ("X", "Y", "ANGLE"):
        df32[col] = df32[col].astype(np.float32)
    out32 = _apply(df32)
    assert out32.dtypes["polarization"] == np.float64
    np.testing.assert_allclose(
        out32["rotation"].to_numpy(), _apply(df64)["rotation"].to_numpy(), atol=1e-7
    )


def test_an_all_nan_frame_emits_no_warning() -> None:
    df = _square([0.0] * 4)
    df["X"] = np.nan
    df["Y"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _apply(df)
    assert out.iloc[0]["n_ids"] == 0
    assert out.iloc[0]["state"] == "Undefined"
