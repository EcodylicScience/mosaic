"""Unit tests for the shared collective-motion primitives.

Every expected value here is hand-computed from the Tunstrom 2013 definitions,
so a change in the arithmetic fails against the paper rather than against a
previously recorded output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mosaic.behavior.feature_library.collective_math import (
    STATE_HIGH,
    STATE_LOW,
    alpha_shape_area,
    backward_dt,
    classify_state,
    cross2,
    hull_area,
    polarization,
    principal_axes,
    resolve_heading_source,
    scrub_positions,
    step_masks,
    unit_headings,
    unit_radial,
)

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


# --- cross2 -----------------------------------------------------------------


def test_cross2_sign() -> None:
    """+x cross +y is +1; the reverse is -1. This pins the chirality convention."""
    one = np.array([1.0])
    zero = np.array([0.0])
    assert cross2(one, zero, zero, one)[0] == 1.0
    assert cross2(zero, one, one, zero)[0] == -1.0


def test_cross2_emits_no_deprecation() -> None:
    """np.cross on 2-vectors is deprecated in numpy 2; cross2 must not use it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = cross2(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))


# --- scrub_positions --------------------------------------------------------


def test_scrub_positions_is_joint() -> None:
    """A row loses both coordinates when either is non-finite, never just one."""
    x = np.array([1.0, np.inf, 2.0, -np.inf])
    y = np.array([1.0, 3.0, np.nan, 4.0])
    sx, sy, finite = scrub_positions(x, y)
    np.testing.assert_array_equal(finite, [True, False, False, False])
    assert np.isnan(sy[1]), "the inf row must lose its finite Y as well"
    assert np.isnan(sx[2]), "the NaN-Y row must lose its finite X as well"
    assert sx[0] == 1.0


# --- step_masks -------------------------------------------------------------


def test_step_masks() -> None:
    ids = np.array([0, 0, 1, 1])
    order = np.array([0.0, 1.0, 0.0, 2.0])
    same_id, dstep = step_masks(ids, order)
    np.testing.assert_array_equal(same_id, [False, True, False, True])
    assert np.isnan(dstep[0])
    assert dstep[1] == 1.0
    assert dstep[3] == 2.0


def test_step_masks_single_row() -> None:
    same_id, dstep = step_masks(np.array([0]), np.array([0.0]))
    assert not same_id[0]
    assert np.isnan(dstep[0])


# --- heading resolution -----------------------------------------------------


def test_resolve_heading_source_auto_prefers_orientation() -> None:
    assert resolve_heading_source("auto", np.array([0.0, 1.0])) == "orientation"


def test_resolve_heading_source_auto_falls_back_on_all_nan() -> None:
    """An all-NaN angle column is not an orientation signal, only a column."""
    assert resolve_heading_source("auto", np.array([np.nan, np.nan])) == "velocity"
    assert resolve_heading_source("auto", None) == "velocity"


def test_resolve_heading_source_orientation_refuses_to_fall_back() -> None:
    with pytest.raises(ValueError, match="no usable orientation column"):
        resolve_heading_source("orientation", None)
    with pytest.raises(ValueError, match="no usable orientation column"):
        resolve_heading_source("orientation", np.array([np.nan]))


def test_unit_headings_from_orientation() -> None:
    ux, uy = unit_headings(
        "orientation",
        np.array([0.0, np.pi / 2]),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_allclose(ux, [1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(uy, [0.0, 1.0], atol=1e-15)


def test_unit_headings_from_velocity_pads_with_nan() -> None:
    """The first row of an individual is NaN, never a copy of the second."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros(3)
    same_id = np.array([False, True, True])
    ux, uy = unit_headings("velocity", None, x, y, same_id)
    assert np.isnan(ux[0]) and np.isnan(uy[0])
    np.testing.assert_allclose(ux[1:], [1.0, 1.0])
    np.testing.assert_allclose(uy[1:], [0.0, 0.0])


def test_unit_headings_from_velocity_nan_when_stationary() -> None:
    """A zero-length displacement has no direction; it must not become one."""
    ux, uy = unit_headings(
        "velocity", None, np.zeros(2), np.zeros(2), np.array([False, True])
    )
    assert np.isnan(ux[1]) and np.isnan(uy[1])


# --- backward_dt ------------------------------------------------------------


def test_backward_dt_prefers_frame_and_fps() -> None:
    order = np.array([0.0, 1.0, 3.0])
    same_id = np.array([False, True, True])
    _, dstep = step_masks(np.array([0, 0, 0]), order)
    dt = backward_dt(order, True, None, 2.0, same_id, dstep, None)
    assert np.isnan(dt[0])
    np.testing.assert_allclose(dt[1:], [0.5, 1.0])


def test_backward_dt_honours_max_frame_gap() -> None:
    order = np.array([0.0, 3.0])
    same_id = np.array([False, True])
    _, dstep = step_masks(np.array([0, 0]), order)
    assert np.isfinite(backward_dt(order, True, None, 1.0, same_id, dstep, None)[1])
    assert np.isnan(backward_dt(order, True, None, 1.0, same_id, dstep, 2)[1])


def test_backward_dt_falls_back_to_time_then_order() -> None:
    order = np.array([0.0, 1.0])
    same_id = np.array([False, True])
    _, dstep = step_masks(np.array([0, 0]), order)
    time = np.array([0.0, 0.25])
    np.testing.assert_allclose(
        backward_dt(order, True, time, None, same_id, dstep, None)[1], 0.25
    )
    np.testing.assert_allclose(
        backward_dt(order, True, None, None, same_id, dstep, None)[1], 1.0
    )


# --- unit_radial and polarization -------------------------------------------


def test_unit_radial_at_centre_is_nan_not_zero() -> None:
    rhx, rhy, at_centre = unit_radial(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert at_centre[0] and not at_centre[1]
    assert np.isnan(rhx[0]) and np.isnan(rhy[0])
    assert rhx[1] == 1.0


def test_unit_radial_flags_non_finite() -> None:
    """at_centre is ~(|r| > 0), so it also catches a NaN radial vector."""
    _, _, at_centre = unit_radial(np.array([np.nan]), np.array([1.0]))
    assert at_centre[0]


def test_polarization_divides_by_the_usable_count() -> None:
    np.testing.assert_allclose(
        polarization(np.array([2.0]), np.array([0.0]), np.array([4.0])), [0.5]
    )
    assert np.isnan(polarization(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0])


# --- principal_axes ---------------------------------------------------------


def test_principal_axes_isotropic() -> None:
    sd_major, sd_minor, elong, angle = principal_axes(
        np.array([1.0]), np.array([1.0]), np.array([0.0])
    )
    np.testing.assert_allclose([sd_major[0], sd_minor[0], elong[0]], [1.0, 1.0, 1.0])
    assert angle[0] == 0.0


def test_principal_axes_collinear() -> None:
    """Three points at 0, 1, 2 have variance 2/3 along x and none across it."""
    sd_major, sd_minor, elong, angle = principal_axes(
        np.array([2.0 / 3.0]), np.array([0.0]), np.array([0.0])
    )
    np.testing.assert_allclose(sd_major[0], np.sqrt(2.0 / 3.0))
    assert sd_minor[0] == 0.0
    assert elong[0] == np.inf
    assert angle[0] == 0.0


def test_principal_axes_near_collinear_does_not_produce_nan() -> None:
    """The smaller eigenvalue rounds negative here; the clamp is the fix."""
    sd_major, sd_minor, _, _ = principal_axes(
        np.array([1.0]), np.array([1e-16]), np.array([0.0])
    )
    assert np.isfinite(sd_major[0]) and np.isfinite(sd_minor[0])


def test_principal_axes_all_zero_is_nan_elongation() -> None:
    _, _, elong, _ = principal_axes(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert np.isnan(elong[0])


def test_principal_axes_angle_range() -> None:
    """A 45-degree major axis, and the angle stays in (-pi/2, pi/2]."""
    _, _, _, angle = principal_axes(np.array([1.0]), np.array([1.0]), np.array([0.9]))
    np.testing.assert_allclose(angle[0], np.pi / 4)


# --- areas ------------------------------------------------------------------


def test_hull_area_is_the_enclosed_area() -> None:
    """ConvexHull.volume is the 2-D area; .area would be the perimeter, 8.0."""
    np.testing.assert_allclose(hull_area(SQUARE), 4.0)


@pytest.mark.parametrize(
    "pts",
    [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]]),
    ],
    ids=["two-points", "collinear", "with-nan"],
)
def test_hull_area_degenerate_is_nan_without_raising(pts: np.ndarray) -> None:
    assert np.isnan(hull_area(pts))


def test_alpha_shape_area_admits_the_square() -> None:
    area, n_tri = alpha_shape_area(SQUARE, 10.0)
    assert np.isclose(area, 4.0)
    assert n_tri == 2


def test_alpha_shape_area_admits_nothing_is_nan_not_zero() -> None:
    """Zero area would make a density infinite; absence must read as absence."""
    area, n_tri = alpha_shape_area(SQUARE, 1e-6)
    assert np.isnan(area)
    assert n_tri == 0


def test_alpha_shape_area_degenerate() -> None:
    assert alpha_shape_area(np.array([[0.0, 0.0], [1.0, 0.0]]), 10.0) == (
        pytest.approx(float("nan"), nan_ok=True),
        0,
    )
    area, n_tri = alpha_shape_area(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 10.0)
    assert np.isnan(area)
    assert n_tri == 0


def test_alpha_shape_excludes_a_sliver_the_hull_includes() -> None:
    """Why an alpha shape at all: a hull spans thin gaps, an alpha shape need not.

    Four points whose hull is the triangle (0,0), (10,0), (5,3), area 15. The
    Delaunay triangulation splits it into two fat triangles totalling 14.5 plus
    one sliver of area 0.5 along the base, whose circumradius is about 125. An
    alpha of 10 admits the fat pair and rejects the sliver; an alpha of 200
    admits all three and recovers the hull.
    """
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.1], [5.0, 3.0]])
    np.testing.assert_allclose(hull_area(pts), 15.0)

    tight_area, tight_n = alpha_shape_area(pts, 10.0)
    assert np.isclose(tight_area, 14.5)
    assert tight_n == 2

    loose_area, loose_n = alpha_shape_area(pts, 200.0)
    assert np.isclose(loose_area, 15.0)
    assert loose_n == 3


# --- classify_state ---------------------------------------------------------


@pytest.mark.parametrize(
    ("rotation", "polarization_value", "expected"),
    [
        (0.0, 1.0, "Polarized"),
        (1.0, 0.0, "Milling"),
        (0.0, 0.0, "Swarm"),
        (0.5, 0.5, "Transitional"),
        (0.0, 0.5, "Transitional"),
        (np.nan, 0.2, "Undefined"),
        (0.2, np.nan, "Undefined"),
    ],
)
def test_classify_state(
    rotation: float, polarization_value: float, expected: str
) -> None:
    got = classify_state(np.array([rotation]), np.array([polarization_value]))
    assert got[0] == expected


def test_classify_state_nan_is_not_transitional() -> None:
    """The reference implementation lets NaN fall through to the transitional
    class, so every tracking failure reads as a behavioral transition."""
    assert classify_state(np.array([np.nan]), np.array([np.nan]))[0] == "Undefined"


def test_state_thresholds_match_the_paper() -> None:
    assert (STATE_LOW, STATE_HIGH) == (0.35, 0.65)
