"""Shape geometry: signed distance, anchors, placement, the center ratio and the fit."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

import mosaic.core.region_geometry as region_geometry
from mosaic.core.region_geometry import (
    EstimateShape,
    anchor_point,
    boundary_center,
    contains,
    estimate_boundary,
    inside,
    radius_norm,
    resolve_roi,
    signed_distance,
)
from mosaic.core.regions import (
    Anchor,
    CircleShape,
    PolygonShape,
    RectShape,
    RingShape,
    Roi,
    Shape,
)

NAN = math.nan

CIRCLE = CircleShape(cx=300.0, cy=200.0, r=100.0)
RING = RingShape(cx=300.0, cy=200.0, r_inner=50.0, r_outer=100.0)
RECT = RectShape(x0=100.0, y0=50.0, x1=300.0, y1=150.0)
"""A 200x100 rect centered on (200, 100)."""

L_POLYGON = PolygonShape(
    vertices=((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0))
)
"""An L of area 12 whose notch is the square (2, 2)-(4, 4).

Its centroid pools the 4x2 bar (area 8, centroid (2, 1)) and the 2x2 arm (area 4,
centroid (1, 3)): ((8 * 2 + 4 * 1) / 12, (8 * 1 + 4 * 3) / 12) = (5/3, 5/3).
"""


def _array(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _distances(shape: Shape, points: list[tuple[float, float]]) -> list[float]:
    x = _array([p[0] for p in points])
    y = _array([p[1] for p in points])
    return [float(d) for d in signed_distance(shape, x, y)]


@pytest.mark.parametrize(
    ("shape", "points", "expected"),
    [
        (
            CIRCLE,
            [(350.0, 200.0), (400.0, 200.0), (300.0, 350.0)],
            [-50.0, 0.0, 50.0],
        ),
        (
            RING,
            [
                (380.0, 200.0),
                (350.0, 200.0),
                (300.0, 100.0),
                (300.0, 200.0),
                (320.0, 200.0),
                (300.0, 330.0),
            ],
            [-20.0, 0.0, 0.0, 50.0, 30.0, 30.0],
        ),
        (
            RECT,
            [(220.0, 110.0), (300.0, 120.0), (200.0, 170.0), (303.0, 154.0)],
            [-40.0, 0.0, 20.0, 5.0],
        ),
        (
            L_POLYGON,
            [(1.0, 0.5), (3.0, 0.0), (3.0, 3.0), (3.5, 2.5), (6.0, 1.0)],
            [-0.5, 0.0, 1.0, 0.5, 2.0],
        ),
    ],
    ids=["circle", "ring", "rect", "l-polygon"],
)
def test_signed_distance_is_negative_inside_zero_on_the_edge_positive_outside(
    shape: Shape, points: list[tuple[float, float]], expected: list[float]
) -> None:
    assert _distances(shape, points) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("shape", "points", "expected"),
    [
        (CIRCLE, [(350.0, 200.0), (400.0, 200.0), (300.0, 350.0)], [1, 1, 0]),
        (
            RING,
            [(380.0, 200.0), (350.0, 200.0), (300.0, 200.0), (300.0, 330.0)],
            [1, 1, 0, 0],
        ),
        (RECT, [(220.0, 110.0), (300.0, 120.0), (303.0, 154.0)], [1, 1, 0]),
        (L_POLYGON, [(1.0, 0.5), (3.0, 0.0), (3.0, 3.0), (6.0, 1.0)], [1, 1, 0, 0]),
    ],
    ids=["circle", "ring-hole-is-outside", "rect", "l-notch-is-outside"],
)
def test_contains_counts_the_edge_as_inside(
    shape: Shape, points: list[tuple[float, float]], expected: list[int]
) -> None:
    x = _array([p[0] for p in points])
    y = _array([p[1] for p in points])
    assert contains(shape, x, y).tolist() == [bool(e) for e in expected]


def test_inside_counts_a_zero_distance_as_inside_and_a_missing_one_as_outside() -> None:
    distances = _array([-2.0, 0.0, 1e-12, 3.0, NAN])
    assert inside(distances).tolist() == [True, True, False, False, False]


@pytest.mark.parametrize(
    "shape", [CIRCLE, RING, RECT, L_POLYGON], ids=["circle", "ring", "rect", "polygon"]
)
def test_a_missing_coordinate_has_no_distance_and_is_not_contained(
    shape: Shape,
) -> None:
    x = _array([NAN, 1.0, NAN])
    y = _array([1.0, NAN, NAN])
    assert np.isnan(signed_distance(shape, x, y)).all()
    assert not contains(shape, x, y).any()


def test_the_polygon_in_blocks_equals_the_polygon_row_by_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 5.0, 10_003)
    y = rng.uniform(-1.0, 5.0, 10_003)
    x[[0, 5000, 10_002]] = NAN
    y[[1, 5000]] = NAN
    monkeypatch.setattr(region_geometry, "_POLYGON_BLOCK_ROWS", 7)

    blocked_distance = signed_distance(L_POLYGON, x, y)
    blocked_ratio = radius_norm(L_POLYGON, x, y)

    by_row_distance = np.array(
        [
            signed_distance(L_POLYGON, x[i : i + 1], y[i : i + 1])[0]
            for i in range(x.size)
        ]
    )
    by_row_ratio = np.array(
        [radius_norm(L_POLYGON, x[i : i + 1], y[i : i + 1])[0] for i in range(x.size)]
    )
    np.testing.assert_array_equal(blocked_distance, by_row_distance)
    np.testing.assert_array_equal(blocked_ratio, by_row_ratio)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (CIRCLE, (300.0, 200.0)),
        (RING, (300.0, 200.0)),
        (RECT, (200.0, 100.0)),
        (L_POLYGON, (5.0 / 3.0, 5.0 / 3.0)),
    ],
    ids=["circle", "ring", "rect", "l-polygon-centroid"],
)
def test_boundary_center(shape: Shape, expected: tuple[float, float]) -> None:
    assert boundary_center(shape) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("center", (200.0, 100.0)),
        ("top", (200.0, 50.0)),
        ("bottom", (200.0, 150.0)),
        ("left", (100.0, 100.0)),
        ("right", (300.0, 100.0)),
        ("top_left", (100.0, 50.0)),
        ("top_right", (300.0, 50.0)),
        ("bottom_left", (100.0, 150.0)),
        ("bottom_right", (300.0, 150.0)),
    ],
)
def test_each_anchor_of_a_rect_boundary(
    anchor: Anchor, expected: tuple[float, float]
) -> None:
    assert anchor_point(RECT, anchor) == expected


def test_the_center_anchor_of_a_polygon_is_its_centroid() -> None:
    assert anchor_point(L_POLYGON, "center") == pytest.approx((5 / 3, 5 / 3))


@pytest.mark.parametrize(
    ("roi", "boundary", "expected"),
    [
        (
            Roi(
                name="center",
                shape=CircleShape(cx=0.0, cy=0.0, r=0.5),
                placement="arena",
            ),
            CIRCLE,
            CircleShape(cx=300.0, cy=200.0, r=50.0),
        ),
        (
            Roi(
                name="wall",
                shape=RingShape(cx=0.0, cy=0.0, r_inner=0.8, r_outer=1.0),
                placement="arena",
            ),
            CIRCLE,
            RingShape(cx=300.0, cy=200.0, r_inner=80.0, r_outer=100.0),
        ),
        (
            Roi(
                name="nest",
                shape=RectShape(x0=0.0, y0=0.0, x1=0.4, y1=0.4),
                placement="arena",
                anchor="top_left",
            ),
            RECT,
            RectShape(x0=100.0, y0=50.0, x1=140.0, y1=70.0),
        ),
        (
            Roi(
                name="inscribed",
                shape=CircleShape(cx=0.0, cy=0.0, r=1.0),
                placement="arena",
            ),
            RECT,
            CircleShape(cx=200.0, cy=100.0, r=50.0),
        ),
        (
            Roi(
                name="corner",
                shape=PolygonShape(
                    vertices=((0.0, 0.0), (-0.5, 0.0), (0.0, -0.5)),
                ),
                placement="arena",
                anchor="bottom_right",
            ),
            RECT,
            PolygonShape(vertices=((300.0, 150.0), (250.0, 150.0), (300.0, 125.0))),
        ),
    ],
    ids=[
        "half-radius-disc",
        "outer-fifth-ring",
        "corner-fifth-rect",
        "inscribed-circle",
        "corner-triangle",
    ],
)
def test_resolve_roi_scales_by_half_extents_and_moves_to_the_anchor(
    roi: Roi, boundary: CircleShape | RectShape, expected: Shape
) -> None:
    assert resolve_roi(roi, boundary) == expected


def test_resolve_roi_returns_an_absolute_shape_unchanged() -> None:
    shape = RectShape(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    roi = Roi(name="feeder", shape=shape)
    assert resolve_roi(roi, CIRCLE) is shape


@pytest.mark.parametrize(
    ("boundary", "points", "expected"),
    [
        (
            CIRCLE,
            [(300.0, 200.0), (400.0, 200.0), (300.0, 100.0), (450.0, 200.0)],
            [0.0, 1.0, 1.0, 1.5],
        ),
        (
            RECT,
            [
                (200.0, 100.0),
                (300.0, 100.0),
                (200.0, 150.0),
                (300.0, 150.0),
                (350.0, 100.0),
                (200.0, 175.0),
            ],
            [0.0, 1.0, 1.0, 1.0, 1.5, 1.5],
        ),
        (
            L_POLYGON,
            [(5 / 3, 5 / 3), (4.0, 1.0), (3.5, 1.0), (3.5, 2.5)],
            [0.0, 1.0, 11 / 14, 2.5],
        ),
    ],
    ids=["circle", "rect", "l-polygon"],
)
def test_radius_norm_is_the_ratio_to_the_first_crossing_from_the_center(
    boundary: CircleShape | RectShape | PolygonShape,
    points: list[tuple[float, float]],
    expected: list[float],
) -> None:
    x = _array([p[0] for p in points])
    y = _array([p[1] for p in points])
    assert radius_norm(boundary, x, y).tolist() == pytest.approx(expected, abs=1e-12)


C_POLYGON = PolygonShape(
    vertices=(
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 1.0),
        (1.0, 1.0),
        (1.0, 9.0),
        (10.0, 9.0),
        (10.0, 10.0),
        (0.0, 10.0),
    )
)
"""A C of area 28 open to the right, whose centroid lies in its opening.

Its bottom and top bars (area 10 each, centroids (5, 0.5) and (5, 9.5)) and its
spine (area 8, centroid (0.5, 5)) pool to ((50 + 50 + 4) / 28, 5) = (26/7, 5).
"""


def test_radius_norm_reads_nan_on_a_ray_that_meets_no_edge() -> None:
    assert boundary_center(C_POLYGON) == pytest.approx((26 / 7, 5.0), abs=1e-12)
    ratio = radius_norm(C_POLYGON, _array([8.0, 0.5]), _array([5.0, 5.0]))
    assert np.isnan(ratio[0])
    assert ratio[1] == pytest.approx((26 / 7 - 0.5) / (26 / 7 - 1.0), abs=1e-12)


def test_radius_norm_on_a_ray_through_a_vertex_is_its_fraction_of_the_way() -> None:
    """Points on the line from the centroid through each vertex of convex polygons.

    A point at fraction ``f`` of the way from the centroid to a vertex reads
    ``f``. The centroids are not exactly representable, so the ray meets the
    vertex at an edge parameter a rounding away from 0 or 1.
    """
    rng = np.random.default_rng(0)
    fractions = np.linspace(0.1, 1.5, 50)
    misread: list[tuple[int, int]] = []
    for polygon_index in range(40):
        count = int(rng.integers(5, 12))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, count))
        polygon = PolygonShape(
            vertices=tuple(
                (float(300.0 + 100.0 * np.cos(a)), float(200.0 + 100.0 * np.sin(a)))
                for a in angles
            )
        )
        center_x, center_y = boundary_center(polygon)
        for vertex_index, (vertex_x, vertex_y) in enumerate(polygon.vertices):
            ratio = radius_norm(
                polygon,
                center_x + fractions * (vertex_x - center_x),
                center_y + fractions * (vertex_y - center_y),
            )
            if not np.allclose(ratio, fractions, rtol=0.0, atol=1e-9):
                misread.append((polygon_index, vertex_index))
    assert misread == []


def test_radius_norm_of_a_missing_coordinate_is_nan() -> None:
    for boundary in (CIRCLE, RECT, L_POLYGON):
        assert np.isnan(radius_norm(boundary, _array([NAN]), _array([1.0]))).all()


def test_estimate_recovers_a_circle_from_points_on_it() -> None:
    angles = np.arange(360) * (2 * np.pi / 360)
    x = 300.0 + 100.0 * np.cos(angles)
    y = 200.0 + 100.0 * np.sin(angles)

    estimate = estimate_boundary(x, y, shape="circle", quantile=0.0)

    assert isinstance(estimate.boundary, CircleShape)
    assert estimate.boundary.cx == pytest.approx(300.0, abs=1e-9)
    assert estimate.boundary.cy == pytest.approx(200.0, abs=1e-9)
    assert estimate.boundary.r == pytest.approx(100.0, abs=1e-9)
    assert estimate.rows == 360


def _corners_and_an_outlier() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """50 rows at each corner of (0, 0)-(10, 20) and one row at (1000, 1000).

    Of the 201 sorted values of each axis, the 0.01 quantile sits at position 2
    and the 0.99 quantile at position 198, both inside a run of 100 equal values.
    """
    corners = [(0.0, 0.0), (10.0, 0.0), (0.0, 20.0), (10.0, 20.0)] * 50
    corners.append((1000.0, 1000.0))
    return _array([c[0] for c in corners]), _array([c[1] for c in corners])


def test_a_quantile_trims_an_outlier() -> None:
    x, y = _corners_and_an_outlier()

    trimmed = estimate_boundary(x, y, shape="rect", quantile=0.01)
    circle = estimate_boundary(x, y, shape="circle", quantile=0.01)
    untrimmed = estimate_boundary(x, y, shape="rect", quantile=0.0)

    assert trimmed.boundary == RectShape(x0=0.0, y0=0.0, x1=10.0, y1=20.0)
    assert circle.boundary == CircleShape(cx=5.0, cy=10.0, r=7.5)
    assert untrimmed.boundary == RectShape(x0=0.0, y0=0.0, x1=1000.0, y1=1000.0)
    assert trimmed.rows == 201


def test_a_rect_fit_is_the_box_of_the_finite_positions() -> None:
    x = _array([1.0, 5.0, 4.0, NAN, 100.0])
    y = _array([2.0, 3.0, 9.0, 50.0, NAN])

    estimate = estimate_boundary(x, y, shape="rect", quantile=0.0)

    assert estimate.boundary == RectShape(x0=1.0, y0=2.0, x1=5.0, y1=9.0)
    assert estimate.rows == 3


@pytest.mark.parametrize(
    ("x", "y"),
    [([], []), ([NAN, 1.0], [2.0, NAN])],
    ids=["no-rows", "no-finite-position"],
)
def test_an_estimate_without_a_finite_position_is_refused(
    x: list[float], y: list[float]
) -> None:
    with pytest.raises(ValueError, match="no finite position"):
        estimate_boundary(_array(x), _array(y), shape="circle", quantile=0.0)


@pytest.mark.parametrize(
    ("x", "y", "shape", "message"),
    [
        ([0.0, 1.0, 2.0], [5.0, 5.0, 5.0], "rect", "encloses no rect"),
        ([3.0, 3.0], [4.0, 4.0], "circle", "fits no circle"),
    ],
    ids=["collinear-rect", "one-point-circle"],
)
def test_an_estimate_over_no_extent_is_refused(
    x: list[float], y: list[float], shape: EstimateShape, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_boundary(_array(x), _array(y), shape=shape, quantile=0.0)
