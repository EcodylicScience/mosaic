"""The geometry of shapes and regions of interest, over numpy arrays of positions.

Every function that reads positions takes arrays ``x`` and ``y`` of one shape and
returns an array of that shape. A row with a missing coordinate reads NaN. The
shapes are those of :mod:`mosaic.core.regions`, whose
:func:`~mosaic.core.regions.bounding_box` gives the box that anchors and boundary
units read. This module imports numpy, the standard library and
:mod:`mosaic.core.regions` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from mosaic.core.regions import (
    Anchor,
    CircleShape,
    PolygonShape,
    RectShape,
    RingShape,
    Roi,
    Shape,
    bounding_box,
)

__all__ = [
    "Boundary",
    "BoundaryEstimate",
    "EstimateShape",
    "anchor_point",
    "boundary_center",
    "contains",
    "estimate_boundary",
    "inside",
    "radius_norm",
    "resolve_roi",
    "signed_distance",
]

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

Boundary: TypeAlias = CircleShape | PolygonShape | RectShape
"""The shapes an arena's boundary can take."""

EstimateShape: TypeAlias = Literal["circle", "rect"]
"""The shape :func:`estimate_boundary` fits."""

_POLYGON_BLOCK_ROWS = 4096
"""Rows of positions a polygon compares against all its edges at once.

A block holds ``rows x edges`` floats per intermediate, so 4096 rows of a
256-vertex polygon stay near 8 MiB each.
"""


_EDGE_PARAMETER_TOLERANCE = 1e-9
"""How far past either end of an edge, as a fraction of the edge, a ray still meets it.

A ray through a vertex meets both of its edges at a parameter of 0 or 1 that
rounding can move to either side, which would miss the vertex.
"""


def _positions(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape:
        message = f"x and y differ in shape: {xs.shape} and {ys.shape}"
        raise ValueError(message)
    return xs, ys


def _edges(
    shape: PolygonShape,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The start point and the vector of each edge, the last closing the outline."""
    vertices = np.array(shape.vertices, dtype=np.float64)
    following = np.roll(vertices, -1, axis=0)
    return (
        vertices[:, 0],
        vertices[:, 1],
        following[:, 0] - vertices[:, 0],
        following[:, 1] - vertices[:, 1],
    )


def _polygon_signed_distance(
    shape: PolygonShape, x: FloatArray, y: FloatArray
) -> FloatArray:
    """The least distance to an edge, negated inside by the even-odd count."""
    ax, ay, ex, ey = _edges(shape)
    length2 = ex * ex + ey * ey
    safe_length2 = np.where(length2 > 0.0, length2, 1.0)
    safe_ey = np.where(ey != 0.0, ey, 1.0)
    out = np.empty(x.shape, dtype=np.float64)
    for start in range(0, x.size, _POLYGON_BLOCK_ROWS):
        stop = start + _POLYGON_BLOCK_ROWS
        px = x[start:stop, np.newaxis]
        py = y[start:stop, np.newaxis]
        wx = px - ax
        wy = py - ay
        t = np.clip((wx * ex + wy * ey) / safe_length2, 0.0, 1.0)
        dx = wx - t * ex
        dy = wy - t * ey
        distance = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
        straddles = (ay > py) != (ay + ey > py)
        crossing_x = ax + (py - ay) * ex / safe_ey
        crossings = np.count_nonzero(straddles & (px < crossing_x), axis=1)
        out[start:stop] = np.where(crossings % 2 == 1, -distance, distance)
    return out


def signed_distance(shape: Shape, x: FloatArray, y: FloatArray) -> FloatArray:
    """The distance from each position to *shape*'s boundary, in its units.

    Negative inside, zero on the boundary, positive outside. A ring's hole is
    outside. A polygon's sign follows the even-odd crossing count.

    Args:
        shape: The shape, in the same coordinates as the positions.
        x: The positions' x coordinates.
        y: The positions' y coordinates, of the same shape as ``x``.

    Returns:
        One float64 distance per position, NaN where a coordinate is NaN.
    """
    xs, ys = _positions(x, y)
    match shape:
        case CircleShape():
            out = np.hypot(xs - shape.cx, ys - shape.cy) - shape.r
        case RingShape():
            radius = np.hypot(xs - shape.cx, ys - shape.cy)
            out = np.maximum(shape.r_inner - radius, radius - shape.r_outer)
        case RectShape():
            qx = np.abs(xs - (shape.x0 + shape.x1) / 2.0) - (shape.x1 - shape.x0) / 2.0
            qy = np.abs(ys - (shape.y0 + shape.y1) / 2.0) - (shape.y1 - shape.y0) / 2.0
            outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            out = outside + np.minimum(np.maximum(qx, qy), 0.0)
        case PolygonShape():
            out = _polygon_signed_distance(shape, xs.ravel(), ys.ravel()).reshape(
                xs.shape
            )
    return np.where(np.isfinite(xs) & np.isfinite(ys), out, np.nan)


def inside(distance: FloatArray) -> BoolArray:
    """Whether each signed *distance* is that of a position in its shape.

    The boundary counts as inside, so a distance of zero reads ``True``. A NaN
    distance reads ``False``.
    """
    return distance <= 0.0


def contains(shape: Shape, x: FloatArray, y: FloatArray) -> BoolArray:
    """Whether each position lies in *shape*, its boundary included.

    A position with a NaN coordinate reads ``False``. Its NaN
    :func:`signed_distance` tells it from a position seen outside.
    """
    return inside(signed_distance(shape, x, y))


def boundary_center(shape: Shape) -> tuple[float, float]:
    """A circle's or ring's center, a rect's middle, or a polygon's area centroid."""
    match shape:
        case CircleShape() | RingShape():
            return (shape.cx, shape.cy)
        case RectShape():
            return ((shape.x0 + shape.x1) / 2.0, (shape.y0 + shape.y1) / 2.0)
        case PolygonShape():
            vertices = shape.vertices
            count = len(vertices)
            twice_area = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for k in range(count):
                x0, y0 = vertices[k]
                x1, y1 = vertices[(k + 1) % count]
                cross = x0 * y1 - x1 * y0
                twice_area += cross
                sum_x += (x0 + x1) * cross
                sum_y += (y0 + y1) * cross
            return (sum_x / (3.0 * twice_area), sum_y / (3.0 * twice_area))


def anchor_point(shape: Shape, anchor: Anchor) -> tuple[float, float]:
    """The point of *shape* that *anchor* names.

    ``center`` is :func:`boundary_center`. The others are points of the shape's
    bounding box, in image axes where ``y`` grows down, so ``top`` is the middle
    of the box's least-``y`` side.
    """
    if anchor == "center":
        return boundary_center(shape)
    x_min, y_min, x_max, y_max = bounding_box(shape)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    match anchor:
        case "top":
            return (x_mid, y_min)
        case "bottom":
            return (x_mid, y_max)
        case "left":
            return (x_min, y_mid)
        case "right":
            return (x_max, y_mid)
        case "top_left":
            return (x_min, y_min)
        case "top_right":
            return (x_max, y_min)
        case "bottom_left":
            return (x_min, y_max)
        case "bottom_right":
            return (x_max, y_max)


def resolve_roi(roi: Roi, boundary: Boundary) -> Shape:
    """The absolute shape of *roi* in the arena that *boundary* encloses.

    An ``absolute`` region is returned as given. An ``arena`` region's offsets
    are scaled by the boundary's bounding box, an ``x`` by its half-width, a ``y``
    by its half-height and a radius by half its shorter side, and moved to the
    region's anchor. For a circle boundary each unit is its radius.

    Args:
        roi: The region of interest.
        boundary: The arena's boundary, in source pixels.

    Returns:
        The region's shape in source pixels.
    """
    if roi.placement == "absolute":
        return roi.shape
    x_min, y_min, x_max, y_max = bounding_box(boundary)
    half_width = (x_max - x_min) / 2.0
    half_height = (y_max - y_min) / 2.0
    radius_unit = min(half_width, half_height)
    origin_x, origin_y = anchor_point(boundary, roi.anchor)

    def place_x(offset: float) -> float:
        return origin_x + offset * half_width

    def place_y(offset: float) -> float:
        return origin_y + offset * half_height

    shape = roi.shape
    match shape:
        case CircleShape():
            return CircleShape(
                cx=place_x(shape.cx), cy=place_y(shape.cy), r=shape.r * radius_unit
            )
        case RingShape():
            return RingShape(
                cx=place_x(shape.cx),
                cy=place_y(shape.cy),
                r_inner=shape.r_inner * radius_unit,
                r_outer=shape.r_outer * radius_unit,
            )
        case RectShape():
            return RectShape(
                x0=place_x(shape.x0),
                y0=place_y(shape.y0),
                x1=place_x(shape.x1),
                y1=place_y(shape.y1),
            )
        case PolygonShape():
            return PolygonShape(
                vertices=tuple((place_x(vx), place_y(vy)) for vx, vy in shape.vertices)
            )


def _polygon_radius_norm(
    shape: PolygonShape, x: FloatArray, y: FloatArray
) -> FloatArray:
    """The ratio of each position's center distance to the ray's first crossing.

    The ray from the centroid through a position meets edge ``a + s e`` at
    ``center + t (p - center)``, and the ratio is ``1 / t`` for the least ``t >
    0``. A ray that meets no edge, from a centroid outside the polygon, reads
    NaN.
    """
    center_x, center_y = boundary_center(shape)
    ax, ay, ex, ey = _edges(shape)
    wx = ax - center_x
    wy = ay - center_y
    out = np.empty(x.shape, dtype=np.float64)
    for start in range(0, x.size, _POLYGON_BLOCK_ROWS):
        stop = start + _POLYGON_BLOCK_ROWS
        dx = x[start:stop, np.newaxis] - center_x
        dy = y[start:stop, np.newaxis] - center_y
        denominator = dx * ey - dy * ex
        safe = np.where(denominator != 0.0, denominator, 1.0)
        t = (wx * ey - wy * ex) / safe
        s = (wx * dy - wy * dx) / safe
        valid = (
            (denominator != 0.0)
            & (t > 0.0)
            & (s >= -_EDGE_PARAMETER_TOLERANCE)
            & (s <= 1.0 + _EDGE_PARAMETER_TOLERANCE)
        )
        first = np.min(np.where(valid, t, np.inf), axis=1)
        at_center = (dx[:, 0] == 0.0) & (dy[:, 0] == 0.0)
        ratio = np.where(np.isfinite(first), 1.0 / first, np.nan)
        out[start:stop] = np.where(at_center, 0.0, ratio)
    return out


def radius_norm(boundary: Boundary, x: FloatArray, y: FloatArray) -> FloatArray:
    """Each position's distance from the boundary's center, as a fraction of the wall's.

    The wall's distance is to the first point where the ray from the center
    through the position crosses the boundary. That gives ``|p - c| / r`` for a
    circle and ``max(|dx| / half-width, |dy| / half-height)`` for a rect: 0 at
    the center, 1 on the wall, above 1 past it. On a concave polygon the first
    crossing is the nearest, so a position past a notch reads above 1.

    Args:
        boundary: The arena's boundary.
        x: The positions' x coordinates.
        y: The positions' y coordinates, of the same shape as ``x``.

    Returns:
        One float64 ratio per position. NaN where a coordinate is NaN, and on a
        polygon whose centroid lies outside it, where the ray meets no edge.
    """
    xs, ys = _positions(x, y)
    match boundary:
        case CircleShape():
            out = np.hypot(xs - boundary.cx, ys - boundary.cy) / boundary.r
        case RectShape():
            center_x, center_y = boundary_center(boundary)
            half_width = (boundary.x1 - boundary.x0) / 2.0
            half_height = (boundary.y1 - boundary.y0) / 2.0
            out = np.maximum(
                np.abs(xs - center_x) / half_width, np.abs(ys - center_y) / half_height
            )
        case PolygonShape():
            out = _polygon_radius_norm(boundary, xs.ravel(), ys.ravel()).reshape(
                xs.shape
            )
    return np.where(np.isfinite(xs) & np.isfinite(ys), out, np.nan)


@dataclass(frozen=True, slots=True)
class BoundaryEstimate:
    """A boundary fit to positions.

    Attributes:
        boundary: The fitted circle or rect.
        rows: The number of finite positions the fit read.
    """

    boundary: CircleShape | RectShape
    rows: int


def estimate_boundary(
    x: FloatArray, y: FloatArray, *, shape: EstimateShape, quantile: float
) -> BoundaryEstimate:
    """Fit a boundary to the positions with both coordinates finite.

    The fit spans the ``quantile`` and ``1 - quantile`` quantiles of each axis.
    A rect is that box. A circle is centered on the box's center, with radius
    the mean of its half-width and half-height.

    Args:
        x: The positions' x coordinates.
        y: The positions' y coordinates, of the same shape as ``x``.
        shape: The shape to fit.
        quantile: The share of positions trimmed at each end of each axis, from
            0 up to but excluding 0.5.

    Returns:
        The fitted boundary and the number of positions it read.

    Raises:
        ValueError: When ``quantile`` is out of range, no position is finite, or
            the positions span no extent the shape needs.
    """
    if not 0.0 <= quantile < 0.5:
        message = f"quantile must lie in [0, 0.5), got {quantile}"
        raise ValueError(message)
    xs, ys = _positions(x, y)
    finite = np.isfinite(xs) & np.isfinite(ys)
    rows = int(np.count_nonzero(finite))
    if rows == 0:
        message = "no finite position to fit a boundary to"
        raise ValueError(message)
    levels = [quantile, 1.0 - quantile]
    x_min, x_max = (float(v) for v in np.quantile(xs[finite], levels))
    y_min, y_max = (float(v) for v in np.quantile(ys[finite], levels))
    if shape == "rect":
        if x_min >= x_max or y_min >= y_max:
            message = (
                f"the positions span ({x_min}, {y_min}) to ({x_max}, {y_max}), "
                f"which encloses no rect"
            )
            raise ValueError(message)
        return BoundaryEstimate(
            boundary=RectShape(x0=x_min, y0=y_min, x1=x_max, y1=y_max), rows=rows
        )
    radius = ((x_max - x_min) / 2.0 + (y_max - y_min) / 2.0) / 2.0
    if radius <= 0.0:
        message = f"the positions all sit at ({x_min}, {y_min}), which fits no circle"
        raise ValueError(message)
    return BoundaryEstimate(
        boundary=CircleShape(
            cx=(x_min + x_max) / 2.0, cy=(y_min + y_max) / 2.0, r=radius
        ),
        rows=rows,
    )
