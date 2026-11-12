"""Shapes and regions of interest, in plain pydantic models.

A shape is a circle, a ring, a rectangle or a polygon, and a region of interest
(:class:`Roi`) is a named shape placed either in source pixels or relative to an
arena's boundary. The models are frozen and hold tuples, so a shape and an arena
holding one stay hashable. This module imports pydantic, the standard library,
:mod:`mosaic.core.params` and :mod:`mosaic.core.strict_model` only, so the arena
layout and the agent tool contract can declare a field over them. The numpy
geometry lives in :mod:`mosaic.core.region_geometry`.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from mosaic.core.params import Declared, DeclaredModel, has_fields

__all__ = [
    "NO_REGION_NAME",
    "Anchor",
    "CircleShape",
    "Placement",
    "PolygonShape",
    "RectShape",
    "RingShape",
    "Roi",
    "RoiName",
    "Shape",
    "bounding_box",
]

Anchor: TypeAlias = Literal[
    "center",
    "top",
    "bottom",
    "left",
    "right",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]
"""The point of an arena's boundary that an arena-relative shape is offset from.

``center`` is the boundary's center. The others are points of its bounding box,
in image axes: ``x`` grows right and ``y`` grows down.
"""

Placement: TypeAlias = Literal["absolute", "arena"]
"""What a region's coordinates are measured from.

``absolute`` coordinates are source pixels. ``arena`` coordinates are offsets
from an anchor of the arena's boundary, in boundary units.
"""

Coordinate: TypeAlias = Annotated[float, Field(allow_inf_nan=False)]
"""A finite coordinate. A shape may sit between pixels."""

MIN_VERTICES = 3
MAX_VERTICES = 256

NO_REGION_NAME = "none"
"""The value of ``roi`` outside every region, which no region may be named."""


def _refuse_the_name_of_no_region(name: str) -> str:
    if name == NO_REGION_NAME:
        message = (
            f"{NO_REGION_NAME!r} is the value of roi outside every region, so no "
            f"region can be named {NO_REGION_NAME!r}"
        )
        raise ValueError(message)
    return name


RoiName: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$"),
    AfterValidator(_refuse_the_name_of_no_region),
]
"""A region's name: a lowercase identifier of at most 32 characters, not ``none``.

The name becomes part of a column name (``in_<name>``, ``dist_<name>``) and a
value of ``roi``, so a ``filter_expr`` can name it unquoted.
"""


class CircleShape(DeclaredModel):
    """A disc."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Annotated[Literal["circle"], Declared("The shape's kind.")] = "circle"
    cx: Annotated[Coordinate, Declared("The x coordinate of the center.")]
    cy: Annotated[Coordinate, Declared("The y coordinate of the center.")]
    r: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False),
        Declared("The radius, above zero."),
    ]


class RingShape(DeclaredModel):
    """The band between two concentric circles."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Annotated[Literal["ring"], Declared("The shape's kind.")] = "ring"
    cx: Annotated[Coordinate, Declared("The x coordinate of the center.")]
    cy: Annotated[Coordinate, Declared("The y coordinate of the center.")]
    r_inner: Annotated[
        float,
        Field(ge=0.0, allow_inf_nan=False),
        Declared("The radius of the hole, below the outer radius."),
    ]
    r_outer: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False),
        Declared("The radius of the outer edge."),
    ]

    @model_validator(mode="after")
    def _check_radii(self) -> Self:
        """Refuse a ring whose hole is not smaller than its outer edge."""
        if not has_fields(self, "r_inner", "r_outer"):
            return self
        if self.r_inner >= self.r_outer:
            message = (
                f"a ring needs r_inner < r_outer, got r_inner={self.r_inner} and "
                f"r_outer={self.r_outer}"
            )
            raise ValueError(message)
        return self


class RectShape(DeclaredModel):
    """An axis-aligned rectangle of float coordinates.

    Not the integer, half-open ``Rect`` of an arena's region.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Annotated[Literal["rect"], Declared("The shape's kind.")] = "rect"
    x0: Annotated[Coordinate, Declared("The left edge, below the right edge.")]
    y0: Annotated[Coordinate, Declared("The top edge, above the bottom edge.")]
    x1: Annotated[Coordinate, Declared("The right edge.")]
    y1: Annotated[Coordinate, Declared("The bottom edge.")]

    @model_validator(mode="after")
    def _check_extent(self) -> Self:
        """Refuse a rectangle without area."""
        if not has_fields(self, "x0", "y0", "x1", "y1"):
            return self
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            message = (
                f"a rect needs x0 < x1 and y0 < y1, got ({self.x0}, {self.y0}) to "
                f"({self.x1}, {self.y1})"
            )
            raise ValueError(message)
        return self


type _Point = tuple[float, float]


def _orientation(a: _Point, b: _Point, c: _Point) -> float:
    """The cross product of ``b - a`` and ``c - a``: positive for a left turn."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _within_box(a: _Point, b: _Point, p: _Point) -> bool:
    """Whether *p*, collinear with ``a`` and ``b``, lies between them."""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[
        1
    ] <= max(a[1], b[1])


def _segments_touch(a: _Point, b: _Point, c: _Point, d: _Point) -> bool:
    """Whether the closed segments ``ab`` and ``cd`` share a point."""
    d1 = _orientation(c, d, a)
    d2 = _orientation(c, d, b)
    d3 = _orientation(a, b, c)
    d4 = _orientation(a, b, d)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        (d1 == 0 and _within_box(c, d, a))
        or (d2 == 0 and _within_box(c, d, b))
        or (d3 == 0 and _within_box(a, b, c))
        or (d4 == 0 and _within_box(a, b, d))
    )


def _first_crossing(vertices: tuple[_Point, ...]) -> tuple[int, int] | None:
    """The first pair of non-adjacent edges that share a point, or ``None``.

    Edge ``i`` runs from vertex ``i`` to the next, and the last edge closes the
    outline. At most 256 vertices give at most 32,640 pairs.
    """
    count = len(vertices)
    for i in range(count):
        a, b = vertices[i], vertices[(i + 1) % count]
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            if _segments_touch(a, b, vertices[j], vertices[(j + 1) % count]):
                return i, j
    return None


def _shoelace_area(vertices: tuple[_Point, ...]) -> float:
    """The signed area of the outline."""
    count = len(vertices)
    twice = sum(
        vertices[k][0] * vertices[(k + 1) % count][1]
        - vertices[(k + 1) % count][0] * vertices[k][1]
        for k in range(count)
    )
    return twice / 2.0


class PolygonShape(DeclaredModel):
    """A simple polygon, convex or concave, in either winding."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Annotated[Literal["polygon"], Declared("The shape's kind.")] = "polygon"
    vertices: Annotated[
        tuple[tuple[Coordinate, Coordinate], ...],
        Field(min_length=MIN_VERTICES, max_length=MAX_VERTICES),
        Declared(
            "The outline's vertices as (x, y) pairs, 3 to 256 of them. The last "
            "vertex joins the first. The outline has an area and no two "
            "non-adjacent edges touch."
        ),
    ]

    @model_validator(mode="after")
    def _check_outline(self) -> Self:
        """Refuse an outline whose edges touch or that encloses no area."""
        if not has_fields(self, "vertices"):
            return self
        crossing = _first_crossing(self.vertices)
        if crossing is not None:
            message = (
                f"a polygon's edges may not cross: edges {crossing[0]} and "
                f"{crossing[1]} share a point"
            )
            raise ValueError(message)
        if _shoelace_area(self.vertices) == 0.0:
            message = "a polygon with no area is not a region"
            raise ValueError(message)
        return self


Shape: TypeAlias = Annotated[
    CircleShape | RingShape | RectShape | PolygonShape,
    Field(discriminator="kind"),
]
"""A region's geometry, told apart by ``kind``."""


class Roi(DeclaredModel):
    """A named region of interest.

    ``absolute`` places the shape in source pixels, and its anchor stays
    ``center``. ``arena`` reads the shape's coordinates as offsets from an anchor
    of the arena's boundary, so one definition fits every arena wherever it sits
    in the frame.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[
        RoiName,
        Declared(
            "The region's name, a lowercase identifier of at most 32 characters "
            "that is not none. It names the in_<name> and dist_<name> columns "
            "and a value of roi."
        ),
    ]
    shape: Annotated[Shape, Declared("The region's geometry.")]
    placement: Annotated[
        Placement,
        Declared(
            "What the shape's coordinates are measured from: source pixels "
            "(absolute) or offsets from an anchor of the arena's boundary, in "
            "boundary units (arena)."
        ),
    ] = "absolute"
    anchor: Annotated[
        Anchor,
        Declared(
            "The point of the arena's boundary that an arena-placed shape is "
            "offset from. Left at center for an absolute shape."
        ),
    ] = "center"

    @model_validator(mode="after")
    def _check_anchor(self) -> Self:
        """Refuse an anchor on a shape placed in source pixels."""
        if not has_fields(self, "placement", "anchor"):
            return self
        if self.placement == "absolute" and self.anchor != "center":
            message = (
                f"an absolute region takes no anchor, got anchor={self.anchor!r}. "
                f"Set placement to 'arena' or leave the anchor at 'center'"
            )
            raise ValueError(message)
        return self


def bounding_box(shape: Shape) -> tuple[float, float, float, float]:
    """The axis-aligned box ``(x_min, y_min, x_max, y_max)`` of *shape*.

    A ring's box is its outer circle's, and a polygon's is its vertices' extremes.
    """
    match shape:
        case CircleShape():
            return (
                shape.cx - shape.r,
                shape.cy - shape.r,
                shape.cx + shape.r,
                shape.cy + shape.r,
            )
        case RingShape():
            return (
                shape.cx - shape.r_outer,
                shape.cy - shape.r_outer,
                shape.cx + shape.r_outer,
                shape.cy + shape.r_outer,
            )
        case RectShape():
            return (shape.x0, shape.y0, shape.x1, shape.y1)
        case PolygonShape():
            xs = [x for x, _ in shape.vertices]
            ys = [y for _, y in shape.vertices]
            return (min(xs), min(ys), max(xs), max(ys))
