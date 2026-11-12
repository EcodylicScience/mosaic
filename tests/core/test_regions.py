"""Shapes and regions of interest: what validates, what round-trips, what hashes."""

from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter, ValidationError

from mosaic.core.regions import (
    CircleShape,
    PolygonShape,
    RectShape,
    RingShape,
    Roi,
    RoiName,
    Shape,
    bounding_box,
)

SHAPE = TypeAdapter(Shape)
ROI_NAME = TypeAdapter(RoiName)

L_VERTICES = [(1.0, 2.0), (5.0, 2.0), (5.0, 4.0), (3.0, 4.0), (3.0, 7.0), (1.0, 7.0)]
"""An L whose notch sits at (3, 4)-(5, 7), so its box is not a vertex."""

CIRCLE = {"kind": "circle", "cx": 10.5, "cy": 20.0, "r": 5.0}
RING = {"kind": "ring", "cx": 10.5, "cy": 20.0, "r_inner": 2.0, "r_outer": 5.0}
RECT = {"kind": "rect", "x0": 1.5, "y0": 2.0, "x1": 8.0, "y1": 9.25}
POLYGON = {"kind": "polygon", "vertices": L_VERTICES}


def regular_polygon(sides: int) -> list[tuple[float, float]]:
    """A convex polygon of *sides* vertices on the unit circle, which is simple."""
    return [
        (math.cos(2 * math.pi * k / sides), math.sin(2 * math.pi * k / sides))
        for k in range(sides)
    ]


@pytest.mark.parametrize(
    ("payload", "model"),
    [
        (CIRCLE, CircleShape),
        (RING, RingShape),
        (RECT, RectShape),
        (POLYGON, PolygonShape),
    ],
    ids=["circle", "ring", "rect", "polygon"],
)
def test_a_shape_validates_and_round_trips_through_json(
    payload: dict[str, object], model: type[object]
) -> None:
    """The union picks the model by ``kind``, and a dump parses back equal."""
    shape = SHAPE.validate_python(payload)
    assert isinstance(shape, model)
    dumped = shape.model_dump(mode="json")
    assert dumped["kind"] == payload["kind"]
    assert SHAPE.validate_python(dumped) == shape
    assert SHAPE.validate_json(shape.model_dump_json()) == shape


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({**CIRCLE, "r": 0.0}, "greater than 0"),
        ({**CIRCLE, "r": -1.0}, "greater than 0"),
        ({**RING, "r_inner": 5.0}, "r_inner"),
        ({**RING, "r_inner": 6.0}, "r_inner"),
        ({**RING, "r_inner": -0.5}, "greater than or equal to 0"),
        ({**RECT, "x1": 1.5}, "x0 < x1"),
        ({**RECT, "x0": 9.0}, "x0 < x1"),
        ({**RECT, "y1": 2.0}, "y0 < y1"),
        ({**POLYGON, "vertices": [(0.0, 0.0), (1.0, 0.0)]}, "at least 3"),
        ({**POLYGON, "vertices": regular_polygon(257)}, "at most 256"),
        (
            {**POLYGON, "vertices": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]},
            "no area",
        ),
        (
            {**POLYGON, "vertices": [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)]},
            "cross",
        ),
        ({**CIRCLE, "color": "red"}, "color"),
        ({"kind": "hexagon", "cx": 0.0, "cy": 0.0, "r": 1.0}, "kind"),
        ({"cx": 0.0, "cy": 0.0, "r": 1.0}, "kind"),
        ({**CIRCLE, "cx": float("nan")}, "finite"),
        ({**CIRCLE, "r": float("inf")}, "finite"),
    ],
    ids=[
        "circle-zero-radius",
        "circle-negative-radius",
        "ring-equal-radii",
        "ring-inverted-radii",
        "ring-negative-inner",
        "rect-x-collapsed",
        "rect-x-inverted",
        "rect-y-collapsed",
        "polygon-two-vertices",
        "polygon-257-vertices",
        "polygon-zero-area",
        "polygon-bow-tie",
        "unknown-key",
        "unknown-kind",
        "missing-kind",
        "nan-coordinate",
        "infinite-radius",
    ],
)
def test_a_shape_that_is_not_valid_is_refused(
    payload: dict[str, object], fragment: str
) -> None:
    with pytest.raises(ValidationError, match=fragment):
        _ = SHAPE.validate_python(payload)


def test_a_polygon_of_256_vertices_is_accepted() -> None:
    shape = PolygonShape(vertices=tuple(regular_polygon(256)))
    assert len(shape.vertices) == 256


def test_an_l_shaped_concave_polygon_is_accepted() -> None:
    shape = SHAPE.validate_python(POLYGON)
    assert isinstance(shape, PolygonShape)
    assert shape.vertices == tuple(L_VERTICES)


def test_a_polygon_wound_clockwise_is_accepted() -> None:
    shape = PolygonShape(vertices=tuple(reversed(L_VERTICES)))
    assert len(shape.vertices) == len(L_VERTICES)


def test_a_polygon_whose_vertex_touches_a_non_adjacent_edge_is_refused() -> None:
    """A vertex on another edge makes the outline touch itself."""
    touching = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)]
    with pytest.raises(ValidationError, match="cross"):
        _ = PolygonShape(vertices=tuple(touching))


def test_a_shape_is_frozen() -> None:
    shape = CircleShape(cx=0.0, cy=0.0, r=1.0)
    with pytest.raises(ValidationError, match="frozen"):
        shape.r = 2.0  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("name", ["center", "a", "nest_2", "a" * 32])
def test_a_roi_name_that_is_a_lowercase_identifier_is_accepted(name: str) -> None:
    assert ROI_NAME.validate_python(name) == name


@pytest.mark.parametrize(
    "name",
    ["none", "Center", "1a", "a" * 33, "a-b", "", "_a", "a b", "a\n"],
    ids=[
        "none",
        "uppercase",
        "digit-first",
        "33-characters",
        "hyphen",
        "empty",
        "underscore-first",
        "space",
        "trailing-newline",
    ],
)
def test_a_roi_name_that_is_not_a_lowercase_identifier_is_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        _ = ROI_NAME.validate_python(name)


def test_a_roi_defaults_to_an_absolute_placement_at_the_center() -> None:
    roi = Roi.model_validate({"name": "feeder", "shape": RECT})
    assert roi.placement == "absolute"
    assert roi.anchor == "center"
    assert isinstance(roi.shape, RectShape)


def test_a_roi_anchored_on_the_arena_keeps_its_anchor() -> None:
    roi = Roi(
        name="nest",
        shape=RectShape(x0=0.0, y0=0.0, x1=0.4, y1=0.4),
        placement="arena",
        anchor="top_left",
    )
    assert roi.model_dump(mode="json")["anchor"] == "top_left"


def test_a_roi_refuses_an_anchor_other_than_center_when_absolute() -> None:
    with pytest.raises(ValidationError, match="anchor"):
        _ = Roi.model_validate({"name": "feeder", "shape": RECT, "anchor": "top"})


def test_a_roi_refuses_an_unknown_anchor_and_an_unknown_key() -> None:
    with pytest.raises(ValidationError):
        _ = Roi.model_validate(
            {"name": "x", "shape": RECT, "placement": "arena", "anchor": "middle"}
        )
    with pytest.raises(ValidationError, match="speed"):
        _ = Roi.model_validate({"name": "x", "shape": RECT, "speed": 1})


def test_a_roi_refuses_a_name_that_is_not_a_roi_name() -> None:
    with pytest.raises(ValidationError):
        _ = Roi.model_validate({"name": "none", "shape": RECT})


def test_a_roi_round_trips_through_json() -> None:
    roi = Roi(
        name="notch",
        shape=PolygonShape(vertices=tuple(L_VERTICES)),
        placement="arena",
        anchor="bottom_right",
    )
    assert Roi.model_validate(roi.model_dump(mode="json")) == roi
    assert Roi.model_validate_json(roi.model_dump_json()) == roi


def test_a_roi_and_a_tuple_of_rois_are_hashable() -> None:
    """An arena holds its ROIs in a tuple and stays hashable."""
    first = Roi(name="feeder", shape=SHAPE.validate_python(POLYGON))
    again = Roi(name="feeder", shape=SHAPE.validate_python(POLYGON))
    other = Roi(name="nest", shape=SHAPE.validate_python(RING))
    assert hash(first) == hash(again)
    assert hash((first, other)) == hash((again, other))
    seen: set[object] = set()
    for roi in (first, again, other):
        seen.add(roi)
    assert len(seen) == 2


@pytest.mark.parametrize(
    ("payload", "box"),
    [
        (CIRCLE, (5.5, 15.0, 15.5, 25.0)),
        (RING, (5.5, 15.0, 15.5, 25.0)),
        (RECT, (1.5, 2.0, 8.0, 9.25)),
        (POLYGON, (1.0, 2.0, 5.0, 7.0)),
    ],
    ids=["circle", "ring-is-its-outer-circle", "rect", "polygon-vertex-extremes"],
)
def test_bounding_box_is_the_float_box_of_the_shape(
    payload: dict[str, object], box: tuple[float, float, float, float]
) -> None:
    assert bounding_box(SHAPE.validate_python(payload)) == box
