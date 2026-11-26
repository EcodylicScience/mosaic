"""The shared world-frame helpers of the 3D converters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ValidationError

from mosaic.core.track_library.world_frame import (
    WorldAxes,
    axis_index_and_sign,
    check_world_axes,
    complete_keypoint_center,
    to_world,
)


class _Model(BaseModel):
    axes: WorldAxes = ("x", "y", "z")


# --- axes ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "expected"),
    [("x", (0, 1)), ("y", (1, 1)), ("z", (2, 1)), ("-x", (0, -1)), ("-z", (2, -1))],
)
def test_an_axis_names_a_source_index_and_a_sign(
    axis: str, expected: tuple[int, int]
) -> None:
    assert axis_index_and_sign(axis) == expected


@pytest.mark.parametrize("axis", ["w", "--z", "", "-", "X"])
def test_an_unknown_axis_is_refused(axis: str) -> None:
    with pytest.raises(ValueError, match="is not an axis"):
        _ = axis_index_and_sign(axis)


@pytest.mark.parametrize(
    "axes",
    [
        ("x", "z", "y"),
        ("-x", "y", "z"),
        ("-x", "-y", "-z"),
    ],
)
def test_a_reflection_is_refused(axes: tuple[str, str, str]) -> None:
    with pytest.raises(ValueError, match="reflection"):
        _ = check_world_axes(axes)
    with pytest.raises(ValidationError, match="reflection"):
        _ = _Model(axes=axes)


@pytest.mark.parametrize("axes", [("x", "-x", "y"), ("x", "y", "y"), ("-z", "x", "z")])
def test_a_repeated_axis_is_refused(axes: tuple[str, str, str]) -> None:
    with pytest.raises(ValueError, match="more than once"):
        _ = check_world_axes(axes)
    with pytest.raises(ValidationError, match="more than once"):
        _ = _Model(axes=axes)


@pytest.mark.parametrize("axes", [("x", "y", "w"), ("x", "y", "--z"), ("x", "y", "")])
def test_an_unknown_axis_is_refused_in_a_triple(axes: tuple[str, str, str]) -> None:
    with pytest.raises(ValidationError, match="axis"):
        _ = _Model(axes=axes)


@pytest.mark.parametrize(
    "axes",
    [
        ("x", "y", "z"),
        ("x", "-z", "y"),
        ("-x", "-y", "z"),
        ("y", "-x", "z"),
        ("y", "x", "-z"),
    ],
)
def test_a_proper_rotation_is_accepted(axes: tuple[str, str, str]) -> None:
    assert check_world_axes(axes) == axes
    assert _Model(axes=axes).axes == axes


# --- to_world -----------------------------------------------------------------


def _source() -> npt.NDArray[np.float64]:
    return np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )


def test_to_world_scales_by_the_unit() -> None:
    world = to_world(_source(), 0.5, ("x", "y", "z"))

    np.testing.assert_allclose(world, _source() * 0.5)


def test_to_world_permutes_and_negates() -> None:
    source = _source()

    world = to_world(source, 1.0, ("x", "-z", "y"))

    np.testing.assert_allclose(world[:, :, 0], source[:, :, 0])
    np.testing.assert_allclose(world[:, :, 1], -source[:, :, 2])
    np.testing.assert_allclose(world[:, :, 2], source[:, :, 1])


def test_to_world_keeps_a_nan_on_the_axis_it_maps_to() -> None:
    source = _source()
    source[0, 1, 2] = np.nan

    world = to_world(source, 0.01, ("x", "-z", "y"))

    assert np.isnan(world[0, 1, 1])
    assert np.isnan(world).sum() == 1


# --- complete_keypoint_center -------------------------------------------------


def test_the_center_leaves_out_a_keypoint_with_one_missing_axis() -> None:
    world = np.array([[[1.0, 2.0, 3.0], [np.nan, 10.0, 10.0], [3.0, 4.0, 5.0]]])

    center = complete_keypoint_center(world)

    np.testing.assert_allclose(center, [[2.0, 3.0, 4.0]])


def test_the_center_is_nan_where_no_keypoint_is_complete() -> None:
    world = np.array(
        [
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[np.nan, 2.0, 3.0], [1.0, np.nan, 3.0]],
        ]
    )

    center = complete_keypoint_center(world)

    np.testing.assert_allclose(center[0], [1.0, 2.0, 3.0])
    assert np.isnan(center[1]).all()


def test_the_center_of_no_rows_is_empty() -> None:
    assert complete_keypoint_center(np.empty((0, 3, 3))).shape == (0, 3)
