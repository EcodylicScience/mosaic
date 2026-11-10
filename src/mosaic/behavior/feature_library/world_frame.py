"""The yaw-only body frame and the 3D point reading of the world-frame features.

``pose-egocentric`` and ``pair-world`` both turn a keypoint pair into a body
frame that rotates about world up only, and both read a keypoint as one 3D
point. One definition keeps the two features' frames, and their notion of a
vertical pair, the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from .derivatives import BoolArray, FloatArray, keypoint_positions

__all__ = [
    "VERTICAL_TOLERANCE",
    "BodyAxes",
    "FrameKind",
    "KeypointRef",
    "body_axes",
    "world_points",
]

KeypointRef = int | str
"""A keypoint named by its position or, given ``keypoint_names``, by its name."""

VERTICAL_TOLERANCE: Final = 1e-6
"""The smallest horizontal share of the frame's pair vector that still fixes a yaw.

Below this the vector points straight up or down, its horizontal part is
rounding error, and a direction normalized from it would be noise reported as a
heading. Relative to the vector's own length, so it does not depend on the
table's length unit.
"""

FrameKind = Literal["lateral", "heading"]
"""Which keypoint pair spans the body frame's horizontal axes."""


@dataclass(frozen=True, slots=True)
class BodyAxes:
    """Each row's forward and left directions in the world XY plane.

    Attributes:
        forward_x: The world x component of the body's +X.
        forward_y: The world y component of the body's +X.
        left_x: The world x component of the body's +Y.
        left_y: The world y component of the body's +Y.
        oriented: Whether the row's pair vector fixes a yaw. The components are
            NaN where it does not.
    """

    forward_x: FloatArray
    forward_y: FloatArray
    left_x: FloatArray
    left_y: FloatArray
    oriented: BoolArray


def world_points(frame: pd.DataFrame, columns: tuple[str, ...]) -> FloatArray:
    """One point's ``(n_rows, 3)`` world coordinates, as float64.

    A row missing any of the three axes is missing all of them. A point with no
    height is not a point on the floor, and keeping its other two axes would
    give it a forward and a leftward coordinate with a NaN height beside them.

    Args:
        frame: The table.
        columns: The point's x, y and z columns.
    """
    points = keypoint_positions(frame, columns, np.arange(len(frame), dtype=np.int64))
    points[np.isnan(points).any(axis=1)] = np.nan
    return points


def body_axes(vector: FloatArray, kind: FrameKind) -> BodyAxes:
    """The body's horizontal axes from each row's pair vector.

    Args:
        vector: ``(n_rows, 3)``, left minus right for a lateral pair and front
            minus rear for a heading pair.
        kind: Which of the two *vector* is.

    Returns:
        The axes, NaN on a row where the vector is missing, zero, or no more
        than :data:`VERTICAL_TOLERANCE` horizontal.
    """
    horizontal = np.hypot(vector[:, 0], vector[:, 1])
    length = np.linalg.norm(vector, axis=1)
    # NaN compares false, so a row missing either keypoint is undefined here too.
    oriented: BoolArray = horizontal > VERTICAL_TOLERANCE * length
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_x: FloatArray = np.where(oriented, vector[:, 0] / horizontal, np.nan)
        unit_y: FloatArray = np.where(oriented, vector[:, 1] / horizontal, np.nan)
    if kind == "lateral":
        # Forward is Y x Z = (left_y, -left_x, 0).
        return BodyAxes(
            forward_x=unit_y,
            forward_y=-unit_x,
            left_x=unit_x,
            left_y=unit_y,
            oriented=oriented,
        )
    # Left is Z x X = (-forward_y, forward_x, 0).
    return BodyAxes(
        forward_x=unit_x,
        forward_y=unit_y,
        left_x=-unit_y,
        left_y=unit_x,
        oriented=oriented,
    )
