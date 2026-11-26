"""The ``mosaic_world_v1`` conventions that a 3D converter applies.

A world-frame table is in meters, in a right-handed frame with +Z up. A 3D
converter reads a file in its own unit and axes, so it declares ``meters_per_unit``
and ``axes`` among its params and records the transform there. ``to_world``
applies them, and ``WorldAxes`` refuses an ``axes`` that would mirror the tracks.

``complete_keypoint_center`` is the body center of a 3D row. It differs from
``pose_columns.keypoint_centroid``, the 2D center, which takes a per-axis
``nanmean``. The 3D center averages only the keypoints with three finite axes, so
a keypoint missing one axis leaves the center unchanged on the other two.

``animal_columns`` and ``stacked_table`` build the table rows of one animal and of
all of them, so the 3D converters write the same columns in the same order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AfterValidator

_SOURCE_AXES: Final = ("x", "y", "z")


def axis_index_and_sign(axis: str) -> tuple[int, int]:
    """Return the source axis index and sign that *axis* names.

    Raises:
        ValueError: If *axis* is not ``x``, ``y`` or ``z`` with an optional
            leading ``-``.
    """
    name = axis[1:] if axis.startswith("-") else axis
    if name not in _SOURCE_AXES:
        msg = f"{axis!r} is not an axis: use x, y or z with an optional leading '-'"
        raise ValueError(msg)
    return _SOURCE_AXES.index(name), -1 if axis.startswith("-") else 1


def _rotation_matrix(axes: tuple[str, str, str]) -> npt.NDArray[np.float64]:
    """The signed permutation that maps source axes to the world axes *axes* name.

    Row ``i`` selects the source axis of world axis ``i``. A leading ``-`` negates
    it. Only its determinant is used.
    """
    matrix = np.zeros((3, 3), dtype=np.float64)
    for world, axis in enumerate(axes):
        source, sign = axis_index_and_sign(axis)
        matrix[world, source] = sign
    return matrix


def check_world_axes(axes: tuple[str, str, str]) -> tuple[str, str, str]:
    """Return *axes* when it is a proper rotation of the source axes.

    Raises:
        ValueError: If an axis is unknown, a source axis is used twice, or the
            signed permutation is a reflection.
    """
    sources = [axis_index_and_sign(axis)[0] for axis in axes]
    if len(set(sources)) != len(sources):
        msg = f"axes {axes} use a source axis more than once"
        raise ValueError(msg)
    if round(float(np.linalg.det(_rotation_matrix(axes)))) < 0:
        msg = (
            f"axes {axes} are a reflection, which would mirror the tracks. "
            "Use a proper rotation: a signed permutation of x, y, z with "
            "determinant +1"
        )
        raise ValueError(msg)
    return axes


WorldAxes = Annotated[tuple[str, str, str], AfterValidator(check_world_axes)]


def to_world(
    source: npt.NDArray[np.float64],
    meters_per_unit: float,
    axes: tuple[str, str, str],
) -> npt.NDArray[np.float64]:
    """Scale source values to meters and permute them onto the world axes.

    A permutation and a sign, not a matrix product, so a NaN on one source axis
    stays on the world axis it maps to.

    Args:
        source: ``(rows, keypoints, 3)`` values in the file's unit and axes.
        meters_per_unit: Meters in one unit of the file's coordinates.
        axes: The source axis that becomes world X, Y and Z, each with an
            optional leading ``-``.

    Returns:
        The values in world meters, in the shape of *source*.
    """
    scaled = source * meters_per_unit
    picks = [axis_index_and_sign(axis) for axis in axes]
    return np.stack([sign * scaled[:, :, index] for index, sign in picks], axis=2)


def complete_keypoint_center(world: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The mean of each row's keypoints that have all three axes.

    Args:
        world: ``(rows, keypoints, 3)`` values in world meters.

    Returns:
        ``(rows, 3)``, NaN for a row with no complete keypoint.
    """
    complete = ~np.isnan(world).any(axis=2)
    counts = complete.sum(axis=1)
    totals = np.where(complete[:, :, None], world, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts[:, None] > 0, totals / counts[:, None], np.nan)


def animal_columns(
    animal_id: int,
    frames: npt.NDArray[np.int64],
    world: npt.NDArray[np.float64],
    rate: float,
    group: str,
    sequence: str,
) -> dict[str, npt.NDArray[np.generic]]:
    """One animal's table columns.

    Args:
        animal_id: The animal's number, written as ``id``.
        frames: The frame of each row.
        world: Those rows' keypoints in world meters, ``(row, keypoint, axis)``.
        rate: The frame rate that times the rows.
        group: The ``group`` cell of every row.
        sequence: The ``sequence`` cell of every row.

    Returns:
        The columns, in the table's order: the base columns, then ``poseX{k}``,
        ``poseY{k}`` and ``poseZ{k}`` for each keypoint ``k``.
    """
    center = complete_keypoint_center(world)

    n_rows = len(frames)
    columns: dict[str, npt.NDArray[np.generic]] = {
        "frame": frames,
        "time": frames.astype(np.float64) / rate,
        "id": np.full(n_rows, animal_id, dtype=np.int64),
        "X": center[:, 0],
        "Y": center[:, 1],
        "Z": center[:, 2],
        "group": np.full(n_rows, group, dtype=object),
        "sequence": np.full(n_rows, sequence, dtype=object),
    }
    for keypoint in range(world.shape[1]):
        for axis, name in enumerate("XYZ"):
            columns[f"pose{name}{keypoint}"] = world[:, keypoint, axis]
    return columns


def stacked_table(
    per_animal: Sequence[dict[str, npt.NDArray[np.generic]]],
) -> pd.DataFrame:
    """The table of every animal's columns, sorted by ``frame`` and then ``id``."""
    columns = {
        name: np.concatenate([animal[name] for animal in per_animal])
        for name in per_animal[0]
    }
    order = np.lexsort((columns["id"], columns["frame"]))
    return pd.DataFrame({name: values[order] for name, values in columns.items()})
