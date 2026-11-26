"""Converter for the CSV export of a zebrafish 3D tracking workflow.

Converts the per-experiment ``.csv`` that the lab's workflow writes to the
``mosaic_world_v1`` schema: meters in a right-handed world frame with +Z up.

The file layout
---------------
The header is ``frame_index`` and then one column per fish, keypoint and axis,
named ``fish{n}_{part}_{axis}`` with ``axis`` one of ``x``, ``y``, ``z``. Each
row is a frame. An empty field is a gap, because the workflow leaves a cell
empty where it has no estimate. The file records neither its unit nor its frame
rate, so the converter takes them from its params: ``meters_per_unit`` (the
workflow's unit is centimeters) and the frame rate through ``conversion_rate``,
like every converter of a file that records none.

The fish numbers and keypoint names come from the header and are not a fixed
list. ``id`` is the fish number and keypoint ``k`` is the ``k``-th part in header
order, written as ``poseX{k}``, ``poseY{k}`` and ``poseZ{k}``. The part names are
not written to the table, so a feature names a keypoint by its index. Every fish
must list the same parts in the same order.

Rows and gaps
-------------
There is one row per fish and frame. A fish-frame whose keypoints are all empty
is dropped, so a gap is an absent row, and a fish-frame with some empty cells
keeps its row with NaN keypoints. ``X``/``Y``/``Z`` is the mean of the keypoints
that have all three axes on that row, and NaN where none does.

The frame
---------
``meters_per_unit`` scales every value, and ``axes`` then names which source axis
becomes world X, Y and Z, each with an optional leading ``-``. It must be a
proper rotation, so a left-handed source is refused and not silently mirrored.
The identity is the workflow's own frame, which is right-handed with +Z up.

Identity continuity is not verified: a file has a fish 1 and a fish 2 column
set, and the workflow removes identity swaps without correcting them, so the
converter declares no answer for ``carries_tracklets``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Final, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    EntryHints,
    TimedTrackConvertParams,
    TrackConverter,
    conversion_rate,
    register_track_converter,
)
from mosaic.core.track_library.csv_columns import read_numeric_csv
from mosaic.core.track_library.helpers import norm_hint
from mosaic.core.track_library.world_frame import (
    WorldAxes,
    animal_columns,
    stacked_table,
    to_world,
)

_FRAME_COLUMN: Final = "frame_index"
_COLUMN_PATTERN: Final = re.compile(r"^fish(\d+)_(.+)_([xyz])$")
_SOURCE_AXES: Final = ("x", "y", "z")

_METERS_PER_UNIT_DESCRIPTION = (
    "Meters in one unit of the file's coordinates. The workflow writes "
    "centimeters, so the default is 0.01."
)
_AXES_DESCRIPTION = (
    "The source axis that becomes world X, Y and Z, each one of x, y or z with an "
    "optional leading minus. ('x', '-z', 'y') takes world Y from minus source z "
    "and world Z from source y. It must be a proper rotation, so a reflection is "
    "refused. The default is the workflow's own frame."
)


class Zebrafish3dWorkflowParams(TimedTrackConvertParams):
    """Everything that determines the converter's output besides the file."""

    meters_per_unit: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False),
        Declared(_METERS_PER_UNIT_DESCRIPTION),
    ] = 0.01
    axes: Annotated[WorldAxes, Declared(_AXES_DESCRIPTION)] = ("x", "y", "z")


class _Header:
    """The parsed header of one file.

    Attributes:
        columns: Every column name, in file order.
        fish: Each fish number and the column of each of its parts' axes, in the
            file's order of fish and of parts. A part's three columns are in the
            order x, y, z.
        parts: The keypoint names every fish lists, in order.
    """

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.columns = columns
        by_fish: dict[int, dict[str, dict[str, str]]] = {}
        seen: set[str] = set()
        for column in columns:
            if column == _FRAME_COLUMN:
                continue
            if column in seen:
                msg = f"{path}: the column {column!r} appears more than once"
                raise ValueError(msg)
            seen.add(column)
            match = _COLUMN_PATTERN.match(column)
            if match is None:
                msg = (
                    f"{path}: the column {column!r} is not {_FRAME_COLUMN} or of the "
                    "form fish{n}_{part}_{axis} with axis x, y or z"
                )
                raise ValueError(msg)
            number, part, axis = int(match.group(1)), match.group(2), match.group(3)
            by_fish.setdefault(number, {}).setdefault(part, {})[axis] = column
        if _FRAME_COLUMN not in columns:
            msg = f"{path}: the header has no {_FRAME_COLUMN} column"
            raise ValueError(msg)
        if not by_fish:
            msg = f"{path}: the header names no fish column"
            raise ValueError(msg)
        for number, parts in by_fish.items():
            for part, axes in parts.items():
                if set(axes) != set(_SOURCE_AXES):
                    missing = ", ".join(
                        f"fish{number}_{part}_{axis}"
                        for axis in _SOURCE_AXES
                        if axis not in axes
                    )
                    msg = f"{path}: the keypoint fish{number}_{part} lacks {missing}"
                    raise ValueError(msg)
        numbers = list(by_fish)
        first = list(by_fish[numbers[0]])
        for number in numbers[1:]:
            listed = list(by_fish[number])
            if listed != first:
                msg = (
                    f"{path}: fish {number} lists the parts {listed}, but fish "
                    f"{numbers[0]} lists {first}. Every fish needs the same parts "
                    "in the same order"
                )
                raise ValueError(msg)
        self.parts = first
        self.fish: dict[int, list[list[str]]] = {
            number: [[parts[part][axis] for axis in _SOURCE_AXES] for part in parts]
            for number, parts in by_fish.items()
        }


def _read_header(path: Path) -> _Header:
    """Parse the first line of the file at *path*.

    Raises:
        ValueError: If the file is empty or its header is outside the format.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        line = handle.readline().rstrip("\r\n")
    if not line:
        msg = f"{path}: the file has no header line"
        raise ValueError(msg)
    return _Header(path, line.split(","))


@register_track_converter
class Zebrafish3dWorkflowConverter(TrackConverter[Zebrafish3dWorkflowParams]):
    """A zebrafish 3D workflow ``.csv`` -> a ``mosaic_world_v1`` table."""

    src_format = "zebrafish_3d_workflow_csv"
    version = "0.1"
    output_schema = "mosaic_world_v1"
    # The workflow removes identity swaps without correcting them, so the file
    # cannot say whether a fish number follows one animal.
    carries_tracklets = None
    enumerable = False
    merges_per_sequence = False
    Params = Zebrafish3dWorkflowParams

    @override
    def convert(
        self, path: Path, params: Zebrafish3dWorkflowParams, hints: EntryHints
    ) -> pd.DataFrame:
        header = _read_header(path)
        rate = conversion_rate(params.fps, params.fallback_fps, hints, str(path))
        cells = read_numeric_csv(path, header.columns, integer_columns=(_FRAME_COLUMN,))
        frames = np.asarray(cells[_FRAME_COLUMN], dtype=np.int64)

        group = norm_hint(hints.group) or ""
        sequence = norm_hint(hints.sequence) or path.stem
        n_parts = len(header.parts)

        per_fish: list[dict[str, npt.NDArray[np.generic]]] = []
        for number, parts in header.fish.items():
            source = np.empty((len(frames), n_parts, 3), dtype=np.float64)
            for keypoint, axis_columns in enumerate(parts):
                for axis, name in enumerate(axis_columns):
                    source[:, keypoint, axis] = cells[name]
            present = ~np.isnan(source).all(axis=(1, 2))
            world = to_world(source[present], params.meters_per_unit, params.axes)
            per_fish.append(
                animal_columns(number, frames[present], world, rate, group, sequence)
            )
        return stacked_table(per_fish)
