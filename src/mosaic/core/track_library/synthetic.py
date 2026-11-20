"""Synthetic track files -> simulated tables, one format per dimension.

A synthetic track file (:mod:`~mosaic.core.track_library.synthetic_spec`) declares
every parameter of one generated sequence: the arena, the motion, the interaction
zones, the body, the dropouts, the frame rate and the seed. The file's checksum
identifies the sequence, and a tracks variant's ``source_md5`` covers all of it.
The converter's params declare the id policy alone, ``relabel_after_gap``, because
``TrackConverter.tracklets(params)`` answers whether ids follow animals from the
params and cannot read the file. A tracks variant then names the policy. The file
records its rate. The params derive from ``TrackConvertParams`` and have no
``fps`` field, as the rule for a file that records its own time has it
(``core/CLAUDE.md``, "Track converters and schemas").

**Each dimension is its own format.** A converter declares one ``output_schema``
for all of its files (``trex.py``). ``synthetic`` writes ``mosaic_v1``, in pixels
and two dimensions, and ``synthetic_3d`` writes ``mosaic_world_v1``, in meters and
three. Each refuses a file of the other dimension and names the format that reads
it.

**3D positions are generated in world meters.** The arena's lengths are meters,
and ``X``, ``Y``, ``Z`` and ``poseZ{k}`` are written as simulated, with no
transform applied and none recorded.

The zone model follows the published description of Couzin, Krause, James, Ruxton
and Franks (2002), "Collective memory and spatial sorting in animal groups",
J. Theor. Biol. 218:1-11.

The simulation is :mod:`~mosaic.core.track_library.synthetic_motion`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, override

import numpy as np
import pandas as pd

from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint
from mosaic.core.track_library.synthetic_motion import SyntheticColumns, simulate
from mosaic.core.track_library.synthetic_spec import read_synthetic_spec

__all__ = ["Synthetic3dConverter", "SyntheticConverter", "SyntheticParams"]

_RELABEL_DESCRIPTION = (
    "Give an individual that returns after a missing frame the next unused id, as "
    "a tracker that loses and re-finds an animal does. Off, an individual keeps "
    "one id throughout, and the table's ids follow animals."
)


class SyntheticParams(TrackConvertParams):
    """The id policy of a synthetic conversion.

    Every other parameter of the simulation is in the file.
    """

    relabel_after_gap: Annotated[bool, Declared(_RELABEL_DESCRIPTION)] = False


def _table(
    columns: SyntheticColumns, *, rate: float, group: str, sequence: str
) -> pd.DataFrame:
    """The table of *columns*, timed at *rate* and labeled, sharing the arrays.

    The order is ``frame``, ``time``, ``id``, the position axes, ``group``,
    ``sequence`` and then the keypoint columns.
    """
    simulated = dict(columns.named_columns())
    rows = len(columns.frame)
    mapping: dict[str, object] = {}
    for name, array in simulated.items():
        if name.startswith("pose") and "group" not in mapping:
            mapping["group"] = np.full(rows, group, dtype=object)
            mapping["sequence"] = np.full(rows, sequence, dtype=object)
        mapping[name] = array
        if name == "frame":
            mapping["time"] = array / rate
    if "group" not in mapping:
        mapping["group"] = np.full(rows, group, dtype=object)
        mapping["sequence"] = np.full(rows, sequence, dtype=object)
    return pd.DataFrame(mapping, copy=False)


class _SyntheticConverter(TrackConverter[SyntheticParams]):
    """What the two synthetic formats share: the recipe and the dimension check."""

    Params: ClassVar[type[TrackConvertParams]] = SyntheticParams
    carries_tracklets = True
    dimensions: ClassVar[Literal[2, 3]]
    other_format: ClassVar[str]

    @override
    def tracklets(self, params: SyntheticParams) -> bool | None:
        return not params.relabel_after_gap

    @override
    def convert(
        self, path: Path, params: SyntheticParams, hints: EntryHints
    ) -> pd.DataFrame:
        spec = read_synthetic_spec(path)
        if spec.dimensions != self.dimensions:
            message = (
                f"{path} declares dimensions = {spec.dimensions}, which the "
                f"{self.src_format} format does not read. Convert it as "
                f"{self.other_format}."
            )
            raise ValueError(message)
        columns = simulate(spec, relabel_after_gap=params.relabel_after_gap)
        return _table(
            columns,
            rate=spec.fps,
            group=norm_hint(hints.group) or "",
            sequence=norm_hint(hints.sequence) or path.stem,
        )


@register_track_converter
class SyntheticConverter(_SyntheticConverter):
    """A two-dimensional synthetic track file -> a ``mosaic_v1`` table, in pixels."""

    src_format = "synthetic"
    version = "0.1"
    output_schema = "mosaic_v1"
    dimensions = 2
    other_format = "synthetic_3d"


@register_track_converter
class Synthetic3dConverter(_SyntheticConverter):
    """A three-dimensional synthetic track file -> a ``mosaic_world_v1`` table."""

    src_format = "synthetic_3d"
    version = "0.1"
    output_schema = "mosaic_world_v1"
    dimensions = 3
    other_format = "synthetic"
