"""Declares the synthetic track file and reads it.

A synthetic track file declares every parameter of one generated sequence: the
arena, the motion, the interaction zones, the body, the dropouts and the seed.
The file's checksum identifies the sequence it produces.
Every length is in the unit of the table the file generates: pixels for a 2D
table, meters for a 3D one.

Imports pydantic, the standard library and :mod:`mosaic.core.strict_model` at
module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mosaic.core.strict_model import StrictModel, read_toml_model

__all__ = [
    "MAX_TABLE_CELLS",
    "SYNTHETIC_FORMAT",
    "ArenaSpec",
    "BodySpec",
    "DropoutSpec",
    "MotionSpec",
    "SyntheticSpec",
    "ZoneSpec",
    "read_synthetic_spec",
]

SYNTHETIC_FORMAT = "mosaic-synthetic/1"

MAX_TABLE_CELLS = 600_000_000
"""The most numeric cells a generated table may contain.

A cell is one value of a numeric column of one row. The generator returns one
table, and the write path keeps a few copies of it in memory. The cap is the
largest shape whose write was measured to fit.
"""

MAX_INDIVIDUALS = 300
"""The most individuals a file may declare, bounded by the dense neighbor search."""

MAX_KEYPOINTS = 32

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Rate = Annotated[float, Field(ge=0, lt=1)]
MeanFrames = Annotated[float, Field(ge=1, allow_inf_nan=False)]


class ArenaSpec(StrictModel):
    """The region the individuals move in.

    A box spans ``[0, width]`` by ``[0, height]``. A disk of ``radius`` is
    centered at ``(radius, radius)``. In 3D the arena also spans ``[0, depth]``
    along z, which the file's ``dimensions`` decides.
    """

    shape: Literal["box", "disk"]
    width: Positive | None = None
    height: Positive | None = None
    radius: Positive | None = None
    depth: Positive | None = None
    start_radius: Positive | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> ArenaSpec:
        if self.shape == "box":
            if self.radius is not None:
                msg = "arena.radius belongs to a disk, and the arena is a box"
                raise ValueError(msg)
            if self.width is None or self.height is None:
                msg = "arena.width and arena.height are required for a box"
                raise ValueError(msg)
        else:
            if self.width is not None or self.height is not None:
                msg = "arena.width and arena.height belong to a box, and the arena is a disk"
                raise ValueError(msg)
            if self.radius is None:
                msg = "arena.radius is required for a disk"
                raise ValueError(msg)
        if self.start_radius is not None and self.start_radius > self.inscribed_radius:
            msg = (
                f"arena.start_radius {self.start_radius} exceeds the arena's "
                f"inscribed radius {self.inscribed_radius}"
            )
            raise ValueError(msg)
        return self

    @property
    def inscribed_radius(self) -> float:
        """The radius of the largest circle centered in the arena's footprint."""
        if self.shape == "disk":
            assert self.radius is not None
            return self.radius
        assert self.width is not None
        assert self.height is not None
        return min(self.width, self.height) / 2

    def smallest_extent(self, dimensions: Literal[2, 3]) -> float:
        """The shortest span of the arena along any axis it has."""
        footprint = 2 * self.inscribed_radius
        if dimensions == 2:
            return footprint
        assert self.depth is not None
        return min(footprint, self.depth)


class MotionSpec(StrictModel):
    """A correlated random walk: speed relaxes toward a mean, heading diffuses."""

    mean_speed: NonNegative
    speed_sd: NonNegative = 0.0
    speed_tau_s: Positive = 1.0
    turning_sd: NonNegative = 0.5

    @property
    def max_speed(self) -> float:
        """The cap on speed: the mean plus four stationary standard deviations."""
        return self.mean_speed + 4 * self.speed_sd


class ZoneSpec(StrictModel):
    """Repulsion, orientation and attraction zones around each individual."""

    repulsion: Positive
    orientation: Positive
    attraction: Positive
    field_of_view_deg: Annotated[float, Field(gt=0, le=360, allow_inf_nan=False)] = (
        270.0
    )
    max_turn_rate: Positive = 6.0

    @model_validator(mode="after")
    def _check_radii(self) -> ZoneSpec:
        if not self.repulsion <= self.orientation <= self.attraction:
            msg = (
                "zones radii must satisfy 0 < repulsion <= orientation <= attraction, "
                f"got repulsion={self.repulsion}, orientation={self.orientation}, "
                f"attraction={self.attraction}"
            )
            raise ValueError(msg)
        return self


class BodySpec(StrictModel):
    """Keypoints placed along the body axis, each with position noise."""

    keypoints: Annotated[int, Field(ge=0, le=MAX_KEYPOINTS)] = 0
    length: Positive | None = None
    keypoint_sd: NonNegative = 0.0
    confidence_sd: NonNegative = 0.05

    @model_validator(mode="after")
    def _check_length(self) -> BodySpec:
        if self.keypoints > 0 and self.length is None:
            msg = "body.length is required when body.keypoints is above 0"
            raise ValueError(msg)
        return self


class DropoutSpec(StrictModel):
    """Per-frame probabilities that an id goes missing or that a frame gap starts.

    A rate applies to every frame, and a run's length is geometric with the mean
    declared in frames. The rows a file writes do not depend on its ``fps``.
    """

    id_rate: Rate = 0.0
    id_mean_frames: MeanFrames = 10.0
    gap_rate: Rate = 0.0
    gap_mean_frames: MeanFrames = 5.0


class SyntheticSpec(StrictModel):
    """Every parameter of one generated sequence."""

    format: str
    seed: Annotated[int, Field(ge=0)]
    dimensions: Literal[2, 3]
    frames: Annotated[int, Field(ge=1)]
    fps: Positive
    individuals: Annotated[int, Field(ge=1, le=MAX_INDIVIDUALS)]
    burn_in_frames: Annotated[int, Field(ge=0)] = 0
    arena: ArenaSpec
    motion: MotionSpec
    zones: ZoneSpec | None = None
    body: BodySpec | None = None
    dropouts: DropoutSpec | None = None

    @property
    def numeric_columns(self) -> int:
        """The numeric columns of a row: ``frame``, ``time``, ``id``, the position
        axes, and per keypoint its axes plus its confidence."""
        keypoints = 0 if self.body is None else self.body.keypoints
        return 3 + self.dimensions + keypoints * (self.dimensions + 1)

    @field_validator("format")
    @classmethod
    def _check_format(cls, value: str) -> str:
        if value != SYNTHETIC_FORMAT:
            msg = f"format must be {SYNTHETIC_FORMAT!r}, got {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check_file(self) -> SyntheticSpec:
        if self.dimensions == 2 and self.arena.depth is not None:
            msg = "arena.depth is for dimensions = 3, and dimensions is 2"
            raise ValueError(msg)
        if self.dimensions == 3 and self.arena.depth is None:
            msg = "arena.depth is required when dimensions is 3"
            raise ValueError(msg)
        step = self.motion.max_speed / self.fps
        extent = self.arena.smallest_extent(self.dimensions)
        if step > extent:
            msg = (
                f"a step at motion.mean_speed plus four speed_sd, {step}, is longer "
                f"than the arena's smallest extent {extent}; lower motion.mean_speed "
                "or raise fps"
            )
            raise ValueError(msg)
        cells = self.frames * self.individuals * self.numeric_columns
        if cells > MAX_TABLE_CELLS:
            msg = (
                f"frames {self.frames} by individuals {self.individuals} by "
                f"{self.numeric_columns} numeric columns is {cells} cells, over the "
                f"cap of {MAX_TABLE_CELLS}; lower frames, individuals or body.keypoints"
            )
            raise ValueError(msg)
        return self


def read_synthetic_spec(path: Path) -> SyntheticSpec:
    """Read and validate the synthetic track file at *path*.

    Raises:
        ValueError: The file is not valid TOML, or a model refuses it. The message
            names *path* and each refused field.
    """
    return read_toml_model(path, SyntheticSpec, "a synthetic track file")
