"""The frame-selecting steps: ``trim`` keeps a source range, ``decimate`` thins.

Neither changes a pixel. Each changes only the placement's frame map, and the
writer reads just the frames that map selects, so both bind the identity.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.preprocess.registry import (
    STEP_DESCRIPTION,
    Frame,
    FrameFn,
    MediaStep,
    register_media_step,
)
from mosaic.core.params import Declared

__all__ = ["DecimateStep", "TrimStep"]

_START_DESCRIPTION = "The first source frame of the range kept."
_STOP_DESCRIPTION = "The source frame the kept range stops before."
_EVERY_DESCRIPTION = "Keep one frame in this many, starting with the first."


def _unchanged(frame: Frame) -> Frame:
    """The frame as it is: a frame-selecting step changes no pixel."""
    return frame


@register_media_step
class TrimStep(MediaStep):
    """Keep the frames from ``start`` up to, not including, ``stop``.

    Both are source frame numbers, and the range applies to the frames the
    steps before it kept (the placement's frame map). After a ``decimate``, a
    ``start`` between two kept frames begins at the next kept frame. A range
    reaching past the source or past the frames already kept, or holding none
    of them, is refused.
    """

    name: ClassVar[str] = "trim"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = False
    appearance: ClassVar[bool] = False

    step: Annotated[Literal["trim"], Declared(STEP_DESCRIPTION)] = "trim"
    start: Annotated[int, Field(ge=0), Declared(_START_DESCRIPTION)]
    stop: Annotated[int, Declared(_STOP_DESCRIPTION)]

    @model_validator(mode="after")
    def _refuse_an_empty_range(self) -> Self:
        """Refuse a *stop* at or before *start*, which keeps no frame."""
        if self.start >= self.stop:
            raise ValueError(
                f"a trim of source frames [{self.start}, {self.stop}) keeps no "
                f"frame; the start must be before the stop"
            )
        return self

    def place(self, placement: Placement) -> Placement:
        """*placement* with its frame map narrowed to ``[start, stop)``.

        Raises:
            ValueError: If *stop* is past the source's last frame, or the range
                reaches outside the current frame map or holds none of its
                frames.
        """
        if self.stop > placement.source_frame_count:
            raise ValueError(
                f"a trim to source frames [{self.start}, {self.stop}) reaches past "
                f"the source, which holds frames [0, {placement.source_frame_count})"
            )
        return dataclasses.replace(
            placement, frames=placement.frames.within(self.start, self.stop)
        )

    def bind(self, placement: Placement) -> FrameFn:
        """The identity: a trim selects frames and changes none."""
        return _unchanged


@register_media_step
class DecimateStep(MediaStep):
    """Keep every n-th frame, where n is ``every``, starting with the first.

    The count runs over the frames the steps before it kept (the placement's
    frame map), so after a ``trim`` it thins only the trimmed range.
    """

    name: ClassVar[str] = "decimate"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = False
    appearance: ClassVar[bool] = False

    step: Annotated[Literal["decimate"], Declared(STEP_DESCRIPTION)] = "decimate"
    every: Annotated[int, Field(ge=2), Declared(_EVERY_DESCRIPTION)]

    def place(self, placement: Placement) -> Placement:
        """*placement* with its frame map thinned to one frame in *every*."""
        return dataclasses.replace(placement, frames=placement.frames.every(self.every))

    def bind(self, placement: Placement) -> FrameFn:
        """The identity: a decimation selects frames and changes none."""
        return _unchanged
