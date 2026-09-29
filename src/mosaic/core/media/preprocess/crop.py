"""Define the ``crop`` step, which makes a source-space rectangle the whole image."""

from __future__ import annotations

import dataclasses
from typing import Annotated, ClassVar, Final, Literal

from pydantic import Field, ValidationInfo, field_validator

from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.preprocess.registry import (
    STEP_DESCRIPTION,
    Frame,
    FrameFn,
    MediaStep,
    register_media_step,
)
from mosaic.core.media.video_io import apply_crop
from mosaic.core.params import Declared

__all__ = ["MIN_CROP_SIDE", "CropStep"]

MIN_CROP_SIDE: Final = 4
"""The smallest width or height a crop may have, in pixels.

It is the AV1 encoder's minimum, established by encoding five frames through
mosaic-media's ``FFmpegVideoWriter`` (``libsvtav1``, SVT-AV1 4.2.0) at sizes from
2 to 1024 pixels a side, each in a separate process, and decoding them back. Every
size with both sides at least 4 wrote and read back all five frames. A side of 2
failed: ``2x2`` raised ``Cannot allocate memory`` from ``avcodec_receive_packet``,
and ``8x2`` and ``16x2`` crashed the process. The H.264 fallback's ``libx264``
encodes ``4x4`` as well.
"""

_X_DESCRIPTION = "The source column of the rectangle's left edge."
_Y_DESCRIPTION = "The source row of the rectangle's top edge."
_WIDTH_DESCRIPTION = (
    "The rectangle's width. Even, because the variant is encoded as yuv420p, "
    "and at least the AV1 encoder's minimum."
)
_HEIGHT_DESCRIPTION = (
    "The rectangle's height. Even, because the variant is encoded as yuv420p, "
    "and at least the AV1 encoder's minimum."
)


@register_media_step
class CropStep(MediaStep):
    """Cut the image down to a rectangle given in source coordinates.

    The rectangle must lie wholly inside the current image, which after an
    earlier crop is that crop's rectangle. The step refuses a rectangle outside it.
    Clamping changes the geometry without recording the change.
    """

    name: ClassVar[str] = "crop"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = True
    appearance: ClassVar[bool] = False

    step: Annotated[Literal["crop"], Declared(STEP_DESCRIPTION)] = "crop"
    x: Annotated[int, Field(ge=0), Declared(_X_DESCRIPTION, unit="px")]
    y: Annotated[int, Field(ge=0), Declared(_Y_DESCRIPTION, unit="px")]
    width: Annotated[
        int, Field(ge=MIN_CROP_SIDE), Declared(_WIDTH_DESCRIPTION, unit="px")
    ]
    height: Annotated[
        int, Field(ge=MIN_CROP_SIDE), Declared(_HEIGHT_DESCRIPTION, unit="px")
    ]

    @field_validator("width", "height")
    @classmethod
    def _refuse_an_odd_side(cls, value: int, info: ValidationInfo) -> int:
        """Refuse an odd side, which yuv420p's half-resolution chroma cannot encode."""
        if value % 2:
            raise ValueError(
                f"a crop {info.field_name} of {value} is odd. The variant is "
                f"encoded as yuv420p, which stores color at half resolution. "
                f"Both sides must be even"
            )
        return value

    def place(self, placement: Placement) -> Placement:
        """Return the placement of the rectangle, with the frames and rate unchanged.

        Raises:
            ValueError: If the rectangle is not wholly inside the current image.
        """
        right = placement.offset_x + placement.width
        bottom = placement.offset_y + placement.height
        if (
            self.x < placement.offset_x
            or self.y < placement.offset_y
            or self.x + self.width > right
            or self.y + self.height > bottom
        ):
            requested = f"({self.x}, {self.y}, {self.width}, {self.height})"
            current = (
                f"({placement.offset_x}, {placement.offset_y}, {placement.width}, "
                f"{placement.height})"
            )
            raise ValueError(
                f"the crop {requested} is not inside the current image, which is "
                f"the source rectangle {current}. Both are (x, y, width, height) "
                f"in source coordinates."
            )
        return dataclasses.replace(
            placement,
            offset_x=self.x,
            offset_y=self.y,
            width=self.width,
            height=self.height,
        )

    def bind(self, placement: Placement) -> FrameFn:
        """Slice the rectangle out of a frame of the current image, gray or BGR."""
        rectangle = (
            self.x - placement.offset_x,
            self.y - placement.offset_y,
            self.width,
            self.height,
        )

        def crop(frame: Frame) -> Frame:
            return apply_crop(frame, rectangle)

        return crop
