"""Define the ``mask`` step, which blacks out the pixels outside or inside a polygon."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field

from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.preprocess.registry import (
    STEP_DESCRIPTION,
    Frame,
    FrameFn,
    MediaStep,
    register_media_step,
)
from mosaic.core.params import Declared

__all__ = ["MaskStep"]

_POLYGON_DESCRIPTION = (
    "The polygon's vertices as (x, y) source coordinates, at least three. It may "
    "extend outside the image but may not lie wholly outside it."
)
_KEEP_DESCRIPTION = (
    "Keep the pixels inside the polygon and black out the rest. False blacks out "
    "the inside instead."
)


@register_media_step
class MaskStep(MediaStep):
    """Fill the pixels on one side of a polygon with black.

    The step leaves every pixel where it was. A polygon that extends outside the
    current image is therefore accepted and drawn clipped to it. One that lies
    wholly outside is refused, because it blacks out the whole image or leaves it
    unchanged.
    """

    name: ClassVar[str] = "mask"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = False
    appearance: ClassVar[bool] = False

    step: Annotated[Literal["mask"], Declared(STEP_DESCRIPTION)] = "mask"
    polygon: Annotated[
        list[tuple[int, int]],
        Field(min_length=3),
        Declared(_POLYGON_DESCRIPTION, unit="px"),
    ]
    keep: Annotated[bool, Declared(_KEEP_DESCRIPTION)] = True

    def place(self, placement: Placement) -> Placement:
        """Return *placement* unchanged, once the polygon is known to cover a pixel.

        Raises:
            ValueError: If the polygon does not cover a pixel of the current image.
        """
        if not self._inside(placement).any():
            xs = [x for x, _ in self.polygon]
            ys = [y for _, y in self.polygon]
            right = placement.offset_x + placement.width - 1
            bottom = placement.offset_y + placement.height - 1
            raise ValueError(
                f"the mask polygon does not cover a pixel of the current image. It "
                f"spans x {min(xs)}..{max(xs)} and y {min(ys)}..{max(ys)}, and the "
                f"image covers x {placement.offset_x}..{right} and y "
                f"{placement.offset_y}..{bottom} in source coordinates"
            )
        return placement

    def bind(self, placement: Placement) -> FrameFn:
        """Render the polygon once, and black out the chosen side of each frame."""
        inside = self._inside(placement)
        blacked_out = ~inside if self.keep else inside

        def mask(frame: Frame) -> Frame:
            masked = frame.copy()
            masked[blacked_out] = 0
            return masked

        return mask

    def _inside(self, placement: Placement) -> npt.NDArray[np.bool_]:
        """Return a mask, True where ``fillPoly`` draws the polygon on the image."""
        vertices = np.array(
            [(x - placement.offset_x, y - placement.offset_y) for x, y in self.polygon],
            dtype=np.int32,
        )
        raster = np.zeros((placement.height, placement.width), dtype=np.uint8)
        _ = cv2.fillPoly(raster, [vertices], 255)
        return raster != 0
