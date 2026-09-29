"""Define the appearance steps: ``grayscale``, ``adjust`` and ``clahe``.

Each changes every pixel by one rule that is independent of the pixel's position.
Each keeps pixel positions and frames as they are, and returns its placement
unchanged.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import cv2
import numpy as np
from pydantic import Field

from mosaic.core.media.preprocess.appearance import apply_clahe, make_clahe, to_gray
from mosaic.core.media.preprocess.geometry import Placement
from mosaic.core.media.preprocess.registry import (
    STEP_DESCRIPTION,
    Frame,
    FrameFn,
    MediaStep,
    register_media_step,
)
from mosaic.core.params import Declared

__all__ = ["AdjustStep", "ClaheStep", "GrayscaleStep"]

_BRIGHTNESS_DESCRIPTION = (
    "The level added to every scaled pixel value, from -255 to 255. Zero leaves the "
    "values unchanged."
)
_CONTRAST_DESCRIPTION = (
    "The factor that every pixel value is scaled by before the brightness is added. "
    "One leaves the contrast unchanged."
)
_GAMMA_DESCRIPTION = (
    "The gamma applied after brightness and contrast, as 255 * (v / 255) ** "
    "(1 / gamma). Above one brightens the mid-tones, and one leaves them unchanged."
)
_CLIP_LIMIT_DESCRIPTION = (
    "The contrast limit of the equalization. Higher equalizes harder and "
    "amplifies more noise."
)
_TILE_GRID_SIZE_DESCRIPTION = (
    "The number of tiles along each side of the image. Each tile is equalized on a "
    "histogram of its pixels alone."
)


@register_media_step
class GrayscaleStep(MediaStep):
    """Convert a BGR frame to gray. The frames stay gray through later steps."""

    name: ClassVar[str] = "grayscale"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = False
    appearance: ClassVar[bool] = True

    step: Annotated[Literal["grayscale"], Declared(STEP_DESCRIPTION)] = "grayscale"

    def place(self, placement: Placement) -> Placement:
        """Return *placement* unchanged. Gray pixels stay where they were."""
        return placement

    def bind(self, placement: Placement) -> FrameFn:
        """Return the gray conversion, which passes a frame that is already gray."""
        return to_gray


@register_media_step
class AdjustStep(MediaStep):
    """Scale, offset and gamma-correct every pixel value through one lookup table.

    A value ``v`` becomes ``clip(contrast * v + brightness, 0, 255)``, then
    ``255 * (v / 255) ** (1 / gamma)``, rounded half to even.
    """

    name: ClassVar[str] = "adjust"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = False
    appearance: ClassVar[bool] = True

    step: Annotated[Literal["adjust"], Declared(STEP_DESCRIPTION)] = "adjust"
    brightness: Annotated[
        float, Field(ge=-255.0, le=255.0), Declared(_BRIGHTNESS_DESCRIPTION)
    ] = 0.0
    contrast: Annotated[
        float, Field(gt=0.0, allow_inf_nan=False), Declared(_CONTRAST_DESCRIPTION)
    ] = 1.0
    gamma: Annotated[
        float, Field(gt=0.0, allow_inf_nan=False), Declared(_GAMMA_DESCRIPTION)
    ] = 1.0

    def place(self, placement: Placement) -> Placement:
        """Return *placement* unchanged. The step changes values and keeps positions."""
        return placement

    def bind(self, placement: Placement) -> FrameFn:
        """Build the 256-level lookup table once, and apply it to each frame."""
        table = self.lookup_table()

        def adjust(frame: Frame) -> Frame:
            return np.asarray(cv2.LUT(frame, table), dtype=np.uint8)

        return adjust

    def lookup_table(self) -> Frame:
        """Return the level that each of the 256 input levels becomes."""
        levels = np.arange(256, dtype=np.float64)
        scaled = np.clip(self.contrast * levels + self.brightness, 0.0, 255.0)
        corrected = 255.0 * (scaled / 255.0) ** (1.0 / self.gamma)
        return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


@register_media_step
class ClaheStep(MediaStep):
    """Equalize contrast locally, tile by tile, with a limit on the contrast gained.

    A gray frame is equalized directly. A BGR frame is equalized on the lightness
    channel of LAB. Its hues are kept.
    """

    name: ClassVar[str] = "clahe"
    version: ClassVar[str] = "0.1"
    moves_pixels: ClassVar[bool] = False
    appearance: ClassVar[bool] = True

    step: Annotated[Literal["clahe"], Declared(STEP_DESCRIPTION)] = "clahe"
    clip_limit: Annotated[
        float, Field(gt=0.0, allow_inf_nan=False), Declared(_CLIP_LIMIT_DESCRIPTION)
    ] = 2.0
    tile_grid_size: Annotated[
        int, Field(ge=1), Declared(_TILE_GRID_SIZE_DESCRIPTION)
    ] = 8

    def place(self, placement: Placement) -> Placement:
        """Return *placement* unchanged. The step changes values and keeps positions."""
        return placement

    def bind(self, placement: Placement) -> FrameFn:
        """Build the CLAHE once, and apply it to each frame."""
        clahe = make_clahe(self.clip_limit, self.tile_grid_size)

        def equalize(frame: Frame) -> Frame:
            return apply_clahe(frame, clahe)

        return equalize
