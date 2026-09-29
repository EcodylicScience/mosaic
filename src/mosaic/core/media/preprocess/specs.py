"""The union of the built-in steps, discriminated by each step's ``step`` field.

A static union rather than one built from :data:`MEDIA_STEPS`, so a parameter
model holding a list of steps publishes every step's full schema, which is what
a pipeline editor draws its step controls from. The set of steps is closed, so a
step added later edits this union, and a test holds the union and the registry
to the same names.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mosaic.core.media.preprocess.crop import CropStep
from mosaic.core.media.preprocess.frames import DecimateStep, TrimStep
from mosaic.core.media.preprocess.mask import MaskStep
from mosaic.core.media.preprocess.steps import AdjustStep, ClaheStep, GrayscaleStep

__all__ = ["MediaStepSpec"]

type MediaStepSpec = Annotated[
    CropStep
    | MaskStep
    | TrimStep
    | DecimateStep
    | GrayscaleStep
    | AdjustStep
    | ClaheStep,
    Field(discriminator="step"),
]
"""One step of a list, validated into its own model by its ``step`` name."""
