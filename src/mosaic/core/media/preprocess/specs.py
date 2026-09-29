"""``MediaStepSpec`` is the union of the built-in steps, discriminated by ``step``.

The union names the step classes statically instead of building itself from
:data:`MEDIA_STEPS`. A parameter model with a list of steps therefore publishes
every step's full schema, and a pipeline editor builds its step controls from that
schema. The set of steps is closed. A new step is added to this union as well as
registered, and a test fails when the union and the registry name different steps.
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
"""One step of a list, validated into the model that its ``step`` name selects."""
