"""Pre-processing media into variants: the steps, and where their output sits.

Importing the package registers every built-in step in :data:`MEDIA_STEPS`.
"""

from mosaic.core.media.preprocess.appearance import apply_clahe, make_clahe, to_gray
from mosaic.core.media.preprocess.crop import MIN_CROP_SIDE, CropStep
from mosaic.core.media.preprocess.frames import DecimateStep, TrimStep
from mosaic.core.media.preprocess.geometry import FrameMap, Placement
from mosaic.core.media.preprocess.mask import MaskStep
from mosaic.core.media.preprocess.registry import (
    MEDIA_STEPS,
    Frame,
    FrameFn,
    MediaStep,
    register_media_step,
)
from mosaic.core.media.preprocess.specs import MediaStepSpec
from mosaic.core.media.preprocess.steps import AdjustStep, ClaheStep, GrayscaleStep

__all__ = [
    "MEDIA_STEPS",
    "MIN_CROP_SIDE",
    "AdjustStep",
    "ClaheStep",
    "CropStep",
    "DecimateStep",
    "Frame",
    "FrameFn",
    "FrameMap",
    "GrayscaleStep",
    "MaskStep",
    "MediaStep",
    "MediaStepSpec",
    "Placement",
    "TrimStep",
    "apply_clahe",
    "make_clahe",
    "register_media_step",
    "to_gray",
]
