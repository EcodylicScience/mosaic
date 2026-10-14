"""Define the media pre-processing steps and the placement of their output.

This file binds every public name on access. Python executes it before any
submodule, and the step modules import cv2, so a reader of a recorded placement
imports ``geometry`` and loads no cv2. Reading any public name of the package
first imports ``specs``, which imports every built-in step and so registers it in
:data:`MEDIA_STEPS`.

Each name is also declared under ``TYPE_CHECKING`` for type checkers and editors.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Final

from mosaic.lazy_exports import lazy_export

if TYPE_CHECKING:
    from .appearance import (
        apply_clahe as apply_clahe,
        make_clahe as make_clahe,
        to_gray as to_gray,
    )
    from .crop import MIN_CROP_SIDE as MIN_CROP_SIDE, CropStep as CropStep
    from .frames import DecimateStep as DecimateStep, TrimStep as TrimStep
    from .geometry import FrameMap as FrameMap, Placement as Placement
    from .mask import MaskStep as MaskStep
    from .registry import (
        MEDIA_STEPS as MEDIA_STEPS,
        Frame as Frame,
        FrameFn as FrameFn,
        MediaStep as MediaStep,
        register_media_step as register_media_step,
    )
    from .specs import MediaStepSpec as MediaStepSpec
    from .steps import (
        AdjustStep as AdjustStep,
        ClaheStep as ClaheStep,
        GrayscaleStep as GrayscaleStep,
    )

_MODULE_OF: Final[dict[str, str]] = {
    "apply_clahe": ".appearance",
    "make_clahe": ".appearance",
    "to_gray": ".appearance",
    "MIN_CROP_SIDE": ".crop",
    "CropStep": ".crop",
    "DecimateStep": ".frames",
    "TrimStep": ".frames",
    "FrameMap": ".geometry",
    "Placement": ".geometry",
    "MaskStep": ".mask",
    "MEDIA_STEPS": ".registry",
    "Frame": ".registry",
    "FrameFn": ".registry",
    "MediaStep": ".registry",
    "register_media_step": ".registry",
    "MediaStepSpec": ".specs",
    "AdjustStep": ".steps",
    "ClaheStep": ".steps",
    "GrayscaleStep": ".steps",
}
"""The submodule that defines each public name."""


def __getattr__(name: str) -> object:
    """Register the built-in steps, then import the submodule that defines *name*."""
    if name in _MODULE_OF:
        _ = import_module(".specs", __name__)
    return lazy_export(__name__, _MODULE_OF, name)


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
