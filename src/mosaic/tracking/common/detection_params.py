"""The detection arguments that the Ultralytics tracker and inference ops share.

Each is a field type carrying its bounds and its prose, so the ``ultralytics``
tracker, ``infer-pose`` and ``infer-points`` declare one argument one way. Every
op that takes one passes it to Ultralytics explicitly, so an upstream retune of
a default cannot re-mean an identifier already minted.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from mosaic.core.params import HASH_EXCLUDE, Declared, Probability
from mosaic.tracking.external.runner.ultralytics_protocol import Precision

__all__ = ["AgnosticNms", "InferencePrecision", "IouThreshold", "MaxDetections"]

_IOU_DESCRIPTION = "The NMS IoU threshold for suppressing overlapping detections."

_MAX_DET_DESCRIPTION = "The most detections kept from one frame."

_AGNOSTIC_NMS_DESCRIPTION = (
    "Suppress overlapping detections across classes instead of within each one."
)

_PRECISION_DESCRIPTION = (
    "The numeric precision inference runs at. fp16 halves the memory a batch "
    "needs and can move a detection across a threshold."
)

IouThreshold: TypeAlias = Annotated[Probability, Declared(_IOU_DESCRIPTION)]
"""Ultralytics' ``iou``. Its shipped default is 0.7."""

MaxDetections: TypeAlias = Annotated[int, Field(gt=0), Declared(_MAX_DET_DESCRIPTION)]
"""Ultralytics' ``max_det``. Its shipped default is 300."""

AgnosticNms: TypeAlias = Annotated[bool, Declared(_AGNOSTIC_NMS_DESCRIPTION)]
"""Ultralytics' ``agnostic_nms``. Its shipped default is False."""

InferencePrecision: TypeAlias = Annotated[
    Precision, HASH_EXCLUDE, Declared(_PRECISION_DESCRIPTION)
]
"""fp32 or fp16, which the runner spells ``half`` or ``quantize`` as the
installed release takes it. Out of identity, as the device is."""
