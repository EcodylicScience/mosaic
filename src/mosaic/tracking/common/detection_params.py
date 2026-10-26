"""The detection arguments that the Ultralytics tracker and inference ops share.

Each is a field type carrying its bounds and its prose, so the ``ultralytics``
tracker, ``infer-pose``, ``infer-points`` and ``predict-frames`` declare one
argument one way. Every op that takes one passes it to Ultralytics or POLO
explicitly, so an upstream retune of a default cannot re-mean an identifier
already minted.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from mosaic.core.params import HASH_EXCLUDE, Declared, Probability
from mosaic.tracking.external.runner.ultralytics_protocol import Precision

__all__ = [
    "AgnosticNms",
    "DistanceOverRadiusThreshold",
    "InferencePrecision",
    "IouThreshold",
    "MaxDetections",
]

_IOU_DESCRIPTION = "The NMS IoU threshold for suppressing overlapping detections."

_MAX_DET_DESCRIPTION = "The most detections kept from one frame."

_AGNOSTIC_NMS_DESCRIPTION = (
    "Suppress overlapping detections across classes instead of within each one."
)

_DOR_DESCRIPTION = (
    "Distance-over-radius threshold of POLO's point suppression. A point whose "
    "distance to a higher-scoring point, divided by its radius, is below it is "
    "dropped. Unset uses the threshold the checkpoint was trained with, or "
    "POLO's 0.3 for a checkpoint that records none."
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

DistanceOverRadiusThreshold: TypeAlias = Annotated[
    float | None, Declared(_DOR_DESCRIPTION)
]
"""POLO's ``dor``. Unset reads the checkpoint's ``train_args.dor``. The type
declares no bound. A field adds its own."""

InferencePrecision: TypeAlias = Annotated[
    Precision, HASH_EXCLUDE, Declared(_PRECISION_DESCRIPTION)
]
"""fp32 or fp16, which the runner spells ``half`` or ``quantize`` as the
installed release takes it. Out of identity, as the device is."""
