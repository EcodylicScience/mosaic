"""Declares the parameters of a predict-frames run.

Imports ``core`` and ``tracking.common`` and no op module. The adapters and the
op both import it without a cycle.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator

from mosaic.core.params import HASH_EXCLUDE, Declared, Params, Probability
from mosaic.tracking.common.detection_params import (
    AgnosticNms,
    DistanceOverRadiusThreshold,
    InferencePrecision,
    IouThreshold,
    MaxDetections,
)
from mosaic.tracking.common.params import DEVICE_INDEX_NOTE

__all__ = ["PredictFramesParams"]

_MODEL_DESCRIPTION = (
    "The run ids of the trained models to predict with: one run, or a SLEAP "
    "top-down pair given as its centroid run, then its centered-instance run."
)

_FRAMES_RUN_ID_DESCRIPTION = (
    "The run id of the extraction run whose frames are predicted."
)

_CONF_THRESHOLD_DESCRIPTION = (
    "yolo-pose and POLO: the instance confidence below which a detection is "
    "dropped. Lower than the 0.25 of the inference ops, to keep weak detections "
    "for an annotator to correct."
)

_IMGSZ_DESCRIPTION = (
    "yolo-pose and POLO: the longest side an image is resized to before the "
    "model reads it."
)

_PEAK_THRESHOLD_DESCRIPTION = (
    "SLEAP: the minimum confidence-map peak that is kept as a keypoint."
)

_MAX_INSTANCES_DESCRIPTION = (
    "SLEAP: the most instances kept in one frame. Unset keeps every one."
)

_DEVICE_DESCRIPTION = (
    "Where the model runs: auto, cpu, or a GPU index. auto leaves the choice "
    "to the tool, which takes the first CUDA device it finds, else the CPU. "
    "Lightning Pose runs on CUDA only."
) + DEVICE_INDEX_NOTE

_BATCH_SIZE_DESCRIPTION = "How many images the model reads in one forward pass."


class PredictFramesParams(Params):
    """Declares how a trained model predicts on the frames of one extraction run.

    Some fields are read only for the models of the frameworks listed:

    - yolo-pose and POLO: ``conf_threshold``, ``imgsz``, ``max_det``,
      ``agnostic_nms`` and ``precision``.
    - yolo-pose: ``iou``.
    - POLO: ``dor``.
    - SLEAP: ``peak_threshold`` and ``max_instances``.

    ``iou``, ``max_det``, ``agnostic_nms`` and ``precision`` take the types,
    the descriptions and the defaults that ``infer-pose`` and ``infer-points``
    declare. Their descriptions name no framework, and the list above names
    it. Each is passed to the tool explicitly. A refreshed tool lock then
    leaves the predictions under a run id unchanged. ``dor`` takes the type of
    ``infer-points`` and must be positive.

    ``max_det`` and ``max_instances`` stay two fields, because Ultralytics
    keeps 300 detections by default and sleap-nn keeps every instance.
    """

    model: Annotated[
        list[str], Field(min_length=1, max_length=2), Declared(_MODEL_DESCRIPTION)
    ]
    frames_run_id: Annotated[
        str, Field(min_length=1), Declared(_FRAMES_RUN_ID_DESCRIPTION)
    ]
    conf_threshold: Annotated[Probability, Declared(_CONF_THRESHOLD_DESCRIPTION)] = 0.05
    imgsz: Annotated[int, Field(gt=0), Declared(_IMGSZ_DESCRIPTION, unit="px")] = 640
    iou: IouThreshold = 0.7
    max_det: MaxDetections = 300
    agnostic_nms: AgnosticNms = False
    precision: InferencePrecision = "fp32"
    dor: Annotated[DistanceOverRadiusThreshold, Field(gt=0.0)] = None
    peak_threshold: Annotated[Probability, Declared(_PEAK_THRESHOLD_DESCRIPTION)] = 0.2
    max_instances: Annotated[
        int | None, Field(ge=1), Declared(_MAX_INSTANCES_DESCRIPTION)
    ] = None
    device: Annotated[str, HASH_EXCLUDE, Declared(_DEVICE_DESCRIPTION)] = "auto"
    batch_size: Annotated[
        int, Field(ge=1), HASH_EXCLUDE, Declared(_BATCH_SIZE_DESCRIPTION)
    ] = 16

    # A consumer that sets the device itself narrows the field out of the model it
    # exposes; the check then has nothing to check there, and still runs here.
    @field_validator("device", check_fields=False)
    @classmethod
    def _device_is_spelled(cls, value: str) -> str:
        """Refuse a device other than ``auto``, ``cpu`` or a CUDA index, at submit time.

        Each framework takes these values, except that Lightning Pose runs on
        CUDA only and the op refuses its ``cpu`` when it plans the run. Any
        other value is refused here, before the run takes a claim.
        """
        if value in ("auto", "cpu") or (value.isascii() and value.isdigit()):
            return value
        msg = (
            f"unusable device {value!r}: give 'auto', 'cpu' or a CUDA index such "
            "as '0'."
        )
        raise ValueError(msg)
