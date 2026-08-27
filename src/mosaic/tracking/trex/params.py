"""What a TREx run is, declared once for every consumer.

One statement per field serves validation, run identity, subprocess invocation
and discovery. ``Field`` states the constraint pydantic enforces,
:class:`~mosaic.core.pipeline.types.Declared` states the prose a client draws a
control from, and :class:`~mosaic.core.pipeline.markers.Phase` names which of
TREx's two subprocess phases consumes the field.

The model is declared beside the integration rather than beside the op, because
:func:`~mosaic.tracking.trex.dataset_runs.run_trex` and both phase functions
take it: declared in ``tracking/ops/trex.py``, the integration would import its
own adapter. ``tracking/trex/version.py`` is declared here for the same reason.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mosaic.core.pipeline.markers import Phase
from mosaic.core.pipeline.types import Declared, JsonValue, Probability
from mosaic.tracking.common.params import PhasedTrackerOpParams

__all__ = ["TrexParams"]

_DETECT_MODEL_DESCRIPTION = (
    "A YOLO .pt path, or the run id of the training op that produced the "
    "weights. Identity records the training run id when the reference is one, "
    "and the weights' content digest when it is a bare path."
)

_DETECT_TYPE_DESCRIPTION = (
    "The detection algorithm. Known values are yolo, background_subtraction and points."
)

_DETECT_CONF_THRESHOLD_DESCRIPTION = (
    "The minimum YOLO detection confidence. Unset, TREx applies 0.1."
)

_DETECT_IOU_THRESHOLD_DESCRIPTION = (
    "The NMS IoU threshold for suppressing overlapping detections. TREx leaves "
    "it unset by default: unset preserves the upstream model's default "
    "postprocessing, and setting it may disable end-to-end NMS-free inference. "
    "A number here is a choice about the detector as well as a threshold."
)

_CM_PER_PIXEL_DESCRIPTION = (
    "The spatial calibration factor. Unset, TREx derives it from "
    "meta_real_width / video_width."
)

_META_ENCODING_DESCRIPTION = "The pixel encoding. Known values are gray and rgb8."

_CONVERT_EXTRA_SETTINGS_DESCRIPTION = (
    "Additional TREx parameters sent as -key value pairs for the conversion "
    "phase. A None value removes a parameter mosaic would otherwise send."
)

_TRACK_MAX_INDIVIDUALS_DESCRIPTION = (
    "The maximum number of simultaneous individuals to track. Unset, TREx applies 1024."
)

_TRACK_MAX_SPEED_DESCRIPTION = (
    "The maximum plausible speed. Its meaning depends on the cm_per_pixel the "
    "conversion applied."
)

_TRACK_MAX_REASSIGN_TIME_DESCRIPTION = (
    "How long to wait before giving up on a lost individual. Unset, TREx applies 0.5."
)

_TRACK_TRUSTED_PROBABILITY_DESCRIPTION = (
    "The probability below which the current tracklet ends and a new one "
    "starts. Unset, TREx applies 0.25. Lowering it produces longer tracklets "
    "held together by weaker evidence."
)

_ANALYSIS_RANGE_DESCRIPTION = (
    "The frame range to analyze. -1 means the beginning or the end of the video."
)

_VISUAL_IDENTIFICATION_MODEL_PATH_DESCRIPTION = (
    "Pre-trained identity weights (.pth, without the extension), or the run id "
    "of the identity-training op that produced them. Identity records the "
    "training run id when the reference is one, and the weights' content digest "
    "when it is a bare path."
)

_AUTO_TRAIN_DESCRIPTION = "Train visual identification automatically after tracking."

_DETECT_KEYPOINT_COUNT_DESCRIPTION = (
    "How many keypoints detect_model reports. Set it whenever that model is a "
    "pose model, or the tracks come back without their keypoint columns. TREx "
    "derives the poseX<i> / poseY<i> names from a keypoint format it learns by "
    "loading the model, and mosaic converts and tracks as two invocations, so "
    "the exporting process has never loaded one."
)

_TRACK_EXTRA_SETTINGS_DESCRIPTION = (
    "Additional TREx parameters sent as -key value pairs for the tracking "
    "phase. A None value removes a parameter mosaic would otherwise send."
)


class TrexParams(PhasedTrackerOpParams):
    """Parameters for the ``trex`` tracking op and for ``run_trex``.

    Every tool-facing parameter but ``auto_train`` defaults to ``None``, which
    sends no flag and leaves TREx's own default in force; each field's
    description names the value that is.

    Leaving a parameter unset and setting it to the value TREx would have
    applied are two different runs. Unset is recorded as a setting of its own,
    so the two hash to different ``run_id`` values and neither reuses the
    other's output.
    """

    # No field states a default of mosaic's own. One that differed from TREx's
    # would impose an opinion a caller cannot decline, and None reaches the argv
    # builder, which omits the flag -- so unset is expressed the whole way down
    # rather than being translated into a stand-in value.
    detect_model: Annotated[
        str | None, Declared(_DETECT_MODEL_DESCRIPTION), Phase("convert")
    ] = None
    # detect_type and meta_encoding stay str with examples rather than narrowing
    # to Literal: TREx owns both vocabularies and runs in a conda environment of
    # its own, so mosaic has no configuration to diff its own list against. A
    # client offers the known values and accepts a typed one.
    detect_type: Annotated[
        str | None,
        Field(examples=["yolo", "background_subtraction", "points"]),
        Declared(_DETECT_TYPE_DESCRIPTION),
        Phase("convert"),
    ] = None
    detect_conf_threshold: Annotated[
        Probability | None,
        Declared(_DETECT_CONF_THRESHOLD_DESCRIPTION),
        Phase("convert"),
    ] = None
    detect_iou_threshold: Annotated[
        Probability | None,
        Declared(_DETECT_IOU_THRESHOLD_DESCRIPTION),
        Phase("convert"),
    ] = None
    cm_per_pixel: Annotated[
        float | None,
        Field(gt=0.0),
        Declared(_CM_PER_PIXEL_DESCRIPTION, unit="cm/px"),
        Phase("convert"),
    ] = None
    meta_encoding: Annotated[
        str | None,
        Field(examples=["gray", "rgb8"]),
        Declared(_META_ENCODING_DESCRIPTION),
        Phase("convert"),
    ] = None
    # JsonValue rather than object on both pass-through dictionaries: they are
    # the only fields carrying arbitrary user values into the run identifier, so
    # an unrepresentable value here is what identity_ready refuses. Typing them
    # moves that refusal to params construction, where pydantic names the field,
    # rather than into hash_params with only a type name to report.
    convert_extra_settings: Annotated[
        dict[str, JsonValue] | None,
        Declared(_CONVERT_EXTRA_SETTINGS_DESCRIPTION),
        Phase("convert"),
    ] = None
    # The conversion caps detections and the tracking phase reads the same cap,
    # so changing it invalidates a conversion as well as a tracking.
    track_max_individuals: Annotated[
        int | None,
        Field(ge=1),
        Declared(_TRACK_MAX_INDIVIDUALS_DESCRIPTION),
        Phase("convert", "track"),
    ] = None
    track_max_speed: Annotated[
        float | None,
        Field(gt=0.0),
        Declared(_TRACK_MAX_SPEED_DESCRIPTION, unit="cm/s"),
        Phase("track"),
    ] = None
    track_max_reassign_time: Annotated[
        float | None,
        Field(ge=0.0),
        Declared(_TRACK_MAX_REASSIGN_TIME_DESCRIPTION, unit="s"),
        Phase("track"),
    ] = None
    track_trusted_probability: Annotated[
        Probability | None,
        Declared(_TRACK_TRUSTED_PROBABILITY_DESCRIPTION),
        Phase("track"),
    ] = None
    analysis_range: Annotated[
        tuple[int, int] | None,
        Declared(_ANALYSIS_RANGE_DESCRIPTION, unit="frames"),
        Phase("track"),
    ] = None
    visual_identification_model_path: Annotated[
        str | None,
        Declared(_VISUAL_IDENTIFICATION_MODEL_PATH_DESCRIPTION),
        Phase("track"),
    ] = None
    auto_train: Annotated[bool, Declared(_AUTO_TRAIN_DESCRIPTION), Phase("track")] = (
        False
    )
    # A track key despite the detect_ prefix: it changes which columns are
    # exported, never what was detected. Naming the keypoints re-keys the
    # tracking run -- the cheap phase -- and reuses the .pv, the expensive one.
    detect_keypoint_count: Annotated[
        int | None,
        Field(ge=0),
        Declared(_DETECT_KEYPOINT_COUNT_DESCRIPTION),
        Phase("track"),
    ] = None
    track_extra_settings: Annotated[
        dict[str, JsonValue] | None,
        Declared(_TRACK_EXTRA_SETTINGS_DESCRIPTION),
        Phase("track"),
    ] = None
