"""What a SLEAP run is, declared once for every consumer.

One statement per field serves validation, run identity, subprocess invocation
and discovery. ``Field`` states the constraint pydantic enforces and
:class:`~mosaic.core.params.Declared` states the prose a client draws a control
from. SLEAP infers and tracks in one gated phase. Its fields do not name a
phase.

The model is declared beside the integration rather than beside the op, because
:func:`~mosaic.tracking.sleap.dataset_runs.run_sleap` and
:func:`~mosaic.tracking.sleap.dataset_runs.sleap_settings` take it: declared in
``tracking/ops/sleap.py``, the integration would import its own adapter.
"""

from __future__ import annotations

from typing import Annotated, Final, Self

from pydantic import Field, field_validator, model_validator

from mosaic.core.pipeline.media_input import MediaInputParams
from mosaic.core.pipeline.ops import ModelReference
from mosaic.core.pipeline.types import JsonValue
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)
from mosaic.tracking.common.params import DEVICE_INDEX_NOTE, TrackerOpParams
from mosaic.tracking.sleap.run import (
    SleapCandidatesMethod,
    SleapFeatures,
    SleapMatchingMethod,
    SleapScoringMethod,
    sleap_track_device_args,
)
from mosaic.tracking.sleap.version import SLEAP_KIND, TRAIN_SLEAP_KIND

__all__ = ["SLEAP_MODEL", "SleapParams"]

_MODEL_PATHS_DESCRIPTION = (
    "One trained SLEAP model directory, or two for a top-down model "
    "(centroid, then centered-instance)."
)

_TRACKING_DESCRIPTION = (
    "Assign identities to detections across frames. When False, no tracker is attached."
)

_USE_FLOW_DESCRIPTION = (
    "Shift each candidate pose by optical flow before matching it to a "
    "detection, sent as --use_flow."
)

_CANDIDATES_METHOD_DESCRIPTION = (
    "Where match candidates come from, sent as --candidates_method: "
    "fixed_window takes every instance of the last few frames, local_queues the "
    "last few instances of each track."
)

_FEATURES_DESCRIPTION = (
    "What a detection and a candidate are compared by, sent as --features: "
    "keypoints, centroids, bboxes or image."
)

_SCORING_METHOD_DESCRIPTION = (
    "How that comparison is scored, sent as --scoring_method: oks, cosine_sim, "
    "iou or euclidean_dist."
)

_TRACK_MATCHING_METHOD_DESCRIPTION = (
    "How detections are assigned to tracks from those scores, sent as "
    "--track_matching_method: hungarian or greedy."
)

_ANALYSIS_RANGE_DESCRIPTION = (
    "The first and last frame to analyze. Unset, SLEAP analyzes the whole video."
)

_TRACKING_WINDOW_SIZE_DESCRIPTION = (
    "How many frames, or instances per track under local_queues, are kept as "
    "match candidates, sent as --tracking_window_size."
)

_MAX_INSTANCES_DESCRIPTION = "The maximum number of instances to detect per frame."

_MAX_TRACKS_DESCRIPTION = (
    "The maximum number of tracks, sent as --max_tracks. sleap-nn enforces it "
    "only through local_queues candidates, so it requires that candidates_method."
)

_PEAK_THRESHOLD_DESCRIPTION = "The minimum confidence for a detected peak."

_SLEAP_EXTRA_SETTINGS_DESCRIPTION = (
    "Additional sleap-nn track options, sent as --key value pairs. A boolean "
    "value becomes a bare --key flag when true and is omitted when false, and a "
    "None value is skipped."
)

_BATCH_SIZE_DESCRIPTION = "The inference batch size."

_DEVICE_DESCRIPTION = (
    "The device to run inference on: cpu, cuda, mps, a CUDA index such as 0, "
    "or cuda:<index>. Unset and auto leave the choice to sleap-nn; a named "
    "device fails where it is absent."
) + DEVICE_INDEX_NOTE


class SleapParams(TrackerOpParams, MediaInputParams):
    """Parameters for the ``sleap`` tracking op and for ``run_sleap``."""

    window_fields = ("analysis_range",)
    # `--frames` is the one `sleap-nn track` option that selects a video's frames
    # by index (sleap_nn/cli.py, sleap-nn 0.3.1), and `analysis_range` is sent as
    # it. The options that filter by annotation status read a .slp and only warn
    # on a video.
    extra_settings_window_keys = {"sleap_extra_settings": ("frames",)}
    op_kind = SLEAP_KIND

    # model: one external model directory, or two for top-down (centroid, then
    # centered-instance). Part of the run_id identity -- via a content digest of
    # the weights, never the paths themselves.
    model_paths: Annotated[list[str], Declared(_MODEL_PATHS_DESCRIPTION)]
    # tracking (part of the run_id identity). Named for the sleap-nn track
    # options they are sent as; the defaults are what the legacy defaults ran
    # (the flow tracker, keypoint OKS, hungarian matching, a five-frame window).
    tracking: Annotated[bool, Declared(_TRACKING_DESCRIPTION)] = True
    use_flow: Annotated[bool, Declared(_USE_FLOW_DESCRIPTION)] = True
    candidates_method: Annotated[
        SleapCandidatesMethod, Declared(_CANDIDATES_METHOD_DESCRIPTION)
    ] = "fixed_window"
    features: Annotated[SleapFeatures, Declared(_FEATURES_DESCRIPTION)] = "keypoints"
    scoring_method: Annotated[
        SleapScoringMethod, Declared(_SCORING_METHOD_DESCRIPTION)
    ] = "oks"
    track_matching_method: Annotated[
        SleapMatchingMethod, Declared(_TRACK_MATCHING_METHOD_DESCRIPTION)
    ] = "hungarian"
    # No unit: it counts frames under fixed_window and instances per track
    # under local_queues.
    tracking_window_size: Annotated[
        int, Declared(_TRACKING_WINDOW_SIZE_DESCRIPTION)
    ] = 5
    max_tracks: Annotated[int | None, Declared(_MAX_TRACKS_DESCRIPTION)] = None
    max_instances: Annotated[int | None, Declared(_MAX_INSTANCES_DESCRIPTION)] = None
    peak_threshold: Annotated[float, Declared(_PEAK_THRESHOLD_DESCRIPTION)] = 0.2
    analysis_range: Annotated[
        tuple[int, int] | None, Declared(_ANALYSIS_RANGE_DESCRIPTION)
    ] = None
    # JsonValue rather than object, so an unrepresentable value is rejected at
    # params construction (where pydantic names the field) instead of deep inside
    # hash_params. Every representable value still validates and none changes the
    # digest.
    sleap_extra_settings: Annotated[
        dict[str, JsonValue] | None, Declared(_SLEAP_EXTRA_SETTINGS_DESCRIPTION)
    ] = None
    # execution knobs -- throughput/environment only, excluded from the run_id.
    batch_size: Annotated[int, HASH_EXCLUDE, Declared(_BATCH_SIZE_DESCRIPTION)] = 4
    # cpu / cuda / mps / a CUDA index / None (auto). Where it ran, not what it
    # produced.
    device: Annotated[
        str | None,
        HASH_EXCLUDE,
        Field(examples=["cpu", "cuda", "mps", "auto", "0"]),
        Declared(_DEVICE_DESCRIPTION),
    ] = None

    @field_validator("device", check_fields=False)
    @classmethod
    def _device_is_usable(cls, value: str | None) -> str | None:
        """Refuse a device sleap-nn cannot be given, at submit time.

        :func:`~mosaic.tracking.sleap.run.sleap_track_device_args` is the
        translation, and calling it here refuses an unusable spelling before the
        job is scheduled rather than on a GPU node once it runs. ``check_fields``
        is off so that a consumer that sets the device itself can declare a
        subclass without the field, as the training ops' validators allow.
        """
        _ = sleap_track_device_args(value)
        return value

    @model_validator(mode="after")
    def _max_tracks_has_local_queues(self) -> Self:
        """Refuse a track cap under candidates that cannot enforce it.

        sleap-nn enforces ``--max_tracks`` only through ``local_queues``
        candidates. Under ``fixed_window``, earlier releases discard the cap and
        later ones switch to ``local_queues`` themselves, so the same params
        would name two tracker configurations depending on the installed
        release, and neither is the one they state.
        """
        if self.max_tracks is not None and self.candidates_method != "local_queues":
            msg = (
                f"max_tracks={self.max_tracks} needs "
                "candidates_method='local_queues', the only candidates sleap-nn "
                f"enforces a track cap through; got {self.candidates_method!r}."
            )
            raise ValueError(msg)
        return self


SLEAP_MODEL: Final = ModelReference(
    field="model_paths", kinds=(TRAIN_SLEAP_KIND,), many=True
)
"""The model SLEAP runs, as ``SleapOp`` declares it.

Declared here rather than in the op module because
:func:`~mosaic.tracking.sleap.dataset_runs.run_sleap` reads it too, and cannot
import the op.
"""
