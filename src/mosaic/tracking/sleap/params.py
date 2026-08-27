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

from typing import Annotated

from pydantic import Field

from mosaic.core.pipeline.types import JsonValue
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)
from mosaic.tracking.common.params import TrackerOpParams

__all__ = ["SleapParams"]

_MODEL_PATHS_DESCRIPTION = (
    "One trained SLEAP model directory, or two for a top-down model "
    "(centroid, then centered-instance)."
)

_TRACKING_DESCRIPTION = (
    "Assign identities to detections across frames. When False, no tracker is attached."
)

_TRACKER_DESCRIPTION = (
    "The tracker algorithm. Known values are simple, flow, simplemaxtracks and "
    "flowmaxtracks."
)

_SIMILARITY_DESCRIPTION = (
    "The similarity metric for matching detections across frames, for example "
    "instance, centroid, or iou."
)

_MATCH_DESCRIPTION = (
    "The assignment algorithm for matching detections across frames. Known "
    "values are hungarian and greedy."
)

_ANALYSIS_RANGE_DESCRIPTION = (
    "The first and last frame to analyze. Unset, SLEAP analyzes the whole video."
)

_TRACK_WINDOW_DESCRIPTION = "The candidate window for track matching."

_MAX_INSTANCES_DESCRIPTION = "The maximum number of instances to detect per frame."

_MAX_TRACKING_DESCRIPTION = (
    "The maximum number of tracks to maintain. Requires a tracker whose name "
    "ends in maxtracks."
)

_PEAK_THRESHOLD_DESCRIPTION = "The minimum confidence for a detected peak."

_SLEAP_EXTRA_SETTINGS_DESCRIPTION = (
    "Additional sleap-track flags, sent as --key value pairs. A boolean value "
    "becomes a bare --key flag when true and is omitted when false, and a None "
    "value is skipped."
)

_BATCH_SIZE_DESCRIPTION = "The inference batch size."

_DEVICE_DESCRIPTION = (
    "The device to run inference on: cpu, or a GPU index. Unset, cuda and auto "
    "all leave the choice to SLEAP."
)


class SleapParams(TrackerOpParams):
    """Parameters for the ``sleap`` tracking op and for ``run_sleap``."""

    # model: one external model directory, or two for top-down (centroid, then
    # centered-instance). Part of the run_id identity -- via a content digest of
    # the weights, never the paths themselves.
    model_paths: Annotated[list[str], Declared(_MODEL_PATHS_DESCRIPTION)]
    # tracking (part of the run_id identity)
    tracking: Annotated[bool, Declared(_TRACKING_DESCRIPTION)] = True
    tracker: Annotated[
        str,
        Field(examples=["simple", "flow", "simplemaxtracks", "flowmaxtracks"]),
        Declared(_TRACKER_DESCRIPTION),
    ] = "flow"
    similarity: Annotated[
        str,
        Field(examples=["instance", "centroid", "iou"]),
        Declared(_SIMILARITY_DESCRIPTION),
    ] = "instance"
    match: Annotated[
        str,
        Field(examples=["hungarian", "greedy"]),
        Declared(_MATCH_DESCRIPTION),
    ] = "hungarian"
    track_window: Annotated[int, Declared(_TRACK_WINDOW_DESCRIPTION, unit="frames")] = 5
    max_instances: Annotated[int | None, Declared(_MAX_INSTANCES_DESCRIPTION)] = None
    max_tracking: Annotated[int | None, Declared(_MAX_TRACKING_DESCRIPTION)] = None
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
    # cpu / cuda / a gpu index / None (auto). Where it ran, not what it produced.
    device: Annotated[
        str | None,
        HASH_EXCLUDE,
        Field(examples=["cpu", "cuda", "auto", "0"]),
        Declared(_DEVICE_DESCRIPTION),
    ] = None
