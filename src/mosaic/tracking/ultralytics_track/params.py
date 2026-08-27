"""What an Ultralytics tracking run is, declared once for every consumer.

One statement per field serves validation, run identity, invocation and
discovery. ``Field`` states the constraint pydantic enforces and
:class:`~mosaic.core.pipeline.types.Declared` states the prose a client draws a
control from. Ultralytics tracks a video in one pass, in an environment of its
own, so no field names a phase.

The model is declared beside the integration rather than beside the op, because
:func:`~mosaic.tracking.ultralytics_track.dataset_runs.run_ultralytics` and
:func:`~mosaic.tracking.ultralytics_track.dataset_runs.ultralytics_settings`
take it: declared in ``tracking/ops/ultralytics.py``, the integration would
import its own adapter. ``tracking/ultralytics_track/version.py`` is declared
here for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.types import HASH_EXCLUDE, Declared, Probability
from mosaic.tracking.common.params import TrackerOpParams
from mosaic.tracking.ultralytics_track.run import ModelTask
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TRACKER_NAMES,
    TRACKER_TYPE_KEY,
    TrackerConfig,
    TrackerName,
    resolve_tracker_config,
    validate_tracker_overrides,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["UltralyticsParams"]

_MODEL_PATH_DESCRIPTION = (
    "The YOLO weights to track with: a .pt path, or the run id of the training "
    "op that produced them. Identity records the weights' content digest when "
    "the reference is a bare path, and the training run id when it is one."
)

_TASK_DESCRIPTION = (
    "Which head the weights carry. Declared rather than read off the file, "
    "because it is part of the identifier and a run must not re-mean itself "
    "when the weights behind a path change. The loaded model is checked "
    "against it and a mismatch is refused by name."
)

_TRACKER_DESCRIPTION = (
    "Which backend associates detections across frames. bytetrack is the "
    "default rather than Ultralytics' own, because botsort, deepocsort and "
    "tracktrack default to optical-flow camera motion compensation whose "
    "RANSAC is unseeded, and a run under one of those is not reproducible bit "
    "for bit. bytetrack constructs no motion estimator. The others stay one "
    "parameter away and are the right choice for a moving camera."
)

_TRACKER_OVERRIDES_DESCRIPTION = (
    "The chosen backend's settings. Every setting that backend has appears "
    "here with its declared default, so a caller states the ones it cares "
    "about and leaves the rest. Identity records the fully resolved table: "
    "restating a default mints the same identifier as passing nothing, and "
    "changing one setting mints a different one."
)

_CONF_DESCRIPTION = (
    "The minimum detection confidence. Ultralytics' track mode replaces a "
    "falsy confidence with 0.1, so its documented predict default of 0.25 "
    "never applies here and 0.0 would be recorded as 0.0 and run as 0.1. The "
    "lower bound refuses that one value."
)

_IOU_DESCRIPTION = (
    "The NMS IoU threshold for suppressing overlapping detections before they "
    "reach the tracker."
)

_IMGSZ_DESCRIPTION = (
    "The side each frame is letterboxed to for inference. Predictions are "
    "mapped back to source pixels, so this changes what the detector sees "
    "rather than what the coordinates mean."
)

_MAX_DET_DESCRIPTION = "The most detections kept from one frame."

_CLASSES_DESCRIPTION = (
    "Which detector class indexes to keep. Unset keeps every class. Identity "
    "records the set sorted and de-duplicated, so the order a filter was typed "
    "in does not move the identifier."
)

_AGNOSTIC_NMS_DESCRIPTION = (
    "Suppress overlapping detections across classes instead of within each one."
)

_START_FRAME_DESCRIPTION = (
    "The first frame index tracked. It is part of identity for a stronger "
    "reason than in the inference ops: a tracker is stateful, so a different "
    "starting frame gives different identities rather than a subset of the "
    "same ones."
)

_END_FRAME_DESCRIPTION = (
    "The frame index tracking stops before. Unset tracks to the end of the video."
)

_FRAME_STEP_DESCRIPTION = (
    "Track every nth frame. A step above 1 changes what the tracker sees "
    "between observations, so it changes the identities as well as the count."
)

_DEVICE_DESCRIPTION = (
    "Which CUDA device index, or cpu, runs inference. A property of the "
    "machine rather than of the result, so it stays out of identity."
)

_PRECISION_DESCRIPTION = (
    "The numeric precision inference runs at. fp16 halves the memory a batch "
    "needs and can move a detection across a threshold."
)

_BATCH_SIZE_DESCRIPTION = (
    "How many frames are decoded and handed to the model per forward pass. "
    "Tracking still advances one frame at a time within a batch."
)

_PREFETCH_DESCRIPTION = (
    "Decode the next batch on a background thread while the current one runs."
)


def _selected_backend(value: object) -> TrackerName | None:
    """*value* as a backend name, or ``None`` where it names none of them.

    ``tracker`` validates before ``tracker_overrides``, so a value that is not
    a backend name is refused by its own field rather than reported as an
    override that could not be tagged.
    """
    return next((name for name in TRACKER_NAMES if name == value), None)


class UltralyticsParams(TrackerOpParams):
    """Parameters for the ``ultralytics`` tracking op and for ``run_ultralytics``.

    ``tracker_overrides`` takes the settings of whichever backend ``tracker``
    names, and only those. Its tag is supplied from ``tracker`` before
    validation, so a caller never writes one, and a tag disagreeing with
    ``tracker`` is refused rather than silently reassigned.

    Every tool-facing field states a concrete default, which is the value the
    installed Ultralytics release applies. ``conf`` is the one that differs from
    the documented default, for the reason its description gives.
    """

    # tracker_overrides is a tagged union over the six backend configurations,
    # so the discovery schema states the settings the selected backend takes
    # with their types, defaults, choices and prose, and a client reads tracker
    # to pick the branch it draws. The tag is tracker_type, which is also the
    # first key of the resolved table.
    #
    # The defaults are stated rather than deferred to because mosaic never
    # imports Ultralytics and so cannot read one at the call site:
    # tracker_defaults transcribes them and the preflight diffs the
    # transcription against the release the external environment holds.
    model_path: Annotated[str, Declared(_MODEL_PATH_DESCRIPTION)]
    task: Annotated[ModelTask, Declared(_TASK_DESCRIPTION)] = "pose"
    tracker: Annotated[TrackerName, Declared(_TRACKER_DESCRIPTION)] = "bytetrack"
    tracker_overrides: Annotated[
        TrackerConfig | None, Declared(_TRACKER_OVERRIDES_DESCRIPTION)
    ] = None
    conf: Annotated[float, Field(gt=0.0, le=1.0), Declared(_CONF_DESCRIPTION)] = 0.1
    iou: Annotated[Probability, Declared(_IOU_DESCRIPTION)] = 0.7
    imgsz: Annotated[int, Field(gt=0), Declared(_IMGSZ_DESCRIPTION, unit="px")] = 640
    max_det: Annotated[int, Field(gt=0), Declared(_MAX_DET_DESCRIPTION)] = 300
    classes: Annotated[list[int] | None, Declared(_CLASSES_DESCRIPTION)] = None
    agnostic_nms: Annotated[bool, Declared(_AGNOSTIC_NMS_DESCRIPTION)] = False
    start_frame: Annotated[int, Field(ge=0), Declared(_START_FRAME_DESCRIPTION)] = 0
    end_frame: Annotated[int | None, Field(ge=0), Declared(_END_FRAME_DESCRIPTION)] = (
        None
    )
    frame_step: Annotated[int, Field(ge=1), Declared(_FRAME_STEP_DESCRIPTION)] = 1

    # Execution: where and how it ran, never what it produced.
    device: Annotated[str, HASH_EXCLUDE, Declared(_DEVICE_DESCRIPTION)] = "0"
    precision: Annotated[
        Literal["fp32", "fp16"], HASH_EXCLUDE, Declared(_PRECISION_DESCRIPTION)
    ] = "fp32"
    batch_size: Annotated[
        int, Field(ge=1), HASH_EXCLUDE, Declared(_BATCH_SIZE_DESCRIPTION)
    ] = 8
    prefetch: Annotated[bool, HASH_EXCLUDE, Declared(_PREFETCH_DESCRIPTION)] = True

    @field_validator("tracker_overrides", mode="before")
    @classmethod
    def _configure_the_backend_that_tracker_names(
        cls,
        value: Mapping[str, JsonValue] | TrackerConfig | None,
        info: ValidationInfo,
    ) -> Mapping[str, JsonValue] | TrackerConfig | None:
        """Turn a mapping of overrides into the configuration ``tracker`` selects.

        A tagged union reads its tag from the input, so a mapping without one
        selects no branch. The backend is named once, by ``tracker``, and the
        tag is filled in from it here. Routing the mapping through
        :func:`~mosaic.tracking.ultralytics_track.tracker_defaults.validate_tracker_overrides`
        is also what keeps the refusals specific: it names the setting, the
        backends that do take it, and why a closed setting is closed, where the
        union alone reports an extra key.

        Anything that is not a mapping is returned for the union to validate,
        which covers ``None``, an already-built configuration model, and a
        value of the wrong type entirely.
        """
        if not isinstance(value, Mapping):
            return value
        backend = _selected_backend(info.data.get("tracker"))
        if backend is None:
            return value
        return validate_tracker_overrides(backend, _untagged(backend, value))

    @model_validator(mode="after")
    def _overrides_configure_the_chosen_backend(self) -> Self:
        """Refuse a configuration built for a backend other than ``tracker``.

        A caller who builds the configuration model itself names the backend
        twice, and the resolved table would otherwise describe a tracker the
        identifier does not name.
        """
        _ = resolve_tracker_config(self.tracker, self.tracker_overrides)
        return self


def _refuse_misordered_tag_fields(model: type[UltralyticsParams]) -> None:
    """Refuse *model* unless ``tracker`` validates before ``tracker_overrides``.

    ``info.data`` in a field validator contains the fields validated so far, so
    the tag is only fillable from ``tracker`` while ``tracker`` is declared
    first. Swapping the two declarations refuses every mapping-form override
    for selecting no branch, and refuses it at run time rather than here.

    Checked at import for this class alone. Pydantic keeps an inherited field
    in the position the base declared it, whatever a subclass redeclares, so
    the order is a property of this class body.
    """
    order = list(model.model_fields)
    if order.index("tracker") > order.index("tracker_overrides"):
        refused = (
            f"{model.__name__} validates tracker_overrides before tracker, so "
            f"an override given as a mapping cannot be tagged from tracker. "
            f"Declare tracker ahead of it."
        )
        raise TypeError(refused)


_refuse_misordered_tag_fields(UltralyticsParams)


def _untagged(
    tracker: TrackerName, overrides: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """*overrides* without its tag, refusing one that names another backend.

    ``tracker_type`` selects the backend, so a caller may not use it to point
    the configuration at a different one than ``tracker`` names.
    """
    named = overrides.get(TRACKER_TYPE_KEY, tracker)
    if named != tracker:
        refused = (
            f"{TRACKER_TYPE_KEY!r} selects the backend and cannot be "
            f"overridden; pass tracker={named!r} instead of overriding it on a "
            f"{tracker} run."
        )
        raise ValueError(refused)
    return {key: value for key, value in overrides.items() if key != TRACKER_TYPE_KEY}
