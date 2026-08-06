"""Ultralytics tracking as a tracking op -- ``mosaic track ultralytics``.

Wraps :func:`mosaic.tracking.ultralytics_track.run_ultralytics` as a registered
``Op``, so it rides the schema-driven runner and every execution backend with
Pydantic parameter validation and discovery, exactly as the other three trackers
do.

``resource_class = "gpu"`` because detection is a network forward pass per frame;
the ``"convert"`` category would otherwise derive ``cpu`` and the router would
send a multi-hour run to the wrong lane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import Field, model_validator

from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.pipeline.types import HASH_EXCLUDE, JsonValue
from mosaic.tracking.common.params import TrackerOpParams
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TrackerName,
    resolve_tracker_config,
)
from mosaic.tracking.ultralytics_track.version import (
    ULTRALYTICS_KIND,
    ULTRALYTICS_VERSION,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


class UltralyticsParams(TrackerOpParams):
    """Parameters for the ``ultralytics`` tracking op.

    The default backend is ``bytetrack`` rather than Ultralytics' own default.
    ``botsort``, ``deepocsort`` and ``tracktrack`` default to optical-flow camera
    motion compensation, whose RANSAC is unseeded -- so a run with one of those
    is not bit-reproducible, which is a poor thing to get by default in a
    pipeline whose identifiers promise that identical inputs give identical
    outputs. ``bytetrack`` constructs no motion estimator at all. The others stay
    one parameter away, and are the right choice for a moving camera.
    """

    # One weights file: a `.pt` path, or the run id of the training op that made
    # it. Identity carries its content digest, never the path.
    model_path: str
    # Declared rather than detected, because it is part of the identifier and a
    # run must not silently re-mean itself when the weights behind a path change.
    # Checked against the loaded model, which refuses a mismatch by name.
    task: Literal["pose", "detect"] = "pose"
    tracker: TrackerName = "bytetrack"
    # Identity, as the fully resolved table: restating a default mints the same
    # identifier as passing nothing, and changing one knob mints a different one.
    tracker_overrides: dict[str, JsonValue] | None = None
    # `Model.track` replaces a falsy confidence with 0.1, so Ultralytics' documented
    # predict default of 0.25 never applies in track mode. Declaring 0.1 makes the
    # default honest, and `gt=0` refuses the one value that would be recorded as
    # 0.0 and executed as 0.1.
    conf: float = Field(default=0.1, gt=0.0)
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 300
    classes: list[int] | None = None
    agnostic_nms: bool = False
    # Identity, and for a stronger reason than in the inference ops: a tracker is
    # stateful, so a different starting frame gives different identities rather
    # than a subset of the same ones.
    start_frame: int = 0
    end_frame: int | None = None
    frame_step: int = 1

    # Execution: where and how it ran, never what it produced.
    device: Annotated[str, HASH_EXCLUDE] = "0"
    precision: Annotated[Literal["fp32", "fp16"], HASH_EXCLUDE] = "fp32"
    batch_size: Annotated[int, HASH_EXCLUDE] = 8
    prefetch: Annotated[bool, HASH_EXCLUDE] = True

    @model_validator(mode="after")
    def _overrides_are_resolvable(self) -> Self:
        """Refuse a bad override here, where pydantic names the field.

        Otherwise the first sign of a typo would be a tracker running with a
        setting it ignored, under an identifier that recorded it.
        """
        _ = resolve_tracker_config(self.tracker, self.tracker_overrides)
        return self


@register_op
class UltralyticsOp(Op[UltralyticsParams]):
    """Track scoped videos with a YOLO model, bridging results into ``tracks/``."""

    kind = ULTRALYTICS_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    version = ULTRALYTICS_VERSION
    Params = UltralyticsParams

    def target(self, params: UltralyticsParams) -> str:
        return "ultralytics-track"

    def run(self, ds: Dataset, params: UltralyticsParams, ctx: JobContext) -> str:
        # Ultralytics and torch stay inside run(), so registration is light.
        from mosaic.tracking.ultralytics_track.dataset_runs import run_ultralytics

        entry_pairs = params.entry_pairs()
        return run_ultralytics(
            ds,
            ctx=ctx,
            model_path=params.model_path,
            groups=params.groups,
            sequences=params.sequences,
            entries=entry_pairs or None,
            task=params.task,
            tracker=params.tracker,
            tracker_overrides=params.tracker_overrides,
            conf=params.conf,
            iou=params.iou,
            imgsz=params.imgsz,
            max_det=params.max_det,
            classes=params.classes,
            agnostic_nms=params.agnostic_nms,
            start_frame=params.start_frame,
            end_frame=params.end_frame,
            frame_step=params.frame_step,
            device=params.device,
            precision=params.precision,
            batch_size=params.batch_size,
            prefetch=params.prefetch,
            convert_to_tracks=params.convert_to_tracks,
            overwrite=params.overwrite,
            idle_timeout=params.idle_timeout,
        )
