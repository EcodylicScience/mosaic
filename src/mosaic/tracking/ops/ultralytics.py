"""Ultralytics tracking as a tracking op -- ``mosaic track ultralytics``.

Wraps :func:`mosaic.tracking.ultralytics_track.run_ultralytics` as a registered
``Op``, so it rides the schema-driven runner and every execution backend with
Pydantic parameter validation and discovery, exactly as the other three trackers
do.

:class:`~mosaic.tracking.ultralytics_track.params.UltralyticsParams` is declared
beside the integration, which both this op and ``run_ultralytics`` read, so the
adapter and the tool cannot drift into naming one run two ways.

``resource_class = "gpu"`` because detection is a network forward pass per frame;
the ``"convert"`` category would otherwise derive ``cpu`` and the router would
send a multi-hour run to the wrong lane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from mosaic.tracking.ultralytics_track.version import (
    ULTRALYTICS_KIND,
    ULTRALYTICS_VERSION,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


@register_op
class UltralyticsOp(Op[UltralyticsParams]):
    """Track scoped videos with a YOLO model, bridging results into ``tracks/``."""

    kind = ULTRALYTICS_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    version = ULTRALYTICS_VERSION
    scope_takes = "any"
    scope_dependent = False
    Params = UltralyticsParams

    def target(self, params: UltralyticsParams) -> str:
        return "ultralytics-track"

    def plan_identity(self, ds: Dataset, params: UltralyticsParams) -> OpIdentity:
        """What an Ultralytics tracking run with these settings will be called.

        The tracker table is resolved in full rather than passed as the
        overrides, matching ``run_ultralytics``: a caller who restates a default
        and one who passes nothing must mint the same identifier.
        """
        from mosaic.core.pipeline.op_identity import parse_op_run_id
        from mosaic.tracking.common.mint import planned_model_id, tracker_identity
        from mosaic.tracking.ultralytics_track.dataset_runs import ultralytics_settings
        from mosaic.tracking.ultralytics_track.tracker_defaults import (
            resolve_tracker_config,
        )

        # Both train-pose and train-points produce runnable weights, and a run id
        # resolves against the index its own training op wrote -- so the kind
        # comes from the reference rather than from this tracker.
        parsed = parse_op_run_id(params.model_path)
        model_kind = parsed.kind if parsed is not None else "train-pose"
        settings = ultralytics_settings(
            params,
            model_id=planned_model_id(ds, self.kind, [params.model_path], model_kind),
            tracker_config=resolve_tracker_config(
                params.tracker, params.tracker_overrides
            ),
        )
        return tracker_identity(self.kind, self.version, settings)

    def run(self, ds: Dataset, params: UltralyticsParams, ctx: JobContext) -> str:
        # Ultralytics and torch stay inside run(), so registration is light.
        from mosaic.tracking.ultralytics_track.dataset_runs import run_ultralytics

        return run_ultralytics(ds, params, ctx=ctx)
