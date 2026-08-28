"""TREx as a tracking op -- ``mosaic run --kind trex``.

Wraps :func:`mosaic.tracking.trex.run_trex` as a registered ``Op`` so TREx rides the
schema-driven runner and every execution backend (local / rq / k8s) with Pydantic param
validation + discovery -- the same one-contract path SLEAP / DeepLabCut follow. ``run_trex``
shells out to the ``trex`` binary in its conda env and hashes the resolved settings for the
``run_id``, and this module hands it the same parameters through a ``JobContext``.

:class:`~mosaic.tracking.trex.params.TrexParams` is declared beside the integration, which
both this op and ``run_trex`` read, so the adapter and the tool cannot drift into naming one
run two ways.

Registering this op loads the whole ``mosaic.tracking.trex`` package, because the params
model and the run function both live in it. Deferring either import would not change that:
the package ``__init__`` imports ``dataset_runs`` and ``run`` at module top, so any path into
it loads the same modules.

``resource_class = "gpu"`` because TREx needs the GPU for YOLO detection -- its ``category``
of ``"convert"`` would not imply that, so it declares the class explicitly, and the execution
router then sends it to the GPU lane / k8s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.trex.dataset_runs import run_trex, trex_settings
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext


@register_op
class TrexOp(Op[TrexParams]):
    """Run TRex (convert + track) over scoped videos, bridging results into ``tracks/``."""

    kind = TREX_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    # Read from the integration rather than restated, so the op and the
    # standalone run_trex cannot drift into naming the same run two ways.
    version = TREX_VERSION
    scope_takes = "any"
    scope_dependent = False
    Params = TrexParams

    def target(self, params: TrexParams, scope: ResolvedScope) -> str:
        return "trex-track"

    def plan_identity(
        self,
        ds: Dataset,
        params: TrexParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What a TREx run with these settings will be called.

        Both model references are resolved to an identity first, which is the
        same ordering ``run_trex`` follows and for the same reason: what the
        settings must carry is what a model *is*, because a bare weights path is
        a mutable key and swapping the file in place would let two different runs
        share one identifier.
        """
        from mosaic.tracking.common.mint import planned_model_id, tracker_identity

        detect_model_id = (
            planned_model_id(
                ds,
                self.kind,
                [params.detect_model],
                _detect_model_kind(params.detect_model),
            )
            if params.detect_model is not None
            else None
        )
        vi_model_id = (
            planned_model_id(
                ds,
                self.kind,
                [params.visual_identification_model_path],
                "train-identity",
            )
            if params.visual_identification_model_path is not None
            else None
        )
        settings = trex_settings(
            params, detect_model_id=detect_model_id, vi_model_id=vi_model_id
        )
        return tracker_identity(self.kind, self.version, settings)

    def run(
        self,
        ds: Dataset,
        params: TrexParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        # Within the op's Job Contract, so the run is not double-wrapped. Which
        # conda environment, binary and display TREx is launched from is read
        # from MOSAIC_TREX_CONDA_ENV / _BIN / _DISPLAY, which keeps the run_id
        # independent of where it ran.
        return run_trex(ds, params, ctx=ctx)


def _detect_model_kind(ref: str | None) -> str:
    """Which training op's index a detection-model reference resolves against.

    The kind comes from the reference itself, matching ``run_trex``: both
    ``train-pose`` and ``train-points`` produce runnable detection weights, and a
    run identifier resolves against the index its own training op wrote. A
    reference that is not a run identifier at all -- a bare weights path -- falls
    back rather than guessing, because the fallback only decides which spec reads
    the artifact and a path is read the same way either way.
    """
    from mosaic.core.pipeline.op_identity import parse_op_run_id

    parsed = parse_op_run_id(str(ref)) if ref else None
    return parsed.kind if parsed is not None else "train-points"
