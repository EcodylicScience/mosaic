"""Lightning Pose as a tracking op -- ``mosaic run --kind litpose``.

Wraps :func:`mosaic.tracking.litpose.run_litpose` as a registered ``Op`` so
Lightning Pose rides the schema-driven runner and every execution backend (local
/ rq / k8s) with Pydantic param validation + discovery -- the same one-contract
path TREx and SLEAP use. The implementation is unchanged: ``run_litpose`` still
drives the Lightning Pose Python API in its own environment and hashes its
*internal settings dict* for the ``run_id`` (the op only re-routes the same call
through a ``JobContext``).

:class:`~mosaic.tracking.litpose.params.LitposeParams` is declared beside the
integration, which both this op and ``run_litpose`` read, so the adapter and the
tool cannot drift into naming one run two ways.

Registering this op loads the whole ``mosaic.tracking.litpose`` package, because
the params model and the run function both live in it. Deferring either import
would not change that: the package ``__init__`` imports ``dataset_runs`` and
``run`` at module top, so any path into it loads the same modules.

``resource_class = "gpu"`` because Lightning Pose video inference requires a CUDA
GPU -- its ``category`` of ``"convert"`` would not imply that, so it declares the
class explicitly and the execution router sends it to the GPU lane / k8s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.litpose.dataset_runs import litpose_settings, run_litpose
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.litpose.version import (
    LITPOSE_KIND,
    LITPOSE_VERSION,
    TRAIN_LITPOSE_KIND,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext


@register_op
class LitposeOp(Op[LitposeParams]):
    """Run Lightning Pose inference over scoped videos, bridging results into ``tracks/``."""

    kind = LITPOSE_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    # Read from the integration rather than restated, so the op and the standalone
    # run_litpose cannot drift into naming the same run two ways.
    version = LITPOSE_VERSION
    scope_takes = "any"
    scope_dependent = False
    Params = LitposeParams

    def target(self, params: LitposeParams, scope: ResolvedScope) -> str:
        return "litpose-predict"

    def plan_identity(
        self,
        ds: Dataset,
        params: LitposeParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What a Lightning Pose run with these settings will be called."""
        from mosaic.tracking.common.mint import planned_model_id, tracker_identity

        settings = litpose_settings(
            params,
            model_id=planned_model_id(
                ds, self.kind, [str(params.model_path)], TRAIN_LITPOSE_KIND
            ),
        )
        return tracker_identity(self.kind, self.version, settings)

    def run(
        self,
        ds: Dataset,
        params: LitposeParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        # conda-env / bin are environment (image) concerns, left unset so the
        # runner resolves them from MOSAIC_LITPOSE_CONDA_ENV / _BIN -- the run_id
        # stays independent of *where* it ran.
        return run_litpose(ds, params, ctx=ctx)
