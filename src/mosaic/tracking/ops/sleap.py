"""SLEAP as a tracking op -- ``mosaic run --kind sleap``.

Wraps :func:`mosaic.tracking.sleap.run_sleap` as a registered ``Op`` so SLEAP rides
the schema-driven runner and every execution backend (local / rq / k8s) with
Pydantic param validation + discovery -- the same one-contract path TREx uses. The
implementation is unchanged: ``run_sleap`` still shells out to the ``sleap-track`` /
``sleap-convert`` console scripts in their own environment and hashes its *internal
settings dict* for the ``run_id`` (the op only re-routes the same call through a
``JobContext``).

:class:`~mosaic.tracking.sleap.params.SleapParams` is declared beside the
integration, which both this op and ``run_sleap`` read, so the adapter and the
tool cannot drift into naming one run two ways.

Registering this op loads the whole ``mosaic.tracking.sleap`` package, because the
params model and the run function both live in it. Deferring either import would
not change that: the package ``__init__`` imports ``dataset_runs`` and ``run`` at
module top, so any path into it loads the same modules.

``resource_class = "gpu"`` because SLEAP inference wants the GPU -- its ``category``
of ``"convert"`` would not imply that, so it declares the class explicitly and the
execution router sends it to the GPU lane / k8s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.sleap.dataset_runs import run_sleap, sleap_settings
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.sleap.version import (
    SLEAP_KIND,
    SLEAP_VERSION,
    TRAIN_SLEAP_KIND,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


@register_op
class SleapOp(Op[SleapParams]):
    """Run SLEAP (infer + track) over scoped videos, bridging results into ``tracks/``."""

    kind = SLEAP_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    # Read from the integration rather than restated, so the op and the standalone
    # run_sleap cannot drift into naming the same run two ways.
    version = SLEAP_VERSION
    scope_takes = "any"
    scope_dependent = False
    Params = SleapParams

    def target(self, params: SleapParams) -> str:
        return "sleap-track"

    def plan_identity(self, ds: Dataset, params: SleapParams) -> OpIdentity:
        """What a SLEAP run with these settings will be called.

        The model set resolves under the *training* kind rather than this
        tracker's, matching ``run_sleap``: a registered reference resolves
        against the index the training op wrote, and ``MODEL_KINDS`` declares
        ``train-sleap`` as SLEAP's own artifact shape for exactly this.
        """
        from mosaic.tracking.common.mint import planned_model_id, tracker_identity

        settings = sleap_settings(
            params,
            model_id=planned_model_id(
                ds, self.kind, list(params.model_paths), TRAIN_SLEAP_KIND
            ),
        )
        return tracker_identity(self.kind, self.version, settings)

    def run(self, ds: Dataset, params: SleapParams, ctx: JobContext) -> str:
        # conda-env / bin are environment (image) concerns, left unset so the
        # runner resolves them from MOSAIC_SLEAP_CONDA_ENV / _BIN -- the run_id
        # stays independent of *where* it ran.
        return run_sleap(ds, params, ctx=ctx)
