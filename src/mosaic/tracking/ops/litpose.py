"""Lightning Pose as a tracking op -- ``mosaic run --kind litpose``.

Wraps :func:`mosaic.tracking.litpose.run_litpose` as a registered ``Op`` so
Lightning Pose rides the schema-driven runner and every execution backend (local
/ rq / k8s) with Pydantic param validation + discovery -- the same one-contract
path TREx and SLEAP use. The implementation is unchanged: ``run_litpose`` still
drives the Lightning Pose Python API in its own environment and hashes its
*internal settings dict* for the ``run_id`` (the op only re-routes the same call
through a ``JobContext``).

``resource_class = "gpu"`` because Lightning Pose video inference requires a CUDA
GPU -- its ``category`` of ``"convert"`` would not imply that, so it declares the
class explicitly and the execution router sends it to the GPU lane / k8s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from mosaic.core.helpers import parse_entry_tokens
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.pipeline.types import HASH_EXCLUDE, JsonValue, Params
from mosaic.tracking.litpose.version import LITPOSE_KIND, LITPOSE_VERSION

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


class LitposeParams(Params):
    """Parameters for the ``litpose`` tracking op (mirrors :func:`run_litpose`'s settings + scope)."""

    # scope (empty -> all indexed media)
    groups: list[str] | None = None
    sequences: list[str] | None = None
    # "group:sequence" pairs (":seq" or "seq" == empty group)
    entries: list[str] | None = None
    # model: one external Lightning Pose model directory (config.yaml plus a
    # checkpoint under tb_logs/). Part of the run_id identity -- via a content
    # digest of the weights + config, never the path itself.
    model_path: str
    # Hydra config overrides applied at inference time. Identity, because they
    # change the produced keypoints. JsonValue rather than object, so an
    # unrepresentable value is rejected at params construction.
    litpose_overrides: dict[str, JsonValue] | None = None
    # execution knobs -- throughput/environment only, excluded from the run_id.
    # fp16/bf16 change the forward pass numerically but are a "how it ran" choice,
    # like SLEAP's device, so precision is excluded from identity.
    precision: Annotated[str, HASH_EXCLUDE] = "fp32"
    convert_to_tracks: Annotated[bool, HASH_EXCLUDE] = True
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    # Inactivity (hang) watchdog: kill a phase after this many seconds with no
    # Lightning Pose output. max_runtime is an optional absolute ceiling (None ->
    # the queue owns it).
    idle_timeout: Annotated[float, HASH_EXCLUDE] = 900
    max_runtime: Annotated[float | None, HASH_EXCLUDE] = None


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
    Params = LitposeParams

    def target(self, params: LitposeParams) -> str:
        return "litpose-predict"

    def run(self, ds: Dataset, params: LitposeParams, ctx: JobContext) -> str:
        # Heavy Lightning Pose imports (subprocess) stay inside run() so registration is light.
        from mosaic.tracking.litpose.dataset_runs import run_litpose

        entry_pairs = parse_entry_tokens(params.entries)
        return run_litpose(
            ds,
            ctx=ctx,  # run within the op's Job Contract -- no double-wrapping
            model_path=params.model_path,
            groups=params.groups,
            sequences=params.sequences,
            entries=entry_pairs or None,
            litpose_overrides=params.litpose_overrides,
            precision=params.precision,
            convert_to_tracks=params.convert_to_tracks,
            overwrite=params.overwrite,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            # conda-env / bin are environment (image) concerns, left unset so the
            # runner resolves them from MOSAIC_LITPOSE_CONDA_ENV / _BIN -- the
            # run_id stays independent of *where* it ran. ``precision`` is passed
            # but is HASH_EXCLUDE, so it likewise never enters the run_id.
        )
