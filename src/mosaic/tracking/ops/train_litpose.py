"""Lightning Pose training as an op -- ``mosaic run --kind train-litpose``.

The mirror of ``train-sleap``, and deliberately the same shape: a project
directory in, a registered model directory out, minted through ``train_run_id``
and resolved back through the ``litpose`` spec so the tracker can name the run.

Where it differs is single-animal. Lightning Pose has no instance axis, which
the label writer enforces rather than this op -- by the time a project directory
exists the question has been answered.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.pipeline.types import HASH_EXCLUDE, JsonValue, Params
from mosaic.tracking.model_refs import resolve_model, resolve_model_set
from mosaic.tracking.ops._common import ensure_models_root, fingerprint_dataset
from mosaic.tracking.ops.train import (
    finalize_training,
    train_run_id,
    training_is_complete,
)
from mosaic.tracking.litpose.templates import default_config_path
from mosaic.tracking.litpose.version import TRAIN_LITPOSE_KIND

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

TRAIN_LITPOSE_VERSION: str = "0.1"

LitposeModelType = Literal[
    "heatmap", "heatmap_mhcrnn", "regression", "heatmap_multiview_transformer"
]
"""Which prediction head. ``heatmap_mhcrnn`` adds temporal context over five
frames; the multiview transformer is for synchronised cameras."""


class TrainLitposeParams(Params):
    """Parameters for the ``train-litpose`` op."""

    # A Lightning Pose project directory, or a prior run to fine-tune from.
    project: str
    # Where the complete Lightning Pose config is, dataset-relative or absolute.
    # A **location**, so it is excluded from the hash and the file's content
    # digest enters the payload instead -- otherwise two different configs at one
    # path mint one identifier, and one config at two paths mints two. See the
    # ``extra`` argument of ``train_run_id``.
    base_config: Annotated[str, HASH_EXCLUDE] = ""
    base_model: str = ""
    model_type: LitposeModelType = "heatmap"
    backbone: str = "resnet50_animal_ap10k"
    max_epochs: int = 300
    litpose_overrides: dict[str, JsonValue] | None = None
    device: Annotated[str, HASH_EXCLUDE] = "auto"
    idle_timeout: Annotated[float, HASH_EXCLUDE] = 1800
    max_runtime: Annotated[float | None, HASH_EXCLUDE] = None
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    """Train again even if this exact run already finished.

    ``HASH_EXCLUDE`` because it is a throughput knob, not a property of the model:
    flipping it must not mint a second identity for the same weights.
    """


@register_op
class TrainLitposeOp(Op[TrainLitposeParams]):
    """Train a Lightning Pose model, registering the directory it produces."""

    kind = TRAIN_LITPOSE_KIND
    category = "train"
    domain = "tracking"
    version = TRAIN_LITPOSE_VERSION
    Params = TrainLitposeParams
    resource_class: ClassVar[str] = "gpu"

    def target(self, params: TrainLitposeParams) -> str:
        return f"litpose-train-{params.model_type}"

    def run(self, ds: Dataset, params: TrainLitposeParams, ctx: JobContext) -> str:
        from mosaic.tracking.litpose.training import train_litpose

        ensure_models_root(ds)
        project = Path(ds.resolve_path(params.project))
        # Named or vendored, resolved once: the digest below and the trainer must
        # read the same file, and the identity is over its contents.
        base_config = (
            Path(ds.resolve_path(params.base_config))
            if params.base_config
            else default_config_path()
        )

        base_run_id = ""
        base_digest = ""
        overrides: dict[str, JsonValue] = dict(params.litpose_overrides or {})
        if params.base_model:
            base = resolve_model(ds, params.base_model, self.kind)
            base_run_id = base.model_id
            base_digest = base.digest
            weights = base.artifacts[0].file_for("weights")
            if weights is not None:
                overrides.setdefault("model.checkpoint", str(weights))

        run_id = train_run_id(
            self.kind,
            self.version,
            params,
            fingerprint_dataset(project),
            base_run_id,
            extra={"config": file_digest(base_config)},
        )
        ctx.set_run_id(run_id)
        if not params.overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            return run_id
        ctx.set_total(params.max_epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        produced = train_litpose(
            project,
            run_root,
            base_config=base_config,
            model_type=params.model_type,
            backbone=params.backbone,
            max_epochs=params.max_epochs,
            overrides=overrides,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            cancel_check=ctx.cancel_token.is_cancelled if ctx.cancel_token else None,
        )
        ctx.check_cancel()

        # Read the model type back off the artifact rather than echoing the
        # request, and let resolution be the check that it is loadable.
        resolved = resolve_model_set(None, [str(produced)], "litpose")
        weights = resolved.artifacts[0].file_for("weights")

        finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            base_digest,
            weights if weights is not None else produced,
            produced / "config.yaml",
            params.max_epochs,
            artifact_shape="directory",
            artifact_path=produced,
            model_type=resolved.model_type,
        )
        return run_id
