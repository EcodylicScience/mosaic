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

from pydantic import Field

from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.types import JsonValue
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)
from mosaic.tracking.model_refs import resolve_model, resolve_model_set
from mosaic.tracking.ops._common import (
    claim_run_root,
    ensure_models_root,
    fingerprint_dataset,
)
from mosaic.tracking.ops.train import (
    finalize_training,
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
frames; the multiview transformer is for synchronized cameras."""


_PROJECT_DESCRIPTION = (
    "A Lightning Pose project directory, written with the labeled data to "
    "train on. Its config.yaml supplies only the data half of the training "
    "configuration."
)

_BASE_CONFIG_DESCRIPTION = (
    "The complete Lightning Pose config to train from, dataset-relative or "
    "absolute. Unset uses the template config included with mosaic. The "
    "project's own config.yaml supplies the data half and is merged over "
    "it."
)

_BASE_MODEL_DESCRIPTION = (
    "Weights to fine-tune from, as a path or as the run id of the training "
    "op that produced them. Identity records the training run id when the "
    "reference is one, and the weights' content digest when it is a bare "
    "path."
)

_MODEL_TYPE_DESCRIPTION = (
    "Which prediction head trains. heatmap_mhcrnn adds temporal context "
    "over five frames. The multiview transformer is for synchronized "
    "cameras."
)

_BACKBONE_DESCRIPTION = (
    "The feature extractor. Defaults to a ResNet-50 pretrained on animal "
    "pose rather than ImageNet."
)

_EPOCHS_DESCRIPTION = "How long the model trains at most."

_LITPOSE_OVERRIDES_DESCRIPTION = (
    "Hydra key=value overrides applied last, over model_type, backbone and "
    "max_epochs as well as anything else Lightning Pose exposes with no "
    "field here. A key set here wins over base_model where they would set "
    "the same key."
)

_DEVICE_DESCRIPTION = "The accelerator to train the model on."

_DEVICE_UNWIRED = "the training subprocess never receives it"

_IDLE_TIMEOUT_DESCRIPTION = (
    "How long the training subprocess may go without output before it is "
    "killed. A generous default, because an epoch on a large set is slow "
    "and a watchdog must not mistake slow for dead."
)

_MAX_RUNTIME_DESCRIPTION = (
    "Absolute wall-clock ceiling for the training run. Unset leaves the "
    "ceiling to whatever queue submitted the run, and idle_timeout still "
    "applies."
)

_OVERWRITE_DESCRIPTION = "Train again even if this exact run already finished."


class TrainLitposeParams(Params):
    """Parameters for the ``train-litpose`` op."""

    project: Annotated[str, Declared(_PROJECT_DESCRIPTION)]
    # A location, excluded from the hash. The file's content digest enters
    # the payload instead, distinguishing two configs at one path and
    # unifying one config reachable at two paths. See the extra argument of
    # train_run_id.
    base_config: Annotated[str, HASH_EXCLUDE, Declared(_BASE_CONFIG_DESCRIPTION)] = ""
    base_model: Annotated[str, Declared(_BASE_MODEL_DESCRIPTION)] = ""
    model_type: Annotated[LitposeModelType, Declared(_MODEL_TYPE_DESCRIPTION)] = (
        "heatmap"
    )
    backbone: Annotated[
        str,
        Field(examples=["resnet50_animal_ap10k", "resnet50"]),
        Declared(_BACKBONE_DESCRIPTION),
    ] = "resnet50_animal_ap10k"
    max_epochs: Annotated[int, Declared(_EPOCHS_DESCRIPTION, unit="epochs")] = 300
    litpose_overrides: Annotated[
        dict[str, JsonValue] | None, Declared(_LITPOSE_OVERRIDES_DESCRIPTION)
    ] = None
    device: Annotated[
        str, HASH_EXCLUDE, Declared(_DEVICE_DESCRIPTION, unwired=_DEVICE_UNWIRED)
    ] = "auto"
    idle_timeout: Annotated[
        float, HASH_EXCLUDE, Declared(_IDLE_TIMEOUT_DESCRIPTION, unit="s")
    ] = 1800
    max_runtime: Annotated[
        float | None, HASH_EXCLUDE, Declared(_MAX_RUNTIME_DESCRIPTION, unit="s")
    ] = None
    # A throughput knob, not a property of the model: flipping it must not
    # mint a second identity for the same weights.
    overwrite: Annotated[bool, HASH_EXCLUDE, Declared(_OVERWRITE_DESCRIPTION)] = False


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

    def plan_identity(
        self, ds: Dataset, params: TrainLitposeParams, *, require_data: bool = True
    ) -> OpIdentity:
        """What this run, and the model it produces, will be called.

        The base config enters the identity as a content digest, because two
        projects trained under different configs are different models.
        *require_data* separates planning from execution; see
        :func:`~mosaic.tracking.ops.train.planned_train_identity`.
        """
        from mosaic.tracking.ops.train import planned_train_identity

        base_config = (
            Path(ds.resolve_path(params.base_config))
            if params.base_config
            else default_config_path()
        )
        return planned_train_identity(
            ds,
            kind=self.kind,
            version=self.version,
            params=params,
            data_path=Path(ds.resolve_path(params.project)),
            fingerprint=fingerprint_dataset,
            base_model=params.base_model,
            extra={"config": file_digest(base_config)},
            require_data=require_data,
        )

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

        run_id = self.plan_identity(ds, params, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not params.overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            ctx.cache_hit()
            return run_id
        ctx.set_total(params.max_epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, run_root, self.kind, params.idle_timeout)
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
