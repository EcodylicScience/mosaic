"""SLEAP training as an op -- ``mosaic run --kind train-sleap``.

The first training op whose artifact is a *directory*, which is the whole point
of the model-reference work underneath it: what it registers can be handed back
to the SLEAP tracker as a run identifier, and resolves to the same shape an
externally-trained model does.

**Not a tracker op.** ``TrackerOpParams`` scopes over media entries and bridges
into ``tracks/``; a training run consumes one labels file and produces one model,
so this subclasses ``Params`` like the other training ops and mints through
``train_run_id``. ``mint_tracker_run`` would additionally write a tracks variant
naming a table that does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import Field

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
from mosaic.tracking.ops._train_descriptions import (
    BASE_MODEL_DESCRIPTION,
    EPOCHS_DESCRIPTION,
    IDLE_TIMEOUT_DESCRIPTION,
    MAX_RUNTIME_DESCRIPTION,
)
from mosaic.tracking.sleap.training import SleapBackbone, SleapHead
from mosaic.tracking.sleap.version import TRAIN_SLEAP_KIND

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext

TRAIN_SLEAP_VERSION: str = "0.1"

_LABELS_DESCRIPTION = "The .slp file to train on."

_HEAD_DESCRIPTION = (
    "Which task the network is trained for. centroid and centered_instance "
    "are the two halves of a top-down model, trained separately and passed "
    "to inference as a pair. The multi_class_ heads add identity "
    "classification."
)

_BACKBONE_DESCRIPTION = "The feature extractor architecture, independent of the head."

_SEED_DESCRIPTION = "Seeds sleap-nn's initialization."

_VALIDATION_FRACTION_DESCRIPTION = (
    "Fraction of labels held out for validation, when no separate "
    "validation file is given."
)

_SLEAP_OVERRIDES_DESCRIPTION = (
    "Hydra key=value overrides applied over the generated config, for "
    "anything sleap-nn exposes with no field here. A key set here wins "
    "over base_model and device where they would set the same key."
)

_DEVICE_DESCRIPTION = (
    "Which accelerator trains the model, forwarded to sleap-nn as "
    "trainer_accelerator. auto leaves the choice to sleap-nn."
)


class TrainSleapParams(Params):
    """Parameters for the ``train-sleap`` op.

    The typed fields are the decisions mosaic has an opinion about. Everything
    else sleap-nn exposes is reachable through *sleap_overrides*, which reaches
    identity for the same reason the typed fields do: a model trained with a
    different learning rate is a different model whether or not there is a field
    for it here.
    """

    labels: Annotated[str, Declared(_LABELS_DESCRIPTION)]
    base_model: Annotated[str, Declared(BASE_MODEL_DESCRIPTION)] = ""
    head: Annotated[SleapHead, Declared(_HEAD_DESCRIPTION)] = "centered_instance"
    backbone: Annotated[SleapBackbone, Declared(_BACKBONE_DESCRIPTION)] = "unet"
    max_epochs: Annotated[int, Declared(EPOCHS_DESCRIPTION, unit="epochs")] = 200
    seed: Annotated[int, Declared(_SEED_DESCRIPTION)] = 42
    validation_fraction: Annotated[
        float, Declared(_VALIDATION_FRACTION_DESCRIPTION)
    ] = 0.1
    sleap_overrides: Annotated[
        dict[str, JsonValue] | None, Declared(_SLEAP_OVERRIDES_DESCRIPTION)
    ] = None
    # Execution knobs: where and how fast, not what was trained.
    device: Annotated[
        str,
        HASH_EXCLUDE,
        Field(examples=["auto", "cpu", "gpu", "0"]),
        Declared(_DEVICE_DESCRIPTION),
    ] = "auto"
    idle_timeout: Annotated[
        float, HASH_EXCLUDE, Declared(IDLE_TIMEOUT_DESCRIPTION, unit="s")
    ] = 1800
    max_runtime: Annotated[
        float | None, HASH_EXCLUDE, Declared(MAX_RUNTIME_DESCRIPTION, unit="s")
    ] = None


@register_op
class TrainSleapOp(Op[TrainSleapParams]):
    """Train a SLEAP model, registering the directory it produces."""

    kind = TRAIN_SLEAP_KIND
    category = "train"
    domain = "tracking"
    version = TRAIN_SLEAP_VERSION
    scope_takes = "none"
    scope_dependent = False
    Params = TrainSleapParams
    resource_class: ClassVar[str] = "gpu"

    def target(self, params: TrainSleapParams, scope: ResolvedScope) -> str:
        return f"sleap-train-{params.head}"

    def plan_identity(
        self,
        ds: Dataset,
        params: TrainSleapParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this run, and the model it produces, will be called.

        *require_data* separates planning from execution; see
        :func:`~mosaic.tracking.ops.train.planned_train_identity`.
        """
        from mosaic.tracking.ops.train import planned_train_identity

        return planned_train_identity(
            ds,
            kind=self.kind,
            version=self.version,
            params=params,
            data_path=Path(ds.resolve_path(params.labels)),
            fingerprint=fingerprint_dataset,
            base_model=params.base_model,
            require_data=require_data,
        )

    def run(
        self,
        ds: Dataset,
        params: TrainSleapParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        from mosaic.tracking.sleap.training import train_sleap

        ensure_models_root(ds)
        labels_path = Path(ds.resolve_path(params.labels))

        base_run_id = ""
        base_digest = ""
        resume_from = ""
        if params.base_model:
            base = resolve_model(ds, params.base_model, self.kind)
            base_run_id = base.model_id
            base_digest = base.digest
            weights = base.artifacts[0].file_for("weights")
            resume_from = str(weights) if weights is not None else ""

        run_id = self.plan_identity(ds, params, scope, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            ctx.cache_hit()
            return run_id
        ctx.set_total(params.max_epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, run_root, self.kind, params.idle_timeout)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        overrides: dict[str, JsonValue] = dict(params.sleap_overrides or {})
        if resume_from:
            overrides.setdefault("trainer_config.resume_ckpt_path", resume_from)
        if params.device != "auto":
            overrides.setdefault("trainer_config.trainer_accelerator", params.device)

        produced = train_sleap(
            labels_path,
            run_root,
            head=params.head,
            backbone=params.backbone,
            max_epochs=params.max_epochs,
            seed=params.seed,
            validation_fraction=params.validation_fraction,
            overrides=overrides,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            cancel_check=ctx.cancel_token.is_cancelled if ctx.cancel_token else None,
        )
        ctx.check_cancel()

        # Read the head back off the artifact rather than echoing the parameter.
        # It is what the directory says it is, and resolving it here is also the
        # check that training produced something loadable before a row claims so.
        resolved = resolve_model_set(None, [str(produced)], "sleap")
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
            run_root / "config.yaml",
            params.max_epochs,
            artifact_shape="directory",
            artifact_path=produced,
            model_type=resolved.model_type,
        )
        return run_id
