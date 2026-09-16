"""SLEAP training as an op -- ``mosaic run --kind train-sleap``.

The first training op whose artifact is a *directory*, which is the whole point
of the model-reference work underneath it: what it registers can be handed back
to the SLEAP tracker as a run identifier, and resolves to the same shape an
externally-trained model does.

**Not a tracker op.** ``TrackerOpParams`` declares the execution knobs a tool
run needs and bridges its output into ``tracks/``. A training run consumes one
labels file and produces one model. This subclasses ``Params`` like the other
training ops and mints through ``train_run_id``. ``mint_tracker_run`` would
additionally write a tracks variant naming a table that does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import Field, field_validator

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
from mosaic.tracking.sleap.probe import (
    probe_sleap,
    require_identity_labels,
    require_sleap_nn,
)
from mosaic.tracking.sleap.training import (
    SleapBackbone,
    SleapHead,
    sleap_device_overrides,
)
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
    "classification, so they need labels that carry it: a .slp whose "
    "instances have no track trains one of them against no classes at all, "
    "which succeeds and produces a model that learned nothing."
)

_BACKBONE_DESCRIPTION = "The feature extractor architecture, independent of the head."

_SEED_DESCRIPTION = "Seeds sleap-nn's initialization."

_VALIDATION_FRACTION_DESCRIPTION = (
    "Fraction of labels held out for validation, when no separate "
    "validation file is given."
)

_SLEAP_OVERRIDES_DESCRIPTION = (
    "Hydra key=value overrides applied over the generated config, for "
    "anything sleap-nn exposes with no field here. A key the generated "
    "config does not carry is appended for you, so it needs no + prefix; a "
    "key written with an explicit + or ~ is passed through as written. A key "
    "set here wins over base_model and device where they would set the same "
    "key."
)

_DEVICE_DESCRIPTION = (
    "Which accelerator trains the model. auto leaves the choice to sleap-nn; "
    "cpu, gpu and mps each name a family; a comma-separated list of CUDA "
    "indices such as 0 or 0,1 names devices within the gpu family."
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
        Field(examples=["auto", "cpu", "gpu", "mps", "0", "0,1"]),
        Declared(_DEVICE_DESCRIPTION),
    ] = "auto"
    idle_timeout: Annotated[
        float, HASH_EXCLUDE, Declared(IDLE_TIMEOUT_DESCRIPTION, unit="s")
    ] = 1800
    max_runtime: Annotated[
        float | None, HASH_EXCLUDE, Declared(MAX_RUNTIME_DESCRIPTION, unit="s")
    ] = None

    @field_validator("device")
    @classmethod
    def _device_is_usable(cls, value: str) -> str:
        """Refuse a device sleap-nn cannot be given, at submit time.

        :func:`~mosaic.tracking.sleap.training.sleap_device_overrides` is the
        translation, and calling it here is what lets mosaic-api answer an
        unusable spelling with a 422 rather than leaving it to fail during
        Hydra config composition on a GPU node once the job is scheduled.
        """
        _ = sleap_device_overrides(value)
        return value


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

        # After the reuse gate and before the root is claimed. A cache hit pays
        # for no cold import, and a refusal is a message rather than a claim
        # left behind on a run that was never going to start. The probe reports
        # and the refusals below are mosaic's, so both are decided here without
        # importing anything the SLEAP environment owns.
        probe = probe_sleap(
            labels_path,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            cancel_check=ctx.cancel_token.is_cancelled if ctx.cancel_token else None,
        )
        require_sleap_nn(probe)
        require_identity_labels(probe, params.head, labels_path)

        ctx.set_total(params.max_epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, run_root, self.kind, params.idle_timeout)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        overrides: dict[str, JsonValue] = dict(params.sleap_overrides or {})
        if resume_from:
            overrides.setdefault("trainer_config.resume_ckpt_path", resume_from)
        for key, value in sleap_device_overrides(params.device).items():
            overrides.setdefault(key, value)

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
            # sleap-nn's CSVLoggerCallback writes this beside the checkpoints,
            # one row per epoch. The config that used to be recorded here is a
            # file that exists, so the index advertised a metrics path holding
            # no metrics, which a reader has to open to discover.
            produced / "training_log.csv",
            params.max_epochs,
            artifact_shape="directory",
            artifact_path=produced,
            model_type=resolved.model_type,
        )
        return run_id
