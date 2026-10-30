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

import csv
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Self

from pydantic import Field, TypeAdapter, field_validator, model_validator

from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.entry_claim import phase_activity
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, parse_op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.core.pipeline.types import JsonValue
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)
from mosaic.tracking.common.params import DEVICE_INDEX_NOTE
from mosaic.tracking.common.training_progress import epoch_reporter
from mosaic.tracking.model_refs import (
    observed_model_source,
    resolve_model,
    resolve_model_set,
)
from mosaic.tracking.ops._common import (
    claim_run_root,
    ensure_models_root,
    fingerprint_dataset,
    prepared_data_row,
    resolve_training_data,
)
from mosaic.tracking.ops.prepare import PREPARE_KIND
from mosaic.tracking.ops.train import finalize_training
from mosaic.tracking.ops.trained_models import training_is_complete
from mosaic.tracking.ops._train_descriptions import (
    BASE_MODEL_DESCRIPTION,
    EPOCHS_DESCRIPTION,
    IDLE_TIMEOUT_DESCRIPTION,
    LOADER_WORKERS_DESCRIPTION,
    MAX_RUNTIME_DESCRIPTION,
)
from mosaic.tracking.sleap.labels import (
    validation_labels_beside,
    validation_labels_path,
)
from mosaic.tracking.sleap.probe import (
    probe_sleap,
    require_embedding_labels,
    require_identity_labels,
    require_sleap_nn,
)
from mosaic.tracking.sleap.training import (
    DEFAULT_EMBEDDING_OBJECTIVE,
    EMBEDDING_SELECT_METRIC,
    EMBEDDING_SELECT_METRIC_MODES,
    PREPARED_EMBEDDING_OBJECTIVE,
    EmbeddingObjective,
    EmbeddingSettings,
    SleapBackbone,
    SleapBackboneSize,
    SleapHead,
    SleapOptimizer,
    SleapTrainConfig,
    SleapTrainSettings,
    effective_train_config,
    effective_train_settings,
    embedding_settings,
    loader_worker_keys_in,
    sleap_device_overrides,
    sleap_loader_overrides,
    sleap_train_config,
)
from mosaic.tracking.sleap.version import (
    PREPARE_EMBEDDING_DATA_KIND,
    SLEAP_NN_EMBEDDING_MIN,
    TRAIN_SLEAP_KIND,
    require_sleap_nn_at_least,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext

# 0.3 initializes the backbone and the head from base_model's weights, where 0.2
# resumed base_model's run. 0.3 also validates on the validation file beside the
# labels when there is one, such as the labels.val.slp that prepare-training-data
# writes beside labels.slp, in place of holding out validation_fraction of the
# labels. That file enters the data fingerprint. ``train-sleap.0.2-*`` runs stay
# readable and resolve as models and as bases under their own ids.
# 0.2 fingerprints the labels by their own bytes, where 0.1 listed every file
# under their folder and so moved with anything written beside them. That moves
# the digest, so the visible segment moves with it: ``train-sleap.0.1-*`` runs
# stay readable and keep their index rows rather than sitting under a name that
# now means something else.
TRAIN_SLEAP_VERSION: str = "0.3"

_LABELS_DESCRIPTION = "The .slp file to train on."

_BASE_MODEL_DESCRIPTION = BASE_MODEL_DESCRIPTION + (
    " Both the backbone and the head start from these weights. The base must "
    "be a training of the same head."
)

_HEAD_DESCRIPTION = (
    "Which task the network is trained for. centroid and centered_instance "
    "are the two halves of a top-down model, trained separately and passed "
    "to inference as a pair. The multi_class_ heads add identity "
    "classification, so they need labels that carry it: a .slp whose "
    "instances have no track trains one of them against no classes at all, "
    "which succeeds and produces a model that learned nothing. An embedding "
    "model maps each detection's crop to an appearance vector, predicts no "
    "pose, and runs only as the last model of a sleap op model list."
)

_BACKBONE_DESCRIPTION = "The feature extractor architecture, independent of the head."

_SEED_DESCRIPTION = "Seeds sleap-nn's initialization."

_UNSET_NOTE = (
    " Unset, a pose head keeps sleap-nn's default and an embedding head takes "
    "the value that trained the four-mice embedding model"
)

_BATCH_SIZE_DESCRIPTION = (
    "Samples per batch of the training and the validation loader. An "
    "embedding head draws each training batch from the objective's sampler, "
    "groups_per_batch times samples_per_group crops, so this sets its "
    "validation batch." + _UNSET_NOTE + ", 64. At sleap-nn's default of 4 an "
    "embedding run's validation pass is slow."
)

_BACKBONE_SIZE_DESCRIPTION = (
    "The size of a convnext or swint backbone; swint has no large. Refused "
    "with unet." + _UNSET_NOTE + ", tiny."
)

_PRETRAINED_BACKBONE_DESCRIPTION = (
    "Whether a convnext or swint backbone starts from the ImageNet weights of "
    "its size. Refused true with unet. base_model's weights replace them."
    + _UNSET_NOTE
    + ", true."
)

_STEM_STRIDE_DESCRIPTION = (
    "The stride of a convnext or swint stem; its kernel stays at 4, so "
    "ImageNet weights load at any stride. sleap-nn sets the backbone's "
    "max_stride to 16 times it. A stride of 2 fills about "
    "20 GB of GPU memory under an embedding head. Refused with unet."
    + _UNSET_NOTE
    + ", 4."
)

_MAX_STRIDE_DESCRIPTION = (
    "The backbone's max_stride, at which an embedding head pools. On convnext "
    "and swint it must be 16 times the stem stride, which sleap-nn sets anyway."
    + _UNSET_NOTE
    + ": 16 times the stem stride on convnext and swint, 16 on unet."
)

_OPTIMIZER_DESCRIPTION = "The optimizer." + _UNSET_NOTE + ", AdamW."

_LEARNING_RATE_DESCRIPTION = "The optimizer's learning rate." + _UNSET_NOTE + ", 1e-4."

_ENSURE_GRAYSCALE_DESCRIPTION = (
    "Whether the images are converted to one channel." + _UNSET_NOTE + ", false: "
    "color can tell animals apart. Set true for grayscale video. Refused true "
    "with pretrained_backbone on a pose head, which sleap-nn trains on RGB."
)

_CACHE_IMAGES_DESCRIPTION = (
    "Whether sleap-nn holds the decoded images in host memory, which is faster "
    "and costs about 17 GB for 2,880 RGB 1080p frames." + _UNSET_NOTE + ", false."
)

_VALIDATION_FRACTION_DESCRIPTION = (
    "Fraction of labels held out for validation. It applies only when the labels "
    "have no validation file beside them, such as the labels.val.slp that "
    "prepare-training-data writes beside labels.slp."
)

_SLEAP_OVERRIDES_DESCRIPTION = (
    "Hydra key=value overrides applied over the generated config, for "
    "anything sleap-nn exposes with no field here. A key the generated "
    "config does not carry is appended for you, so it needs no + prefix; a "
    "key written with an explicit + or ~ is passed through as written. A key "
    "set here wins over base_model and device where they would set the same "
    "key. The data-loader worker count is refused here: set num_workers, "
    "which leaves the run identity alone. An embedding run sizes its crops "
    "from the labels, and data_config.preprocessing.crop_size sets the size. "
    "sleap-nn raises trainer_config.train_steps_per_epoch to "
    "trainer_config.min_train_steps_per_epoch without a message. A shorter "
    "epoch sets both."
)

_DEVICE_DESCRIPTION = (
    "Which accelerator trains the model. auto leaves the choice to sleap-nn; "
    "cpu, gpu and mps each name a family; a comma-separated list of CUDA "
    "indices such as 0 or 0,1 names devices within the gpu family."
) + DEVICE_INDEX_NOTE

_NUM_WORKERS_DESCRIPTION = LOADER_WORKERS_DESCRIPTION + (
    " Sets both the training and the validation loader. sleap-nn's own "
    "caveat: under its default data pipeline, workers above 0 can fail on "
    "labels that read frames from video, which does not pickle."
)


def fingerprint_sleap_labels(path: Path) -> str:
    """Return the content digest of a ``.slp`` and of the validation file beside it.

    Labels without a validation file are digested as
    :func:`~mosaic.tracking.ops._common.fingerprint_dataset` digests any file.
    With one, the digest covers both files' digests, so a changed validation file
    names a different model.
    """
    validation = validation_labels_beside(path)
    if validation is None:
        return fingerprint_dataset(path)
    return hash_params(
        {
            "labels": fingerprint_dataset(path),
            "validation": fingerprint_dataset(validation),
        }
    )


def embedding_objective(labels: str) -> EmbeddingObjective:
    """The embedding objective for the labels that *labels* names.

    Labels named by a ``prepare-embedding-data`` run id take that preparation's
    objective. Any other run id, a path, and a pre-version id take sleap-nn's
    default. The run id names the objective that trained, because the labels
    reference enters it.
    """
    if _labels_kind(labels) == PREPARE_EMBEDDING_DATA_KIND:
        return PREPARED_EMBEDDING_OBJECTIVE
    return DEFAULT_EMBEDDING_OBJECTIVE


def _labels_kind(labels: str) -> str:
    """The op kind of the run that the labels reference *labels* names.

    Empty for a path and for a pre-version id, which name no run kind.
    """
    parsed = parse_op_run_id(labels)
    return parsed.kind if parsed is not None else ""


class TrainSleapInvalid(ClassifiedError, ValueError):
    """Refuses labels or settings that cannot train the requested head."""

    error_kind: ClassVar[ErrorKind] = "invalid"


_DOCUMENT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])

MODEL_DIRECTORY: Final = "model"
"""The directory under a run root that sleap-nn writes the model to."""
_TRAINING_LOG: Final = "training_log.csv"


def _epochs_trained(training_log: Path) -> int:
    """Return the epochs that *training_log* holds, 0 when the run wrote none.

    sleap-nn's CSV logger writes one row per finished epoch, so the count is
    what the run trained, which is fewer than the cap when it stopped early.
    """
    if not training_log.is_file():
        return 0
    with training_log.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


_DECLARATION_KEY: Final = "data_config.identity.track_names_are_global"


def _with_article(noun: str) -> str:
    """*noun* after "a" or "an"."""
    return f"{'an' if noun.startswith(('a', 'e', 'i', 'o', 'u')) else 'a'} {noun}"


def _training_document(
    params: TrainSleapParams,
    labels_path: Path,
    run_root: Path,
    base_weights: Path | None,
) -> tuple[SleapTrainConfig, dict[str, JsonValue]]:
    """The document a run writes and the overrides it hands sleap-nn, in order.

    The overrides are *sleap_overrides*, then the device's assignments for keys
    not already set, then the loader worker count. An embedding document keeps
    its checkpoint on the ``eval.select_metric`` that the overrides leave in
    effect, the metric sleap-nn early-stops on.

    Raises:
        TrainSleapInvalid: Hydra would reject an override, or the embedding
            objective names a positive scope that sleap-nn does not.
    """

    def document(select_metric: str) -> SleapTrainConfig:
        return sleap_train_config(
            labels_path,
            run_root,
            head=params.head,
            backbone=params.backbone,
            max_epochs=params.max_epochs,
            seed=params.seed,
            validation_fraction=params.validation_fraction,
            validation_labels=validation_labels_beside(labels_path),
            base_weights=base_weights,
            run_name=MODEL_DIRECTORY,
            objective=embedding_objective(params.labels),
            settings=params.train_settings(),
            select_metric=select_metric,
        )

    config = document(EMBEDDING_SELECT_METRIC)
    overrides: dict[str, JsonValue] = dict(params.sleap_overrides or {})
    for key, value in sleap_device_overrides(params.device).items():
        overrides.setdefault(key, value)
    # These keys are set with update, because the bag refuses them and no
    # override can name them.
    overrides.update(sleap_loader_overrides(params.num_workers))
    settings = _effective_settings(params, config, overrides)
    if settings is not None:
        select = settings.select_metric
        # An unknown metric keeps the default document, and
        # _refuse_a_checkpoint_off_the_selection_metric refuses it.
        if isinstance(select, str) and select in EMBEDDING_SELECT_METRIC_MODES:
            config = document(select)
    return config, overrides


def _effective_settings(
    params: TrainSleapParams,
    config: SleapTrainConfig,
    overrides: dict[str, JsonValue],
) -> EmbeddingSettings | None:
    """The embedding settings of the document sleap-nn composes.

    ``None`` for a pose head, whose overrides are still checked.

    Raises:
        TrainSleapInvalid: Hydra would reject an override, or the embedding
            objective names a positive scope that sleap-nn does not.
    """
    try:
        effective = effective_train_config(_DOCUMENT.validate_python(config), overrides)
        if params.head != "embedding":
            return None
        return embedding_settings(effective)
    except ValueError as exc:
        raise TrainSleapInvalid(str(exc)) from exc


def _refuse_untrainable_settings(
    params: TrainSleapParams,
    config: SleapTrainConfig,
    overrides: dict[str, JsonValue],
) -> EmbeddingSettings | None:
    """Refuse overrides and embedding settings that no labels can train.

    Reads the effective document, the one sleap-nn composes from *config* and
    *overrides*. These checks read neither the labels nor the probe.

    Returns:
        The embedding settings of the effective document, ``None`` for a
        pose head.

    Raises:
        TrainSleapInvalid: Hydra would reject an override, or the embedding
            settings can never train: ``global_id`` without the declaration,
            ``tracklet`` across videos, more than one device, or a checkpoint
            kept on a metric other than the one early stopping follows.
    """
    settings = _effective_settings(params, config, overrides)
    if settings is None:
        return None
    if settings.scope == "global_id" and not settings.track_names_are_global:
        msg = (
            f"An embedding run with positive scope 'global_id' needs "
            f"sleap_overrides to set {_DECLARATION_KEY} to true. The "
            "declaration asserts that each track name is one individual in "
            "every video."
        )
        raise TrainSleapInvalid(msg)
    if settings.scope == "tracklet" and not settings.restrict_same_video:
        msg = (
            "An embedding run with positive scope 'tracklet' needs "
            "objective.negatives.restrict_same_video true. Otherwise in-batch "
            "negatives draw the same animal from another window of the "
            "preparation."
        )
        raise TrainSleapInvalid(msg)
    _refuse_more_than_one_device(params.device, settings)
    _refuse_a_checkpoint_off_the_selection_metric(settings)
    return settings


def _refuse_a_checkpoint_off_the_selection_metric(settings: EmbeddingSettings) -> None:
    """Refuse a checkpoint monitor that differs from the selection metric.

    sleap-nn early-stops an embedding run on ``eval.select_metric`` and keeps the
    best epoch of any ``model_ckpt.monitor`` other than ``val/loss``, which
    defers to the selection metric.
    """
    select = settings.select_metric
    mode = (
        EMBEDDING_SELECT_METRIC_MODES.get(select) if isinstance(select, str) else None
    )
    if mode is None:
        accepted = ", ".join(EMBEDDING_SELECT_METRIC_MODES)
        msg = (
            f"trainer_config.eval.select_metric is {select!r}, and sleap-nn "
            f"selects an embedding checkpoint on one of {accepted}."
        )
        raise TrainSleapInvalid(msg)
    monitor = f"eval/val/{select}"
    if settings.checkpoint_monitor == "val/loss" or (
        settings.checkpoint_monitor == monitor and settings.checkpoint_mode == mode
    ):
        return
    msg = (
        "An embedding run early-stops on trainer_config.eval.select_metric and "
        "keeps the best epoch of trainer_config.model_ckpt.monitor. The monitor "
        f"is {settings.checkpoint_monitor!r} with mode "
        f"{settings.checkpoint_mode!r}, and select_metric {select!r} needs "
        f"{monitor!r} with mode {mode!r}."
    )
    raise TrainSleapInvalid(msg)


def _refuse_more_than_one_device(device: str, settings: EmbeddingSettings) -> None:
    """Refuse an embedding run that would train on more than one device."""
    indices = settings.trainer_device_indices
    if len(device.split(",")) > 1:
        offender = f"device {device!r} names more than one CUDA index"
    elif not (
        isinstance(settings.trainer_devices, int)
        and not isinstance(settings.trainer_devices, bool)
        and settings.trainer_devices == 1
    ):
        offender = (
            "trainer_config.trainer_devices is "
            f"{settings.trainer_devices!r} rather than 1"
        )
    elif isinstance(indices, list) and len(indices) > 1:
        offender = f"trainer_config.trainer_device_indices is {indices!r}"
    else:
        return
    msg = (
        f"An embedding run trains on one device, and {offender}. Name a single "
        "CUDA index in device, and leave trainer_config.trainer_devices at 1."
    )
    raise TrainSleapInvalid(msg)


def _refuse_untrainable_labels(
    ds: Dataset,
    params: TrainSleapParams,
    settings: EmbeddingSettings,
    labels_path: Path,
    *,
    planning: bool = False,
) -> bool:
    """Refuse embedding labels that cannot train the effective scope.

    Reads the labels' preparation row and the validation file beside them, so
    it runs only once the labels are on disk.

    A ``prepare-training-data`` run with no row or a blank ``track_by`` cell
    has not recorded how its tracks were named yet: a recipe that runs an
    older preparation again fills the cell on its cache hit. When *planning*,
    that ``track_by`` check is skipped rather than reading the blank as "not
    alias", and :meth:`TrainSleapOp.run` reads the filled cell.

    Returns:
        Whether the ``track_by`` check was skipped for that reason.

    Raises:
        TrainSleapInvalid: ``global_id`` on labels other than a
            ``prepare-training-data`` run that named tracks by alias,
            ``tracklet`` on labels that no ``prepare-embedding-data`` run cut,
            or labels with no validation file beside them.
    """
    kind = _labels_kind(params.labels)
    track_by_deferred = False
    if settings.scope == "global_id":
        row = prepared_data_row(ds, params.labels) if kind == PREPARE_KIND else None
        track_by = "" if row is None else row.get("track_by", "")
        if planning and kind == PREPARE_KIND and not track_by:
            track_by_deferred = True
        elif row is not None and not track_by:
            msg = (
                "An embedding run with positive scope 'global_id' needs labels "
                'from a prepare-training-data run with track_by="alias", and '
                f"{params.labels} records no track_by. It was prepared without "
                "it, or before the index recorded the column. Running "
                "prepare-training-data again with the same params records "
                'track_by="alias" on a run that set it.'
            )
            raise TrainSleapInvalid(msg)
        elif track_by != "alias":
            msg = (
                "An embedding run with positive scope 'global_id' needs labels "
                'from a prepare-training-data run with track_by="alias". A path '
                "to a .slp is refused, because mosaic cannot read how its "
                "tracks were named."
            )
            raise TrainSleapInvalid(msg)
    if settings.scope == "tracklet" and kind != PREPARE_EMBEDDING_DATA_KIND:
        msg = (
            "An embedding run with positive scope 'tracklet' needs labels from "
            "a prepare-embedding-data run. A mosaic annotation preparation "
            "merges recordings into one SLEAP video per image size. Its tracks "
            "are not per-video tracks."
        )
        raise TrainSleapInvalid(msg)
    if validation_labels_beside(labels_path) is None:
        msg = (
            "An embedding run validates on "
            f"{validation_labels_path(labels_path).name} beside its labels, "
            f"and {labels_path} has none. It takes no validation_fraction "
            "split, because a random split puts frames of one track on both "
            "sides and inflates the validation retrieval score."
        )
        raise TrainSleapInvalid(msg)
    return track_by_deferred


class TrainSleapParams(Params):
    """Parameters for the ``train-sleap`` op.

    The typed fields are the decisions mosaic has an opinion about. Everything
    else sleap-nn exposes is reachable through *sleap_overrides*, which reaches
    identity for the same reason the typed fields do: a model trained with a
    different learning rate is a different model whether or not there is a field
    for it here.
    """

    labels: Annotated[str, Declared(_LABELS_DESCRIPTION)]
    base_model: Annotated[str, Declared(_BASE_MODEL_DESCRIPTION)] = ""
    head: Annotated[SleapHead, Declared(_HEAD_DESCRIPTION)] = "centered_instance"
    backbone: Annotated[SleapBackbone, Declared(_BACKBONE_DESCRIPTION)] = "unet"
    max_epochs: Annotated[int, Declared(EPOCHS_DESCRIPTION, unit="epochs")] = 200
    seed: Annotated[int, Declared(_SEED_DESCRIPTION)] = 42
    validation_fraction: Annotated[
        float, Declared(_VALIDATION_FRACTION_DESCRIPTION)
    ] = 0.1
    batch_size: Annotated[
        int | None, Field(ge=1, examples=[16, 64]), Declared(_BATCH_SIZE_DESCRIPTION)
    ] = None
    backbone_size: Annotated[
        SleapBackboneSize | None, Declared(_BACKBONE_SIZE_DESCRIPTION)
    ] = None
    pretrained_backbone: Annotated[
        bool | None, Declared(_PRETRAINED_BACKBONE_DESCRIPTION)
    ] = None
    stem_stride: Annotated[
        int | None, Field(ge=1, examples=[2, 4]), Declared(_STEM_STRIDE_DESCRIPTION)
    ] = None
    max_stride: Annotated[
        int | None, Field(ge=1, examples=[16, 64]), Declared(_MAX_STRIDE_DESCRIPTION)
    ] = None
    optimizer: Annotated[SleapOptimizer | None, Declared(_OPTIMIZER_DESCRIPTION)] = None
    learning_rate: Annotated[
        float | None, Field(gt=0), Declared(_LEARNING_RATE_DESCRIPTION)
    ] = None
    ensure_grayscale: Annotated[
        bool | None, Declared(_ENSURE_GRAYSCALE_DESCRIPTION)
    ] = None
    cache_images: Annotated[bool | None, Declared(_CACHE_IMAGES_DESCRIPTION)] = None
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
    num_workers: Annotated[
        int | None,
        HASH_EXCLUDE,
        Field(ge=0, examples=[4, 8]),
        Declared(_NUM_WORKERS_DESCRIPTION),
    ] = None

    def train_settings(self) -> SleapTrainSettings:
        """The backbone and training settings, as the document builder takes them."""
        return SleapTrainSettings(
            batch_size=self.batch_size,
            backbone_size=self.backbone_size,
            pretrained_backbone=self.pretrained_backbone,
            stem_stride=self.stem_stride,
            max_stride=self.max_stride,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            ensure_grayscale=self.ensure_grayscale,
            cache_images=self.cache_images,
        )

    @model_validator(mode="after")
    def _settings_are_buildable(self) -> Self:
        """Refuse, at submit time, a combination of settings sleap-nn refuses."""
        _ = effective_train_settings(self.head, self.backbone, self.train_settings())
        return self

    # A consumer that sets the device itself narrows the field out of the model it
    # exposes; the check then has nothing to check there, and still runs here.
    @field_validator("device", check_fields=False)
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

    @field_validator("sleap_overrides")
    @classmethod
    def _worker_count_is_not_an_override(
        cls, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        """Refuse the loader worker count in the hashed bag, naming its field.

        Everything in *sleap_overrides* reaches the run identity, so a worker
        count set there mints a new identifier and retrains a model that
        nothing about has changed. *num_workers* is the one route to it.
        """
        shadowed = loader_worker_keys_in(value or {})
        if shadowed:
            names = ", ".join(shadowed)
            msg = (
                f"sleap_overrides may not set {names}: set num_workers "
                "instead, which does not retrain the model when it changes."
            )
            raise ValueError(msg)
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

        Refuses the settings that no labels can train, as :meth:`run` does.
        The checks that read the labels run once they are on disk. Before
        then, *require_data* defers the identity; see
        :func:`~mosaic.tracking.ops.train.planned_train_identity`. It defers
        the same way on a preparation whose ``track_by`` cell is still blank.

        The document is built before the run id exists. The kind's model root
        stands in for the run root, and the base reference for its weights.
        No check reads a path.
        """
        labels_path = resolve_training_data(ds, params.labels)
        config, overrides = _training_document(
            params,
            labels_path,
            model_index_path(ds, self.kind).parent,
            Path(params.base_model) if params.base_model else None,
        )
        settings = _refuse_untrainable_settings(params, config, overrides)
        if (
            settings is not None
            and labels_path.exists()
            and _refuse_untrainable_labels(
                ds, params, settings, labels_path, planning=True
            )
            and require_data
        ):
            raise IdentityDeferred(
                self.kind,
                f"the preparation {params.labels} has not recorded its track_by "
                "yet, and positive scope 'global_id' needs it to be alias",
            )
        return self._identity(ds, params, labels_path, require_data=require_data)

    def _identity(
        self,
        ds: Dataset,
        params: TrainSleapParams,
        labels_path: Path,
        *,
        require_data: bool,
    ) -> OpIdentity:
        from mosaic.tracking.ops.train import planned_train_identity

        return planned_train_identity(
            ds,
            kind=self.kind,
            version=self.version,
            params=params,
            data_path=labels_path,
            fingerprint=fingerprint_sleap_labels,
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
        labels_path = resolve_training_data(ds, params.labels)

        base_run_id = ""
        base_digest = ""
        base_origin = ""
        base_head = ""
        base_weights: Path | None = None
        if params.base_model:
            base = resolve_model(ds, params.base_model, self.kind)
            base_run_id = base.model_id
            base_digest = base.digest
            base_origin = observed_model_source(base).get("model_source", "")
            base_head = base.model_type
            base_weights = base.artifacts[0].file_for("weights")

        run_id = self._identity(ds, params, labels_path, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            ctx.cache_hit()
            return run_id
        # Before training, never after: see TrainPoseOp.run.
        data_fingerprint = fingerprint_sleap_labels(labels_path)

        # After the reuse gate and before the root is claimed. A cache hit pays
        # for no cold import, and a refusal is a message rather than a claim
        # left behind on a run that was never going to start. The probe reports
        # and the refusals below are mosaic's. All are decided here without
        # importing anything the SLEAP environment owns.
        run_root = model_run_root(ds, self.kind, run_id)
        config, overrides = _training_document(
            params, labels_path, run_root, base_weights
        )
        settings = _refuse_untrainable_settings(params, config, overrides)
        if settings is not None:
            _ = _refuse_untrainable_labels(ds, params, settings, labels_path)
        probe = probe_sleap(
            labels_path,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            cancel_check=ctx.cancel_token.is_cancelled,
        )
        require_sleap_nn(probe)
        require_identity_labels(probe, params.head, labels_path)
        if settings is not None:
            require_sleap_nn_at_least(probe, SLEAP_NN_EMBEDDING_MIN)
            require_embedding_labels(
                probe, scope=settings.scope, labels_kind=_labels_kind(params.labels)
            )
        # A base whose config does not read reports an empty model_type, which
        # names no head to compare. Otherwise the base is a training of this
        # run's head: a pose head takes no embedding base, and the embedding
        # head takes only an embedding one.
        if base_head and base_head != params.head:
            msg = (
                f"base_model {params.base_model!r} is {_with_article(base_head)} "
                f"model. {_with_article(params.head).capitalize()} run "
                f"fine-tunes only from {_with_article(params.head)} model."
            )
            raise ValueError(msg)

        ctx.set_total(params.max_epochs)
        run_root.mkdir(parents=True, exist_ok=True)
        marker = claim_run_root(
            ds, ctx.execution_id, run_root, self.kind, params.idle_timeout
        )
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)
        # sleap-nn writes "model-1" beside a "model" that exists, and the row
        # would then register the directory of an earlier attempt.
        stale_model = run_root / MODEL_DIRECTORY
        if stale_model.exists():
            shutil.rmtree(stale_model)

        produced = train_sleap(
            labels_path,
            run_root,
            config=config,
            overrides=overrides,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            cancel_check=ctx.cancel_token.is_cancelled,
            on_activity=phase_activity(ctx, run_root, marker, params.idle_timeout),
            on_output=epoch_reporter(ctx, params.max_epochs),
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
            produced / _TRAINING_LOG,
            _epochs_trained(produced / _TRAINING_LOG),
            artifact_shape="directory",
            artifact_path=produced,
            model_type=resolved.model_type,
            data_path=labels_path,
            data_fingerprint=data_fingerprint,
            base_origin=base_origin,
            tool_version=probe.sleap_nn_version,
        )
        return run_id
