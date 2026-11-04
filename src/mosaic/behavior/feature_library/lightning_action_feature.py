"""Lightning-action supervised temporal action segmentation feature.

Wraps the ``lightning-action`` package (Paninski lab, MIT license) as a
mosaic global feature.  Trains a temporal neural network classifier
(DilatedTCN, RNN, or TemporalMLP) on runs of consecutive frames and predicts
per-frame action probabilities with temporal context.

A temporal model reads one animal's frames in time order. The feature therefore
splits each table into series, one per individual or per ordered pair, and trains
and predicts on each series as a sequence of its own.

**Cross-validation.** With ``cv_folds`` the training entries, or with
``cv_unit="block"`` blocks of their frames, are dealt into folds as ``xgboost`` and
``sequence-classifier`` deal them. One package model is trained without each fold,
then the full model on every training run, so a fit costs ``cv_folds`` more training
runs on the GPU. A fold's model trains on the whole windows of the training runs
outside its fold, standardized by statistics of those frames alone, and a block
fold's model also leaves out ``cv_gap_frames`` frames on each side of its blocks.
``apply`` scores a training entry, or a row of a block, with the model of its fold,
and every other row with the full model. ``cv_folds.parquet`` lists the folds.

Requires the optional ``lightning-action`` package::

    pip install lightning-action

Or install mosaic with the extra::

    pip install mosaic-behavior[lightning-action]
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    final,
    override,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.helpers import UNLABELED
from mosaic.core.params import Declared
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Payload,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.types import (
    COLUMNS,
    EmitsLevel,
    DependencyLookup,
    GlobalModelParams,
    GroundTruthLabelsSource,
    LabeledTemplatesRef,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)
from mosaic.core.track_quality import consecutive_runs
from mosaic.optional_dependency import (
    MissingOptionalDependency,
    OptionalModule,
    Prerequisite,
)

from .classifier_common import (
    FIT_SCOPE_REMEDY,
    DEFAULT_CLASS_DESCRIPTION,
    CvBlocksPerSequence,
    CvRecord,
    CvGapFrames,
    CvGroupBy,
    CvSeed,
    CvUnit,
    DecisionThreshold,
    FitReporting,
    FoldLookup,
    RunningMoments,
    assemble_prediction,
    block_rows,
    check_cross_validation,
    cv_record,
    deal_runs,
    drop_unused_cv_settings,
    note_labeled_range,
    refuse_entries_outside_scope,
    scoring_folds,
    standardize,
    stored_folds,
    write_bundle,
)
from .helpers import (
    ensure_columns,
    entry_identity,
    feature_columns,
    iter_series,
    iter_series_positions,
    refuse_repeated_series_rows,
    entries_of_templates,
)
from .registry import register_feature
from .sequence_training import sub_seed

if TYPE_CHECKING:
    import numpy.typing as npt

log = logging.getLogger(__name__)


# --- Import guard ---

LIGHTNING_ACTION: Final = OptionalModule(
    "lightning_action", "lightning-action", "the lightning-action classifier"
)
"""The lightning-action package, imported when the feature fits or applies."""

LIGHTNING_ACTION_TOOL_NAME: Final = "lightning-action"
"""The name that a :class:`~mosaic.core.pipeline.refusal.ToolUnavailable` gives it."""


# --- Model artifact ---

_MODEL_DIR: Final = "lightning_action_model"
"""The directory of the full model in a run root, and the stem of each fold model's."""


class LightningActionFoldModel(TypedDict):
    """The package model fitted without one fold, and the statistics it standardizes by."""

    model_dir: str  # relative to the bundle's directory
    feature_mean: list[float]
    feature_std: list[float]


class LightningActionCv(CvRecord):
    """The cross-validation half of a bundle, written when ``cv_folds`` is set.

    The folds are those of :class:`CvRecord`, and ``models[fold]`` is the model
    fitted without that fold.
    """

    models: list[LightningActionFoldModel]


class LightningActionModelBundle(TypedDict):
    model_dir: str  # relative path to checkpoint dir within run_root
    feature_columns: list[str]
    classes: list[int]
    class_names: list[str]
    config: dict[str, object]
    version: str
    # A bundle written by 0.4 or earlier has none of these four.
    feature_mean: NotRequired[list[float]]
    feature_std: NotRequired[list[float]]
    sequence_length: NotRequired[int]
    sequence_pad: NotRequired[int]
    # Written only by a fit with ``cv_folds``.
    cv: NotRequired[LightningActionCv]


class LightningActionModelArtifact(JoblibArtifact[LightningActionModelBundle]):
    """Fitted lightning-action model bundle."""

    feature: str = "lightning-action"
    pattern: str = "lightning_action_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


class _PackageModel(Protocol):
    """The part of ``lightning_action.api.model.Model`` that the feature calls."""

    @property
    def config(self) -> Mapping[str, object]: ...

    def train(self, output_dir: str, post_inference: bool) -> None: ...

    def predict(self, *, data_path: str, input_dir: str, output_dir: str) -> None: ...


class _PackageModelClass(Protocol):
    """The part of the package's ``Model`` class that builds a fitted model."""

    def from_config(self, config: dict[str, object], /) -> _PackageModel: ...

    def from_dir(self, path: str, /) -> _PackageModel: ...


@runtime_checkable
class _Assignable(Protocol):
    def __setitem__(self, key: str, value: int, /) -> None: ...


def _require_package() -> None:
    """Raise unless the lightning-action package imports.

    Raises:
        ToolUnavailable: The package is not installed.
    """
    try:
        _ = LIGHTNING_ACTION.load()
    except MissingOptionalDependency as missing:
        raise ToolUnavailable(
            LIGHTNING_ACTION_TOOL_NAME, "environment", str(missing)
        ) from missing


def _package_model_class() -> _PackageModelClass:
    """Return the package's ``Model``, imported when it is first needed."""
    from lightning_action.api.model import Model

    return Model


def _predicting_in_process(model: _PackageModel) -> _PackageModel:
    """Have a loaded *model* read its prediction batches in this process.

    The package starts ``training.num_workers`` worker processes, four by default,
    for every series that it predicts. Two series of about 1,700 frames took 90
    seconds, and 0.2 seconds with none. Training reads the setting from the config
    that is saved with the model, which this leaves as it is.
    """
    training = model.config["training"]
    assert isinstance(training, _Assignable)
    training["num_workers"] = 0
    return model


@dataclass(frozen=True, slots=True)
class _Run:
    """One run of whole windows of consecutive labeled frames, kept in ``path``.

    ``y`` holds the class index of each frame, in the order of the classes, and
    ``first_frame`` the ``frame`` of its first row.
    """

    path: Path
    entry: tuple[str, str]
    y: npt.NDArray[np.intp]
    first_frame: int


_Piece = tuple[_Run, int, int]
"""The rows ``start`` up to ``stop`` (half-open) of a run that one model trains on."""


@dataclass(frozen=True, slots=True)
class _TrainingRuns:
    """The runs a pass over the training entries kept, each in a file of its own.

    ``ranges`` holds the labeled frames of each entry as ``(first, stop)`` for
    cross-validation by block, and is empty otherwise.
    """

    columns: list[str]
    classes: list[int]
    runs: list[_Run]
    ranges: dict[tuple[str, str], tuple[int, int]]


# --- Feature class ---


_TEMPLATES_DESCRIPTION = (
    "The extract-labeled-templates run whose train entries the model fits on. "
    "Only its template_provenance.parquet is read: the model trains on the "
    "frame runs of those entries, not on the sampled templates. An entry the "
    "run lists as test, or does not list, never trains. The run names the "
    "training set, so the step's scope is not part of its identity: a wider "
    "scope applies the fit it has and does not fit again. Mutually exclusive "
    "with model."
)

_MODEL_DESCRIPTION = (
    "Pre-fitted model to load instead of training. Mutually exclusive with templates."
)

_LABELS_DESCRIPTION = (
    "The per-frame ground-truth labels that the frames of the training entries "
    "are aligned to, resolved from labels/<kind>/index.csv. A frame no label "
    "covers is left out of training. Required with templates."
)

_HEAD_DESCRIPTION = (
    "Temporal encoder architecture: dtcn (dilated temporal convolution), "
    "rnn (LSTM/GRU), or temporalmlp."
)

_NUM_HID_UNITS_DESCRIPTION = "Number of hidden units in the temporal encoder's layers."

_NUM_LAYERS_DESCRIPTION = "Number of layers in the temporal encoder."

_NUM_LAGS_DESCRIPTION = (
    "Number of temporal lags the encoder convolves over. Used by the "
    "temporalmlp and dtcn heads and ignored by rnn."
)

_ACTIVATION_DESCRIPTION = (
    "Activation function used in the temporal encoder. Known values are "
    "relu, lrelu, sigmoid, tanh and linear."
)

_DROPOUT_RATE_DESCRIPTION = (
    "Dropout rate applied in the temporal encoder for regularization."
)

_SEQUENCE_LENGTH_DESCRIPTION = (
    "Length of each training window. A run of consecutive labeled rows trains in "
    "whole windows from its first row, so a run shorter than this does not "
    "train, and neither do the rows past a run's last whole window. It must "
    "exceed twice the context that the temporal encoder reads on each side of a "
    "row, or the loss is empty."
)

_NUM_EPOCHS_DESCRIPTION = "How long the model trains."

_BATCH_SIZE_DESCRIPTION = (
    "How many training sequence chunks the model reads in one forward pass."
)

_LEARNING_RATE_DESCRIPTION = "The optimizer's learning rate."

_WEIGHT_DECAY_DESCRIPTION = "The optimizer's weight decay coefficient."

_OPTIMIZER_DESCRIPTION = "The optimizer used to train the network."

_WEIGHT_CLASSES_DESCRIPTION = (
    "Weight the training loss by inverse class frequency, to correct for "
    "class imbalance."
)

_DEVICE_DESCRIPTION = (
    "Which device trains the model. Known values are cpu and gpu. The step is "
    "offered to a GPU worker whichever it is, and gpu trains on the CPU where "
    "the machine has no CUDA device."
)

_RANDOM_STATE_DESCRIPTION = (
    "The random seed for model initialization and training. The model fitted "
    "on every training run uses it as it is, and each fold's model a seed "
    "derived from it."
)

_CV_FOLDS_DESCRIPTION = (
    "Unset, one model is fitted and scores every entry. Set to K (at least 2), "
    "the training entries of the templates are dealt into K folds and K more "
    "package models are trained, each on the training runs outside one fold, "
    "so a fit costs K more GPU training runs. A training entry is then scored "
    "by the model that did not see it, and every other entry by the model "
    "fitted on all training runs. The fold of each entry is in "
    "cv_folds.parquet. With cv_unit block, the folds hold out blocks of frames "
    "and not whole entries. Needs templates and is refused beside model."
)

_BLOCK_FOLD_REMEDY: Final = (
    "Lower cv_gap_frames, cv_folds or sequence_length, or cut fewer, larger blocks"
)
"""What to change when a block fold's whole windows hold fewer than two classes."""

_DEFAULT_CLASS_DESCRIPTION = (
    DEFAULT_CLASS_DESCRIPTION
    + " It is also assigned to a frame that belongs to no series because its "
    "identity columns are missing."
)


@final
@register_feature
class LightningActionFeature(FitReporting):
    """Supervised temporal action segmentation via lightning-action.

    Trains a temporal neural network classifier (DilatedTCN, RNN, or
    TemporalMLP head + linear classifier) on runs of consecutive frames and
    predicts per-frame action probabilities.

    A table is read as series: one per ``id``, or one per ordered pair
    (``id1``, ``id2``, ``perspective``) when it has no ``id``. Training uses the
    runs of consecutive labeled frames of each series, in frame order, of the
    entries that the ``templates`` run assigned to ``train``. ``fit`` streams the
    step's inputs and aligns each series to ``labels``, and skips the entries
    that ``templates`` did not assign to train. The ``templates`` run names the
    training set, so the run id does not cover the step's scope: a wider scope
    applies the stored model to the new entries and trains nothing again. The
    first run has to be asked for over a scope that holds the train entries. A
    model fitted elsewhere is applied through ``model``. Frames are standardized by statistics computed
    once over the training frames and stored with the model, so ``apply``
    reads a frame as ``fit`` did. ``apply`` predicts each series as a sequence of
    its own and returns a row for every input row, in the input's order.

    The step is offered to a GPU worker whatever ``device`` says, as ``feral``
    is, and an apply-only run from a ``model`` queues there too.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.lightning_action_feature.LightningActionFeature.Params`.
    """

    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    resource_class = "gpu"
    name = "lightning-action"
    tool: ClassVar[str] = LIGHTNING_ACTION_TOOL_NAME
    # 0.3: ``nn_id`` and ``attention_target_id``, track ids, are no longer
    # fitted, from templates written before ``extract-labeled-templates``
    # 0.3 too (``BOOKKEEPING_COLS``).
    # 0.4: the training, label and apply CSVs carry the index column that
    # lightning-action reads with ``index_col=0``, and the predictions are
    # read by class column. Through 0.3 the package dropped the first feature
    # and class column, so every fit raised.
    # 0.5: the model trains on runs of consecutive frames of one series,
    # read from the step's inputs, where it trained on the class-sampled
    # templates in no time order, and the entries the templates run held out
    # no longer train. A pair table is read as one series per ``(id1, id2,
    # perspective)``. Frames are standardized by statistics stored with the
    # model, where the package standardized each file by its own. The optimizer
    # block sits where the package reads it, so ``optimizer``,
    # ``learning_rate`` and ``weight_decay`` take effect.
    # 0.6: the run id no longer covers the step's scope. The ``templates`` run
    # names the training set, so a wider scope applies the fit it has.
    version = "0.6"
    parallelizable = False  # one loaded network per instance, on one device
    # The fit streams the step's inputs but trains only on the entries that
    # ``templates`` assigned to train, and skips the others before touching state.
    scope_dependent = False
    fit_set_pinned_by: ClassVar[str] = "templates"
    # A run given a ``model`` fits nothing, and ``__init__`` clears this for it.
    fits = True  # learns a network
    relates: RelatesLevel = "none"
    accepts_overlap = False  # reads each series as its own sequence
    consumed_roots: tuple[str, ...] = ()
    prerequisites: tuple[Prerequisite, ...] = (LIGHTNING_ACTION,)
    emits: EmitsLevel = "as-input"
    follows_ids: bool = True
    # Apply reads the template columns and writes one probability per class.
    requires: ClassVar[tuple[Requirement, ...]] = (Payload(), Column(COLUMNS.frame_col))
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("prob_"), Column("predicted_label")),
        conditional=META_COLUMNS_KEPT,
    )
    ModelArtifact = LightningActionModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(GlobalModelParams[LightningActionModelArtifact, LabeledTemplatesRef]):
        templates: Annotated[
            LabeledTemplatesRef | None, Declared(_TEMPLATES_DESCRIPTION)
        ] = None
        model: Annotated[
            LightningActionModelArtifact | None, Declared(_MODEL_DESCRIPTION)
        ] = Field(default_factory=LightningActionModelArtifact)
        labels: Annotated[
            GroundTruthLabelsSource | None, Declared(_LABELS_DESCRIPTION)
        ] = None
        # --- Network architecture ---
        head: Annotated[
            Literal["temporalmlp", "rnn", "dtcn"], Declared(_HEAD_DESCRIPTION)
        ] = "dtcn"
        num_hid_units: Annotated[
            int, Field(ge=1), Declared(_NUM_HID_UNITS_DESCRIPTION)
        ] = 64
        num_layers: Annotated[int, Field(ge=1), Declared(_NUM_LAYERS_DESCRIPTION)] = 2
        num_lags: Annotated[
            int, Field(ge=1), Declared(_NUM_LAGS_DESCRIPTION, unit="frames")
        ] = 4
        activation: Annotated[
            str,
            Field(examples=["relu", "lrelu", "sigmoid", "tanh", "linear"]),
            Declared(_ACTIVATION_DESCRIPTION),
        ] = "lrelu"
        dropout_rate: Annotated[
            float, Field(ge=0, le=1), Declared(_DROPOUT_RATE_DESCRIPTION)
        ] = 0.1
        # --- Training ---
        sequence_length: Annotated[
            int, Field(ge=10), Declared(_SEQUENCE_LENGTH_DESCRIPTION, unit="frames")
        ] = 500
        num_epochs: Annotated[
            int, Field(ge=1), Declared(_NUM_EPOCHS_DESCRIPTION, unit="epochs")
        ] = 200
        batch_size: Annotated[int, Field(ge=1), Declared(_BATCH_SIZE_DESCRIPTION)] = 32
        learning_rate: Annotated[
            float, Field(gt=0), Declared(_LEARNING_RATE_DESCRIPTION)
        ] = 1e-3
        weight_decay: Annotated[
            float, Field(ge=0), Declared(_WEIGHT_DECAY_DESCRIPTION)
        ] = 0.0
        optimizer: Annotated[
            Literal["Adam", "AdamW"], Declared(_OPTIMIZER_DESCRIPTION)
        ] = "Adam"
        weight_classes: Annotated[bool, Declared(_WEIGHT_CLASSES_DESCRIPTION)] = True
        device: Annotated[
            str, Field(examples=["cpu", "gpu"]), Declared(_DEVICE_DESCRIPTION)
        ] = "gpu"
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        # --- Cross-validation ---
        cv_folds: Annotated[
            int | None, Field(ge=2), Declared(_CV_FOLDS_DESCRIPTION)
        ] = None
        cv_seed: CvSeed = 0
        cv_unit: CvUnit = "sequence"
        cv_blocks_per_sequence: CvBlocksPerSequence = None
        cv_gap_frames: CvGapFrames = 0
        cv_group_by: CvGroupBy = None
        # --- Inference ---
        decision_threshold: DecisionThreshold = None
        default_class: Annotated[int, Declared(_DEFAULT_CLASS_DESCRIPTION)]

        @model_validator(mode="after")
        def _labels_belong_to_a_fit(self) -> Self:
            """Name the labels of a fit, and of no run that loads a model."""
            if self.templates is not None and self.labels is None:
                msg = (
                    "labels is required with templates: the model trains on the "
                    "labeled frames of the entries the templates run assigned to "
                    "train, and reads their labels from it"
                )
                raise ValueError(msg)
            if self.model is not None and self.labels is not None:
                msg = (
                    "labels is read when fitting from templates, and a run given "
                    "a model fits nothing"
                )
                raise ValueError(msg)
            self.cv_blocks_per_sequence = check_cross_validation(
                self.cv_folds,
                self.cv_seed,
                model_given="model" in self.model_fields_set and self.model is not None,
                cv_unit=self.cv_unit,
                cv_blocks_per_sequence=self.cv_blocks_per_sequence,
                cv_gap_frames=self.cv_gap_frames,
                cv_group_by=self.cv_group_by,
            )
            return self

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, less the cross-validation settings a run leaves unused.

            A run without ``cv_folds`` has the identity it had before the step
            cross-validated. ``check_cross_validation`` holds every other
            setting at its default then, so leaving them out loses nothing.
            """
            dumped = drop_unused_cv_settings(
                super().identity_dump(), self.cv_unit, self.cv_group_by
            )
            if self.cv_folds is None:
                del dumped["cv_folds"]
                del dumped["cv_seed"]
            return dumped

    def __init__(
        self,
        inputs: LightningActionFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.fits = self.params.model is None
        self._reset()

    def _reset(self) -> None:
        self._feature_columns: list[str] | None = None
        self._feature_mean: npt.NDArray[np.float64] | None = None
        self._feature_std: npt.NDArray[np.float64] | None = None
        self._classes: list[int] | None = None
        self._class_names: list[str] | None = None
        self._sequence_length: int | None = None
        self._sequence_pad: int | None = None
        self._train_entries: set[tuple[str, str]] | None = None
        self._label_files: DependencyLookup = {}
        # The directory that holds the model directories the bundle names.
        self._model_dir: Path | None = None
        self._full_model: str = _MODEL_DIR
        self._la_models: dict[str, _PackageModel] = {}  # cached package Models
        self._config: dict[str, object] | None = None
        self._cv: LightningActionCv | None = None
        self._fold_of: dict[tuple[str, str], int] = {}
        self._blocks: FoldLookup | None = None
        self._cv_table: pd.DataFrame | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._reset()

        # Branch 1: cached model in run_root
        cached_path = run_root / "lightning_action_model.joblib"
        if cached_path.exists():
            bundle: LightningActionModelBundle = (
                LightningActionModelArtifact().from_path(cached_path)
            )
            self._load_bundle(bundle, run_root, folds=self.params.cv_folds is not None)
            return True

        # Branch 2: pre-fitted model from artifact_paths. Its folds are not used:
        # a fitted model scores every entry with its full model.
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._load_bundle(bundle, artifact_paths["model"].parent, folds=False)
            return True

        # Branch 3: the entries of the templates run to fit on
        if self.params.templates is not None and "templates" in artifact_paths:
            train_entries = entries_of_templates(artifact_paths["templates"], "train")
            if train_entries is None:
                msg = (
                    f"[{self.name}] the templates run left no "
                    "template_provenance.parquet beside its templates, so the "
                    "entries it held out are unknown. Rerun extract-labeled-templates"
                )
                raise ValueError(msg)
            self._train_entries = set(train_entries)
            self._label_files = dependency_lookups.get("labels", {})
            return False

        return False

    def fit(self, inputs: InputStream) -> None:
        _require_package()
        from lightning_action.data.utils import compute_sequence_pad

        if self._train_entries is None:
            msg = (
                f"[lightning-action] model {self.params.model.run_id!r} did not "
                "resolve to a stored model, so there is nothing to apply and the "
                "step does not fit in its place. Check the model reference"
                if self.params.model is not None
                else "[lightning-action] No templates loaded, so check load_state."
            )
            raise RuntimeError(msg)

        p = self.params
        pad = int(
            compute_sequence_pad(p.head, num_layers=p.num_layers, num_lags=p.num_lags)
        )
        if p.sequence_length <= 2 * pad:
            msg = (
                f"[{self.name}] sequence_length {p.sequence_length} must exceed twice "
                f"the {pad} frames of context that head {p.head!r} reads on each side "
                f"of a frame ({2 * pad}), or the loss of every window is empty. "
                "Raise sequence_length or lower num_layers or num_lags"
            )
            raise ValueError(msg)

        with tempfile.TemporaryDirectory(prefix="la_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            runs = self._write_runs(inputs, tmpdir / "runs")
            self._classes = runs.classes
            self._class_names = [str(c) for c in runs.classes]
            self._feature_columns = runs.columns
            self._sequence_length = p.sequence_length
            self._sequence_pad = pad

            plan = (
                deal_runs(
                    self.name,
                    p,
                    runs.runs,
                    runs.classes,
                    train_entries=sorted(self._train_entries),
                    ranges=runs.ranges,
                    min_length=p.sequence_length,
                    multiple=p.sequence_length,
                    block_remedy=_BLOCK_FOLD_REMEDY,
                )
                if p.cv_folds is not None
                else None
            )

            print(
                f"[lightning-action] Training: {len(runs.runs)} runs, "
                f"{len(runs.columns)} features, {len(runs.classes)} classes, "
                f"head={p.head}, epochs={p.num_epochs}, "
                f"models={1 + (p.cv_folds or 0)}",
                file=sys.stderr,
            )

            # The trained models wait here until save_state copies them to run_root.
            self._model_dir = Path(tempfile.mkdtemp(prefix="la_model_"))
            total = 1 + (p.cv_folds or 0)
            fold_models: list[LightningActionFoldModel] = []
            if plan is not None:
                for fold, pieces in enumerate(plan.training):
                    self._before_fit(fold, total, whole="all runs")
                    name = f"{_MODEL_DIR}_fold{fold}"
                    mean, std, _ = self._train(
                        pieces,
                        runs,
                        tmpdir / name,
                        sub_seed(p.random_state, "fold", fold),
                    )
                    fold_models.append(
                        {
                            "model_dir": name,
                            "feature_mean": mean.tolist(),
                            "feature_std": std.tolist(),
                        }
                    )
                    self._after_fit(fold + 1, total)
            self._before_fit(total - 1, total, whole="all runs")
            whole: list[_Piece] = [(run, 0, len(run.y)) for run in runs.runs]
            mean, std, config = self._train(
                whole, runs, tmpdir / _MODEL_DIR, p.random_state
            )
            self._after_fit(total, total)
            self._feature_mean = mean
            self._feature_std = std
            self._config = config

        if plan is not None and p.cv_folds is not None:
            self._cv = {
                **cv_record(
                    p.cv_folds,
                    p.cv_seed,
                    plan.fold_of,
                    plan.blocks,
                    blocks_per_sequence=p.cv_blocks_per_sequence,
                    gap_frames=p.cv_gap_frames,
                ),
                "models": fold_models,
            }
            self._fold_of = dict(plan.fold_of)
            self._blocks = (
                FoldLookup.by_block(block_rows(plan.blocks)) if plan.blocks else None
            )
            self._cv_table = plan.table
        print("[lightning-action] Training complete.", file=sys.stderr)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _require_package()
        if (
            self._feature_columns is None
            or self._feature_mean is None
            or self._feature_std is None
            or self._classes is None
            or self._sequence_length is None
        ):
            msg = (
                "[lightning-action] Model not fitted "
                "and fit() or load_state() has not run."
            )
            raise RuntimeError(msg)

        feat_cols = self._feature_columns
        ensure_columns(df, [COLUMNS.frame_col, *feat_cols])
        refuse_repeated_series_rows(df, self.name)

        probs = self._predict_series(df, df[feat_cols].to_numpy(dtype=np.float64))

        return assemble_prediction(
            df,
            probs,
            self._classes,
            self.params.decision_threshold,
            self.params.default_class,
        )

    def save_state(self, run_root: Path) -> None:
        if (
            self._model_dir is None
            or self._feature_columns is None
            or self._feature_mean is None
            or self._feature_std is None
            or self._classes is None
            or self._sequence_length is None
            or self._sequence_pad is None
        ):
            return
        run_root.mkdir(parents=True, exist_ok=True)

        # Copy the model directories into run_root
        if self._model_dir != run_root:
            names = [self._full_model]
            if self._cv is not None:
                names += [model["model_dir"] for model in self._cv["models"]]
            for name in names:
                dest = run_root / name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(self._model_dir / name, dest)
        if self._cv_table is not None:
            _ = write_parquet_atomic(self._cv_table, run_root / "cv_folds.parquet")

        # Save metadata bundle
        bundle: LightningActionModelBundle = {
            "model_dir": self._full_model,
            "feature_columns": self._feature_columns,
            "feature_mean": self._feature_mean.tolist(),
            "feature_std": self._feature_std.tolist(),
            "classes": self._classes,
            "class_names": self._class_names or [],
            "sequence_length": self._sequence_length,
            "sequence_pad": self._sequence_pad,
            "config": self._config or {},
            "version": self.version,
        }
        if self._cv is not None:
            bundle["cv"] = self._cv
        write_bundle(run_root / "lightning_action_model.joblib", bundle)

        # Clean up temp model dir (if it's separate from run_root)
        if self._model_dir != run_root and self._model_dir.exists():
            shutil.rmtree(self._model_dir, ignore_errors=True)
        self._model_dir = run_root
        self._la_models = {}  # force reload from new location on next apply

    # --- Helpers ---

    def _load_bundle(
        self, bundle: LightningActionModelBundle, base_dir: Path, *, folds: bool
    ) -> None:
        """Take a loaded bundle, and its fold models when *folds* asks for them."""
        mean = bundle.get("feature_mean")
        std = bundle.get("feature_std")
        length = bundle.get("sequence_length")
        pad = bundle.get("sequence_pad")
        if mean is None or std is None or length is None or pad is None:
            msg = (
                f"[{self.name}] this model was fitted by version "
                f"{bundle.get('version', '0.4 or earlier')}, which stood each "
                "file on its own statistics and stored none. Fit it again"
            )
            raise ValueError(msg)
        self._feature_columns = bundle["feature_columns"]
        self._feature_mean = np.asarray(mean, dtype=np.float64)
        self._feature_std = np.asarray(std, dtype=np.float64)
        self._classes = bundle["classes"]
        self._class_names = bundle["class_names"]
        self._sequence_length = length
        self._sequence_pad = pad
        self._config = bundle["config"]
        self._model_dir = base_dir
        self._full_model = bundle["model_dir"]
        self._la_models = {}  # lazy-load on first apply()
        if folds:
            cv = bundle.get("cv")
            self._fold_of, self._blocks = stored_folds(
                self.name, cv, self.params.cv_unit, "lightning_action_model.joblib"
            )
            self._cv = cv

    def _write_runs(self, inputs: InputStream, runs_dir: Path) -> _TrainingRuns:
        """Stream the training entries and keep each of their frame runs on disk.

        A run is a stretch of consecutive labeled frames of one series. It is
        kept in whole windows from its first frame, which is how the package
        cuts it, and a run with no whole window is left out. For cross-validation
        by block the labeled frames of each entry are noted too.

        Raises:
            ValueError: A train entry that has labels is missing from the stream.
            RuntimeError: No run is left, or the runs hold one class.
        """
        assert self._train_entries is not None
        source = self.params.labels
        assert source is not None, "templates validate with labels"
        window = self.params.sequence_length
        by_block = self.params.cv_folds is not None and self.params.cv_unit == "block"
        runs_dir.mkdir(parents=True)

        columns: list[str] | None = None
        kept: list[tuple[Path, tuple[str, str], npt.NDArray[np.int64], int]] = []
        ranges: dict[tuple[str, str], tuple[int, int]] = {}
        seen: set[tuple[str, str]] = set()
        unlabeled_entries = 0
        too_short = 0
        past_last_window = 0

        for _, df in inputs():
            entry = entry_identity(df)
            if entry not in self._train_entries:
                continue
            seen.add(entry)
            label_file = self._label_files.get(entry)
            if label_file is None:
                unlabeled_entries += 1
                continue
            ensure_columns(df, [COLUMNS.frame_col])
            refuse_repeated_series_rows(df, self.name)
            if columns is None:
                columns = feature_columns(df)
            else:
                ensure_columns(df, columns)

            for _, rows in iter_series(df):
                labels = labels_for_rows(label_file, rows, source)
                frames = rows[COLUMNS.frame_col].to_numpy()
                matrix = rows[columns].to_numpy(dtype=np.float64)
                if by_block:
                    note_labeled_range(ranges, entry, frames, labels != UNLABELED)
                for start, stop in consecutive_runs(frames, labels != UNLABELED):
                    whole = (stop - start) // window * window
                    if whole == 0:
                        too_short += 1
                        continue
                    past_last_window += stop - start - whole
                    stop = start + whole
                    path = runs_dir / f"run_{len(kept):06d}.npz"
                    np.savez(path, x=matrix[start:stop], y=labels[start:stop])
                    kept.append(
                        (
                            path,
                            entry,
                            labels[start:stop].astype(np.int64),
                            int(frames[start]),
                        )
                    )

        refuse_entries_outside_scope(
            self.name,
            "train entries of the templates run",
            self._train_entries,
            self._label_files,
            seen,
            consequence="The model would train on fewer entries than its templates name.",
            remedy=FIT_SCOPE_REMEDY,
        )
        if unlabeled_entries or too_short or past_last_window:
            print(
                f"[lightning-action] {unlabeled_entries} train entries have no label "
                f"file, {too_short} runs are shorter than sequence_length {window}, "
                f"and {past_last_window} frames lie past the last whole window of a "
                "run. None of them train",
                file=sys.stderr,
            )
        if columns is None or not kept:
            msg = (
                f"[{self.name}] no run of consecutive labeled frames of "
                f"sequence_length {window} frames or more is left to train on: "
                f"{unlabeled_entries} train entries have no label file and "
                f"{too_short} runs are shorter than sequence_length"
            )
            raise RuntimeError(msg)
        classes = sorted(
            {int(c) for _, _, labels, _ in kept for c in np.unique(labels)}
        )
        if len(classes) < 2:
            msg = (
                f"[{self.name}] the training runs hold {len(classes)} class "
                f"({classes}), and a classifier needs two or more"
            )
            raise RuntimeError(msg)
        runs = [
            _Run(
                path,
                entry,
                np.searchsorted(classes, labels).astype(np.intp),
                first_frame,
            )
            for path, entry, labels, first_frame in kept
        ]
        return _TrainingRuns(columns, classes, runs, ranges)

    def _train(
        self, pieces: list[_Piece], runs: _TrainingRuns, work_dir: Path, seed: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, object]]:
        """Train one package model on *pieces* and keep it under ``self._model_dir``.

        Returns:
            The mean and standard deviation that the model's frames were
            standardized by, and the config it was trained with.
        """
        assert self._model_dir is not None
        model_class = _package_model_class()
        data_dir = work_dir / "data"
        output_dir = work_dir / "output"
        mean, std = self._write_training_csvs(pieces, runs, data_dir)
        config = self._build_config(
            len(runs.columns), len(runs.classes), str(data_dir), seed
        )
        model = model_class.from_config(config)
        model.train(output_dir=str(output_dir), post_inference=False)
        shutil.copytree(output_dir, self._model_dir / work_dir.name)
        return mean, std, config

    def _write_training_csvs(
        self, pieces: list[_Piece], runs: _TrainingRuns, data_dir: Path
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Write the pieces as standardized CSVs in lightning-action layout.

        One file per piece. Both files carry the index column, because the package
        reads its first column as the index (``index_col=0``) and would otherwise
        drop the first feature or class. The package standardizes nothing, so the
        frames are standardized here, by the statistics of these pieces, which
        ``apply`` repeats.

        Returns:
            The mean and standard deviation of the pieces' frames.
        """
        features_dir = data_dir / "features"
        labels_dir = data_dir / "labels"
        features_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        moments = RunningMoments(len(runs.columns))
        for run, start, stop in pieces:
            with np.load(run.path) as stored:
                moments.add(stored["x"][start:stop])
        mean, std = moments.standardizer()

        names = [str(c) for c in runs.classes]
        for run, start, stop in pieces:
            with np.load(run.path) as stored:
                matrix = stored["x"][start:stop]
            stem = run.path.stem if start == 0 else f"{run.path.stem}_{start:06d}"
            pd.DataFrame(standardize(matrix, mean, std), columns=runs.columns).to_csv(
                features_dir / f"{stem}.csv", index=True, float_format="%.7g"
            )
            one_hot = (
                run.y[start:stop, None] == np.arange(len(runs.classes))[None, :]
            ).astype(int)
            pd.DataFrame(one_hot, columns=names).to_csv(
                labels_dir / f"{stem}.csv", index=True
            )
        return mean, std

    def _build_config(
        self, n_features: int, n_classes: int, data_path: str, seed: int
    ) -> dict[str, object]:
        """Translate mosaic Params into a lightning-action config dict.

        The package reads the optimizer from the top level of the config, and
        takes weight decay as ``wd``. Under ``training`` it is ignored without a
        warning. ``transforms`` is empty because the frames are standardized
        before they are written.
        """
        p = self.params
        return {
            "data": {
                "data_path": data_path,
                "input_dir": "features",
                "weight_classes": p.weight_classes,
                "transforms": [],
            },
            "model": {
                "input_size": n_features,
                "output_size": n_classes,
                "head": p.head,
                "num_hid_units": p.num_hid_units,
                "num_layers": p.num_layers,
                "num_lags": p.num_lags,
                "activation": p.activation,
                "dropout_rate": p.dropout_rate,
                "seed": seed,
            },
            "training": {
                "batch_size": p.batch_size,
                "num_epochs": p.num_epochs,
                "device": p.device,
                "sequence_length": p.sequence_length,
                "seed": seed,
                "train_probability": 0.9,
                "val_probability": 0.1,
            },
            "optimizer": {
                "type": p.optimizer,
                "lr": p.learning_rate,
                "wd": p.weight_decay,
            },
        }

    def _loaded_model(self, name: str) -> _PackageModel:
        """Return the package model of directory *name*, loading it once."""
        cached = self._la_models.get(name)
        if cached is not None:
            return cached
        if self._model_dir is None:
            msg = "[lightning-action] No model available for prediction."
            raise RuntimeError(msg)
        model = _predicting_in_process(
            _package_model_class().from_dir(str(self._model_dir / name))
        )
        self._la_models[name] = model
        return model

    def _scoring_model(
        self, fold: int
    ) -> tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the directory and the statistics of the model of *fold*, ``-1`` for the full one."""
        if fold < 0:
            assert self._feature_mean is not None
            assert self._feature_std is not None
            return self._full_model, self._feature_mean, self._feature_std
        assert self._cv is not None
        model = self._cv["models"][fold]
        return (
            model["model_dir"],
            np.asarray(model["feature_mean"], dtype=np.float64),
            np.asarray(model["feature_std"], dtype=np.float64),
        )

    def _predict_series(
        self, df: pd.DataFrame, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predict the class probabilities of every row of *df*, one series at a time.

        Each row is scored by the model that :func:`scoring_folds` names: a
        training entry, or a row of a block, by the model fitted without its fold,
        and every other row by the full model. A model standardizes *features*
        by its own statistics and predicts each series that holds a row it
        scores whole.

        Returns:
            One row of probabilities per row of *df*, NaN for a row that belongs
            to no series because a key column is missing.
        """
        assert self._classes is not None
        probs = np.full((len(df), len(self._classes)), np.nan)
        series = list(iter_series_positions(df))
        if not series:
            return probs
        folds = scoring_folds(df, self._fold_of, self._blocks, self.name)
        for fold in np.unique(folds):
            needed = folds == fold
            name, mean, std = self._scoring_model(int(fold))
            scored = self._predict_with(
                self._loaded_model(name),
                standardize(features, mean, std),
                [rows for rows in series if needed[rows].any()],
            )
            probs[needed] = scored[needed]
        return probs

    def _predict_with(
        self,
        model: _PackageModel,
        matrix: npt.NDArray[np.float64],
        series: list[npt.NDArray[np.intp]],
    ) -> npt.NDArray[np.float64]:
        """Predict the rows of each of *series* with *model*, from the standardized *matrix*.

        The package predicts whole windows from a file's first row, and only a file
        of at least one window and its context. Each series is therefore written
        as its own file, with zero rows (the column mean) ahead of its first frame
        to fill a whole number of windows, and at least two. The last frame ends
        the final window, so no row falls past it, and the zero rows sit where a
        training run also starts with no context.

        Returns:
            One row of probabilities per row of *matrix*, NaN for a row in none of
            *series*.
        """
        assert self._classes is not None
        assert self._class_names is not None
        assert self._sequence_length is not None
        feat_cols = self._feature_columns
        assert feat_cols is not None
        window = self._sequence_length
        probs = np.full((len(matrix), len(self._classes)), np.nan)

        with tempfile.TemporaryDirectory(prefix="la_apply_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            features_dir = tmpdir / "features"
            pred_dir = tmpdir / "predictions"
            features_dir.mkdir()
            pred_dir.mkdir()

            # The package reads its first column as the index (``index_col=0``),
            # so the index is written too.
            leads: list[int] = []
            for index, rows in enumerate(series):
                n_windows = max(2, math.ceil(len(rows) / window))
                padded = np.zeros((n_windows * window, len(feat_cols)))
                lead = len(padded) - len(rows)
                padded[lead:] = matrix[rows]
                pd.DataFrame(padded, columns=feat_cols).to_csv(
                    features_dir / f"s{index:06d}.csv", index=True, float_format="%.7g"
                )
                leads.append(lead)

            model.predict(
                data_path=str(tmpdir),
                input_dir="features",
                output_dir=str(pred_dir),
            )

            # The package writes the row index first and one column per class
            # name, so the probabilities are read by name rather than position.
            for index, rows in enumerate(series):
                pred_file = pred_dir / f"s{index:06d}_predictions.csv"
                if not pred_file.exists():
                    msg = "[lightning-action] No prediction output found."
                    raise RuntimeError(msg)
                pred_df = pd.read_csv(pred_file, index_col=0)
                ensure_columns(pred_df, self._class_names)
                values = pred_df[self._class_names].to_numpy(dtype=np.float64)
                values = values[leads[index] :]
                if len(values) != len(rows) or np.isnan(values).any():
                    msg = (
                        f"[lightning-action] the package returned {len(values)} "
                        f"rows, some of them without probabilities, for a series "
                        f"of {len(rows)} rows"
                    )
                    raise RuntimeError(msg)
                probs[rows] = values
        return probs
