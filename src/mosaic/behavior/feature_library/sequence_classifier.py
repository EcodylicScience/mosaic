"""``sequence-classifier``: a neural network that labels each frame of a series.

A head from the registry of :mod:`~mosaic.behavior.feature_library.sequence_heads`
(``tcn``, ``gru``, ``mlp``, or one registered in Python) reads the frames of a series
and writes a probability for each class at every frame. The classifier fits it on
runs of consecutive labeled frames and predicts each series whole.

**Series.** A table is read as series: one per ``id``, or one per ordered pair
(``id1``, ``id2``, ``perspective``) when it has no ``id``. A series is cut where its
frames are not consecutive, and the network never convolves or recurs across the cut.

**Fitting.** The entries that the ``templates`` run assigned to ``train`` train, on
the runs of consecutive frames of their series that ``labels`` covers. A batch is a
set of random crops of whole runs, and the drawn templates are not read. Frames are
standardized by the mean and standard deviation of the training frames, which the
model stores. The ``templates`` run names the training set, so the run id does not
cover the step's scope: the step fits once, on the train entries it names, and a
wider scope applies the stored model to the new entries and fits nothing again. The
first run has to be asked for over a scope that holds the train entries. A model
fitted elsewhere is applied through ``model``.

**Cross-validation.** With ``cv_folds`` the training entries are dealt into folds as
``xgboost`` deals them, one model is fitted without each fold, and an entry of a fold
is scored by the model that did not see it, and an entry outside the training set by
the model fitted on all of it. ``cv_folds.parquet`` lists the folds.

**Cross-validation by block.** ``cv_unit="block"`` holds out blocks of frames instead of
whole sequences, so that a dataset with one training sequence is still cross-validated.
Each training entry is cut into ``cv_blocks_per_sequence`` contiguous blocks that tile
the frames from the first to the last labeled frame of any of its series (the frames
the step's inputs hold, so the cut does not depend on the scope or on the order the
entries arrive in), differing in length by at most one frame. The blocks of all entries
are ordered by the sha256 of ``[cv_seed, group, sequence, block]`` and dealt to the folds
in turn. The model of a fold trains on the labeled runs of every entry minus the blocks
of that fold and ``cv_gap_frames`` frames on each side of them; a piece of a run that this
leaves shorter than ``min_run`` does not train. A row is scored by the model of the fold
whose block holds its ``frame``, an unlabeled frame between two labeled ones included.
Only the frames outside every block, which no label covers, and the entries outside the
training set are scored by the model fitted on all of it, so every labeled frame of a
training entry is out of fold. The network predicts each series whole for every model
that scores some of its rows, so a block-mode apply costs up to one pass per fold more.
``cv_folds.parquet`` then has one row per block, with its ``start`` and ``stop`` frames
(half-open) beside ``fold``.

**Determinism.** The same params and inputs fit the same model on a CPU at one
thread count. A GPU repeats on the same machine with the settings below, and is
not promised to give the CPU's bytes. ``device`` is part of the run id for this
reason, and ``torch``'s version is recorded with the model and not hashed.
"""

from __future__ import annotations

import logging
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
    Self,
    TypedDict,
    final,
    override,
)

import numpy as np
import pandas as pd
from pydantic import (
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)

from mosaic.core.helpers import UNLABELED
from mosaic.core.json_value import JsonValue
from mosaic.core.params import HASH_EXCLUDE, Declared
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Payload,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types import (
    COLUMNS,
    DependencyLookup,
    EmitsLevel,
    GlobalModelParams,
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    LabeledTemplatesRef,
    Result,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.track_quality import consecutive_runs
from mosaic.optional_dependency import OptionalModule, Prerequisite

from .classifier_common import (
    FIT_SCOPE_REMEDY,
    DEFAULT_CLASS_DESCRIPTION,
    CvBlock,
    CvBlocksPerSequence,
    CvGapFrames,
    CvSeed,
    CvUnit,
    DecisionThreshold,
    FitReporting,
    FoldLookup,
    RunningMoments,
    assemble_prediction,
    block_fold_table,
    block_list,
    block_rows,
    check_cross_validation,
    check_folds,
    deal_blocks,
    deal_folds,
    drop_block_settings,
    fold_map,
    fold_rows,
    fold_table,
    held_out_frames,
    refuse_entries_outside_scope,
    standardize,
    write_bundle,
)
from .helpers import (
    check_memory_headroom,
    ensure_columns,
    entry_identity,
    feature_columns,
    iter_series,
    iter_series_positions,
    out_of_fold_choice,
    refuse_repeated_series_rows,
    train_entries_of_templates,
)
from .registry import register_feature
from .sequence_heads import (
    HeadConfig,
    assemble_network,
    get_head,
    head_source_digest,
)
from .sequence_training import (
    Sampler,
    TrainingRun,
    class_weights,
    deterministic_torch,
    network_state,
    predict_probabilities,
    seed_torch,
    sub_seed,
    train_network,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch import nn

log = logging.getLogger(__name__)

TORCH: Final = OptionalModule("torch", "deep-learning", "the sequence classifier")
"""The torch package, imported when the first model is fitted or applied."""

_MIN_STD: Final = 1e-6
"""A column with less spread than this standardizes by 1, as a constant column does."""


@dataclass(frozen=True)
class _FoldPlan:
    """The folds a fit deals: ``cv_folds.parquet``, who is in which fold, and each fold's runs.

    ``fold_of`` holds the fold of each entry when the unit is the sequence, and
    ``blocks`` the folded blocks when it is the block. ``training[k]`` holds the runs
    that the model without fold ``k`` trains on.
    """

    table: pd.DataFrame
    fold_of: dict[tuple[str, str], int]
    blocks: list[CvBlock]
    training: list[list[TrainingRun]]


# --- Model artifact ---


class SequenceClassifierModel(TypedDict):
    """One fitted network: the statistics it standardizes by and its weights.

    ``fold`` is the fold it was fitted without, ``None`` for the model fitted on
    every training run. ``weights`` holds the parameters and buffers of the
    network by name, as numpy arrays.
    """

    fold: int | None
    mean: list[float]
    std: list[float]
    weights: dict[str, npt.NDArray[np.float32]]


class SequenceClassifierCv(TypedDict):
    """The cross-validation half of a bundle, written when ``cv_folds`` is set.

    By sequence, ``fold_of`` holds ``(group, sequence, fold)`` for every train entry.
    By block, the keys ``unit``, ``blocks_per_sequence``, ``gap_frames`` and
    ``blocks`` are written too, ``fold_of`` is empty, and ``blocks`` holds
    ``(group, sequence, block, start, stop, fold)`` for every block.
    """

    folds: int
    seed: int
    fold_of: list[tuple[str, str, int]]
    unit: NotRequired[Literal["block"]]
    blocks_per_sequence: NotRequired[int]
    gap_frames: NotRequired[int]
    blocks: NotRequired[list[tuple[str, str, int, int, int, int]]]


class SequenceClassifierBundle(TypedDict):
    """What a fit leaves: the models and what is needed to rebuild their network."""

    version: str
    head: str
    head_config: dict[str, object]
    head_revision: int
    head_source_digest: str
    input_dropout: float
    clip: float
    columns: list[str]
    classes: list[int]
    models: list[SequenceClassifierModel]
    environment: dict[str, str]
    cv: NotRequired[SequenceClassifierCv]


class SequenceClassifierModelArtifact(JoblibArtifact[SequenceClassifierBundle]):
    """Fitted sequence-classifier bundle (sequence_classifier_model.joblib)."""

    feature: str = "sequence-classifier"
    pattern: str = "sequence_classifier_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Parameters ---

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
    "A pre-fitted SequenceClassifierModelArtifact to load instead of training. "
    "Mutually exclusive with templates."
)

_LABELS_DESCRIPTION = (
    "The per-frame ground-truth labels that the frames of the training entries "
    "are aligned to, resolved from labels/<kind>/index.csv. A frame no label "
    "covers is left out of training. Required with templates."
)

_COLUMNS_DESCRIPTION = (
    "The payload columns the network reads, in this order. Unset, every numeric "
    "payload column of the inputs. A narrow set can beat a wide one, so compare "
    "sets by cross-validated score."
)

_HEAD_DESCRIPTION = (
    "The network. A built-in head is tcn (dilated convolutions), gru (a "
    "bidirectional GRU) or mlp (each frame by itself). A head registered in "
    "Python with register_head in the process that runs the step is named "
    "here too."
)

_HEAD_CONFIG_DESCRIPTION = (
    "The options of the chosen head. Unset, the head's defaults. tcn takes "
    "hidden (128), blocks (6), kernel_size (3, odd) and dropout (0.3). gru "
    "takes hidden (128), layers (2), bidirectional (true) and dropout (0.3). "
    "mlp takes hidden (128), layers (2) and dropout (0.3). Options that the "
    "chosen head does not have are refused."
)

_HEAD_REVISION_DESCRIPTION = (
    "A number that the author of a head registered in Python raises when its "
    "code changes, so that a changed head is a new run. A built-in head ignores "
    "it, because the version of the step names its code."
)

_INPUT_DROPOUT_DESCRIPTION = "Dropout applied to the input frames while training."

_CLIP_DESCRIPTION = (
    "Standardized values are limited to this many standard deviations on both "
    "sides, so that an outlier frame does not dominate a batch."
)

_CLASS_WEIGHT_DESCRIPTION = (
    "How the loss weighs a class. balanced weights it by the inverse of its "
    "frequency, sqrt by the square root of that, none equally. Each is scaled to "
    "a mean of 1 over the classes of the training runs."
)

_CROP_LENGTH_DESCRIPTION = (
    "The longest crop of a run in a training batch. A batch is cut to its "
    "shortest crop."
)

_BATCH_SIZE_DESCRIPTION = "The crops in a training batch."

_STEPS_DESCRIPTION = "How long each model trains, counted in batches."

_LEARNING_RATE_DESCRIPTION = "The peak learning rate of the one-cycle schedule."

_WEIGHT_DECAY_DESCRIPTION = "The weight decay of the AdamW optimizer."

_PCT_START_DESCRIPTION = "The share of the steps that the learning rate climbs for."

_GRAD_CLIP_DESCRIPTION = "The norm that gradients are clipped to."

_MIN_RUN_DESCRIPTION = (
    "The shortest run of consecutive labeled rows that trains. A shorter run is "
    "left out."
)

_RANDOM_STATE_DESCRIPTION = (
    "The random seed of the batches, the initial weights and the dropout. The "
    "model fitted on every training run uses it as it is, and each fold's model "
    "a seed derived from it."
)

_CV_FOLDS_DESCRIPTION = (
    "Unset, one model is fitted and scores every entry. Set to K (at least 2), "
    "the training entries of the templates are dealt into K folds and K more "
    "models are fitted, each on the training runs outside one fold. A training "
    "entry is then scored by the model that did not see it, and every other "
    "entry by the model fitted on all training runs. The fold of each entry is "
    "in cv_folds.parquet. With cv_unit block, the folds hold out blocks of frames "
    "and not whole entries. Needs templates and is refused beside model."
)

_DEVICE_DESCRIPTION = (
    "Where the network trains. cuda needs a CUDA device and fails without one. "
    "cpu is for small fits and tests. The device is part of the run id, since a "
    "GPU and a CPU need not give one model. The step is offered to a GPU worker "
    "whichever it is."
)

_APPLY_DEVICE_DESCRIPTION = (
    "Where a fitted network predicts. A model fitted on a GPU applies on the CPU. "
    "It changes the time of an apply and not the model, so it is not part of the "
    "run id. Probabilities agree across devices to about 1e-5."
)

_N_THREADS_DESCRIPTION = (
    "The CPU threads a fit or an apply uses. It changes the time, not the model "
    "beyond rounding, so it is not part of the run id. One thread makes a CPU fit "
    "repeat across machines."
)


@final
@register_feature
class SequenceClassifier(FitReporting):
    """Neural sequence classifier: a registered head on runs of consecutive frames.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.sequence_classifier.SequenceClassifier.Params`.
    """

    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    resource_class = "gpu"
    name = "sequence-classifier"
    # 0.2: the run id no longer covers the step's scope. The ``templates`` run names
    # the training set, so a wider scope applies the fit it has.
    version = "0.2"
    parallelizable = False  # one loaded network per instance, and a pass is a series
    # The fit streams the step's inputs but trains only on the entries that
    # ``templates`` assigned to train, and skips the others before touching state.
    scope_dependent = False
    fit_set_pinned_by: ClassVar[str] = "templates"
    # A run given a ``model`` fits nothing, and ``__init__`` clears this for it.
    fits = True  # learns a network
    accepts_overlap = False  # reads each series as its own sequence
    consumed_roots: tuple[str, ...] = ()
    prerequisites: tuple[Prerequisite, ...] = (TORCH,)
    emits: EmitsLevel = "as-input"
    requires: ClassVar[tuple[Requirement, ...]] = (Payload(), Column(COLUMNS.frame_col))
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("prob_"), Column("predicted_label")),
        conditional=META_COLUMNS_KEPT,
    )
    ModelArtifact = SequenceClassifierModelArtifact

    @staticmethod
    def temporal_reach(params: SequenceClassifier.Params) -> int | None:
        """Frames on each side that the head reads of its input, the step's own reach.

        ``None`` for a head whose reach no option bounds (a GRU), or one that is not
        registered in the process that asks.
        """
        try:
            return get_head(params.head).temporal_reach(params.head_options())
        except ValueError:
            return None

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(
        GlobalModelParams[SequenceClassifierModelArtifact, LabeledTemplatesRef]
    ):
        templates: Annotated[
            LabeledTemplatesRef | None, Declared(_TEMPLATES_DESCRIPTION)
        ] = None
        model: Annotated[
            SequenceClassifierModelArtifact | None, Declared(_MODEL_DESCRIPTION)
        ] = Field(default_factory=SequenceClassifierModelArtifact)
        labels: Annotated[
            GroundTruthLabelsSource | None, Declared(_LABELS_DESCRIPTION)
        ] = None
        columns: Annotated[
            list[str] | None, Field(min_length=1), Declared(_COLUMNS_DESCRIPTION)
        ] = None
        # --- Network ---
        head: Annotated[str, Declared(_HEAD_DESCRIPTION)] = "gru"
        head_config: Annotated[
            SerializeAsAny[HeadConfig] | None, Declared(_HEAD_CONFIG_DESCRIPTION)
        ] = None
        head_revision: Annotated[
            int, Field(ge=0), Declared(_HEAD_REVISION_DESCRIPTION)
        ] = 0
        input_dropout: Annotated[
            float, Field(ge=0.0, lt=1.0), Declared(_INPUT_DROPOUT_DESCRIPTION)
        ] = 0.1
        clip: Annotated[float, Field(gt=0.0), Declared(_CLIP_DESCRIPTION)] = 10.0
        # --- Training ---
        class_weight: Annotated[
            Literal["sqrt", "balanced", "none"], Declared(_CLASS_WEIGHT_DESCRIPTION)
        ] = "sqrt"
        crop_length: Annotated[
            int, Field(ge=2), Declared(_CROP_LENGTH_DESCRIPTION, unit="frames")
        ] = 1024
        batch_size: Annotated[int, Field(ge=1), Declared(_BATCH_SIZE_DESCRIPTION)] = 16
        steps: Annotated[
            int, Field(ge=1), Declared(_STEPS_DESCRIPTION, unit="steps")
        ] = 2500
        learning_rate: Annotated[
            float, Field(gt=0.0), Declared(_LEARNING_RATE_DESCRIPTION)
        ] = 1e-3
        weight_decay: Annotated[
            float, Field(ge=0.0), Declared(_WEIGHT_DECAY_DESCRIPTION)
        ] = 1e-3
        pct_start: Annotated[
            float, Field(gt=0.0, lt=1.0), Declared(_PCT_START_DESCRIPTION)
        ] = 0.1
        grad_clip: Annotated[float, Field(gt=0.0), Declared(_GRAD_CLIP_DESCRIPTION)] = (
            1.0
        )
        min_run: Annotated[
            int, Field(ge=2), Declared(_MIN_RUN_DESCRIPTION, unit="frames")
        ] = 64
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        cv_folds: Annotated[
            int | None, Field(ge=2), Declared(_CV_FOLDS_DESCRIPTION)
        ] = None
        cv_seed: CvSeed = 0
        cv_unit: CvUnit = "sequence"
        cv_blocks_per_sequence: CvBlocksPerSequence = None
        cv_gap_frames: CvGapFrames = 0
        # Not HASH_EXCLUDE: a GPU and a CPU need not give one model.
        device: Annotated[Literal["cpu", "cuda"], Declared(_DEVICE_DESCRIPTION)] = (
            "cuda"
        )
        # HASH_EXCLUDE: where and how fast a fitted network predicts changes no model.
        apply_device: Annotated[
            Literal["cpu", "cuda"], HASH_EXCLUDE, Declared(_APPLY_DEVICE_DESCRIPTION)
        ] = "cpu"
        n_threads: Annotated[
            int, Field(ge=1), HASH_EXCLUDE, Declared(_N_THREADS_DESCRIPTION)
        ] = 1
        # --- Inference ---
        decision_threshold: DecisionThreshold = None
        default_class: Annotated[int, Declared(DEFAULT_CLASS_DESCRIPTION)]

        @field_validator("head")
        @classmethod
        def _head_is_registered(cls, value: str) -> str:
            _ = get_head(value)
            return value

        @field_validator("head_config", mode="before")
        @classmethod
        def _options_of_the_chosen_head(
            cls,
            value: Mapping[str, JsonValue] | HeadConfig | None,
            info: ValidationInfo,
        ) -> Mapping[str, JsonValue] | HeadConfig | None:
            """Read a mapping of options as those of the head that ``head`` names.

            ``head`` is declared first, so it is validated by now. A head that
            failed validation leaves the mapping to the error already raised.
            """
            head = info.data.get("head")
            if not isinstance(head, str) or not isinstance(value, Mapping):
                return value
            return get_head(head).config.model_validate(value)

        @model_validator(mode="after")
        def _a_fit_and_its_labels(self) -> Self:
            """Settle the options, and name the labels of a fit and of no loaded model."""
            self.head_config = self.head_options()
            if self.pct_start * self.steps <= 1.0:
                msg = (
                    f"pct_start {self.pct_start} of {self.steps} steps leaves no "
                    "step for the schedule to climb over, and the one-cycle "
                    "schedule needs more than one. Raise steps or pct_start"
                )
                raise ValueError(msg)
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
            )
            return self

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, with the block settings left out under ``cv_unit="sequence"``.

            A run that cross-validates by sequence, or not at all, has the identity it
            had before the block settings existed. ``check_cross_validation`` leaves
            them at their defaults then, so leaving them out loses nothing.
            """
            return drop_block_settings(super().identity_dump(), self.cv_unit)

        def head_options(self) -> HeadConfig:
            """Return the options of the chosen head: those given, else its defaults.

            Raises:
                ValueError: The options are those of another head.
            """
            config_class = get_head(self.head).config
            if self.head_config is None:
                return config_class()
            if not isinstance(self.head_config, config_class):
                msg = (
                    f"head_config holds {type(self.head_config).__name__} options, "
                    f"and head {self.head!r} takes {config_class.__name__}"
                )
                raise ValueError(msg)
            return self.head_config

    def __init__(
        self,
        inputs: SequenceClassifier.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.fits = self.params.model is None
        self._reset()

    def _reset(self) -> None:
        self._bundle: SequenceClassifierBundle | None = None
        self._fold_of: dict[tuple[str, str], int] = {}
        self._blocks: FoldLookup | None = None
        self._cv_table: pd.DataFrame | None = None
        self._train_entries: list[tuple[str, str]] | None = None
        self._label_files: DependencyLookup = {}
        self._networks: dict[int, nn.Module] = {}

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._reset()

        cached_path = run_root / "sequence_classifier_model.joblib"
        if cached_path.exists():
            bundle = SequenceClassifierModelArtifact().from_path(cached_path)
            self._adopt(bundle, folds=self.params.cv_folds is not None)
            return True

        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._adopt(bundle, folds=False)
            return True

        if self.params.templates is not None and "templates" in artifact_paths:
            entries = train_entries_of_templates(artifact_paths["templates"])
            if entries is None:
                msg = (
                    f"[{self.name}] the templates run left no "
                    "template_provenance.parquet beside its templates, so the "
                    "entries it held out are unknown. Rerun extract-labeled-templates"
                )
                raise ValueError(msg)
            self._train_entries = entries
            self._label_files = dependency_lookups.get("labels", {})
        return False

    def _adopt(self, bundle: SequenceClassifierBundle, *, folds: bool) -> None:
        """Take a loaded bundle, and its folds when *folds* asks for them."""
        current = head_source_digest(bundle["head"])
        if bundle["head_source_digest"] and current != bundle["head_source_digest"]:
            log.warning(
                "[%s] the code of head %r changed since this model was fitted "
                "(%s, now %s). The model applies the head as it is now",
                self.name,
                bundle["head"],
                bundle["head_source_digest"][:12],
                current[:12],
            )
        self._bundle = bundle
        cv = bundle.get("cv")
        if folds and cv is None:
            msg = (
                f"[{self.name}] cv_folds is set and the cached model has no folds. "
                "Remove sequence_classifier_model.joblib from the run directory "
                "to fit again"
            )
            raise RuntimeError(msg)
        if folds and cv is not None:
            by_block = self.params.cv_unit == "block"
            if by_block != (cv.get("unit") == "block"):
                msg = (
                    f"[{self.name}] cv_unit is {self.params.cv_unit!r} and the cached "
                    "model was fitted with the other unit. Remove "
                    "sequence_classifier_model.joblib from the run directory to fit "
                    "again"
                )
                raise RuntimeError(msg)
            if by_block:
                self._blocks = FoldLookup.by_block(block_list(cv.get("blocks", [])))
            else:
                self._fold_of = fold_map(cv["fold_of"])

    def fit(self, inputs: InputStream) -> None:
        if self._train_entries is None:
            msg = (
                f"[{self.name}] model {self.params.model.run_id!r} did not resolve "
                "to a stored model, so there is nothing to apply and the step "
                "does not fit in its place. Check the model reference"
                if self.params.model is not None
                else f"[{self.name}] No templates loaded, so check load_state."
            )
            raise RuntimeError(msg)
        torch = TORCH.load()
        p = self.params

        columns, runs, ranges = self._collect_runs(inputs)
        classes = sorted({int(c) for run in runs for c in np.unique(run.y)})
        # The runs hold the labels themselves until here, and class indices after.
        index_of = {label: index for index, label in enumerate(classes)}
        runs = [
            TrainingRun(
                run.entry,
                run.x,
                np.array([index_of[int(v)] for v in run.y], dtype=np.intp),
                run.first_frame,
            )
            for run in runs
        ]
        n_frames = sum(len(run.y) for run in runs)
        check_memory_headroom(
            n_frames * len(columns) * 4 * 3,
            self.name,
            f"holding {n_frames:,} training frames of {len(columns)} columns",
            "Name fewer columns with `columns`, or narrow the scope.",
        )

        plan: _FoldPlan | None = None
        if p.cv_folds is not None:
            plan = self._deal(runs, ranges, classes)
            self._cv_table = plan.table
            self._fold_of = dict(plan.fold_of)
            self._blocks = (
                FoldLookup.by_block(block_rows(plan.blocks)) if plan.blocks else None
            )

        total = 1 + (p.cv_folds or 0)
        models: list[SequenceClassifierModel] = []
        environment = self._environment(torch.__version__)
        with deterministic_torch(device=p.device, n_threads=p.n_threads):
            if plan is not None and p.cv_folds is not None:
                for fold in range(p.cv_folds):
                    self._before_fit(fold, total, whole="all runs")
                    models.append(
                        self._fit_model(
                            plan.training[fold],
                            columns,
                            classes,
                            fold,
                            sub_seed(p.random_state, "fold", fold),
                        )
                    )
                    self._after_fit(fold + 1, total)
            self._before_fit(total - 1, total, whole="all runs")
            models.append(self._fit_model(runs, columns, classes, None, p.random_state))
            self._after_fit(total, total)

        bundle: SequenceClassifierBundle = {
            "version": self.version,
            "head": p.head,
            "head_config": p.head_options().model_dump(),
            "head_revision": p.head_revision,
            "head_source_digest": head_source_digest(p.head),
            "input_dropout": p.input_dropout,
            "clip": p.clip,
            "columns": columns,
            "classes": classes,
            "models": models,
            "environment": environment,
        }
        if plan is not None and p.cv_folds is not None:
            cv: SequenceClassifierCv = {
                "folds": p.cv_folds,
                "seed": p.cv_seed,
                "fold_of": fold_rows(plan.fold_of),
            }
            if plan.blocks and p.cv_blocks_per_sequence is not None:
                cv["unit"] = "block"
                cv["blocks_per_sequence"] = p.cv_blocks_per_sequence
                cv["gap_frames"] = p.cv_gap_frames
                cv["blocks"] = block_rows(plan.blocks)
            bundle["cv"] = cv
        self._bundle = bundle

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        bundle = self._bundle
        if bundle is None:
            msg = (
                f"[{self.name}] Model not fitted and fit() or load_state() has not run."
            )
            raise RuntimeError(msg)
        columns = bundle["columns"]
        ensure_columns(df, [COLUMNS.frame_col, *columns])
        refuse_repeated_series_rows(df, self.name)

        choice = self._scoring_models(df, bundle)
        probs = np.full((len(df), len(bundle["classes"])), np.nan)
        order = df[COLUMNS.frame_col].to_numpy()
        features = df[columns].to_numpy(dtype=np.float64)
        for index in np.unique(choice):
            needed = choice == index
            model = bundle["models"][int(index)]
            mean = np.asarray(model["mean"], dtype=np.float64)
            std = np.asarray(model["std"], dtype=np.float64)
            frames = standardize(features, mean, std, bundle["clip"]).astype(np.float32)
            scored = np.full_like(probs, np.nan)
            network = self._network(int(index), bundle)
            for positions in iter_series_positions(df):
                if not needed[positions].any():
                    continue
                for start, stop in consecutive_runs(
                    order[positions], np.ones(len(positions), dtype=bool)
                ):
                    stretch = positions[start:stop]
                    scored[stretch] = predict_probabilities(
                        network, frames[stretch], device=self.params.apply_device
                    )
            probs[needed] = scored[needed]
        return assemble_prediction(
            df,
            probs,
            bundle["classes"],
            self.params.decision_threshold,
            self.params.default_class,
        )

    def save_state(self, run_root: Path) -> None:
        if self._bundle is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        if self._cv_table is not None:
            _ = write_parquet_atomic(self._cv_table, run_root / "cv_folds.parquet")
        write_bundle(run_root / "sequence_classifier_model.joblib", self._bundle)

    # --- Fitting ---

    def _collect_runs(
        self, inputs: InputStream
    ) -> tuple[list[str], list[TrainingRun], dict[tuple[str, str], tuple[int, int]]]:
        """Stream the training entries and keep each run of consecutive labeled frames.

        Returns:
            The columns the network reads, the runs, each with the class labels of
            its frames, and for cross-validation by block the labeled frames of
            each entry as ``(first, stop)``, from the first labeled frame of any of
            its series to one past the last. Without it the last is empty.

        Raises:
            ValueError: A train entry that has labels is missing from the stream.
            RuntimeError: No run is left, or the runs hold one class.
        """
        assert self._train_entries is not None
        p = self.params
        train = set(self._train_entries)
        columns: list[str] | None = None
        runs: list[TrainingRun] = []
        ranges: dict[tuple[str, str], tuple[int, int]] = {}
        by_block = p.cv_folds is not None and p.cv_unit == "block"
        seen: set[tuple[str, str]] = set()
        unlabeled_entries = 0
        too_short = 0
        for _, df in inputs():
            entry = entry_identity(df)
            if entry not in train:
                continue
            seen.add(entry)
            label_file = self._label_files.get(entry)
            if label_file is None:
                unlabeled_entries += 1
                continue
            ensure_columns(df, [COLUMNS.frame_col])
            refuse_repeated_series_rows(df, self.name)
            if columns is None:
                columns = (
                    list(p.columns) if p.columns is not None else feature_columns(df)
                )
            ensure_columns(df, columns)
            for _, rows in iter_series(df):
                labels = labels_for_rows(label_file, rows)
                frames = rows[COLUMNS.frame_col].to_numpy()
                matrix = rows[columns].to_numpy(dtype=np.float32)
                if by_block and (labels != UNLABELED).any():
                    labeled = frames[labels != UNLABELED]
                    first, last = ranges.get(entry, (int(labeled[0]), 0))
                    ranges[entry] = (
                        min(first, int(labeled.min())),
                        max(last, int(labeled.max()) + 1),
                    )
                for start, stop in consecutive_runs(frames, labels != UNLABELED):
                    if stop - start < p.min_run:
                        too_short += 1
                        continue
                    runs.append(
                        TrainingRun(
                            entry,
                            matrix[start:stop].copy(),
                            labels[start:stop].astype(np.intp),
                            int(frames[start]) if by_block else None,
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
        if unlabeled_entries or too_short:
            log.warning(
                "[%s] %d train entries have no label file and %d runs are shorter than "
                "min_run %d. None of them train",
                self.name,
                unlabeled_entries,
                too_short,
                p.min_run,
            )
        if columns is None or not runs:
            msg = (
                f"[{self.name}] no run of {p.min_run} or more consecutive labeled "
                f"frames is left to train on: {unlabeled_entries} train entries have "
                f"no label file and {too_short} runs are shorter than min_run"
            )
            raise RuntimeError(msg)
        found = {int(c) for run in runs for c in np.unique(run.y)}
        if len(found) < 2:
            msg = (
                f"[{self.name}] the training runs hold {len(found)} class "
                f"({sorted(found)}), and a classifier needs two or more"
            )
            raise RuntimeError(msg)
        return columns, runs, ranges

    def _deal(
        self,
        runs: list[TrainingRun],
        ranges: dict[tuple[str, str], tuple[int, int]],
        classes: list[int],
    ) -> _FoldPlan:
        """Deal the folds and name the runs each fold's model trains on.

        By sequence, a fold's model trains on the runs of every entry outside the
        fold. By block, it trains on every entry's runs but the frames of its own
        blocks and the ``cv_gap_frames`` on each side of them.
        """
        p = self.params
        assert p.cv_folds is not None
        if p.cv_unit == "block":
            assert p.cv_blocks_per_sequence is not None
            blocks = deal_blocks(
                self.name, ranges, p.cv_blocks_per_sequence, p.cv_folds, p.cv_seed
            )
            table = self._block_table(blocks, runs, classes)
            training = [
                self._outside_blocks(
                    runs, held_out_frames(blocks, fold, p.cv_gap_frames)
                )
                for fold in range(p.cv_folds)
            ]

            def enough_classes(
                fold: int, present: npt.NDArray[np.intp], _: npt.NDArray[np.intp]
            ) -> None:
                held = np.concatenate(
                    [r.y for r in training[fold]] or [np.empty(0, np.intp)]
                )
                if len(np.unique(held)) < 2:
                    msg = (
                        f"[{self.name}] holding out the blocks of fold {fold} and "
                        f"{p.cv_gap_frames} frames around them leaves training runs "
                        f"of {len(np.unique(held))} class (of {len(present)} outside "
                        "the blocks), and a classifier needs two. Lower "
                        "cv_gap_frames or cv_folds, or cut fewer, larger blocks"
                    )
                    raise ValueError(msg)

            check_folds(
                self.name,
                p.cv_folds,
                table,
                classes,
                rows="labeled frames",
                unit="block",
                per_fold=enough_classes,
            )
            return _FoldPlan(table, {}, blocks, training)
        fold_of = deal_folds(
            self.name, self._train_entries or [], p.cv_folds, p.cv_seed
        )
        table = self._fold_table(fold_of, runs, classes)
        check_folds(self.name, p.cv_folds, table, classes, rows="labeled frames")
        training = [
            [r for r in runs if fold_of[r.entry] != fold] for fold in range(p.cv_folds)
        ]
        return _FoldPlan(table, fold_of, [], training)

    def _outside_blocks(
        self,
        runs: list[TrainingRun],
        held: dict[tuple[str, str], list[tuple[int, int]]],
    ) -> list[TrainingRun]:
        """Cut the frames of *held* out of the runs, and keep the pieces of ``min_run`` frames or more."""
        kept: list[TrainingRun] = []
        for run in runs:
            ranges = held.get(run.entry)
            if not ranges:
                kept.append(run)
                continue
            assert run.first_frame is not None
            frames = run.first_frame + np.arange(len(run.y))
            keep = np.ones(len(frames), dtype=bool)
            for first, stop in ranges:
                keep &= (frames < first) | (frames >= stop)
            for start, stop in consecutive_runs(frames, keep):
                if stop - start >= self.params.min_run:
                    kept.append(
                        TrainingRun(
                            run.entry,
                            run.x[start:stop],
                            run.y[start:stop],
                            int(frames[start]),
                        )
                    )
        return kept

    @staticmethod
    def _block_table(
        blocks: list[CvBlock], runs: list[TrainingRun], classes: list[int]
    ) -> pd.DataFrame:
        """``cv_folds.parquet`` by block: the fold of every block and its labeled frames."""
        in_entry: dict[tuple[str, str], list[CvBlock]] = {}
        for block in blocks:
            in_entry.setdefault((block.group, block.sequence), []).append(block)
        counts = {
            (block.group, block.sequence, block.block): np.zeros(
                len(classes), dtype=np.int64
            )
            for block in blocks
        }
        for run in runs:
            assert run.first_frame is not None
            of_entry = in_entry[run.entry]
            starts = np.array([block.start for block in of_entry], dtype=np.int64)
            frames = run.first_frame + np.arange(len(run.y))
            where = np.clip(np.searchsorted(starts, frames, side="right") - 1, 0, None)
            tally = np.bincount(
                where * len(classes) + run.y, minlength=len(of_entry) * len(classes)
            ).reshape(len(of_entry), len(classes))
            for position, block in enumerate(of_entry):
                counts[(block.group, block.sequence, block.block)] += tally[position]
        frame = pd.DataFrame(
            list(counts.values()),
            index=pd.MultiIndex.from_tuples(
                list(counts), names=["group", "sequence", "block"]
            ),
            columns=classes,
        )
        return block_fold_table(blocks, frame, classes)

    @staticmethod
    def _fold_table(
        fold_of: dict[tuple[str, str], int], runs: list[TrainingRun], classes: list[int]
    ) -> pd.DataFrame:
        """``cv_folds.parquet``: the fold of every train entry and its labeled frames."""
        frame = pd.DataFrame(
            [
                (entry[0], entry[1], classes[int(c)], int(n))
                for run in runs
                for entry in [run.entry]
                for c, n in zip(*np.unique(run.y, return_counts=True))
            ],
            columns=["group", "sequence", "label", "count"],
        )
        counts = (
            frame.groupby(["group", "sequence", "label"])["count"]
            .sum()
            .unstack(fill_value=0)
        )
        return fold_table(fold_of, counts, classes)

    def _fit_model(
        self,
        runs: list[TrainingRun],
        columns: list[str],
        classes: list[int],
        fold: int | None,
        seed: int,
    ) -> SequenceClassifierModel:
        """Fit one network on *runs*, standardized by their own statistics."""
        p = self.params
        moments = RunningMoments(len(columns))
        for run in runs:
            moments.add(run.x.astype(np.float64))
        mean, std = moments.standardizer(min_std=_MIN_STD)
        frames = np.concatenate(
            [standardize(r.x.astype(np.float64), mean, std, p.clip) for r in runs]
        ).astype(np.float32)
        targets = np.concatenate([r.y for r in runs])
        counts = np.bincount(targets, minlength=len(classes)).astype(np.intp)
        sampler = Sampler.of(
            np.array([len(r.y) for r in runs], dtype=np.intp),
            p.crop_length,
            p.batch_size,
        )
        seed_torch(seed, device=p.device)
        network = assemble_network(
            p.head,
            p.head_options(),
            input_dropout=p.input_dropout,
            n_features=len(columns),
            n_classes=len(classes),
        )
        train_network(
            network,
            frames,
            targets,
            sampler,
            class_weights(counts, p.class_weight),
            steps=p.steps,
            learning_rate=p.learning_rate,
            weight_decay=p.weight_decay,
            pct_start=p.pct_start,
            grad_clip=p.grad_clip,
            seed=seed,
            device=p.device,
            check=self._check_cancel,
        )
        return {
            "fold": fold,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": network_state(network),
        }

    def _check_cancel(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.raise_if_cancelled()

    def _environment(self, torch_version: str) -> dict[str, str]:
        """Record what fitted the models, which no run id names."""
        import torch

        gpu = (
            torch.cuda.get_device_name()
            if self.params.device == "cuda" and torch.cuda.is_available()
            else ""
        )
        return {
            "torch": torch_version,
            "cuda": str(torch.version.cuda),
            "device": self.params.device,
            "gpu": gpu,
        }

    # --- Applying ---

    def _scoring_models(
        self, df: pd.DataFrame, bundle: SequenceClassifierBundle
    ) -> npt.NDArray[np.intp]:
        """Return the index of the model that scores each row of the entry *df* holds.

        By sequence the entry is scored whole, by the model of its fold or the full
        model. By block each row is scored by the model of the fold whose block holds
        its frame, and a row in no block by the full model.
        """
        full = len(bundle["models"]) - 1
        if "cv" not in bundle:
            return np.full(len(df), full, dtype=np.intp)
        if self._blocks is not None:
            folds = self._blocks.scoring_folds(df, self.name)
            return np.where(folds < 0, full, folds).astype(np.intp)
        if not self._fold_of:
            return np.full(len(df), full, dtype=np.intp)
        fold = out_of_fold_choice(df, self._fold_of, self.name)
        return np.full(len(df), full if fold is None else fold, dtype=np.intp)

    def _network(self, index: int, bundle: SequenceClassifierBundle) -> nn.Module:
        """Return the network of model *index* in evaluation mode, built once."""
        cached = self._networks.get(index)
        if cached is not None:
            return cached
        import torch

        config = get_head(bundle["head"]).config.model_validate(bundle["head_config"])
        network = assemble_network(
            bundle["head"],
            config,
            input_dropout=bundle["input_dropout"],
            n_features=len(bundle["columns"]),
            n_classes=len(bundle["classes"]),
        )
        weights = bundle["models"][index]["weights"]
        _ = network.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
        network.to(self.params.apply_device).eval()
        self._networks[index] = network
        return network
