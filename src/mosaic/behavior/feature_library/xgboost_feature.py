from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Annotated,
    ClassVar,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    final,
    override,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, field_validator, model_validator
from xgboost import XGBClassifier

from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Payload,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    GlobalModelParams,
    LabeledTemplatesRef,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)
from mosaic.core.params import HASH_EXCLUDE, Declared
from mosaic.core.pipeline.writers import write_parquet_atomic

from .classifier_common import (
    DEFAULT_CLASS_DESCRIPTION,
    CvBlock,
    CvBlocksPerSequence,
    CvGapFrames,
    CvSeed,
    CvUnit,
    DecisionThreshold,
    FitReporting,
    FoldLookup,
    assemble_prediction,
    balanced_class_weights,
    block_fold_table,
    block_list,
    block_rows,
    check_cross_validation,
    check_folds,
    deal_blocks,
    deal_folds,
    fold_map,
    fold_rows,
    fold_table,
    held_out_frames,
    write_bundle,
)
from .helpers import (
    check_memory_headroom,
    ensure_columns,
    feature_columns,
    finite_rows,
    out_of_fold_choice,
    refuse_unlabeled_templates,
    train_entries_of_templates,
    train_labeled_frames_of_templates,
)
from .registry import register_feature

log = logging.getLogger(__name__)

_SMOTE_MIN_ROWS = 6
"""The rows of a class that SMOTE needs to oversample it: its five neighbours and the row."""

_TEMPLATES_DESCRIPTION = (
    "The labeled templates artifact to train on. Mutually exclusive with model."
)

_MODEL_DESCRIPTION = (
    "A pre-fitted XgboostModelArtifact to load, skipping training. Mutually "
    "exclusive with templates."
)

_STRATEGY_DESCRIPTION = (
    "The classification strategy. multiclass fits one multi-class model. "
    "one_vs_rest fits one binary classifier per class."
)

_CLASS_WEIGHT_DESCRIPTION = (
    "Unset, every training sample is weighted equally. balanced weights "
    "each sample inversely to its class's frequency. A mapping from class "
    "to weight gives each sample its class's weight, 1 for a class it does "
    "not list, and applies no balanced weighting."
)

_USE_SMOTE_DESCRIPTION = (
    "Apply SMOTE oversampling to the training set, after any "
    "undersample_ratio undersampling."
)

_UNDERSAMPLE_RATIO_DESCRIPTION = (
    "Unset, no class is undersampled before training. Set, every class "
    "larger than the minority class is undersampled to this many times "
    "the minority class's size, never below the minority class's own "
    "size."
)

_N_ESTIMATORS_DESCRIPTION = "The number of boosting rounds the model fits."

_MAX_DEPTH_DESCRIPTION = "The maximum depth of each boosted tree."

_LEARNING_RATE_DESCRIPTION = (
    "The boosting learning rate (XGBoost's eta), scaling each tree's "
    "contribution to the ensemble."
)

_SUBSAMPLE_DESCRIPTION = (
    "The fraction of training rows sampled for each boosting round."
)

_COLSAMPLE_BYTREE_DESCRIPTION = "The fraction of feature columns sampled for each tree."

_RANDOM_STATE_DESCRIPTION = (
    "The random seed for XGBoost training and, when enabled, for "
    "undersampling and SMOTE."
)

_CV_FOLDS_DESCRIPTION = (
    "Unset, one model is fitted and scores every entry. Set to K (at least "
    "2), the training sequences of the templates are dealt into K folds and "
    "K more models are fitted, each on the training rows outside one fold. "
    "A training sequence is then scored by the model that did not see it, "
    "and every other entry by the model fitted on all training rows. The "
    "fold of each sequence is in cv_folds.parquet. With cv_unit block, the "
    "folds hold out blocks of frames and not whole sequences, which needs "
    "templates that record their frames (extract-labeled-templates 0.7 or "
    "later). Needs templates and is refused beside model."
)

_N_JOBS_DESCRIPTION = (
    "The number of threads each fit uses. Unset, XGBoost uses every core. "
    "It changes the time of a fit and not its model, so it is not part of "
    "the run id."
)

_DEVICE_DESCRIPTION = (
    "Which device fits the model. cpu fits on the processor. cuda fits on the "
    "GPU and fails where XGBoost has no CUDA device. The step stays on a CPU "
    "worker either way, so cuda runs only where the run is placed on a GPU "
    "machine. A fitted model predicts on the CPU, so a model fitted on cuda "
    "applies anywhere."
)


class XgboostCv(TypedDict):
    """The cross-validation half of a bundle, written when ``cv_folds`` is set.

    By sequence, ``fold_of`` holds ``(group, sequence, fold)`` for every train
    entry of the templates. By block, the keys ``unit``, ``blocks_per_sequence``,
    ``gap_frames`` and ``blocks`` are written too, ``fold_of`` is empty, and
    ``blocks`` holds ``(group, sequence, block, start, stop, fold)`` for every
    block. ``models[fold]`` was fitted on the train rows outside that fold, and
    outside the gap around its blocks, over the classes in ``classes[fold]``,
    which are its probability columns in order and are fewer than the bundle's
    when the rows held none of a class.
    """

    folds: int
    seed: int
    fold_of: list[tuple[str, str, int]]
    models: list[XGBClassifier | list[XGBClassifier]]
    classes: list[list[int]]
    unit: NotRequired[Literal["block"]]
    blocks_per_sequence: NotRequired[int]
    gap_frames: NotRequired[int]
    blocks: NotRequired[list[tuple[str, str, int, int, int, int]]]


class XgboostModelBundle(TypedDict):
    model: XGBClassifier | list[XGBClassifier]
    feature_columns: list[str]
    classes: list[int]
    strategy: str
    version: str
    cv: NotRequired[XgboostCv]


class XgboostModelArtifact(JoblibArtifact[XgboostModelBundle]):
    """Fitted XGBoost model bundle (xgboost_model.joblib)."""

    feature: str = "xgboost"
    pattern: str = "xgboost_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


@dataclass(frozen=True)
class _FoldPlan:
    """How the train rows of the templates fall into folds."""

    fold_of: dict[tuple[str, str], int]
    """The fold of every train entry the templates run saw, empty by block."""
    blocks: list[CvBlock]
    """Every block and its fold, empty by sequence."""
    training: list[np.ndarray]
    """``training[k]`` marks the train rows that the model without fold ``k`` fits."""
    table: pd.DataFrame
    """``cv_folds.parquet``: the fold and the rows of each train entry, or block."""


@final
@register_feature
class XgboostFeature(FitReporting):
    """XGBoost behavior classifier as a pipeline feature.

    Trains on labeled templates (from ExtractLabeledTemplates) and runs
    per-sequence inference. Supports multiclass and one-vs-rest strategies.

    **Cross-validation by sequence.** With ``cv_folds`` set to K, the train
    entries listed in the templates run's ``template_provenance.parquet`` are
    dealt into K folds by the sha256 of ``[cv_seed, group, sequence]``, and the
    fit trains K fold models, each on the train rows outside its fold, and then
    the full model on all train rows. ``apply`` scores an entry of a fold with
    its fold's model, so no training sequence is scored by a model that saw it,
    and every other entry, such as one of a test group, with the full model.
    ``evaluate-labels`` over the training entries is then the pooled
    out-of-fold score, and the probabilities are honest inputs to a second
    stage. ``cv_folds.parquet`` in the run directory gives each entry's fold and
    its rows per class. The score is not nested: the full model and the sample
    were chosen with every training sequence in view, as were label-free fits
    upstream of the templates.

    **Cross-validation by block.** ``cv_unit="block"`` holds out blocks of frames
    instead of whole sequences, so that a dataset with one training sequence is
    still cross-validated. Each train entry's labeled frames (``labeled_start`` up
    to ``labeled_stop`` in the templates run's provenance, the frames that
    ``sequence-classifier`` cuts from its own input) are cut into
    ``cv_blocks_per_sequence`` contiguous blocks, and the blocks of all entries are
    dealt to the folds by the sha256 of ``[cv_seed, group, sequence, block]``, so
    that this step and ``sequence-classifier`` deal alike and a
    ``probability-ensemble`` can pool them. A template row belongs to the block
    that holds its ``frame``. The model of a fold trains on the template rows
    outside its blocks and outside the ``cv_gap_frames`` frames on each side of
    them. A row is scored by the model of the fold whose block holds its frame,
    and a row in no block, and every entry outside the training set, by the model
    fitted on all train rows. ``cv_folds.parquet`` then has one row per block, with
    its ``start`` and ``stop`` frames (half-open) beside ``fold``. The templates
    must record their frames, so extract them again with
    ``extract-labeled-templates`` 0.7 or later. The templates are a sample of the
    labeled frames, so the rows of a block are those the sample drew.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.xgboost_feature.XgboostFeature.Params`.
    """

    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    fits = True  # learns from its inputs, so a held-out group must stay out of its fit
    name = "xgboost"
    # 0.2: ``nn_id`` and ``attention_target_id``, track ids, are no longer
    # fitted, from templates written before ``extract-labeled-templates``
    # 0.3 too (``BOOKKEEPING_COLS``).
    # 0.3: ``cv_folds`` and ``cv_seed`` cross-validate by sequence, ``class_weight``
    # takes a mapping, ``n_jobs`` bounds the threads of a fit, and the bundle is
    # written atomically. A 0.2 bundle still loads through ``model``.
    # 0.4: ``device`` picks the CPU or a CUDA device for a fit.
    version = "0.4"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # computes within a frame, so gains nothing
    temporal_reach = 0  # one row in, one row out
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    # Apply reads the template columns and writes one probability per class.
    requires: ClassVar[tuple[Requirement, ...]] = (Payload(),)
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("prob_"), Column("predicted_label")),
        conditional=META_COLUMNS_KEPT,
    )
    ModelArtifact = XgboostModelArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(GlobalModelParams[XgboostModelArtifact, LabeledTemplatesRef]):
        templates: Annotated[
            LabeledTemplatesRef | None, Declared(_TEMPLATES_DESCRIPTION)
        ] = None
        model: Annotated[XgboostModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = (
            Field(default_factory=XgboostModelArtifact)
        )
        strategy: Annotated[
            Literal["multiclass", "one_vs_rest"], Declared(_STRATEGY_DESCRIPTION)
        ] = "multiclass"
        decision_threshold: DecisionThreshold = None
        default_class: Annotated[int, Declared(DEFAULT_CLASS_DESCRIPTION)]
        class_weight: Annotated[
            Literal["balanced"] | Mapping[int, float] | None,
            Declared(_CLASS_WEIGHT_DESCRIPTION),
        ] = "balanced"
        use_smote: Annotated[bool, Declared(_USE_SMOTE_DESCRIPTION)] = False
        undersample_ratio: Annotated[
            float | None, Declared(_UNDERSAMPLE_RATIO_DESCRIPTION)
        ] = None
        n_estimators: Annotated[int, Declared(_N_ESTIMATORS_DESCRIPTION)] = Field(
            default=100, ge=1
        )
        max_depth: Annotated[int, Declared(_MAX_DEPTH_DESCRIPTION)] = Field(
            default=6, ge=1
        )
        learning_rate: Annotated[float, Declared(_LEARNING_RATE_DESCRIPTION)] = Field(
            default=0.1, gt=0
        )
        subsample: Annotated[float, Declared(_SUBSAMPLE_DESCRIPTION)] = Field(
            default=0.8, gt=0, le=1
        )
        colsample_bytree: Annotated[float, Declared(_COLSAMPLE_BYTREE_DESCRIPTION)] = (
            Field(default=0.8, gt=0, le=1)
        )
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        cv_folds: Annotated[int | None, Declared(_CV_FOLDS_DESCRIPTION)] = Field(
            default=None, ge=2
        )
        cv_seed: CvSeed = 0
        cv_unit: CvUnit = "sequence"
        cv_blocks_per_sequence: CvBlocksPerSequence = None
        cv_gap_frames: CvGapFrames = 0
        # Not HASH_EXCLUDE: a GPU histogram build need not give the model a CPU one does.
        device: Annotated[Literal["cpu", "cuda"], Declared(_DEVICE_DESCRIPTION)] = "cpu"
        # HASH_EXCLUDE: the threads of a fit change its time and not its model. One
        # fit predicted identically at 1, 2 and 4 threads.
        n_jobs: Annotated[int | None, HASH_EXCLUDE, Declared(_N_JOBS_DESCRIPTION)] = (
            Field(default=None, ge=1)
        )

        @field_validator("class_weight")
        @classmethod
        def _weights_are_positive(
            cls, value: Literal["balanced"] | Mapping[int, float] | None
        ) -> Literal["balanced"] | Mapping[int, float] | None:
            if isinstance(value, Mapping):
                for label, weight in value.items():
                    if not (math.isfinite(weight) and weight > 0):
                        msg = (
                            f"class_weight must be positive for every class, "
                            f"not {weight} for class {label}"
                        )
                        raise ValueError(msg)
            return value

        @model_validator(mode="after")
        def _cross_validation_fits_from_templates(self) -> Self:
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
            dumped = super().identity_dump()
            if self.cv_unit == "sequence":
                for name in ("cv_unit", "cv_blocks_per_sequence", "cv_gap_frames"):
                    del dumped[name]
            return dumped

    def __init__(
        self,
        inputs: XgboostFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._model: XGBClassifier | list[XGBClassifier] | None = None
        self._classes: list[int] | None = None
        self._templates: pd.DataFrame | None = None
        self._metrics: dict[str, object] | None = None
        # The train entries of the templates run, for the folds.
        self._train_entries: list[tuple[str, str]] | None = None
        self._labeled_frames: dict[tuple[str, str], tuple[int, int]] | None = None
        self._cv: XgboostCv | None = None
        self._fold_of: dict[tuple[str, str], int] = {}
        self._blocks: FoldLookup | None = None
        self._cv_table: pd.DataFrame | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._feature_columns = None
        self._model = None
        self._classes = None
        self._templates = None
        self._metrics = None
        self._train_entries = None
        self._labeled_frames = None
        self._cv = None
        self._fold_of = {}
        self._blocks = None
        self._cv_table = None

        # Branch 1: cached model in run_root
        cached_path = run_root / "xgboost_model.joblib"
        if cached_path.exists():
            bundle: XgboostModelBundle = XgboostModelArtifact().from_path(cached_path)
            self._adopt(bundle, folds=self.params.cv_folds is not None)
            return True

        # Branch 2: pre-fitted model from artifact_paths. Its folds are not used:
        # a fitted model scores every entry as the full model.
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._adopt(bundle, folds=False)
            return True

        # Branch 3: labeled templates to fit from
        if self.params.templates is not None and "templates" in artifact_paths:
            self._templates = self.params.templates.from_path(
                artifact_paths["templates"]
            )
            self._feature_columns = feature_columns(self._templates)
            if self.params.cv_folds is not None:
                self._train_entries = train_entries_of_templates(
                    artifact_paths["templates"]
                )
                if self.params.cv_unit == "block":
                    self._labeled_frames = train_labeled_frames_of_templates(
                        artifact_paths["templates"]
                    )
            return False

        return False

    def _adopt(self, bundle: XgboostModelBundle, *, folds: bool) -> None:
        """Take a loaded bundle's model, and its folds when *folds* asks for them."""
        self._model = bundle["model"]
        self._feature_columns = bundle["feature_columns"]
        self._classes = bundle["classes"]
        if not folds:
            return
        cv = bundle.get("cv")
        if cv is None:
            msg = (
                f"[{self.name}] cv_folds is set and the cached model has no folds. "
                "Remove xgboost_model.joblib from the run directory to fit again"
            )
            raise RuntimeError(msg)
        by_block = self.params.cv_unit == "block"
        if by_block != (cv.get("unit") == "block"):
            msg = (
                f"[{self.name}] cv_unit is {self.params.cv_unit!r} and the cached "
                "model was fitted with the other unit. Remove xgboost_model.joblib "
                "from the run directory to fit again"
            )
            raise RuntimeError(msg)
        self._cv = cv
        if by_block:
            self._blocks = FoldLookup.by_block(block_list(cv.get("blocks", [])))
        else:
            self._fold_of = fold_map(cv["fold_of"])

    def _undersampled_counts(self, counts: np.ndarray) -> list[int]:
        """The rows each class keeps under ``undersample_ratio``, given its rows in *counts*."""
        ratio = self.params.undersample_ratio
        if ratio is None:
            return [int(n) for n in counts]
        minority = int(counts.min())
        return [max(minority, min(int(n), int(minority * ratio))) for n in counts]

    def _undersample_then_smote(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multiclass-aware undersampling with optional SMOTE oversampling."""
        from imblearn.over_sampling import SMOTE  # type: ignore[import-untyped]

        classes, counts = np.unique(labels, return_counts=True)

        # Undersample majority classes
        if self.params.undersample_ratio is not None:
            keep_indices: list[np.intp] = []
            rng = np.random.default_rng(self.params.random_state)
            targets = self._undersampled_counts(counts)
            for cls, count, target in zip(classes, counts, targets):
                cls_indices = np.where(labels == cls)[0]
                if count > target:
                    chosen = rng.choice(cls_indices, size=target, replace=False)
                    keep_indices.extend(chosen.tolist())
                else:
                    keep_indices.extend(cls_indices.tolist())
            indices_array = np.array(keep_indices)
            features = features[indices_array]
            labels = labels[indices_array]

        # SMOTE oversampling. XGBoost reads NaN as a missing value, and SMOTE
        # interpolates between whole vectors and refuses one. The synthetic rows
        # come from the finite rows, and the rest are kept as they are.
        if self.params.use_smote:
            finite = finite_rows(features)
            smote = SMOTE(random_state=self.params.random_state)
            resampled = smote.fit_resample(features[finite], labels[finite])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            features = np.vstack([np.asarray(resampled[0]), features[~finite]])  # pyright: ignore[reportUnknownArgumentType]
            labels = np.concatenate([np.asarray(resampled[1]), labels[~finite]])  # pyright: ignore[reportUnknownArgumentType]

        return features, labels

    def _from_index(self, indices: np.ndarray) -> np.ndarray:
        """Column indices -> the class labels they stand for."""
        assert self._classes is not None
        return np.array([self._classes[int(i)] for i in indices], dtype=np.intp)

    def _sample_weight(
        self, y_index: np.ndarray, classes: list[int]
    ) -> np.ndarray | None:
        """The weight of each training sample, ``None`` when all weigh the same.

        Args:
            y_index: The class of each sample as a position in *classes*.
            classes: The class labels the fit sees, in column order.
        """
        weighting = self.params.class_weight
        if weighting == "balanced":
            weights = balanced_class_weights(
                np.bincount(y_index, minlength=len(classes))
            )
            return weights[y_index]
        if isinstance(weighting, Mapping):
            per_class = np.array([weighting.get(c, 1.0) for c in classes])
            return per_class[y_index]
        return None

    def _fit_model(
        self, x: np.ndarray, y: np.ndarray, classes: list[int]
    ) -> XGBClassifier | list[XGBClassifier]:
        """Fit one model on the rows *x* with labels *y*, over the labels *classes*.

        Every fit of a run goes through here, so the model of a fold is fitted
        as the full model is, on fewer rows. *classes* are the labels present in
        *y*, sorted. The model's probability columns are in this order.
        """
        p = self.params
        if p.strategy == "multiclass":
            # XGBoost's multiclass objective requires labels 0..num_class-1, while
            # these are behaviour ids -- and a training split need not contain every
            # one. A class missing here (an interaction that does not occur in the
            # training recordings) would otherwise leave `num_class` and the label
            # values disagreeing, and XGBClassifier refuses the fit outright. Index
            # space is what the rest of this class already speaks: `apply` maps a
            # probability column back through the classes, and so does the
            # one-vs-rest branch below.
            lookup = {c: i for i, c in enumerate(classes)}
            y_index = np.array([lookup[int(v)] for v in y], dtype=np.intp)
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(classes),
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                learning_rate=p.learning_rate,
                subsample=p.subsample,
                colsample_bytree=p.colsample_bytree,
                random_state=p.random_state,
                eval_metric="mlogloss",
                n_jobs=p.n_jobs,
                device=p.device,
            )
            model.fit(x, y_index, sample_weight=self._sample_weight(y_index, classes))
            return self._predict_on_cpu(model)

        # one_vs_rest strategy
        lookup = {c: i for i, c in enumerate(classes)}
        y_index = np.array([lookup[int(v)] for v in y], dtype=np.intp)
        weight = (
            self._sample_weight(y_index, classes)
            if isinstance(p.class_weight, Mapping)
            else None
        )
        models: list[XGBClassifier] = []
        for cls in classes:
            binary_y = (y == cls).astype(int)
            ovr_model = XGBClassifier(
                objective="binary:logistic",
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                learning_rate=p.learning_rate,
                subsample=p.subsample,
                colsample_bytree=p.colsample_bytree,
                random_state=p.random_state,
                eval_metric="logloss",
                n_jobs=p.n_jobs,
                device=p.device,
            )
            if p.class_weight == "balanced":
                n_pos = int(binary_y.sum())
                n_neg = len(binary_y) - n_pos
                scale = n_neg / max(n_pos, 1)
                ovr_model.set_params(scale_pos_weight=scale)
            ovr_model.fit(x, binary_y, sample_weight=weight)
            models.append(self._predict_on_cpu(ovr_model))
        return models

    @staticmethod
    def _predict_on_cpu(model: XGBClassifier) -> XGBClassifier:
        """Move a fitted model to the CPU, so that a bundle applies on any machine.

        The trees are the same, and a booster left on a CUDA device would predict
        there, or warn and fall back, on a machine that has none.
        """
        model.set_params(device="cpu")
        return model

    def _fit_rows(
        self, x: np.ndarray, y: np.ndarray, classes: list[int] | None = None
    ) -> tuple[XGBClassifier | list[XGBClassifier], list[int]]:
        """Rebalance the rows if asked, then fit them. Returns the model and its classes."""
        if self.params.use_smote or self.params.undersample_ratio is not None:
            x, y = self._undersample_then_smote(x, y)
        present = sorted(int(c) for c in np.unique(y)) if classes is None else classes
        return self._fit_model(x, y, present), present

    def _check_fit_memory(self, n_rows: int, n_features: int, n_classes: int) -> None:
        """Refuse a fit whose working copies of the training rows would not fit.

        Per cell of the train rows the peak holds a float64 matrix (8 bytes), one
        working copy of it (8: the rows of the frame, then the rows outside a
        fold) and XGBoost's quantized copy of the rows a fit uses (6), and the
        models are counted at their largest.
        """
        p = self.params
        fits = 1 + (p.cv_folds or 0)
        trees = fits * p.n_estimators * max(n_classes, 2) * 2 ** (p.max_depth + 1)
        check_memory_headroom(
            n_rows * n_features * 22 + trees * 64,
            self.name,
            f"fitting {fits} model{'s' if fits > 1 else ''} on {n_rows:,} rows of "
            f"{n_features} columns",
            "Draw fewer templates (n_per_class in extract-labeled-templates), name "
            "fewer columns there with `columns`, or lower cv_folds.",
        )

    def _plan_folds(self, train: pd.DataFrame, labels: np.ndarray) -> _FoldPlan:
        """Deal the train entries, or their blocks, into folds and place every train row in one.

        Args:
            train: The train rows of the templates.
            labels: Their labels.

        Raises:
            ValueError: The templates lack the identity of their rows or their
                provenance, name an entry the provenance does not list, hold
                fewer entries (or blocks) than folds, or leave a fold's training
                rows with fewer than two classes. By block, also templates that
                record no frames.
        """
        k = self.params.cv_folds
        assert k is not None
        missing = [c for c in ("group", "sequence") if c not in train.columns]
        if missing:
            msg = (
                f"[{self.name}] cv_folds names the fold of each template row by "
                f"its group and sequence, and these templates lack {missing}. "
                "Rerun extract-labeled-templates 0.6 or later and point "
                "templates at the new run"
            )
            raise ValueError(msg)
        if self._train_entries is None:
            msg = (
                f"[{self.name}] cv_folds deals the train entries that "
                "template_provenance.parquet lists beside the templates, and "
                "found none. Rerun extract-labeled-templates 0.5 or later"
            )
            raise ValueError(msg)
        keys = list(zip(map(str, train["group"]), map(str, train["sequence"])))
        classes = sorted(int(c) for c in np.unique(labels))
        if self.params.cv_unit == "block":
            return self._plan_block_folds(train, keys, labels, classes)
        fold_of = deal_folds(self.name, self._train_entries, k, self.params.cv_seed)

        row_fold = np.array([fold_of.get(key, -1) for key in keys], dtype=np.intp)
        self._refuse_unlisted_entries(keys, row_fold < 0)

        rows = pd.DataFrame(
            {
                "group": [key[0] for key in keys],
                "sequence": [key[1] for key in keys],
                "label": labels,
            }
        )
        per_class = (
            rows.groupby(["group", "sequence", "label"]).size().unstack(fill_value=0)
        )
        table = fold_table(fold_of, per_class, classes)
        check_folds(
            self.name,
            k,
            table,
            classes,
            rows="template rows",
            per_fold=self._check_smote_fold if self.params.use_smote else None,
        )
        return _FoldPlan(fold_of, [], [row_fold != fold for fold in range(k)], table)

    def _refuse_unlisted_entries(
        self, keys: list[tuple[str, str]], unknown: np.ndarray
    ) -> None:
        """Refuse template rows that belong to an entry with no fold.

        Args:
            keys: The ``(group, sequence)`` of each train row.
            unknown: Marks the rows whose entry the provenance does not list, or
                gives no block.

        Raises:
            ValueError: A row is marked.
        """
        gone = sorted({key for key, bad in zip(keys, unknown) if bad})
        if gone:
            msg = (
                f"[{self.name}] the template rows of {len(gone)} "
                f"entr{'y' if len(gone) == 1 else 'ies'}, such as {gone[0]}, "
                "belong to entries the provenance does not list as train entries, "
                "or fall outside the labeled frames it records for them. The "
                "templates and their provenance disagree"
            )
            raise ValueError(msg)

    def _plan_block_folds(
        self,
        train: pd.DataFrame,
        keys: list[tuple[str, str]],
        labels: np.ndarray,
        classes: list[int],
    ) -> _FoldPlan:
        """Cut each train entry's labeled frames into blocks, deal them, and place every row in one.

        Raises:
            ValueError: The templates or their provenance record no frames, a row
                falls in no block, there are fewer blocks than folds, or a fold's
                training rows, outside its blocks and their gap, hold fewer than
                two classes.
        """
        p = self.params
        k = p.cv_folds
        assert k is not None and p.cv_blocks_per_sequence is not None
        if "frame" not in train.columns or not self._labeled_frames:
            what = (
                "these templates record no frame for their rows"
                if "frame" not in train.columns
                else "the provenance beside these templates records no labeled frames"
            )
            msg = (
                f"[{self.name}] cv_unit='block' cuts each sequence into blocks of "
                f"frames and places each template row in one by its frame, and "
                f"{what}. They come from extract-labeled-templates before 0.7. "
                "Run extract-labeled-templates again, and point templates at the "
                "new run"
            )
            raise ValueError(msg)
        blocks = deal_blocks(
            self.name, self._labeled_frames, p.cv_blocks_per_sequence, k, p.cv_seed
        )
        frames = train["frame"].to_numpy(dtype=np.int64)
        of_entry: dict[tuple[str, str], list[CvBlock]] = {}
        for block in blocks:
            of_entry.setdefault((block.group, block.sequence), []).append(block)
        listed: dict[tuple[str, str], list[int]] = {}
        for position, key in enumerate(keys):
            listed.setdefault(key, []).append(position)
        positions = {key: np.array(rows, dtype=np.intp) for key, rows in listed.items()}
        row_block = np.full(len(keys), -1, dtype=np.intp)
        for entry, rows in positions.items():
            entry_blocks = of_entry.get(entry, [])
            if not entry_blocks:
                continue
            starts = np.array([b.start for b in entry_blocks], dtype=np.int64)
            stops = np.array([b.stop for b in entry_blocks], dtype=np.int64)
            at_frame: npt.NDArray[np.int64] = frames[rows]
            after: npt.NDArray[np.intp] = np.searchsorted(
                starts, at_frame, side="right"
            )
            at = np.maximum(after - 1, 0)
            inside = (at_frame >= starts[at]) & (at_frame < stops[at])
            row_block[rows[inside]] = np.array(
                [b.block for b in entry_blocks], dtype=np.intp
            )[at[inside]]
        self._refuse_unlisted_entries(keys, row_block < 0)

        table = block_fold_table(
            blocks,
            pd.DataFrame(
                {
                    "group": [key[0] for key in keys],
                    "sequence": [key[1] for key in keys],
                    "block": row_block,
                    "label": labels,
                }
            )
            .groupby(["group", "sequence", "block", "label"])
            .size()
            .unstack(fill_value=0),
            classes,
        )
        training: list[np.ndarray] = []
        for fold in range(k):
            fits = np.ones(len(keys), dtype=bool)
            for entry, held in held_out_frames(blocks, fold, p.cv_gap_frames).items():
                rows = positions.get(entry)
                if rows is None:
                    continue
                for first, stop in held:
                    fits[rows[(frames[rows] >= first) & (frames[rows] < stop)]] = False
            training.append(fits)

        def enough_classes(fold: int, _: np.ndarray, __: np.ndarray) -> None:
            present, counts = np.unique(labels[training[fold]], return_counts=True)
            if len(present) < 2:
                msg = (
                    f"[{self.name}] holding out the blocks of fold {fold} and "
                    f"{p.cv_gap_frames} frames around them leaves training rows "
                    f"of {len(present)} class, and a classifier needs two. Lower "
                    "cv_gap_frames or cv_folds, or cut fewer, larger blocks"
                )
                raise ValueError(msg)
            if p.use_smote:
                self._check_smote_fold(fold, present, counts)

        check_folds(
            self.name,
            k,
            table,
            classes,
            rows="template rows",
            unit="block",
            per_fold=enough_classes,
        )
        return _FoldPlan({}, blocks, training, table)

    def _check_smote_fold(
        self, fold: int, present: np.ndarray, counts: np.ndarray
    ) -> None:
        """Refuse a fold in which SMOTE could not oversample a class.

        A class that SMOTE oversamples from fewer rows fails in imblearn, late
        and without naming the fold, where the full fit would have passed. SMOTE
        leaves the largest class as it is.
        """
        kept = self._undersampled_counts(counts)
        sparse = [
            f"class {int(c)} ({int(n)} row{'' if n == 1 else 's'})"
            for c, n in zip(present, kept)
            if n < _SMOTE_MIN_ROWS and n < max(kept)
        ]
        if sparse:
            msg = (
                f"[{self.name}] cv_folds={self.params.cv_folds} with use_smote: the "
                f"train rows outside fold {fold} hold {', '.join(sparse)}, and SMOTE "
                f"needs {_SMOTE_MIN_ROWS} rows of a class to oversample it. Lower "
                "cv_folds, draw more rows of that class, or leave use_smote off"
            )
            raise ValueError(msg)

    def fit(self, inputs: InputStream) -> None:
        if self._templates is None:
            msg = "No templates loaded -- call load_state first"
            raise RuntimeError(msg)
        if self._feature_columns is None:
            msg = "No feature columns determined"
            raise RuntimeError(msg)

        templates = self._templates
        ensure_columns(templates, ["label", "split"])
        refuse_unlabeled_templates(templates, self.name)
        train_mask = templates["split"] == "train"
        test_mask = templates["split"] == "test"

        self._check_fit_memory(
            int(train_mask.sum()),
            len(self._feature_columns),
            int(templates.loc[train_mask, "label"].nunique()),
        )

        train_df = templates.loc[train_mask]
        test_df = templates.loc[test_mask]

        x_train = train_df[self._feature_columns].to_numpy(dtype=np.float64)
        y_train = train_df["label"].to_numpy(dtype=np.intp)
        classes = sorted(int(c) for c in np.unique(y_train))
        plan = (
            None
            if self.params.cv_folds is None
            else self._plan_folds(train_df, y_train)
        )
        del train_df

        n_folds = self.params.cv_folds or 0
        total = 1 + n_folds
        fold_models: list[XGBClassifier | list[XGBClassifier]] = []
        fold_classes: list[list[int]] = []
        if plan is not None:
            for fold in range(n_folds):
                self._before_fit(fold, total, whole="all rows")
                outside = plan.training[fold]
                model, present = self._fit_rows(x_train[outside], y_train[outside])
                fold_models.append(model)
                fold_classes.append(present)
                self._after_fit(fold + 1, total)
        self._before_fit(total - 1, total, whole="all rows")
        full, _ = self._fit_rows(x_train, y_train, classes)
        self._after_fit(total, total)

        self._classes = classes
        self._model = full
        if plan is not None:
            assert self.params.cv_folds is not None
            cv: XgboostCv = {
                "folds": self.params.cv_folds,
                "seed": self.params.cv_seed,
                "fold_of": fold_rows(plan.fold_of),
                "models": fold_models,
                "classes": fold_classes,
            }
            if plan.blocks and self.params.cv_blocks_per_sequence is not None:
                cv["unit"] = "block"
                cv["blocks_per_sequence"] = self.params.cv_blocks_per_sequence
                cv["gap_frames"] = self.params.cv_gap_frames
                cv["blocks"] = block_rows(plan.blocks)
                self._blocks = FoldLookup.by_block(cv["blocks"])
            self._cv = cv
            self._fold_of = dict(plan.fold_of)
            self._cv_table = plan.table

        # Evaluate on test split if available
        if len(test_df) > 0:
            from sklearn.metrics import classification_report as _cr  # pyright: ignore[reportUnknownVariableType]

            x_test = test_df[self._feature_columns].to_numpy(dtype=np.float64)
            y_test = test_df["label"].to_numpy(dtype=np.intp)

            if self.params.strategy == "multiclass":
                assert isinstance(self._model, XGBClassifier)
                # The model was fitted in index space, so its predictions are
                # column indices; the report has to be in label space. `predict`
                # of a two-class softprob model returns the probabilities and
                # not the class, so the class is read from them.
                y_pred: np.ndarray = self._from_index(
                    np.argmax(self._model.predict_proba(x_test), axis=1)
                )
            else:
                assert isinstance(self._model, list)
                test_probs = np.column_stack(
                    [m.predict_proba(x_test)[:, 1] for m in self._model]
                )
                y_pred = self._from_index(np.argmax(test_probs, axis=1))

            # `zero_division=0` because the splits are by sequence, so the test
            # split can hold a class the training one did not -- and precision or
            # recall for a class the model was never given is genuinely 0, not an
            # undefined value worth warning about on every fit.
            report: dict[str, object] = _cr(  # pyright: ignore[reportUnknownVariableType, reportAssignmentType]
                y_test, y_pred, output_dict=True, zero_division=0
            )
            self._metrics = report

    def _scoring_model(
        self, df: pd.DataFrame
    ) -> tuple[XGBClassifier | list[XGBClassifier], list[int]]:
        """The model that scores the entry *df* holds, and its classes.

        Without folds, the model. With them, the fold's model for a training
        sequence, and the full model for any other entry. By block the entry is
        scored by row: see :meth:`_scoring_folds`.
        """
        assert self._model is not None and self._classes is not None
        if self._cv is None or self._blocks is not None:
            return self._model, self._classes
        fold = out_of_fold_choice(df, self._fold_of, self.name)
        if fold is None:
            return self._model, self._classes
        return self._cv["models"][fold], self._cv["classes"][fold]

    def _scoring_folds(self, df: pd.DataFrame) -> np.ndarray | None:
        """The fold whose model scores each row of the entry *df* holds, by block.

        Returns:
            The fold of each row, ``-1`` for a row in no block, which the full
            model scores. ``None`` when the run does not cross-validate by block.
        """
        if self._blocks is None:
            return None
        ensure_columns(df, ["frame"])
        return self._blocks.scoring_folds(df, self.name)

    def _probabilities(
        self,
        model: XGBClassifier | list[XGBClassifier],
        model_classes: list[int],
        x: np.ndarray,
    ) -> np.ndarray:
        """The probability of every class of the run, 0 for one the model never saw."""
        assert self._classes is not None
        if isinstance(model, list):
            probs = np.column_stack([m.predict_proba(x)[:, 1] for m in model])
        else:
            probs = model.predict_proba(x)
        if model_classes == self._classes:
            return probs
        full = np.zeros((probs.shape[0], len(self._classes)))
        full[:, [self._classes.index(c) for c in model_classes]] = probs
        return full

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._feature_columns is None:
            msg = "No feature columns -- model not loaded"
            raise RuntimeError(msg)
        if self._model is None:
            msg = "No model fitted -- call fit() or load_state() first"
            raise RuntimeError(msg)
        if self._classes is None:
            msg = "No classes determined -- model not loaded"
            raise RuntimeError(msg)

        ensure_columns(df, self._feature_columns)

        feat_matrix = df[self._feature_columns].to_numpy(dtype=np.float64)

        # Get probabilities
        folds = self._scoring_folds(df)
        if folds is None:
            model, model_classes = self._scoring_model(df)
            probs = self._probabilities(model, model_classes, feat_matrix)
        else:
            assert self._cv is not None
            probs = np.zeros((len(df), len(self._classes)))
            for fold in np.unique(folds):
                rows = folds == fold
                if fold < 0:
                    model, model_classes = self._model, self._classes
                else:
                    model = self._cv["models"][int(fold)]
                    model_classes = self._cv["classes"][int(fold)]
                probs[rows] = self._probabilities(
                    model, model_classes, feat_matrix[rows]
                )

        return assemble_prediction(
            df,
            probs,
            self._classes,
            self.params.decision_threshold,
            self.params.default_class,
        )

    def save_state(self, run_root: Path) -> None:
        if (
            self._model is None
            or self._feature_columns is None
            or self._classes is None
        ):
            return
        run_root.mkdir(parents=True, exist_ok=True)

        # The bundle goes last: its presence is what makes the run a cache, so the
        # files that belong to it are in place before it is.
        if self._metrics is not None:
            import json

            (run_root / "reports.json").write_text(json.dumps(self._metrics, indent=2))
            summary = pd.DataFrame(self._metrics).T
            summary.to_csv(run_root / "summary.csv")
        if self._cv_table is not None:
            _ = write_parquet_atomic(self._cv_table, run_root / "cv_folds.parquet")

        bundle: XgboostModelBundle = {
            "model": self._model,
            "feature_columns": self._feature_columns,
            "classes": self._classes,
            "strategy": self.params.strategy,
            "version": self.version,
        }
        if self._cv is not None:
            bundle["cv"] = self._cv
        write_bundle(run_root / "xgboost_model.joblib", bundle)
