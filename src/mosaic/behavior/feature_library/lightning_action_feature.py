"""Lightning-action supervised temporal action segmentation feature.

Wraps the ``lightning-action`` package (Paninski lab, MIT license) as a
mosaic global feature.  Trains a temporal neural network classifier
(DilatedTCN, RNN, or TemporalMLP) on runs of consecutive frames and predicts
per-frame action probabilities with temporal context.

A temporal model reads one animal's frames in time order. The feature therefore
splits each table into series, one per individual or per ordered pair, and trains
and predicts on each series as a sequence of its own.

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
    runtime_checkable,
)

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

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
from mosaic.optional_dependency import OptionalModule, Prerequisite

from .classifier_common import (
    DEFAULT_CLASS_DESCRIPTION,
    DecisionThreshold,
    RunningMoments,
    assemble_prediction,
    standardize,
    write_bundle,
)
from .helpers import (
    consecutive_runs,
    ensure_columns,
    entry_identity,
    feature_columns,
    iter_series,
    iter_series_positions,
    refuse_repeated_series_rows,
    train_entries_of_templates,
)
from .registry import register_feature

if TYPE_CHECKING:
    import numpy.typing as npt

log = logging.getLogger(__name__)


# --- Import guard ---

LIGHTNING_ACTION: Final = OptionalModule(
    "lightning_action", "lightning-action", "the lightning-action classifier"
)
"""The lightning-action package, imported when the feature is constructed."""


# --- Model artifact ---


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
class _TrainingRuns:
    """What a pass over the training entries left in a directory of runs."""

    runs_dir: Path
    columns: list[str]
    moments: RunningMoments
    classes: list[int]
    n_runs: int


# --- Feature class ---


_TEMPLATES_DESCRIPTION = (
    "The extract-labeled-templates run whose train entries the model fits on. "
    "Only its template_provenance.parquet is read: the model trains on the "
    "frame runs of those entries, not on the sampled templates. An entry the "
    "run lists as test, or does not list, never trains. Mutually exclusive "
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

_RANDOM_STATE_DESCRIPTION = "The random seed for model initialization and training."

_DEFAULT_CLASS_DESCRIPTION = (
    DEFAULT_CLASS_DESCRIPTION
    + " It is also assigned to a frame that belongs to no series because its "
    "identity columns are missing."
)


@final
@register_feature
class LightningActionFeature:
    """Supervised temporal action segmentation via lightning-action.

    Trains a temporal neural network classifier (DilatedTCN, RNN, or
    TemporalMLP head + linear classifier) on runs of consecutive frames and
    predicts per-frame action probabilities.

    A table is read as series: one per ``id``, or one per ordered pair
    (``id1``, ``id2``, ``perspective``) when it has no ``id``. Training uses the
    runs of consecutive labeled frames of each series, in frame order, of the
    entries that the ``templates`` run assigned to ``train``. ``fit`` streams the
    step's inputs and aligns each series to ``labels``, so the step's scope is
    its training set, and a model applied to further entries is named through
    ``model`` by a second step. Frames are standardized by statistics computed
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
    version = "0.5"
    parallelizable = True
    # A fit reads its own stream, so its scope is its training set. A run given a
    # ``model`` fits nothing, and ``__init__`` clears this for it.
    scope_dependent = True
    accepts_overlap = False  # reads each series as its own sequence
    consumed_roots: tuple[str, ...] = ()
    prerequisites: tuple[Prerequisite, ...] = (LIGHTNING_ACTION,)
    emits: EmitsLevel = "as-input"
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
            return self

    def __init__(
        self,
        inputs: LightningActionFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        _ = LIGHTNING_ACTION.load()
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.scope_dependent = self.params.model is None

        self._feature_columns: list[str] | None = None
        self._feature_mean: npt.NDArray[np.float64] | None = None
        self._feature_std: npt.NDArray[np.float64] | None = None
        self._classes: list[int] | None = None
        self._class_names: list[str] | None = None
        self._sequence_length: int | None = None
        self._sequence_pad: int | None = None
        self._train_entries: set[tuple[str, str]] | None = None
        self._label_files: DependencyLookup = {}
        self._model_dir: Path | None = None
        self._la_model: _PackageModel | None = None  # cached package Model
        self._config: dict[str, object] | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._feature_columns = None
        self._feature_mean = None
        self._feature_std = None
        self._classes = None
        self._class_names = None
        self._sequence_length = None
        self._sequence_pad = None
        self._train_entries = None
        self._label_files = {}
        self._model_dir = None
        self._la_model = None
        self._config = None

        # Branch 1: cached model in run_root
        cached_path = run_root / "lightning_action_model.joblib"
        if cached_path.exists():
            bundle: LightningActionModelBundle = (
                LightningActionModelArtifact().from_path(cached_path)
            )
            self._load_bundle(bundle, run_root)
            return True

        # Branch 2: pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            bundle = self.params.model.from_path(artifact_paths["model"])
            self._load_bundle(bundle, artifact_paths["model"].parent)
            return True

        # Branch 3: the entries of the templates run to fit on
        if self.params.templates is not None and "templates" in artifact_paths:
            train_entries = train_entries_of_templates(artifact_paths["templates"])
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
        from lightning_action.data.utils import compute_sequence_pad

        model_class = _package_model_class()
        if self._train_entries is None:
            msg = "[lightning-action] No templates loaded, so check load_state."
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
            data_dir = tmpdir / "data"
            output_dir = tmpdir / "output"

            runs = self._write_runs(inputs, tmpdir / "runs")
            mean, std = runs.moments.standardizer()
            self._classes = runs.classes
            self._class_names = [str(c) for c in runs.classes]
            self._feature_columns = runs.columns
            self._feature_mean = mean
            self._feature_std = std
            self._sequence_length = p.sequence_length
            self._sequence_pad = pad

            print(
                f"[lightning-action] Training: {runs.n_runs} runs, "
                f"{len(runs.columns)} features, {len(runs.classes)} classes, "
                f"head={p.head}, epochs={p.num_epochs}",
                file=sys.stderr,
            )

            # Write temp CSVs in lightning-action layout
            self._write_training_csvs(runs, data_dir)

            # Build config
            config = self._build_config(
                len(runs.columns), len(runs.classes), str(data_dir)
            )

            # Train
            model = model_class.from_config(config)
            model.train(output_dir=str(output_dir), post_inference=False)

            # Persist trained model to a temp location (save_state copies to run_root)
            self._model_dir = Path(tempfile.mkdtemp(prefix="la_model_"))
            shutil.copytree(output_dir, self._model_dir / "model", dirs_exist_ok=True)
            self._la_model = _predicting_in_process(
                model_class.from_dir(str(self._model_dir / "model"))
            )
            self._config = config

        print("[lightning-action] Training complete.", file=sys.stderr)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
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

        matrix = standardize(
            df[feat_cols].to_numpy(dtype=np.float64),
            self._feature_mean,
            self._feature_std,
        )
        probs = self._predict_series(df, matrix)

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

        # Copy model directory into run_root
        dest = run_root / "lightning_action_model"
        src = self._model_dir / "model"
        if not src.exists():
            src = self._model_dir / "lightning_action_model"
        if src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

        # Save metadata bundle
        bundle: LightningActionModelBundle = {
            "model_dir": "lightning_action_model",
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
        write_bundle(run_root / "lightning_action_model.joblib", bundle)

        # Clean up temp model dir (if it's separate from run_root)
        if self._model_dir != run_root and self._model_dir.exists():
            shutil.rmtree(self._model_dir, ignore_errors=True)
        self._model_dir = run_root
        self._la_model = None  # force reload from new location on next apply

    # --- Helpers ---

    def _load_bundle(self, bundle: LightningActionModelBundle, base_dir: Path) -> None:
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
        self._la_model = None  # lazy-load on first apply()

    def _write_runs(self, inputs: InputStream, runs_dir: Path) -> _TrainingRuns:
        """Stream the training entries and keep each of their frame runs on disk.

        A run is a stretch of consecutive labeled frames of one series. It is
        kept in whole windows from its first frame, which is how the package
        cuts it, and a run with no whole window is left out. The statistics of
        the frames that are kept accumulate as the runs are written, so the
        stream is read once.

        Raises:
            ValueError: A train entry that has labels is missing from the stream.
            RuntimeError: No run is left, or the runs hold one class.
        """
        assert self._train_entries is not None
        window = self.params.sequence_length
        runs_dir.mkdir(parents=True)

        columns: list[str] | None = None
        moments: RunningMoments | None = None
        classes: set[int] = set()
        seen: set[tuple[str, str]] = set()
        unlabeled_entries = 0
        too_short = 0
        past_last_window = 0
        n_runs = 0

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
                moments = RunningMoments(len(columns))
            else:
                ensure_columns(df, columns)
            assert moments is not None

            for _, rows in iter_series(df):
                labels = labels_for_rows(label_file, rows)
                frames = rows[COLUMNS.frame_col].to_numpy()
                matrix = rows[columns].to_numpy(dtype=np.float64)
                for start, stop in consecutive_runs(frames, labels != UNLABELED):
                    whole = (stop - start) // window * window
                    if whole == 0:
                        too_short += 1
                        continue
                    past_last_window += stop - start - whole
                    stop = start + whole
                    np.savez(
                        runs_dir / f"run_{n_runs:06d}.npz",
                        x=matrix[start:stop],
                        y=labels[start:stop],
                    )
                    moments.add(matrix[start:stop])
                    classes.update(int(c) for c in np.unique(labels[start:stop]))
                    n_runs += 1

        absent = sorted(
            entry
            for entry in self._train_entries
            if entry in self._label_files and entry not in seen
        )
        if absent:
            shown = ", ".join(f"{group}/{sequence}" for group, sequence in absent[:5])
            msg = (
                f"[{self.name}] {len(absent)} train entries of the templates run "
                f"have labels and are not in this step's scope: {shown}. The model "
                "would train on fewer entries than its templates name. Add them to "
                "the scope, or apply a model fitted elsewhere through `model`"
            )
            raise ValueError(msg)
        if unlabeled_entries or too_short or past_last_window:
            print(
                f"[lightning-action] {unlabeled_entries} train entries have no label "
                f"file, {too_short} runs are shorter than sequence_length {window}, "
                f"and {past_last_window} frames lie past the last whole window of a "
                "run. None of them train",
                file=sys.stderr,
            )
        if columns is None or moments is None or n_runs == 0:
            msg = (
                f"[{self.name}] no run of consecutive labeled frames of "
                f"sequence_length {window} frames or more is left to train on: "
                f"{unlabeled_entries} train entries have no label file and "
                f"{too_short} runs are shorter than sequence_length"
            )
            raise RuntimeError(msg)
        if len(classes) < 2:
            msg = (
                f"[{self.name}] the training runs hold {len(classes)} class "
                f"({sorted(classes)}), and a classifier needs two or more"
            )
            raise RuntimeError(msg)
        return _TrainingRuns(runs_dir, columns, moments, sorted(classes), n_runs)

    def _write_training_csvs(self, runs: _TrainingRuns, data_dir: Path) -> None:
        """Write the runs as standardized CSVs in lightning-action layout.

        One file per run. Both files carry the index column, because the package
        reads its first column as the index (``index_col=0``) and would otherwise
        drop the first feature or class. The package standardizes nothing, so
        the standardization that ``apply`` repeats is done here.
        """
        features_dir = data_dir / "features"
        labels_dir = data_dir / "labels"
        features_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        assert self._class_names is not None
        mean, std = runs.moments.standardizer()
        classes = np.asarray(runs.classes)

        for path in sorted(runs.runs_dir.glob("run_*.npz")):
            with np.load(path) as stored:
                matrix = stored["x"]
                labels = stored["y"]
            pd.DataFrame(standardize(matrix, mean, std), columns=runs.columns).to_csv(
                features_dir / f"{path.stem}.csv", index=True, float_format="%.7g"
            )
            one_hot = (labels[:, None] == classes[None, :]).astype(int)
            pd.DataFrame(one_hot, columns=self._class_names).to_csv(
                labels_dir / f"{path.stem}.csv", index=True
            )

    def _build_config(
        self, n_features: int, n_classes: int, data_path: str
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
                "seed": p.random_state,
            },
            "training": {
                "batch_size": p.batch_size,
                "num_epochs": p.num_epochs,
                "device": p.device,
                "sequence_length": p.sequence_length,
                "seed": p.random_state,
                "train_probability": 0.9,
                "val_probability": 0.1,
            },
            "optimizer": {
                "type": p.optimizer,
                "lr": p.learning_rate,
                "wd": p.weight_decay,
            },
        }

    def _loaded_model(self) -> _PackageModel:
        """Return the fitted package model, loading it from its directory once."""
        model_class = _package_model_class()
        if self._la_model is None and self._model_dir is not None:
            model_path = self._model_dir / "model"
            if not model_path.exists():
                # model_dir IS the model directory (loaded from run_root)
                model_path = self._model_dir / "lightning_action_model"
            self._la_model = _predicting_in_process(
                model_class.from_dir(str(model_path))
            )

        if self._la_model is None:
            msg = "[lightning-action] No model available for prediction."
            raise RuntimeError(msg)
        return self._la_model

    def _predict_series(
        self, df: pd.DataFrame, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predict the class probabilities of every row of *df*, one series at a time.

        The package predicts whole windows from a file's first row, and only a file
        of at least one window and its context. Each series is therefore written
        as its own file, with zero rows (the column mean) ahead of its first frame
        to fill a whole number of windows, and at least two. The last frame ends
        the final window, so no row falls past it, and the zero rows sit where a
        training run also starts with no context.

        Returns:
            One row of probabilities per row of *df*, NaN for a row that belongs
            to no series because a key column is missing.
        """
        assert self._classes is not None
        assert self._class_names is not None
        assert self._sequence_length is not None
        model = self._loaded_model()
        window = self._sequence_length
        n_classes = len(self._classes)
        probs = np.full((len(df), n_classes), np.nan)

        series = list(iter_series_positions(df))
        if not series:
            return probs

        feat_cols = self._feature_columns
        assert feat_cols is not None
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
