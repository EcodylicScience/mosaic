"""The VAME feature: a recurrent variational autoencoder of keypoint windows.

VAME runs in a separate Python environment, located through :data:`VAME_ENV`. A
runner program in that environment imports VAME. Nothing in this module imports
it, and mosaic does not install it. One runner serves a feature from its first
request until :meth:`VameFeature.close`, because importing VAME takes seconds and
``apply`` runs once per entry.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Self, final

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from mosaic.core.helpers import make_entry_key
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Prefix,
    Provides,
    Requirement,
    Template,
)
from mosaic.core.pipeline.job import CancelCheck, CancelToken, cancel_check
from mosaic.core.pipeline.progress import ProgressCallback
from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
)
from mosaic.core.pipeline.types import (
    COLUMNS,
    POSE_CONFIG_DESCRIPTION,
    DependencyLookup,
    EmitsLevel,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    PoseConfig,
    TrackInputs,
)
from mosaic.optional_dependency import Prerequisite

from .external.runner_wire import HEARTBEAT, Event
from .external.vame_protocol import (
    VAME_VERSION,
    AnnealFunction,
    AnnealTime,
    BatchSize,
    Device,
    Embed,
    EmbedResult,
    EmbedSession,
    EpochEvent,
    Fit,
    FitResult,
    FittedModel,
    KlStart,
    LoadModel,
    LossReduction,
    Payload,
    PredictionSteps,
    Preprocessing,
    SavgolOrder,
    Session,
    SplitMode,
    StageEvent,
    StepsPerEpoch,
    TestFraction,
    Training,
    TrainSet,
    check_distinct_references,
    check_model_is_saved,
)
from .helpers import entry_identity
from .registry import register_feature
from .runner_client import RunnerInterpreter, RunnerProcess
from .vame_sessions import (
    LATENT_PREFIX,
    EntrySessions,
    SessionPurpose,
    SessionSettings,
    SessionTrack,
    VameKeypoints,
    build_sessions,
    min_session_frames,
    place_latents,
    vame_keypoints,
    write_session_tracks,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

__all__ = [
    "HEARTBEAT_REPORT_SECONDS",
    "MODEL_SETTINGS",
    "VAME_CONDA_ENV_VAR",
    "VAME_ENV",
    "VAME_MODEL_FILE",
    "VAME_PYTHON_ENV",
    "VAME_RUNNER_LOG",
    "VAME_RUNNER_SCRIPT",
    "FitReport",
    "FitSession",
    "VameExitError",
    "VameFeature",
    "VameModelArtifact",
    "VameModelBundle",
    "VameNotFoundError",
]

log = logging.getLogger(__name__)

VAME_CONDA_ENV_VAR: Final = "MOSAIC_VAME_CONDA_ENV"
"""Names a conda environment that has VAME installed."""

VAME_PYTHON_ENV: Final = "MOSAIC_VAME_PYTHON"
"""Names a Python interpreter that has VAME installed."""

VAME_MODEL_FILE: Final = "vame_model.joblib"
"""The model bundle that a fit writes to its run root."""

VAME_RUNNER_LOG: Final = "vame_runner.log"
"""The runners' standard error in the run root.

:meth:`VameFeature.save_state` appends the log of the fit, and
:meth:`VameFeature.close` the rest.
"""

VAME_RUNNER_SCRIPT: Final = Path(__file__).parent / "external" / "vame_runner.py"
"""The runner program, which runs in VAME's environment."""

HEARTBEAT_REPORT_SECONDS: Final = 60.0
"""The least time between two heartbeats that the progress callback receives.

The runner writes a heartbeat every 30 s while a command runs. ``on_phase``
receives one only after this long without another report. The run log of a long
command then records that the command is alive, and omits most beats.
"""

MODEL_SETTINGS: Final = tuple(
    sorted(
        {field.name for field in dataclasses.fields(SessionSettings)}
        | set(Preprocessing.model_fields)
    )
)
"""The params that build and preprocess sessions.

A model bundle records their values, and a run that applies the model must use
them. A run's identifier names its params, and a run that applies a model runs
the model's values.
"""

_NOT_FOUND_HELP: Final = (
    "VAME is not installed with mosaic. It runs in a separate environment. "
    "Build it with the command in "
    "mosaic/behavior/feature_library/external/vame-env/README.md, point "
    f"{VAME_PYTHON_ENV} at an interpreter that has vame-py {VAME_VERSION} "
    f"installed, or name a conda environment with {VAME_CONDA_ENV_VAR}."
)


class VameNotFoundError(ToolNotFoundError):
    """Raised when the location ladder cannot locate VAME's interpreter."""

    default_message = _NOT_FOUND_HELP


class VameExitError(ToolExitError):
    """Raised when the VAME runner exits during a command.

    Also raised when the runner writes a line that breaks the runner protocol.
    """

    tool_name = "The VAME runner"


# A $PATH lookup cannot single out VAME's interpreter. Rung 5 reads the one under
# external/vame-env/.venv.
VAME_ENV: Final = ToolEnv(
    tool="VAME",
    conda_env_var=VAME_CONDA_ENV_VAR,
    bin_var=VAME_PYTHON_ENV,
    bin_mode="direct",
    not_found=VameNotFoundError,
    repo_venv=Path(__file__).parent / "external" / "vame-env",
)


class VameModelArtifact(JoblibArtifact[dict[str, object]]):
    """A fitted VAME model bundle, a dict of builtins (``vame_model.joblib``)."""

    feature: str = "vame"
    pattern: str = VAME_MODEL_FILE
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


class FitSession(BaseModel):
    """One session of a fit: an individual's frames in one entry.

    Attributes:
        name: The session's name in VAME's project.
        group: The entry's group.
        sequence: The entry's sequence.
        individual: The individual's id as text, or ``None`` for a table without
            an id column.
        first_frame: The session's first frame.
        last_frame: The session's last frame.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    group: str
    sequence: str
    individual: str | None
    first_frame: int
    last_frame: int


class FitReport(BaseModel):
    """What a fit trained on, and how long it trained.

    Attributes:
        epochs: The epochs that training ran.
        sessions: VAME's training sessions, in the order of their
            concatenation.
        excluded: Each excluded session's entry, frames and reason.
        gap_rows: Each entry's rows at gap frames outside every session and
            exclusion, by entry key. An entry without such a row is left out.
        train_frames: The frames of the training set.
        test_frames: The frames of the test set.
        straddling_fraction: The fraction of training-window starts whose window
            spans a join between two chunks of the training set. A chunk is a
            session, or under ``mode_2`` one of the two parts of a session around
            its test chunk.
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int
    sessions: list[FitSession]
    excluded: list[str]
    gap_rows: dict[str, int]
    train_frames: int
    test_frames: int
    straddling_fraction: float


class VameModelBundle(FittedModel):
    """The model bundle that a fit writes, ``vame_model.joblib``.

    The file is a dict of this model's fields, of builtin types only, and loads in
    mosaic's environment and in VAME's. The runner reads the fields of
    :class:`~mosaic.behavior.feature_library.external.vame_protocol.FittedModel`.
    The bundle does not record a path or a time. Two fits on the CPU therefore
    write equal bundles.

    Attributes:
        vame_version: The VAME release that trained the model.
        device: The device of the trained parameters.
        settings: The value of each param of :data:`MODEL_SETTINGS`.
        fit_report: What the fit trained on.
    """

    vame_version: str
    device: str
    settings: dict[str, JsonValue]
    fit_report: FitReport


_MODEL_DESCRIPTION = (
    "A fitted VAME model to apply, such as a fit step scoped to the training "
    "groups. Unset, the feature fits a new model on the entries in scope. With a "
    "model set, nothing is fitted, the entries in scope leave the run id, and the "
    "settings that build and preprocess the sessions must equal the model's."
)

_VAME_PYTHON_DESCRIPTION = (
    "Path to a Python interpreter with VAME installed. Unset, resolution reads "
    "MOSAIC_VAME_CONDA_ENV, then MOSAIC_VAME_PYTHON, then the in-repository "
    "environment under feature_library/external/vame-env/.venv."
)

_DEVICE_DESCRIPTION = (
    "Where VAME trains and embeds. The value auto selects CUDA, then MPS, then "
    "the CPU, as VAME does. The run identifier leaves the device out. One recipe "
    "therefore runs under one name on any machine. Training on different devices "
    "gives different models, and only training on the CPU is bit-reproducible. "
    "The model bundle and the run log record the device that ran."
)

_CENTERED_REFERENCE_DESCRIPTION = (
    "The keypoint that egocentric alignment moves to the origin of each frame, "
    "as a keypoint index or a keypoint name. It must be one of the selected "
    "keypoints."
)

_ORIENTATION_REFERENCE_DESCRIPTION = (
    "The keypoint whose direction from the centered reference keypoint "
    "egocentric alignment rotates onto a fixed axis, as a keypoint index or a "
    "keypoint name. It must be one of the selected keypoints, and a keypoint "
    "other than the centered one."
)

_RUN_LOWCONF_CLEANING_DESCRIPTION = (
    "Replace each point whose confidence is below pose_confidence by linear "
    "interpolation along its series. Off, every frame with a missing point ends "
    "its session, and max_gap_frames must be 0."
)

_POSE_CONFIDENCE_DESCRIPTION = (
    "The confidence below which low-confidence cleaning replaces a point. "
    "Confidence is read from the confidence columns when every selected "
    "keypoint has one, and is otherwise 1 for every finite point."
)

_RUN_OUTLIER_CLEANING_DESCRIPTION = (
    "Replace each point whose absolute z-score exceeds iqr_factor times the "
    "interquartile range of its series' z-scores, by linear interpolation."
)

_IQR_FACTOR_DESCRIPTION = (
    "The multiple of the interquartile range beyond which outlier cleaning "
    "replaces a point."
)

_RUN_SAVGOL_FILTERING_DESCRIPTION = (
    "Smooth each coordinate series with a Savitzky-Golay filter."
)

_SAVGOL_LENGTH_DESCRIPTION = (
    "The Savitzky-Golay filter's window. It is odd, longer than savgol_order, "
    "and at most time_window."
)

_SAVGOL_ORDER_DESCRIPTION = "The order of the Savitzky-Golay filter's polynomial."

_RUN_RESCALING_DESCRIPTION = (
    "Divide each session's positions by the session's median distance between the "
    "two reference keypoints, after the other preprocessing. A session is a run of "
    "an individual's frames that no gap longer than max_gap_frames splits."
)

_MAX_GAP_FRAMES_DESCRIPTION = (
    "The longest gap inside a session that low-confidence cleaning fills by "
    "interpolation. It must be 0 without run_lowconf_cleaning. A gap is a run in "
    "which either reference keypoint lacks a finite point at or above "
    "pose_confidence. A longer gap splits the individual into separate sessions, "
    "and a gap at its start or end is trimmed. The rows of these gaps do not "
    "receive a latent, and their count is reported. "
    "The missing points of the other keypoints are interpolated however long they "
    "run. A session is excluded when one of them lacks such a point at every frame."
)

_TEST_FRACTION_DESCRIPTION = (
    "The fraction of the training data that VAME sets aside to compute the test "
    "loss. Training saves the model of the epoch with the lowest test "
    "reconstruction loss, among the epochs with a KL weight above 0.99."
)

_SPLIT_MODE_DESCRIPTION = (
    "How VAME chooses the test data. mode_1 takes the first test_fraction of "
    "the concatenated sessions, and mode_2 a random contiguous chunk of that "
    "fraction from each session."
)

_RANDOM_STATE_DESCRIPTION = (
    "Seeds every random number generator that training draws from."
)

_ZDIMS_DESCRIPTION = (
    "The dimensions of the latent space. Each dimension is one output column, "
    "vame_z0 to vame_z{zdims-1}."
)

_TIME_WINDOW_DESCRIPTION = (
    "The window that the encoder reads to produce one latent. Each latent is "
    "written on the row at the center of its window."
)

_MAX_EPOCHS_DESCRIPTION = (
    "VAME trains from epoch 1 to epoch max_epochs - 1, unless model_convergence "
    "stops it first. VAME saves a model only from the first epoch whose KL weight "
    "exceeds 0.99. A max_epochs that ends training earlier is refused."
)

_MODEL_CONVERGENCE_DESCRIPTION = (
    "How long VAME may train in a row without saving a model. It stops training "
    "at the next epoch without a save. Every epoch before the first one whose KL "
    "weight exceeds 0.99 is one without a save. A model_convergence that stops "
    "training earlier is refused."
)

_BATCH_SIZE_DESCRIPTION = (
    "The windows of one training batch. A test batch contains a quarter of them."
)

_BETA_DESCRIPTION = "The factor on the KL loss, which the annealed KL weight scales."

_LEARNING_RATE_DESCRIPTION = "The learning rate of the Adam optimizer."

_PREDICTION_DECODER_DESCRIPTION = (
    "Train a second decoder that predicts the frames after each window. The "
    "first decoder reconstructs the window."
)

_PREDICTION_STEPS_DESCRIPTION = (
    "How far past each window the prediction decoder predicts. It is at most "
    "time_window."
)

_NOISE_DESCRIPTION = (
    "Add Gaussian noise to the training input, scaled by each input series' "
    "standard deviation in the window."
)

_SCHEDULER_DESCRIPTION = (
    "Multiply the learning rate by scheduler_gamma when the loss of each epoch's "
    "last training batch has not fallen a relative 1e-3 below its lowest value "
    "for more than scheduler_step_size epochs. Off, the learning rate stays "
    "constant."
)

_SCHEDULER_STEP_SIZE_DESCRIPTION = (
    "How long the loss of each epoch's last training batch may go without "
    "improving. The scheduler lowers the learning rate at the next epoch without "
    "an improvement."
)

_SCHEDULER_GAMMA_DESCRIPTION = (
    "The factor by which the scheduler multiplies the learning rate."
)

_SOFTPLUS_DESCRIPTION = "Pass the encoder's log-variance through a softplus activation."

_STEPS_PER_EPOCH_DESCRIPTION = (
    "The batches of one training epoch, at least 1. Unset, an epoch draws one "
    "window per training frame."
)

_HIDDEN_SIZE_LAYER_1_DESCRIPTION = (
    "The hidden units of each of the encoder's two recurrent layers."
)

_DROPOUT_ENCODER_DESCRIPTION = (
    "The dropout probability between the encoder's two recurrent layers."
)

_HIDDEN_SIZE_REC_DESCRIPTION = (
    "The hidden units of the reconstruction decoder's recurrent layer."
)

_HIDDEN_SIZE_PRED_DESCRIPTION = (
    "The hidden units of the prediction decoder's recurrent layer."
)

_MSE_RECONSTRUCTION_REDUCTION_DESCRIPTION = (
    "How the reconstruction loss reduces over a batch: sum or mean."
)

_MSE_PREDICTION_REDUCTION_DESCRIPTION = (
    "How the prediction loss reduces over a batch: sum or mean."
)

_KMEANS_LOSS_DESCRIPTION = (
    "The cluster loss sums the square roots of this many of the largest singular "
    "values of the latent Gram matrix."
)

_KMEANS_LAMBDA_DESCRIPTION = "The weight of the cluster loss."

_ANNEAL_FUNCTION_DESCRIPTION = (
    "The schedule on which the KL weight rises after kl_start: linear or sigmoid."
)

_KL_START_DESCRIPTION = "The last epoch at which the KL weight is 0."

_ANNEALTIME_DESCRIPTION = (
    "How the KL weight rises after kl_start. The linear schedule raises the "
    "weight to 1 at epoch kl_start + annealtime, and the sigmoid schedule is "
    "centered on epoch annealtime."
)

_EMBED_BATCH_SIZE_DESCRIPTION = (
    "How many windows VAME encodes at once when it applies a model. It changes "
    "latents only at float32 rounding, about 1e-6."
)

_EMBED_CHUNK_FRAMES_DESCRIPTION = (
    "The longest stretch of a session that one embedding call reads. A longer "
    "session is embedded in chunks that overlap by time_window - 1. Chunking "
    "changes latents only at float32 rounding. Lower it when embedding runs out "
    "of memory. It is at least time_window."
)

_SCRATCH_DIR_DESCRIPTION = (
    "The directory in which the feature creates its exchange directory for the "
    "runner's files. Unset, the system's temporary directory."
)


@final
@register_feature
class VameFeature:
    """Fits VAME to keypoint tracks, and embeds each frame in its latent space.

    VAME (Luxem et al. 2022, Communications Biology,
    doi:10.1038/s42003-022-04080-7) trains a recurrent variational autoencoder
    on windows of egocentrically aligned keypoints. A fit trains one model on
    every individual of every entry in its scope. Apply encodes each window of
    ``time_window`` frames and writes its latent on the row at the window's
    center, as ``vame_z0`` to ``vame_z{zdims-1}``. A row without a latent is left
    out: a row near either end of a session, a row of an excluded session, a row
    in a gap that splits a session, and a row in a gap at either end of an
    individual's frames.

    Each individual's rows are one session on contiguous frames. A gap longer
    than ``max_gap_frames`` splits the session, and a gap at either end of the
    individual's frames is trimmed. The rows of both are reported. A session is
    excluded and reported when one of its keypoints lacks a usable point, or when
    it is too short: under ``time_window`` frames to embed, and under
    ``2 * time_window + 1`` frames to fit. VAME's preprocessing runs on each
    session, with egocentric alignment always on.

    VAME is licensed GPL-3.0. It runs in a separate environment that you build
    (``external/vame-env/README.md`` in this package), and mosaic drives it as a
    program over that program's standard input and output. ``vame_python``,
    ``MOSAIC_VAME_CONDA_ENV`` or ``MOSAIC_VAME_PYTHON`` names another
    environment.

    One runner process serves the feature from its first request until
    :meth:`close`. :meth:`save_state` writes the model bundle and the runner's log
    to the run root, and the runner then loads the bundle and embeds each entry of
    the run. A runner that exits is started again at the next request, and loads
    the bundle. A fired ``cancel_token`` kills the runner, and the run ends
    cancelled.

    ``device`` is left out of the run identifier. Only training on the CPU is
    bit-reproducible. The model bundle and the run log record the device that ran.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.vame_feature.VameFeature.Params`.
    """

    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    # Learns from its inputs. ``__init__`` clears this and ``scope_dependent`` for a
    # run given a model, which applies it.
    fits = True
    # VAME trains a recurrent network, and is built for a GPU.
    resource_class = "gpu"
    name = "vame"
    version = "0.1"
    parallelizable = False
    # The entries in scope are the training set, so a fit hashes them. A run given
    # a ``model`` embeds each entry from its own rows and hashes none.
    scope_dependent = True
    # Apply builds its sessions from the rows of the entry alone.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    prerequisites: tuple[Prerequisite, ...] = (RunnerInterpreter(VAME_ENV),)
    emits: EmitsLevel = "individual"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(COLUMNS.frame_col),
        Template("{pose.x_prefix}", family=True),
        Template("{pose.y_prefix}", family=True),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix(LATENT_PREFIX),), conditional=META_COLUMNS_KEPT
    )
    VameModelArtifact = VameModelArtifact

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        model: Annotated[VameModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = None
        # Two machines run one model from interpreters at two paths.
        vame_python: Annotated[
            str | None, HASH_EXCLUDE, Declared(_VAME_PYTHON_DESCRIPTION)
        ] = None
        device: Annotated[Device, HASH_EXCLUDE, Declared(_DEVICE_DESCRIPTION)] = "auto"
        pose: Annotated[PoseConfig, Declared(POSE_CONFIG_DESCRIPTION)] = Field(
            default_factory=PoseConfig
        )
        centered_reference_keypoint: Annotated[
            int | str, Declared(_CENTERED_REFERENCE_DESCRIPTION)
        ]
        orientation_reference_keypoint: Annotated[
            int | str, Declared(_ORIENTATION_REFERENCE_DESCRIPTION)
        ]

        # Preprocessing
        run_lowconf_cleaning: Annotated[
            bool, Declared(_RUN_LOWCONF_CLEANING_DESCRIPTION)
        ] = True
        pose_confidence: Annotated[float, Declared(_POSE_CONFIDENCE_DESCRIPTION)] = 0.99
        run_outlier_cleaning: Annotated[
            bool, Declared(_RUN_OUTLIER_CLEANING_DESCRIPTION)
        ] = True
        iqr_factor: Annotated[int, Declared(_IQR_FACTOR_DESCRIPTION)] = 4
        run_savgol_filtering: Annotated[
            bool, Declared(_RUN_SAVGOL_FILTERING_DESCRIPTION)
        ] = True
        savgol_length: Annotated[
            int, Declared(_SAVGOL_LENGTH_DESCRIPTION, unit="frames")
        ] = 5
        savgol_order: Annotated[SavgolOrder, Declared(_SAVGOL_ORDER_DESCRIPTION)] = 2
        run_rescaling: Annotated[bool, Declared(_RUN_RESCALING_DESCRIPTION)] = False

        # Sessions
        max_gap_frames: Annotated[
            int, Declared(_MAX_GAP_FRAMES_DESCRIPTION, unit="frames")
        ] = Field(default=30, ge=0)

        # Train set
        test_fraction: Annotated[TestFraction, Declared(_TEST_FRACTION_DESCRIPTION)] = (
            0.1
        )
        split_mode: Annotated[SplitMode, Declared(_SPLIT_MODE_DESCRIPTION)] = "mode_2"

        # Training
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        zdims: Annotated[int, Declared(_ZDIMS_DESCRIPTION)] = Field(default=30, ge=1)
        time_window: Annotated[
            int, Declared(_TIME_WINDOW_DESCRIPTION, unit="frames")
        ] = Field(default=30, ge=1)
        max_epochs: Annotated[int, Declared(_MAX_EPOCHS_DESCRIPTION, unit="epochs")] = (
            500
        )
        model_convergence: Annotated[
            int, Declared(_MODEL_CONVERGENCE_DESCRIPTION, unit="epochs")
        ] = 50
        batch_size: Annotated[BatchSize, Declared(_BATCH_SIZE_DESCRIPTION)] = 256
        beta: Annotated[float, Declared(_BETA_DESCRIPTION)] = 1.0
        learning_rate: Annotated[float, Declared(_LEARNING_RATE_DESCRIPTION)] = 5e-4
        prediction_decoder: Annotated[
            bool, Declared(_PREDICTION_DECODER_DESCRIPTION)
        ] = True
        prediction_steps: Annotated[
            PredictionSteps, Declared(_PREDICTION_STEPS_DESCRIPTION, unit="frames")
        ] = 15
        noise: Annotated[bool, Declared(_NOISE_DESCRIPTION)] = False
        scheduler: Annotated[bool, Declared(_SCHEDULER_DESCRIPTION)] = True
        scheduler_step_size: Annotated[
            int, Declared(_SCHEDULER_STEP_SIZE_DESCRIPTION, unit="epochs")
        ] = 100
        scheduler_gamma: Annotated[float, Declared(_SCHEDULER_GAMMA_DESCRIPTION)] = 0.2
        softplus: Annotated[bool, Declared(_SOFTPLUS_DESCRIPTION)] = False
        steps_per_epoch: Annotated[
            StepsPerEpoch | None, Declared(_STEPS_PER_EPOCH_DESCRIPTION)
        ] = None
        hidden_size_layer_1: Annotated[
            int, Declared(_HIDDEN_SIZE_LAYER_1_DESCRIPTION)
        ] = 256
        dropout_encoder: Annotated[float, Declared(_DROPOUT_ENCODER_DESCRIPTION)] = 0.0
        hidden_size_rec: Annotated[int, Declared(_HIDDEN_SIZE_REC_DESCRIPTION)] = 256
        hidden_size_pred: Annotated[int, Declared(_HIDDEN_SIZE_PRED_DESCRIPTION)] = 256
        mse_reconstruction_reduction: Annotated[
            LossReduction, Declared(_MSE_RECONSTRUCTION_REDUCTION_DESCRIPTION)
        ] = "sum"
        mse_prediction_reduction: Annotated[
            LossReduction, Declared(_MSE_PREDICTION_REDUCTION_DESCRIPTION)
        ] = "sum"
        kmeans_loss: Annotated[int, Declared(_KMEANS_LOSS_DESCRIPTION)] = 30
        kmeans_lambda: Annotated[float, Declared(_KMEANS_LAMBDA_DESCRIPTION)] = 0.1
        anneal_function: Annotated[
            AnnealFunction, Declared(_ANNEAL_FUNCTION_DESCRIPTION)
        ] = "linear"
        kl_start: Annotated[KlStart, Declared(_KL_START_DESCRIPTION, unit="epochs")] = 2
        annealtime: Annotated[
            AnnealTime, Declared(_ANNEALTIME_DESCRIPTION, unit="epochs")
        ] = 4

        # Throughput and scratch space
        embed_batch_size: Annotated[
            int, HASH_EXCLUDE, Declared(_EMBED_BATCH_SIZE_DESCRIPTION)
        ] = Field(default=64, ge=1)
        embed_chunk_frames: Annotated[
            int,
            HASH_EXCLUDE,
            Declared(_EMBED_CHUNK_FRAMES_DESCRIPTION, unit="frames"),
        ] = Field(default=50_000, ge=1)
        scratch_dir: Annotated[
            str | None, HASH_EXCLUDE, Declared(_SCRATCH_DIR_DESCRIPTION)
        ] = None

        @model_validator(mode="after")
        def _check_references(self) -> Self:
            check_distinct_references(
                self.centered_reference_keypoint, self.orientation_reference_keypoint
            )
            return self

        @model_validator(mode="after")
        def _check_model_is_saved(self) -> Self:
            check_model_is_saved(
                max_epochs=self.max_epochs,
                model_convergence=self.model_convergence,
                kl_start=self.kl_start,
                annealtime=self.annealtime,
                anneal_function=self.anneal_function,
            )
            return self

        @model_validator(mode="after")
        def _check_windows(self) -> Self:
            if self.prediction_steps > self.time_window:
                msg = (
                    f"prediction_steps={self.prediction_steps} exceeds "
                    f"time_window={self.time_window}, the most frames that VAME "
                    "predicts past a window"
                )
                raise ValueError(msg)
            if self.embed_chunk_frames < self.time_window:
                msg = (
                    f"embed_chunk_frames={self.embed_chunk_frames} is shorter than "
                    f"time_window={self.time_window}, and each chunk must contain "
                    "a whole window"
                )
                raise ValueError(msg)
            return self

        @model_validator(mode="after")
        def _check_savgol(self) -> Self:
            length = self.savgol_length
            if length % 2 == 0:
                msg = f"savgol_length={length} is even, and the window must be odd"
                raise ValueError(msg)
            if length <= self.savgol_order:
                msg = (
                    f"savgol_length={length} must be longer than "
                    f"savgol_order={self.savgol_order}"
                )
                raise ValueError(msg)
            if length > self.time_window:
                msg = f"savgol_length={length} exceeds time_window={self.time_window}"
                raise ValueError(msg)
            return self

        @model_validator(mode="after")
        def _check_gaps(self) -> Self:
            if self.max_gap_frames > 0 and not self.run_lowconf_cleaning:
                msg = (
                    f"max_gap_frames={self.max_gap_frames} needs "
                    "run_lowconf_cleaning, which alone interpolates a gap. Set "
                    "max_gap_frames to 0 to end a session at every missing point."
                )
                raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: VameFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        # A run given a model applies it and learns nothing: a held-out group may be
        # in its scope, and the scope is not part of what the run is.
        self.scope_dependent = self.params.model is None
        self.fits = self.params.model is None
        self.progress_callback: ProgressCallback | None = None
        self.cancel_token: CancelToken | None = None
        self._runner = RunnerProcess(
            VAME_ENV.placed(bin_path=self.params.vame_python),
            VAME_RUNNER_SCRIPT,
            failure=VameExitError,
            version=VAME_VERSION,
            on_start=self._prepare,
            scratch_dir=self.params.scratch_dir,
        )
        # The bundle that a started runner loads.
        self._model: Path | None = None
        # The fit that save_state writes.
        self._fitted: VameModelBundle | None = None
        # The step of the command in progress, which a heartbeat reports.
        self._stage = ""
        self._request_started = 0.0
        self._last_report = 0.0
        self._embed_device: str | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        """Find the model of *run_root*, else the model that ``model`` names.

        The runner of an earlier run is stopped. No runner starts here. The next
        request starts one, which loads the model found.

        Raises:
            FileNotFoundError: ``model`` is set, and the run that it names lacks
                a bundle. A run with ``model`` set does not fit.
            ValueError: The bundle was trained by another VAME release, or a
                param of :data:`MODEL_SETTINGS` differs from the bundle's.
            VameNotFoundError: ``model`` is set, the run root lacks a bundle, and
                VAME's interpreter cannot be located. The run is refused even
                when every entry is cached, because ``run_feature`` calls
                load_state before it checks any entry against the cache.
        """
        self.close()
        self._runner.keep_log(run_root / VAME_RUNNER_LOG)
        self._model = None
        self._fitted = None
        self._embed_device = None
        cached = run_root / VAME_MODEL_FILE
        model = self.params.model
        if cached.exists():
            path = cached
            self._check_model(path)
        elif model is not None:
            resolved = artifact_paths.get("model")
            if resolved is None:
                msg = (
                    f"vame: the model of the {model.feature} run {model.run_id} "
                    f"has no {model.pattern}. A run with model set applies that "
                    "model and does not fit one."
                )
                raise FileNotFoundError(msg)
            path = resolved
            self._check_model(path)
            # The run is refused here, once, before any entry is read. Locating
            # the interpreter does not start a process.
            _ = self._runner.locate()
        else:
            return False
        self._model = path
        return True

    def fit(self, inputs: InputStream) -> None:
        """Train a model on the sessions of every entry, in the runner.

        Raises:
            ValueError: Every session of the scope is excluded from the fit, or as
                :func:`~mosaic.behavior.feature_library.vame_sessions.build_sessions`
                raises, after the key of the entry that it refused.
        """
        self._model = None
        self._fitted = None
        settings = self._session_settings()
        keypoints = vame_keypoints(settings)
        entries: list[tuple[str, str, EntrySessions]] = []
        for _, df in inputs():
            group, sequence = entry_identity(df)
            try:
                built = build_sessions(df, settings, "fit")
            except ValueError as error:
                msg = f"{make_entry_key(group, sequence)}: {error}"
                raise ValueError(msg) from error
            entries.append((group, sequence, built))
        # The order of the sessions sets VAME's split of training and test frames.
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        unused: list[str] = []
        excluded: list[str] = []
        gap_rows: dict[str, int] = {}
        sessions: list[tuple[str, str, SessionTrack]] = []
        for group, sequence, entry in entries:
            key = make_entry_key(group, sequence)
            unused.extend(self._report_unused("fit", key, entry))
            excluded.extend(
                f"{key}: {exclusion.describe()}" for exclusion in entry.excluded
            )
            if entry.gap_rows.count:
                gap_rows[key] = entry.gap_rows.count
            sessions.extend((group, sequence, session) for session in entry.sessions)
        if not sessions:
            least = min_session_frames(settings.time_window, "fit")
            msg = (
                "vame: a fit needs a session of "
                f"2 * time_window + 1 = {least} frames in which every selected "
                "keypoint has a usable point, and the scope lacks one. Excluded: "
                f"{'; '.join(unused) or 'nothing'}"
            )
            raise ValueError(msg)
        payload, result, state_dict = self._fit_sessions(
            [session for _, _, session in sessions], keypoints
        )
        self._phase("device", f"fit on {result.device}")
        names = [session.name for session in payload.sessions]
        self._fitted = VameModelBundle(
            project_name=result.project_name,
            state_dict=state_dict,
            metadata=result.metadata,
            num_features=result.num_features,
            keypoints=payload.keypoints,
            preprocessing=payload.preprocessing,
            training=payload.training,
            seq_mean=result.seq_mean,
            seq_std=result.seq_std,
            vame_version=VAME_VERSION,
            device=result.device,
            settings=self._settings(),
            fit_report=FitReport(
                epochs=result.epochs,
                sessions=[
                    FitSession(
                        name=name,
                        group=group,
                        sequence=sequence,
                        individual=(
                            None
                            if session.individual is None
                            else str(session.individual)
                        ),
                        first_frame=session.first_frame,
                        last_frame=session.last_frame,
                    )
                    for name, (group, sequence, session) in zip(
                        names, sessions, strict=True
                    )
                ],
                excluded=excluded,
                gap_rows=gap_rows,
                train_frames=result.train_frames,
                test_frames=result.test_frames,
                straddling_fraction=result.straddling_fraction,
            ),
        )

    def save_state(self, run_root: Path) -> None:
        """Write the bundle and the runner's log, and load the bundle in the runner.

        The runner keeps running, and embeds with the bundle that it loaded.

        Raises:
            RuntimeError: No fit ran.
        """
        fitted = self._fitted
        if fitted is None:
            msg = "vame: save_state needs a fit, and none ran"
            raise RuntimeError(msg)
        path = run_root / VAME_MODEL_FILE
        bundle = fitted.model_dump()
        atomic_write(path, lambda target: joblib.dump(bundle, target))
        self._runner.keep_log(run_root / VAME_RUNNER_LOG)
        # self._model is set after this request. A runner that the request
        # starts then skips the load in _prepare, and loads the bundle once.
        _ = self._request(LoadModel(path=str(path), device=self.params.device))
        self._model = path

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Embed one entry's sessions, and place each latent on its window's center.

        Raises:
            RuntimeError: Neither fit nor load_state provided a model.
            ValueError: No row of the entry receives a latent, naming the
                excluded frames, or as
                :func:`~mosaic.behavior.feature_library.vame_sessions.build_sessions`
                raises.
        """
        if self._model is None:
            msg = "vame: apply needs a model, which fit or load_state provides"
            raise RuntimeError(msg)
        settings = self._session_settings()
        entry = build_sessions(df, settings, "embed")
        _ = self._report_unused("embed", make_entry_key(*entry_identity(df)), entry)
        latents = self._embed(entry.sessions) if entry.sessions else []
        return place_latents(df, entry, latents, time_window=settings.time_window)

    def close(self) -> None:
        """Stop the runner, append its log to the run root, and remove its files.

        The log in the run root receives the runner's standard error that
        :meth:`save_state` did not append, and the exchange directory is removed.
        A second call does nothing. A later request starts a new runner.
        """
        _ = self._runner.close()

    # --- Runner ---

    def _request(self, payload: Payload) -> JsonValue:
        """Run *payload*'s command in the runner, reporting its events."""
        self._stage = payload.command
        self._request_started = self._last_report = time.monotonic()
        return self._runner.request(
            payload.command,
            payload,
            on_event=self._on_event,
            cancel=cancel_check(self.cancel_token),
        )

    def _prepare(self, runner: RunnerProcess, cancel: CancelCheck | None) -> None:
        """Load the model into a runner that has started, when there is one."""
        if self._model is None:
            return
        load = LoadModel(path=str(self._model), device=self.params.device)
        _ = runner.request(load.command, load, on_event=self._on_event, cancel=cancel)

    def _fit_sessions(
        self, sessions: list[SessionTrack], keypoints: VameKeypoints
    ) -> tuple[Fit, FitResult, bytes]:
        """Train on *sessions* in the runner.

        Returns:
            The request, its result and the bytes of the trained state dict. The
            files of the request are removed.
        """
        params = self.params
        exchange = self._runner.exchange_dir()
        tracks = Path(tempfile.mkdtemp(prefix="fit-", dir=exchange))
        state_dict = tracks / "state_dict.pkl"
        try:
            written = write_session_tracks(sessions, tracks)
            payload = Fit(
                sessions=[
                    Session(name=name, track=str(path))
                    for name, path in written.items()
                ],
                keypoints=list(keypoints.names),
                preprocessing=self._preprocessing(keypoints),
                trainset=TrainSet(
                    test_fraction=params.test_fraction, split_mode=params.split_mode
                ),
                training=self._training(),
                device=params.device,
                exchange_dir=str(exchange),
                state_dict=str(state_dict),
            )
            result = FitResult.model_validate(self._request(payload))
            return payload, result, state_dict.read_bytes()
        finally:
            shutil.rmtree(tracks, ignore_errors=True)

    def _embed(
        self, sessions: tuple[SessionTrack, ...]
    ) -> list[npt.NDArray[np.float32]]:
        """Embed *sessions* in the runner, and return each one's latents."""
        exchange = self._runner.exchange_dir()
        work = Path(tempfile.mkdtemp(prefix="embed-", dir=exchange))
        try:
            written = write_session_tracks(sessions, work)
            payload = Embed(
                sessions=[
                    EmbedSession(
                        name=name,
                        track=str(path),
                        latents=str(work / f"{name}-latents.npz"),
                    )
                    for name, path in written.items()
                ],
                batch_size=self.params.embed_batch_size,
                chunk_frames=self.params.embed_chunk_frames,
                exchange_dir=str(exchange),
            )
            result = EmbedResult.model_validate(self._request(payload))
            if result.device != self._embed_device:
                self._embed_device = result.device
                self._phase("device", f"embed on {result.device}")
            return [
                _read_latents(Path(session.latents)) for session in payload.sessions
            ]
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # --- Progress ---

    def _on_event(self, event: Event) -> None:
        """Pass an event of the runner on to the progress callback."""
        if event.name == EpochEvent.name:
            epoch = EpochEvent.model_validate(event.data)
            # A loss that is not finite arrives as None, and is left out.
            metrics = {
                name: value for name, value in epoch.losses.items() if value is not None
            }
            if self.progress_callback is not None:
                # VAME counts epochs from 1, and runs epochs 1 to total.
                self.progress_callback.on_epoch_end(
                    epoch.epoch - 1, epoch.total, metrics
                )
            self._last_report = time.monotonic()
        elif event.name == StageEvent.name:
            stage = StageEvent.model_validate(event.data)
            self._stage = stage.stage
            self._phase(stage.stage, stage.message)
        elif event.name == HEARTBEAT:
            now = time.monotonic()
            if now - self._last_report >= HEARTBEAT_REPORT_SECONDS:
                elapsed = now - self._request_started
                self._phase(self._stage, f"running for {elapsed:.0f} s")
        else:
            log.debug("vame: the runner reported the event %s", event.name)

    def _report_unused(
        self, purpose: SessionPurpose, key: str, entry: EntrySessions
    ) -> list[str]:
        """Report each exclusion of the entry *key*, and its gap rows, as ``exclude``.

        Returns:
            The reported lines, each ``"<key>: <description>"``.
        """
        lines = [f"{key}: {line}" for line in entry.describe_unused()]
        for line in lines:
            self._phase("exclude", f"{purpose}, {line}")
        return lines

    def _phase(self, phase: str, message: str) -> None:
        """Report *phase* and *message* to the progress callback, when there is one."""
        if self.progress_callback is not None:
            self.progress_callback.on_phase(phase, message)
        self._last_report = time.monotonic()

    # --- Settings ---

    def _settings(self) -> dict[str, JsonValue]:
        """Return the value of each param of :data:`MODEL_SETTINGS`."""
        return self.params.model_dump(mode="json", include=set(MODEL_SETTINGS))

    def _check_model(self, path: Path) -> None:
        """Refuse a bundle of another VAME release, or of other session settings."""
        loaded: object = joblib.load(path)
        bundle = VameModelBundle.model_validate(loaded)
        if bundle.vame_version != VAME_VERSION:
            msg = (
                f"vame: the model {path} was trained by VAME {bundle.vame_version}, "
                f"and this feature runs VAME {VAME_VERSION}. Fit the model again."
            )
            raise ValueError(msg)
        ours = self._settings()
        differing = [
            f"{name}={ours.get(name)!r} (the model's {bundle.settings.get(name)!r})"
            for name in MODEL_SETTINGS
            if ours.get(name) != bundle.settings.get(name)
        ]
        if differing:
            msg = (
                f"vame: the model {path} built and preprocessed its sessions with "
                "other settings, and a run that applies it must use them. These "
                f"params differ: {'; '.join(differing)}"
            )
            raise ValueError(msg)

    def _session_settings(self) -> SessionSettings:
        params = self.params
        return SessionSettings(
            pose=params.pose,
            centered_reference_keypoint=params.centered_reference_keypoint,
            orientation_reference_keypoint=params.orientation_reference_keypoint,
            run_lowconf_cleaning=params.run_lowconf_cleaning,
            pose_confidence=params.pose_confidence,
            max_gap_frames=params.max_gap_frames,
            time_window=params.time_window,
        )

    def _preprocessing(self, keypoints: VameKeypoints) -> Preprocessing:
        params = self.params
        return Preprocessing(
            centered_reference_keypoint=keypoints.centered,
            orientation_reference_keypoint=keypoints.orientation,
            run_lowconf_cleaning=params.run_lowconf_cleaning,
            pose_confidence=params.pose_confidence,
            run_outlier_cleaning=params.run_outlier_cleaning,
            iqr_factor=params.iqr_factor,
            run_savgol_filtering=params.run_savgol_filtering,
            savgol_length=params.savgol_length,
            savgol_order=params.savgol_order,
            run_rescaling=params.run_rescaling,
        )

    def _training(self) -> Training:
        """Return the training settings, each a param of the same name.

        ``project_random_state`` is the param ``random_state``.
        """
        names = set(Training.model_fields) - {"project_random_state"}
        values = self.params.model_dump(include=names)
        return Training.model_validate(
            {**values, "project_random_state": self.params.random_state}
        )


def _read_latents(path: Path) -> npt.NDArray[np.float32]:
    """Read the latents that the runner wrote for one session."""
    with np.load(path) as stored:
        return np.asarray(stored["latents"], dtype=np.float32)
