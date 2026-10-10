"""The payloads, results and events of the VAME runner's commands.

mosaic's ``vame`` feature sends each payload in a request of ``runner_wire``, and
``vame_runner.py`` validates it with the same model. The runner answers with a
result model, which mosaic validates in turn, and reports progress in events.
Arrays cross as ``.npz`` files that a payload names by path, and a fitted model
as the state-dict file that a fit result names.

This module imports only the standard library and pydantic, which both
environments install, and keeps the syntax of Python 3.11. The runner imports it
by its bare name, and mosaic as
``mosaic.behavior.feature_library.external.vame_protocol``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Final, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "VAME_VERSION",
    "AnnealFunction",
    "AnnealTime",
    "BatchSize",
    "Command",
    "Device",
    "Embed",
    "EmbedResult",
    "EmbedSession",
    "EmbeddedSession",
    "EpochEvent",
    "EventData",
    "EventName",
    "Fit",
    "FitResult",
    "FittedModel",
    "KlStart",
    "LoadModel",
    "LossReduction",
    "Payload",
    "PredictionSteps",
    "Preprocessing",
    "SavgolOrder",
    "Session",
    "SessionName",
    "SplitMode",
    "Stage",
    "StageEvent",
    "StepsPerEpoch",
    "TestFraction",
    "TrainSet",
    "Training",
    "TrainingMetadata",
    "check_distinct_references",
    "check_model_is_saved",
    "first_saving_epoch",
    "least_training_frames",
    "repeated_names",
    "training_window_frames",
]

VAME_VERSION: Final = "0.14.4"
"""The VAME release that the feature drives, as ``vame-env/pyproject.toml`` pins it."""

Command: TypeAlias = Literal["fit", "load_model", "embed"]
"""The commands that the runner serves, besides ``shutdown``."""

Device: TypeAlias = Literal["auto", "cpu", "cuda", "mps"]
"""Where VAME computes. ``auto`` selects as VAME does: CUDA, then MPS, then the CPU."""

AnnealFunction: TypeAlias = Literal["linear", "sigmoid"]
"""The schedule on which VAME raises the weight of the KL loss."""

LossReduction: TypeAlias = Literal["sum", "mean"]
"""How a reconstruction loss reduces over the elements of a batch."""

SplitMode: TypeAlias = Literal["mode_1", "mode_2"]
"""How VAME splits the training frames from the test frames.

``mode_1`` takes the first ``test_fraction`` of the concatenated sessions as the
test set. ``mode_2`` takes a random contiguous chunk of that fraction from each
session.
"""

Stage: TypeAlias = Literal["prepare", "preprocess", "trainset", "train", "embed"]
"""A step of a command that the runner reports when it starts."""

EventName: TypeAlias = Literal["epoch", "stage"]
"""The events that the runner writes besides ``runner_wire``'s heartbeat."""

_SAVING_WEIGHT: Final = 0.99
"""The KL weight above which VAME's ``train_model`` saves a model at an epoch."""

SessionName: TypeAlias = Annotated[str, Field(pattern=r"^[A-Za-z0-9_]+$")]
"""A session's name in VAME's project, which VAME uses in file and directory names."""

BatchSize: TypeAlias = Annotated[int, Field(ge=4)]
"""The windows of a training batch.

At least 4, because a test batch contains ``batch_size // 4`` windows and torch's
data loader refuses a batch of 0.
"""

KlStart: TypeAlias = Annotated[int, Field(ge=0)]
"""The last epoch at which the KL weight is 0."""

AnnealTime: TypeAlias = Annotated[int, Field(ge=1)]
"""How the KL weight rises after ``kl_start``.

The linear schedule raises the weight to 1 over this many epochs after
``kl_start``. The sigmoid schedule is centered on this epoch.
"""

TestFraction: TypeAlias = Annotated[float, Field(gt=0, lt=1)]
"""The fraction of the frames that the test set takes, strictly between 0 and 1."""

StepsPerEpoch: TypeAlias = Annotated[int, Field(ge=1)]
"""The batches of a training epoch.

At least 1, because VAME reads 0 as unset and draws one window per training frame.
"""

PredictionSteps: TypeAlias = Annotated[int, Field(ge=1)]
"""The frames past a window that the prediction decoder predicts.

At least 1, because below 1 VAME slices an empty target for the prediction decoder.
"""

SavgolOrder: TypeAlias = Annotated[int, Field(ge=0)]
"""The order of the Savitzky-Golay filter's polynomial, at least 0."""


def training_window_frames(time_window: int) -> int:
    """Return the frames of one training window, ``2 * time_window``.

    VAME's encoder reads the first half of a training window, and its prediction
    decoder predicts frames of the second half.
    """
    return 2 * time_window


def least_training_frames(time_window: int) -> int:
    """Return the fewest frames that contain a training window, ``2 * time_window + 1``.

    VAME draws a training window's start from the first ``N - window`` frames of
    ``N``, and draws from none when ``N`` equals the window.
    """
    return training_window_frames(time_window) + 1


def repeated_names(names: Sequence[str]) -> list[str]:
    """Return each name that occurs more than once in *names*, sorted."""
    return sorted({name for name in names if names.count(name) > 1})


def first_saving_epoch(
    kl_start: int, annealtime: int, anneal_function: AnnealFunction
) -> int:
    """Return the first epoch whose KL weight exceeds 0.99.

    VAME's ``train_model`` saves a model only at an epoch whose KL weight exceeds
    0.99, and the weight is 0 up to epoch *kl_start*. For the linear schedule
    this evaluates VAME's gate, ``min(1, (epoch - kl_start) / annealtime) >
    0.99``, at each integer epoch after *kl_start*. The sigmoid schedule first
    exceeds 0.99 at epoch ``annealtime + 6``, or at ``kl_start + 1`` when that is
    later.

    Raises:
        ValueError: If *kl_start* is negative or *annealtime* is below 1.
    """
    if kl_start < 0:
        msg = f"kl_start={kl_start} is negative"
        raise ValueError(msg)
    if annealtime < 1:
        msg = f"annealtime={annealtime} is below 1"
        raise ValueError(msg)
    if anneal_function == "sigmoid":
        return max(kl_start + 1, annealtime + 6)
    epoch = kl_start + 1
    while min(1.0, (epoch - kl_start) / annealtime) <= _SAVING_WEIGHT:
        epoch += 1
    return epoch


def check_model_is_saved(
    *,
    max_epochs: int,
    model_convergence: int,
    kl_start: int,
    annealtime: int,
    anneal_function: AnnealFunction,
) -> None:
    """Raise ValueError unless training runs until the first epoch that saves a model.

    VAME runs epochs 1 to ``max_epochs - 1``. It stops early once more than
    *model_convergence* epochs in a row have not saved, and the epochs before the
    first saving epoch do not save. When training ends before the first saving
    epoch, ``train_model`` returns without a model and without raising.
    """
    saving = first_saving_epoch(kl_start, annealtime, anneal_function)
    if max_epochs < saving + 1:
        msg = (
            f"max_epochs={max_epochs} does not save a model: the KL weight first "
            f"exceeds 0.99 at epoch {saving}, and VAME runs epochs 1 to "
            f"max_epochs - 1. Raise max_epochs to at least {saving + 1}."
        )
        raise ValueError(msg)
    if model_convergence < saving - 1:
        msg = (
            f"model_convergence={model_convergence} does not save a model: training "
            f"stops after {model_convergence + 1} epochs without a save, before "
            f"epoch {saving}, where the KL weight first exceeds 0.99. Raise "
            f"model_convergence to at least {saving - 1}."
        )
        raise ValueError(msg)


def check_distinct_references(centered: int | str, orientation: int | str) -> None:
    """Raise ValueError when the two reference keypoints are equal.

    Egocentric alignment needs two keypoints: one that it moves to the origin, and
    one whose direction from the first it rotates onto a fixed axis.
    """
    if centered == orientation:
        msg = (
            "egocentric alignment needs two distinct reference keypoints, and "
            "centered_reference_keypoint and orientation_reference_keypoint are "
            f"both {centered!r}"
        )
        raise ValueError(msg)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Preprocessing(_Model):
    """VAME's preprocessing settings.

    Egocentric alignment always runs, because VAME's model input reads dataset
    attributes that only alignment writes. Outlier cleaning always uses VAME's
    robust IQR rule.

    Attributes:
        centered_reference_keypoint: The keypoint that alignment moves to the
            origin of each frame.
        orientation_reference_keypoint: The keypoint whose direction from the
            centered one alignment rotates onto a fixed axis.
        run_lowconf_cleaning: Whether points whose confidence is below
            *pose_confidence* are replaced by interpolation.
        pose_confidence: The confidence below which low-confidence cleaning
            replaces a point.
        run_outlier_cleaning: Whether points outside *iqr_factor* interquartile
            ranges of their z-scored series are replaced by interpolation.
        iqr_factor: The interquartile ranges that outlier cleaning allows.
        run_savgol_filtering: Whether a Savitzky-Golay filter smooths each series.
        savgol_length: The filter's window length.
        savgol_order: The filter's polynomial order.
        run_rescaling: Whether VAME's rescaling runs last.
    """

    centered_reference_keypoint: str
    orientation_reference_keypoint: str
    run_lowconf_cleaning: bool
    pose_confidence: float
    run_outlier_cleaning: bool
    iqr_factor: int
    run_savgol_filtering: bool
    savgol_length: int
    savgol_order: SavgolOrder
    run_rescaling: bool

    @model_validator(mode="after")
    def _check_distinct_references(self) -> Self:
        check_distinct_references(
            self.centered_reference_keypoint, self.orientation_reference_keypoint
        )
        return self

    def check_references(self, keypoints: Sequence[str]) -> None:
        """Raise ValueError unless both reference keypoints are among *keypoints*."""
        for reference in (
            self.centered_reference_keypoint,
            self.orientation_reference_keypoint,
        ):
            if reference not in keypoints:
                msg = (
                    f"the reference keypoint {reference!r} is not among the "
                    f"keypoints {list(keypoints)}"
                )
                raise ValueError(msg)


class Training(_Model):
    """The model and training settings, under the names of VAME's project config.

    Each field sets the config key of its name. ``project_random_state`` seeds
    every random number generator that training draws from.
    ``prediction_decoder`` adds the decoder that predicts the next
    ``prediction_steps`` frames, and ``steps_per_epoch`` caps the batches of an
    epoch, which are otherwise one per training frame.

    A validator refuses settings under which training does not save a model; see
    :func:`check_model_is_saved`.
    """

    project_random_state: int
    zdims: int = Field(ge=1)
    time_window: int = Field(ge=1)
    max_epochs: int
    model_convergence: int
    batch_size: BatchSize
    beta: float
    learning_rate: float
    prediction_decoder: bool
    prediction_steps: PredictionSteps
    noise: bool
    scheduler: bool
    scheduler_step_size: int
    scheduler_gamma: float
    softplus: bool
    steps_per_epoch: StepsPerEpoch | None
    hidden_size_layer_1: int
    dropout_encoder: float
    hidden_size_rec: int
    hidden_size_pred: int
    mse_reconstruction_reduction: LossReduction
    mse_prediction_reduction: LossReduction
    kmeans_loss: int
    kmeans_lambda: float
    anneal_function: AnnealFunction
    kl_start: KlStart
    annealtime: AnnealTime

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


class TrainSet(_Model):
    """How VAME divides the training sessions into training and test frames.

    Attributes:
        test_fraction: The fraction of the frames that the test set takes.
        split_mode: How the test frames are chosen.
    """

    test_fraction: TestFraction
    split_mode: SplitMode


class Session(_Model):
    """One session's track, an ``.npz`` file that mosaic writes.

    The file contains ``frames``, the int64 frame numbers ``(T,)``, which are
    contiguous; ``position``, float ``(T, K, 2)``, the x and y of each keypoint
    in the order of the request's keypoints, NaN where a point is missing; and
    ``confidence``, float ``(T, K)``. Either array may be float32 or float64. The
    runner reads both as float64, because VAME's preprocessing computes in the
    dtype of its input.

    Attributes:
        name: The session's name in VAME's project: letters, digits and
            underscores, distinct within a request.
        track: The ``.npz`` file.
    """

    name: SessionName
    track: str


class EmbedSession(Session):
    """One session to embed, and the file for its latents.

    Attributes:
        latents: The ``.npz`` file that the runner writes. It contains
            ``latents``, float32 ``(T - time_window + 1, zdims)``. Latent ``i``
            encodes the window of frames ``i`` to ``i + time_window - 1``.
    """

    latents: str


def _check_distinct_names(sessions: Sequence[Session]) -> None:
    repeated = repeated_names([session.name for session in sessions])
    if repeated:
        msg = f"session names must be distinct, and {repeated} repeat"
        raise ValueError(msg)


class Payload(_Model):
    """The payload of one command. Its ``command`` names the command."""

    command: ClassVar[Command]


class Fit(Payload):
    """Train a model on the sessions.

    The runner builds VAME's project in a new directory inside *exchange_dir*,
    and removes the project once it has copied the trained model's state dict to
    *state_dict*.

    Attributes:
        sessions: The sessions to train on, in the order that VAME concatenates
            them.
        keypoints: The keypoint names, in the order of each track's keypoint
            axis.
        preprocessing: VAME's preprocessing settings.
        trainset: How the frames divide into training and test frames.
        training: The model and training settings.
        device: Where VAME trains.
        exchange_dir: The directory of the request's files.
        state_dict: The file that the runner writes, the trained model's state
            dict as VAME saves it.
    """

    command: ClassVar[Command] = "fit"
    sessions: list[Session] = Field(min_length=1)
    keypoints: list[str] = Field(min_length=2)
    preprocessing: Preprocessing
    trainset: TrainSet
    training: Training
    device: Device
    exchange_dir: str
    state_dict: str

    @model_validator(mode="after")
    def _check_sessions_and_references(self) -> Self:
        _check_distinct_names(self.sessions)
        self.preprocessing.check_references(self.keypoints)
        return self


class LoadModel(Payload):
    """Load a fitted model, which every later ``embed`` uses.

    Attributes:
        path: The model bundle, a joblib file of a dict that contains the fields of
            :class:`FittedModel`.
        device: Where VAME embeds.
    """

    command: ClassVar[Command] = "load_model"
    path: str
    device: Device


class Embed(Payload):
    """Encode each session's windows with the loaded model.

    Attributes:
        sessions: The sessions to embed.
        batch_size: How many windows VAME encodes at once. It changes latents at
            float32 rounding only.
        chunk_frames: The most frames that one embedding call reads. A longer
            session is embedded in chunks that overlap by ``time_window - 1``
            frames, and their latents are joined.
        exchange_dir: The directory of the request's files.
    """

    command: ClassVar[Command] = "embed"
    sessions: list[EmbedSession] = Field(min_length=1)
    batch_size: int = Field(ge=1)
    chunk_frames: int = Field(ge=1)
    exchange_dir: str

    @model_validator(mode="after")
    def _check_sessions(self) -> Self:
        _check_distinct_names(self.sessions)
        return self


class TrainingMetadata(_Model):
    """The fields of VAME's training metadata that embedding reads.

    Attributes:
        keypoints_used: The keypoints whose coordinates the model reads.
        extra_features: The further input series that the model reads.
    """

    keypoints_used: list[str]
    extra_features: list[str]


class FitResult(_Model):
    """What a fit trained.

    Attributes:
        project_name: The name of VAME's project, which VAME's state-dict file
            name contains.
        metadata: The fields of VAME's training metadata that embedding reads.
        num_features: The model's input features per frame, ``2K - 3`` for ``K``
            keypoints.
        seq_mean: The mean of every training value, which training subtracts.
        seq_std: The standard deviation of every training value, by which
            training divides.
        device: The device of the trained parameters.
        epochs: The epochs that training ran.
        sessions: The sessions that VAME trained on.
        train_frames: The frames of the training set.
        test_frames: The frames of the test set.
        straddling_fraction: The fraction of training-window starts whose window
            spans a join between two chunks of the training set. VAME
            concatenates the chunks, which are not adjacent in time.
    """

    project_name: str
    metadata: TrainingMetadata
    num_features: int
    seq_mean: float
    seq_std: float
    device: str
    epochs: int
    sessions: list[str]
    train_frames: int
    test_frames: int
    straddling_fraction: float


class FittedModel(BaseModel):
    """The fields of a model bundle that embedding reads.

    This model ignores the bundle's further entries, which mosaic reads.

    Attributes:
        project_name: The name of the fit's VAME project.
        state_dict: The bytes of the trained model's state-dict file.
        metadata: The fields of VAME's training metadata that embedding reads.
        num_features: The model's input features per frame.
        keypoints: The keypoint names, in the order of each track's keypoint
            axis.
        preprocessing: The preprocessing settings of the fit.
        training: The model and training settings of the fit.
        seq_mean: The fit's training mean, which embedding subtracts.
        seq_std: The fit's training standard deviation, by which embedding
            divides.
    """

    model_config = ConfigDict(extra="ignore")

    project_name: str
    state_dict: bytes
    metadata: TrainingMetadata
    num_features: int
    keypoints: list[str] = Field(min_length=2)
    preprocessing: Preprocessing
    training: Training
    seq_mean: float
    seq_std: float

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        self.preprocessing.check_references(self.keypoints)
        return self


class EmbeddedSession(_Model):
    """One embedded session.

    Attributes:
        name: The session's name.
        windows: The latents written, one per window of ``time_window`` frames.
    """

    name: str
    windows: int


class EmbedResult(_Model):
    """What an embedding wrote.

    Attributes:
        device: The device that VAME embedded on.
        sessions: Each session, in the order of the request.
    """

    device: str
    sessions: list[EmbeddedSession]


class EventData(_Model):
    """The data of an event. Its ``name`` names the event."""

    name: ClassVar[EventName]


class EpochEvent(EventData):
    """The end of a training epoch.

    Attributes:
        epoch: VAME's number of the epoch, from 1.
        total: The last epoch that training runs unless it converges first,
            ``max_epochs - 1``.
        losses: The epoch's losses and KL weight by name. A value that is not
            finite is ``None``, because JSON does not represent NaN.
    """

    name: ClassVar[EventName] = "epoch"
    epoch: int
    total: int
    losses: dict[str, float | None]


class StageEvent(EventData):
    """The start of a step of a command.

    Attributes:
        stage: The step.
        message: What the step works on.
    """

    name: ClassVar[EventName] = "stage"
    stage: Stage
    message: str
