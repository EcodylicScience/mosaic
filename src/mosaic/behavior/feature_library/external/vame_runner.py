"""The VAME runner, which trains VAME models and embeds tracks in VAME's environment.

mosaic's ``vame`` feature starts this program in the environment built under
``vame-env/``. The program serves the commands that ``vame_protocol`` declares
through ``runner_wire``, reading requests on standard input and writing replies
and events on standard output. Tracks cross as ``.npz`` files that a payload names
by path.

The program calls VAME's functions directly: ``init_new_project`` on movement
netCDF files, ``preprocessing`` with egocentric alignment, ``create_trainset`` and
``train_model`` for a fit, and ``embed_latent_vectors_optimized`` for an
embedding. It does not call ``segment_session``, which records any exception as
success, or ``VAMEPipeline``, which refuses movement files. Each command builds
its VAME project in a new directory, and removes it before it replies.

An embedding reads ``(x - seq_mean) / seq_std``, the normalization that training
applies and VAME's embedding functions omit. A long session is embedded in chunks
that overlap by ``time_window - 1`` frames, because VAME's embedding allocates an
array of every window of a session.

The module scope imports the standard library, pydantic and the two siblings
only. VAME, torch, xarray and joblib are imported at the first command that needs
them, after ``main`` has pointed standard output at standard error, set
``MPLBACKEND`` and configured logging. The program therefore imports, and answers
``Ready``, in an environment without them.

Usage, from this directory::

    vame-env/.venv/bin/python vame_runner.py

This is the only file in mosaic that imports VAME. VAME is licensed GPL-3.0, and
this program is part of mosaic, under AGPL-3.0-or-later. Section 13 of each
license permits combining a work under the other.
"""

# VAME and xarray are installed in the VAME environment only. The rules disabled
# below report the consequences, all of them in load_libraries:
# - reportMissingImports: their imports do not resolve in mosaic's environment;
# - reportUnknownVariableType: each name imported from them has an unknown type;
# - reportUnknownArgumentType: load_libraries passes those names to the typed
#   fields of Libraries, through which the rest of this file calls them.
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from runner_wire import (
    Dispatch,
    Emit,
    claim_stdout,
    installed_version,
    log_to_stderr,
    serve,
    serving,
)
from vame_protocol import (
    Device,
    Embed,
    EmbeddedSession,
    EmbedResult,
    EpochEvent,
    Fit,
    FitResult,
    FittedModel,
    LoadModel,
    Preprocessing,
    Session,
    SplitMode,
    Stage,
    StageEvent,
    Training,
    TrainingMetadata,
    least_training_frames,
    training_window_frames,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

TOOL: Final = "vame"
"""The tool name that ``Ready`` reports."""

DISTRIBUTION: Final = "vame-py"
"""VAME's distribution name, whose installed version ``Ready`` reports."""

PROJECT_NAME: Final = "mosaic"
"""The name of every VAME project that the runner creates."""

MODEL_NAME: Final = "VAME"
"""VAME's model name, which its state-dict file name contains."""

NORMALIZED: Final = "position_normalized"
"""The dataset variable that an embedding reads, ``(x - seq_mean) / seq_std``."""

INIT_KEYS: Final = (
    "vame_version",
    "project_name",
    "project_path",
    "creation_datetime",
    "session_names",
    "pose_estimation_filetype",
    "num_features",
    "keypoints",
    "video_type",
)
"""The config keys that ``init_new_project`` sets itself.

It raises on the first six when they arrive in ``config_kwargs``, overwrites
``num_features`` and ``keypoints``, and keeps a given ``video_type``.
"""

PREPROCESSING_CONFIG_KEYS: Final = (
    "pose_confidence",
    "iqr_factor",
    "savgol_length",
    "savgol_order",
)
"""The preprocessing settings that VAME reads from the project config.

The others are arguments of ``vame.preprocessing``.
"""

ARCHITECTURE_KEYS: Final = (
    "zdims",
    "time_window",
    "prediction_decoder",
    "prediction_steps",
    "hidden_size_layer_1",
    "hidden_size_rec",
    "hidden_size_pred",
    "dropout_encoder",
    "softplus",
)
"""The training settings that loading a model and embedding with it read."""

_INDIVIDUAL: Final = "individual_0"
"""The one individual of each session's netCDF file."""

log = logging.getLogger("vame_runner")


class RunnerError(RuntimeError):
    """A refusal of a request that VAME cannot complete as given."""


# --- The VAME functions that the runner calls ---


VameConfig: TypeAlias = dict[str, object]
"""A VAME project config, which VAME's functions read and update in place."""


class _InitNewProject(Protocol):
    """``vame.init_new_project``."""

    def __call__(
        self,
        *,
        project_name: str,
        poses_estimations: list[str],
        source_software: Literal["movement"],
        working_directory: str,
        config_kwargs: VameConfig,
    ) -> tuple[str, VameConfig]: ...


class _Preprocess(Protocol):
    """``vame.preprocessing``. It returns the name of the variable written last."""

    def __call__(
        self,
        *,
        config: VameConfig,
        centered_reference_keypoint: str,
        orientation_reference_keypoint: str,
        run_lowconf_cleaning: bool,
        run_egocentric_alignment: bool,
        run_outlier_cleaning: bool,
        run_savgol_filtering: bool,
        run_rescaling: bool,
        save_logs: bool,
    ) -> str: ...


class _CreateTrainset(Protocol):
    """``vame.create_trainset``."""

    def __call__(
        self,
        *,
        config: VameConfig,
        test_fraction: float,
        read_from_variable: str,
        split_mode: SplitMode,
        save_logs: bool,
    ) -> None: ...


class _TrainModel(Protocol):
    """``vame.train_model``."""

    def __call__(self, *, config: VameConfig, save_logs: bool) -> None: ...


class _UpdateConfig(Protocol):
    """``vame.update_config``, which updates *config* and its file."""

    def __call__(
        self, *, config: VameConfig, config_update: VameConfig
    ) -> VameConfig: ...


class _EmbedLatents(Protocol):
    """``vame.analysis.pose_segmentation.embed_latent_vectors_optimized``."""

    def __call__(
        self,
        *,
        config: VameConfig,
        sessions: list[str],
        fixed: bool,
        read_from_variable: str,
        overwrite: bool,
        batch_size: int,
    ) -> list[npt.NDArray[np.float32]]: ...


class TrainingSteps(Protocol):
    """``vame.model.rnn_vae``, whose ``train_model`` calls these once per epoch.

    In VAME 0.14.4 ``train`` returns the KL weight, the training loss, its
    k-means, KL, reconstruction and prediction terms, and a step count, and
    ``test`` returns the test set's reconstruction loss, loss and k-means term.
    """

    train: Callable[..., Sequence[float]]
    test: Callable[..., Sequence[float]]


class _DataArray(Protocol):
    @property
    def dims(self) -> tuple[str, ...]: ...

    @property
    def values(self) -> npt.NDArray[np.floating]: ...


_Variable: TypeAlias = tuple[tuple[str, ...], "npt.NDArray[np.floating]"]
"""A dataset variable as xarray takes it: its dimensions and its values."""


class _Dataset(Protocol):
    @property
    def sizes(self) -> Mapping[str, int]: ...

    def __getitem__(self, name: str) -> _DataArray: ...

    def __setitem__(self, name: str, value: _Variable) -> None: ...

    def isel(self, *, time: slice) -> _Dataset: ...

    def load(self) -> _Dataset: ...

    def to_netcdf(self, path: str, *, engine: Literal["netcdf4"]) -> None: ...


class _NewDataset(Protocol):
    """``xarray.Dataset``."""

    def __call__(
        self,
        *,
        data_vars: Mapping[str, _Variable],
        coords: Mapping[str, npt.NDArray[np.int64] | list[str]],
        attrs: Mapping[str, str],
    ) -> _Dataset: ...


class _OpenDataset(Protocol):
    """``xarray.open_dataset``."""

    def __call__(
        self, path: str, *, engine: Literal["netcdf4"]
    ) -> contextlib.AbstractContextManager[_Dataset]: ...


@dataclass(frozen=True)
class Libraries:
    """The parts of VAME and xarray that the runner calls, imported once per process.

    Attributes:
        init_new_project: Creates a VAME project from pose files.
        preprocessing: Runs VAME's preprocessing.
        create_trainset: Writes VAME's training and test arrays.
        train_model: Trains VAME's model.
        update_config: Updates a project's config.
        embed_latent_vectors: Encodes a session's windows in batches.
        training_steps: The module whose epoch functions the epoch hook wraps.
        project_schema: VAME's project config model. Its fields are the keys
            that the config accepts.
        get_device: VAME's device selection: CUDA, then MPS, then the CPU.
        new_dataset: Builds an xarray dataset.
        open_dataset: Opens a netCDF file as an xarray dataset.
    """

    init_new_project: _InitNewProject
    preprocessing: _Preprocess
    create_trainset: _CreateTrainset
    train_model: _TrainModel
    update_config: _UpdateConfig
    embed_latent_vectors: _EmbedLatents
    training_steps: TrainingSteps
    project_schema: type[BaseModel]
    get_device: Callable[[], torch.device]
    new_dataset: _NewDataset
    open_dataset: _OpenDataset


@functools.cache
def load_libraries() -> Libraries:
    """Import VAME and xarray, once per process.

    ``main`` has set ``MPLBACKEND`` and configured logging before this runs, so
    VAME's import neither opens a window nor adds a root log handler.
    """
    from vame import (
        create_trainset,
        init_new_project,
        preprocessing,
        train_model,
        update_config,
    )
    from vame.analysis.pose_segmentation import embed_latent_vectors_optimized
    from vame.model import rnn_vae
    from vame.schemas.project import ProjectSchema
    from vame.util.auxiliary import get_device
    from xarray import Dataset, open_dataset

    return Libraries(
        init_new_project=init_new_project,
        preprocessing=preprocessing,
        create_trainset=create_trainset,
        train_model=train_model,
        update_config=update_config,
        embed_latent_vectors=embed_latent_vectors_optimized,
        training_steps=rnn_vae,
        project_schema=ProjectSchema,
        get_device=get_device,
        new_dataset=Dataset,
        open_dataset=open_dataset,
    )


# --- The project config ---


def pinned_config() -> VameConfig:
    """The config keys that the runner sets for every project.

    The input is not egocentric data, because ``preprocessing`` aligns it.
    Extra features and pretrained weights are off, every session trains, and
    outlier cleaning uses VAME's robust rule, without which it changes nothing.
    """
    return {
        "egocentric_data": False,
        "extra_features": [],
        "pretrained_weights": False,
        "pretrained_model": "None",
        "all_data": "yes",
        "model_name": MODEL_NAME,
        "model_snapshot": 50,
        "robust": True,
    }


def _preprocessing_config(preprocessing: Preprocessing) -> VameConfig:
    settings = preprocessing.model_dump()
    return {key: settings[key] for key in PREPROCESSING_CONFIG_KEYS}


def fit_config(training: Training, preprocessing: Preprocessing) -> VameConfig:
    """The project config of a fit: every training setting, preprocessing and pins."""
    return {
        **training.model_dump(),
        **_preprocessing_config(preprocessing),
        **pinned_config(),
    }


def embed_config(model: FittedModel) -> VameConfig:
    """The project config of an embedding, built from the bundle's settings.

    It contains the architecture, the preprocessing and the pins. It is not read
    from a fitted ``config.yaml``, whose project keys ``init_new_project`` refuses.
    """
    training = model.training.model_dump()
    return {
        **{key: training[key] for key in ARCHITECTURE_KEYS},
        **_preprocessing_config(model.preprocessing),
        **pinned_config(),
    }


def check_config_keys(config: Mapping[str, object], known: Collection[str]) -> None:
    """Raise RunnerError unless VAME accepts every key of *config* from the runner.

    ``init_new_project`` drops a key that its project schema lacks, without
    logging, and sets the keys of :data:`INIT_KEYS` itself.

    Args:
        config: The config passed to ``init_new_project``.
        known: The fields of VAME's project schema.
    """
    unknown = sorted(set(config) - set(known))
    if unknown:
        msg = f"VAME's project config does not define {unknown}"
        raise RunnerError(msg)
    owned = sorted(set(config) & set(INIT_KEYS))
    if owned:
        msg = f"VAME's project creation sets {owned} itself"
        raise RunnerError(msg)


def preprocess(
    run: _Preprocess, config: VameConfig, preprocessing: Preprocessing
) -> str:
    """Run VAME's preprocessing with egocentric alignment.

    Args:
        run: VAME's ``preprocessing``.
        config: The project's config.
        preprocessing: The settings.

    Returns:
        The dataset variable that the last step wrote.
    """
    return run(
        config=config,
        centered_reference_keypoint=preprocessing.centered_reference_keypoint,
        orientation_reference_keypoint=preprocessing.orientation_reference_keypoint,
        run_lowconf_cleaning=preprocessing.run_lowconf_cleaning,
        run_egocentric_alignment=True,
        run_outlier_cleaning=preprocessing.run_outlier_cleaning,
        run_savgol_filtering=preprocessing.run_savgol_filtering,
        run_rescaling=preprocessing.run_rescaling,
        save_logs=False,
    )


# --- Sessions ---


def write_session(
    new_dataset: _NewDataset,
    session: Session,
    keypoints: Sequence[str],
    directory: Path,
) -> Path:
    """Write *session*'s track as a movement netCDF file in *directory*.

    The file contains one individual, in float64. Its ``time`` coordinate is the
    track's frame numbers, which VAME does not read.

    Returns:
        The file's path, whose stem is the session's name.

    Raises:
        RunnerError: If the track's arrays do not match *keypoints*.
    """
    import numpy as np

    # VAME's preprocessing computes in the file's dtype. With some BLAS builds,
    # Apple's among them, egocentric alignment in float32 leaves the orientation
    # keypoint's x as rounding residue of a few distinct values. When more than
    # half of them are equal, the series has an interquartile range of 0, and
    # outlier cleaning replaces all of it with NaN.
    with np.load(session.track) as track:
        frames = np.asarray(track["frames"], dtype=np.int64)
        position = np.asarray(track["position"], dtype=np.float64)
        confidence = np.asarray(track["confidence"], dtype=np.float64)
    expected = (len(frames), len(keypoints))
    if position.shape != (*expected, 2) or confidence.shape != expected:
        msg = (
            f"session {session.name}: position {position.shape} and confidence "
            f"{confidence.shape} do not match {expected[0]} frames of "
            f"{expected[1]} keypoints"
        )
        raise RunnerError(msg)
    dataset = new_dataset(
        data_vars={
            "position": (
                ("time", "space", "keypoints", "individuals"),
                position.transpose(0, 2, 1)[..., np.newaxis],
            ),
            "confidence": (
                ("time", "keypoints", "individuals"),
                confidence[..., np.newaxis],
            ),
        },
        coords={
            "time": frames,
            "space": ["x", "y"],
            "keypoints": list(keypoints),
            "individuals": [_INDIVIDUAL],
        },
        attrs={"source_software": "mosaic"},
    )
    path = directory / f"{session.name}.nc"
    dataset.to_netcdf(str(path), engine="netcdf4")
    return path


def _new_project(
    libraries: Libraries,
    work: Path,
    sessions: Sequence[Session],
    keypoints: Sequence[str],
    config: VameConfig,
    project_name: str,
) -> tuple[VameConfig, Path]:
    """Create a VAME project in *work* from *sessions*.

    Returns:
        The project's config and directory.
    """
    poses = work / "poses"
    poses.mkdir()
    paths = [
        write_session(libraries.new_dataset, session, keypoints, poses)
        for session in sessions
    ]
    check_config_keys(config, libraries.project_schema.model_fields)
    _, created = libraries.init_new_project(
        project_name=project_name,
        poses_estimations=[str(path) for path in paths],
        source_software="movement",
        working_directory=str(work / "project"),
        config_kwargs=dict(config),
    )
    return created, Path(str(created["project_path"]))


def _stage(emit: Emit, stage: Stage, message: str) -> None:
    log.info("%s: %s", stage, message)
    emit(StageEvent.name, StageEvent(stage=stage, message=message).model_dump())


# --- The split ---


def check_split(
    train: tuple[int, int],
    test: tuple[int, int],
    *,
    time_window: int,
    batch_size: int,
    steps_per_epoch: int | None,
) -> None:
    """Raise RunnerError unless VAME can train on the split.

    Args:
        train: The shape of VAME's training array, ``(features, frames)``.
        test: The shape of VAME's test array.
        time_window: The frames of one embedding window. A training window is
            twice as long.
        batch_size: The windows of a training batch. A test batch contains a
            quarter of them.
        steps_per_epoch: The batches of an epoch, or ``None`` for one window per
            frame.
    """
    features = train[0]
    window = training_window_frames(time_window)
    least = least_training_frames(time_window)
    for name, (_, frames), remedy in (
        ("training", train, "Add frames, or lower test_fraction or time_window."),
        ("test", test, "Raise test_fraction, or lower time_window."),
    ):
        if frames < least:
            msg = (
                f"the {name} set contains {frames} frames, and a training window of "
                f"2 * time_window = {window} frames needs at least {least}. "
                f"{remedy}"
            )
            raise RunnerError(msg)
        if frames < features:
            msg = (
                f"the {name} set contains {frames} frames, fewer than its {features} "
                "features, and VAME would read it transposed. Add frames."
            )
            raise RunnerError(msg)
    if steps_per_epoch is None:
        for name, frames, batch in (
            ("training", train[1], batch_size),
            ("test", test[1], batch_size // 4),
        ):
            if frames < batch:
                msg = (
                    f"the {name} set contains {frames} frames, fewer than its batch "
                    f"of {batch} windows, which batch_size={batch_size} sets. "
                    "Lower batch_size or set steps_per_epoch."
                )
                raise RunnerError(msg)
    if features == time_window:
        msg = (
            f"the model reads {features} features per frame, as many as "
            "time_window, and VAME's embedding would read every window "
            "transposed. Change time_window."
        )
        raise RunnerError(msg)


class _Span(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: int
    end: int


class _SessionSplit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_chunks: list[_Span]


class _ChunkSplit(BaseModel):
    """The split of ``mode_2``: a random test chunk from each session."""

    model_config = ConfigDict(extra="ignore")

    method: Literal["random_continuous_chunks"]
    session_splits: list[_SessionSplit]


class _SequentialSplit(BaseModel):
    """The split of ``mode_1``: the first frames of the concatenated sessions."""

    model_config = ConfigDict(extra="ignore")

    method: Literal["sequential_split"]
    split_index: int
    session_boundaries: list[_Span]


_SPLIT_DETAILS: Final[TypeAdapter[_ChunkSplit | _SequentialSplit]] = TypeAdapter(
    Annotated[_ChunkSplit | _SequentialSplit, Field(discriminator="method")]
)


def train_chunk_lengths(details: Mapping[str, object]) -> list[int]:
    """The lengths of the chunks that VAME concatenates into its training array.

    Args:
        details: ``split_information.split_details`` of VAME's training
            metadata.
    """
    split = _SPLIT_DETAILS.validate_python(details)
    if isinstance(split, _ChunkSplit):
        return [
            chunk.end - chunk.start
            for session in split.session_splits
            for chunk in session.train_chunks
        ]
    lengths = [
        span.end - max(span.start, split.split_index)
        for span in split.session_boundaries
    ]
    return [length for length in lengths if length > 0]


def straddling_fraction(chunk_lengths: Sequence[int], window: int) -> float:
    """The fraction of window starts whose window spans a join between two chunks.

    VAME draws a training window's start uniformly from the first ``N - window``
    frames of the concatenated chunks, ``N`` frames in all. A window spans the
    join at frame ``b`` when it starts after ``b - window`` and before ``b``.
    """
    starts = sum(chunk_lengths) - window
    straddling = 0
    counted_until = 0
    join = 0
    for length in chunk_lengths[:-1]:
        join += length
        low = max(join - window + 1, counted_until)
        high = min(join, starts)
        if high > low:
            straddling += high - low
            counted_until = high
    return straddling / starts


# --- Chunks of a long session ---


def chunk_bounds(
    frames: int, chunk_frames: int, time_window: int
) -> list[tuple[int, int]]:
    """The frame range ``[start, stop)`` of each chunk of a session.

    Consecutive chunks overlap by ``time_window - 1`` frames. Each window of the
    session then lies in exactly one chunk, and each chunk contains a whole window.

    Raises:
        RunnerError: If the session or a chunk is shorter than *time_window*.
    """
    if frames < time_window or chunk_frames < time_window:
        msg = (
            f"a session of {frames} frames, in chunks of {chunk_frames}, does not "
            f"contain a whole window of time_window={time_window} frames"
        )
        raise RunnerError(msg)
    step = chunk_frames - time_window + 1
    return [
        (start, min(start + chunk_frames, frames))
        for start in range(0, frames - time_window + 1, step)
    ]


def write_latents(path: str, latents: npt.NDArray[np.floating]) -> None:
    """Write *latents* as float32 to the ``.npz`` file *path*, under that exact name.

    ``np.savez`` appends ``.npz`` to a path that lacks the suffix. It does not
    rename an open file.
    """
    import numpy as np

    with Path(path).open("wb") as handle:
        np.savez(handle, latents=latents.astype(np.float32))


def stitch(
    chunks: Sequence[npt.NDArray[np.float32]],
    bounds: Sequence[tuple[int, int]],
    time_window: int,
) -> npt.NDArray[np.float32]:
    """Join the latents of a session's chunks into the session's latents.

    Raises:
        RunnerError: If a chunk's latents are not one per window.
    """
    import numpy as np

    for latents, (start, stop) in zip(chunks, bounds, strict=True):
        windows = stop - start - time_window + 1
        if len(latents) != windows:
            msg = (
                f"the chunk of frames {start} to {stop} returned {len(latents)} "
                f"latents for its {windows} windows"
            )
            raise RunnerError(msg)
    return np.concatenate(chunks, axis=0)


# --- The device ---


def bind_device(
    modules: Mapping[str, object],
    lookup: object,
    device: object,
    *,
    is_device: Callable[[object], bool],
) -> list[str]:
    """Point every binding of VAME's device lookup, and every ``DEVICE``, at *device*.

    Each VAME module binds the lookup by importing its name, and some store its
    result in a module-level ``DEVICE`` at import. Replacing the lookup in
    ``vame.util.auxiliary`` alone changes neither binding.

    Args:
        modules: The loaded modules by name, ``sys.modules`` at run time.
        lookup: VAME's ``get_device``.
        device: The device to compute on.
        is_device: Whether a value is a device.

    Returns:
        The rebound attributes, as ``module.attribute``, sorted.
    """

    def chosen() -> object:
        return device

    bound: list[str] = []
    for name in sorted(modules):
        module = modules[name]
        if module is None or not (name == "vame" or name.startswith("vame.")):
            continue
        for attribute, value in list(vars(module).items()):
            if value is lookup:
                setattr(module, attribute, chosen)
            elif attribute == "DEVICE" and is_device(value):
                setattr(module, attribute, device)
            else:
                continue
            bound.append(f"{name}.{attribute}")
    return sorted(bound)


def _resolve_device(libraries: Libraries, requested: Device) -> torch.device:
    """The device that *requested* names in this environment.

    Raises:
        RunnerError: If the named device is not available.
    """
    import torch

    if requested == "auto":
        return libraries.get_device()
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if not available[requested]:
        msg = f"device={requested!r} is not available in this environment"
        raise RunnerError(msg)
    return torch.device(requested)


# --- Epochs ---

_TRAIN_LOSSES: Final = (
    "kl_weight",
    "train_loss",
    "train_kmeans_loss",
    "train_kl_loss",
    "train_mse_loss",
    "train_future_loss",
)
_TEST_LOSSES: Final = ("test_mse_loss", "test_loss", "test_kmeans_loss")


def _finite_or_none(value: float) -> float | None:
    import math

    return value if math.isfinite(value) else None


def epoch_losses(
    train: Sequence[float], test: Sequence[float]
) -> dict[str, float | None]:
    """Name an epoch's losses, from what VAME's ``train`` and ``test`` returned."""
    named = [
        *zip(_TRAIN_LOSSES, train, strict=False),
        *zip(_TEST_LOSSES, test, strict=False),
    ]
    return {name: _finite_or_none(float(value)) for name, value in named}


@dataclass
class EpochCount:
    """The epochs that a fit has run, and the last training losses.

    Attributes:
        epochs: The epochs whose test pass has finished.
        train: What VAME's ``train`` returned last.
    """

    epochs: int = 0
    train: Sequence[float] = ()


@contextlib.contextmanager
def epoch_events(steps: TrainingSteps, emit: Emit, total: int) -> Generator[EpochCount]:
    """Emit an :class:`EpochEvent` after each epoch that VAME trains.

    ``train`` and ``test`` of *steps* are replaced by wrappers for the duration,
    and restored on exit. ``train_model`` does not report the end of an epoch,
    and its loss files miss the epoch that ends training at convergence.

    Args:
        steps: The module whose functions ``train_model`` calls each epoch.
        emit: Writes an event.
        total: The last epoch that training can run.

    Yields:
        The epoch count, which grows as training runs.
    """
    count = EpochCount()
    train, test = steps.train, steps.test

    def counted_train(*arguments: object, **keywords: object) -> Sequence[float]:
        result = train(*arguments, **keywords)
        count.train = result
        return result

    def counted_test(*arguments: object, **keywords: object) -> Sequence[float]:
        result = test(*arguments, **keywords)
        count.epochs += 1
        event = EpochEvent(
            epoch=count.epochs, total=total, losses=epoch_losses(count.train, result)
        )
        emit(EpochEvent.name, event.model_dump())
        return result

    steps.train, steps.test = counted_train, counted_test
    try:
        yield count
    finally:
        steps.train, steps.test = train, test


# --- VAME's training metadata ---


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    keypoints_used: list[str]
    extra_features: list[str] = Field(default_factory=list[str])


class _SplitInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sessions_used: list[str]
    split_details: dict[str, JsonValue]


class _TrainingMetadataFile(BaseModel):
    """The parts of VAME's ``metadata.json`` that the runner reads."""

    model_config = ConfigDict(extra="ignore")

    parameters: _Parameters
    split_information: _SplitInformation


# --- Commands ---


class VameRunner:
    """The runner's state: the device it computes on and the loaded model."""

    def __init__(self) -> None:
        self._requested: Device | None = None
        self._device: str = ""
        self._model: FittedModel | None = None

    def _bind(self, requested: Device) -> str:
        """Compute on the device that *requested* names, from the first command on.

        Returns:
            The device's type.

        Raises:
            RunnerError: If an earlier command named another device.
        """
        if self._requested is not None:
            if requested != self._requested:
                msg = (
                    f"this runner computes on device={self._requested!r} and "
                    f"cannot switch to {requested!r}"
                )
                raise RunnerError(msg)
            return self._device
        import torch

        libraries = load_libraries()
        device = _resolve_device(libraries, requested)
        bound = bind_device(
            sys.modules,
            libraries.get_device,
            device,
            is_device=lambda value: isinstance(value, torch.device),
        )
        log.info("device %s, bound at %s", device, ", ".join(bound))
        self._requested = requested
        self._device = device.type
        return self._device

    def handle_fit(self, request: Fit, emit: Emit) -> FitResult:
        """Train a model, and copy its state dict to the request's file.

        The model's project is removed before the reply.
        """
        device = self._bind(request.device)
        libraries = load_libraries()
        work = Path(tempfile.mkdtemp(prefix="vame-fit-", dir=request.exchange_dir))
        try:
            return self._fit(libraries, request, device, work, emit)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _fit(
        self,
        libraries: Libraries,
        request: Fit,
        device: str,
        work: Path,
        emit: Emit,
    ) -> FitResult:
        import numpy as np
        import torch

        training = request.training
        _stage(emit, "prepare", f"{len(request.sessions)} sessions")
        config, project = _new_project(
            libraries,
            work,
            request.sessions,
            request.keypoints,
            fit_config(training, request.preprocessing),
            PROJECT_NAME,
        )
        _stage(emit, "preprocess", "egocentric alignment and cleaning")
        variable = preprocess(libraries.preprocessing, config, request.preprocessing)
        _stage(emit, "trainset", f"test fraction {request.trainset.test_fraction}")
        libraries.create_trainset(
            config=config,
            test_fraction=request.trainset.test_fraction,
            read_from_variable=variable,
            split_mode=request.trainset.split_mode,
            save_logs=False,
        )
        train_directory = project / "data" / "train"
        train = np.load(train_directory / "train_seq.npy", mmap_mode="r").shape
        test = np.load(train_directory / "test_seq.npy", mmap_mode="r").shape
        check_split(
            (train[0], train[1]),
            (test[0], test[1]),
            time_window=training.time_window,
            batch_size=training.batch_size,
            steps_per_epoch=training.steps_per_epoch,
        )
        _stage(emit, "train", f"up to {training.max_epochs - 1} epochs on {device}")
        with epoch_events(
            libraries.training_steps, emit, training.max_epochs - 1
        ) as count:
            libraries.train_model(config=config, save_logs=False)
        state_dict = _state_dict_path(project, PROJECT_NAME)
        if not state_dict.is_file():
            msg = (
                f"training ran {count.epochs} epochs and did not save a model. VAME "
                "saves a model only when an epoch's KL weight exceeds 0.99 and its "
                "test loss is finite."
            )
            raise RunnerError(msg)
        _ = shutil.copyfile(state_dict, request.state_dict)
        state: Mapping[str, torch.Tensor] = torch.load(state_dict, weights_only=True)
        metadata = _TrainingMetadataFile.model_validate_json(
            (train_directory / "metadata.json").read_text()
        )
        return FitResult(
            project_name=PROJECT_NAME,
            metadata=TrainingMetadata(
                keypoints_used=metadata.parameters.keypoints_used,
                extra_features=metadata.parameters.extra_features,
            ),
            num_features=int(train[0]),
            seq_mean=float(np.load(train_directory / "seq_mean.npy")),
            seq_std=float(np.load(train_directory / "seq_std.npy")),
            device=next(iter(state.values())).device.type,
            epochs=count.epochs,
            sessions=metadata.split_information.sessions_used,
            train_frames=int(train[1]),
            test_frames=int(test[1]),
            straddling_fraction=straddling_fraction(
                train_chunk_lengths(metadata.split_information.split_details),
                training_window_frames(training.time_window),
            ),
        )

    def handle_load_model(self, request: LoadModel, emit: Emit) -> None:
        """Load the bundle that every later embedding uses."""
        import joblib

        _ = self._bind(request.device)
        bundle: object = joblib.load(request.path)
        self._model = FittedModel.model_validate(bundle)
        log.info("loaded %s", request.path)

    def handle_embed(self, request: Embed, emit: Emit) -> EmbedResult:
        """Embed each session with the loaded model, in a project removed after."""
        model = self._model
        if model is None:
            msg = "embed needs a loaded model: send load_model first"
            raise RunnerError(msg)
        libraries = load_libraries()
        work = Path(tempfile.mkdtemp(prefix="vame-embed-", dir=request.exchange_dir))
        try:
            return self._embed(libraries, model, request, work, emit)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _embed(
        self,
        libraries: Libraries,
        model: FittedModel,
        request: Embed,
        work: Path,
        emit: Emit,
    ) -> EmbedResult:
        time_window = model.training.time_window
        _stage(emit, "prepare", f"{len(request.sessions)} sessions")
        config, project = _new_project(
            libraries,
            work,
            request.sessions,
            model.keypoints,
            embed_config(model),
            model.project_name,
        )
        state_dict = _state_dict_path(project, model.project_name)
        state_dict.parent.mkdir(parents=True, exist_ok=True)
        _ = state_dict.write_bytes(model.state_dict)
        metadata = project / "data" / "train" / "metadata.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        _ = metadata.write_text(json.dumps({"parameters": model.metadata.model_dump()}))
        _ = libraries.update_config(
            config=config, config_update={"num_features": model.num_features}
        )
        _stage(emit, "preprocess", "egocentric alignment and cleaning")
        variable = preprocess(libraries.preprocessing, config, model.preprocessing)
        processed = project / "data" / "processed"
        chunks: dict[str, list[tuple[int, int]]] = {}
        names: list[str] = []
        for session in request.sessions:
            with libraries.open_dataset(
                str(processed / f"{session.name}_processed.nc"), engine="netcdf4"
            ) as opened:
                dataset = opened.load()
            bounds = chunk_bounds(
                dataset.sizes["time"], request.chunk_frames, time_window
            )
            chunks[session.name] = bounds
            for index, (start, stop) in enumerate(bounds):
                name = f"{session.name}-{index:06d}"
                chunk = dataset.isel(time=slice(start, stop))
                source = chunk[variable]
                chunk[NORMALIZED] = (
                    source.dims,
                    (source.values - model.seq_mean) / model.seq_std,
                )
                chunk.to_netcdf(
                    str(processed / f"{name}_processed.nc"), engine="netcdf4"
                )
                (project / "results" / name / MODEL_NAME).mkdir(parents=True)
                names.append(name)
        _stage(
            emit, "embed", f"{len(names)} chunks of {len(request.sessions)} sessions"
        )
        latents = iter(
            libraries.embed_latent_vectors(
                config=config,
                sessions=names,
                fixed=False,
                read_from_variable=NORMALIZED,
                overwrite=True,
                batch_size=request.batch_size,
            )
        )
        embedded: list[EmbeddedSession] = []
        for session in request.sessions:
            bounds = chunks[session.name]
            joined = stitch([next(latents) for _ in bounds], bounds, time_window)
            write_latents(session.latents, joined)
            embedded.append(EmbeddedSession(name=session.name, windows=len(joined)))
        return EmbedResult(device=self._device, sessions=embedded)


def _state_dict_path(project: Path, project_name: str) -> Path:
    """Where VAME keeps the best model of *project*."""
    return project / "model" / "best_model" / f"{MODEL_NAME}_{project_name}.pkl"


# --- Dispatch ---


def dispatch(runner: VameRunner) -> Dispatch:
    """The handler of each command, run on *runner*."""
    return {
        Fit.command: serving(Fit, runner.handle_fit),
        LoadModel.command: serving(LoadModel, runner.handle_load_model),
        Embed.command: serving(Embed, runner.handle_embed),
    }


# --- Entry point ---


def main() -> None:
    protocol = claim_stdout()
    os.environ["MPLBACKEND"] = "Agg"
    log_to_stderr()
    serve(
        dispatch(VameRunner()),
        TOOL,
        installed_version(DISTRIBUTION),
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
