"""The VAME runner program, run in mosaic's environment, without VAME.

The program imports VAME, torch and xarray at its first command. It therefore
imports, and serves ``Ready`` and ``shutdown``, in an environment without them,
and its helpers run here on plain values.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Final, get_args

import numpy as np
import pytest
from pydantic import JsonValue

from mosaic.behavior.feature_library.external.runner_wire import (
    NOT_INSTALLED,
)
from mosaic.behavior.feature_library.external.vame_protocol import (
    Command,
    Preprocessing,
    Session,
    Training,
)
from mosaic.behavior.feature_library.vame_sessions import (
    SessionSettings,
    build_sessions,
    write_session_tracks,
)
from mosaic.core.pipeline.types import PoseConfig
from mosaic.core.pipeline.toolenv import subprocess_env
from tests.helpers import (
    SOURCE_ROOT,
    VAME_RUNNER,
    EventRecorder,
    LineChildren,
    assert_ready_then_shutdown,
    import_runner_program,
    keypoint_rows,
)

_RUNNER: Final = SOURCE_ROOT / VAME_RUNNER
"""The runner program."""

_SIBLINGS: Final = ("vame_protocol", "runner_wire")

PROJECT_FIELDS: Final = frozenset(
    {
        "vame_version",
        "project_name",
        "project_path",
        "creation_datetime",
        "session_names",
        "pose_estimation_filetype",
        "video_type",
        "project_random_state",
        "all_data",
        "keypoints",
        "extra_features",
        "egocentric_data",
        "pose_confidence",
        "robust",
        "iqr_factor",
        "savgol_filter",
        "savgol_length",
        "savgol_order",
        "test_fraction",
        "model_name",
        "pretrained_model",
        "pretrained_weights",
        "num_features",
        "batch_size",
        "max_epochs",
        "model_snapshot",
        "model_convergence",
        "transition_function",
        "beta",
        "beta_norm",
        "zdims",
        "learning_rate",
        "time_window",
        "prediction_decoder",
        "prediction_steps",
        "noise",
        "scheduler",
        "scheduler_step_size",
        "scheduler_gamma",
        "scheduler_threshold",
        "softplus",
        "steps_per_epoch",
        "n_clusters",
        "segmentation_algorithms",
        "hmm_trained",
        "hmm_n_iter",
        "individual_segmentation",
        "n_init_kmeans",
        "length_of_motif_video",
        "min_dist",
        "n_neighbors",
        "num_points",
        "hidden_size_layer_1",
        "hidden_size_layer_2",
        "dropout_encoder",
        "hidden_size_rec",
        "dropout_rec",
        "n_layers",
        "hidden_size_pred",
        "dropout_pred",
        "mse_reconstruction_reduction",
        "mse_prediction_reduction",
        "kmeans_loss",
        "kmeans_lambda",
        "anneal_function",
        "kl_start",
        "annealtime",
    }
)
"""The fields of VAME 0.14.4's ``ProjectSchema``, the keys of a project config."""


@pytest.fixture(scope="module")
def runner() -> Iterator[ModuleType]:
    """The runner program, imported into this process and then unimported."""
    with import_runner_program(_RUNNER, _SIBLINGS) as program:
        yield program


def _preprocessing() -> Preprocessing:
    return Preprocessing(
        centered_reference_keypoint="nose",
        orientation_reference_keypoint="tail_base",
        run_lowconf_cleaning=True,
        pose_confidence=0.5,
        run_outlier_cleaning=False,
        iqr_factor=3,
        run_savgol_filtering=True,
        savgol_length=7,
        savgol_order=3,
        run_rescaling=False,
    )


def _training() -> Training:
    return Training(
        project_random_state=7,
        zdims=4,
        time_window=10,
        max_epochs=8,
        model_convergence=5,
        batch_size=16,
        beta=1.0,
        learning_rate=1e-3,
        prediction_decoder=True,
        prediction_steps=5,
        noise=False,
        scheduler=True,
        scheduler_step_size=100,
        scheduler_gamma=0.2,
        softplus=False,
        steps_per_epoch=3,
        hidden_size_layer_1=32,
        dropout_encoder=0.0,
        hidden_size_rec=32,
        hidden_size_pred=32,
        mse_reconstruction_reduction="sum",
        mse_prediction_reduction="mean",
        kmeans_loss=4,
        kmeans_lambda=0.1,
        anneal_function="linear",
        kl_start=0,
        annealtime=1,
    )


# --- The program ---


def test_the_program_imports_without_its_libraries(runner: ModuleType) -> None:
    assert callable(runner.main)
    assert "vame" not in sys.modules
    assert "xarray" not in sys.modules


def test_the_dispatch_serves_exactly_the_commands_of_the_protocol(
    runner: ModuleType,
) -> None:
    dispatch = runner.dispatch(runner.VameRunner())

    assert sorted(dispatch) == sorted(get_args(Command))


def test_an_embedding_without_a_loaded_model_is_refused(
    runner: ModuleType, tmp_path: Path
) -> None:
    dispatch = runner.dispatch(runner.VameRunner())
    payload: dict[str, JsonValue] = {
        "sessions": [
            {
                "name": "s000000",
                "track": str(tmp_path / "s000000.npz"),
                "latents": str(tmp_path / "s000000.latents.npz"),
            }
        ],
        "batch_size": 64,
        "chunk_frames": 50_000,
        "exchange_dir": str(tmp_path),
    }

    with pytest.raises(runner.RunnerError, match="load_model"):
        _ = dispatch["embed"](payload, EventRecorder())


def test_the_program_answers_ready_and_then_shutdown(
    line_children: LineChildren,
) -> None:
    """The program runs under this interpreter, without VAME."""
    program = line_children.spawn([sys.executable, str(_RUNNER)], env=subprocess_env())

    ready = assert_ready_then_shutdown(program)

    assert ready.tool == "vame"
    assert ready.version == NOT_INSTALLED


# --- The project config ---


def test_a_fit_config_holds_the_training_preprocessing_and_pinned_keys(
    runner: ModuleType,
) -> None:
    config = runner.fit_config(_training(), _preprocessing())

    assert {key: config[key] for key in Training.model_fields} == (
        _training().model_dump()
    )
    assert config["pose_confidence"] == 0.5
    assert config["iqr_factor"] == 3
    assert config["savgol_length"] == 7
    assert config["savgol_order"] == 3
    assert config["egocentric_data"] is False
    assert config["extra_features"] == []
    assert config["pretrained_weights"] is False
    assert config["robust"] is True
    assert config["model_name"] == "VAME"
    runner.check_config_keys(config, PROJECT_FIELDS)


def test_an_embedding_config_holds_the_architecture_and_preprocessing_only(
    runner: ModuleType,
) -> None:
    model = SimpleNamespace(training=_training(), preprocessing=_preprocessing())

    config = runner.embed_config(model)

    assert config["time_window"] == 10
    assert config["zdims"] == 4
    assert config["softplus"] is False
    assert config["savgol_length"] == 7
    for training_only in ("max_epochs", "learning_rate", "kl_start", "batch_size"):
        assert training_only not in config
    runner.check_config_keys(config, PROJECT_FIELDS)


def test_a_config_holds_no_key_that_project_creation_sets(runner: ModuleType) -> None:
    for config in (
        runner.fit_config(_training(), _preprocessing()),
        runner.embed_config(
            SimpleNamespace(training=_training(), preprocessing=_preprocessing())
        ),
    ):
        assert not set(config) & set(runner.INIT_KEYS)


def test_each_config_gets_a_list_of_its_own(runner: ModuleType) -> None:
    """Each call returns new values, which VAME may change in place."""
    first = runner.fit_config(_training(), _preprocessing())
    second = runner.fit_config(_training(), _preprocessing())

    first["extra_features"].append("speed")

    assert second["extra_features"] == []


def test_a_config_key_that_vame_does_not_know_is_refused(runner: ModuleType) -> None:
    config = {**runner.fit_config(_training(), _preprocessing()), "zdim": 3}

    with pytest.raises(runner.RunnerError, match="zdim"):
        runner.check_config_keys(config, PROJECT_FIELDS)


def test_a_config_key_that_project_creation_sets_is_refused(
    runner: ModuleType,
) -> None:
    config = {**runner.fit_config(_training(), _preprocessing()), "num_features": 9}

    with pytest.raises(runner.RunnerError, match="num_features"):
        runner.check_config_keys(config, PROJECT_FIELDS)


def test_every_preprocessing_setting_reaches_the_config_or_the_call(
    runner: ModuleType,
) -> None:
    """Each setting is a config key or an argument of VAME's ``preprocessing``."""
    calls: list[dict[str, object]] = []

    def preprocessing(**arguments: object) -> str:
        calls.append(arguments)
        return "position_processed"

    settings = _preprocessing()
    config = runner.fit_config(_training(), settings)

    written = runner.preprocess(preprocessing, config, settings)

    assert written == "position_processed"
    (arguments,) = calls
    assert arguments["config"] is config
    assert arguments["run_egocentric_alignment"] is True
    assert arguments["save_logs"] is False
    for field, value in settings.model_dump().items():
        passed = arguments[field] if field in arguments else config[field]
        assert passed == value, field


# --- Sessions ---

_KEYPOINTS: Final = ["nose", "neck", "tail_base"]


@dataclasses.dataclass
class _Dataset:
    """A dataset built in place of ``xarray.Dataset``, which records its writes."""

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]]
    coords: dict[str, object]
    attrs: dict[str, str]
    written: list[tuple[str, str]] = dataclasses.field(
        default_factory=list[tuple[str, str]]
    )

    def to_netcdf(self, path: str, *, engine: str) -> None:
        self.written.append((path, engine))


def _read_track(
    runner: ModuleType, directory: Path, track: Path
) -> tuple[Path, _Dataset]:
    """Have the runner write the track file *track* as session ``s000000``."""
    built: list[_Dataset] = []

    def new_dataset(
        *,
        data_vars: Mapping[str, tuple[tuple[str, ...], np.ndarray]],
        coords: Mapping[str, object],
        attrs: Mapping[str, str],
    ) -> _Dataset:
        built.append(_Dataset(dict(data_vars), dict(coords), dict(attrs)))
        return built[-1]

    session = Session(name="s000000", track=str(track))
    path = runner.write_session(new_dataset, session, _KEYPOINTS, directory)
    (dataset,) = built
    return path, dataset


def _write_session(
    runner: ModuleType, directory: Path, position: np.ndarray, confidence: np.ndarray
) -> tuple[Path, _Dataset]:
    """Write a track of *position* and *confidence* as session ``s000000``."""
    track = directory / "s000000.npz"
    frames = np.arange(len(position), dtype=np.int64) + 100
    np.savez(track, frames=frames, position=position, confidence=confidence)
    return _read_track(runner, directory, track)


def test_a_session_is_written_as_a_float64_movement_dataset(
    runner: ModuleType, tmp_path: Path
) -> None:
    """VAME's preprocessing computes in the dtype of the file that it reads."""
    position = np.random.default_rng(0).normal(size=(6, 3, 2)).astype(np.float32)
    confidence = np.full((6, 3), 0.75, np.float32)

    path, dataset = _write_session(runner, tmp_path, position, confidence)

    assert path == tmp_path / "s000000.nc"
    assert dataset.written == [(str(path), "netcdf4")]
    dims, values = dataset.data_vars["position"]
    assert dims == ("time", "space", "keypoints", "individuals")
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values[..., 0], position.transpose(0, 2, 1))
    dims, values = dataset.data_vars["confidence"]
    assert dims == ("time", "keypoints", "individuals")
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values[..., 0], confidence)
    np.testing.assert_array_equal(dataset.coords["time"], np.arange(6) + 100)
    assert dataset.coords["space"] == ["x", "y"]
    assert dataset.coords["keypoints"] == _KEYPOINTS
    assert dataset.coords["individuals"] == ["individual_0"]


def test_a_session_track_that_mosaic_writes_is_read(
    runner: ModuleType, tmp_path: Path
) -> None:
    rows = keypoint_rows(range(100, 106), 0, n_keypoints=len(_KEYPOINTS))
    pose = PoseConfig(pose_n=len(_KEYPOINTS), keypoint_names=_KEYPOINTS)
    settings = SessionSettings(
        pose=pose,
        centered_reference_keypoint="nose",
        orientation_reference_keypoint="tail_base",
        run_lowconf_cleaning=True,
        pose_confidence=0.5,
        max_gap_frames=0,
        time_window=4,
    )
    [session] = build_sessions(rows, settings, "embed").sessions
    written = write_session_tracks([session], tmp_path / "exchange")

    _, dataset = _read_track(runner, tmp_path, written["s000000"])

    _, position = dataset.data_vars["position"]
    np.testing.assert_array_equal(
        position[..., 0], session.track.position.transpose(0, 2, 1)
    )
    _, confidence = dataset.data_vars["confidence"]
    np.testing.assert_array_equal(confidence[..., 0], session.track.confidence)
    np.testing.assert_array_equal(dataset.coords["time"], np.arange(100, 106))


@pytest.mark.parametrize(
    ("position_shape", "confidence_shape"),
    [((6, 4, 2), (6, 4)), ((6, 3, 3), (6, 3)), ((6, 3, 2), (5, 3))],
    ids=["keypoints", "space", "frames"],
)
def test_a_track_that_does_not_match_its_keypoints_is_refused(
    runner: ModuleType,
    tmp_path: Path,
    position_shape: tuple[int, ...],
    confidence_shape: tuple[int, ...],
) -> None:
    with pytest.raises(runner.RunnerError, match="s000000"):
        _ = _write_session(
            runner, tmp_path, np.zeros(position_shape), np.ones(confidence_shape)
        )


def test_latents_are_written_under_the_name_that_the_payload_gives(
    runner: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "s000000.latents"
    latents = np.arange(6, dtype=np.float64).reshape(3, 2)

    runner.write_latents(str(path), latents)

    assert list(tmp_path.iterdir()) == [path]
    with np.load(path) as written:
        assert written["latents"].dtype == np.float32
        np.testing.assert_array_equal(written["latents"], latents)


# --- The split ---


def _check_split(
    runner: ModuleType,
    *,
    train: int = 400,
    test: int = 100,
    features: int = 9,
    steps_per_epoch: int | None = None,
) -> None:
    """Check a split for ``time_window`` 10 and ``batch_size`` 16."""
    runner.check_split(
        (features, train),
        (features, test),
        time_window=10,
        batch_size=16,
        steps_per_epoch=steps_per_epoch,
    )


def test_a_split_that_holds_a_window_and_a_batch_passes(runner: ModuleType) -> None:
    _check_split(runner, train=21, test=21, steps_per_epoch=3)
    _check_split(runner, train=21, test=21)


@pytest.mark.parametrize(("train", "test"), [(20, 100), (100, 20)])
def test_each_split_holds_more_frames_than_a_training_window(
    runner: ModuleType, train: int, test: int
) -> None:
    """VAME draws windows of ``2 * time_window`` frames from a split."""
    with pytest.raises(runner.RunnerError, match="21"):
        _check_split(runner, train=train, test=test, steps_per_epoch=3)


@pytest.mark.parametrize(("train", "test"), [(29, 100), (100, 29)])
def test_each_split_holds_at_least_as_many_frames_as_features(
    runner: ModuleType, train: int, test: int
) -> None:
    """VAME reads a split of fewer frames than features transposed."""
    _check_split(runner, train=train + 1, test=test + 1, features=30, steps_per_epoch=3)
    with pytest.raises(runner.RunnerError, match="transposed"):
        _check_split(runner, train=train, test=test, features=30, steps_per_epoch=3)


@pytest.mark.parametrize(("train", "test"), [(399, 100), (400, 99)])
def test_without_a_step_cap_each_split_holds_one_batch(
    runner: ModuleType, train: int, test: int
) -> None:
    """A training batch is ``batch_size`` windows, and a test batch a quarter."""
    with pytest.raises(runner.RunnerError, match="batch_size"):
        runner.check_split(
            (9, train), (9, test), time_window=10, batch_size=400, steps_per_epoch=None
        )
    runner.check_split(
        (9, 400), (9, 100), time_window=10, batch_size=400, steps_per_epoch=None
    )
    runner.check_split(
        (9, train), (9, test), time_window=10, batch_size=400, steps_per_epoch=3
    )


def test_a_split_whose_features_equal_the_time_window_is_refused(
    runner: ModuleType,
) -> None:
    """VAME's embedding reads such windows transposed, without an error."""
    with pytest.raises(runner.RunnerError, match="time_window"):
        _check_split(runner, features=10)


# --- Windows that span a join ---


def test_train_chunks_of_a_random_split_are_each_sessions_training_frames(
    runner: ModuleType,
) -> None:
    details = {
        "method": "random_continuous_chunks",
        "session_splits": [
            {
                "session": "s000000",
                "session_length": 100,
                "test_chunk": {"start": 30, "end": 40},
                "train_chunks": [{"start": 0, "end": 30}, {"start": 40, "end": 100}],
            },
            {
                "session": "s000001",
                "session_length": 50,
                "test_chunk": {"start": 0, "end": 5},
                "train_chunks": [{"start": 5, "end": 50}],
            },
        ],
    }

    assert runner.train_chunk_lengths(details) == [30, 60, 45]


def test_train_chunks_of_a_sequential_split_follow_its_split_index(
    runner: ModuleType,
) -> None:
    details = {
        "method": "sequential_split",
        "split_index": 15,
        "total_length": 150,
        "session_boundaries": [
            {"session": "s000000", "start": 0, "end": 10, "length": 10},
            {"session": "s000001", "start": 10, "end": 100, "length": 90},
            {"session": "s000002", "start": 100, "end": 150, "length": 50},
        ],
    }

    assert runner.train_chunk_lengths(details) == [85, 50]


def test_the_straddling_fraction_counts_starts_whose_window_spans_a_join(
    runner: ModuleType,
) -> None:
    """A window of 10 frames spans the join at 100 when it starts at 91 to 99."""
    assert runner.straddling_fraction([100, 100], 10) == pytest.approx(9 / 190)
    assert runner.straddling_fraction([200], 10) == 0.0


def test_the_straddling_fraction_counts_a_start_once(runner: ModuleType) -> None:
    """A window may span two joins, between which a chunk is shorter than it."""
    lengths = [50, 4, 50]
    window = 10
    starts = sum(lengths) - window
    joins = [50, 54]
    expected = sum(
        1
        for start in range(starts)
        if any(start < join < start + window for join in joins)
    )

    assert runner.straddling_fraction(lengths, window) == pytest.approx(
        expected / starts
    )


# --- Chunks of a long session ---


def test_chunks_cover_each_window_once(runner: ModuleType) -> None:
    assert runner.chunk_bounds(100, 40, 10) == [(0, 40), (31, 71), (62, 100)]
    assert runner.chunk_bounds(40, 40, 10) == [(0, 40)]
    assert runner.chunk_bounds(25, 40, 10) == [(0, 25)]
    assert runner.chunk_bounds(10, 40, 10) == [(0, 10)]


@pytest.mark.parametrize(("frames", "chunk_frames"), [(9, 40), (100, 9)])
def test_chunks_need_a_whole_window(
    runner: ModuleType, frames: int, chunk_frames: int
) -> None:
    with pytest.raises(runner.RunnerError, match="time_window"):
        _ = runner.chunk_bounds(frames, chunk_frames, 10)


def _encode(series: np.ndarray, time_window: int) -> np.ndarray:
    """One latent per window: the window's sum and its first value."""
    windows = np.lib.stride_tricks.sliding_window_view(series, time_window)
    return np.stack([windows.sum(axis=1), windows[:, 0]], axis=1).astype(np.float32)


@pytest.mark.parametrize(
    ("frames", "chunk_frames", "time_window"),
    [(100, 40, 10), (1000, 37, 5), (64, 64, 8), (64, 8, 8), (12, 50, 12)],
)
def test_stitched_chunks_equal_one_unchunked_embedding(
    runner: ModuleType, frames: int, chunk_frames: int, time_window: int
) -> None:
    series = np.random.default_rng(0).normal(size=frames)
    bounds = runner.chunk_bounds(frames, chunk_frames, time_window)
    chunks = [_encode(series[start:stop], time_window) for start, stop in bounds]

    stitched = runner.stitch(chunks, bounds, time_window)

    np.testing.assert_array_equal(stitched, _encode(series, time_window))


def test_a_chunk_with_the_wrong_number_of_latents_is_refused(
    runner: ModuleType,
) -> None:
    bounds = [(0, 40), (31, 71)]
    chunks = [np.zeros((31, 2), np.float32), np.zeros((30, 2), np.float32)]

    with pytest.raises(runner.RunnerError, match="31"):
        _ = runner.stitch(chunks, bounds, 10)


# --- The device ---


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _is_device(value: object) -> bool:
    return isinstance(value, _Device)


def test_every_vame_binding_of_the_device_lookup_is_replaced(
    runner: ModuleType,
) -> None:
    def get_device() -> _Device:
        return _Device("cuda")

    cpu = _Device("cpu")
    package = _module("vame", get_device=get_device)
    auxiliary = _module("vame.util.auxiliary", get_device=get_device)
    training = _module(
        "vame.model.rnn_vae", get_device=get_device, DEVICE=_Device("cuda")
    )
    renamed = _module("vame.analysis.renamed", lookup=get_device)
    similar = _module("vamex", get_device=get_device, DEVICE=_Device("cuda"))
    other = _module("numpy", get_device=get_device)
    modules: dict[str, ModuleType | None] = {
        module.__name__: module
        for module in (package, auxiliary, training, renamed, similar, other)
    }
    modules["vame.unloaded"] = None

    bound = runner.bind_device(modules, get_device, cpu, is_device=_is_device)

    assert bound == [
        "vame.analysis.renamed.lookup",
        "vame.get_device",
        "vame.model.rnn_vae.DEVICE",
        "vame.model.rnn_vae.get_device",
        "vame.util.auxiliary.get_device",
    ]
    for module in (package, auxiliary, training):
        assert module.get_device() is cpu
    assert renamed.lookup() is cpu
    assert training.DEVICE is cpu
    assert similar.get_device is get_device
    assert similar.DEVICE is not cpu
    assert other.get_device is get_device


# --- Epochs ---


def test_an_epoch_reports_its_losses_by_name(runner: ModuleType) -> None:
    losses = runner.epoch_losses(
        (0.5, 120.0, 3.0, 2.0, 100.0, 15.0, 42), (90.0, float("nan"), 1.5)
    )

    assert losses == {
        "kl_weight": 0.5,
        "train_loss": 120.0,
        "train_kmeans_loss": 3.0,
        "train_kl_loss": 2.0,
        "train_mse_loss": 100.0,
        "train_future_loss": 15.0,
        "test_mse_loss": 90.0,
        "test_loss": None,
        "test_kmeans_loss": 1.5,
    }


def test_the_epoch_hook_emits_each_epoch_and_then_restores_training(
    runner: ModuleType,
) -> None:
    def train(*_: object) -> tuple[float, ...]:
        return (1.0, 10.0, 0.0, 0.0, 9.0, 1.0, 3)

    def test(*_: object) -> tuple[float, ...]:
        return (8.0, 8.5, 0.5)

    steps = SimpleNamespace(train=train, test=test)
    recorder = EventRecorder()

    with runner.epoch_events(steps, recorder, total=4) as counter:
        for _ in range(2):
            _ = steps.train("loader", 1)
            _ = steps.test("loader", 2)

    events = recorder.events
    assert counter.epochs == 2
    assert [name for name, _ in events] == ["epoch", "epoch"]
    assert [data["epoch"] for _, data in events] == [1, 2]
    assert events[0][1]["total"] == 4
    losses = events[1][1]["losses"]
    assert isinstance(losses, dict)
    assert losses["test_loss"] == 8.5
    assert steps.train is train
    assert steps.test is test
