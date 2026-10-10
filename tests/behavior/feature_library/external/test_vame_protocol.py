"""The payloads, results and events of the VAME runner, as they cross the wire.

mosaic dumps a payload into a request, and the runner validates the request's
payload with the same model. A result and an event make the opposite trip. Each
must therefore survive it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, get_args

import pytest
from pydantic import BaseModel, ValidationError

from mosaic.behavior.feature_library.external.runner_wire import (
    REPLIES,
    Done,
    Event,
    Request,
)
from mosaic.behavior.feature_library.external.vame_protocol import (
    VAME_VERSION,
    AnnealFunction,
    Command,
    Embed,
    EmbeddedSession,
    EmbedResult,
    EmbedSession,
    EpochEvent,
    EventData,
    EventName,
    Fit,
    FitResult,
    FittedModel,
    LoadModel,
    Payload,
    Preprocessing,
    Session,
    StageEvent,
    Training,
    TrainingMetadata,
    TrainSet,
    check_model_is_saved,
    first_saving_epoch,
    least_training_frames,
    repeated_names,
    training_window_frames,
)
from tests.helpers import VAME_TOOL, distribution_names, project_requirements

(_ENVIRONMENT,) = VAME_TOOL.environments

KEYPOINTS: Final = ["nose", "left_ear", "right_ear", "neck", "tail_base"]


def _preprocessing(**overrides: object) -> Preprocessing:
    fields: dict[str, object] = {
        "centered_reference_keypoint": "nose",
        "orientation_reference_keypoint": "tail_base",
        "run_lowconf_cleaning": True,
        "pose_confidence": 0.99,
        "run_outlier_cleaning": True,
        "iqr_factor": 4,
        "run_savgol_filtering": True,
        "savgol_length": 5,
        "savgol_order": 2,
        "run_rescaling": False,
    }
    return Preprocessing.model_validate({**fields, **overrides})


def _training(**overrides: object) -> Training:
    fields: dict[str, object] = {
        "project_random_state": 42,
        "zdims": 30,
        "time_window": 30,
        "max_epochs": 500,
        "model_convergence": 50,
        "batch_size": 256,
        "beta": 1.0,
        "learning_rate": 5e-4,
        "prediction_decoder": True,
        "prediction_steps": 15,
        "noise": False,
        "scheduler": True,
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.2,
        "softplus": False,
        "steps_per_epoch": None,
        "hidden_size_layer_1": 256,
        "dropout_encoder": 0.0,
        "hidden_size_rec": 256,
        "hidden_size_pred": 256,
        "mse_reconstruction_reduction": "sum",
        "mse_prediction_reduction": "sum",
        "kmeans_loss": 30,
        "kmeans_lambda": 0.1,
        "anneal_function": "linear",
        "kl_start": 2,
        "annealtime": 4,
    }
    return Training.model_validate({**fields, **overrides})


def _fit(**overrides: object) -> Fit:
    fields: dict[str, object] = {
        "sessions": [
            {"name": "s000000", "track": "/exchange/s000000.npz"},
            {"name": "s000001", "track": "/exchange/s000001.npz"},
        ],
        "keypoints": KEYPOINTS,
        "preprocessing": _preprocessing(),
        "trainset": TrainSet(test_fraction=0.1, split_mode="mode_2"),
        "training": _training(),
        "device": "cpu",
        "exchange_dir": "/exchange",
        "state_dict": "/exchange/fit/state_dict.pkl",
    }
    return Fit.model_validate({**fields, **overrides})


def _fitted_model() -> FittedModel:
    return FittedModel(
        project_name="mosaic",
        state_dict=b"\x80\x02weights",
        metadata=TrainingMetadata(keypoints_used=KEYPOINTS, extra_features=[]),
        num_features=7,
        keypoints=KEYPOINTS,
        preprocessing=_preprocessing(),
        training=_training(),
        seq_mean=0.25,
        seq_std=12.5,
    )


PAYLOADS: list[Payload] = [
    _fit(),
    _fit(training=_training(steps_per_epoch=20, anneal_function="sigmoid")),
    LoadModel(path="/run/vame_model.joblib", device="auto"),
    Embed(
        sessions=[
            EmbedSession(
                name="s000000",
                track="/exchange/s000000.npz",
                latents="/exchange/s000000.latents.npz",
            )
        ],
        batch_size=64,
        chunk_frames=50_000,
        exchange_dir="/exchange",
    ),
]


@pytest.mark.parametrize(
    "payload", PAYLOADS, ids=["fit", "fit-with-options", "load-model", "embed"]
)
def test_each_payload_round_trips_through_a_request(payload: Payload) -> None:
    line = Request(
        command=payload.command, payload=payload.model_dump(mode="json")
    ).model_dump_json()

    request = Request.model_validate_json(line)

    assert request.command == payload.command
    assert type(payload).model_validate(request.payload) == payload


def test_each_command_has_exactly_one_payload() -> None:
    commands = [model.command for model in Payload.__subclasses__()]

    assert sorted(commands) == sorted(get_args(Command))


RESULTS: list[BaseModel] = [
    FitResult(
        project_name="mosaic",
        metadata=TrainingMetadata(keypoints_used=KEYPOINTS, extra_features=[]),
        num_features=7,
        seq_mean=0.25,
        seq_std=12.5,
        device="cpu",
        epochs=12,
        sessions=["s000000", "s000001"],
        train_frames=5400,
        test_frames=600,
        straddling_fraction=0.02,
    ),
    EmbedResult(
        device="mps",
        sessions=[EmbeddedSession(name="s000000", windows=571)],
    ),
]


@pytest.mark.parametrize("result", RESULTS, ids=["fit", "embed"])
def test_each_result_round_trips_through_done(result: BaseModel) -> None:
    line = Done(result=result.model_dump(mode="json")).model_dump_json()

    reply = REPLIES.validate_json(line)

    assert isinstance(reply, Done)
    assert type(result).model_validate(reply.result) == result


EVENTS: list[EventData] = [
    EpochEvent(epoch=3, total=499, losses={"train_loss": 1.5, "test_loss": None}),
    StageEvent(stage="train", message="training on 2 sessions"),
]


@pytest.mark.parametrize("event", EVENTS, ids=["epoch", "stage"])
def test_each_event_round_trips_through_an_event(event: EventData) -> None:
    line = Event(name=event.name, data=event.model_dump(mode="json")).model_dump_json()

    reply = REPLIES.validate_json(line)

    assert isinstance(reply, Event)
    assert reply.name == event.name
    assert type(event).model_validate(reply.data) == event


def test_each_event_name_has_exactly_one_model() -> None:
    names = [model.name for model in EventData.__subclasses__()]

    assert sorted(names) == sorted(get_args(EventName))


def test_a_payload_refuses_an_unknown_field() -> None:
    """A field that one side renamed is refused."""
    with pytest.raises(ValidationError, match="bundle"):
        _ = LoadModel.model_validate(
            {"path": "/run/vame_model.joblib", "device": "cpu", "bundle": "/run"}
        )


def test_a_fit_refuses_a_reference_outside_its_keypoints() -> None:
    with pytest.raises(ValidationError, match="'tail' is not among the keypoints"):
        _ = _fit(preprocessing=_preprocessing(orientation_reference_keypoint="tail"))


def test_preprocessing_refuses_one_keypoint_as_both_references() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        _ = _preprocessing(orientation_reference_keypoint="nose")


def test_a_fit_refuses_two_sessions_of_one_name() -> None:
    session = {"name": "s000000", "track": "/exchange/s000000.npz"}

    with pytest.raises(ValidationError, match="s000000"):
        _ = _fit(sessions=[session, session])


def test_repeated_names_are_each_named_once_in_order() -> None:
    assert repeated_names(["tail", "nose", "tail", "ear", "nose", "tail"]) == [
        "nose",
        "tail",
    ]
    assert repeated_names(["nose", "tail"]) == []


def test_a_training_window_is_two_time_windows_and_needs_one_more_frame() -> None:
    assert training_window_frames(30) == 60
    assert least_training_frames(30) == 61


@pytest.mark.parametrize("name", ["s/0", "s-0", "", "s 0"])
def test_a_session_name_is_letters_digits_and_underscores(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        _ = Session(name=name, track="/exchange/s.npz")


def test_a_train_set_fraction_lies_strictly_between_zero_and_one() -> None:
    for fraction in (0.0, 1.0):
        with pytest.raises(ValidationError, match="test_fraction"):
            _ = TrainSet(test_fraction=fraction, split_mode="mode_2")


@pytest.mark.parametrize("batch_size", [1, 3])
def test_training_refuses_a_batch_whose_quarter_is_empty(batch_size: int) -> None:
    """VAME's test batches contain ``batch_size // 4`` windows."""
    with pytest.raises(ValidationError, match="batch_size"):
        _ = _training(batch_size=batch_size)

    assert _training(batch_size=4).batch_size == 4


@pytest.mark.parametrize(
    ("build", "field", "least"),
    [
        (_training, "steps_per_epoch", 1),
        (_training, "prediction_steps", 1),
        (_preprocessing, "savgol_order", 0),
    ],
    ids=["steps_per_epoch", "prediction_steps", "savgol_order"],
)
def test_a_count_below_its_least_value_is_refused(
    build: Callable[..., BaseModel], field: str, least: int
) -> None:
    """VAME reads 0 batches per epoch as unset, and slices an empty target at 0 steps."""
    assert getattr(build(**{field: least}), field) == least
    with pytest.raises(ValidationError, match=field):
        _ = build(**{field: least - 1})


def test_a_fitted_model_ignores_the_entries_that_mosaic_keeps_for_itself() -> None:
    """A bundle contains further entries beside the ones that embedding reads."""
    model = _fitted_model()
    bundle = {**model.model_dump(), "vame_version": VAME_VERSION, "device": "cpu"}

    assert FittedModel.model_validate(bundle) == model


def test_the_version_is_the_pin_of_the_environment() -> None:
    manifest = _ENVIRONMENT.source_directory / "pyproject.toml"
    dependencies = project_requirements(manifest).dependencies

    pins = [
        entry
        for entry in dependencies
        if distribution_names([entry]) == {VAME_TOOL.distribution}
    ]

    assert pins == [f"vame-py=={VAME_VERSION}"]


# --- The epoch at which VAME first saves a model ---


@pytest.mark.parametrize("kl_start", range(10))
def test_the_linear_saving_epoch_matches_its_closed_form(kl_start: int) -> None:
    """``kl_start + floor(0.99 * annealtime) + 1``, in integer arithmetic."""
    for annealtime in range(1, 400):
        expected = kl_start + 99 * annealtime // 100 + 1

        assert first_saving_epoch(kl_start, annealtime, "linear") == expected


@pytest.mark.parametrize("kl_start", range(10))
def test_the_sigmoid_saving_epoch_matches_its_closed_form(kl_start: int) -> None:
    for annealtime in range(1, 400):
        expected = max(kl_start + 1, annealtime + 6)

        assert first_saving_epoch(kl_start, annealtime, "sigmoid") == expected


def test_the_linear_saving_epoch_needs_a_weight_above_the_threshold() -> None:
    """At ``annealtime`` 100, epoch 99 has a weight of 0.99 and does not save."""
    assert first_saving_epoch(0, 100, "linear") == 100
    assert first_saving_epoch(2, 4, "linear") == 6
    assert first_saving_epoch(2, 4, "sigmoid") == 10


@pytest.mark.parametrize(
    ("kl_start", "annealtime"), [(-1, 4), (2, 0)], ids=["kl-start", "annealtime"]
)
def test_the_saving_epoch_refuses_a_schedule_without_one(
    kl_start: int, annealtime: int
) -> None:
    with pytest.raises(ValueError, match="kl_start|annealtime"):
        _ = first_saving_epoch(kl_start, annealtime, "linear")


ANNEAL_FUNCTIONS: Final[tuple[AnnealFunction, ...]] = get_args(AnnealFunction)


def _check(
    *, max_epochs: int, model_convergence: int, anneal_function: AnnealFunction
) -> None:
    """Check training on VAME's default schedule, ``kl_start`` 2, ``annealtime`` 4."""
    check_model_is_saved(
        max_epochs=max_epochs,
        model_convergence=model_convergence,
        kl_start=2,
        annealtime=4,
        anneal_function=anneal_function,
    )


@pytest.mark.parametrize("anneal_function", ANNEAL_FUNCTIONS)
def test_training_must_reach_the_saving_epoch(anneal_function: AnnealFunction) -> None:
    """VAME runs epochs 1 to ``max_epochs - 1``."""
    saving = first_saving_epoch(2, 4, anneal_function)

    _check(max_epochs=saving + 1, model_convergence=50, anneal_function=anneal_function)
    with pytest.raises(ValueError, match="max_epochs"):
        _check(max_epochs=saving, model_convergence=50, anneal_function=anneal_function)


@pytest.mark.parametrize("anneal_function", ANNEAL_FUNCTIONS)
def test_convergence_must_not_stop_training_before_the_saving_epoch(
    anneal_function: AnnealFunction,
) -> None:
    """Each epoch without a save counts towards ``model_convergence``."""
    saving = first_saving_epoch(2, 4, anneal_function)

    _check(
        max_epochs=500, model_convergence=saving - 1, anneal_function=anneal_function
    )
    with pytest.raises(ValueError, match="model_convergence"):
        _check(
            max_epochs=500,
            model_convergence=saving - 2,
            anneal_function=anneal_function,
        )


def test_training_refuses_settings_that_save_no_model() -> None:
    with pytest.raises(ValidationError, match="max_epochs"):
        _ = _training(max_epochs=6)
