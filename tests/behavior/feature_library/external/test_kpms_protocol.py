"""The payloads of the keypoint-MoSeq runner's commands, as they cross the wire.

mosaic dumps a payload into a request, and the runner validates the request's
payload with the same model. Each payload must therefore survive that trip.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.external.kpms_protocol import (
    AddTrack,
    Apply,
    Command,
    Fit,
    FitConfig,
    LoadModel,
    Payload,
    SaveModel,
)
from mosaic.behavior.feature_library.external.runner_wire import Request


def _config(**overrides: object) -> FitConfig:
    fields: dict[str, object] = {
        "bodyparts": ["nose", "left", "right"],
        "anterior_bodyparts": ["nose"],
        "posterior_bodyparts": ["left", "right"],
        "latent_dim": 4,
    }
    return FitConfig.model_validate({**fields, **overrides})


PAYLOADS: list[Payload] = [
    AddTrack(key="g__s__id0", track="/exchange/track.npz"),
    Fit(config=_config()),
    Fit(
        config=_config(
            use_bodyparts=["nose", "left"],
            latent_dim=2,
            kappa_ar=1e4,
            mixed_map_iters=4,
            parallel_message_passing=False,
            checkpoint_dir="/checkpoints",
        )
    ),
    LoadModel(path="/run/kpms_model.joblib"),
    Apply(
        key="g__s",
        num_iters=10,
        track="/exchange/track.npz",
        syllables="/exchange/syllables.npz",
    ),
    SaveModel(path="/run/kpms_model.joblib"),
]


@pytest.mark.parametrize(
    "payload",
    PAYLOADS,
    ids=[
        "add-track",
        "fit",
        "fit-with-options",
        "load-model",
        "apply",
        "save-model",
    ],
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


def test_a_payload_refuses_an_unknown_field() -> None:
    """A field that one side renamed is refused."""
    with pytest.raises(ValidationError, match="coords"):
        _ = AddTrack.model_validate(
            {"key": "g__s", "track": "/exchange/track.npz", "coords": []}
        )


def test_a_fit_refuses_a_latent_dimension_beyond_its_keypoints() -> None:
    with pytest.raises(ValidationError, match="exceeds maximum"):
        _ = _config(latent_dim=5)
