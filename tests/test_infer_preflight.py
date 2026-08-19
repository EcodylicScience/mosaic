"""What `infer-pose` and `infer-points` refuse, and the names they exchange.

Every refusal here is decided from a :class:`ProbeResponse` alone, with no
environment anywhere, which is the whole point of the runner reporting rather
than refusing: the messages name mosaic commands and mosaic's own installation
documentation, and they are testable with no Ultralytics installed at all.
"""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.tracking.common.ultralytics_env import (
    ModelLoadError,
    UnsupportedModelError,
)
from mosaic.tracking.external.runner.ultralytics_protocol import ProbeResponse
from mosaic.tracking.pose_training.ultralytics_infer import (
    INFER_REQUEST_NAME,
    INFER_RESPONSE_NAME,
    require_points_model,
    require_pose_model,
)

pytestmark = pytest.mark.tracker


def _probe(**overrides: object) -> ProbeResponse:
    """A healthy upstream probe, with the fields a case is about replaced."""
    fields: dict[str, object] = {
        "has_ultralytics": True,
        "has_lap": True,
        "has_locate": False,
        "ultralytics_version": "8.4.121",
        "tracker_names": ["botsort"],
        "model_task": "pose",
        "n_keypoints": 17,
        "model_load_error": "",
        "installed_tracker_table": {},
    }
    fields.update(overrides)
    return ProbeResponse.model_validate(fields)


def test_point_inference_refuses_an_upstream_environment() -> None:
    """The trap the location ladder cannot see.

    POLO and upstream ship the same ``yolo`` console script under the same
    distribution name, so a ``$PATH`` lookup resolves to whichever is installed
    and cannot report which it found. Without this refusal an unset
    ``MOSAIC_POLO_BIN`` runs point detection against upstream, which has no
    ``locate`` task -- and the user is told only that their weights would not
    load, about an environment they never chose.
    """
    with pytest.raises(Exception) as raised:
        require_points_model(_probe(has_locate=False, model_task="locate"), "m.pt")
    message = str(raised.value)
    assert "not the POLO fork" in message
    assert "MOSAIC_POLO_BIN" in message


def test_point_inference_checks_the_fork_before_the_weights() -> None:
    """Order matters: an upstream build fails to load POLO weights *because* it
    is upstream, so naming the environment is the actionable half."""
    probe = _probe(
        has_locate=False,
        model_task="",
        model_load_error="AttributeError: Can't get attribute 'LocalizationModel'",
    )
    with pytest.raises(Exception) as raised:
        require_points_model(probe, "m.pt")
    assert "not the POLO fork" in str(raised.value)


def test_a_checkpoint_the_environment_cannot_load_is_refused_by_name() -> None:
    """A load failure reaches the user as a refusal, not as a traceback.

    And when the running build is not the fork, the refusal says so and routes
    the weights to the op that can run them -- which is the whole reason the
    probe reports the failure instead of dying of it.
    """
    probe = _probe(
        model_task="",
        model_load_error="AttributeError: Can't get attribute 'LocalizationModel'",
    )
    with pytest.raises(ModelLoadError) as raised:
        require_pose_model(probe, "polo-weights.pt")
    message = str(raised.value)
    assert "polo-weights.pt" in message
    assert "infer-points" in message
    assert "8.4.121" in message


def test_pose_inference_refuses_a_point_model_and_says_where_it_goes() -> None:
    with pytest.raises(UnsupportedModelError) as raised:
        require_pose_model(_probe(model_task="locate"), "m.pt")
    assert "infer-points" in str(raised.value)


def test_point_inference_refuses_a_pose_model() -> None:
    with pytest.raises(UnsupportedModelError) as raised:
        require_points_model(_probe(has_locate=True, model_task="pose"), "m.pt")
    assert "infer-pose" in str(raised.value)


def test_a_healthy_probe_is_accepted() -> None:
    """The guard has to let a good run through, or the refusals prove nothing."""
    require_pose_model(_probe(model_task="pose"), "m.pt")
    require_points_model(_probe(has_locate=True, model_task="locate"), "m.pt")


@pytest.mark.parametrize("kind", ["infer-pose", "infer-points"])
def test_a_re_run_clears_the_files_the_exchange_left(kind: str) -> None:
    """The two spellings of the request and response names agree.

    ``TRACKING_ROOTS`` is in ``core``, which cannot import the module that names
    these files, so the strings exist in two places and only a test holds them
    together. They are byproducts of one attempt rather than results of one, so a
    re-run of the phase must delete them: a stale request beside fresh output is
    exactly what a reuse gate would adopt.
    """
    globs = TRACKING_ROOTS[kind].phase_outputs[0].clear_globs
    assert INFER_REQUEST_NAME in globs
    assert INFER_RESPONSE_NAME in globs
    # And they are byproducts, so they are not evidence of real output.
    assert INFER_REQUEST_NAME not in TRACKING_ROOTS[kind].outputs
    assert INFER_RESPONSE_NAME not in TRACKING_ROOTS[kind].outputs


def test_the_localizer_exchanges_nothing_and_clears_nothing_extra() -> None:
    """It runs in mosaic's own process, so there is no request to go stale."""
    globs = TRACKING_ROOTS["infer-localizer"].phase_outputs[0].clear_globs
    assert INFER_REQUEST_NAME not in globs
    assert INFER_RESPONSE_NAME not in globs
