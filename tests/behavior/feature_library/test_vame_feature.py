"""VameFeature's params and its declaration.

The params refuse settings under which VAME ends training without a model, caps a
value without recording it, or reads a partial window. The run identity leaves out
the fields that place a run or set its throughput, and contains the rest.
Constructing the feature does not start a process.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.external.vame_protocol import AnnealFunction
from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.behavior.feature_library.vame_feature import (
    VAME_MODEL_FILE,
    VameFeature,
    VameModelArtifact,
)

_REFERENCES: Final[dict[str, object]] = {
    "centered_reference_keypoint": 0,
    "orientation_reference_keypoint": 3,
}
"""The two reference keypoints, which the params require."""


def _params(**overrides: object) -> VameFeature.Params:
    return VameFeature.Params.model_validate({**_REFERENCES, **overrides})


def _make_feature(**overrides: object) -> VameFeature:
    return VameFeature(
        inputs=VameFeature.Inputs(("tracks",)), params={**_REFERENCES, **overrides}
    )


# --- the declaration --------------------------------------------------------


def test_the_feature_is_registered_with_its_declaration() -> None:
    assert FEATURES["VameFeature"] is VameFeature
    declared = {
        "name": VameFeature.name,
        "version": VameFeature.version,
        "category": VameFeature.category,
        "parallelizable": VameFeature.parallelizable,
        "scope_dependent": VameFeature.scope_dependent,
        "accepts_overlap": VameFeature.accepts_overlap,
        "consumed_roots": VameFeature.consumed_roots,
        "emits": VameFeature.emits,
        "resource_class": VameFeature.resource_class,
    }

    assert declared == {
        "name": "vame",
        "version": "0.1",
        "category": "global",
        "parallelizable": False,
        "scope_dependent": True,
        "accepts_overlap": False,
        "consumed_roots": (),
        "emits": "individual",
        "resource_class": "gpu",
    }


def test_the_feature_computes_in_the_planar_spaces() -> None:
    """A feature that does not declare track spaces is refused a world-frame table."""
    assert not hasattr(VameFeature, "track_spaces")


def test_the_model_artifact_names_the_bundle_that_a_fit_writes() -> None:
    artifact = VameModelArtifact()

    assert artifact.feature == "vame"
    assert artifact.pattern == VAME_MODEL_FILE == "vame_model.joblib"
    assert artifact.load.kind == "joblib"


# --- params -----------------------------------------------------------------


def test_the_reference_keypoints_are_required() -> None:
    with pytest.raises(ValidationError) as refused:
        _ = VameFeature.Params.model_validate({})

    message = str(refused.value)
    assert "centered_reference_keypoint" in message
    assert "orientation_reference_keypoint" in message


def test_a_reference_keypoint_is_an_index_or_a_name() -> None:
    by_index = _params()
    by_name = _params(
        centered_reference_keypoint="nose", orientation_reference_keypoint="tail"
    )

    assert by_index.centered_reference_keypoint == 0
    assert by_index.orientation_reference_keypoint == 3
    assert by_name.centered_reference_keypoint == "nose"
    assert by_name.orientation_reference_keypoint == "tail"


@pytest.mark.parametrize("reference", [2, "nose"], ids=["index", "name"])
def test_one_keypoint_cannot_be_both_references(reference: int | str) -> None:
    with pytest.raises(ValidationError, match="distinct"):
        _ = _params(
            centered_reference_keypoint=reference,
            orientation_reference_keypoint=reference,
        )


@pytest.mark.parametrize(
    ("anneal_function", "kl_start", "annealtime", "saving_epoch"),
    [
        ("linear", 2, 4, 6),
        ("linear", 0, 100, 100),
        ("sigmoid", 2, 4, 10),
        ("sigmoid", 20, 4, 21),
    ],
    ids=["linear-default", "linear-long", "sigmoid-default", "sigmoid-late-start"],
)
def test_training_runs_until_the_first_epoch_that_saves_a_model(
    anneal_function: AnnealFunction,
    kl_start: int,
    annealtime: int,
    saving_epoch: int,
) -> None:
    """VAME runs epochs 1 to ``max_epochs - 1`` and saves from *saving_epoch* on.

    It also stops after ``model_convergence + 1`` epochs in a row without a save,
    and every epoch before *saving_epoch* is one.
    """
    schedule: dict[str, object] = {
        "anneal_function": anneal_function,
        "kl_start": kl_start,
        "annealtime": annealtime,
    }

    _ = _params(**schedule, max_epochs=saving_epoch + 1, model_convergence=500)
    with pytest.raises(ValidationError, match="max_epochs"):
        _ = _params(**schedule, max_epochs=saving_epoch, model_convergence=500)
    _ = _params(**schedule, max_epochs=1000, model_convergence=saving_epoch - 1)
    with pytest.raises(ValidationError, match="model_convergence"):
        _ = _params(**schedule, max_epochs=1000, model_convergence=saving_epoch - 2)


@pytest.mark.parametrize(
    ("field", "value"), [("kl_start", -1), ("annealtime", 0)], ids=lambda v: str(v)
)
def test_the_kl_schedule_starts_at_or_after_epoch_0_and_rises_over_an_epoch(
    field: str, value: int
) -> None:
    with pytest.raises(ValidationError, match=field):
        _ = _params(**{field: value})


def test_a_batch_holds_at_least_four_windows() -> None:
    """A test batch contains ``batch_size // 4`` windows."""
    assert _params(batch_size=4).batch_size == 4
    with pytest.raises(ValidationError, match="batch_size"):
        _ = _params(batch_size=3)


@pytest.mark.parametrize(
    ("field", "least"),
    [("steps_per_epoch", 1), ("prediction_steps", 1), ("savgol_order", 0)],
    ids=["steps_per_epoch", "prediction_steps", "savgol_order"],
)
def test_a_count_below_its_least_value_is_refused(field: str, least: int) -> None:
    """VAME reads 0 batches per epoch as unset, and slices an empty target at 0 steps."""
    assert getattr(_params(**{field: least}), field) == least
    with pytest.raises(ValidationError, match=field):
        _ = _params(**{field: least - 1})


def test_the_predicted_frames_fit_in_one_window() -> None:
    """VAME predicts at most ``time_window`` frames, and caps a larger value."""
    assert _params(time_window=20, prediction_steps=20).prediction_steps == 20
    with pytest.raises(ValidationError, match="prediction_steps"):
        _ = _params(time_window=20, prediction_steps=21)


@pytest.mark.parametrize(
    ("changes", "accepted"),
    [
        ({"savgol_length": 7, "savgol_order": 2}, True),
        ({"savgol_length": 6, "savgol_order": 2}, False),
        ({"savgol_length": 3, "savgol_order": 2}, True),
        ({"savgol_length": 3, "savgol_order": 3}, False),
        ({"savgol_length": 29, "time_window": 29}, True),
        ({"savgol_length": 31, "time_window": 29}, False),
    ],
    ids=[
        "odd",
        "even",
        "longer-than-order",
        "not-longer-than-order",
        "as-long-as-the-window",
        "longer-than-the-window",
    ],
)
def test_the_savgol_window_is_odd_longer_than_its_order_and_fits_a_window(
    changes: dict[str, object], accepted: bool
) -> None:
    if accepted:
        _ = _params(**changes)
        return
    with pytest.raises(ValidationError, match="savgol_length"):
        _ = _params(**changes)


def test_a_gap_is_at_least_zero_frames() -> None:
    with pytest.raises(ValidationError, match="max_gap_frames"):
        _ = _params(max_gap_frames=-1)


def test_only_low_confidence_cleaning_bridges_a_gap() -> None:
    """Without it, every frame with a missing point ends its session."""
    assert _params(run_lowconf_cleaning=True, max_gap_frames=5).max_gap_frames == 5
    assert _params(run_lowconf_cleaning=False, max_gap_frames=0).max_gap_frames == 0
    with pytest.raises(ValidationError, match="max_gap_frames"):
        _ = _params(run_lowconf_cleaning=False, max_gap_frames=5)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_the_test_set_is_a_fraction_strictly_between_0_and_1(fraction: float) -> None:
    with pytest.raises(ValidationError, match="test_fraction"):
        _ = _params(test_fraction=fraction)


def test_an_embedding_chunk_holds_a_window() -> None:
    assert _params(time_window=30, embed_chunk_frames=30).embed_chunk_frames == 30
    with pytest.raises(ValidationError, match="embed_chunk_frames"):
        _ = _params(time_window=30, embed_chunk_frames=29)


def test_the_device_is_one_that_vame_selects() -> None:
    assert _params(device="mps").device == "mps"
    with pytest.raises(ValidationError, match="device"):
        _ = _params(device="tpu")


# --- identity ---------------------------------------------------------------


def test_placement_and_throughput_are_left_out_of_the_run_identity() -> None:
    params = _params()

    excluded = set(params.model_dump()) - set(params.identity_dump())

    assert excluded == {
        "vame_python",
        "device",
        "embed_batch_size",
        "embed_chunk_frames",
        "scratch_dir",
    }


def test_settings_that_vame_does_not_read_or_that_are_pinned_are_not_params() -> None:
    """A field without effect would name two runs that compute the same model."""
    unread = {
        "savgol_filter",
        "transition_function",
        "beta_norm",
        "scheduler_threshold",
        "n_layers",
        "hidden_size_layer_2",
        "dropout_rec",
        "dropout_pred",
    }
    pinned = {
        "egocentric_data",
        "extra_features",
        "pretrained_weights",
        "pretrained_model",
        "all_data",
        "model_name",
        "model_snapshot",
        "robust",
    }

    assert not (unread | pinned) & set(VameFeature.Params.model_fields)


# --- construction ----------------------------------------------------------


def test_construction_starts_no_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constructing the feature is not use of VAME, and ``mosaic reconcile`` does it.

    The interpreter named here does not exist. Locating it raises.
    """
    started: list[object] = []

    def refuse_to_start(*args: object, **kwargs: object) -> None:
        started.append(args)
        msg = "construction started a process"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "Popen", refuse_to_start)

    feature = _make_feature(vame_python=str(tmp_path / "missing" / "python"))

    assert feature.params.vame_python == str(tmp_path / "missing" / "python")
    assert started == []
