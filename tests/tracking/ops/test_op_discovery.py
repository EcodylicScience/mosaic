"""What the tracking ops present before they run.

Their kinds and categories, their parameter schemas, and the training parameters
refused at submit time rather than on the GPU node.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline.ops import OPS, describe_op, list_ops, run_op
from mosaic.tracking.ops.train import PointTrainParams, PoseTrainParams
from tests.helpers import CONSUMER_OWNED, narrowed_params


def test_registry_has_builtin_ops():
    kinds = set(OPS)
    assert kinds >= {
        "extract-frames",
        "train-pose",
        "train-points",
        "train-localizer",
        "infer-pose",
        "infer-points",
        "infer-localizer",
    }
    assert "trex" in kinds
    assert "sleap" in kinds
    for op in list_ops(domain="tracking"):
        assert op["kind"] in kinds and op["category"] in {
            "extract",
            "train",
            "infer",
            "convert",
        }


def test_describe_returns_params_schema():
    d = describe_op("train-pose")
    schema = d["params_schema"]
    assert "properties" in schema
    assert {"data", "epochs", "model", "train_overrides"} <= set(schema["properties"])


def test_train_points_describes_the_polo_knobs():
    """What a front-end form renders for POLO training.

    ``train_overrides`` is what carries the hyperparameters mosaic has no field
    for -- the learning-rate schedule the deployed detector is trained on -- so a
    caller can reproduce it through the op rather than around it.
    """
    d = describe_op("train-points")
    assert d["category"] == "train"
    assert {"data", "epochs", "loc", "loc_loss", "dor", "train_overrides"} <= set(
        d["params_schema"]["properties"]
    )


def test_train_overrides_may_not_shadow_what_the_op_supplies():
    """Refuse at submit time what would otherwise fail on the GPU node.

    A key naming a trainer parameter arrives as a duplicate keyword and raises
    ``TypeError`` from Python itself, after the job has been accepted, queued and
    scheduled. ``data`` and ``task`` are worse: the trainer only builds those
    internally, so an override replaces them silently and the run trains on data
    its identifier does not describe.
    """
    assert PointTrainParams.model_validate(
        {"data": "d.yaml", "train_overrides": {"lr0": 0.0044, "weight_decay": 0.000139}}
    ).train_overrides == {"lr0": 0.0044, "weight_decay": 0.000139}

    for shadowed in ("epochs", "imgsz", "patience", "loc", "backend"):
        with pytest.raises(ValidationError, match=shadowed):
            _ = PointTrainParams.model_validate(
                {"data": "d.yaml", "train_overrides": {shadowed: 1}}
            )
    for supplied in ("data", "task", "project", "name", "callback", "cancel_check"):
        with pytest.raises(ValidationError, match=supplied):
            _ = PointTrainParams.model_validate(
                {"data": "d.yaml", "train_overrides": {supplied: "x"}}
            )


@pytest.mark.parametrize(
    ("kind", "model"),
    [("train-pose", PoseTrainParams), ("train-points", PointTrainParams)],
)
def test_a_narrowed_trainer_reserves_the_fields_it_lacks(
    kind: str, model: type[PoseTrainParams]
):
    """A control plane that owns some trainer fields still gets mosaic's refusals.

    The model it exposes lacks ``data``, ``device``, ``workers``, ``resume`` and
    ``base_model``. An override naming one must be refused at submit time, as the
    full model refuses it, rather than fail on the worker.
    """
    narrowed = narrowed_params(model, CONSUMER_OWNED[kind])

    for owned in ("data", "device", "workers", "resume", "base_model"):
        with pytest.raises(ValidationError, match=owned):
            _ = narrowed.model_validate({"train_overrides": {owned: "x"}})
    assert narrowed.model_validate(
        {"train_overrides": {"lr0": 0.0044}}
    ).train_overrides == {"lr0": 0.0044}


def test_a_narrowed_trainer_does_not_point_at_a_field_it_lacks():
    """The refusal tells the caller where to set a name its model has no field for.

    The full model ends "Use the typed field instead."; on a model without the
    field that advice cannot be followed.
    """
    full = "train_overrides may not set epochs: the op supplies these itself."
    with pytest.raises(ValidationError) as full_refusal:
        _ = PoseTrainParams.model_validate(
            {"data": "d.yaml", "train_overrides": {"epochs": 1}}
        )
    assert f"{full} Use the typed field instead." in str(full_refusal.value)

    narrowed = narrowed_params(PoseTrainParams, {"epochs"})
    with pytest.raises(ValidationError) as absent:
        _ = narrowed.model_validate(
            {"data": "d.yaml", "train_overrides": {"epochs": 1}}
        )
    message = str(absent.value)
    assert f"{full} epochs is set by whoever supplies the job's parameters." in message
    assert "typed field" not in message

    with pytest.raises(ValidationError) as mixed:
        _ = narrowed.model_validate(
            {"data": "d.yaml", "train_overrides": {"epochs": 1, "imgsz": 1}}
        )
    assert (
        "Use the typed field instead for imgsz. epochs is set by whoever supplies "
        "the job's parameters." in str(mixed.value)
    )


def test_a_narrowed_point_trainer_still_reserves_what_only_it_declares():
    """``loc`` and ``backend`` belong to ``PointTrainParams``, not its base."""
    narrowed = narrowed_params(PointTrainParams, CONSUMER_OWNED["train-points"])

    for reserved in ("loc", "loc_loss", "dor", "backend"):
        with pytest.raises(ValidationError, match=reserved):
            _ = narrowed.model_validate({"train_overrides": {reserved: 1}})


def test_a_trainer_narrowed_without_train_overrides_gives_no_verdict_on_it():
    """With no ``train_overrides`` field the rule has nothing to read."""
    narrowed = narrowed_params(PoseTrainParams, {"train_overrides"})

    assert narrowed.model_validate({"data": "d.yaml"}).model_fields_set == {"data"}


def test_augmentation_accepts_the_dict_forms():
    """``resolve_augmentation`` has always taken a dict; the op used to narrow it.

    Both forms it accepts -- a preset plus overrides, and a bare override set --
    were unreachable through the op and so through the CLI and the API.
    """
    preset_plus = PoseTrainParams.model_validate(
        {"data": "d.yaml", "augmentation": {"preset": "medium", "flipud": 0.5}}
    )
    assert preset_plus.augmentation == {"preset": "medium", "flipud": 0.5}
    assert (
        PoseTrainParams.model_validate(
            {"data": "d.yaml", "augmentation": "medium"}
        ).augmentation
        == "medium"
    )


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        run_op(object(), "nope", {})
