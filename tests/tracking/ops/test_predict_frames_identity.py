"""A predict-frames run is named by its params and the kind and identity of its models.

One run id is its own identity and is read from no file. A SLEAP pair is named by
the content digest of both trainings. Every refusal reads the run ids and the
params alone. It is raised before a model is read, also for a pair whose
trainings are not on disk yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

import mosaic.tracking.model_refs as model_refs
import mosaic.tracking.ops.predict_frames as predict_frames
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred
from mosaic.tracking.common.mint import planned_model_id
from mosaic.tracking.frame_predictions.errors import PredictFramesInvalid
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ModelReferenceRefusedError
from mosaic.tracking.ops.predict_frames import (
    predict_frames_model,
    predict_frames_run_id,
    refuse_device_for_kind,
    refuse_foreign_framework_fields,
    refuse_mixed_kinds,
    refuse_pair_unless_sleap,
)
from tests.helpers import make_dataset, register_sleap_model

KIND: Final = "predict-frames"
VERSION: Final = "0.1"
FRAMES_RUN_ID: Final = "uniform-0123456789"
POSE_RUN_ID: Final = "train-pose.0.2-abcdef0123"
CENTROID_RUN_ID: Final = "train-sleap.0.3-0123456789"
INSTANCE_RUN_ID: Final = "train-sleap.0.3-abcdef0123"

ACCEPTED_KINDS: Final = ("train-pose", "train-points", "train-sleap", "train-litpose")

FRAMEWORK_OF: Final[dict[str, str]] = {
    "train-pose": "yolo-pose",
    "train-points": "POLO",
    "train-sleap": "SLEAP",
    "train-litpose": "Lightning Pose",
}

READ_BY: Final[dict[str, frozenset[str]]] = {
    "conf_threshold": frozenset({"train-pose", "train-points"}),
    "imgsz": frozenset({"train-pose", "train-points"}),
    "iou": frozenset({"train-pose"}),
    "max_det": frozenset({"train-pose", "train-points"}),
    "agnostic_nms": frozenset({"train-pose", "train-points"}),
    "precision": frozenset({"train-pose", "train-points"}),
    "dor": frozenset({"train-points"}),
    "peak_threshold": frozenset({"train-sleap"}),
    "max_instances": frozenset({"train-sleap"}),
}
"""The training kinds whose models read each framework field."""

COMMON: Final = frozenset({"model", "frames_run_id", "device", "batch_size"})
"""The fields that the models of every framework read."""

NON_DEFAULT: Final[dict[str, object]] = {
    "conf_threshold": 0.1,
    "imgsz": 320,
    "iou": 0.5,
    "max_det": 10,
    "agnostic_nms": True,
    "precision": "fp16",
    "dor": 2.0,
    "peak_threshold": 0.5,
    "max_instances": 2,
}

FOREIGN: Final = [
    (field, kind)
    for field, readers in READ_BY.items()
    for kind in ACCEPTED_KINDS
    if kind not in readers
]
OWN: Final = [
    (field, kind)
    for field, readers in READ_BY.items()
    for kind in ACCEPTED_KINDS
    if kind in readers
]


def _run_of(kind: str, digest: str = "abcdef0123") -> str:
    return f"{kind}.0.1-{digest}"


def _params(**overrides: object) -> PredictFramesParams:
    return PredictFramesParams.model_validate(
        {"model": [POSE_RUN_ID], "frames_run_id": FRAMES_RUN_ID, **overrides}
    )


def _unreachable(*_args: object, **_kwargs: object) -> str:
    raise AssertionError("a model was read")


@pytest.fixture
def no_model_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that reaches the model identity, which may read the models."""
    monkeypatch.setattr(predict_frames, "planned_model_id", _unreachable)


@pytest.fixture
def no_model_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that resolves a model, which reads its index and its files."""
    monkeypatch.setattr(model_refs, "resolve_model", _unreachable)
    monkeypatch.setattr(model_refs, "resolve_model_set", _unreachable)


# --- References ---------------------------------------------------------------


@pytest.mark.usefixtures("no_model_identity")
@pytest.mark.parametrize("before", [[], [CENTROID_RUN_ID]], ids=["alone", "second"])
def test_a_weights_path_is_refused_before_any_model_is_read(
    tmp_path: Path, before: list[str]
) -> None:
    """Without the refusal, the identity would be the digest of the file."""
    weights = tmp_path / "best.pt"
    _ = weights.write_bytes(b"weights")
    ds = make_dataset(tmp_path / "ds")

    with pytest.raises(ModelReferenceRefusedError) as caught:
        _ = predict_frames_model(ds, _params(model=[*before, str(weights)]))

    assert caught.value.reason == "not_a_run_id"
    assert caught.value.error_kind == "invalid"
    assert caught.value.op_kind == KIND
    assert caught.value.reference == str(weights)
    assert caught.value.accepted_kinds == ACCEPTED_KINDS
    assert str(caught.value).startswith(f"{str(weights)!r} is not a run id.")


@pytest.mark.usefixtures("no_model_identity")
def test_a_run_of_a_kind_predict_frames_does_not_run_is_refused(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path)
    localizer = _run_of("train-localizer")

    with pytest.raises(ModelReferenceRefusedError) as caught:
        _ = predict_frames_model(ds, _params(model=[localizer]))

    assert caught.value.reason == "kind_not_accepted"
    assert caught.value.reference == localizer


@pytest.mark.usefixtures("no_model_identity")
def test_models_of_mixed_kinds_are_refused_naming_each(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = predict_frames_model(ds, _params(model=[POSE_RUN_ID, CENTROID_RUN_ID]))

    assert caught.value.error_kind == "invalid"
    assert str(caught.value) == (
        f"predict-frames takes models of one training kind. {POSE_RUN_ID!r} is "
        f"a train-pose run and {CENTROID_RUN_ID!r} is a train-sleap run."
    )


def test_models_of_one_kind_pass_the_mixed_kinds_check() -> None:
    refuse_mixed_kinds([CENTROID_RUN_ID, INSTANCE_RUN_ID])
    refuse_mixed_kinds([POSE_RUN_ID])


def test_the_mixed_kinds_check_names_every_reference() -> None:
    points = _run_of("train-points")

    with pytest.raises(PredictFramesInvalid) as caught:
        refuse_mixed_kinds([POSE_RUN_ID, CENTROID_RUN_ID, points])

    assert str(caught.value).endswith(
        f"{POSE_RUN_ID!r} is a train-pose run, {CENTROID_RUN_ID!r} is a "
        f"train-sleap run, and {points!r} is a train-points run."
    )


@pytest.mark.usefixtures("no_model_identity")
@pytest.mark.parametrize("kind", ["train-pose", "train-points", "train-litpose"])
def test_two_models_other_than_a_sleap_pair_are_refused(
    tmp_path: Path, kind: str
) -> None:
    """Only a SLEAP top-down model is two trainings, and the identity of any
    other pair would digest its first model alone."""
    ds = make_dataset(tmp_path)
    first, second = _run_of(kind, "0123456789"), _run_of(kind, "abcdef0123")

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = predict_frames_model(ds, _params(model=[first, second]))

    assert caught.value.error_kind == "invalid"
    assert str(caught.value) == (
        "predict-frames takes two models only as a SLEAP top-down pair of "
        f"train-sleap runs, centroid first. {first!r} is a {kind} run and "
        f"{second!r} is a {kind} run."
    )


def test_a_sleap_pair_and_one_model_pass_the_pair_check() -> None:
    refuse_pair_unless_sleap([CENTROID_RUN_ID, INSTANCE_RUN_ID])
    refuse_pair_unless_sleap([POSE_RUN_ID])
    refuse_pair_unless_sleap([CENTROID_RUN_ID])


# --- Framework fields ---------------------------------------------------------


def test_every_field_is_common_or_listed_with_the_kinds_that_read_it() -> None:
    """A new field must be classified, so the foreign-field cases cover it."""
    assert not COMMON & set(READ_BY)
    assert set(PredictFramesParams.model_fields) == COMMON | set(READ_BY)


@pytest.mark.usefixtures("no_model_identity")
@pytest.mark.parametrize(
    ("field", "kind"), FOREIGN, ids=[f"{f}-{k}" for f, k in FOREIGN]
)
def test_a_field_set_for_another_framework_is_refused(
    tmp_path: Path, field: str, kind: str
) -> None:
    ds = make_dataset(tmp_path)
    params = _params(model=[_run_of(kind)], **{field: NON_DEFAULT[field]})

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = predict_frames_model(ds, params)

    message = str(caught.value)
    assert caught.value.error_kind == "invalid"
    assert message.startswith(f"{FRAMEWORK_OF[kind]} models do not read {field}.")
    readers = " and ".join(
        FRAMEWORK_OF[reader] for reader in ACCEPTED_KINDS if reader in READ_BY[field]
    )
    assert f"{field} is read only for {readers} models." in message


@pytest.mark.usefixtures("no_model_resolved")
@pytest.mark.parametrize(("field", "kind"), OWN, ids=[f"{f}-{k}" for f, k in OWN])
def test_a_field_set_for_its_own_framework_is_accepted(
    tmp_path: Path, field: str, kind: str
) -> None:
    ds = make_dataset(tmp_path)
    run_id = _run_of(kind)
    params = _params(model=[run_id], **{field: NON_DEFAULT[field]})

    assert predict_frames_model(ds, params) == (kind, run_id)


def test_every_foreign_field_is_named_in_one_refusal() -> None:
    params = _params(model=[CENTROID_RUN_ID], max_det=10, precision="fp16")

    with pytest.raises(PredictFramesInvalid) as caught:
        refuse_foreign_framework_fields(params, "train-sleap")

    assert str(caught.value) == (
        "SLEAP models do not read max_det or precision. max_det is read only for "
        "yolo-pose and POLO models. precision is read only for yolo-pose and POLO "
        "models. Leave them at their defaults."
    )


def test_a_foreign_field_is_refused_for_a_sleap_pair_not_on_disk(
    tmp_path: Path,
) -> None:
    """The refusal comes before the pair's identity, which needs both trainings."""
    ds = make_dataset(tmp_path)
    pair = [CENTROID_RUN_ID, INSTANCE_RUN_ID]

    with pytest.raises(PredictFramesInvalid, match="SLEAP models do not read iou"):
        _ = predict_frames_model(ds, _params(model=pair, iou=0.5))
    with pytest.raises(IdentityDeferred):
        _ = predict_frames_model(ds, _params(model=pair))


# --- Device -------------------------------------------------------------------


@pytest.mark.usefixtures("no_model_identity")
def test_cpu_is_refused_for_a_lightning_pose_model(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    params = _params(model=[_run_of("train-litpose")], device="cpu")

    with pytest.raises(PredictFramesInvalid) as caught:
        _ = predict_frames_model(ds, params)

    assert caught.value.error_kind == "invalid"
    assert str(caught.value) == (
        "unusable device 'cpu': Lightning Pose runs on CUDA only, so give 'auto' "
        "or a CUDA index such as '0'."
    )


@pytest.mark.parametrize("device", ["auto", "0", "1"])
def test_a_cuda_device_is_accepted_for_a_lightning_pose_model(device: str) -> None:
    params = _params(model=[_run_of("train-litpose")], device=device)

    refuse_device_for_kind(params.device, "train-litpose")


@pytest.mark.parametrize("kind", ["train-pose", "train-points", "train-sleap"])
def test_cpu_is_accepted_for_every_other_framework(kind: str) -> None:
    refuse_device_for_kind("cpu", kind)


# --- Model identity -----------------------------------------------------------


@pytest.mark.usefixtures("no_model_resolved")
@pytest.mark.parametrize("kind", ACCEPTED_KINDS)
def test_one_run_id_is_its_own_model_id_and_reads_no_model(
    tmp_path: Path, kind: str
) -> None:
    """The dataset registers no training, so a resolution would fail."""
    ds = make_dataset(tmp_path)
    run_id = _run_of(kind)

    assert predict_frames_model(ds, _params(model=[run_id])) == (kind, run_id)


def test_a_sleap_pair_is_named_by_the_digest_of_both_trainings(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path)
    centroid = register_sleap_model(ds, CENTROID_RUN_ID, b"centroid-weights")
    instance = register_sleap_model(ds, INSTANCE_RUN_ID, b"instance-weights")
    pair = [CENTROID_RUN_ID, INSTANCE_RUN_ID]
    checkpoints = [
        file_digest(directory / "best.ckpt") for directory in (centroid, instance)
    ]
    digest = hash_params({"sleap_weights": checkpoints})

    assert planned_model_id(ds, KIND, pair, "train-sleap") == digest
    assert predict_frames_model(ds, _params(model=pair)) == ("train-sleap", digest)
    reversed_pair = _params(model=pair[::-1])
    assert predict_frames_model(ds, reversed_pair)[1] != digest


# --- Run id -------------------------------------------------------------------


def _run_id(params: PredictFramesParams, model_id: str = POSE_RUN_ID) -> str:
    return predict_frames_run_id(
        VERSION, params, model_id=model_id, model_kind="train-pose"
    )


def test_the_run_id_names_the_op_and_its_version() -> None:
    parsed = parse_op_run_id(_run_id(_params()))

    assert parsed is not None
    assert (parsed.kind, parsed.version) == (KIND, VERSION)


def test_the_run_id_is_stable_under_field_order() -> None:
    fields: dict[str, object] = {
        "model": [POSE_RUN_ID],
        "frames_run_id": FRAMES_RUN_ID,
        "conf_threshold": 0.1,
        "imgsz": 320,
        "iou": 0.5,
    }
    reordered = dict(reversed(list(fields.items())))

    assert list(reordered) != list(fields)
    assert _run_id(PredictFramesParams.model_validate(reordered)) == _run_id(
        PredictFramesParams.model_validate(fields)
    )


@pytest.mark.parametrize(
    "overrides",
    [{"device": "cpu"}, {"batch_size": 4}, {"precision": "fp16"}],
    ids=["device", "batch_size", "precision"],
)
def test_the_run_id_leaves_out_throughput_and_precision(
    overrides: dict[str, object],
) -> None:
    assert _run_id(_params(**overrides)) == _run_id(_params())


def test_the_run_id_moves_with_each_term() -> None:
    base = _run_id(_params())
    moved = {
        "model_id": _run_id(_params(), model_id="0123456789abcdef"),
        "model_kind": predict_frames_run_id(
            VERSION, _params(), model_id=POSE_RUN_ID, model_kind="train-points"
        ),
        "frames_run_id": _run_id(_params(frames_run_id="kmeans-0123456789")),
        "conf_threshold": _run_id(_params(conf_threshold=0.1)),
        "model": _run_id(_params(model=[_run_of("train-pose", "0123456789")])),
    }

    assert all(run_id != base for run_id in moved.values()), moved
    assert len(set(moved.values())) == len(moved)
