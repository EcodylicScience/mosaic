"""The parameters, errors and adapter types of predict-frames.

The detection fields share their types and defaults with ``infer-pose``. The
throughput fields and the precision stay out of the run identity. The modules
below the op leave every op module unloaded.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_predictions import PredictedObject
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameAdapter,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.errors import (
    PredictFramesConflict,
    PredictFramesInvalid,
    PredictFramesNotFound,
    PredictFramesUnavailable,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ResolvedModel
from mosaic.tracking.ops.infer import PointInferParams, PoseInferParams
from tests.helpers import run_fresh

POSE_RUN_ID = "train-pose.0.2-abcdef0123"
CENTROID_RUN_ID = "train-sleap.0.1-abcdef0123"
INSTANCE_RUN_ID = "train-sleap.0.1-0123456789"
FRAMES_RUN_ID = "uniform-0123456789"

DETECTION_FIELDS = ("iou", "max_det", "agnostic_nms", "precision")
"""The fields whose types ``tracking/common/detection_params.py`` declares."""

OUT_OF_IDENTITY = frozenset({"device", "batch_size", "precision"})


def _params(**overrides: object) -> PredictFramesParams:
    return PredictFramesParams.model_validate(
        {"model": [POSE_RUN_ID], "frames_run_id": FRAMES_RUN_ID, **overrides}
    )


def test_every_optional_field_has_its_default() -> None:
    params = _params()

    assert params.model == [POSE_RUN_ID]
    assert params.frames_run_id == FRAMES_RUN_ID
    assert params.conf_threshold == 0.05
    assert params.imgsz == 640
    assert params.iou == 0.7
    assert params.max_det == 300
    assert params.agnostic_nms is False
    assert params.precision == "fp32"
    assert params.dor is None
    assert params.peak_threshold == 0.2
    assert params.max_instances is None
    assert params.device == "auto"
    assert params.batch_size == 16


@pytest.mark.parametrize("field", DETECTION_FIELDS)
def test_a_detection_field_is_declared_as_infer_pose_declares_it(field: str) -> None:
    """One field type gives both models infer-pose's bounds, prose and default."""
    ours = PredictFramesParams.model_json_schema()["properties"][field]
    theirs = PoseInferParams.model_json_schema()["properties"][field]

    assert ours == theirs
    assert (
        PredictFramesParams.model_fields[field].default
        == PoseInferParams.model_fields[field].default
    )


def test_dor_is_described_as_infer_points_describes_it() -> None:
    ours = PredictFramesParams.model_json_schema()["properties"]["dor"]
    theirs = PointInferParams.model_json_schema()["properties"]["dor"]

    assert ours["description"] == theirs["description"]


@pytest.mark.parametrize("missing", ["model", "frames_run_id"])
def test_the_models_and_the_extraction_run_are_required(missing: str) -> None:
    given: dict[str, object] = {"model": [POSE_RUN_ID], "frames_run_id": FRAMES_RUN_ID}
    del given[missing]

    with pytest.raises(ValidationError, match=missing):
        _ = PredictFramesParams.model_validate(given)


def test_the_model_field_is_a_list() -> None:
    annotation = PredictFramesParams.model_fields["model"].annotation

    assert typing.get_origin(annotation) is list


@pytest.mark.parametrize(
    "model", [[POSE_RUN_ID], [CENTROID_RUN_ID, INSTANCE_RUN_ID]], ids=["one", "pair"]
)
def test_one_model_or_a_pair_is_accepted(model: list[str]) -> None:
    assert _params(model=model).model == model


@pytest.mark.parametrize(
    "model",
    [[], [CENTROID_RUN_ID, INSTANCE_RUN_ID, POSE_RUN_ID]],
    ids=["none", "three"],
)
def test_no_model_or_more_than_two_is_refused(model: list[str]) -> None:
    with pytest.raises(ValidationError, match="model"):
        _ = _params(model=model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frames_run_id", ""),
        ("conf_threshold", -0.01),
        ("conf_threshold", 1.01),
        ("imgsz", 0),
        ("iou", -0.01),
        ("iou", 1.01),
        ("max_det", 0),
        ("precision", "int8"),
        ("dor", 0.0),
        ("dor", -1.0),
        ("peak_threshold", -0.01),
        ("peak_threshold", 1.01),
        ("max_instances", 0),
        ("batch_size", 0),
    ],
)
def test_a_value_outside_its_bounds_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _ = _params(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conf_threshold", 0.0),
        ("conf_threshold", 1.0),
        ("imgsz", 1),
        ("iou", 0.0),
        ("iou", 1.0),
        ("max_det", 1),
        ("agnostic_nms", True),
        ("precision", "fp16"),
        ("dor", 0.5),
        ("peak_threshold", 0.0),
        ("peak_threshold", 1.0),
        ("max_instances", 1),
        ("device", "cpu"),
        ("device", "1"),
        ("batch_size", 1),
    ],
)
def test_a_value_on_or_inside_its_bounds_is_accepted(field: str, value: object) -> None:
    assert getattr(_params(**{field: value}), field) == value


def test_the_device_the_batch_size_and_the_precision_leave_the_identity() -> None:
    params = _params(device="cpu", batch_size=2, precision="fp16")

    dumped = params.identity_dump()

    assert set(dumped) == set(PredictFramesParams.model_fields) - OUT_OF_IDENTITY
    assert dumped == _params().identity_dump()


@pytest.mark.parametrize(
    ("error", "kind", "base"),
    [
        (PredictFramesInvalid, "invalid", ValueError),
        (PredictFramesConflict, "conflict", ValueError),
        (PredictFramesNotFound, "not_found", LookupError),
        (PredictFramesUnavailable, "unavailable", RuntimeError),
    ],
)
def test_each_error_declares_its_kind(
    error: type[ClassifiedError], kind: ErrorKind, base: type[Exception]
) -> None:
    raised = error("the message")

    assert error.error_kind == kind
    assert isinstance(raised, base)
    envelope = error_envelope(raised)
    assert (envelope.kind, envelope.message) == (kind, "the message")


def test_the_errors_import_nothing_outside_core() -> None:
    """Beyond its packages, the errors module loads only ``mosaic.core`` modules."""
    code = (
        "import sys\n"
        "import mosaic.tracking.frame_predictions\n"
        "packages = set(sys.modules)\n"
        "import mosaic.tracking.frame_predictions.errors\n"
        "loaded = [m for m in set(sys.modules) - packages\n"
        "          if m.startswith('mosaic.')\n"
        "          and not m.startswith('mosaic.core.')\n"
        "          and m != 'mosaic.core'\n"
        "          and m != 'mosaic.tracking.frame_predictions.errors']\n"
        "print(sorted(loaded) or 'none')"
    )
    assert run_fresh(code) == "none"


@pytest.mark.parametrize("module", ["params", "adapter"])
def test_the_params_and_adapter_types_load_no_op_module(module: str) -> None:
    code = (
        "import sys\n"
        f"import mosaic.tracking.frame_predictions.{module}\n"
        "loaded = [m for m in sys.modules if m.startswith('mosaic.tracking.ops')]\n"
        "print(sorted(loaded) or 'none')"
    )
    assert run_fresh(code) == "none"


def test_the_probe_records_a_keypoint_count() -> None:
    names = [field.name for field in dataclasses.fields(ToolProbe)]

    assert names == ["tool_version", "n_keypoints"]
    assert ToolProbe(tool_version="2.3.1", n_keypoints=None).n_keypoints is None


def _empty_adapter(
    images: Sequence[FrameImage],
    model: ResolvedModel,
    params: PredictFramesParams,
    probe: ToolProbe,
    work_dir: Path,
    *,
    on_output: Callable[[str], None],
    cancel_check: Callable[[], bool],
) -> AdapterResult:
    """Find nothing in any image, reporting the probe's keypoint count as a pose."""
    del model, params, work_dir, cancel_check
    on_output(f"{len(images)} images")
    count = probe.n_keypoints or 0
    nothing: tuple[PredictedObject, ...] = ()
    return AdapterResult(
        objects=tuple(nothing for _ in images),
        tool_pose=KeypointSchema(names=tuple(f"kp{i}" for i in range(count))),
    )


def test_an_adapter_returns_one_tuple_of_objects_per_image(tmp_path: Path) -> None:
    adapter: FrameAdapter = _empty_adapter
    images = [
        FrameImage(
            position=0, key="a.png", path=tmp_path / "a.png", width=64, height=48
        ),
        FrameImage(
            position=1, key="b.png", path=tmp_path / "b.png", width=64, height=48
        ),
    ]
    model = ResolvedModel(
        artifacts=(), run_id=POSE_RUN_ID, digest="", members=(POSE_RUN_ID,)
    )
    lines: list[str] = []

    result = adapter(
        images,
        model,
        _params(),
        ToolProbe(tool_version="8.3.0", n_keypoints=2),
        tmp_path,
        on_output=lines.append,
        cancel_check=lambda: False,
    )

    assert result.objects == ((), ())
    assert result.tool_pose.names == ("kp0", "kp1")
    assert lines == ["2 images"]
