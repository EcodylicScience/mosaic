"""The yolo-pose and POLO frame adapters, and the runner exchange they share.

The adapters build a ``predict-images`` request from a predict-frames run's
params and the tool's probe, and turn the rows the runner writes into predicted
objects. A fake stands in for ``run_predict_images_tool``. The exchange tests
stand in for ``run_runner`` alone. Each tool runs in the environment of its
task, and returns what the runner's response reports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

import mosaic.tracking.pose_training.ultralytics_infer as ultralytics_infer
from mosaic.core.annotations.model import Bbox, Keypoint, KeypointSchema
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    PredictedPose,
    pose_predictions_payload,
)
from mosaic.core.pipeline.toolenv import ToolEnv
from mosaic.tracking.common.ultralytics_env import (
    POLO_ENV,
    ULTRALYTICS_ENV,
    PoloError,
    UltralyticsError,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    POINT_COLUMNS,
    ImageSource,
    InferPointsRequest,
    InferPoseRequest,
    InferResponse,
    InferTask,
    PredictImagesRequest,
    PredictImagesResponse,
    ProbeResponse,
    SourceFile,
    pose_columns,
)
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameAdapter,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.model_refs import ModelArtifact, ResolvedModel
from mosaic.tracking.pose_training.ultralytics_infer import InferenceOutcome
from tests.helpers import (
    fake_tool_probe,
    install_fake_predict_images,
    point_image_rows,
    pose_image_rows,
)

pytestmark = pytest.mark.tracker

_POSE_RUN_ID = "train-pose.0.2-abcdef0123"
_POINTS_RUN_ID = "train-points.0.2-abcdef0123"


def _adapter(task: InferTask) -> FrameAdapter:
    """Return the frame adapter of a *task* model."""
    if task == "pose":
        return ultralytics_infer.predict_pose_frames
    return ultralytics_infer.predict_point_frames


def _model(tmp_path: Path, run_id: str) -> ResolvedModel:
    """Return a resolved model whose weights are ``best.pt`` under *tmp_path*."""
    weights = tmp_path / "models" / "best.pt"
    artifact = ModelArtifact(root=weights, files=(), exec_path=weights, digest="d")
    return ResolvedModel(
        artifacts=(artifact,), run_id=run_id, digest="d", members=(run_id,)
    )


def _images(tmp_path: Path, positions: list[int]) -> list[FrameImage]:
    """Return images at *positions*, in that order, each named by its position."""
    return [
        FrameImage(
            position=position,
            key=f"frames/frame_{position:06d}.png",
            path=tmp_path / "frames" / f"frame_{position:06d}.png",
            width=64,
            height=48,
        )
        for position in positions
    ]


def _params(task: InferTask, **overrides: object) -> PredictFramesParams:
    """Return the params of a run of one *task* model, with *overrides* set."""
    run_id = _POSE_RUN_ID if task == "pose" else _POINTS_RUN_ID
    fields: dict[str, object] = {"model": [run_id], "frames_run_id": "frames.0.1-x"}
    fields.update(overrides)
    return PredictFramesParams.model_validate(fields)


def _predict(
    tmp_path: Path,
    task: InferTask,
    positions: list[int],
    *,
    lines: list[str] | None = None,
    cancel_check: Callable[[], bool] = lambda: False,
    **overrides: object,
) -> AdapterResult:
    """Run the *task* adapter over images at *positions*, with the fake probe's report."""
    run_id = _POSE_RUN_ID if task == "pose" else _POINTS_RUN_ID
    received: list[str] = [] if lines is None else lines
    return _adapter(task)(
        _images(tmp_path, positions),
        _model(tmp_path, run_id),
        _params(task, **overrides),
        fake_tool_probe(task),
        tmp_path / "work",
        on_output=received.append,
        cancel_check=cancel_check,
    )


# --- the request -----------------------------------------------------------


@pytest.mark.parametrize("task", ["pose", "locate"])
@pytest.mark.parametrize(("device", "sent"), [("auto", ""), ("cpu", "cpu"), ("1", "1")])
def test_the_device_reaches_the_request_in_the_spelling_of_ultralytics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: InferTask,
    device: str,
    sent: str,
) -> None:
    """``auto`` is sent empty, and Ultralytics then chooses the device."""
    fake = install_fake_predict_images(monkeypatch, task)

    _ = _predict(tmp_path, task, [0], device=device)

    assert [request.device for request in fake.requests] == [sent]


def test_a_pose_request_names_the_pose_and_box_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request is sent with the columns and task that mosaic builds, and the op's ``iou``."""
    fake = install_fake_predict_images(monkeypatch, "pose")
    n_keypoints = fake_tool_probe("pose").n_keypoints
    assert n_keypoints is not None

    _ = _predict(tmp_path, "pose", [0], iou=0.55)

    (request,) = fake.requests
    assert request.columns == [
        *pose_columns(n_keypoints),
        "det_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    assert request.task == "pose"
    assert request.iou == 0.55
    assert request.n_keypoints == n_keypoints
    assert request.dor is None


def test_a_point_request_names_the_point_columns_and_no_iou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POLO's point suppression reads no ``iou``, and the runner refuses one."""
    fake = install_fake_predict_images(monkeypatch, "locate")

    _ = _predict(tmp_path, "locate", [0], dor=0.4)

    (request,) = fake.requests
    assert request.columns == list(POINT_COLUMNS)
    assert request.task == "locate"
    assert request.iou is None
    assert request.dor == 0.4
    assert request.n_keypoints == 1


@pytest.mark.parametrize("task", ["pose", "locate"])
def test_a_request_carries_the_images_the_model_and_the_detection_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: InferTask
) -> None:
    fake = install_fake_predict_images(monkeypatch, task)

    _ = _predict(
        tmp_path,
        task,
        [3, 1],
        conf_threshold=0.2,
        imgsz=320,
        max_det=12,
        agnostic_nms=True,
        precision="fp16",
        batch_size=4,
    )

    (request,) = fake.requests
    assert request.model_path == str(tmp_path / "models" / "best.pt")
    assert request.images == [
        ImageSource(position=3, path=str(tmp_path / "frames" / "frame_000003.png")),
        ImageSource(position=1, path=str(tmp_path / "frames" / "frame_000001.png")),
    ]
    assert (request.conf, request.imgsz, request.max_det) == (0.2, 320, 12)
    assert request.agnostic_nms is True
    assert request.precision == "fp16"
    assert request.batch_size == 4
    assert Path(request.rows_path).parent == tmp_path / "work"
    assert fake.work_dirs == [tmp_path / "work"]


@pytest.mark.parametrize("task", ["pose", "locate"])
def test_the_adapter_passes_its_callbacks_to_the_tool_and_probes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: InferTask
) -> None:
    """The op probes the environment and checks the model, before any claim."""
    fake = install_fake_predict_images(monkeypatch, task)
    checked: list[str] = []

    def record_check(_probe: ProbeResponse, model_path: str) -> None:
        checked.append(model_path)

    monkeypatch.setattr(ultralytics_infer, "require_pose_model", record_check)
    monkeypatch.setattr(ultralytics_infer, "require_points_model", record_check)
    lines: list[str] = []

    def never_cancelled() -> bool:
        return False

    _ = _predict(tmp_path, task, [0, 1], lines=lines, cancel_check=never_cancelled)

    assert lines == ["predicted 2 images"]
    assert fake.cancel_checks == [never_cancelled]
    assert fake.probed == []
    assert checked == []


# --- the objects -----------------------------------------------------------


def test_a_pose_object_has_its_keypoints_its_confidence_and_its_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The box corners become a box of an origin, a width and a height."""
    _ = install_fake_predict_images(monkeypatch, "pose")

    result = _predict(tmp_path, "pose", [2])

    ((found,),) = result.objects
    assert found.keypoints == (
        Keypoint(x=20.0, y=40.0, visibility=2, score=0.5),
        Keypoint(x=21.0, y=41.0, visibility=2, score=0.5),
    )
    assert found.score == pytest.approx(0.9)
    assert found.bbox == Bbox(x=2.0, y=4.0, width=30.0, height=40.0)
    assert found.track_id is None


def test_a_pose_without_a_visibility_channel_writes_null_keypoint_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner writes NaN for a score the model does not predict."""

    def without_scores(request: PredictImagesRequest) -> pd.DataFrame:
        table = pose_image_rows(request)
        for k in range(request.n_keypoints):
            table[f"poseP{k}"] = math.nan
        return table

    _ = install_fake_predict_images(monkeypatch, "pose", without_scores)

    result = _predict(tmp_path, "pose", [0])

    ((found,),) = result.objects
    assert [point.score for point in found.keypoints] == [None, None]
    predictions = FramePredictions(
        pose=PredictedPose(id=None, name="", schema=result.tool_pose),
        frames_run_id="frames.0.1-x",
        group="",
        sequence="s",
        camera="",
        frames=(
            PredictedFrame(
                image="a.png", frame_index=0, width=64, height=48, objects=(found,)
            ),
        ),
    )
    document = json.loads(pose_predictions_payload(predictions))
    assert document["frames"][0]["objects"][0]["keypoints"] == [
        [0.0, 0.0, None],
        [1.0, 1.0, None],
    ]


def test_a_point_object_is_one_keypoint_scored_by_its_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POLO does not report a keypoint confidence or a box."""
    _ = install_fake_predict_images(monkeypatch, "locate")

    result = _predict(tmp_path, "locate", [3])

    ((found,),) = result.objects
    assert found.keypoints == (Keypoint(x=30.0, y=60.0, visibility=2, score=None),)
    assert found.score == pytest.approx(0.8)
    assert found.bbox is None
    assert found.track_id is None


@pytest.mark.parametrize(
    ("task", "id_column", "x_column", "rows"),
    [
        ("pose", "id", "poseX0", pose_image_rows),
        ("locate", "detection_id", "x", point_image_rows),
    ],
)
def test_the_objects_follow_the_images_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: InferTask,
    id_column: str,
    x_column: str,
    rows: Callable[[PredictImagesRequest], pd.DataFrame],
) -> None:
    """Each image gets the detections at its position, in detection order.

    Position 2 has a second detection, 100 pixels to the right of its first and
    written before it. The rows of position 0 are dropped.
    """

    def found_at_two_and_five(request: PredictImagesRequest) -> pd.DataFrame:
        table = rows(request)
        at_two = table[table["frame"] == 2]
        second = at_two.assign(**{id_column: 1, x_column: at_two[x_column] + 100})
        table = pd.concat([second, table], ignore_index=True)
        return table[table["frame"] != 0].reset_index(drop=True)

    _ = install_fake_predict_images(monkeypatch, task, found_at_two_and_five)

    result = _predict(tmp_path, task, [5, 0, 2])

    assert [len(objects) for objects in result.objects] == [1, 0, 2]
    assert [found.keypoints[0].x for found in result.objects[0]] == [50.0]
    assert [found.keypoints[0].x for found in result.objects[2]] == [20.0, 120.0]


@pytest.mark.parametrize(
    ("task", "names"), [("pose", ("kp0", "kp1")), ("locate", ("kp0",))]
)
def test_the_tool_pose_is_one_keypoint_per_index_with_no_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: InferTask,
    names: tuple[str, ...],
) -> None:
    _ = install_fake_predict_images(monkeypatch, task)

    result = _predict(tmp_path, task, [0])

    assert result.tool_pose == KeypointSchema(names=names)


@pytest.mark.parametrize("task", ["pose", "locate"])
def test_an_image_list_with_nothing_found_returns_an_empty_tuple_per_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: InferTask
) -> None:
    def nothing(request: PredictImagesRequest) -> pd.DataFrame:
        default = pose_image_rows if request.task == "pose" else point_image_rows
        return default(request).iloc[0:0]

    _ = install_fake_predict_images(monkeypatch, task, nothing)

    result = _predict(tmp_path, task, [0, 1])

    assert result.objects == ((), ())


@pytest.mark.parametrize("task", ["pose", "locate"])
def test_a_probe_without_a_keypoint_count_is_refused_before_the_tool_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: InferTask
) -> None:
    """The request's columns are built from the count. Without one, the tool is not run."""
    fake = install_fake_predict_images(monkeypatch, task)
    run_id = _POSE_RUN_ID if task == "pose" else _POINTS_RUN_ID

    with pytest.raises(ValueError, match="keypoint count"):
        _ = _adapter(task)(
            _images(tmp_path, [0]),
            _model(tmp_path, run_id),
            _params(task),
            ToolProbe(tool_version="8.4.63", n_keypoints=None),
            tmp_path / "work",
            on_output=lambda _line: None,
            cancel_check=lambda: False,
        )
    assert fake.requests == []


# --- the exchange ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RecordedRun:
    """One call of the faked ``run_runner``."""

    tool: str
    failure: str
    subcommand: str
    request: str


def _install_runner(
    monkeypatch: pytest.MonkeyPatch,
    response: BaseModel,
    output: Path | None,
) -> list[_RecordedRun]:
    """Stand in for ``run_runner``: write *response* and touch *output*.

    Returns the list that records each call. *output* None leaves the output unwritten.
    """
    calls: list[_RecordedRun] = []

    def run_runner(
        env: ToolEnv,
        failure: type[Exception],
        subcommand: str,
        request_path: Path,
        response_path: Path,
        **_kwargs: object,
    ) -> tuple[str, str]:
        calls.append(
            _RecordedRun(
                tool=env.tool,
                failure=failure.__name__,
                subcommand=subcommand,
                request=request_path.read_text(),
            )
        )
        _ = response_path.write_text(response.model_dump_json())
        if output is not None:
            _ = output.write_bytes(b"rows")
        return "", ""

    monkeypatch.setattr(ultralytics_infer, "run_runner", run_runner)
    return calls


def _predict_images_request(task: InferTask, rows_path: Path) -> PredictImagesRequest:
    return PredictImagesRequest(
        model_path="best.pt",
        task=task,
        images=[ImageSource(position=0, path="/frames/a.png")],
        columns=list(POINT_COLUMNS),
        conf=0.05,
        iou=0.7 if task == "pose" else None,
        imgsz=640,
        max_det=300,
        agnostic_nms=False,
        device="",
        precision="fp32",
        batch_size=16,
        n_keypoints=1,
        dor=None,
        rows_path=str(rows_path),
    )


@pytest.mark.parametrize(
    ("task", "tool", "failure"),
    [
        ("pose", ULTRALYTICS_ENV.tool, UltralyticsError.__name__),
        ("locate", POLO_ENV.tool, PoloError.__name__),
    ],
)
def test_predict_images_runs_in_the_environment_of_its_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: InferTask,
    tool: str,
    failure: str,
) -> None:
    """yolo-pose runs upstream Ultralytics, and POLO its fork."""
    rows_path = tmp_path / "work" / "rows.parquet"
    request = _predict_images_request(task, rows_path)
    response = PredictImagesResponse(n_images=1, n_rows=3, rows_path=str(rows_path))
    calls = _install_runner(monkeypatch, response, rows_path)

    answered = ultralytics_infer.run_predict_images_tool(
        request, work_dir=tmp_path / "work"
    )

    assert answered == response
    assert calls == [
        _RecordedRun(
            tool=tool,
            failure=failure,
            subcommand="predict-images",
            request=request.model_dump_json(),
        )
    ]


def test_predict_images_refuses_an_exchange_that_wrote_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_path = tmp_path / "work" / "rows.parquet"
    request = _predict_images_request("pose", rows_path)
    response = PredictImagesResponse(n_images=1, n_rows=0, rows_path=str(rows_path))
    _ = _install_runner(monkeypatch, response, None)

    with pytest.raises(FileNotFoundError, match="rows.parquet"):
        _ = ultralytics_infer.run_predict_images_tool(
            request, work_dir=tmp_path / "work"
        )


def _infer_fields(output: Path) -> dict[str, object]:
    """The fields that both inference requests share, publishing to *output*."""
    return {
        "model_path": "best.pt",
        "sources": [SourceFile(path="/videos/a.mp4", media_facts={})],
        "output_parquet": str(output),
        "annotated_dir": "",
        "columns": pose_columns(1),
        "conf": 0.25,
        "imgsz": 640,
        "max_det": 300,
        "agnostic_nms": False,
        "device": "0",
        "precision": "fp32",
        "start_frame": 0,
        "end_frame": None,
        "frame_step": 1,
        "max_frames": None,
        "batch_size": 16,
        "prefetch": True,
    }


def test_pose_inference_returns_its_outcome_from_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "work" / "predictions.parquet"
    request = InferPoseRequest.model_validate(
        {**_infer_fields(output), "task": "pose", "n_keypoints": 1, "iou": 0.7}
    )
    calls = _install_runner(monkeypatch, InferResponse(n_frames=20, n_rows=7), output)

    outcome = ultralytics_infer.run_pose_inference_tool(
        request, work_dir=tmp_path / "work"
    )

    assert outcome == InferenceOutcome(predictions_path=output, n_frames=20, n_rows=7)
    assert [(call.tool, call.failure, call.subcommand) for call in calls] == [
        (ULTRALYTICS_ENV.tool, UltralyticsError.__name__, "infer-pose")
    ]
    assert (tmp_path / "work" / ultralytics_infer.INFER_RESPONSE_NAME).is_file()


def test_point_inference_returns_its_outcome_from_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "work" / "predictions.parquet"
    request = InferPointsRequest.model_validate(
        {**_infer_fields(output), "task": "locate", "dor": None, "radii": None}
    )
    calls = _install_runner(monkeypatch, InferResponse(n_frames=9, n_rows=2), output)

    outcome = ultralytics_infer.run_point_inference_tool(
        request, work_dir=tmp_path / "work"
    )

    assert outcome == InferenceOutcome(predictions_path=output, n_frames=9, n_rows=2)
    assert [(call.tool, call.failure, call.subcommand) for call in calls] == [
        (POLO_ENV.tool, PoloError.__name__, "infer-points")
    ]


def test_inference_refuses_an_exchange_that_wrote_no_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "work" / "predictions.parquet"
    request = InferPoseRequest.model_validate(
        {**_infer_fields(output), "task": "pose", "n_keypoints": 1, "iou": 0.7}
    )
    _ = _install_runner(monkeypatch, InferResponse(n_frames=1, n_rows=0), None)

    with pytest.raises(FileNotFoundError, match="predictions.parquet"):
        _ = ultralytics_infer.run_pose_inference_tool(
            request, work_dir=tmp_path / "work"
        )
