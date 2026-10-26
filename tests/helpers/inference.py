"""Stand in for the environments that ``infer-pose`` and ``infer-points`` run in.

Both ops spawn a runner in an Ultralytics environment. A test stands in for the
environment probe and the tool call, two module-level seams, and the installers
here replace both. The fake runner writes a predictions table, in the
runner's layout, at the path that the request names, because the op reads it back
to bridge it. The table is fixed unless a test passes a function of the video. A
request of several files gets each file's table, its frames moved onto the
entry's frame axis as the runner numbers them.

:func:`install_fake_predict_images` stands in for the same probe and for the
``predict-images`` call of a yolo-pose or POLO frame adapter. Its fake writes the
rows of a list of images, in the runner's layout, at the request's rows path.
:func:`fake_tool_probe` is the :class:`ToolProbe` that the installed probe reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import probe_media

from mosaic.tracking.external.runner.ultralytics_protocol import (
    InferPointsRequest,
    InferPoseRequest,
    InferTask,
    PredictImagesRequest,
    PredictImagesResponse,
    ProbeResponse,
)
from mosaic.tracking.frame_predictions.adapter import ToolProbe
from mosaic.tracking.pose_training.ultralytics_infer import InferenceOutcome

from tests.helpers.ultralytics import (
    frames_in_window,
    source_frame_count,
    ultralytics_probe_response,
)


def pose_predictions(frames: int = 4) -> pd.DataFrame:
    """Return *frames* frames of one two-keypoint animal, in a pose runner's layout.

    The two keypoints have different values. The body center that the bridge
    derives is their mean, ``(3, 5)``. A test that asserts it cannot pass by the
    bridge copying either keypoint.
    """
    return pd.DataFrame(
        {
            "frame": range(frames),
            "id": [0] * frames,
            "poseX0": [1.0] * frames,
            "poseY0": [2.0] * frames,
            "poseP0": [0.9] * frames,
            "poseX1": [5.0] * frames,
            "poseY1": [8.0] * frames,
            "poseP1": [0.8] * frames,
        }
    )


def pose_per_frame(video: Path) -> pd.DataFrame:
    """Return :func:`pose_predictions` at every frame of *video*, numbered from 0."""
    return pose_predictions(probe_media(video).frame_count)


def point_predictions() -> pd.DataFrame:
    """Return three detections over two frames, as a POLO point runner writes them."""
    return pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "detection_id": [0, 1, 0],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "confidence": [0.9, 0.8, 0.7],
            "class_id": [0, 0, 1],
            "class_name": ["bee", "bee", "feeder"],
        }
    )


type PredictionsFor = Callable[[Path], pd.DataFrame]
"""Return the predictions table that a fake runner writes for the video at a path."""


@dataclass
class FakeInference:
    """Stand in for the runner, and record every video that it is handed.

    The runner writes ``predictions(video)`` for each video of a request, with
    each video's frames after those of the videos before it, as the runner reads
    an entry's files.

    Attributes:
        predictions: The table for each video, as a function of the video's path.
        videos: Every video that the runner was handed, in order, over every
            request.
        probed: Every model whose environment was probed, in order. A run probes
            before any model runs, after every entry's input is resolved.
    """

    predictions: PredictionsFor
    videos: list[Path] = field(default_factory=list)
    probed: list[str] = field(default_factory=list)
    frames_read: int | None = None
    """How many frames the runner reports reading.

    ``None`` reports the frames of the request's window of its videos, which the
    runner reads whether or not it detects anything in them
    (:func:`~tests.helpers.ultralytics.frames_in_window`).
    """

    def run(
        self,
        request: InferPoseRequest | InferPointsRequest,
        *,
        work_dir: Path,
        **_kwargs: object,
    ) -> InferenceOutcome:
        tables: list[pd.DataFrame] = []
        first_frame = 0
        for source in request.sources:
            video = Path(source.path)
            self.videos.append(video)
            table = self.predictions(video)
            if first_frame:
                table = table.assign(frame=table["frame"] + first_frame)
            tables.append(table)
            first_frame += source_frame_count(source.media_facts)
        table = tables[0] if len(tables) == 1 else pd.concat(tables, ignore_index=True)
        published = Path(request.output_parquet)
        published.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(published, index=False)
        read = self.frames_read
        if read is None:
            read = frames_in_window(
                request.sources,
                request.start_frame,
                request.end_frame,
                request.frame_step,
                request.max_frames,
            )
        return InferenceOutcome(
            predictions_path=published, n_frames=read, n_rows=len(table)
        )


def _pose_probe_response() -> ProbeResponse:
    """Return the fake pose probe's report: a two-keypoint model."""
    return ultralytics_probe_response("pose", n_keypoints=2)


def _point_probe_response() -> ProbeResponse:
    """Return the fake point probe's report: a POLO model of one keypoint."""
    return ultralytics_probe_response(
        "locate", has_locate=True, version="8.4.84", n_keypoints=1
    )


def install_fake_pose_probe(
    monkeypatch: pytest.MonkeyPatch, probed: list[str] | None = None
) -> list[str]:
    """Stand in for the probe of ``infer-pose``'s environment: a two-keypoint model.

    Returns the list that records each probed model path, *probed* when given.
    """
    import mosaic.tracking.common.ultralytics_env as tool_env

    calls: list[str] = [] if probed is None else probed

    def probe(model_path: str, **_kwargs: object) -> ProbeResponse:
        calls.append(model_path)
        return _pose_probe_response()

    monkeypatch.setattr(tool_env, "probe_environment", probe)
    return calls


def install_fake_point_probe(
    monkeypatch: pytest.MonkeyPatch, probed: list[str] | None = None
) -> list[str]:
    """Stand in for the probe of ``infer-points``'s POLO environment.

    Returns the list that records each probed model path, *probed* when given.
    """
    import mosaic.tracking.common.ultralytics_env as tool_env

    calls: list[str] = [] if probed is None else probed

    def probe(model_path: str, **_kwargs: object) -> ProbeResponse:
        calls.append(model_path)
        return _point_probe_response()

    monkeypatch.setattr(tool_env, "probe_environment", probe)
    return calls


def install_fake_pose_inference(
    monkeypatch: pytest.MonkeyPatch, predictions: PredictionsFor | None = None
) -> FakeInference:
    """Stand in for ``infer-pose``'s environment, writing ``predictions(video)``.

    Without *predictions*, every video gets :func:`pose_predictions`.
    """
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    fake = FakeInference(predictions or (lambda _video: pose_predictions()))
    _ = install_fake_pose_probe(monkeypatch, fake.probed)
    monkeypatch.setattr(infer_run, "run_pose_inference_tool", fake.run)
    return fake


def install_fake_point_inference(
    monkeypatch: pytest.MonkeyPatch, predictions: PredictionsFor | None = None
) -> FakeInference:
    """Stand in for ``infer-points``'s POLO environment, writing ``predictions(video)``.

    Without *predictions*, every video gets :func:`point_predictions`.
    """
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    fake = FakeInference(predictions or (lambda _video: point_predictions()))
    _ = install_fake_point_probe(monkeypatch, fake.probed)
    monkeypatch.setattr(infer_run, "run_point_inference_tool", fake.run)
    return fake


def pose_image_rows(request: PredictImagesRequest) -> pd.DataFrame:
    """Return one detection per image of *request*, as the runner writes pose rows.

    The detection at position ``p`` has keypoint ``k`` at ``(10p + k, 20p + k)``
    with score 0.5, confidence 0.9, and the box from ``(p, 2p)`` to
    ``(p + 30, 2p + 40)``. The columns are ``request.columns``, in their order.
    """
    rows: list[dict[str, float]] = []
    for image in request.images:
        position = float(image.position)
        row = {"frame": position, "id": 0.0, "det_conf": 0.9}
        for k in range(request.n_keypoints):
            row |= {
                f"poseX{k}": 10 * position + k,
                f"poseY{k}": 20 * position + k,
                f"poseP{k}": 0.5,
            }
        row |= {
            "bbox_x1": position,
            "bbox_y1": 2 * position,
            "bbox_x2": position + 30,
            "bbox_y2": 2 * position + 40,
        }
        rows.append(row)
    table = pd.DataFrame(rows, columns=request.columns)
    return table.astype({"frame": "int64", "id": "int64"})


def point_image_rows(request: PredictImagesRequest) -> pd.DataFrame:
    """Return one point per image of *request*, as the runner writes point rows.

    The point at position ``p`` is at ``(10p, 20p)`` with confidence 0.8.
    """
    rows = [
        {
            "frame": image.position,
            "detection_id": 0,
            "x": 10.0 * image.position,
            "y": 20.0 * image.position,
            "confidence": 0.8,
            "class_id": 0,
            "class_name": "bee",
        }
        for image in request.images
    ]
    return pd.DataFrame(rows, columns=request.columns)


type ImageRowsFor = Callable[[PredictImagesRequest], pd.DataFrame]
"""Return the rows table that a fake runner writes for a ``predict-images`` request."""


@dataclass
class FakePredictImages:
    """Stand in for ``run_predict_images_tool``, and record every request.

    Attributes:
        rows: The rows table for a request.
        requests: Every request, in order.
        work_dirs: The work directory of each request, in order.
        cancel_checks: The cancel check of each request, in order.
        idle_timeouts: The inactivity bound of each request, in order.
        probed: Every model whose environment was probed, in order.
    """

    rows: ImageRowsFor
    requests: list[PredictImagesRequest] = field(default_factory=list)
    work_dirs: list[Path] = field(default_factory=list)
    cancel_checks: list[Callable[[], bool] | None] = field(default_factory=list)
    idle_timeouts: list[float] = field(default_factory=list)
    probed: list[str] = field(default_factory=list)

    def run(
        self,
        request: PredictImagesRequest,
        *,
        work_dir: Path,
        idle_timeout: float = 900,
        cancel_check: Callable[[], bool] | None = None,
        on_output: Callable[[str], None] | None = None,
        **_kwargs: object,
    ) -> PredictImagesResponse:
        """Write ``rows(request)`` at the request's rows path, and report one line."""
        self.requests.append(request)
        self.work_dirs.append(work_dir)
        self.cancel_checks.append(cancel_check)
        self.idle_timeouts.append(idle_timeout)
        table = self.rows(request)
        rows_path = Path(request.rows_path)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(rows_path, index=False)
        if on_output is not None:
            on_output(f"predicted {len(request.images)} images")
        return PredictImagesResponse(
            n_images=len(request.images), n_rows=len(table), rows_path=request.rows_path
        )


def fake_tool_probe(task: InferTask) -> ToolProbe:
    """Return the :class:`ToolProbe` of the fake probe of a *task* model."""
    response = _pose_probe_response() if task == "pose" else _point_probe_response()
    return ToolProbe(
        tool_version=response.ultralytics_version, n_keypoints=response.n_keypoints
    )


def install_fake_predict_images(
    monkeypatch: pytest.MonkeyPatch,
    task: InferTask,
    rows: ImageRowsFor | None = None,
) -> FakePredictImages:
    """Stand in for the probe and ``predict-images`` call of a *task* model.

    Without *rows*, a ``pose`` request gets :func:`pose_image_rows` and a
    ``locate`` request :func:`point_image_rows`. The probe is the fake of
    ``infer-pose`` or ``infer-points``, and reports :func:`fake_tool_probe`.
    """
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    if task == "pose":
        fake = FakePredictImages(rows or pose_image_rows)
        _ = install_fake_pose_probe(monkeypatch, fake.probed)
    else:
        fake = FakePredictImages(rows or point_image_rows)
        _ = install_fake_point_probe(monkeypatch, fake.probed)
    monkeypatch.setattr(infer_run, "run_predict_images_tool", fake.run)
    return fake
