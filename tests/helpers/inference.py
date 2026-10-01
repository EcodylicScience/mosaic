"""Stand in for the environments that ``infer-pose`` and ``infer-points`` run in.

Both ops spawn a runner in an Ultralytics environment. A test stands in for the
environment probe and the tool call, two module-level seams, and the installers
here replace both. The fake runner writes a predictions table, in the
runner's layout, at the path that the request names, because the op reads it back
to bridge it. The table is fixed unless a test passes a function of the video. A
request of several files gets each file's table, its frames moved onto the
entry's frame axis as the runner numbers them.
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
    ProbeResponse,
)
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
        return ultralytics_probe_response("pose", n_keypoints=2)

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
        return ultralytics_probe_response(
            "locate", has_locate=True, version="8.4.84", n_keypoints=1
        )

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
