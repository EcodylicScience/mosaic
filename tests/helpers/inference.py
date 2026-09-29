"""Recording stand-ins for the environments ``infer-pose`` and ``infer-points`` run in.

Both ops spawn a runner in an Ultralytics environment, so what a test stands in
for is the environment probe and the tool call, two module-level seams. The
installers here replace both. The fake runner writes a fixed predictions table
at the path the request names, because the op reads it back to bridge it,
exactly as the runner would have written it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from mosaic.tracking.external.runner.ultralytics_protocol import (
    InferPointsRequest,
    InferPoseRequest,
    ProbeResponse,
)
from mosaic.tracking.pose_training.ultralytics_infer import InferenceOutcome


def pose_predictions() -> pd.DataFrame:
    """Four frames of one animal with two keypoints, as a pose runner writes them.

    The two keypoints deliberately hold different values: the body centre the
    bridge derives is their mean, ``(3, 5)``, so a test asserting it cannot pass
    by the bridge having copied either one.
    """
    return pd.DataFrame(
        {
            "frame": range(4),
            "id": [0] * 4,
            "poseX0": [1.0] * 4,
            "poseY0": [2.0] * 4,
            "poseP0": [0.9] * 4,
            "poseX1": [5.0] * 4,
            "poseY1": [8.0] * 4,
            "poseP1": [0.8] * 4,
        }
    )


def point_predictions() -> pd.DataFrame:
    """Three detections over two frames, as a POLO point runner writes them."""
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


@dataclass
class FakeInference:
    """A runner that writes *table* for every video, recording each one."""

    table: pd.DataFrame
    videos: list[Path] = field(default_factory=list)

    def run(
        self,
        request: InferPoseRequest | InferPointsRequest,
        *,
        work_dir: Path,
        **_kwargs: object,
    ) -> InferenceOutcome:
        self.videos.append(Path(request.video_path))
        published = Path(request.output_parquet)
        published.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(published, index=False)
        return InferenceOutcome(
            predictions_path=published,
            n_frames=int(self.table["frame"].nunique()),
            n_rows=len(self.table),
        )


def _probe(
    *, has_locate: bool, version: str, task: str, n_keypoints: int
) -> ProbeResponse:
    return ProbeResponse(
        has_ultralytics=True,
        has_lap=True,
        has_locate=has_locate,
        ultralytics_version=version,
        tracker_names=[],
        model_task=task,
        n_keypoints=n_keypoints,
        model_load_error="",
        installed_tracker_table={},
    )


def install_fake_pose_inference(
    monkeypatch: pytest.MonkeyPatch, table: pd.DataFrame | None = None
) -> FakeInference:
    """Stand in for ``infer-pose``'s environment, writing *table* per video.

    *table* defaults to :func:`pose_predictions`.
    """
    import mosaic.tracking.common.ultralytics_env as tool_env
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    fake = FakeInference(pose_predictions() if table is None else table)

    def probe(_model_path: str, **_kwargs: object) -> ProbeResponse:
        return _probe(has_locate=False, version="8.4.63", task="pose", n_keypoints=2)

    monkeypatch.setattr(tool_env, "probe_environment", probe)
    monkeypatch.setattr(infer_run, "run_pose_inference_tool", fake.run)
    return fake


def install_fake_point_inference(
    monkeypatch: pytest.MonkeyPatch, table: pd.DataFrame | None = None
) -> FakeInference:
    """Stand in for ``infer-points``'s POLO environment, writing *table* per video.

    *table* defaults to :func:`point_predictions`.
    """
    import mosaic.tracking.common.ultralytics_env as tool_env
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    fake = FakeInference(point_predictions() if table is None else table)

    def probe(_model_path: str, **_kwargs: object) -> ProbeResponse:
        return _probe(has_locate=True, version="8.4.84", task="locate", n_keypoints=1)

    monkeypatch.setattr(tool_env, "probe_environment", probe)
    monkeypatch.setattr(infer_run, "run_point_inference_tool", fake.run)
    return fake
