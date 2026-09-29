"""Stand in for the environments that ``infer-pose`` and ``infer-points`` run in.

Both ops spawn a runner in an Ultralytics environment. A test therefore stands in
for the environment probe and the tool call, two module-level seams. The
installers here replace both. The fake runner writes a fixed predictions table,
in the runner's layout, at the path that the request names, because the op reads
it back to bridge it.
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

from tests.helpers.ultralytics import ultralytics_probe_response


def pose_predictions() -> pd.DataFrame:
    """Return four frames of one animal with two keypoints, as a pose runner does.

    The two keypoints have different values. The body center that the bridge
    derives is their mean, ``(3, 5)``. A test that asserts it therefore cannot
    pass by the bridge copying either keypoint.
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


@dataclass
class FakeInference:
    """Return a runner that writes *table* for every video, recording each one."""

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
        return ultralytics_probe_response("pose", n_keypoints=2)

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
        return ultralytics_probe_response(
            "locate", has_locate=True, version="8.4.84", n_keypoints=1
        )

    monkeypatch.setattr(tool_env, "probe_environment", probe)
    monkeypatch.setattr(infer_run, "run_point_inference_tool", fake.run)
    return fake
