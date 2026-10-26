"""``predict-images`` in the built Ultralytics and POLO environments.

The suite's other predict-images tests fake the runner. This one runs it in the
environment a user builds, on weights that the environment constructs from a model
configuration, so nothing is downloaded. The weights are untrained. The test
therefore checks the shape of the exchange, not what the model finds: the rows'
columns, positions, boxes and scores, and that two runs write the same rows.

A case skips where its environment is not built, unless the job that builds it
sets ``MOSAIC_CI_TRACKING``, which refuses to start the suite instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    ToolNotFoundError,
    tool_invocation,
)
from mosaic.tracking.common.ultralytics_env import (
    POLO_ENV,
    ULTRALYTICS_ENV,
    PoloError,
    UltralyticsError,
    probe_environment,
)
from mosaic.tracking.external.runner.ultralytics_protocol import (
    ImageSource,
    InferTask,
    PredictImagesRequest,
    request_columns,
)
from mosaic.tracking.pose_training.ultralytics_infer import run_predict_images_tool
from tests.helpers import write_marked_images

pytestmark = pytest.mark.tracker

_SHAPES: Final = ((64, 96), (80, 64), (64, 96))
"""The ``(height, width)`` of each image: two shapes, so two batches."""

_BUILD_MODEL: Final = (
    "import sys\n"
    "from ultralytics import YOLO\n"
    "model = YOLO(sys.argv[1])\n"
    "shape = model.model.yaml.get('kpt_shape')\n"
    "if shape is not None:\n"
    "    model.model.kpt_shape = tuple(shape)\n"
    "head = model.model.model[-1]\n"
    "for name, parameter in head.named_parameters():\n"
    "    if 'cv3' in name and name.endswith('.2.bias'):\n"
    "        parameter.data.zero_()\n"
    "model.save(sys.argv[2])\n"
)
"""Builds untrained weights from a model configuration and saves them.

Ultralytics' trainer gives a pose model its ``kpt_shape``, and the predictor reads it
there, so the script sets it from the configuration as the trainer does. It zeroes the
class branch's output bias. An untrained head otherwise scores every candidate near
0.01, and nothing passes the confidence threshold.
"""


def _environment(task: InferTask) -> tuple[ToolEnv, list[str]]:
    """Return the environment of *task* and its python's argv.

    Skips when the environment does not resolve.
    """
    env = ULTRALYTICS_ENV if task == "pose" else POLO_ENV
    name = "Ultralytics" if task == "pose" else "POLO"
    try:
        argv = tool_invocation(env, executable="python")
    except ToolNotFoundError:
        pytest.skip(f"the {name} environment is not built here")
    return env, argv


def _built_weights(argv: list[str], configuration: str, tmp_path: Path) -> Path:
    """Build untrained weights from *configuration* in the environment of *argv*."""
    weights = tmp_path / "untrained.pt"
    _ = subprocess.run(
        [*argv, "-c", _BUILD_MODEL, configuration, str(weights)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return weights


def _request(
    task: InferTask,
    weights: Path,
    images: list[ImageSource],
    n_keypoints: int,
    rows_path: Path,
) -> PredictImagesRequest:
    """Return a request that keeps every detection the untrained model proposes."""
    return PredictImagesRequest(
        model_path=str(weights),
        task=task,
        images=images,
        columns=request_columns(task, n_keypoints),
        conf=0.001,
        iou=0.7 if task == "pose" else None,
        imgsz=64,
        max_det=5,
        agnostic_nms=False,
        device="cpu",
        precision="fp32",
        batch_size=2,
        n_keypoints=n_keypoints,
        dor=None,
        rows_path=str(rows_path),
    )


def test_predict_images_runs_in_the_ultralytics_environment(tmp_path: Path) -> None:
    """Rows, boxes and scores come back for images of two shapes, the same on a rerun."""
    env, argv = _environment("pose")
    weights = _built_weights(argv, "yolo11n-pose.yaml", tmp_path)
    probe = probe_environment(weights, env=env, failure=UltralyticsError)
    assert probe.model_load_error == "", probe.model_load_error
    assert probe.model_task == "pose"
    n_keypoints = probe.n_keypoints
    assert n_keypoints > 0
    images = write_marked_images(tmp_path, _SHAPES)

    tables: list[pd.DataFrame] = []
    for attempt in range(2):
        request = _request(
            "pose", weights, images, n_keypoints, tmp_path / f"rows_{attempt}.parquet"
        )
        response = run_predict_images_tool(
            request, work_dir=tmp_path / f"work_{attempt}"
        )
        assert response.n_images == len(images)
        tables.append(pd.read_parquet(response.rows_path))
        assert len(tables[-1]) == response.n_rows

    rows = tables[0]
    assert list(rows.columns) == request_columns("pose", n_keypoints)
    assert set(rows["frame"]) == {0, 1, 2}
    assert rows.groupby("frame").size().max() <= 5
    assert (rows["bbox_x2"] > rows["bbox_x1"]).all()
    assert (rows["bbox_y2"] > rows["bbox_y1"]).all()
    shapes = np.array(_SHAPES, dtype=np.float64)[rows["frame"].to_numpy(dtype=np.int64)]
    assert (rows["bbox_x1"].to_numpy(dtype=np.float64) >= 0).all()
    assert (rows["bbox_y1"].to_numpy(dtype=np.float64) >= 0).all()
    assert (rows["bbox_x2"].to_numpy(dtype=np.float64) <= shapes[:, 1]).all()
    assert (rows["bbox_y2"].to_numpy(dtype=np.float64) <= shapes[:, 0]).all()
    scores = rows["det_conf"].to_numpy(dtype=np.float64)
    assert ((scores >= 0.001) & (scores <= 1.0)).all()
    pd.testing.assert_frame_equal(tables[0], tables[1])


def test_predict_images_runs_in_the_polo_environment(tmp_path: Path) -> None:
    """Point rows come back for images of two shapes, the same on a rerun."""
    env, argv = _environment("locate")
    weights = _built_weights(argv, "polo26n.yaml", tmp_path)
    probe = probe_environment(weights, env=env, failure=PoloError)
    assert probe.model_load_error == "", probe.model_load_error
    assert probe.model_task == "locate"
    images = write_marked_images(tmp_path, _SHAPES)

    tables: list[pd.DataFrame] = []
    for attempt in range(2):
        request = _request(
            "locate", weights, images, 1, tmp_path / f"rows_{attempt}.parquet"
        )
        response = run_predict_images_tool(
            request, work_dir=tmp_path / f"work_{attempt}"
        )
        assert response.n_images == len(images)
        tables.append(pd.read_parquet(response.rows_path))

    assert list(tables[0].columns) == request_columns("locate", 1)
    assert set(tables[0]["frame"]) <= {0, 1, 2}
    pd.testing.assert_frame_equal(tables[0], tables[1])
