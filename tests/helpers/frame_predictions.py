"""Frame-predictions runs on disk: an entry camera's predictions file and index row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    PredictedObject,
    PredictedPose,
    pose_predictions_payload,
)
from mosaic.core.helpers import entry_camera_path
from mosaic.tracking.frame_predictions.index import (
    FramePredictionsRow,
    predict_frames_index,
    prediction_index_path,
    prediction_kind_root,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["entry_predictions", "record_entry_predictions"]

_PREDICTING_MODEL = "train-pose.0.1-0123456789"
"""The training run that every recorded predictions row names as its model."""

_POSE = PredictedPose(id=1, name="mouse", schema=KeypointSchema(names=("nose", "tail")))


def entry_predictions(
    *, frames_run_id: str, group: str, sequence: str, camera: str = ""
) -> FramePredictions:
    """Build one entry camera's predictions: one frame, one object, two keypoints."""
    frames_root = Path("media/frames/uniform") / frames_run_id
    image = entry_camera_path(frames_root, group, sequence, camera) / "frame_000004.png"
    found = PredictedObject(
        keypoints=(
            Keypoint(x=10.0, y=20.0, visibility=2, score=0.9),
            Keypoint.absent(),
        ),
        score=0.8,
    )
    return FramePredictions(
        pose=_POSE,
        frames_run_id=frames_run_id,
        group=group,
        sequence=sequence,
        camera=camera,
        frames=(
            PredictedFrame(
                image=image.as_posix(),
                frame_index=4,
                width=64,
                height=48,
                objects=(found,),
            ),
        ),
    )


def record_entry_predictions(
    ds: Dataset,
    predictions: FramePredictions,
    *,
    run_id: str,
    finished: bool = True,
    path: Path | None = None,
    consumed_frames_composition: str = "",
) -> Path:
    """Write *predictions* and append its predict-frames row.

    Args:
        ds: The dataset whose frame_predictions root gains the file and row.
        predictions: The entry camera's predictions.
        run_id: The predict-frames run.
        finished: Mark every unfinished row of *run_id* finished afterwards.
        path: The file to write. By default, the entry camera's
            ``predictions.json`` in the run's layout.
        consumed_frames_composition: The frames row's composition that the row
            records as read.

    Returns:
        The file written, which the row names dataset-relative.
    """
    run_root = prediction_kind_root(ds, "predict-frames") / run_id
    target = path or (
        entry_camera_path(
            run_root, predictions.group, predictions.sequence, predictions.camera
        )
        / "predictions.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_bytes(pose_predictions_payload(predictions))
    index = predict_frames_index(prediction_index_path(ds, "predict-frames"))
    index.append(
        [
            FramePredictionsRow(
                run_id=run_id,
                abs_path=Path(ds.relative_to_root(target)),
                group=predictions.group,
                sequence=predictions.sequence,
                camera=predictions.camera,
                frames_method="uniform",
                frames_run_id=predictions.frames_run_id,
                model=json.dumps([_PREDICTING_MODEL]),
                model_kind="train-pose",
                model_id=_PREDICTING_MODEL,
                n_frames=len(predictions.frames),
                n_objects=sum(len(frame.objects) for frame in predictions.frames),
                consumed_frames_composition=consumed_frames_composition,
                tool_version="",
                params_hash="h",
            )
        ]
    )
    if finished:
        index.mark_finished(run_id)
    return target
