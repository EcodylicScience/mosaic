"""Frame-predictions runs on disk: an entry camera's predictions file and index row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    PredictedObject,
    PredictedPose,
    pose_predictions_payload,
)
from mosaic.core.annotations.scoring import (
    DEFAULT_PCK_ALPHAS,
    ErrorQuantiles,
    KeypointMapSummary,
    PoseScoreSummary,
    empty_frame_table,
    empty_instance_table,
)
from mosaic.core.helpers import entry_camera_path
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.tracking.frame_predictions.index import (
    FramePredictionsRow,
    ScoreRunRow,
    predict_frames_index,
    prediction_index_path,
    prediction_kind_root,
    score_predictions_index,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["entry_predictions", "record_entry_predictions", "record_score_run"]

_PREDICTING_MODEL = "train-pose.0.1-0123456789"
"""The training run that a recorded predictions row names as its model by default."""

_PREDICTING_RUN = "predict-frames.0.1-0123456789"
"""The predict-frames run that a recorded score names by default."""

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
    model: str = _PREDICTING_MODEL,
    model_kind: str = "train-pose",
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
        model: The training run that predicted, which the row names as its
            model and its model id.
        model_kind: The training kind of *model*.

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
    payload = pose_predictions_payload(predictions)
    atomic_write(target, lambda written: written.write_bytes(payload))
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
                model=json.dumps([model]),
                model_kind=model_kind,
                model_id=model,
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


def record_score_run(
    ds: Dataset,
    run_id: str,
    *,
    prediction_runs: tuple[str, ...] = (_PREDICTING_RUN,),
    finished: bool = True,
    frames: pd.DataFrame | None = None,
    outputs: tuple[str, ...] = ("frames.parquet", "instances.parquet", "summary.json"),
) -> Path:
    """Write a score run's outputs and its score-predictions row.

    Args:
        ds: The dataset whose frame_predictions root gains the run.
        run_id: The score run.
        prediction_runs: The predict-frames runs that the row and the summary name.
        finished: Mark the row finished afterwards.
        frames: The frames table to write. By default, one with no rows and the
            columns of the default ``pck_alphas``.
        outputs: The files to write in the run directory. A name left out is
            absent from disk while the row still names the run.

    Returns:
        The run directory, which the row names dataset-relative.
    """
    run_root = prediction_kind_root(ds, "score-predictions") / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if "frames.parquet" in outputs:
        _ = write_parquet_atomic(
            empty_frame_table(DEFAULT_PCK_ALPHAS) if frames is None else frames,
            run_root / "frames.parquet",
        )
    if "instances.parquet" in outputs:
        _ = write_parquet_atomic(
            empty_instance_table(len(_POSE.schema.names)),
            run_root / "instances.parquet",
        )
    if "summary.json" in outputs:
        text = _summary(prediction_runs).model_dump_json()
        atomic_write(run_root / "summary.json", lambda target: target.write_text(text))
    index = score_predictions_index(prediction_index_path(ds, "score-predictions"))
    index.append(
        [
            ScoreRunRow(
                run_id=run_id,
                abs_path=Path(ds.relative_to_root(run_root)),
                prediction_runs=json.dumps(list(prediction_runs)),
                model=json.dumps([_PREDICTING_MODEL]),
                model_kind="train-pose",
                model_id=_PREDICTING_MODEL,
                consumed_sets="[]",
                pose_id="1",
                n_scored=0,
                n_excluded_training=0,
                n_not_covered=0,
                training_frames_known=True,
                params_hash="h",
            )
        ]
    )
    if finished:
        index.mark_finished(run_id)
    return run_root


def _summary(prediction_runs: tuple[str, ...]) -> PoseScoreSummary:
    """Build the summary of a score that scored nothing."""
    none = ErrorQuantiles(median=None, p90=None)
    return PoseScoreSummary(
        keypoints=["nose", "tail"],
        keypoint_map=KeypointMapSummary(
            identity=True, coverage=["nose", "tail"], unmapped_sources=[]
        ),
        n_truth_frames=0,
        n_scored=0,
        n_excluded_training=0,
        n_not_covered=0,
        coverage=0.0,
        n_gt=0,
        n_pred=0,
        n_matched=0,
        n_fp=0,
        n_fp_duplicate=0,
        recall=None,
        precision=None,
        fp_per_frame=None,
        frac_frames_all_found=None,
        moks=None,
        swap_rate=None,
        pck={},
        pck_per_keypoint={},
        miss_share_per_keypoint={},
        error_px_per_keypoint={},
        error_px=none,
        error_norm=none,
        n_kpt_eligible=0,
        n_kpt_untouched=0,
        n_kpt_unnormalized=0,
        n_objects_source_unknown=0,
        intervals=None,
        interval_reason="fewer than 2 sequences",
        n_sequences=0,
        n_boot=1000,
        confidence=0.95,
        random_state=0,
        prediction_runs=list(prediction_runs),
        model_kind="train-pose",
        model_id=_PREDICTING_MODEL,
        model=[_PREDICTING_MODEL],
        labels=[],
        training_frames_known=True,
        training_frames_reason="",
    )
