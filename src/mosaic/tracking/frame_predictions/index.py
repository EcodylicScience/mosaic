"""Defines the indexes of the frame_predictions root and their inventory records.

The root contains one index per kind directory. ``predict-frames/index.csv`` has
one :class:`FramePredictionsRow` per run and entry camera, and
``score-predictions/index.csv`` one :class:`ScoreRunRow` per run. Rows are written
under ``index_lock`` through :class:`~mosaic.core.pipeline.index_csv.IndexCSV`.
Reconcile opens both through :func:`frame_predictions_index`, and the inventory
reads them through :func:`frame_prediction_records` and
:func:`prediction_score_records`. Each run root also contains the run's identity
payload, which :func:`write_run_params` writes. This module imports no tool or
trainer code. Of frame extraction it imports only the frames index, which the
drift check reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from mosaic.core.pipeline._utils import atomic_write, json_ready
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.dataset_indexes import (
    register_reconcilable_index,
    root_subdirectories,
)
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.inventory._runs import keyed_run_records, run_keyed_record
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    FramePredictionsRef,
    PredictionScoreRef,
)
from mosaic.tracking.frame_extraction.frame_runs import (
    frames_index_path,
    list_frame_runs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import CameraEntry, Entry
    from mosaic.core.pipeline.inventory._read import IndexReader
    from mosaic.core.pipeline.inventory.model import InventoryScope

__all__ = [
    "FRAME_PREDICTIONS_COLUMNS",
    "FRAME_PREDICTIONS_ROOT",
    "RUN_PARAMS_NAME",
    "SCORE_RUN_COLUMNS",
    "SCORE_RUN_OUTPUTS",
    "FramePredictionsRow",
    "PredictionKind",
    "ScoreRunRow",
    "frame_prediction_records",
    "frame_predictions_index",
    "predict_frames_index",
    "prediction_index_path",
    "prediction_kind_root",
    "prediction_score_records",
    "score_predictions_index",
    "score_run_is_complete",
    "write_run_params",
]

FRAME_PREDICTIONS_ROOT: Final = "frame_predictions"
"""The root key, which the default manifest also uses as the directory name."""

PredictionKind = Literal["predict-frames", "score-predictions"]
"""A kind directory of the root, named for the op that writes its runs."""

RUN_PARAMS_NAME: Final = "run_params.json"
"""What a run's identity payload is called, in its run root."""


@dataclass(frozen=True, slots=True)
class FramePredictionsRow(RunIndexRowBase):
    """Records one entry camera that a predict-frames run predicted.

    ``abs_path`` names the entry camera's ``predictions.json``, dataset-relative.
    No other column is a path.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The entry's camera as the frames row names it, ``""`` for
            single-camera media.
        frames_method: The extraction method directory (``uniform``, ``kmeans``
            or ``list``).
        frames_run_id: The extraction run.
        model: The model references as given, a JSON list of run ids in order.
        model_kind: The training kind that the references resolve under.
        model_id: The planned model id of the references: the run id, or the
            content digest of a pair.
        n_frames: Frames predicted in the file.
        n_objects: Objects across those frames.
        consumed_frames_composition: The frames row's ``media_composition`` when
            the run read it, ``""`` when unknown. Recorded and never hashed.
        tool_version: The release of the tool that predicted, ``""`` when the
            tool reports none. Recorded and never hashed.
        params_hash: ``hash_params`` of the run's identity payload.
    """

    group: str
    sequence: str
    camera: str
    frames_method: str
    frames_run_id: str
    model: str
    model_kind: str
    model_id: str
    n_frames: int
    n_objects: int
    consumed_frames_composition: str
    tool_version: str
    params_hash: str


FRAME_PREDICTIONS_COLUMNS: Final = tuple(
    field.name for field in fields(FramePredictionsRow)
)
"""The predict-frames index schema, in CSV order."""


@dataclass(frozen=True, slots=True)
class ScoreRunRow(RunIndexRowBase):
    """Records one score-predictions run.

    ``abs_path`` names the run directory, dataset-relative.

    Attributes:
        prediction_runs: The scored predict-frames runs, a JSON list.
        model: The model references of those runs, a JSON list.
        model_kind: The training kind that the references resolve under.
        model_id: The planned model id of the references.
        consumed_sets: The scored revisions, a JSON list of ``{origin_uuid,
            set_key, revision, digest}`` with ``revision`` an integer.
        pose_id: The scored pose id in decimal, ``""`` when unknown.
        n_scored: Truth frames scored.
        n_excluded_training: Truth frames excluded because the model trained or
            validated on them.
        n_not_covered: Truth frames whose image key no prediction frame has.
        training_frames_known: Whether the frames that the model trained or
            validated on are known. False when its provenance is unreadable or
            seen frames are not excluded.
        params_hash: ``hash_params`` of the run's identity payload.
    """

    prediction_runs: str
    model: str
    model_kind: str
    model_id: str
    consumed_sets: str
    pose_id: str
    n_scored: int
    n_excluded_training: int
    n_not_covered: int
    training_frames_known: bool
    params_hash: str


SCORE_RUN_COLUMNS: Final = tuple(field.name for field in fields(ScoreRunRow))
"""The score-predictions index schema, in CSV order."""

SCORE_RUN_OUTPUTS: Final = ("frames.parquet", "instances.parquet", "summary.json")
"""The files that a finished score-predictions run has written in its directory."""


def prediction_kind_root(ds: Dataset, kind: PredictionKind) -> Path:
    """Return the directory that contains the *kind* runs and their index."""
    return ds.get_root(FRAME_PREDICTIONS_ROOT) / kind


def prediction_index_path(ds: Dataset, kind: PredictionKind) -> Path:
    """Return the index of the *kind* runs, which may not exist yet."""
    return prediction_kind_root(ds, kind) / "index.csv"


def write_run_params(run_root: Path, payload: Mapping[str, object]) -> None:
    """Write a run's identity payload to :data:`RUN_PARAMS_NAME` in *run_root*."""
    text = json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True)
    atomic_write(run_root / RUN_PARAMS_NAME, lambda target: target.write_text(text))


def predict_frames_index(path: Path) -> IndexCSV[FramePredictionsRow]:
    """Open a predict-frames index, one row per run and entry camera.

    The camera is part of the key, as in the frames index. The cameras of one
    recording share a run, group and sequence.
    """
    return IndexCSV(
        path,
        FramePredictionsRow,
        dedup_keys=["run_id", "group", "sequence", "camera"],
        adopt=partial(project_to_schema, columns=FRAME_PREDICTIONS_COLUMNS),
    )


def score_predictions_index(path: Path) -> IndexCSV[ScoreRunRow]:
    """Open a score-predictions index, one row per run."""
    return IndexCSV(
        path,
        ScoreRunRow,
        dedup_keys=["run_id"],
        adopt=partial(project_to_schema, columns=SCORE_RUN_COLUMNS),
    )


def frame_predictions_index(
    path: Path,
) -> IndexCSV[FramePredictionsRow] | IndexCSV[ScoreRunRow]:
    """Open the index at *path* with the row class that its kind directory names.

    Reconcile opens every index of the root through this function. The registry
    keeps one opener per root key, and this root contains one index per kind
    directory.

    Raises:
        ValueError: *path* is in a directory that names no kind.
    """
    kind = path.parent.name
    if kind == "predict-frames":
        return predict_frames_index(path)
    if kind == "score-predictions":
        return score_predictions_index(path)
    msg = (
        f"{path} is not an index of the {FRAME_PREDICTIONS_ROOT} root: its "
        f"directory {kind!r} names no kind. The kinds are predict-frames and "
        "score-predictions."
    )
    raise ValueError(msg)


def frame_prediction_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[CameraEntry]]:
    """Return every predict-frames run, keyed by ``(group, sequence, camera)``.

    A run is covered by the entry cameras of its rows, as a frame run is. A row
    whose file is gone is an orphan row. A run is drifted on every entry whose
    extraction was cut again from other media after the run read it.

    Args:
        ds: The dataset. Never written to.
        scope: What was asked about. Its entry selector drops rows outside it.
        reader: The scan's reader, which reads each index once.

    Returns:
        The records, ordered by run id. Empty when the root is undeclared or
        absent.
    """
    kinds = (
        ["predict-frames"]
        if "predict-frames" in root_subdirectories(ds, FRAME_PREDICTIONS_ROOT)
        else []
    )
    return keyed_run_records(
        ds,
        scope,
        reader,
        root_key=FRAME_PREDICTIONS_ROOT,
        kinds=kinds,
        read_index=_read_predict_frames_index,
        key_of=lambda record: (
            record.get("group", ""),
            record.get("sequence", ""),
            record.get("camera", ""),
        ),
        make_ref=lambda _kind, run_id, rows: FramePredictionsRef(
            run_id=run_id, frames_run_id=index_records(rows)[0].get("frames_run_id", "")
        ),
        drifted=lambda _run_id, rows: _drifted_entries(ds, reader, rows),
    )


def _read_predict_frames_index(ds: Dataset, _kind: str) -> pd.DataFrame:
    """Read the predict-frames index, which raises when it is absent."""
    return predict_frames_index(prediction_index_path(ds, "predict-frames")).read()


def _drifted_entries(
    ds: Dataset, reader: IndexReader, rows: pd.DataFrame
) -> tuple[Entry, ...]:
    """Return the entries of *rows* that were extracted again after the run read them.

    Each row's ``consumed_frames_composition`` is compared with the
    ``media_composition`` of the current frames row of the same extraction run
    and entry camera, under
    :func:`~mosaic.core.pipeline.composition.compositions_disagree`. A blank
    side, or no current row, is no drift. A drifted camera drifts its entry.

    Returns:
        The drifted entries, sorted and distinct.
    """
    methods = frozenset(root_subdirectories(ds, "frames"))
    current: dict[tuple[str, str, str, str, str], str] = {}
    read: set[str] = set()
    drifted: set[Entry] = set()
    for record in index_records(rows):
        method = record.get("frames_method", "")
        if method in methods and method not in read:
            read.add(method)
            frame = reader.frame(
                frames_index_path(ds, method), lambda m=method: list_frame_runs(ds, m)
            )
            for row in index_records(frame):
                key = (
                    method,
                    row.get("run_id", ""),
                    row.get("group", ""),
                    row.get("sequence", ""),
                    row.get("camera", ""),
                )
                current[key] = row.get("media_composition", "")
        group, sequence = record.get("group", ""), record.get("sequence", "")
        frames_row = (
            method,
            record.get("frames_run_id", ""),
            group,
            sequence,
            record.get("camera", ""),
        )
        if compositions_disagree(
            record.get("consumed_frames_composition", ""), current.get(frames_row, "")
        ):
            drifted.add((group, sequence))
    return tuple(sorted(drifted))


def prediction_score_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[str]]:
    """Return every score-predictions run, each one artifact that covers its run id.

    A run is present when its row is finished and its outputs exist, and a
    finished row without them is an orphan row. ``scope`` does not narrow it,
    because a score run compares annotated sets with predictions that span
    entries.

    Args:
        ds: The dataset. Never written to.
        scope: What was asked about. Unused.
        reader: The scan's reader, which reads each index once.

    Returns:
        The records, ordered by run id. Empty when the root is undeclared or
        absent.
    """
    _ = scope
    if "score-predictions" not in root_subdirectories(ds, FRAME_PREDICTIONS_ROOT):
        return []
    index_path = prediction_index_path(ds, "score-predictions")
    reader.note(index_path)
    frame = reader.frame(index_path, lambda: score_predictions_index(index_path).read())
    rows: dict[str, dict[str, str]] = {}
    for record in index_records(frame):
        rows.setdefault(record.get("run_id", ""), record)
    run_roots = prediction_kind_root(ds, "score-predictions")
    return [
        run_keyed_record(
            ref=PredictionScoreRef(run_id=run_id),
            name="score-predictions",
            run_id=run_id,
            row=rows[run_id],
            recorded_finished=bool(rows[run_id].get("finished_at", "")),
            complete=score_run_is_complete(ds, rows[run_id]),
            index_path=index_path,
            run_root=run_roots / run_id,
        )
        for run_id in sorted(rows)
    ]


def score_run_is_complete(ds: Dataset, row: Mapping[str, str]) -> bool:
    """Return whether a score-predictions row finished and its outputs exist."""
    stored = row.get("abs_path", "")
    if not row.get("finished_at", "") or not stored:
        return False
    run_root = ds.resolve_path(stored)
    return all((run_root / name).is_file() for name in SCORE_RUN_OUTPUTS)


register_reconcilable_index(FRAME_PREDICTIONS_ROOT, frame_predictions_index)
