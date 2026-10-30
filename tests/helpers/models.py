"""Rows of the ``models/`` indexes, written without running the ops.

A trained model is registered the way a training op registers one, and a
prepared-data index is rewritten the way an older mosaic wrote it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.tracking.ops.trained_models import TrainedModelIndexRow, trained_model_index
from tests.helpers.sleap import write_sleap_model

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "register_sleap_model",
    "register_trained_model",
    "rewrite_as_older_prepared_index",
]

_NEWEST_PREPARED_COLUMNS: Final = (
    "held_out_sets",
    "n_held_out",
    "keypoint_schema",
    "track_by",
)
"""The columns that the newest rows of a prepared-data index added."""


def register_trained_model(
    ds: Dataset,
    kind: str,
    run_id: str,
    weights: Path,
    *,
    directory: Path | None = None,
    data_path: Path | None = None,
    base_run_id: str = "",
) -> None:
    """Append a finished row for *run_id* to ``models/<kind>/index.csv``.

    The caller writes the artifact under ``models/<kind>/<run_id>/`` first.

    Args:
        ds: The dataset whose model index receives the row.
        kind: The training op kind that would have written it.
        run_id: The training run identifier.
        weights: The weights file.
        directory: The model directory holding *weights*, for a kind whose
            artifact is a directory. ``None`` registers the file itself.
        data_path: What the run trained on, or ``None`` for nothing recorded.
        base_run_id: The model that the run fine-tuned, or ``""`` for none.
    """
    artifact = directory if directory is not None else weights
    index = trained_model_index(model_index_path(ds, kind))
    index.ensure()
    index.append(
        [
            TrainedModelIndexRow(
                run_id=run_id,
                kind=kind,
                base_model="",
                base_run_id=base_run_id,
                best_model_path=ds.relative_to_root(weights),
                metrics_path="",
                n_epochs=1,
                status="finished",
                artifact_shape="file" if directory is None else "directory",
                artifact_path=ds.relative_to_root(artifact),
                model_type="",
                data_path="" if data_path is None else ds.relative_to_root(data_path),
                abs_path=Path(ds.relative_to_root(model_run_root(ds, kind, run_id))),
            )
        ]
    )
    index.mark_finished(run_id)


def register_sleap_model(
    ds: Dataset,
    run_id: str,
    weights: bytes = b"weights",
    *,
    head: str = "",
    data_path: Path | None = None,
) -> Path:
    """Write and register one finished ``train-sleap`` model directory.

    Args:
        ds: The dataset whose model index receives the row.
        run_id: The training run identifier.
        weights: The checkpoint's bytes, whose digest is the model's identity.
        head: The head that the directory's training config selects, or ``""``
            for a directory with no training config.
        data_path: What the run trained on, or ``None`` for nothing recorded.

    Returns:
        The model directory, which holds ``best.ckpt``.
    """
    directory = write_sleap_model(
        model_run_root(ds, "train-sleap", run_id) / "model", weights, head=head
    )
    register_trained_model(
        ds,
        "train-sleap",
        run_id,
        directory / "best.ckpt",
        directory=directory,
        data_path=data_path,
    )
    return directory


def rewrite_as_older_prepared_index(ds: Dataset) -> None:
    """Drop the newest columns from the ``prepare-training-data`` index of *ds*.

    Its rows then read as rows written before those columns.
    """
    path = model_index_path(ds, "prepare-training-data")
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    written.drop(columns=list(_NEWEST_PREPARED_COLUMNS)).to_csv(path, index=False)
