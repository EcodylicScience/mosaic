"""A trained model registered the way a training op registers one, without training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.tracking.ops.train import TrainedModelIndexRow, trained_model_index

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["register_trained_model"]


def register_trained_model(
    ds: Dataset, kind: str, run_id: str, weights: Path, *, directory: Path | None = None
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
                base_run_id="",
                best_model_path=ds.relative_to_root(weights),
                metrics_path="",
                n_epochs=1,
                status="finished",
                artifact_shape="file" if directory is None else "directory",
                artifact_path=ds.relative_to_root(artifact),
                model_type="",
                abs_path=Path(ds.relative_to_root(model_run_root(ds, kind, run_id))),
            )
        ]
    )
    index.mark_finished(run_id)
