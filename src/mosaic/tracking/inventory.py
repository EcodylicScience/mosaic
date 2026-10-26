"""A dataset's inventory with the tracking kinds reported.

``core`` reports the kinds it produces itself, and imports nothing from
``tracking``. Tracker runs, frame runs, frame predictions, prediction scores,
frame selections, prepared datasets and trained models are reported through the
contributor registry, and :func:`register_inventory_contributors` registers all
seven. Each record builder lives beside the index that it reads. Its module
leaves the tracker subpackages and the trainers unimported, and loads only the
media modules that ``mosaic.core.pipeline.inventory`` loads itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.inventory.contributors import register_inventory_contributor
from mosaic.tracking.common.index import tracker_run_records
from mosaic.tracking.frame_extraction.frame_runs import frame_run_records
from mosaic.tracking.frame_extraction.selection_index import frame_selection_records
from mosaic.tracking.frame_predictions.index import (
    frame_prediction_records,
    prediction_score_records,
)
from mosaic.tracking.ops.prepared_datasets import prepared_dataset_records
from mosaic.tracking.ops.trained_models import trained_model_records

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.inventory import ArtifactKind, DatasetInventory
    from mosaic.core.scope import Scope

__all__ = [
    "dataset_inventory",
    "register_inventory_contributors",
]


def register_inventory_contributors() -> None:
    """Register the contributor of every tracking kind.

    A caller that holds an ``InventoryCache`` calls this before its first read.
    Registering again replaces each contributor with itself.
    """
    register_inventory_contributor("tracker-run", tracker_run_records)
    register_inventory_contributor("frame-run", frame_run_records)
    register_inventory_contributor("frame-predictions", frame_prediction_records)
    register_inventory_contributor("prediction-score", prediction_score_records)
    register_inventory_contributor("frame-selection", frame_selection_records)
    register_inventory_contributor("prepared-dataset", prepared_dataset_records)
    register_inventory_contributor("trained-model", trained_model_records)


def dataset_inventory(
    ds: Dataset,
    *,
    kinds: Iterable[ArtifactKind] | None = None,
    scope: Scope | None = None,
    tracks_run_id: str | None = None,
) -> DatasetInventory:
    """Return what *ds* holds, every tracking kind included.

    Args:
        ds: The dataset to read. Never written to.
        kinds: Which artifact kinds to report. ``None`` reports every kind.
        scope: Narrow coverage to what this selector names.
        tracks_run_id: Which tracks variant defines the entry universe.

    Returns:
        The inventory that :func:`~mosaic.core.pipeline.inventory.inventory`
        reads once the contributors are registered.
    """
    register_inventory_contributors()
    return inventory(ds, kinds=kinds, scope=scope, tracks_run_id=tracks_run_id)
