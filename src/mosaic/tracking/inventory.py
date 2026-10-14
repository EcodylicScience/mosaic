"""A dataset's inventory with the tracking kinds reported.

``core`` reports the kinds it produces itself. Tracker runs, frame runs, trained
models and prepared datasets register their inventory contributors when their
modules are imported, and ``core`` imports nothing from ``tracking``. This module
imports them, then reads.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.inventory import inventory

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.inventory import ArtifactKind, DatasetInventory
    from mosaic.core.scope import Scope

__all__ = [
    "INVENTORY_CONTRIBUTOR_MODULES",
    "dataset_inventory",
    "register_inventory_contributors",
]

INVENTORY_CONTRIBUTOR_MODULES: Final = (
    "mosaic.tracking.common.index",
    "mosaic.tracking.frame_extraction.dataset_runs",
    "mosaic.tracking.ops.prepared_datasets",
    "mosaic.tracking.ops.train",
)
"""The modules that register an inventory contributor when imported.

Tracker runs, frame runs, prepared datasets and trained models, in that order.
"""


def register_inventory_contributors() -> None:
    """Import every module of :data:`INVENTORY_CONTRIBUTOR_MODULES`.

    A caller that holds an ``InventoryCache`` calls this before its first read.
    """
    for module in INVENTORY_CONTRIBUTOR_MODULES:
        _ = import_module(module)


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
