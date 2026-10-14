"""Registration of every tracking op, tracker index and inventory contributor.

The tracking ops register when their modules are imported, and so do the tracker
run indexes and the frame readers that ``core`` looks up by kind. ``mosaic.tracking``
imports none of those modules, and :func:`register_ops` imports all of them.
``tests/tracking/test_registration.py`` fails when a module under
``mosaic.tracking`` registers something at import and :data:`REGISTERING_MODULES`
leaves it out.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

__all__ = ["REGISTERING_MODULES", "register_ops"]

REGISTERING_MODULES: Final = (
    "mosaic.tracking.frame_extraction.dataset_runs",
    "mosaic.tracking.litpose.dataset_runs",
    "mosaic.tracking.ops.convert",
    "mosaic.tracking.ops.infer",
    "mosaic.tracking.ops.litpose",
    "mosaic.tracking.ops.prepare",
    "mosaic.tracking.ops.resample",
    "mosaic.tracking.ops.sleap",
    "mosaic.tracking.ops.train",
    "mosaic.tracking.ops.train_litpose",
    "mosaic.tracking.ops.train_sleap",
    "mosaic.tracking.ops.trex",
    "mosaic.tracking.ops.ultralytics",
    "mosaic.tracking.sleap.dataset_runs",
    "mosaic.tracking.trex.conversion_cache",
    "mosaic.tracking.trex.dataset_runs",
    "mosaic.tracking.ultralytics_track.dataset_runs",
)
"""The modules under ``mosaic.tracking`` that register something when imported."""


def register_ops() -> None:
    """Register every built-in op, tracker index and tracking inventory contributor.

    The ops under ``core`` register through
    :func:`~mosaic.core.pipeline.ops.ensure_core_ops_registered`, the modules of
    :data:`REGISTERING_MODULES` on import, and the inventory contributors through
    :func:`~mosaic.tracking.inventory.register_inventory_contributors`. A caller
    that imports only the generic registry from ``mosaic.core.pipeline.ops``,
    such as the CLI, calls this before it enumerates or runs an op.
    """
    from mosaic.core.pipeline.ops import ensure_core_ops_registered
    from mosaic.tracking.inventory import register_inventory_contributors

    ensure_core_ops_registered()
    for module in REGISTERING_MODULES:
        _ = import_module(module)
    register_inventory_contributors()
