"""Tracking utilities: pose estimation, frame extraction, TREx, schema re-exports.

This file binds every public name on access. Python executes it before any
``mosaic.tracking.*`` submodule, and every consumer of a submodule pays for each
name bound at module scope. The tracker subpackages import pandas, cv2 and av.
``tests/meta/test_tracking_lazy_exports.py`` checks that importing the package
loads none of them.

Each name is also declared under ``TYPE_CHECKING`` for type checkers and editors.

The tracking ops, the tracker indexes and the inventory contributors register
when their modules are imported, and this file imports none of them.
:func:`~mosaic.tracking.registration.register_ops` imports every module that
registers something.
"""

from typing import TYPE_CHECKING, Final

from mosaic.lazy_exports import lazy_export

if TYPE_CHECKING:
    from mosaic.core.schema import (
        TRACK_SCHEMAS as TRACK_SCHEMAS,
        TrackSchema as TrackSchema,
        ensure_track_schema as ensure_track_schema,
        register_track_schema as register_track_schema,
    )
    from mosaic.tracking.frame_extraction import (
        extract_frames as extract_frames,
        get_frame_manifests as get_frame_manifests,
        get_frame_paths as get_frame_paths,
        list_frame_runs as list_frame_runs,
    )
    from mosaic.tracking.inventory import (
        dataset_inventory as dataset_inventory,
        register_inventory_contributors as register_inventory_contributors,
    )
    from mosaic.tracking.litpose import (
        list_litpose_runs as list_litpose_runs,
        run_litpose as run_litpose,
    )
    from mosaic.tracking.model_refs import (
        resolve_model as resolve_model,
        resolve_model_set as resolve_model_set,
    )
    from mosaic.tracking.ops.delete_model import (
        delete_trained_model as delete_trained_model,
    )
    from mosaic.tracking.prerequisites import (
        EntryRefusal as EntryRefusal,
        PrerequisiteKind as PrerequisiteKind,
        PrerequisitePlan as PrerequisitePlan,
        PrerequisiteRun as PrerequisiteRun,
        RefusalCause as RefusalCause,
        indexed_entries as indexed_entries,
        plan_prerequisites as plan_prerequisites,
    )
    from mosaic.tracking.registration import register_ops as register_ops
    from mosaic.tracking.sleap import (
        list_sleap_runs as list_sleap_runs,
        run_sleap as run_sleap,
    )
    from mosaic.tracking.trex import (
        list_trex_runs as list_trex_runs,
        run_trex as run_trex,
    )
    from mosaic.tracking.ultralytics_track import (
        list_ultralytics_runs as list_ultralytics_runs,
        run_ultralytics as run_ultralytics,
    )

_MODULE_OF: Final[dict[str, str]] = {
    "TRACK_SCHEMAS": "mosaic.core.schema",
    "TrackSchema": "mosaic.core.schema",
    "ensure_track_schema": "mosaic.core.schema",
    "register_track_schema": "mosaic.core.schema",
    "extract_frames": "mosaic.tracking.frame_extraction",
    "get_frame_manifests": "mosaic.tracking.frame_extraction",
    "get_frame_paths": "mosaic.tracking.frame_extraction",
    "list_frame_runs": "mosaic.tracking.frame_extraction",
    "dataset_inventory": "mosaic.tracking.inventory",
    "register_inventory_contributors": "mosaic.tracking.inventory",
    "list_litpose_runs": "mosaic.tracking.litpose",
    "run_litpose": "mosaic.tracking.litpose",
    "resolve_model": "mosaic.tracking.model_refs",
    "resolve_model_set": "mosaic.tracking.model_refs",
    "delete_trained_model": "mosaic.tracking.ops.delete_model",
    "EntryRefusal": "mosaic.tracking.prerequisites",
    "PrerequisiteKind": "mosaic.tracking.prerequisites",
    "PrerequisitePlan": "mosaic.tracking.prerequisites",
    "PrerequisiteRun": "mosaic.tracking.prerequisites",
    "RefusalCause": "mosaic.tracking.prerequisites",
    "indexed_entries": "mosaic.tracking.prerequisites",
    "plan_prerequisites": "mosaic.tracking.prerequisites",
    "register_ops": "mosaic.tracking.registration",
    "list_sleap_runs": "mosaic.tracking.sleap",
    "run_sleap": "mosaic.tracking.sleap",
    "list_trex_runs": "mosaic.tracking.trex",
    "run_trex": "mosaic.tracking.trex",
    "list_ultralytics_runs": "mosaic.tracking.ultralytics_track",
    "run_ultralytics": "mosaic.tracking.ultralytics_track",
}
"""The module that defines each public name."""


def __getattr__(name: str) -> object:
    """Import the module that defines *name* on first access."""
    return lazy_export(__name__, _MODULE_OF, name)


__all__ = [
    "TrackSchema",
    "TRACK_SCHEMAS",
    "register_track_schema",
    "ensure_track_schema",
    "register_ops",
    "dataset_inventory",
    "delete_trained_model",
    "register_inventory_contributors",
    "EntryRefusal",
    "PrerequisiteKind",
    "PrerequisitePlan",
    "PrerequisiteRun",
    "RefusalCause",
    "indexed_entries",
    "plan_prerequisites",
    "resolve_model",
    "resolve_model_set",
    "extract_frames",
    "get_frame_manifests",
    "get_frame_paths",
    "list_frame_runs",
    "list_litpose_runs",
    "list_sleap_runs",
    "list_trex_runs",
    "list_ultralytics_runs",
    "run_litpose",
    "run_sleap",
    "run_trex",
    "run_ultralytics",
]
