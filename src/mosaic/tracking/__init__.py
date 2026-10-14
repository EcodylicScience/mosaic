"""Tracking utilities — pose estimation, frame extraction, TREx, schema re-exports.

All long-running tracking operations are registered ``Op``\\s discoverable and
runnable through the shared registry in :mod:`mosaic.core.pipeline.ops`.
Importing this package registers the tracking ops (extract / train / infer), and
:func:`register_ops` registers every built-in op, the media ops under ``core``
included.
"""

from mosaic.core.pipeline.ops import ensure_core_ops_registered
from mosaic.core.schema import (
    TrackSchema,
    TRACK_SCHEMAS,
    register_track_schema,
    ensure_track_schema,
)
from mosaic.tracking.model_refs import resolve_model, resolve_model_set
from mosaic.tracking.frame_extraction import (
    extract_frames,
    get_frame_manifests,
    get_frame_paths,
    list_frame_runs,
)
from mosaic.tracking.litpose import list_litpose_runs, run_litpose
from mosaic.tracking.sleap import list_sleap_runs, run_sleap
from mosaic.tracking.trex import list_trex_runs, run_trex
from mosaic.tracking.ultralytics_track import (
    list_ultralytics_runs,
    run_ultralytics,
)

# Explicit re-export: importing the ops subpackage runs its registration side
# effects. This is not an import-light path. The four tracker packages above are
# imported at module top, so pandas, numpy, cv2 and h5py are loaded well before
# this line is reached.
from mosaic.tracking import ops as ops
from mosaic.tracking.ops.delete_model import delete_trained_model
from mosaic.tracking.inventory import (
    dataset_inventory,
    register_inventory_contributors,
)


def register_ops() -> None:
    """Ensure every built-in op is registered in the shared registry.

    Registration of the tracking ops is a side effect of importing
    ``mosaic.tracking``: this package imports its op subpackages, whose
    ``@register_op`` decorators run on import. The ops under ``core`` register
    through :func:`~mosaic.core.pipeline.ops.ensure_core_ops_registered`. A
    caller that imports only the generic registry from
    ``mosaic.core.pipeline.ops``, such as the CLI, calls this before it
    enumerates or runs an op.
    """
    ensure_core_ops_registered()


__all__ = [
    "TrackSchema",
    "TRACK_SCHEMAS",
    "register_track_schema",
    "ensure_track_schema",
    "register_ops",
    "dataset_inventory",
    "delete_trained_model",
    "register_inventory_contributors",
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
