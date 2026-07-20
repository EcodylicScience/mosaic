"""Tracking utilities — pose estimation, frame extraction, TREx, schema re-exports.

All long-running tracking operations are registered ``Op``\\s discoverable and
runnable through the shared registry in :mod:`mosaic.core.pipeline.ops`.
Importing this package registers the built-in ops (extract / train / infer).
"""
from mosaic.core.schema import TrackSchema, TRACK_SCHEMAS, register_track_schema, ensure_track_schema
from mosaic.tracking.model_refs import resolve_model
from mosaic.tracking.frame_extraction import (
    extract_frames,
    get_frame_manifests,
    get_frame_paths,
    list_frame_runs,
)
from mosaic.tracking.trex import list_trex_runs, run_trex

# Explicit re-export: importing the ops subpackage runs its registration side
# effects (kept import-light; heavy backends load lazily inside each op's run()).
from mosaic.tracking import ops as ops


def register_ops() -> None:
    """Ensure the built-in tracking ops are registered in the shared registry.

    Registration is a side effect of importing ``mosaic.tracking``: this package
    imports its op subpackages, whose ``@register_op`` decorators run on import.
    A caller that otherwise holds only the generic registry from
    ``mosaic.core.pipeline.ops`` -- the CLI -- imports and calls this so the
    tracking kinds are present before it enumerates or runs them.
    """


__all__ = [
    "TrackSchema",
    "TRACK_SCHEMAS",
    "register_track_schema",
    "ensure_track_schema",
    "register_ops",
    "resolve_model",
    "extract_frames",
    "get_frame_manifests",
    "get_frame_paths",
    "list_frame_runs",
    "list_trex_runs",
    "run_trex",
]
