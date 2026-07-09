"""Tracking utilities — pose estimation, frame extraction, TREx, schema re-exports.

All long-running tracking operations are registered ``TrackingOp``\\s discoverable
and runnable through the registry (:func:`run_tracking_op`, :data:`TRACKING_OPS`).
Importing this package registers the built-in ops (extract / train / infer).
"""
from mosaic.core.schema import TrackSchema, TRACK_SCHEMAS, register_track_schema, ensure_track_schema
from mosaic.tracking.registry import (
    TRACKING_OPS,
    TrackingOp,
    describe_tracking_op,
    list_tracking_ops,
    register_tracking_op,
    resolve_model,
    run_tracking_op,
)
from mosaic.tracking.frame_extraction import (
    extract_frames,
    get_frame_manifests,
    get_frame_paths,
    list_frame_runs,
)
from mosaic.tracking.trex import list_trex_runs, run_trex

# Import op modules for their registration side effects (kept import-light; heavy
# backends load lazily inside each op's run()).
from mosaic.tracking import ops as _ops  # noqa: F401

__all__ = [
    "TrackSchema",
    "TRACK_SCHEMAS",
    "register_track_schema",
    "ensure_track_schema",
    "TRACKING_OPS",
    "TrackingOp",
    "describe_tracking_op",
    "list_tracking_ops",
    "register_tracking_op",
    "resolve_model",
    "run_tracking_op",
    "extract_frames",
    "get_frame_manifests",
    "get_frame_paths",
    "list_frame_runs",
    "list_trex_runs",
    "run_trex",
]
