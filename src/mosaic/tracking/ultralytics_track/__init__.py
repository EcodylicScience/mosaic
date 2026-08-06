"""Ultralytics multi-object tracking, run in mosaic's own process.

The one integrated tracker with no second environment: Ultralytics is already a
mosaic dependency (the ``pose`` extra), so there is no ``ToolEnv``, no
``MOSAIC_*_BIN`` ladder and no subprocess. Everything else -- content-addressed
run identifiers, phase markers and reuse, the tracks bridge, sweeping -- is the
shared machinery in :mod:`mosaic.tracking.common`, exactly as for TREx, SLEAP and
Lightning Pose.
"""

from __future__ import annotations

from mosaic.tracking.ultralytics_track.dataset_runs import (
    UltralyticsIndexRow,
    list_ultralytics_runs,
    run_ultralytics,
)
from mosaic.tracking.ultralytics_track.run import (
    UltralyticsNotFoundError,
    UltralyticsTrackResult,
    UnsupportedTaskError,
    UnsupportedTrackerError,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TRACKER_NAMES,
    TrackerConfigError,
    TrackerName,
    resolve_tracker_config,
)
from mosaic.tracking.ultralytics_track.version import (
    ULTRALYTICS_KIND,
    ULTRALYTICS_VERSION,
)

__all__ = [
    "TRACKER_NAMES",
    "ULTRALYTICS_KIND",
    "ULTRALYTICS_VERSION",
    "TrackerConfigError",
    "TrackerName",
    "UltralyticsIndexRow",
    "UltralyticsNotFoundError",
    "UltralyticsTrackResult",
    "UnsupportedTaskError",
    "UnsupportedTrackerError",
    "list_ultralytics_runs",
    "resolve_tracker_config",
    "run_ultralytics",
]
