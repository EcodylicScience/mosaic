"""Ultralytics multi-object tracking, run in an environment of its own.

Ultralytics is AGPL-3.0, and a mosaic that imported it would be one work with it,
so mosaic spawns a program in an environment the user builds and exchanges JSON
files with it. That makes this tracker like the other three: a ``ToolEnv``, the
same five-step ``MOSAIC_ULTRALYTICS_CONDA_ENV`` / ``MOSAIC_ULTRALYTICS_BIN``
location ladder, and a supervised subprocess per entry. Everything around it --
content-addressed run identifiers, phase markers and reuse, the tracks bridge,
sweeping -- is the shared machinery in :mod:`mosaic.tracking.common`, exactly as
for TREx, SLEAP and Lightning Pose.

The environment and the program it runs live in
:mod:`mosaic.tracking.external`; ``src/mosaic/tracking/external/README.md`` is
the bootstrap, with the license terms attached to that step.
"""

from __future__ import annotations

from mosaic.tracking.ultralytics_track.dataset_runs import (
    UltralyticsIndexRow,
    list_ultralytics_runs,
    run_ultralytics,
)
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from mosaic.tracking.ultralytics_track.run import (
    ULTRALYTICS_ENV,
    UltralyticsError,
    UltralyticsNotFoundError,
    UltralyticsTrackResult,
    UnsupportedTaskError,
    UnsupportedTrackerError,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    TRACKER_NAMES,
    BotsortConfig,
    BytetrackConfig,
    DeepocsortConfig,
    FasttrackConfig,
    OcsortConfig,
    TrackerConfig,
    TrackerConfigError,
    TrackerName,
    TracktrackConfig,
    resolve_tracker_config,
)
from mosaic.tracking.ultralytics_track.version import (
    ULTRALYTICS_KIND,
    ULTRALYTICS_VERSION,
)

__all__ = [
    "TRACKER_NAMES",
    "ULTRALYTICS_ENV",
    "ULTRALYTICS_KIND",
    "ULTRALYTICS_VERSION",
    "BotsortConfig",
    "BytetrackConfig",
    "DeepocsortConfig",
    "FasttrackConfig",
    "OcsortConfig",
    "TrackerConfig",
    "TrackerConfigError",
    "TrackerName",
    "TracktrackConfig",
    "UltralyticsError",
    "UltralyticsIndexRow",
    "UltralyticsNotFoundError",
    "UltralyticsParams",
    "UltralyticsTrackResult",
    "UnsupportedTaskError",
    "UnsupportedTrackerError",
    "list_ultralytics_runs",
    "resolve_tracker_config",
    "run_ultralytics",
]
