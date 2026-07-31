"""Machinery every integrated tracker shares.

TREx, SLEAP and Lightning Pose differ in what they run and what they emit. They
do not differ in how a run is *driven*: locate the tool in its own environment,
resolve the model to a content digest, mint a run identifier, claim each entry's
working directory, decide per phase whether the recorded marker still proves the
work, run what is stale, record what completed, bridge the output into
``tracks/``, and write a row. That algorithm lives here once.

This package imports ``core`` and nothing from ``behavior``; ``core`` imports
nothing from here. A tracker's own module keeps what is genuinely its own -- the
argv it builds, the settings that define its identity, the phases it runs, and
the converter it bridges through -- and reaches this package for everything else.
"""

from __future__ import annotations

from mosaic.tracking.common.bridge import (
    BridgeCounts,
    existing_counts,
    frame_counts,
    publish_tracks_table,
    tracks_table_path,
)
from mosaic.tracking.common.entry import (
    INFLIGHT_REFRESH_SECONDS,
    AdoptEvidence,
    adopt_completed_directory,
    claim,
    clear_outputs,
    open_entry,
    phase_activity,
    record_phase,
    release_entry,
    reusable_marker,
    reusable_output,
)
from mosaic.tracking.common.index import (
    TrackerRunRowBase,
    list_tracker_runs,
    tracker_index,
    tracker_index_path,
)
from mosaic.tracking.common.mint import MintedRun, mint_tracker_run, tracker_run_root
from mosaic.tracking.common.scope import TrackerWorkItem, build_work_items
from mosaic.tracking.common.toolenv import (
    BinMode,
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    conda_invocation,
    subprocess_env,
    tool_invocation,
)

__all__ = [
    "INFLIGHT_REFRESH_SECONDS",
    "AdoptEvidence",
    "BinMode",
    "BridgeCounts",
    "MintedRun",
    "ToolEnv",
    "ToolExitError",
    "ToolNotFoundError",
    "TrackerRunRowBase",
    "TrackerWorkItem",
    "adopt_completed_directory",
    "build_work_items",
    "claim",
    "clear_outputs",
    "conda_invocation",
    "existing_counts",
    "frame_counts",
    "list_tracker_runs",
    "mint_tracker_run",
    "open_entry",
    "phase_activity",
    "publish_tracks_table",
    "record_phase",
    "release_entry",
    "reusable_marker",
    "reusable_output",
    "subprocess_env",
    "tool_invocation",
    "tracker_index",
    "tracker_index_path",
    "tracker_run_root",
    "tracks_table_path",
]
