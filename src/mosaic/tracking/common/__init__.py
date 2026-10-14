"""Machinery every integrated tracker shares.

TREx, SLEAP, Lightning Pose and Ultralytics differ in what they run and what
they emit. They do not differ in how a run is *driven*: resolve the model to a
content digest, mint a run identifier, claim each entry's working directory,
decide per phase whether the recorded marker still proves the work, run what is
stale, record what completed, bridge the output into ``tracks/``, and write a
row. That algorithm lives here once, calling into
:mod:`mosaic.core.pipeline.toolenv` to locate the tool's environment.

This package imports ``core`` and nothing from ``behavior``; ``core`` imports
nothing from here. A tracker's own module keeps what is genuinely its own -- the
argv it builds, the settings that define its identity, the phases it runs, and
the converter it bridges through -- and reaches this package for everything else.

A tracker also declares one
:class:`~mosaic.core.pipeline.tracking_roots.TrackingRoot` row.

This file binds every public name on access. A submodule such as ``index``
then imports without the bridge, the scope builder and the media modules that
they import.
"""

from typing import TYPE_CHECKING, Final

from mosaic.lazy_exports import lazy_export

if TYPE_CHECKING:
    from .bridge import (
        BridgeCounts as BridgeCounts,
        readable_tracks_table as readable_tracks_table,
        frame_counts as frame_counts,
        publish_tracks_table as publish_tracks_table,
        tracks_table_path as tracks_table_path,
    )
    from .entry import (
        AdoptEvidence as AdoptEvidence,
        adopt_completed_directory as adopt_completed_directory,
        clear_outputs as clear_outputs,
        record_phase as record_phase,
        reusable_marker as reusable_marker,
        reusable_output as reusable_output,
    )
    from .index import (
        TrackerRunRowBase as TrackerRunRowBase,
        list_tracker_runs as list_tracker_runs,
        tracker_index as tracker_index,
        tracker_index_path as tracker_index_path,
    )
    from .mint import (
        MintedRun as MintedRun,
        mint_tracker_run as mint_tracker_run,
        tracker_run_root as tracker_run_root,
    )
    from .scope import (
        JoinedSourceMismatchError as JoinedSourceMismatchError,
        TrackerWorkItem as TrackerWorkItem,
        UnresolvedEntry as UnresolvedEntry,
        WorkItems as WorkItems,
        build_work_items as build_work_items,
    )

_MODULE_OF: Final[dict[str, str]] = {
    "BridgeCounts": ".bridge",
    "readable_tracks_table": ".bridge",
    "frame_counts": ".bridge",
    "publish_tracks_table": ".bridge",
    "tracks_table_path": ".bridge",
    "AdoptEvidence": ".entry",
    "adopt_completed_directory": ".entry",
    "clear_outputs": ".entry",
    "record_phase": ".entry",
    "reusable_marker": ".entry",
    "reusable_output": ".entry",
    "TrackerRunRowBase": ".index",
    "list_tracker_runs": ".index",
    "tracker_index": ".index",
    "tracker_index_path": ".index",
    "MintedRun": ".mint",
    "mint_tracker_run": ".mint",
    "tracker_run_root": ".mint",
    "JoinedSourceMismatchError": ".scope",
    "TrackerWorkItem": ".scope",
    "UnresolvedEntry": ".scope",
    "WorkItems": ".scope",
    "build_work_items": ".scope",
}
"""The submodule that defines each public name."""


def __getattr__(name: str) -> object:
    """Import the submodule that defines *name* on first access."""
    return lazy_export(__name__, _MODULE_OF, name)


__all__ = [
    "AdoptEvidence",
    "BridgeCounts",
    "MintedRun",
    "TrackerRunRowBase",
    "JoinedSourceMismatchError",
    "TrackerWorkItem",
    "UnresolvedEntry",
    "WorkItems",
    "adopt_completed_directory",
    "build_work_items",
    "clear_outputs",
    "readable_tracks_table",
    "frame_counts",
    "list_tracker_runs",
    "mint_tracker_run",
    "publish_tracks_table",
    "record_phase",
    "reusable_marker",
    "reusable_output",
    "tracker_index",
    "tracker_index_path",
    "tracker_run_root",
    "tracks_table_path",
]
