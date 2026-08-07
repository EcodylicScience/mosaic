"""The path a tracker hands to an external tool, which is not always the source.

Three of the four integrated trackers run their tool as a subprocess and give it
a path to open. That works for a video file and fails for an imgstore recording,
which is a *directory* of chunk files: T-Rex converts it to nothing and reports a
missing ``.pv``, and SLEAP and Lightning Pose fail comparably. mosaic's own
readers handle a store natively, so the mismatch is only ever at this boundary --
the moment a path leaves mosaic for a tool that does its own decoding.

:func:`resolve_tool_input` is that boundary. A plain video passes through
untouched; a store resolves to the plain video ``export-store`` wrote for it, or
raises naming the command that would produce one. The in-process Ultralytics
tracker does not call this and must not: it decodes through
``open_frame_reader`` and reads the store directly, which is the better path and
the one that keeps working when no export exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.media.facts_columns import derivative_path_for_target, row_mapping
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.pipeline.store_export import EXPORT_TARGET

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.tracking.common.scope import TrackerWorkItem

__all__ = ["StoreExportMissingError", "resolve_tool_input"]


class StoreExportMissingError(FileNotFoundError):
    """An imgstore has no exported video for a subprocess tool to open."""


def resolve_tool_input(ds: "Dataset", item: "TrackerWorkItem", *, kind: str) -> Path:
    """The path *kind*'s external tool should open for *item*.

    Returns ``item.video_path`` unchanged unless it is an imgstore directory, in
    which case it returns the registered export. The store row is found by path
    rather than by camera: a work item carries no camera (per-camera tracker
    output is not built), and the path is what unambiguously identifies which
    store of a multi-camera sequence this item resolved to.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        item: The work item whose source path is being resolved.
        kind: The tracker's kind, so a failure names the tool the user invoked.

    Raises:
        StoreExportMissingError: If the source is a store with no export
            registered, or with a link pointing at a file that is gone.
    """
    source = item.video_path
    if not is_imgstore(source):
        return source

    export = _registered_export(ds, item)
    if export is None:
        message = (
            f"[{kind}] ({item.group}, {item.sequence}) is an imgstore recording, "
            f"which {kind} cannot open -- it reads a video file, not a store "
            f"directory. Export it first:\n"
            f"    mosaic run -m <manifest> --kind export-store --params "
            f'\'{{"entry": ["{item.group}", "{item.sequence}"]}}\''
        )
        raise StoreExportMissingError(message)
    if not export.is_file():
        message = (
            f"[{kind}] ({item.group}, {item.sequence}) links to an exported video "
            f"at {export}, which does not exist; re-run 'mosaic run --kind "
            f"export-store' to rebuild it"
        )
        raise StoreExportMissingError(message)
    return export


def _registered_export(ds: "Dataset", item: "TrackerWorkItem") -> Path | None:
    """The export linked from *item*'s own store row, or ``None`` if unlinked."""
    matched = ds.match_media_rows(item.group, item.sequence)
    media_root = ds.get_root("media")
    for _, row in matched.iterrows():
        # Through row_mapping rather than indexing the Series: a Series subscript
        # is untyped, and the path is compared, not merely printed.
        cells = row_mapping(row)
        if ds.resolve_path(str(cells["abs_path"])) != item.video_path:
            continue
        return derivative_path_for_target(cells, EXPORT_TARGET, media_root)
    return None
