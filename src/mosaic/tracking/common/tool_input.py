"""The path a tracker hands to an external tool, which is not always the source.

All four integrated trackers run their tool as a subprocess and give it a path to
open, so all four resolve through here. That works for a video file and fails for
an imgstore recording, which is a *directory* of chunk files: T-Rex converts it
to nothing and reports a missing ``.pv``, and SLEAP, Lightning Pose and
Ultralytics fail comparably. mosaic's own readers handle a store natively, so the
mismatch is only ever at this boundary -- the moment a path leaves mosaic for a
tool that does its own decoding.

:func:`resolve_tool_input` is that boundary. A plain video passes through
untouched; a store resolves to the plain video ``export-store`` wrote for it, or
raises naming the command that would produce one.

Ultralytics used to be the exception, decoding through ``open_frame_reader`` and
reading a store directly -- genuinely the better path, and one that worked with
no export on disk. That capability is gone: Ultralytics is AGPL-3.0, so it runs
in an environment of its own and opens a path like every other tool, and tracking
a store now costs an ``export-store`` run and a copy of the pixels first. Pose and
point *inference* pay the same cost for the same reason. The heatmap localizer
does not: it is mosaic's own PyTorch and still reads a store natively.
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

__all__ = [
    "StoreExportMissingError",
    "resolve_entry_input",
    "resolve_tool_input",
    "resolve_tool_inputs",
]


class StoreExportMissingError(FileNotFoundError):
    """An imgstore has no exported video for a subprocess tool to open."""


def resolve_tool_inputs(
    ds: "Dataset", item: "TrackerWorkItem", *, kind: str
) -> tuple[Path, ...]:
    """Every path *kind*'s external tool should open for *item*, in order.

    One element per clip, so a tool that reads a session as one video gets the
    whole arrangement. Each clip is resolved independently: a store becomes its
    registered export, a plain video passes through, and a sequence mixing the
    two is fine here because an export *is* a plain video by the time the tool
    sees it.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        item: The work item whose source paths are being resolved.
        kind: The tracker's kind, so a failure names the tool the user invoked.

    Raises:
        StoreExportMissingError: If a source is a store with no export
            registered, or with a link pointing at a file that is gone.
    """
    return tuple(
        resolve_entry_input(ds, item.group, item.sequence, source, kind=kind)
        for source in item.video_paths
    )


def resolve_tool_input(ds: "Dataset", item: "TrackerWorkItem", *, kind: str) -> Path:
    """The path *kind*'s external tool should open for *item*'s first clip.

    The single-source view of :func:`resolve_tool_inputs`, for the trackers that
    read one video file. One rule, two views -- a second implementation is how
    the two would come to disagree about what a store resolves to.
    """
    return resolve_tool_inputs(ds, item, kind=kind)[0]


def resolve_entry_input(
    ds: "Dataset", group: str, sequence: str, source: Path, *, kind: str
) -> Path:
    """*source* itself, or the export registered for it when it is a store.

    The entry named by *group* and *sequence* rather than a
    :class:`~mosaic.tracking.common.scope.TrackerWorkItem`, because the inference
    ops reach this boundary too and build no work items: they walk a media scope
    directly. Those two names are all a work item ever supplied here.

    The store row is found by path rather than by camera: a work item carries no
    camera (per-camera tracker output is not built), and the path is what
    unambiguously identifies which store of a multi-camera sequence this is.
    """
    if not is_imgstore(source):
        return source

    export = _registered_export(ds, group, sequence, source)
    if export is None:
        message = (
            f"[{kind}] ({group}, {sequence}) is an imgstore recording, "
            f"which {kind} cannot open -- it reads a video file, not a store "
            f"directory. Export it first:\n"
            f"    mosaic run -m <manifest> --kind export-store --params "
            f'\'{{"entry": ["{group}", "{sequence}"]}}\''
        )
        raise StoreExportMissingError(message)
    if not export.is_file():
        message = (
            f"[{kind}] ({group}, {sequence}) links to an exported video "
            f"at {export}, which does not exist; re-run 'mosaic run --kind "
            f"export-store' to rebuild it"
        )
        raise StoreExportMissingError(message)
    return export


def _registered_export(
    ds: "Dataset", group: str, sequence: str, source: Path
) -> Path | None:
    """The export linked from *source*'s own store row, or ``None`` if unlinked."""
    matched = ds.match_media_rows(group, sequence)
    media_root = ds.get_root("media")
    for _, row in matched.iterrows():
        # Through row_mapping rather than indexing the Series: a Series subscript
        # is untyped, and the path is compared, not merely printed.
        cells = row_mapping(row)
        if ds.resolve_path(str(cells["abs_path"])) != source:
            continue
        return derivative_path_for_target(cells, EXPORT_TARGET, media_root)
    return None
