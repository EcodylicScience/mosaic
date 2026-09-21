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
    "JoinedExportMissingError",
    "StoreExportMissingError",
    "resolve_entry_input",
    "resolve_tool_input",
    "resolve_tool_inputs",
]


class StoreExportMissingError(FileNotFoundError):
    """An imgstore has no exported video for a subprocess tool to open."""


class JoinedExportMissingError(FileNotFoundError):
    """A multi-clip entry has no joined video for a subprocess tool to open.

    Its own class rather than a reuse of the one above, because the two have
    different remedies -- ``export-store`` and ``export-joined`` -- and a caller
    catching one should not silently swallow the other.
    """


def resolve_tool_inputs(
    ds: "Dataset", item: "TrackerWorkItem", *, kind: str
) -> tuple[Path, ...]:
    """Every path *kind*'s external tool should open for *item*, in order.

    Each clip is resolved independently first: a store becomes its registered
    export, a plain video passes through, and a sequence mixing the two is fine
    here because an export *is* a plain video by the time the tool sees it.

    **A multi-clip entry then resolves to exactly one path: the join of those
    clips.** Not the list. Handing a tool several files was wrong in both
    directions -- the three tools that cannot take a list tracked clip 0 and
    dropped the rest of the recording, and TREx, which can, under-counts every
    file it opens and so lost the tail of each clip, leaving a table whose
    ``frame`` column no longer addressed the video. One file has one
    unambiguous frame index and needs neither tool to be trusted with the
    arrangement. See :mod:`mosaic.core.pipeline.joined_export`.

    A single-clip entry is unchanged, and is the overwhelming majority: it
    resolves to its one file with nothing built and nothing required.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        item: The work item whose source paths are being resolved.
        kind: The tracker's kind, so a failure names the tool the user invoked.

    Raises:
        StoreExportMissingError: If a source is a store with no export
            registered, or with a link pointing at a file that is gone.
        JoinedExportMissingError: If the entry has several clips and no joined
            export has been built for exactly that clip set.
    """
    clips = tuple(
        resolve_entry_input(ds, item.group, item.sequence, source, kind=kind)
        for source in item.video_paths
    )
    if len(clips) < 2:
        return clips
    return (_joined_input(ds, item, kind=kind),)


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


def _joined_input(ds: "Dataset", item: "TrackerWorkItem", *, kind: str) -> Path:
    """The one video holding *item*'s clips, or a refusal naming how to build it.

    Found by the clip set's own ordered composition digest -- the value
    ``item.source_uid`` already computes for the reuse gate -- so this looks for
    the join of *these* clips in *this* order and never for whatever join
    happens to be on disk.

    **The recipe is deliberately not part of the question.** A joined export's
    filename carries its recipe so that two of them cannot overwrite each other,
    but a consumer is asking "give me these clips as one video", and every
    complete join of one clip set answers that: they hold the same frames in the
    same order, which the op verifies before publishing either. Re-deriving the
    name from default parameters instead made every non-default join invisible
    -- a ``reencode`` run wrote one file and the tracker looked for another,
    then reported the join missing on a session that had just been joined.

    Two joins of one clip set are refused rather than chosen between, for the
    reason ``select_variant_rows`` refuses two recipes for one entry: they are
    different inputs, and picking by sort order would make what a tracker read
    -- and so what it published -- depend on a filesystem accident.

    Refused rather than built here. Joining is minutes of I/O over tens of
    gigabytes: it belongs to an op with a ledger entry, a claim and a
    cancellation point, not to a path resolution that a planner also calls.
    """
    from mosaic.core.pipeline.joined_export import JOINED_KIND_DIRECTORY

    source_uid = item.source_uid
    where = (
        f"    mosaic run -m <manifest> --kind export-joined "
        f'--entries "{item.group}:{item.sequence}"'
    )
    if not source_uid:
        message = (
            f"[{kind}] ({item.group}, {item.sequence}) has {item.n_sources} "
            f"clips and at least one carries no content identity, so the join "
            f"of them cannot be addressed. Run 'mosaic reprobe-media --apply' "
            f"to mint one for every clip, then:\n{where}"
        )
        raise JoinedExportMissingError(message)

    root = ds.get_root("media") / JOINED_KIND_DIRECTORY
    found = sorted(root.glob(f"{source_uid}.*.joined.mp4")) if root.is_dir() else []
    if not found:
        message = (
            f"[{kind}] ({item.group}, {item.sequence}) is one recording in "
            f"{item.n_sources} clips. {kind} is handed one video file, and a "
            f"tool that joins clips itself loses frames at every boundary, so "
            f"mosaic joins them first. Build it:\n{where}"
        )
        raise JoinedExportMissingError(message)
    if len(found) > 1:
        listed = "\n".join(f"      {p.name}" for p in found)
        message = (
            f"[{kind}] ({item.group}, {item.sequence}) has {len(found)} joins of "
            f"the same clips, made by different recipes:\n{listed}\n"
            f"They are different inputs, and choosing between them here would "
            f"make what this run read depend on which sorts first. Delete the "
            f"ones you do not want and re-run."
        )
        raise JoinedExportMissingError(message)
    return found[0]


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
