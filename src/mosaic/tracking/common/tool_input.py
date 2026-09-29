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

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic_media import SOFTWARE_DECODABLE_CODECS
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.transcode import TranscodeError

from mosaic.core.media.facts_columns import derivative_path_for_target, row_mapping
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.pipeline.joined_export import current_join
from mosaic.core.pipeline.store_export import EXPORT_TARGET
from mosaic.core.pipeline.tracking_roots import (
    CONSERVATIVE_DECODER,
    TRACKING_ROOTS,
    ToolDecoder,
)
from mosaic.core.pipeline.variant_source import preprocess_command

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.tracking.common.scope import TrackerWorkItem

__all__ = [
    "StoreExportMissingError",
    "ToolCodecError",
    "refuse_undecodable_codec",
    "resolve_entry_input",
    "resolve_tool_input",
    "resolve_tool_inputs",
]

_ALLOW_CODECS_VAR: Final = "MOSAIC_ALLOW_TOOL_CODECS"
_CODEC_PROBE_TIMEOUT_SECONDS: Final = 120.0


class StoreExportMissingError(FileNotFoundError):
    """An imgstore has no exported video for a subprocess tool to open."""


class ToolCodecError(TranscodeError):
    """A tool is being handed a file its decoder stack may not open.

    Its own class rather than a reuse of the two above, and for the same reason
    they are separate from each other: the remedy differs. Those two say build
    the file; this one says the file exists and is in the wrong codec.
    """


def _stream_codec(path: Path) -> str:
    """*path*'s video codec, from its header.

    A header read, not :func:`~mosaic_media.probe_media` -- that scans every
    packet, which is minutes over a joined session, and the codec is in the
    first few bytes. This runs once per file handed to a tool.

    An unreadable header answers ``""``, which passes. A file a tool cannot open
    at all is the tool's own error to report, with its own message; inventing a
    codec refusal for it here would name the wrong cause.
    """
    try:
        out = run_to_completion(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(path),
            ],
            timeout=_CODEC_PROBE_TIMEOUT_SECONDS,
            action=f"reading the codec of {path.name}",
            error_type=TranscodeError,
        )
    except TranscodeError:
        return ""
    return out.strip().splitlines()[0].strip().lower() if out.strip() else ""


def _allowed_codecs(decoder: ToolDecoder) -> frozenset[str]:
    """What *decoder* may be handed: the baseline, its own extras, and the override.

    ``MOSAIC_ALLOW_TOOL_CODECS`` is a comma-separated list of extra codec names.
    It exists because the refusal is an inference about a decoder mosaic does
    not own: someone who knows their tool environment links ``libdav1d`` is
    right, and should not have to re-encode a corpus to prove it. It widens the
    set and never narrows it, so the variable cannot turn a working run into a
    broken one.
    """
    extra = os.environ.get(_ALLOW_CODECS_VAR, "")
    named = {part.strip().lower() for part in extra.split(",") if part.strip()}
    return SOFTWARE_DECODABLE_CODECS | decoder.also_reads | named


def refuse_undecodable_codec(
    ds: "Dataset",
    path: Path,
    *,
    kind: str,
    group: str,
    sequence: str,
    variant: str = "",
) -> None:
    """Raise unless *kind*'s decoder can be expected to open *path*.

    **Per tool, because the answer differs per tool.** TREx links its own
    environment's libavcodec and Ultralytics reads through mosaic-media's PyAV;
    both carry ``libdav1d`` and read AV1. SLEAP decodes with OpenCV, whose
    manylinux wheel carries no software AV1 decoder at all, and Lightning Pose
    decodes through NVDEC, which reads AV1 only on compute capability 8.6 or
    newer. One global answer would refuse the first two in order to refuse the
    second two. Each tool's declaration lives on its
    :class:`~mosaic.core.pipeline.tracking_roots.TrackingRoot`.

    **Checked on what is handed over, never on what it was resolved from.** A
    clip that is about to be joined away may be in any codec: the tool never
    opens it. Refusing there would block a run that would have worked.

    The failure this prevents is silent, which is why it is worth a probe. A
    reader with no decoder for the file returns zero frames and its caller exits
    0, so SLEAP wrote a `.slp` with no labeled frames, the bridge read a valid
    empty table, and the run was recorded as a success that produced nothing --
    a result indistinguishable, at every gate mosaic has, from a video with no
    animals in it.

    A kind with no registered root gets the conservative declaration, which
    assumes nothing beyond the universal baseline.

    *variant* names the media variant that *path* is the file of, when it is one.
    A variant's codec is chosen when it is made. The refusal then also names making
    the variant again in H.264, from the recipe that *ds* recorded for it.
    """
    root = TRACKING_ROOTS.get(kind)
    decoder = root.decoder if root is not None else CONSERVATIVE_DECODER
    codec = _stream_codec(path)
    if not codec or codec in _allowed_codecs(decoder):
        return
    remedy = f"\n    {decoder.remedy}." if decoder.remedy else ""
    remake = (
        f"\n    Or make the media variant in a codec that {kind} reads. Run the "
        f'recipe of {variant} with "codec" set to "h264" in a copy of it, and name '
        f"the variant that it writes in media. The recipe of {variant} is run "
        f"with:\n"
        f"{preprocess_command(ds, variant, [(group, sequence)])}"
        if variant
        else ""
    )
    message = (
        f"[{kind}] ({group}, {sequence}) resolves to {path.name}, which is "
        f"{codec}. {kind} decodes with {decoder.stack}, which cannot be "
        f"expected to open {codec}. A tool that cannot decode returns zero "
        f"frames and exits 0, so this would otherwise be recorded as a run "
        f"that succeeded and found nothing.{remedy}{remake}\n"
        f"    To say this environment does handle it, set "
        f"{_ALLOW_CODECS_VAR}={codec}."
    )
    raise ToolCodecError(message)


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

    An item that reads a media variant resolves to the variant file. It is a plain
    video that mosaic wrote, one file for the whole entry. An export and a join
    therefore do not apply to it.

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
    handed: tuple[Path, ...]
    if item.variant is not None:
        handed = (item.variant.path,)
    else:
        clips = tuple(
            resolve_entry_input(ds, item.group, item.sequence, source, kind=kind)
            for source in item.video_paths
        )
        # Several clips resolve to their join and the clips themselves are
        # dropped, so the codec gate runs over what is returned rather than
        # inside the comprehension above: an AV1 derivative about to be
        # re-encoded into a uniform join is not a file any tool will open, and
        # refusing it here would block a run that would have worked.
        handed = clips if len(clips) < 2 else (_joined_input(ds, item, kind=kind),)
    for target in handed:
        refuse_undecodable_codec(
            ds,
            target,
            kind=kind,
            group=item.group,
            sequence=item.sequence,
            variant=item.media,
        )
    return handed


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

    The store row is found by path rather than by camera, because the path
    alone identifies the store within a multi-camera sequence.
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

    The lookup is :func:`~mosaic.core.pipeline.joined_export.current_join`'s,
    keyed by ``item.source_uid`` -- the value the reuse gate already computes.
    """
    why = (
        f"{kind} is handed one video file, and a tool that joins clips itself "
        f"loses frames at every boundary, so mosaic joins them first."
    )
    return current_join(
        ds,
        item.group,
        item.sequence,
        item.source_uid,
        item.n_sources,
        asker=kind,
        why=why,
    )


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
