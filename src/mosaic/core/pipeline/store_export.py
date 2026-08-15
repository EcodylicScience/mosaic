"""Exporting an imgstore recording as a plain video an external tool can open.

mosaic reads a store natively: ``open_frame_reader`` returns an
:class:`~mosaic.core.media.imgstore_io.ImgStoreFrameReader` for a store
directory, so every in-process consumer -- features, frame extraction, the
Ultralytics tracker -- works against a store unchanged. An external binary
cannot. T-Rex, SLEAP and Lightning Pose are handed a path and open it
themselves, and a store is a *directory* of chunk files that none of them
understands.

``StoreExportOp`` (``kind="export-store"``, ``domain="media"``) closes that gap
by writing one plain constant-rate mp4 per store, holding the store's frames in
its own contiguous ``frame_index`` order. Exported frame *i* is store frame *i*,
which is what lets a table produced from the export be compared frame for frame
with one produced from the store.

**The export does not change what mosaic itself reads.**
:meth:`Dataset.route_media_row` routes to a derivative only when a row's verdict
says ``analysis_transcode="required"``, and a store's verdict says nothing of the
kind -- a store needs no transcode negotiation, so both its verdict axes are
null. Registering an export therefore leaves routing inert: mosaic keeps reading
the store, and only a caller that explicitly asks for a plain file (see
:func:`mosaic.tracking.common.tool_input.resolve_tool_input`) follows the link.
Marking a store "required" instead would divert every consumer onto the mp4 and
retire the native read path by accident.

Registration is otherwise exactly a transcode's, and deliberately shares its
machinery: the derivative lands in the same ``transcode`` kind directory under
``media``, named ``<video_uuid>.<recipe_hash>.analysis.mp4``, with a back-link
row in the ``media`` index and a forward link on the store's ``media_raw`` row.
Sharing that layout is what makes ``mosaic prune-media`` reach an orphaned export
without knowing this op exists.

An entry's cameras each export separately -- a store per camera, a
``video_uuid`` per store, so the recipe-addressed filenames never collide.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import pandas as pd
from mosaic_media import CHROME_149, derive, probe_media
from mosaic_media.transcode import ANALYSIS_ENCODING, Target, TranscodeError

from mosaic.core.media.facts_columns import (
    derivative_cell,
    media_row_uuid,
    row_mapping,
    series_facts_or_none,
)
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.media.video_io import FFmpegVideoWriter, open_frame_reader
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.transcode import (
    TRANSCODE_KIND_DIRECTORY,
    relative_to_anchor,
    set_back_link,
    set_forward_link,
)
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.media_probe_config import media_thresholds

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

EXPORT_TARGET: Final[Target] = "analysis"
"""Which forward-link column an export claims.

An export exists to be read frame by frame by an analysis tool, which is the
analysis target's whole meaning. It never claims the playback link: a store has
no playback consumer, and writing both would make a pruner keep two copies of one
recipe.
"""

_EXPORT_PRESET: Final = ANALYSIS_ENCODING.cpu_preset
"""Encoder preset, taken from the analysis transcode's own settings.

:class:`mosaic_media.io.FFmpegVideoWriter` encodes AV1, which is exactly what an
analysis transcode produces, and ``EncodingParameters.quality`` /
``.cpu_preset`` are that encoder's CRF and preset. So an export is not merely
*like* an analysis derivative, it is encoded by the same recipe -- which is what
makes sharing the transcode kind directory and the analysis forward link honest
rather than a convenient place to put the file.
"""

_TICKS: Final = 1000
"""Progress denominator per exported store."""

_HEARTBEAT_EVERY: Final = 25
"""Frames between progress heartbeats and cancellation checks."""


class StoreExportParams(Params):
    """Parameters for one entry's store export."""

    # entry and camera both select WHICH stores are exported rather than what
    # comes out of one, so both stay out of the recipe -- the identities of the
    # stores actually exported are hashed in their place. Without the exclusion,
    # exporting one camera and exporting both would name the same camera's file
    # two different things.
    entry: Annotated[tuple[str, str], HASH_EXCLUDE]
    camera: Annotated[str | None, HASH_EXCLUDE] = None
    # The AV1 CRF, 0 (lossless) to 63, defaulting to what an analysis transcode
    # uses. Named for its scale because it is one: this writer encodes AV1, and
    # its `crf` argument is a deprecated shim in x264's scale.
    av1_crf: int = ANALYSIS_ENCODING.quality


def export_recipe_hash(params: StoreExportParams) -> str:
    """The recipe every export of this job is named after.

    Everything that varies the output bytes and is not the store itself: the op
    version and the encode settings. No playback profile and no verdict
    thresholds, unlike a transcode's recipe -- neither reaches this encode,
    because there is no source stream to derive an operation from. Every frame is
    decoded and re-encoded unconditionally.

    As with a transcode, the installed ffmpeg build is deliberately absent and
    :attr:`StoreExportOp.version` stands in for it, which means **the version is
    bumped by hand when an upstream change alters what this writes.**
    """
    fingerprint = {
        "op_version": StoreExportOp.version,
        "params": params.identity_dump(),
        "encoding": {
            "preset": _EXPORT_PRESET,
            "pixel_format": ANALYSIS_ENCODING.pixel_format,
        },
    }
    return hash_params(fingerprint)


def export_run_id(recipe_hash: str, source_uuids: list[str]) -> str:
    """Ledger key: the recipe plus the sorted identities of the stores it ran over.

    Mirrors :func:`mosaic.core.pipeline.transcode.transcode_run_id`, and for the
    same reason: a store enters by ``video_uuid``, never by position or path, so
    a reorder or a rename leaves the identity where it is. This value addresses
    nothing -- the filename carries the recipe -- and reaches only the run log.
    """
    fingerprint = {"recipe": recipe_hash, "sources": sorted(source_uuids)}
    return f"export-store-{hash_params(fingerprint)}"


def _stores_for(
    ds: "Dataset", params: StoreExportParams
) -> list[tuple[int, Path, "pd.Series"]]:
    """The imgstore recordings one export would read, in index order.

    ``match_media_rows`` rather than ``resolve_media``: this reads the raw cells
    (``video_uuid``, ``video_order``, ``media_type``), and it takes a camera
    without raising on a multi-camera sequence the way ``resolve_media`` does. A
    camera of ``None`` means every camera of the entry, which is what a caller
    exporting a whole recording wants.
    """
    group, sequence = params.entry
    matched = ds.match_media_rows(group, sequence, params.camera)
    stores: list[tuple[int, Path, "pd.Series"]] = []
    for _, row in matched.iterrows():
        cells = row_mapping(row)
        if str(cells.get("media_type", "")) != "imgstore":
            continue
        video_order = int(str(cells.get("video_order", "") or 0))
        stores.append((video_order, ds.resolve_path(str(cells["abs_path"])), row))
    if not stores:
        camera_note = f" camera {params.camera}" if params.camera else ""
        raise TranscodeError(
            f"{group}/{sequence}{camera_note}: no imgstore rows to export; "
            f"a plain video needs no export and is read directly"
        )
    return stores


def _store_uuids(ds: "Dataset", params: StoreExportParams) -> list[str]:
    """Each store's identity, read from the index rather than probed.

    Resolved before any encoding, so a corpus that has not been re-probed fails
    immediately -- and so a planner can ask what this run will be called without
    opening a store.
    """
    group, sequence = params.entry
    uuids: list[str] = []
    for _, store, row in _stores_for(ds, params):
        source_uuid = media_row_uuid(row_mapping(row))
        if not source_uuid:
            raise TranscodeError(
                f"{group}/{sequence}: {store} has no video_uuid in the media "
                f"index; run 'mosaic reprobe-media --apply' before exporting"
            )
        uuids.append(source_uuid)
    return uuids


@register_op
class StoreExportOp(Op[StoreExportParams]):
    """Export one entry's imgstore recordings as plain video and link them."""

    kind = "export-store"
    domain = "media"
    category = "transcode"
    version = "0.1"
    # Encoding thousands of full-resolution frames is long and CPU-bound, and
    # nothing here touches a GPU: the writer runs a CPU AV1 encode.
    resource_class = "heavy"
    Params = StoreExportParams

    def target(self, params: StoreExportParams) -> str:
        group, sequence = params.entry
        return f"{group}/{sequence}"

    def plan_identity(self, ds: "Dataset", params: StoreExportParams) -> OpIdentity:
        """What this export will be called, without encoding anything.

        The recipe plus the identities of the stores it will read, both of which
        the media index already holds -- so nothing is deferred. Like a
        transcode's, this value addresses nothing: the filename carries the
        recipe, so it names the attempt rather than the output.
        """
        return OpIdentity(
            run_id=export_run_id(export_recipe_hash(params), _store_uuids(ds, params))
        )

    def run(self, ds: "Dataset", params: StoreExportParams, ctx: "JobContext") -> str:
        group, sequence = params.entry
        # The same refusal TranscodeOp makes, for the same reason: on a dataset
        # with no media_raw root, the originals index and media/index.csv are one
        # file, so the back-link would append a derivative row into the originals
        # index and the forward link would land in the same place -- and
        # route_derivatives is then False, so nothing would ever read it.
        if ds.resolve_media_root() != "media_raw":
            message = (
                f"{group}/{sequence}: this dataset has no media_raw root, so "
                f"media/index.csv is its originals index; an export written "
                f"there would never be read"
            )
            raise TranscodeError(message)

        # match_media_rows rather than resolve_media: this reads the raw rows
        # (video_uuid, video_order, media_type), and it takes a camera without
        # raising on a multi-camera sequence the way resolve_media does. A
        # camera of None means every camera of the entry, which is what a caller
        # exporting a whole recording wants.
        stores = _stores_for(ds, params)
        source_uuids = _store_uuids(ds, params)

        recipe_hash = export_recipe_hash(params)
        # Named in one place, so a planner and this run cannot disagree.
        run_id = self.plan_identity(ds, params).run_id
        ctx.set_run_id(run_id)

        media_root = ds.get_root("media")
        export_root = media_root / TRANSCODE_KIND_DIRECTORY
        export_root.mkdir(parents=True, exist_ok=True)
        ctx.set_total(len(stores) * _TICKS)

        for index, (video_order, store, row) in enumerate(stores):
            ctx.check_cancel()
            label = f"{group}/{sequence}[{index}]"
            dest = (
                export_root / f"{source_uuids[index]}.{recipe_hash}.{EXPORT_TARGET}.mp4"
            )
            derivative_rel = relative_to_anchor(dest, media_root)

            # The name carries the whole recipe, so a file at this path is this
            # recipe's output. The link is checked too, and it is a completion
            # marker only because registration writes the back-link row first and
            # the forward link last: an interrupted registration leaves an
            # unlinked file, which this re-exports, rather than a linked file with
            # no row, which nothing repairs.
            already_linked = (
                derivative_cell(row_mapping(row), EXPORT_TARGET) == derivative_rel
            )
            if dest.is_file() and already_linked:
                ctx.progress.on_phase("export-store", f"{label}: reused")
                ctx.heartbeat(done=(index + 1) * _TICKS)
                continue

            ctx.progress.on_phase("export-store", f"{label}: {store.name}")
            write_export(store, dest, row, params.av1_crf, ctx, index)

            facts = probe_media(dest)
            verdict = derive(facts, CHROME_149, media_thresholds())
            set_back_link(
                ds,
                group,
                sequence,
                store,
                dest,
                facts,
                verdict,
                video_order,
                source_video_uuid=source_uuids[index],
                recipe_hash=recipe_hash,
            )
            set_forward_link(
                ds, store, source_uuids[index], derivative_rel, EXPORT_TARGET
            )
            ctx.heartbeat(done=(index + 1) * _TICKS)

        return run_id


def write_export(
    store: Path,
    dest: Path,
    row: "pd.Series",
    av1_crf: int,
    ctx: "JobContext",
    position: int,
) -> None:
    """Decode every frame of *store* in order and encode it into *dest*.

    Writes to a sibling partial file and renames, so an interrupted encode never
    leaves a truncated video at the recipe address -- where the name alone would
    otherwise claim it is that recipe's complete output.

    Frames go out in the reader's order with nothing dropped, duplicated or
    resampled, at the store's own frame rate. That is the property the whole op
    exists for: exported frame *i* is store ``frame_index`` *i*.
    """
    if not is_imgstore(store):
        message = f"{store} is not an imgstore directory"
        raise TranscodeError(message)

    # A frame read is a raw read. The store branch of open_frame_reader calls no
    # gate at all -- target is the caller's declaration of intent -- and a
    # store's verdict carries nothing to gate on.
    reader = open_frame_reader(store, facts=series_facts_or_none(row), target="raw")
    # The partial keeps the .mp4 suffix: the writer picks its output format from
    # the extension, and a bare ".partial" leaves it with nothing to go on.
    partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    written = 0
    try:
        total = reader.frame_count
        with FFmpegVideoWriter(
            partial,
            width=reader.width,
            height=reader.height,
            fps=reader.fps,
            av1_crf=av1_crf,
            av1_preset=_EXPORT_PRESET,
        ) as writer:
            for _, frame in reader:
                writer.write(frame)
                written += 1
                if total and written % _HEARTBEAT_EVERY == 0:
                    ctx.check_cancel()
                    ctx.heartbeat(
                        done=position * _TICKS + int(_TICKS * written / total)
                    )
    finally:
        reader.close()

    if written != reader.frame_count:
        partial.unlink(missing_ok=True)
        message = (
            f"{store}: exported {written} frames but the store reports "
            f"{reader.frame_count}; the export would not line up with the store"
        )
        raise TranscodeError(message)
    partial.replace(dest)
