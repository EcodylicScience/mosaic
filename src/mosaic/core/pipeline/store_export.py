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
from mosaic_media import (
    CHROME_149,
    SOFTWARE_DECODABLE_CODECS,
    derive,
    probe_media,
)
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.transcode import ANALYSIS_ENCODING, Target, TranscodeError

from mosaic.core.entry import Entry
from mosaic.core.media.facts_columns import (
    derivative_cell,
    media_row_uuid,
    row_mapping,
    series_facts_or_none,
)
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.media.video_io import FFmpegVideoWriter, open_frame_reader
from mosaic.core.pipeline._utils import ResolvedScope, hash_params
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.transcode import (
    TRANSCODE_KIND_DIRECTORY,
    relative_to_anchor,
    set_back_link,
    set_forward_link,
)
from mosaic.core.params import (
    Declared,
    Params,
)
from mosaic.core.pipeline.stream_copy import (
    coded_frame_count,
    first_packet_dts,
    restamp_expression,
    stream_codec,
    stream_timescale,
    write_concat_listing,
)
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

_COPY_TIMEOUT_SECONDS: Final = 21600.0
"""Ceiling for a chunk copy. Generous: it is I/O over tens of gigabytes."""


_AV1_CRF_DESCRIPTION = (
    "AV1 constant-rate factor, 0 (lossless) to 63, defaulting to what an "
    "analysis transcode encodes at. Named for its scale because this writer "
    "encodes AV1, whose `crf` argument is a deprecated shim in x264's scale."
)


class StoreExportParams(Params):
    """Parameters for one entry's store export.

    The encode settings alone. Which entry is exported is an argument to the
    run, and the op's declaration restricts it to one. Which cameras of that entry
    are exported comes from the same selector. A triple names one camera, and
    a pair names every camera of the entry.

    Nothing stands in the coverage's place either: the run identifier is the
    recipe, so exporting one camera and exporting both name one run. What
    they covered is told apart in the ledger instead, by the label
    :meth:`StoreExportOp.target` writes.
    """

    av1_crf: Annotated[int, Declared(_AV1_CRF_DESCRIPTION)] = ANALYSIS_ENCODING.quality


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


def _store_frame_count(store: Path) -> int:
    """How many frames *store*'s own index records.

    The store's count, not the chunks': it is what every consumer addresses the
    export by, and what a copy has to reproduce exactly.
    """
    from mosaic.core.media.imgstore_native import NativeStore

    with NativeStore(store) as native:
        return int(native.frame_count)


def export_run_id(recipe_hash: str) -> str:
    """Ledger key: the recipe, namespaced.

    Mirrors :func:`mosaic.core.pipeline.transcode.transcode_run_id`, and is not
    re-digested for the same reason: ``export_recipe_hash`` already returns a
    ``hash_params`` digest, and a hash over it would carry nothing new. This
    value addresses nothing -- the filename carries the recipe -- and reaches
    only the run log. What a run covered reaches it too, through the
    ``runs.target`` column :meth:`StoreExportOp.target` writes, which names
    the entry and the cameras for exactly that reason.
    """
    return f"export-store-{recipe_hash}"


def _one_entry(scope: ResolvedScope) -> Entry:
    """The single entry *scope* covers.

    ``scope_takes = "exactly-one"`` restricts a run to one entry.
    :func:`~mosaic.core.pipeline.ops.check_scope_takes` raises before any op body
    runs, and the planner and the runner both call it. This unwraps rather
    than re-checks. What it reports is the bypass, not the arity, because a
    caller that arrives here has skipped the check and needs to hear that. The
    vocabulary of a scope refusal belongs to the checker, and repeating its
    opening clause here would give one refusal two wordings to drift apart.
    """
    entries = sorted(scope.entries)
    if len(entries) != 1:
        message = (
            f"export-store was called outside run_op, with a scope of "
            f"{len(entries)} entries. Call "
            f"mosaic.core.pipeline.ops.run_op, which resolves the scope, "
            f"refuses one this op does not accept, and states the arity."
        )
        raise TranscodeError(message)
    return entries[0]


def _stores_for(
    ds: "Dataset", scope: ResolvedScope
) -> list[tuple[int, Path, "pd.Series"]]:
    """The imgstore recordings one export would read, in index order.

    ``match_media_rows`` rather than ``resolve_media``: this reads the raw cells
    (``video_uuid``, ``video_order``, ``media_type``) and does not raise on a
    multi-camera sequence the way ``resolve_media`` does. The transcode job
    needs the originals, not their derivatives.

    Every camera of the entry is matched and the selector's cameras filter
    them, rather than one camera being passed down. A selector naming two
    cameras of one entry is one entry to the arity declaration and two stores
    here, and a filter states both without a count to reconcile.
    """
    group, sequence = _one_entry(scope)
    cameras = scope.selector.cameras
    matched = ds.match_media_rows(group, sequence)
    stores: list[tuple[int, Path, "pd.Series"]] = []
    held: set[str] = set()
    for _, row in matched.iterrows():
        cells = row_mapping(row)
        if str(cells.get("media_type", "")) != "imgstore":
            continue
        held.add(str(cells.get("camera", "")))
        if cameras and str(cells.get("camera", "")) not in cameras:
            continue
        video_order = int(str(cells.get("video_order", "") or 0))
        stores.append((video_order, ds.resolve_path(str(cells["abs_path"])), row))
    if not stores:
        raise TranscodeError(_nothing_to_export(group, sequence, cameras, held))
    return stores


def _nothing_to_export(
    group: str, sequence: str, cameras: set[str], held: set[str]
) -> str:
    """Why an export found no store, told apart from why it found no camera.

    Two failures produce one empty list. The entry records no imgstore at all,
    and the entry records imgstores under other camera names. Reporting the
    first for the second sends a caller to the media type when the camera name
    is what is wrong.
    """
    if cameras and held:
        named = ", ".join(sorted(cameras))
        available = ", ".join(sorted(name for name in held if name)) or "none named"
        return (
            f"{group}/{sequence}: no imgstore is recorded under camera "
            f"{named}. This entry's cameras are: {available}."
        )
    camera_note = f" camera {', '.join(sorted(cameras))}" if cameras else ""
    return (
        f"{group}/{sequence}{camera_note}: no imgstore rows to export; "
        f"a plain video needs no export and is read directly"
    )


def _store_uuids(ds: "Dataset", scope: ResolvedScope) -> list[str]:
    """Each store's identity, read from the index rather than probed.

    Resolved before any encoding, so a corpus that has not been re-probed fails
    immediately -- and so a planner can ask what this run will be called without
    opening a store.
    """
    group, sequence = _one_entry(scope)
    uuids: list[str] = []
    for _, store, row in _stores_for(ds, scope):
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
    """Export one entry's imgstore recordings as plain video and link them.

    Reuse is decided per store by the recipe-addressed filename plus the forward
    link, and ``overwrite`` opens that gate. An attempt that passes it re-encodes
    and relinks an export already on disk. That is how a file written by a
    build whose output is no longer trusted is replaced.
    """

    kind = "export-store"
    domain = "media"
    category = "transcode"
    version = "0.2"
    # Encoding thousands of full-resolution frames is long and CPU-bound, and
    # nothing here touches a GPU: the writer runs a CPU AV1 encode.
    resource_class = "heavy"
    scope_takes = "exactly-one"
    scope_dependent = False
    Params = StoreExportParams

    def target(self, params: StoreExportParams, scope: ResolvedScope) -> str:
        """The entry, and the cameras when the selector names any.

        The cameras are the whole of what one attempt here can cover
        differently from another: they narrow what is encoded and they reach
        no identifier, so a label without them records a one-camera export
        and a whole-entry one identically. A short human label, not a key --
        the ledger column is read, not matched on.
        """
        group, sequence = _one_entry(scope)
        cameras = sorted(scope.selector.cameras)
        if cameras:
            return f"{group}/{sequence}[{', '.join(cameras)}]"
        return f"{group}/{sequence}"

    def plan_identity(
        self,
        ds: "Dataset",
        params: StoreExportParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this export will be called, without encoding anything.

        The recipe alone, which is the params -- so nothing is deferred. Like a
        transcode's, this value addresses nothing: the filename carries the
        recipe, so it names the attempt rather than the output.

        The scope is read for the precondition and for nothing the returned
        value depends on. :func:`_store_uuids` refuses a store with no
        ``video_uuid``, and it is here rather than in :meth:`run` alone
        because a plan is where a corpus that cannot be exported should be
        refused.
        """
        _ = _store_uuids(ds, scope)
        return OpIdentity(run_id=export_run_id(export_recipe_hash(params)))

    def run(
        self,
        ds: "Dataset",
        params: StoreExportParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: "JobContext",
    ) -> str:
        group, sequence = _one_entry(scope)
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

        stores = _stores_for(ds, scope)
        source_uuids = _store_uuids(ds, scope)

        recipe_hash = export_recipe_hash(params)
        # Named in one place, so a planner and this run cannot disagree.
        run_id = self.plan_identity(ds, params, scope).run_id
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
            if dest.is_file() and already_linked and not overwrite:
                ctx.progress.on_phase("export-store", f"{label}: reused")
                ctx.heartbeat(done=(index + 1) * _TICKS)
                continue

            # Copied when the recorder already wrote video, encoded only when it
            # did not. The choice is made per store and reported, because the two
            # differ by two orders of magnitude in time and by everything in
            # fidelity.
            chunks = copyable_chunks(store)
            if chunks:
                ctx.progress.on_phase(
                    "export-store",
                    f"{label}: {store.name}, copying {len(chunks)} chunks",
                )
                encoder = copy_export(
                    store, dest, chunks, _store_frame_count(store), ctx, index
                )
            else:
                ctx.progress.on_phase("export-store", f"{label}: {store.name}")
                encoder = write_export(store, dest, row, params.av1_crf, ctx, index)

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
                encoder=encoder,
            )
            set_forward_link(
                ds, store, source_uuids[index], derivative_rel, EXPORT_TARGET
            )
            ctx.heartbeat(done=(index + 1) * _TICKS)

        return run_id


def copyable_chunks(store: Path) -> list[Path]:
    """*store*'s chunk files when they can be copied out verbatim, else empty.

    **A store whose chunks are already video does not need re-encoding, and must
    not get it.** The recorder wrote H.264; decoding every frame and encoding it
    again is slower, larger, lossy, and -- while this op encoded AV1 -- produced
    a file the tools it exists to feed could not open. Measured on a real
    20,000-frame Motif store: 0.30 s copied against about 140 s re-encoded, with
    the copy the smaller and the lossless of the two.

    Three conditions, and each rules out a store whose chunks are not what
    mosaic would read:

    * the chunks are video files at all -- a raw ``npy`` or image-directory
      store has no stream to copy;
    * the store applies no colour conversion. A Bayer or YUV store is turned
      into BGR by ``cv2.cvtColor`` on every read, so its chunks hold different
      pixels from the ones mosaic serves and a copy would hand a tool something
      mosaic never sees;
    * the chunks are in a codec any libavcodec build decodes. A store written
      in AV1 has to be re-encoded for exactly the reason this op's own output
      did.
    """
    from mosaic.core.media.imgstore_native import NativeStore

    with NativeStore(store) as native:
        if not native.is_video or native.encoding is not None:
            return []
        chunks = native.chunk_paths()
    if not chunks or stream_codec(chunks[0]) not in SOFTWARE_DECODABLE_CODECS:
        return []
    return chunks


def copy_export(
    store: Path,
    dest: Path,
    chunks: list[Path],
    expected: int,
    ctx: "JobContext",
    position: int,
) -> str:
    """Concatenate *chunks* into *dest* without decoding, and report no encoder.

    Returns ``""``: nothing encoded, which is what the derivative row's
    ``encoder`` cell already means for a copy remux.

    The timeline is imposed rather than inherited, the same way
    ``export-joined`` imposes one. A store's chunks agree on tick rate far more
    often than an entry's clips do, so the restamp is usually a formality here
    -- but it costs nothing, and it makes the guarantee the same one in both
    places: exported frame *i* sits at *i* periods, for a tool that seeks.
    """
    partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    listing = dest.with_name(f"{dest.stem}.concat.txt")
    write_concat_listing(chunks, listing)
    try:
        ctx.check_cancel()
        restamp = restamp_expression(
            timescale=stream_timescale(chunks[0]),
            fps=_chunk_frame_rate(chunks[0], store),
            origin=first_packet_dts(chunks[0]),
        )
        _ = run_to_completion(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-bsf:v",
                restamp,
                "-map",
                "0:v:0",
                str(partial),
            ],
            timeout=_COPY_TIMEOUT_SECONDS,
            action=f"copying {len(chunks)} chunks of {store.name} into {dest.name}",
            error_type=TranscodeError,
        )
        written = coded_frame_count(partial)
        if written != expected:
            message = (
                f"{store}: copied {written} frames from {len(chunks)} chunks "
                f"but the store's index records {expected}; the export would "
                f"not line up with the store, which is the one thing it exists "
                f"to guarantee."
            )
            raise TranscodeError(message)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)
    ctx.heartbeat(done=(position + 1) * _TICKS)
    return ""


def _chunk_frame_rate(chunk: Path, store: Path) -> float:
    """The rate *chunk* was recorded at, for the grid the copy is stamped onto."""
    rate = probe_media(chunk).fps
    if rate <= 0:
        message = (
            f"{store}: chunk {chunk.name} reports no frame rate, so the export "
            f"could not be given a uniform timeline."
        )
        raise TranscodeError(message)
    return rate


def write_export(
    store: Path,
    dest: Path,
    row: "pd.Series",
    av1_crf: int,
    ctx: "JobContext",
    position: int,
) -> str:
    """Decode every frame of *store* in order and encode it into *dest*, and
    report the encoder that wrote it.

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
            # Read inside the block: the writer resolves its encoder when it
            # opens, and the value is what goes in the derivative's index cell.
            encoder_name = writer.encoder_name
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
    return encoder_name
