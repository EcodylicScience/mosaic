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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import pandas as pd
from mosaic_media import (
    CHROME_149,
    SOFTWARE_DECODABLE_CODECS,
    MediaFacts,
    MediaProbeError,
    derive,
    probe_media,
)
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.probe.ffprobe import read_header
from mosaic_media.transcode import ANALYSIS_ENCODING, Target, TranscodeError

from mosaic.core.entry import Entry
from mosaic.core.media.facts_columns import (
    derivative_cell,
    media_row_uuid,
    row_mapping,
    series_facts_or_none,
)
from mosaic.core.media.imgstore_io import fps_from_duration, is_imgstore
from mosaic.core.media.store_rate import store_rate_mismatch
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
    header_codec,
    restamp_expression,
    stream_timescale,
    write_concat_listing,
)
from mosaic.media_probe_config import media_thresholds

if TYPE_CHECKING:
    from collections.abc import Iterable

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

_ENCODED_EXPORT_CODEC: Final = "av1"
"""The codec that :func:`write_export` encodes a store in."""

_TICKS: Final = 1000
"""Progress denominator per exported store."""

_HEARTBEAT_EVERY: Final = 25
"""Frames between progress heartbeats and cancellation checks."""

_COPY_TIMEOUT_SECONDS: Final = 21600.0
"""Ceiling for a chunk copy. Generous: it is I/O over tens of gigabytes."""

_LARGEST_COPY_GAP_FRAME_PERIODS: Final = 1.5
"""The widest step between neighboring frames of a copy, in frame periods.

A copy places frame *i* at *i* periods, so its neighbors are one period apart. A
chunk whose container duration is one frame too long moves every later frame by
one period, and the step at that chunk boundary measures two.
"""


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


@dataclass(frozen=True, slots=True)
class StoreTiming:
    """Record the frames of a store and their rate, as the store's indexes list them.

    Attributes:
        frame_count: The frames the store's chunk indexes list. Every consumer
            addresses the export by them, and a copy reproduces each of them.
        fps: The rate of the store's frame times, ``(frames - 1) / span``, which
            mosaic reads the store at. The chunk files can declare another.
    """

    frame_count: int
    fps: float


def _store_timing(store: Path) -> StoreTiming:
    """Return *store*'s frame count and rate, read from its metadata and chunk indexes.

    It does not open a chunk.

    Raises:
        MediaProbeError: If the store's metadata or a chunk index does not read.
        TranscodeError: If the frame times do not give a positive rate.
    """
    from mosaic.core.media.imgstore_native import NativeStore

    with NativeStore(store) as native:
        frame_count = int(native.frame_count)
        fps = fps_from_duration(frame_count, float(native.duration))
    if fps <= 0:
        message = (
            f"{store}: the frame times of its {frame_count} frames do not give "
            f"a frame rate. An export is stamped at the store's rate."
        )
        raise TranscodeError(message)
    return StoreTiming(frame_count=frame_count, fps=fps)


def export_run_id(params: StoreExportParams) -> str:
    """Ledger key: the recipe of *params*, namespaced.

    Mirrors :func:`mosaic.core.pipeline.transcode.transcode_run_id`, and is not
    re-digested for the same reason: ``export_recipe_hash`` already returns a
    ``hash_params`` digest, and a hash over it would carry nothing new. This
    value addresses nothing -- the filename carries the recipe -- and reaches
    only the run log. What a run covered reaches it too, through the
    ``runs.target`` column :meth:`StoreExportOp.target` writes, which names
    the entry and the cameras for exactly that reason.
    """
    return f"export-store-{export_recipe_hash(params)}"


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


def entry_stores(
    ds: "Dataset",
    group: str,
    sequence: str,
    cameras: frozenset[str] | None = None,
) -> list[tuple[int, Path, "pd.Series"]]:
    """Return an entry's imgstore recordings in index order, with each one's row.

    ``match_media_rows`` rather than ``resolve_media``: this reads the raw cells
    (``video_uuid``, ``video_order``, ``media_type``) and does not raise on a
    multi-camera sequence the way ``resolve_media`` does. An export reads the
    originals and does not read their derivatives.

    Args:
        ds: The dataset, read for its originals index.
        group: The entry's group.
        sequence: The entry's sequence.
        cameras: The cameras to keep, or ``None`` for every camera.

    Returns:
        One ``(video_order, store path, index row)`` per store, empty when the
        entry records none under *cameras*.

    Raises:
        FileNotFoundError: If the media index is missing or empty, or no row
            matches the entry.
    """
    stores: list[tuple[int, Path, "pd.Series"]] = []
    for _, row in ds.match_media_rows(group, sequence).iterrows():
        cells = row_mapping(row)
        if str(cells.get("media_type", "")) != "imgstore":
            continue
        if cameras is not None and str(cells.get("camera", "")) not in cameras:
            continue
        video_order = int(str(cells.get("video_order", "") or 0))
        stores.append((video_order, ds.resolve_path(str(cells["abs_path"])), row))
    return stores


def _stores_for(
    ds: "Dataset", scope: ResolvedScope
) -> list[tuple[int, Path, "pd.Series"]]:
    """The imgstore recordings one export would read, in index order.

    Every camera of the entry is matched and the selector's cameras filter
    them, rather than one camera being passed down. A selector naming two
    cameras of one entry is one entry to the arity declaration and two stores
    here, and a filter states both without a count to reconcile.
    """
    group, sequence = _one_entry(scope)
    cameras = scope.selector.cameras
    stores = entry_stores(ds, group, sequence, frozenset(cameras) if cameras else None)
    if not stores:
        held = {
            str(row_mapping(row).get("camera", ""))
            for _, _, row in entry_stores(ds, group, sequence)
        }
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


def unnamed_stores(stores: Iterable[tuple[int, Path, pd.Series]]) -> list[Path]:
    """Return each of *stores* whose media-index row records no ``video_uuid``.

    An export is named by its store's ``video_uuid``. ``export-store`` refuses
    a store without one.

    Args:
        stores: What :func:`entry_stores` returned.
    """
    return [store for _, store, row in stores if not media_row_uuid(row_mapping(row))]


def _store_uuids(ds: "Dataset", scope: ResolvedScope) -> list[str]:
    """Each store's identity, read from the index rather than probed.

    Resolved before any encoding, so a corpus that has not been re-probed fails
    immediately -- and so a planner can ask what this run will be called without
    opening a store.
    """
    group, sequence = _one_entry(scope)
    stores = _stores_for(ds, scope)
    unnamed = unnamed_stores(stores)
    if unnamed:
        raise TranscodeError(
            f"{group}/{sequence}: the media index does not record a video_uuid "
            f"for {unnamed[0]}; run 'mosaic reprobe-media --apply' before exporting"
        )
    return [media_row_uuid(row_mapping(row)) for _, _, row in stores]


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
    version = "0.3"
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
        return OpIdentity(run_id=export_run_id(params))

    def run(
        self,
        ds: "Dataset",
        params: StoreExportParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: "JobContext",
    ) -> str:
        group, sequence = _one_entry(scope)
        # TranscodeOp makes the same refusal, for the same reason. On a dataset with
        # no media_raw root, the originals index and media/index.csv are one file.
        # The back-link would append a derivative row to the originals index, and
        # the forward link would be written to the same file. routes_derivatives()
        # is then False, and media reads do not route to the export.
        if not ds.routes_derivatives():
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
                facts = copy_export(
                    store, dest, chunks, _store_timing(store), ctx, index
                )
                encoder = ""
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


def readable_chunks(store: Path) -> list[tuple[Path, int]]:
    """*store*'s chunk files and their frame counts, when they hold what mosaic reads.

    Empty otherwise. Two conditions, and each rules out a store whose chunk
    files are not the frames mosaic reads from it:

    * the chunks are video files at all -- a raw ``npy`` or image-directory
      store has no stream to hand over;
    * the store applies no colour conversion. A Bayer or YUV store is turned
      into BGR by ``cv2.cvtColor`` on every read, so its chunks hold different
      pixels from the ones mosaic serves, and a tool reading them would see
      something mosaic never sees.

    Read from the store's metadata and chunk indexes. No chunk is opened.
    """
    from mosaic.core.media.imgstore_native import NativeStore

    with NativeStore(store) as native:
        if not native.is_video or native.encoding is not None:
            return []
        return native.chunk_spans()


def _copied_chunks(store: Path) -> tuple[list[Path], str]:
    """*store*'s chunk files when an export copies them, and their codec.

    Empty and ``""`` when an export encodes the store instead. The rule is
    :func:`copyable_chunks`'s.
    """
    chunks = [path for path, _ in readable_chunks(store)]
    codec = header_codec(chunks[0]) if chunks else ""
    if codec not in SOFTWARE_DECODABLE_CODECS:
        return [], ""
    return chunks, codec


def copyable_chunks(store: Path) -> list[Path]:
    """*store*'s chunk files when they can be copied out verbatim, else empty.

    **A store whose chunks are already video does not need re-encoding, and must
    not get it.** The recorder wrote H.264; decoding every frame and encoding it
    again is slower, larger, lossy, and -- while this op encoded AV1 -- produced
    a file the tools it exists to feed could not open. Measured on a real
    20,000-frame Motif store: 0.30 s copied against about 140 s re-encoded, with
    the copy the smaller and the lossless of the two.

    The chunks must hold what mosaic reads (:func:`readable_chunks`), and be in a
    codec any libavcodec build decodes. A store written in AV1 has to be
    re-encoded for exactly the reason this op's own output did.
    """
    return _copied_chunks(store)[0]


def export_codec(store: Path) -> str:
    """Return the codec of *store*'s export, before the export is written.

    An export copies the chunks when :func:`copyable_chunks` lists them, and keeps
    their codec. Otherwise :func:`write_export` encodes the store in AV1. Reads the
    store's metadata and chunk indexes, and the first chunk's header.

    A first chunk whose header does not read is encoded.

    Raises:
        MediaProbeError: If the store's metadata or a chunk index cannot be read.
    """
    return _copied_chunks(store)[1] or _ENCODED_EXPORT_CODEC


def copy_export(
    store: Path,
    dest: Path,
    chunks: list[Path],
    timing: StoreTiming,
    ctx: "JobContext",
    position: int,
) -> MediaFacts:
    """Concatenate *chunks* into *dest* without decoding, and return its facts.

    The timeline is imposed rather than inherited, the same way
    ``export-joined`` imposes one: exported frame *i* is at *i* periods, for a
    tool that seeks. A period lasts one frame at *timing*'s rate. *timing*
    measures the rate from the store's frame times. Chunk files declare a rate
    too, and Motif declares 25 fps on the chunks of a store recorded at another
    rate. The copy is stamped on the time base of
    :func:`~mosaic.core.pipeline.stream_copy.copy_time_base`, which states the
    store's rate whatever ticks the chunks count.

    The copy is checked before it is renamed to *dest*. It contains *timing*'s
    frames, each at a distinct presentation timestamp, and no two neighbors are
    more than :data:`_LARGEST_COPY_GAP_FRAME_PERIODS` apart. Its measured rate is
    the store's under :func:`~mosaic.core.media.store_rate.store_rate_mismatch`.

    Raises:
        TranscodeError: If the copy fails, contains another number of frames,
            presents two frames at one timestamp, skips a frame period or
            measures at another rate, or the first chunk does not declare a rate.
    """
    partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    listing = dest.with_name(f"{dest.stem}.concat.txt")
    write_concat_listing(chunks, listing)
    try:
        ctx.check_cancel()
        restamp = restamp_expression(
            timescale=stream_timescale(chunks[0]),
            fps=_declared_frame_rate(chunks[0], store),
            origin=first_packet_dts(chunks[0]),
            out_fps=timing.fps,
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
        if written != timing.frame_count:
            message = (
                f"{store}: copied {written} frames from {len(chunks)} chunks "
                f"but the store's index records {timing.frame_count}. A tool "
                f"addresses the export by store frame index."
            )
            raise TranscodeError(message)
        facts = probe_media(partial)
        if facts.frame_count != timing.frame_count:
            message = (
                f"{store}: the copy of its {len(chunks)} chunks presents "
                f"{timing.frame_count} frames at {facts.frame_count} distinct "
                f"timestamps. A tool that seeks by timestamp reads one frame for "
                f"each pair that shares a timestamp."
            )
            raise TranscodeError(message)
        gap = facts.max_timestamp_gap_frame_periods
        if gap > _LARGEST_COPY_GAP_FRAME_PERIODS:
            message = (
                f"{store}: the copy of its {len(chunks)} chunks presents two "
                f"neighboring frames {gap:.3g} frame periods apart. Every later "
                f"frame is presented at least one period after its place."
            )
            raise TranscodeError(message)
        if store_rate_mismatch([timing.fps, facts.fps]) is not None:
            message = (
                f"{store}: the copy of its {len(chunks)} chunks measures "
                f"{facts.fps:.6g} fps, and the store's frame times give "
                f"{timing.fps:.6g} fps. A tool times each frame of an export by "
                f"the export's rate."
            )
            raise TranscodeError(message)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)
    ctx.heartbeat(done=(position + 1) * _TICKS)
    return facts


def _declared_frame_rate(chunk: Path, store: Path) -> float:
    """Return the rate that *chunk*'s header declares and its timestamps count."""
    try:
        rate = read_header(chunk).declared_fps
    except MediaProbeError as exc:
        message = f"{store}: the header of chunk {chunk.name} does not read: {exc}"
        raise TranscodeError(message) from exc
    if rate <= 0:
        message = (
            f"{store}: chunk {chunk.name} does not declare a frame rate. The "
            f"copy counts the chunk's timestamps in frames of that rate."
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
