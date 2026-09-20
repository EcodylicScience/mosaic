"""Concatenating an entry's clips into the one video an external tool can read.

A recorder that chops a session into clips leaves an entry whose frames are one
continuous axis spread over several files. mosaic reads that natively --
``MultiVideoReader`` and :class:`~mosaic.core.media.timeline.ConcatenatedTimeline`
present the clips as one timeline, so every in-process consumer sees the whole
recording. An external binary cannot: it is handed a path and opens it itself.

**Both ways of coping with that were wrong, in opposite directions.** SLEAP,
Lightning Pose and Ultralytics declared ``joins_sources=False``, so the scope
builder truncated the entry to clip 0 and tracked none of the rest. TREx declared
``True`` and was handed the whole list -- and its ``FFmpegVideoCapture``
under-counts every file it opens, then reads only as many frames as it counted,
so each clip lost its tail and the published table's ``frame`` column stopped
addressing the video. Measured on a six-clip fixture carrying its own frame
numbers: 1,800 media frames converted to 1,788, the offset constant inside each
clip and stepping by two at every boundary.

``JoinedExportOp`` (``kind="export-joined"``, ``domain="media"``) removes the
choice by removing the premise. It writes **one** video holding the entry's clips
back to back in ``video_order``, so exported frame *i* is global frame *i*, and
every tool then opens one file with one unambiguous frame index. The same fixture
concatenated converts with the ``.pv`` index equal to the media index at every
former boundary; only the tail is lost, and a tail loss costs no registration
because frame ``F`` still means frame ``F``.

**Stream copy, or a refusal.** The clips are copied, never re-encoded: a copy is
minutes rather than hours, it cannot lose a frame to a rate conversion, and it
leaves the pixels exactly as the tracker would have seen them clip by clip.
Clips that cannot be copied as one stream -- different codecs, geometry or pixel
format, which is what a single analysis derivative among the originals produces
-- are **refused**, naming ``mosaic run --kind transcode`` as the way to make
them uniform first. Re-encoding them here would hide a real disagreement inside a
long CPU-bound pass and produce a file whose relationship to the originals nobody
recorded.

**The count is verified, never assumed.** ``ffmpeg``'s concat demuxer is
frame-exact for matching streams and silently is not for mismatched ones, so the
result is decoded and counted against the sum of the clips' own frame counts, and
a short concatenation is deleted rather than published. That check is the whole
value of the op: it is the thing TREx does not do.

**Addressed by what it holds, not by a run.** The filename carries the ordered
composition digest of the clips it joined -- the same value
:attr:`~mosaic.tracking.common.scope.TrackerWorkItem.source_uid` computes for the
reuse gate -- plus the recipe. So adding, removing, reordering or replacing a
clip addresses a different file, and a file at this path is this clip set joined
by this recipe. There is no index row and no forward link: nothing routes to a
joined export, and only a caller that explicitly asks for one
(:func:`mosaic.tracking.common.tool_input.resolve_tool_inputs`) follows it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic_media import MediaFacts, probe_media
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.transcode import TranscodeError

from mosaic.core.entry import Entry
from mosaic.core.pipeline._utils import ResolvedScope, hash_params
from mosaic.core.params import Params
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "JOINED_KIND_DIRECTORY",
    "JoinedExportOp",
    "JoinedExportParams",
    "joined_export_path",
    "joined_recipe_hash",
    "joined_source_uid",
    "write_joined_export",
]

JOINED_KIND_DIRECTORY: Final = "joined"
"""Where a joined export lives, under the ``media`` root.

Its own kind directory rather than ``transcode``'s. A transcode derivative is one
file per source row and is reached through that row's forward link; this is one
file per *entry*, derived from several rows, and belongs to none of them. Sharing
the directory would put two addressing schemes in one place and let
``prune-media`` judge one by the other's rules.
"""

_CONCAT_TIMEOUT_SECONDS: Final = 3600.0
"""How long one concatenation may take.

A stream copy runs at disk speed -- a 30 GB session is minutes -- so an hour is
not a budget, it is the point at which something has gone wrong and the operator
should hear about it rather than wait.
"""


class JoinedExportParams(Params):
    """Parameters for one entry's joined export.

    Empty, and that is a statement rather than an omission: a stream copy has no
    settings. Nothing about this op can vary the bytes it writes except which
    clips it was given, and those are the scope, not a parameter. The recipe hash
    is therefore over the op version alone.
    """


def joined_recipe_hash(params: JoinedExportParams) -> str:
    """The recipe every joined export is named after.

    The op version and the params, exactly as a transcode's and an export's are,
    and for the same reason: the installed ``ffmpeg`` build is deliberately
    absent and :attr:`JoinedExportOp.version` stands in for it. **So the version
    is bumped by hand when an upstream change alters what this writes.**
    """
    return hash_params(
        {"op_version": JoinedExportOp.version, "params": params.identity_dump()}
    )


def joined_source_uid(facts: "list[MediaFacts] | tuple[MediaFacts, ...]") -> str:
    """The ordered composition digest of the clips a joined export holds.

    The same value
    :attr:`~mosaic.tracking.common.scope.TrackerWorkItem.source_uid` computes, by
    the same call, so the name a tracker looks for and the name this op writes
    cannot drift. ``""`` when any clip carries no content identity, which is the
    one state a joined export cannot be addressed in -- see
    :meth:`JoinedExportOp.run`.
    """
    from mosaic.core.pipeline.composition import MediaMember, media_composition

    if not facts:
        return ""
    if any(not clip.video_uuid for clip in facts):
        return ""
    members = [
        MediaMember(camera="", video_order=order, uid=clip.video_uuid)
        for order, clip in enumerate(facts)
    ]
    return media_composition(members).digest


def joined_export_path(ds: "Dataset", source_uid: str, recipe_hash: str) -> Path:
    """Where the join of one clip set by one recipe lives."""
    root = ds.get_root("media") / JOINED_KIND_DIRECTORY
    return root / f"{source_uid}.{recipe_hash}.joined.mp4"


def _refuse_uncopyable(paths: list[Path], facts: list[MediaFacts]) -> None:
    """Raise unless every clip can be copied into one stream.

    The concat demuxer copies packets; it does not reconcile streams. Clips that
    disagree on codec, geometry or pixel format produce a file that plays as far
    as the first change and then does not, or one whose frame count silently
    differs -- which the verification below would catch, but far too late to say
    anything useful about why.

    Refused rather than re-encoded, because the remedy is a real one and already
    exists: making the clips uniform is what ``transcode`` does, with progress and
    cancellation this op has no business reimplementing.
    """
    first = facts[0]
    for path, clip in zip(paths[1:], facts[1:], strict=True):
        differences = [
            f"{name} {getattr(first, name)!r} then {getattr(clip, name)!r}"
            for name in (
                "codec_name",
                "width",
                "height",
                "pixel_format",
                "rotation_degrees",
            )
            if getattr(first, name) != getattr(clip, name)
        ]
        if differences:
            message = (
                f"{path.name} cannot be joined to {paths[0].name} by copying: "
                f"{'; '.join(differences)}. A joined export copies packets and "
                f"never re-encodes, so the clips have to agree. Run "
                f"`mosaic run --kind transcode` over this entry to make them "
                f"uniform, then export the join of the derivatives."
            )
            raise TranscodeError(message)


def write_joined_export(paths: list[Path], facts: list[MediaFacts], dest: Path) -> int:
    """Copy *paths* into *dest* back to back, and return the frames written.

    Writes to a sibling partial and renames, so an interrupted copy never leaves
    a truncated video at the recipe address -- where the name alone would
    otherwise claim it is that clip set's complete join.

    Raises:
        TranscodeError: If the clips cannot be copied as one stream, or if the
            result does not decode to exactly the sum of their frame counts.
    """
    _refuse_uncopyable(paths, facts)
    expected = sum(int(clip.frame_count) for clip in facts)
    if expected <= 0:
        message = (
            f"{dest.name}: the clips report no frame counts, so a join of them "
            f"could not be checked against anything. Run `mosaic reprobe-media "
            f"--apply` to measure them."
        )
        raise TranscodeError(message)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    listing = dest.with_name(f"{dest.stem}.concat.txt")
    # ffmpeg's concat demuxer reads a file of paths. Single quotes are its
    # quoting, and a literal one is escaped the way its own documentation
    # specifies; a path holding one is rare and silently wrong without this.
    listing.write_text(
        "".join(
            f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
            for p in paths
        )
    )
    try:
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
                "-map",
                "0:v:0",
                str(partial),
            ],
            timeout=_CONCAT_TIMEOUT_SECONDS,
            action=f"joining {len(paths)} clips into {dest.name}",
            error_type=TranscodeError,
        )
        written = int(probe_media(partial).frame_count)
        if written != expected:
            message = (
                f"{dest.name}: joined {written} frames from clips reporting "
                f"{expected}; the join would not line up with the media, which "
                f"is the one thing it exists to guarantee. The clips may "
                f"disagree on frame rate or carry stale frame counts -- "
                f"`mosaic reprobe-media --apply` re-measures them."
            )
            raise TranscodeError(message)
        partial.replace(dest)
        return written
    finally:
        partial.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)


def _one_entry(scope: ResolvedScope) -> Entry:
    """The single entry *scope* covers.

    ``scope_takes = "exactly-one"`` restricts a run to one entry and
    :func:`~mosaic.core.pipeline.ops.check_scope_takes` enforces it before any op
    body runs. This unwraps rather than re-checks, and what it reports is the
    bypass -- a caller arriving here has skipped the check.
    """
    entries = sorted(scope.entries)
    if len(entries) != 1:
        message = (
            f"export-joined was called outside run_op, with a scope of "
            f"{len(entries)} entries. Call mosaic.core.pipeline.ops.run_op, "
            f"which resolves the scope, refuses one this op does not accept, "
            f"and states the arity."
        )
        raise TranscodeError(message)
    return entries[0]


@register_op
class JoinedExportOp(Op[JoinedExportParams]):
    """Join one entry's clips into the single video an external tool opens.

    Reuse is decided by the recipe-addressed filename alone: the name carries the
    ordered composition of the clips *and* the recipe, so a file at that path is
    this clip set joined this way. ``overwrite`` rewrites it, which is how a file
    left by a build no longer trusted is replaced.

    A single-clip entry is a no-op that reports itself: there is nothing to join,
    and the tool already opens the one file.
    """

    kind = "export-joined"
    domain = "media"
    category = "transcode"
    version = "0.1"
    # A stream copy is disk-bound, not CPU-bound, and touches no GPU -- but a
    # session is tens of gigabytes and two of these on one host will contend for
    # the same disk, which is what this class exists to serialize.
    resource_class = "heavy"
    scope_takes = "exactly-one"
    scope_dependent = False
    Params = JoinedExportParams

    def target(self, params: JoinedExportParams, scope: ResolvedScope) -> str:
        """The entry. A short human label for the ledger, not a key."""
        group, sequence = _one_entry(scope)
        return f"{group}/{sequence}"

    def plan_identity(
        self,
        ds: "Dataset",
        params: JoinedExportParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this join will be called, without reading a frame.

        The recipe alone, so nothing is deferred. Like a transcode's and an
        export's, the value addresses nothing: the filename carries both the
        recipe and the clip set, so this names the attempt rather than its output.
        """
        return OpIdentity(run_id=f"export-joined-{joined_recipe_hash(params)}")

    def run(
        self,
        ds: "Dataset",
        params: JoinedExportParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: "JobContext",
    ) -> str:
        group, sequence = _one_entry(scope)
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        ctx.set_total(1)

        resolved = ds.resolve_media(group, sequence)
        paths = list(resolved.paths)
        facts = list(resolved.facts)
        if len(paths) < 2:
            # Reported rather than refused: "this entry is already one file" is a
            # correct answer to "join it", and a pipeline running this over every
            # entry of a dataset should not fail on the single-clip ones.
            ctx.progress.on_phase(
                "export-joined", f"{group}/{sequence}: one clip, nothing to join"
            )
            ctx.heartbeat(done=1)
            return run_id

        source_uid = joined_source_uid(facts)
        if not source_uid:
            # The name is the registration, so a clip set that cannot be named
            # cannot be addressed -- and a join written under a guessed name would
            # be served for a clip set it does not hold.
            message = (
                f"{group}/{sequence}: a clip carries no content identity, so this "
                f"clip set cannot be addressed. Run `mosaic reprobe-media --apply` "
                f"to mint one for every clip, then join them."
            )
            raise TranscodeError(message)

        dest = joined_export_path(ds, source_uid, joined_recipe_hash(params))
        if dest.is_file() and not overwrite:
            ctx.progress.on_phase("export-joined", f"{group}/{sequence}: reused")
            ctx.heartbeat(done=1)
            return run_id

        ctx.check_cancel()
        ctx.progress.on_phase(
            "export-joined", f"{group}/{sequence}: joining {len(paths)} clips"
        )
        written = write_joined_export(paths, facts, dest)
        ctx.progress.on_phase(
            "export-joined", f"{group}/{sequence}: {written} frames -> {dest.name}"
        )
        ctx.heartbeat(done=1)
        return run_id
