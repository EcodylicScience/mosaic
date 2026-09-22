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
result is counted against the sum of the clips' own frame counts, and a short
concatenation is refused rather than published. That check is the whole value of
the op: it is the thing TREx does not do.

**Frames are counted, not timestamps** (:func:`_coded_frame_count`), and the
timeline is then *imposed* rather than inherited. Those are two rules, and the
second one replaced a mistake.

The count is what proves nothing was lost: ``probe_media`` reports distinct
presentation timestamps, which undercounts a join across a frame-rate change, so
packets are counted instead.

The timeline is what makes the file addressable. The concat demuxer does not
rescale between inputs -- it offsets each segment by the previous one's duration
expressed in the *first* clip's ticks -- so a clip that counts time differently
lands at a wildly wrong timestamp and ffmpeg then nudges every following packet
to keep the stream monotonic. This op used to *note* that and publish anyway, on
the reasoning that exported frame ``i`` is still global frame ``i`` and that
nothing reads this file's timing, ``retime_joined_frame`` taking ``time`` from
the source clips' own facts.

**The second half of that was false.** TREx seeks by timestamp:
``FFmpegVideoCapture`` maps each packet's PTS back to a frame index and seeks
backwards whenever it disagrees, and ``video_conversion_range`` addresses the
file by time. Measured on a 17-clip session, one re-encoded clip carrying its
encoder's own tick rate made TREx read frame 108,324 for every frame between
1,745 and 216,760: it ran at 2 fps instead of 45 -- about thirty hours for the
session -- and returned the wrong pixels the whole way.

So the clips are written in one tick rate (:func:`_stream_timescale`,
``-video_track_timescale``), the copy restamps every packet onto a uniform grid
by its index, and a join whose timestamps still do not step evenly is refused
before it is published rather than noted after. Frame ``i`` at ``i`` periods is
the promise; it is now enforced, not assumed. What the grid is *not* is a clock:
a mixed-rate session is labelled at its first clip's rate, and real time per
frame still comes from the clips' own facts.

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
from typing import TYPE_CHECKING, Annotated, Final

from mosaic_media import MediaFacts, probe_media
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.transcode import TranscodeError

from mosaic.core.entry import Entry
from mosaic.core.params import Declared
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

_ENCODERS: Final[dict[str, str]] = {
    "h264": "libx264",
    "av1": "libsvtav1",
    "hevc": "libx265",
}
"""Which encoder re-makes a clip in a given codec.

Only the three this corpus produces. An outlier in any other codec is refused by
name rather than guessed at: picking an encoder mosaic did not choose would put
an unrecorded decision in the middle of a file a tracker then reads as truth.
"""

_NORMALISE_CRF: Final = 18
"""Quality for a normalised clip -- visually lossless, and not archival.

A joined export is a tool input, addressed by recipe and disposable, not a
derivative anything keeps. What matters is that it decodes to the same frames in
the same order; this is high enough that a second generation does not show.
"""

_MAX_TIMESTAMP_GAP_FRAME_PERIODS: Final = 1.5
"""How far apart neighbouring timestamps may be before the join is refused.

Not a taste threshold. ``MediaFacts.max_timestamp_gap_frame_periods`` is what
:class:`mosaic_media.io.VideoReader` uses to decide whether a decoded frame
arrived suspiciously late, and it *stops checking* once that value reaches 1.5
(``reader.py``: ``threshold = gap + 0.5``, returning early at ``>= 2.0``). Above
this number mosaic's own reader has no per-frame protection either, so it is the
exact point past which a file cannot be vouched for.

Measured on the 17-clip session this check was written for: 106,546.65 on the
join that made TREx crawl, 1.008 on the restamped one, 1.0000 on a session whose
clips shared a time base.
"""

_CONCAT_TIMEOUT_SECONDS: Final = 3600.0
"""How long one concatenation may take.

A stream copy runs at disk speed -- a 30 GB session is minutes -- so an hour is
not a budget, it is the point at which something has gone wrong and the operator
should hear about it rather than wait.
"""


_REENCODE_DESCRIPTION = (
    "Normalise clips whose stream profile differs from the rest before joining "
    "them, instead of refusing. Off by default: it re-encodes, which costs time "
    "and a generation of quality, so it is opted into rather than inherited."
)


class JoinedExportParams(Params):
    """Parameters for one entry's joined export.

    One knob, and it exists because of a real corpus. ``transcode`` gives a
    *defective* clip an analysis derivative and leaves its clean siblings alone,
    and :meth:`~mosaic.core.dataset.Dataset.route_media_row` then follows the
    link only for the rows whose verdict demanded it -- so a session where one
    clip of seventeen was defective resolves to sixteen h264 originals and one
    AV1 derivative. That mix cannot be stream-copied, and nothing upstream can
    make it uniform: routing is per row and verdict-driven by design.
    """

    reencode: Annotated[bool, Declared(_REENCODE_DESCRIPTION)] = False


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


def _refuse_mismatched_geometry(paths: list[Path], facts: list[MediaFacts]) -> None:
    """Raise if the clips do not share a frame geometry. Never negotiable.

    Unlike a codec difference this one cannot be normalised away, and must not
    be: making the clips agree would mean rescaling or rotating one of them, and
    every coordinate a tracker then reported for those frames would be in a
    different space from the rest of the session -- a plausible number recorded
    nowhere, which is what the schema's forbidden set exists to refuse.

    The remedy is to fix the arrangement, the same one
    :class:`~mosaic.tracking.common.scope.JoinedSourceMismatchError` names.
    """
    first = facts[0]
    for path, clip in zip(paths[1:], facts[1:], strict=True):
        differences = [
            f"{name} {getattr(first, name)!r} then {getattr(clip, name)!r}"
            for name in ("width", "height", "rotation_degrees")
            if getattr(first, name) != getattr(clip, name)
        ]
        if differences:
            message = (
                f"{path.name} cannot be joined to {paths[0].name}: "
                f"{'; '.join(differences)}. Clips of one session have to share a "
                f"frame geometry -- joining them by rescaling or rotating one "
                f"would put its coordinates in a different space from the rest "
                f"of the session. Fix the arrangement instead."
            )
            raise TranscodeError(message)


def _stream_profile(clip: MediaFacts) -> tuple[str, str]:
    """What has to match for two clips to be copied into one stream."""
    return (clip.codec_name, clip.pixel_format)


def _outliers(facts: list[MediaFacts]) -> tuple[tuple[str, str], list[int]]:
    """The profile most clips share, and the positions of the ones that do not.

    Majority rather than "whatever the first clip is", because normalising one
    derivative back to its sixteen siblings is minutes and normalising sixteen
    originals to one derivative is hours. Ties go to the earliest profile, which
    keeps the answer deterministic.
    """
    profiles = [_stream_profile(clip) for clip in facts]
    ranked = sorted(
        dict.fromkeys(profiles),
        key=lambda p: (-profiles.count(p), profiles.index(p)),
    )
    majority = ranked[0]
    return majority, [i for i, p in enumerate(profiles) if p != majority]


def _coded_frame_count(path: Path) -> int:
    """How many coded video frames *path* holds.

    Deliberately **not** ``probe_media(path).frame_count``. That value is the
    number of *distinct presentation timestamps* -- literally ``len({packet.time
    for packet in packets})`` -- so two frames sharing a timestamp contribute
    one. It is the right measure for "does frame ``i`` sit at ``i / fps``", which
    is what the probe exists to answer, and the wrong one for "did every frame
    survive the copy".

    The two come apart exactly where this op works. Concatenating clips across a
    frame-rate change makes ffmpeg re-time each segment by the previous one's
    duration, and the rounding lands one frame of the new clip on the timestamp
    of the old clip's last. Nothing is lost and the distinct-timestamp count
    drops anyway. Measured on a real 17-clip session recorded at 30, 29.948 and
    31 fps: the join held all 390,986 frames, the probe reported 390,984, and
    this op refused a complete video and stopped the session being tracked.

    Packets rather than decoded frames, because the count has to be exact over
    tens of gigabytes: ``-count_packets`` demuxes without decoding, which is
    minutes where a full decode is hours. One video packet is one coded frame
    for every codec this op will copy.
    """
    out = run_to_completion(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=_CONCAT_TIMEOUT_SECONDS,
        action=f"counting the coded frames of {path.name}",
        error_type=TranscodeError,
    ).strip()
    try:
        return int(out)
    except ValueError as exc:
        message = (
            f"{path.name}: ffprobe reported {out!r} where a frame count was "
            f"expected, so the join could not be checked against anything."
        )
        raise TranscodeError(message) from exc


def _joined_frame_rate(facts: list[MediaFacts], dest: Path) -> float:
    """The rate the joined file's uniform timeline is built on.

    The **first** clip's, because the first clip is what the output's tick rate
    comes from, and a grid has to be expressed in the ticks it is written in.

    A session recorded at several rates therefore gets one label for all of it,
    and that is deliberate: this file's timing is not authoritative and never
    was. Real time per frame comes from the source clips' own facts through
    ``retime_joined_frame``. What the grid has to be is *uniform*, so that frame
    ``i`` is findable at ``i`` periods by a tool that seeks -- which is the
    property the clips' real rates cannot provide and the concat demuxer
    destroys.
    """
    rate = facts[0].fps if facts else 0.0
    if rate <= 0:
        message = (
            f"{dest.name}: the first clip reports no frame rate, so the join "
            f"could not be given a uniform timeline. Run `mosaic reprobe-media "
            f"--apply` to measure it."
        )
        raise TranscodeError(message)
    return rate


def _stream_timescale(path: Path) -> int:
    """The denominator of *path*'s video time base -- its ticks per second.

    Not on :class:`~mosaic_media.MediaFacts`, which models what a stream *shows*
    and not how it counts. It is needed here because the concat demuxer does not
    rescale: it writes each input's raw ticks into a track whose timescale comes
    from the *first* input, so two clips that disagree produce timestamps wrong
    by the ratio between them. Measured once, from the clip the join is built
    around.
    """
    out = run_to_completion(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=_CONCAT_TIMEOUT_SECONDS,
        action=f"reading the time base of {path.name}",
        error_type=TranscodeError,
    ).strip()
    _, _, denominator = out.partition("/")
    try:
        timescale = int(denominator)
    except ValueError as exc:
        message = (
            f"{path.name}: ffprobe reported a time base of {out!r}, which has no "
            f"tick rate in it, so the join could not be given a uniform timeline."
        )
        raise TranscodeError(message) from exc
    if timescale <= 0:
        message = (
            f"{path.name}: ffprobe reported a time base of {out!r}, a tick rate "
            f"of {timescale}, which cannot carry a timeline."
        )
        raise TranscodeError(message)
    return timescale


def _first_packet_dts(path: Path) -> int:
    """*path*'s first video packet's decode timestamp, in its own ticks.

    Reproduced onto the join so the result starts where a single-clip file
    would. H.264 with B-frames conventionally opens at a negative DTS -- the
    reorder delay -- and a join that silently started at zero would shift its
    whole presentation relative to the clip it was built from.

    ``N/A`` is a real answer for a stream carrying no decode timestamps, and it
    means the same thing as zero here: there is no offset to preserve.
    """
    out = run_to_completion(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            "%+#1",
            "-show_entries",
            "packet=dts",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=_CONCAT_TIMEOUT_SECONDS,
        action=f"reading the first packet timestamp of {path.name}",
        error_type=TranscodeError,
    ).strip()
    first = out.splitlines()[0].strip() if out else ""
    try:
        return int(first)
    except ValueError:
        return 0


def _normalise_clip(
    source: Path,
    clip: MediaFacts,
    profile: tuple[str, str],
    dest: Path,
    *,
    timescale: int,
) -> None:
    """Re-encode *source* into *profile*, keeping every frame and its order.

    ``-fps_mode passthrough`` is what makes this safe to do to one clip of a
    session: it writes one output frame per input frame and never resamples, so
    a mixed-rate session keeps each clip's own count. Verified afterwards
    anyway, because a silent frame drop here would be indistinguishable from the
    concat defect this whole op exists to prevent.

    ``-video_track_timescale`` is the other half, and it is not cosmetic. An
    encoder picks its own tick rate -- libx264 chose 1/953497 for the clip this
    argument was added for -- and :func:`write_joined_export` then copies packets
    into a track whose timescale came from a *different* clip. The concat
    demuxer does not rescale between them, so the re-encoded clip's timestamps
    landed 62x too large and every frame after it was addressed wrongly. Writing
    the clip in its neighbours' ticks means its numbers mean what theirs mean.
    """
    codec, pixel_format = profile
    encoder = _ENCODERS.get(codec)
    if encoder is None:
        message = (
            f"{source.name} would have to be re-encoded to {codec!r} to join its "
            f"siblings, and mosaic declares no encoder for that codec (it knows "
            f"{', '.join(sorted(_ENCODERS))}). Make the clips uniform with "
            f"`mosaic run --kind transcode` instead."
        )
        raise TranscodeError(message)
    _ = run_to_completion(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            encoder,
            "-crf",
            str(_NORMALISE_CRF),
            "-pix_fmt",
            pixel_format,
            "-fps_mode",
            "passthrough",
            "-video_track_timescale",
            str(timescale),
            str(dest),
        ],
        timeout=_CONCAT_TIMEOUT_SECONDS,
        action=f"re-encoding {source.name} to {codec} so it can be joined",
        error_type=TranscodeError,
    )
    written = _coded_frame_count(dest)
    expected = _coded_frame_count(source)
    if written != expected:
        dest.unlink(missing_ok=True)
        message = (
            f"{source.name}: re-encoding it to {codec} produced {written} frames "
            f"where the clip holds {expected}. A normalisation that loses a frame "
            f"would shift every later frame of the session."
        )
        raise TranscodeError(message)


def write_joined_export(
    paths: list[Path],
    facts: list[MediaFacts],
    dest: Path,
    *,
    reencode: bool = False,
) -> int:
    """Copy *paths* into *dest* back to back, and return the frames written.

    Writes to a sibling partial and renames, so an interrupted copy never leaves
    a truncated video at the recipe address -- where the name alone would
    otherwise claim it is that clip set's complete join.

    The clips are joined by **copying packets**, which is minutes rather than
    hours and cannot lose a frame to a rate conversion. *reencode* handles the
    corpus where that is not possible outright: clips whose stream profile
    differs from the majority are re-made in the majority's profile first, one
    file at a time, and the copy proceeds over the result. The majority is
    normalised *to*, never *from*, so a single derivative among sixteen
    originals costs one re-encode and not sixteen.

    Raises:
        TranscodeError: If the clips disagree on frame geometry; if they
            disagree on stream profile and *reencode* is off; if normalising one
            of them changes its frame count; or if the join does not decode to
            exactly the sum of the clips' own counts.
    """
    _refuse_mismatched_geometry(paths, facts)
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
    majority, odd = _outliers(facts)
    if odd and not reencode:
        named = ", ".join(
            f"{paths[i].name} ({facts[i].codec_name}/{facts[i].pixel_format})"
            for i in odd[:3]
        )
        message = (
            f"{dest.name}: {len(odd)} of {len(paths)} clips do not share the "
            f"stream profile of the rest ({majority[0]}/{majority[1]}): {named}"
            f"{' ...' if len(odd) > 3 else ''}. A joined export copies packets, "
            f"so the clips have to agree. Either make them uniform with "
            f"`mosaic run --kind transcode`, or pass reencode=true to re-make "
            f"just the odd ones in the majority profile before joining."
        )
        raise TranscodeError(message)
    # The ticks every clip in this join is written in. Taken from a clip already
    # in the majority profile, never from an outlier: an outlier is about to be
    # re-encoded, and an encoder picks a tick rate of its own that no other clip
    # shares. `majority` is non-empty by construction, so this always resolves.
    odd_positions = set(odd)
    reference = next(i for i in range(len(paths)) if i not in odd_positions)
    timescale = _stream_timescale(paths[reference])
    # Normalised beside the partial and swept with it, so an interrupted run
    # leaves no half-encoded clip at a name a later run would trust.
    normalised: list[Path] = []
    for i in odd:
        temp = dest.with_name(f"{dest.stem}.normalised{i}{dest.suffix}")
        _normalise_clip(paths[i], facts[i], majority, temp, timescale=timescale)
        normalised.append(temp)
        paths = [*paths[:i], temp, *paths[i + 1 :]]
    # ffmpeg's concat demuxer reads a file of paths. Single quotes are its
    # quoting, and a literal one is escaped the way its own documentation
    # specifies; a path holding one is rare and silently wrong without this.
    listing.write_text(
        "".join(
            f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
            for p in paths
        )
    )
    # One uniform grid for the whole join, imposed rather than inherited. The
    # concat demuxer's own arithmetic is what this op exists to distrust: it
    # offsets each segment by the previous one's duration in the *first* clip's
    # ticks, so any clip that counts differently lands at the wrong timestamp
    # and drags every frame after it along. Restamping by packet index makes
    # that arithmetic unreachable -- frame `i` is at `i` ticks*period, whatever
    # the inputs disagreed about.
    #
    # `PTS-DTS` is carried through unchanged, which is what preserves B-frame
    # reordering: the concat demuxer shifts both ends of that difference
    # equally, so the difference itself is the one quantity it cannot corrupt.
    period = max(1, round(timescale / _joined_frame_rate(facts, dest)))
    origin = _first_packet_dts(paths[0])
    restamp = f"setts=dts=N*{period}{origin:+d}:pts=N*{period}{origin:+d}+PTS-DTS"
    keep_partial = False
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
                "-bsf:v",
                restamp,
                "-map",
                "0:v:0",
                str(partial),
            ],
            timeout=_CONCAT_TIMEOUT_SECONDS,
            action=f"joining {len(paths)} clips into {dest.name}",
            error_type=TranscodeError,
        )
        written = _coded_frame_count(partial)
        if written != expected:
            # Measured, not guessed. The index's counts and the clips on disk
            # are different things, and a message that cannot say which one is
            # short sends the reader to re-probe media that was never wrong.
            measured = [_coded_frame_count(path) for path in paths]
            stale = [
                f"{path.name} holds {held} where the index records {int(clip.frame_count)}"
                for path, clip, held in zip(paths, facts, measured, strict=True)
                if held != int(clip.frame_count)
            ]
            if stale:
                detail = (
                    f"The clips do not hold what the index says they do: "
                    f"{'; '.join(stale[:3])}{' ...' if len(stale) > 3 else ''}. "
                    f"Run `mosaic reprobe-media --apply` to re-measure them."
                )
            else:
                detail = (
                    f"The clips hold exactly what the index records, so the "
                    f"copy itself lost them. The short join is kept at "
                    f"{partial.name} for inspection."
                )
            keep_partial = not stale
            message = (
                f"{dest.name}: joined {written} frames from clips holding "
                f"{sum(measured)} and recorded as {expected}; the join would "
                f"not line up with the media, which is the one thing it exists "
                f"to guarantee. {detail}"
            )
            raise TranscodeError(message)
        # Measured on the partial, before it is published: a join whose timeline
        # is wrong must not reach the recipe address, where the name alone would
        # claim it is this clip set joined correctly. This is the probe pass the
        # op used to spend *after* publishing, moved to where it can still
        # withhold the file.
        #
        # The gap, and deliberately not `frame_count`. That value counts
        # distinct timestamps, and gating on it is the regression
        # `_coded_frame_count` exists to document: it refused a complete
        # 390,986-frame session. The widest step between neighbours answers a
        # different question -- is this timeline uniform enough to seek -- and
        # answers it without depending on how the probe counts.
        gap = probe_media(partial).max_timestamp_gap_frame_periods
        if gap > _MAX_TIMESTAMP_GAP_FRAME_PERIODS:
            keep_partial = True
            message = (
                f"{dest.name}: neighbouring timestamps in the join step as far "
                f"as {gap:.2f} frame periods apart, where a uniform timeline "
                f"steps 1.00. A tool that seeks by timestamp -- TREx does -- "
                f"lands on the wrong frame past the gap and reads the wrong "
                f"pixels, and mosaic's own reader stops checking for missing "
                f"frames above {_MAX_TIMESTAMP_GAP_FRAME_PERIODS}. The join is "
                f"kept at {partial.name} for inspection."
            )
            raise TranscodeError(message)
        partial.replace(dest)
        return written
    finally:
        if not keep_partial:
            partial.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)
        for temp in normalised:
            temp.unlink(missing_ok=True)


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
    version = "0.2"
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
            # Nothing was joined, so nothing is reported: zero means "not
            # reported" by the run-log's own convention, which is the honest
            # answer for an entry that needed no join.
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
            ctx.entries_written(1)
            ctx.heartbeat(done=1)
            return run_id

        ctx.check_cancel()
        _, odd = _outliers(facts)
        note = (
            f", re-encoding {len(odd)} of them first" if odd and params.reencode else ""
        )
        ctx.progress.on_phase(
            "export-joined", f"{group}/{sequence}: joining {len(paths)} clips{note}"
        )
        written = write_joined_export(paths, facts, dest, reencode=params.reencode)
        ctx.progress.on_phase(
            "export-joined", f"{group}/{sequence}: {written} frames -> {dest.name}"
        )
        # No timestamp check here any more. It used to live at this point as a
        # note, on the premise that seeking this file was nobody's business;
        # TREx seeks it, so the check became a refusal and moved inside
        # write_joined_export, where it runs on the partial and can still
        # withhold the file. Checking after `partial.replace(dest)` could only
        # ever describe a video already published at its recipe address.
        # Reported, because "finished" alone could not tell a joined session
        # from one the op decided needed no join -- which is exactly the
        # ambiguity that hid a recipe-addressing bug behind a clean exit.
        ctx.entries_written(1)
        ctx.heartbeat(done=1)
        return run_id
