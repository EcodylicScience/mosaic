"""Copying coded video packets from one or more files into a single mp4.

Two ops need the same four things and used to have none of them in common: an
exact frame count, the tick rate a file counts time in, the offset it starts at,
and a way to put packets from several files onto one timeline that a tool can
seek. ``export-joined`` puts an entry's clips end to end; ``export-store`` hands
over an imgstore's chunks. A store's chunks are usually already H.264, so
copying them is both lossless and about four hundred times faster than decoding
and re-encoding every frame.

**Counting is packets, never timestamps.** ``probe_media(...).frame_count`` is
the number of *distinct presentation timestamps*, which is the right measure for
"does frame ``i`` sit at ``i / fps``" and the wrong one for "did every frame
survive the copy". The two part company exactly where these ops work: joining
across a frame-rate change makes ffmpeg re-time each segment and the rounding
puts two frames on one timestamp. Measured on a real 17-clip session, the join
held all 390,986 frames and the probe reported 390,984.

**Timing is imposed, never inherited.** ffmpeg's concat demuxer offsets each
segment by the previous one's duration expressed in the first input's ticks and
does not rescale. An input that counts time differently is offset to a wrong
timestamp, and every later frame moves with it. :func:`restamp_expression`
writes each packet's timestamps in one of two forms:

- ``export-joined`` stamps each packet by its index, and the demuxer's offsets do
  not enter the result. ``PTS-DTS`` is copied unchanged, which keeps B-frame
  reordering. The demuxer adds one offset to both timestamps and leaves their
  difference intact.
- ``export-store`` copies onto the time base of :func:`copy_time_base` and takes
  each packet's presentation frame from the demuxer's PTS, offset included. The
  form is sound only when every input counts the same time base and rate, as one
  recorder's chunks do. ``PTS-DTS`` is not used there, because chunks that
  reorder by different depths give two packets one presentation timestamp under
  it.
  :func:`~mosaic.core.pipeline.store_export.copy_export` refuses a copy that
  presents two frames at one timestamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic_media import MediaProbeError
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.probe.ffprobe import read_header
from mosaic_media.transcode import TranscodeError

__all__ = [
    "CopyTimeBase",
    "coded_frame_count",
    "copy_time_base",
    "first_packet_dts",
    "header_codec",
    "restamp_expression",
    "stream_timescale",
    "write_concat_listing",
]

PROBE_TIMEOUT_SECONDS: Final = 3600.0
"""Ceiling for a packet scan, which is minutes over tens of gigabytes."""

_MINIMUM_TICKS_PER_FRAME: Final = 1000
"""The fewest ticks per frame of a copy restamped onto a time base of its own."""

_MINIMUM_TICKS_PER_SECOND: Final = 20_000
"""The fewest ticks per second of a copy restamped onto a time base of its own.

Rounding the ticks per second to a whole number then moves the stated rate by at
most 2.5e-5 of itself, a twentieth of the store rate tolerance.
"""


@dataclass(frozen=True, slots=True)
class CopyTimeBase:
    """Record the time base that a copy is restamped onto, and its frame length.

    Attributes:
        ticks_per_second: The denominator of the copy's time base.
        ticks_per_frame: The duration of one frame, in those ticks.
    """

    ticks_per_second: int
    ticks_per_frame: int


def copy_time_base(fps: float) -> CopyTimeBase:
    """Return the time base that states *fps* to within 2.5e-5 of itself.

    A frame is 1000 ticks at 20 fps and above, and more at a lower rate, so that
    the time base counts at least 20,000 ticks per second. A 0.5003 fps
    time-lapse store gets 39,977 ticks to a frame. At 1000 ticks to a frame, its
    time base would state 0.5 fps.
    """
    ticks_per_frame = max(
        _MINIMUM_TICKS_PER_FRAME, math.ceil(_MINIMUM_TICKS_PER_SECOND / fps)
    )
    return CopyTimeBase(
        ticks_per_second=round(fps * ticks_per_frame), ticks_per_frame=ticks_per_frame
    )


def coded_frame_count(path: Path) -> int:
    """How many coded video frames *path* holds.

    Packets rather than decoded frames, because the count has to be exact over
    tens of gigabytes: ``-count_packets`` demuxes without decoding, which is
    minutes where a full decode is hours. One video packet is one coded frame
    for every codec these ops copy.
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
        timeout=PROBE_TIMEOUT_SECONDS,
        action=f"counting the coded frames of {path.name}",
        error_type=TranscodeError,
    ).strip()
    try:
        return int(out)
    except ValueError as exc:
        message = (
            f"{path.name}: ffprobe reported {out!r} where a frame count was "
            f"expected, so the copy could not be checked against anything."
        )
        raise TranscodeError(message) from exc


def header_codec(path: Path) -> str:
    """Return *path*'s video codec name from its header, or ``""`` if unreadable.

    The header is read in place of :func:`~mosaic_media.probe_media`, which scans
    every packet, because the codec is in the first few bytes. A header that does
    not read, or a read that outlasts mosaic-media's header timeout, answers
    ``""``. A codec gate passes it, and the tool that cannot open the file reports
    the error. A store export encodes chunks of an unknown codec.
    """
    try:
        return read_header(path).codec_name
    except MediaProbeError:
        return ""


def stream_timescale(path: Path) -> int:
    """The denominator of *path*'s video time base -- its ticks per second.

    Not on :class:`~mosaic_media.MediaFacts`, which models what a stream *shows*
    and not how it counts. It is needed because the concat demuxer does not
    rescale between inputs that disagree.
    """
    out = _header_field(path, "time_base", "time base")
    _, _, denominator = out.partition("/")
    try:
        timescale = int(denominator)
    except ValueError as exc:
        message = (
            f"{path.name}: ffprobe reported a time base of {out!r}, which has no "
            f"tick rate in it, so the copy could not be given a uniform timeline."
        )
        raise TranscodeError(message) from exc
    if timescale <= 0:
        message = (
            f"{path.name}: ffprobe reported a time base of {out!r}, a tick rate "
            f"of {timescale}, which cannot carry a timeline."
        )
        raise TranscodeError(message)
    return timescale


def first_packet_dts(path: Path) -> int:
    """*path*'s first video packet's decode timestamp, in its own ticks.

    Reproduced onto the copy so the result starts where its first input does.
    H.264 with B-frames conventionally opens at a negative DTS -- the reorder
    delay -- and a copy that silently started at zero would shift its whole
    presentation relative to the file it was built from.

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
        timeout=PROBE_TIMEOUT_SECONDS,
        action=f"reading the first packet timestamp of {path.name}",
        error_type=TranscodeError,
    ).strip()
    first = out.splitlines()[0].strip() if out else ""
    try:
        return int(first)
    except ValueError:
        return 0


def restamp_expression(
    *, timescale: int, fps: float, origin: int, out_fps: float | None = None
) -> str:
    """A ``setts`` bitstream filter putting every packet on a uniform grid.

    *timescale* is the input's ticks per second, *fps* the rate the input counts
    frames at, and *origin* its first packet's decode timestamp, in input ticks.

    Without *out_fps*, the grid is in the input's ticks and the output keeps
    the input's time base. A frame is then ``round(timescale / fps)`` ticks, and
    *fps* is the **first** input's. A set of files recorded at several rates gets
    one label for all of it. Real time per frame comes from the sources' facts,
    and this file's timing is uniform: frame ``i`` is at ``i`` periods for a tool
    that seeks.

    With *out_fps*, the output is stamped at that rate on the time base of
    :func:`copy_time_base`. A copy can then be stamped at a rate that the input's
    ticks cannot express or that the input does not declare. Every offset is
    converted in frames: *origin*, and each packet's presentation frame. A
    presentation frame is the packet's presentation timestamp counted in
    input frames. It keeps a reordered picture in its place, and keeps the order
    of a file whose segments reorder by different depths. The conversion assumes
    inputs that all count *timescale* ticks per second and *fps* frames per
    second, as one recorder's chunks do.
    """
    if out_fps is None:
        period = max(1, round(timescale / fps))
        return f"setts=dts=N*{period}{origin:+d}:pts=N*{period}{origin:+d}+PTS-DTS"
    time_base = copy_time_base(out_fps)
    ticks = time_base.ticks_per_frame
    start = round(origin * fps / timescale) * ticks
    frame = f"round(PTS*{fps!r}/{timescale})"
    return (
        f"setts=time_base=1/{time_base.ticks_per_second}"
        f":dts=N*{ticks}{start:+d}"
        f":pts={frame}*{ticks}"
        f":duration={ticks}"
    )


def write_concat_listing(paths: list[Path], listing: Path) -> None:
    """Write the file of paths ffmpeg's concat demuxer reads.

    Single quotes are its quoting, and a literal one is escaped the way its own
    documentation specifies; a path holding one is rare and silently wrong
    without this.
    """
    quote = chr(39)
    escaped = quote + chr(92) + quote + quote
    listing.write_text(
        "".join(f"file '{str(p).replace(quote, escaped)}'\n" for p in paths)
    )


def _header_field(path: Path, entry: str, label: str) -> str:
    out = run_to_completion(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"stream={entry}",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=PROBE_TIMEOUT_SECONDS,
        action=f"reading the {label} of {path.name}",
        error_type=TranscodeError,
    ).strip()
    return out.splitlines()[0].strip() if out else ""
