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
segment by the previous one's duration expressed in the *first* input's ticks
and never rescales, so an input that counts time differently lands at a wrong
timestamp and drags every later frame with it. :func:`restamp_expression`
rewrites every packet by its index instead, which puts that arithmetic out of
reach. ``PTS-DTS`` is carried through unchanged, so B-frame reordering survives:
the demuxer shifts both ends of that difference equally, making it the one
quantity it cannot corrupt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.transcode import TranscodeError

__all__ = [
    "coded_frame_count",
    "first_packet_dts",
    "restamp_expression",
    "stream_codec",
    "stream_timescale",
    "write_concat_listing",
]

PROBE_TIMEOUT_SECONDS: Final = 3600.0
"""Ceiling for a packet scan, which is minutes over tens of gigabytes."""


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


def stream_codec(path: Path) -> str:
    """*path*'s video codec name, read from its header."""
    return _header_field(path, "codec_name", "codec").lower()


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


def restamp_expression(*, timescale: int, fps: float, origin: int) -> str:
    """A ``setts`` bitstream filter putting every packet on a uniform grid.

    *fps* is the rate the grid is expressed at, and it is the **first** input's:
    a grid has to be written in the ticks the output counts in, and the output's
    tick rate comes from the first input. A set of files recorded at several
    rates therefore gets one label for all of it, which is deliberate. This
    file's timing is not authoritative -- real time per frame comes from the
    sources' own facts -- and what it has to be is *uniform*, so that frame
    ``i`` is findable at ``i`` periods by a tool that seeks.
    """
    period = max(1, round(timescale / fps))
    return f"setts=dts=N*{period}{origin:+d}:pts=N*{period}{origin:+d}+PTS-DTS"


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
