"""What a TREx conversion read, from its ``.pv`` header: frames, and of which files.

A conversion writes one ``.pv`` frame for each video frame that TREx read, and
records their number in the file's header. It is the count that TREx's own
under-count shows in: TREx reads a file only as far as it counted, which was two
frames short of a 60-frame clip in a measured conversion. The per-individual
exports cannot show it, because each one runs from the individual's first tracked
frame to its last. From version 15 the header also records the conversion's
source, the file or files TREx read, whose headers say how short it read them.

The layout follows ``Header::read`` in TREx's
``Application/src/ProcessedVideo/pv.cpp``. Each version adds fields before the
count, so the version decides where the count is. Every number is little-endian,
and every string ends in a NUL byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

__all__ = ["PvHeader", "read_pv_header"]

_CURRENT_VERSION: Final = 15
"""The newest header that TREx writes, ``PV15``. A newer one is not read."""

_LONGEST_STRING: Final = 1 << 16
"""The longest header string read.

A file that is not a ``.pv`` may contain no NUL byte to end a string, and the
read of such a file stops here.
"""

_VERSION_NAME: Final = re.compile(rb"PV(\d+)")


@dataclass(frozen=True, slots=True)
class PvHeader:
    """What one ``.pv`` header records about its conversion.

    Attributes:
        frames: How many frames the conversion read. Zero for a conversion that
            TREx did not finish, because TREx writes the count when it closes
            the file.
        sources: The files the conversion read, in order, as TREx recorded them.
            Empty for a header older than ``PV15``, which records none.
    """

    frames: int
    sources: tuple[str, ...]


def read_pv_header(path: Path) -> PvHeader | None:
    """Return what the ``.pv`` at *path* records, or ``None`` when unknown.

    ``None`` when the file is missing or unreadable, is not a ``.pv``, or has a
    header newer than ``PV15``.
    """
    try:
        with path.open("rb") as handle:
            return _read_header(handle)
    except (OSError, ValueError):
        return None


def _read_header(handle: BinaryIO) -> PvHeader | None:
    """Read the header of *handle* up to the frame count.

    Raises:
        ValueError: If the file ends inside the header, or a string runs past
            :data:`_LONGEST_STRING`.
    """
    name = _string(handle)
    matched = _VERSION_NAME.search(name) if len(name) > 2 else None
    version = int(matched.group(1)) if matched is not None else 1
    if version > _CURRENT_VERSION:
        return None
    if version >= 14:
        _ = _string(handle)  # the encoding's name
    else:
        _skip(handle, 2 if version >= 12 else 1)  # channels, and encoding index
    _skip(handle, 4)  # the width and height, two bytes each
    if version >= 3:
        _skip(handle, 8)  # the four crop offsets, two bytes each
    sources: tuple[str, ...] = ()
    if version >= 15:
        _skip(handle, 16)  # the conversion range, eight bytes per end
        sources = _sources(_string(handle).decode("utf-8", errors="replace"))
    _skip(handle, 1)  # the line size
    frames = int.from_bytes(_bytes(handle, 4), "little")
    return PvHeader(frames=frames, sources=sources)


def _sources(recorded: str) -> tuple[str, ...]:
    """Split the source that TREx recorded into the files it names.

    TREx records one file as its path, and several between brackets, separated
    by commas and unquoted, as TREx build ``4b48601`` recorded two clips:
    ``[/data/a.mp4,/data/b.mp4]``. That form cannot tell a comma inside a path
    from one between paths, so such a path is read as two, which still names
    several files.
    """
    if not recorded:
        return ()
    if not (recorded.startswith("[") and recorded.endswith("]")):
        return (recorded,)
    return tuple(recorded[1:-1].split(","))


def _string(handle: BinaryIO) -> bytes:
    """Read one NUL-terminated string, without its NUL.

    Raises:
        ValueError: If no NUL byte comes before the file ends or within
            :data:`_LONGEST_STRING` bytes.
    """
    read = bytearray()
    while len(read) <= _LONGEST_STRING:
        byte = _bytes(handle, 1)
        if byte == b"\0":
            return bytes(read)
        read += byte
    message = "not a TREx .pv header: a string does not end"
    raise ValueError(message)


def _skip(handle: BinaryIO, size: int) -> None:
    """Read past *size* bytes.

    Raises:
        ValueError: If the file ends first.
    """
    _ = _bytes(handle, size)


def _bytes(handle: BinaryIO, size: int) -> bytes:
    """Read exactly *size* bytes.

    Raises:
        ValueError: If the file ends first.
    """
    read = handle.read(size)
    if len(read) < size:
        message = "not a TREx .pv header: the file ends inside it"
        raise ValueError(message)
    return read
