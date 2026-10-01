"""The number of frames that a TREx conversion read, from its ``.pv`` header.

A conversion writes one ``.pv`` frame for each video frame that TREx read, and
records their number in the file's header. It is the count that TREx's own
under-count shows in: TREx reads a file only as far as it counted, which was two
frames short of a 60-frame clip in a measured conversion. The per-individual
exports cannot show it, because each one runs from the individual's first tracked
frame to its last.

The layout follows ``Header::read`` in TREx's
``Application/src/ProcessedVideo/pv.cpp``. Each version adds fields before the
count, so the version decides where the count is. Every number is little-endian,
and every string ends in a NUL byte.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Final

__all__ = ["pv_frame_count"]

_CURRENT_VERSION: Final = 15
"""The newest header that TREx writes, ``PV15``. A newer one is not read."""

_LONGEST_STRING: Final = 1 << 16
"""The longest header string read, so that a file that is not a ``.pv`` ends the read."""

_VERSION_NAME: Final = re.compile(rb"PV(\d+)")


def pv_frame_count(path: Path) -> int | None:
    """Return the number of frames in the ``.pv`` at *path*, or ``None`` when unknown.

    ``None`` when the file is missing or unreadable, is not a ``.pv``, or has a
    header newer than ``PV15``. A conversion that TREx did not finish has a count
    of zero, because TREx writes the count when it closes the file.
    """
    try:
        with path.open("rb") as handle:
            return _read_count(handle)
    except (OSError, ValueError):
        return None


def _read_count(handle: BinaryIO) -> int | None:
    """Read the header of *handle* up to the frame count, and return the count.

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
    if version >= 15:
        _skip(handle, 16)  # the conversion range, eight bytes per end
        _ = _string(handle)  # the source
    _skip(handle, 1)  # the line size
    return int.from_bytes(_bytes(handle, 4), "little")


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
