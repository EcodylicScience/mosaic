"""Load raw files that a user, or an upload, supplied.

A supplied file is untrusted. A loader here never unpickles one, so the file
cannot run code, and refuses arrays whose declared sizes add up past
`raw_array_limit_bytes`, so it cannot exhaust memory. A format that can only be
read by unpickling loads only when `MOSAIC_TRUST_PICKLED_FILES=1` says the
files are trusted.

The refusals are `UnsafeRawFileError`, not a `TrackSchemaError`, so a batch
conversion records the refused file as failed and goes on to the next one.
"""

from __future__ import annotations

import math
from pathlib import Path
import zipfile

import numpy as np

from mosaic.env_settings import flag_env, int_env

__all__ = [
    "LIMIT_VARIABLE",
    "TRUST_VARIABLE",
    "PickledFormatNotTrustedError",
    "UnsafeRawFileError",
    "check_npz",
    "pickled_files_trusted",
    "raw_array_limit_bytes",
    "require_pickled_trust",
]

LIMIT_VARIABLE = "MOSAIC_RAW_ARRAY_LIMIT_BYTES"
"""The environment variable that sets the size limit, in bytes."""

TRUST_VARIABLE = "MOSAIC_TRUST_PICKLED_FILES"
"""The environment variable that, set to ``1``, trusts pickled files."""

_DEFAULT_LIMIT_BYTES = 2 * 1024**3
_MAX_NPZ_MEMBERS = 4096
_ZIP_LOCAL_HEADER = b"PK\x03\x04"
_ZIP_SIGNATURES = (_ZIP_LOCAL_HEADER, b"PK\x05\x06")
_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


class UnsafeRawFileError(ValueError):
    """A supplied raw file is refused because loading it would be unsafe.

    Attributes:
        path: The refused file.
        member: The archive member refused, or ``None`` for the whole file.
    """

    def __init__(self, path: Path, reason: str, *, member: str | None = None) -> None:
        where = f"{path} (member {member})" if member is not None else str(path)
        super().__init__(f"{where}: {reason}")
        self.path = path
        self.member = member


class PickledFormatNotTrustedError(UnsafeRawFileError):
    """A file in a format that loads by unpickling, without the trust flag."""


def raw_array_limit_bytes() -> int:
    """The most bytes the arrays of one supplied file may declare in total.

    Returns:
        ``MOSAIC_RAW_ARRAY_LIMIT_BYTES``, or 2 GiB when it is unset.
    """
    return int_env(LIMIT_VARIABLE, _DEFAULT_LIMIT_BYTES)


def pickled_files_trusted() -> bool:
    """Whether ``MOSAIC_TRUST_PICKLED_FILES=1`` trusts files that unpickle."""
    return flag_env(TRUST_VARIABLE)


def require_pickled_trust(path: Path, fmt: str, safe_form: str) -> None:
    """Refuse *path*, a file that loads by unpickling, unless pickles are trusted.

    Args:
        path: The file about to be loaded.
        fmt: The file's format, as the refusal names it.
        safe_form: A form of the same data that loads without unpickling, such
            as JSON or CSV.

    Raises:
        PickledFormatNotTrustedError: If `pickled_files_trusted` is false.
    """
    if pickled_files_trusted():
        return
    reason = (
        f"a {fmt} file loads by unpickling, which can run code. Convert it to "
        f"{safe_form}, or set {TRUST_VARIABLE}=1 if you trust where it came from."
    )
    raise PickledFormatNotTrustedError(path, reason)


def check_npz(path: Path) -> None:
    """Refuse an ``.npz`` that cannot be loaded safely, from its headers alone.

    No member's data is read. The file must start as a zip archive does, the
    test ``np.load`` makes. Each member must be a ``.npy`` of format 1.0 or 2.0
    whose dtype holds no Python objects, which ``np.load`` would unpickle, and
    whose shape, with no negative dimension, declares a size that fits in the
    member. The declared sizes together must fit in `raw_array_limit_bytes`.

    Args:
        path: The file to check.

    Raises:
        UnsafeRawFileError: If the file fails any of these checks.
    """
    with path.open("rb") as stream:
        leading = stream.read(len(_ZIP_LOCAL_HEADER))
    # `np.load` decides from the leading bytes, and `is_zipfile` from the end
    # record, so a file must pass both to be read as the archive checked here.
    if leading not in _ZIP_SIGNATURES or not zipfile.is_zipfile(path):
        raise UnsafeRawFileError(path, "not a zip archive, so not an .npz")
    limit = raw_array_limit_bytes()
    total = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > _MAX_NPZ_MEMBERS:
            reason = f"holds {len(members)} members, more than {_MAX_NPZ_MEMBERS}"
            raise UnsafeRawFileError(path, reason)
        for info in members:
            total += _declared_bytes(path, archive, info)
    if total > limit:
        reason = (
            f"its arrays declare {total} bytes, more than the {limit} that "
            f"{LIMIT_VARIABLE} allows"
        )
        raise UnsafeRawFileError(path, reason)


def _declared_bytes(path: Path, archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> int:
    """The bytes that member *info* declares, refusing an unsafe header."""
    member = info.filename
    if not member.endswith(".npy"):
        raise UnsafeRawFileError(path, "not an .npy array", member=member)
    with archive.open(info) as stream:
        try:
            version = np.lib.format.read_magic(stream)
        except ValueError as exc:
            raise UnsafeRawFileError(
                path, f"not an .npy array: {exc}", member=member
            ) from exc
        read_header = _HEADER_READERS.get(version)
        if read_header is None:
            reason = f"format version {version[0]}.{version[1]} is not 1.0 or 2.0"
            raise UnsafeRawFileError(path, reason, member=member)
        try:
            shape, _, dtype = read_header(stream)
        except ValueError as exc:
            reason = f"unreadable .npy header: {exc}"
            raise UnsafeRawFileError(path, reason, member=member) from exc
    if dtype.hasobject:
        reason = "holds Python objects, which load only by unpickling"
        raise UnsafeRawFileError(path, reason, member=member)
    if any(dimension < 0 for dimension in shape):
        reason = f"declares a negative dimension in shape {shape}"
        raise UnsafeRawFileError(path, reason, member=member)
    # A zero-width dtype costs memory per element once loaded, so each element
    # counts as at least one byte.
    declared = math.prod(shape) * max(dtype.itemsize, 1)
    if declared > info.file_size:
        reason = f"declares {declared} bytes but holds {info.file_size}"
        raise UnsafeRawFileError(path, reason, member=member)
    return declared
