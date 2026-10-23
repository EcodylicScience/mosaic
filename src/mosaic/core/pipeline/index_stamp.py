"""Detect a change to an index file between two reads by comparing its stamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["IndexStamp"]


@dataclass(frozen=True, slots=True)
class IndexStamp:
    """The modification time and size of an index file when it was read.

    A stamp costs a stat, where a digest costs a read of the file. A rewrite that
    keeps the file's size is missed when the file system records the same
    modification time for it. A file that was absent stamps as absent, and its
    appearing is then a change.
    """

    path: Path
    exists: bool
    mtime_ns: int = 0
    size: int = 0

    @classmethod
    def of(cls, path: Path) -> IndexStamp:
        """Stat *path* now. Return an absent stamp for a missing file."""
        try:
            status = path.stat()
        except OSError:
            return cls(path=path, exists=False)
        return cls(
            path=path, exists=True, mtime_ns=status.st_mtime_ns, size=status.st_size
        )
