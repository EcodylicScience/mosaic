"""Every index one inventory reads, read exactly once.

``mosaic`` holds no caching decorators anywhere, and the cost shows: one
``Dataset.reconcile`` call makes three full passes over every index in the
dataset, because three separate walks each open the files for themselves. An
inventory would make that worse -- the run enumerator, the row reader and the
drift check all want the same feature index.

This is not a store and it is not shared between calls. It is one object living
for the length of one scan, holding what it has already read so it does not read
it again a moment later. Anything longer-lived revalidates through
``cache.py``, and the stamps that makes possible are collected here as a
by-product of reading rather than by a second walk over the same files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["IndexReader", "IndexStamp"]


@dataclass(frozen=True, slots=True)
class IndexStamp:
    """What an index file looked like when it was read.

    ``(mtime_ns, size)`` rather than a digest: an index only ever grows by
    appends, so size alone catches almost every change and neither costs a read.
    A stamp for a file that was absent is still a stamp -- its appearing is a
    change, and recording nothing would make that invisible.
    """

    path: Path
    exists: bool
    mtime_ns: int = 0
    size: int = 0

    @classmethod
    def of(cls, path: Path) -> IndexStamp:
        """Stat *path* now. An absent file stamps as absent rather than raising."""
        try:
            status = path.stat()
        except OSError:
            return cls(path=path, exists=False)
        return cls(
            path=path, exists=True, mtime_ns=status.st_mtime_ns, size=status.st_size
        )


class IndexReader:
    """One scan's view of the index files, each read at most once.

    Keyed on the resolved path, so two roots reaching one file through different
    routes share the read rather than disagreeing about it.
    """

    def __init__(self) -> None:
        self._frames: dict[Path, pd.DataFrame] = {}
        self._stamps: dict[Path, IndexStamp] = {}

    def frame(self, path: Path, reader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        """The frame at *path*, read once per instance.

        *reader* is required rather than defaulted to a bare ``read_csv``,
        because every index here has a typed reader and a bare one corrupts
        cells: an all-digit ``run_id`` comes back as an integer and a blank cell
        as a float NaN. Making the caller name its reader is what stops this
        becoming the place that reintroduces that.

        Args:
            path: The index CSV, stamped whether or not the read succeeds.
            reader: Produces the frame. Expected to answer an absent file with an
                empty frame, which every typed reader in the package does.

        Returns:
            The frame, or an empty one if *reader* raised. Never an exception:
            each pass over an inventory decides what an absent index means, and
            for all of them it means "nothing here" rather than "error".
        """
        key = self._key(path)
        cached = self._frames.get(key)
        if cached is not None:
            return cached
        self._stamps[key] = IndexStamp.of(path)
        try:
            frame = reader()
        except (OSError, ValueError, KeyError):
            frame = pd.DataFrame()
        self._frames[key] = frame
        return frame

    def stamps(self) -> dict[Path, IndexStamp]:
        """What this reader touched, and what it looked like. A copy."""
        return dict(self._stamps)

    def note(self, path: Path) -> None:
        """Stamp *path* without reading it.

        For a file whose *existence* was consulted but whose contents were not,
        so a later revalidation still notices it appearing or going away.
        """
        key = self._key(path)
        if key not in self._stamps:
            self._stamps[key] = IndexStamp.of(path)

    @staticmethod
    def _key(path: Path) -> Path:
        try:
            return path.resolve()
        except OSError:
            return path
