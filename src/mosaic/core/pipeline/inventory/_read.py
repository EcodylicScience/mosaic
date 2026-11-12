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

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mosaic.core.pipeline.index_stamp import IndexStamp


__all__ = ["IndexReader", "ParquetFooterMemo", "parquet_is_readable"]


def parquet_is_readable(path: Path) -> bool:
    """Is this parquet whole? Footer-only, so it costs one seek rather than a read.

    A parquet's footer is written last, so a file torn by an interrupted write
    has no readable metadata. That is cheap enough to run per file on a gate
    while still catching the failure presence alone cannot.
    """
    import pyarrow.parquet as pq

    try:
        # pyarrow ships no type stubs, the same gap ``track_universe`` documents
        # at its own ``read_schema`` call. Only whether the footer parsed
        # matters, so the result is discarded rather than described.
        pq.read_metadata(path)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
    except Exception:
        return False
    return True


class ParquetFooterMemo:
    """Which parquet files of a directory parsed, by path, size and modification time.

    A feature run holds one output per entry, and reading every footer again on
    each scan of a dataset of thousands of entries costs seconds. A file that was
    rewritten has another modification time or size and is read again, so a torn
    output is still seen. Each directory keeps what its latest scan found, so a
    deleted output is forgotten. Held by one :class:`~.cache.InventoryCache` and
    dropped with it.
    """

    def __init__(self) -> None:
        self._known: dict[Path, dict[str, tuple[int, int, bool]]] = {}

    def whole_stems(self, root: Path) -> set[str]:
        """The stems of the parquet files of *root* whose footer parses.

        Each file's footer is read only when its size or modification time is not
        the one the previous scan of *root* saw.
        """
        before = self._known.get(root, {})
        after: dict[str, tuple[int, int, bool]] = {}
        stems: set[str] = set()
        with os.scandir(root) as listing:
            for item in listing:
                if not item.name.endswith(".parquet"):
                    continue
                try:
                    state = item.stat()
                except OSError:
                    continue
                seen = before.get(item.name)
                if seen is not None and seen[:2] == (state.st_size, state.st_mtime_ns):
                    parsed = seen[2]
                else:
                    parsed = parquet_is_readable(Path(item.path))
                after[item.name] = (state.st_size, state.st_mtime_ns, parsed)
                if parsed:
                    stems.add(Path(item.name).stem)
        self._known[root] = after
        return stems


class IndexReader:
    """One scan's view of the index files, each read at most once.

    Keyed on the resolved path, so two roots reaching one file through different
    routes share the read rather than disagreeing about it. *footers* is the
    memo of parquet footers that a holder of several scans keeps between them.
    """

    def __init__(self, footers: ParquetFooterMemo | None = None) -> None:
        self.footers: ParquetFooterMemo = (
            ParquetFooterMemo() if footers is None else footers
        )
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
