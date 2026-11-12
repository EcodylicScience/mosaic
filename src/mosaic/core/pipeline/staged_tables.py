"""Tracks tables that a worker writes and the process that owns the index publishes.

A worker writes each table beside its final name, under a hidden name that ends in
``.staged``, so that no reader, scan or glob for ``*.parquet`` takes it for a table.
The process that owns the index then moves each batch of tables into place and writes
their rows in one step, in the order of the raw index. Two consequences follow, and
both are what a run with one worker gives:

* A table on disk has its row, apart from the moment between the move and the write
  of its batch. A run killed earlier leaves staged files, which :func:`sweep_staged`
  removes at the start of the next run of the variant, and a table without a row is
  made again.
* Two files that name one entry are settled by the order of the raw index, not by
  which worker finished last. The first keeps the entry unless the run overwrites, and
  then the last replaces it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd

from mosaic.core.helpers import make_entry_key, text_cell

from .entry_pool import EntryFailure
from .pacing import Pacer
from .tracks_index import TracksRowDraft, refuse_colliding_drafts, write_tracks_rows
from .writers import write_parquet_atomic

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "STAGED_SUFFIX",
    "StagedTableLost",
    "StagedTrackTable",
    "TableCommitter",
    "stage_table",
    "sweep_staged",
]

STAGED_SUFFIX = ".staged"
"""The ending of a staged file's name. It is not ``.parquet``."""

_STAGED_PREFIX = ".stage-"

_FLUSH_COST_FACTOR = 4.0
"""How many times the cost of a flush a batch waits before the next one."""


class StagedTableLost(RuntimeError):
    """A staged table was gone when its owner came to publish it."""


@dataclass(frozen=True, slots=True)
class StagedTrackTable:
    """A table written under its staging name, and the row that describes it.

    Attributes:
        staged: Where the table is.
        row: The index row. Its ``out_path`` is where the table belongs.
    """

    staged: Path
    row: TracksRowDraft


def stage_table(df: pd.DataFrame, final: Path) -> Path:
    """Write *df* beside *final* under a staging name, and return that name."""
    staged = final.with_name(f"{_STAGED_PREFIX}{uuid4().hex}{STAGED_SUFFIX}")
    _ = write_parquet_atomic(df, staged)
    return staged


def sweep_staged(variant_root: Path) -> int:
    """Remove the staged files in *variant_root*, and return how many there were.

    Called at the start of a run of the variant, when the files of an earlier run
    that was killed are the only ones there. A variant directory that does not
    exist has none.
    """
    removed = 0
    if not variant_root.is_dir():
        return removed
    for path in variant_root.glob(f"{_STAGED_PREFIX}*{STAGED_SUFFIX}"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


class TableCommitter:
    """Publish staged tables and write their rows, in the order they are given.

    Tables are added and published in batches. A flush moves the batch's tables into
    place and then writes all their rows in one read-modify-write of the index, so
    the cost of a run in the index is one pass for each batch, not one for each
    table. A table is published, and its row written, by the flush that follows its
    addition, which happens when the batch reaches ``max_batch`` tables, when the
    longer of ``interval`` seconds and four times the duration of the last flush has
    passed since that flush ended (checked as tables are added), or when the caller
    flushes. The caller flushes at the end of a run, and when it
    stops early.

    A run killed between flushes leaves the batch's tables staged or in place
    without their rows, and a table without a row is made again by the next run.

    Stretching the wait with the duration of the last flush keeps writing the index
    to at most a fifth of a long run however large the index grows. A batch that
    names a colliding entry is refused before any table moves.

    Args:
        ds: The dataset whose tracks index receives the rows.
        overwrite: Whether a later table of an entry replaces the one a run
            published before it. Without it the first table keeps the entry and the
            later one is discarded. The order is that of :meth:`add`, within a
            batch and across batches.
        max_batch: The number of tables that makes a batch due.
        interval: The least seconds between the end of one flush and the next
            batch being due.
        clock: The clock that measures the interval. ``None`` reads
            :func:`~mosaic.core.pipeline.pacing.now`.
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        overwrite: bool,
        max_batch: int = 1000,
        interval: float = 10.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ds: Dataset = ds
        self._overwrite: bool = overwrite
        self._max_batch: int = max_batch
        self._pacer: Pacer = Pacer(
            interval=interval, cost_factor=_FLUSH_COST_FACTOR, clock=clock
        )
        self._published: set[Path] = set()
        self._pending: list[StagedTrackTable] = []

    def add(self, tables: Iterable[StagedTrackTable]) -> list[EntryFailure]:
        """Add *tables* to the batch, and publish the batch when it is due.

        Returns:
            The failures of the tables a flush found gone, as :meth:`flush` does.
        """
        self._pending.extend(tables)
        if len(self._pending) >= self._max_batch or (
            self._pending and self._pacer.due()
        ):
            return self.flush()
        return []

    def flush(self) -> list[EntryFailure]:
        """Move each table of the batch into place and write their rows, or discard them.

        Returns:
            One failure for each table that was gone, which a concurrent run of the
            same variant removes when it sweeps the staged files. The remaining
            tables are published.
        """
        batch, self._pending = self._pending, []
        if not batch:
            return []
        return self._pacer.timed(lambda: self._publish(batch))

    def _publish(self, batch: list[StagedTrackTable]) -> list[EntryFailure]:
        """Move the tables of *batch* into place, and write the rows of those kept.

        What is kept is settled first and checked for key collisions before any
        table moves, so that a refused batch leaves only staged files.
        """
        lost: list[EntryFailure] = []
        discarded: list[StagedTrackTable] = []
        kept: list[StagedTrackTable] = []
        claimed = set(self._published)
        for table in batch:
            final = table.row.out_path
            if final in claimed and not self._overwrite:
                discarded.append(table)
            elif not table.staged.exists():
                lost.append(self._lost(table, FileNotFoundError(str(table.staged))))
            else:
                kept.append(table)
                claimed.add(final)
        refuse_colliding_drafts(self._ds, [table.row for table in kept])
        for table in discarded:
            table.staged.unlink(missing_ok=True)
        drafts: list[TracksRowDraft] = []
        for table in kept:
            try:
                os.replace(table.staged, table.row.out_path)
            except FileNotFoundError as exc:
                if table.staged.exists():
                    raise
                lost.append(self._lost(table, exc))
                continue
            drafts.append(table.row)
            self._published.add(table.row.out_path)
        write_tracks_rows(self._ds, drafts)
        return lost

    @staticmethod
    def _lost(table: StagedTrackTable, cause: FileNotFoundError) -> EntryFailure:
        """The failure of an entry whose staged table was removed."""
        key = make_entry_key(text_cell(table.row.group), text_cell(table.row.sequence))
        message = (
            f"the staged table of entry {key} was removed before it was published. "
            f"A concurrent run of variant {table.row.run_id} removes the staged "
            "files of the variant when it starts."
        )
        try:
            raise StagedTableLost(message) from cause
        except StagedTableLost as lost:
            return EntryFailure.capture(key, lost)
