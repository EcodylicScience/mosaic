"""Tracks tables that a worker writes and the process that owns the index publishes.

A worker writes each table beside its final name, under a hidden name that ends in
``.staged``, so that no reader, scan or glob for ``*.parquet`` takes it for a table.
The process that owns the index then moves it into place and writes its row in one
step, in the order of the raw index. Two consequences follow, and both are what a run
with one worker gives:

* A table on disk has its row, apart from the moment between the move and the write.
  A run killed earlier leaves staged files, which :func:`sweep_staged` removes at the
  start of the next run of the variant, and a table without a row is made again.
* Two files that name one entry are settled by the order of the raw index, not by
  which worker finished last. The first keeps the entry unless the run overwrites, and
  then the last replaces it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd

from mosaic.core.helpers import make_entry_key, text_cell

from .entry_pool import EntryFailure
from .tracks_index import TracksRowDraft
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

    Args:
        ds: The dataset whose tracks index receives the rows.
        overwrite: Whether a later table of an entry replaces the one a run
            published before it. Without it the first table keeps the entry and the
            later one is discarded.
    """

    def __init__(self, ds: Dataset, *, overwrite: bool) -> None:
        self._ds: Dataset = ds
        self._overwrite: bool = overwrite
        self._published: set[Path] = set()

    def commit(self, tables: Iterable[StagedTrackTable]) -> list[EntryFailure]:
        """Move each table into place and write its row, or discard it.

        Returns:
            One failure for each table that was gone, which a concurrent run of the
            same variant removes when it sweeps the staged files. The remaining
            tables are published.
        """
        lost: list[EntryFailure] = []
        for table in tables:
            final = table.row.out_path
            if final in self._published and not self._overwrite:
                table.staged.unlink(missing_ok=True)
                continue
            try:
                os.replace(table.staged, final)
            except FileNotFoundError as exc:
                if table.staged.exists():
                    raise
                lost.append(self._lost(table, exc))
                continue
            table.row.write(self._ds)
            self._published.add(final)
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
