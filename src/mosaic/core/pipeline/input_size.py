"""The size of an entry's input, read from its input files' footers.

Rows are the proxy a run's memory estimate is learned against
(:class:`~mosaic.core.pipeline.admission.RowCostEstimate`), so the run, the order its
entries start in and a report of a step's input size all count them by this one
rule: the footer rows of the entry's own input files, its neighbors left out. No
table is read. A file whose footer cannot be read counts nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mosaic.core.entry import Entry

from .manifest import Manifest, ManifestEntry
from .parquet_reader import ParquetUnreadable, read_footer

__all__ = [
    "EntryInputSize",
    "by_rows",
    "entry_input_size",
    "manifest_input_sizes",
]


@dataclass(frozen=True, slots=True)
class EntryInputSize:
    """How much one entry reads.

    Attributes:
        rows: The footer rows summed over the entry's input files, neighbors
            left out.
        columns: The columns summed over the same files.
    """

    rows: int
    columns: int


def entry_input_size(entry: ManifestEntry) -> EntryInputSize:
    """Return the rows and columns of *entry*'s input files, from their footers.

    Args:
        entry: The manifest entry. Its neighbors' files are not counted.

    Returns:
        The sums over its readable files. An unreadable file adds nothing.
    """
    rows = 0
    columns = 0
    for path, _spec in entry.file_specs:
        try:
            footer = read_footer(path)
        except ParquetUnreadable:
            continue
        rows += footer.n_rows
        columns += len(footer.columns)
    return EntryInputSize(rows=rows, columns=columns)


def manifest_input_sizes(manifest: Manifest) -> dict[str, EntryInputSize]:
    """Return :func:`entry_input_size` of every entry, keyed as *manifest* is."""
    return {key: entry_input_size(entry) for key, entry in manifest.items()}


def by_rows(rows: Mapping[Entry, int]) -> list[Entry]:
    """Return the entries of *rows* largest first, ties by ``(group, sequence)``."""
    return sorted(rows, key=lambda entry: (-rows[entry], entry))
