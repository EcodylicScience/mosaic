"""The input size of an entry, read from its input files' footers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.pipeline.input_size import (
    EntryInputSize,
    by_rows,
    entry_input_size,
    manifest_input_sizes,
)
from mosaic.core.pipeline.manifest import ManifestEntry
from mosaic.core.pipeline.types import ParquetLoadSpec


def _table(path: Path, rows: int, columns: int) -> Path:
    pd.DataFrame({f"c{i}": range(rows) for i in range(columns)}).to_parquet(path)
    return path


def test_an_entry_sums_rows_and_columns_over_its_inputs_and_skips_neighbors(
    tmp_path: Path,
) -> None:
    first = _table(tmp_path / "a.parquet", rows=10, columns=3)
    second = _table(tmp_path / "b.parquet", rows=4, columns=2)
    neighbor = _table(tmp_path / "n.parquet", rows=1000, columns=9)
    entry = ManifestEntry(
        file_specs=[(first, ParquetLoadSpec()), (second, ParquetLoadSpec())],
        prev_file_specs=[(neighbor, ParquetLoadSpec())],
        next_file_specs=[(neighbor, ParquetLoadSpec())],
    )

    assert entry_input_size(entry) == EntryInputSize(rows=14, columns=5)


def test_an_unreadable_footer_counts_no_rows(tmp_path: Path) -> None:
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not a parquet file")
    readable = _table(tmp_path / "a.parquet", rows=6, columns=2)
    entry = ManifestEntry(
        file_specs=[
            (broken, ParquetLoadSpec()),
            (tmp_path / "absent.parquet", ParquetLoadSpec()),
            (readable, ParquetLoadSpec()),
        ]
    )

    assert entry_input_size(entry) == EntryInputSize(rows=6, columns=2)


def test_a_manifest_is_sized_by_entry_key(tmp_path: Path) -> None:
    manifest = {
        "g__a": ManifestEntry(
            file_specs=[(_table(tmp_path / "a.parquet", 3, 1), ParquetLoadSpec())]
        ),
        "g__b": ManifestEntry(
            file_specs=[(_table(tmp_path / "b.parquet", 5, 2), ParquetLoadSpec())]
        ),
    }

    assert manifest_input_sizes(manifest) == {
        "g__a": EntryInputSize(rows=3, columns=1),
        "g__b": EntryInputSize(rows=5, columns=2),
    }


def test_entries_go_largest_first_and_ties_go_by_group_then_sequence() -> None:
    rows = {
        ("g2", "a"): 10,
        ("g1", "b"): 10,
        ("g1", "a"): 10,
        ("g0", "z"): 3,
        ("g9", "y"): 50,
    }

    assert by_rows(rows) == [
        ("g9", "y"),
        ("g1", "a"),
        ("g1", "b"),
        ("g2", "a"),
        ("g0", "z"),
    ]
