"""The memo of parquet footers that a run and an inventory scan share."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline import parquet_reader
from mosaic.core.pipeline.parquet_reader import ParquetFooterMemo, ParquetUnreadable


def _write(path: Path, columns: dict[str, list[int]]) -> None:
    pd.DataFrame(columns).to_parquet(path, index=False)


def _count_parses(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record each file that the memo reads from disk."""
    parsed: list[Path] = []
    real = parquet_reader.parse_footer

    def counted(path: Path) -> parquet_reader.ParquetFooter:
        parsed.append(path)
        return real(path)

    monkeypatch.setattr(parquet_reader, "parse_footer", counted)
    return parsed


def test_two_footer_calls_on_one_file_parse_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1, 2, 3]})
    parsed = _count_parses(monkeypatch)
    memo = ParquetFooterMemo()

    first = memo.footer(path)
    again = memo.footer(path)

    assert first is again
    assert first.n_rows == 3
    assert [column.name for column in first.columns] == ["x"]
    assert parsed == [path]


def test_a_rewrite_with_another_size_parses_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1]})
    memo = ParquetFooterMemo()
    _ = memo.footer(path)
    parsed = _count_parses(monkeypatch)
    stamp = path.stat()

    _write(path, {"x": list(range(500))})
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

    assert memo.footer(path).n_rows == 500
    assert parsed == [path]


def test_a_rewrite_with_another_modification_time_parses_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1, 2]})
    memo = ParquetFooterMemo()
    _ = memo.footer(path)
    parsed = _count_parses(monkeypatch)
    stamp = path.stat()

    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))

    assert memo.footer(path).n_rows == 2
    assert parsed == [path]


def test_a_file_replaced_by_a_rename_parses_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    other = tmp_path / "other.parquet"
    _write(path, {"x": [1, 2]})
    _write(other, {"y": [3, 4]})
    stamp = path.stat()
    assert other.stat().st_size == stamp.st_size
    os.utime(other, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    memo = ParquetFooterMemo()
    assert [column.name for column in memo.footer(path).columns] == ["x"]
    parsed = _count_parses(monkeypatch)

    _ = other.replace(path)

    assert path.stat().st_size == stamp.st_size
    assert path.stat().st_mtime_ns == stamp.st_mtime_ns
    assert [column.name for column in memo.footer(path).columns] == ["y"]
    assert parsed == [path]


def test_a_torn_file_raises_and_is_parsed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _ = path.write_bytes(b"torn")
    parsed = _count_parses(monkeypatch)
    memo = ParquetFooterMemo()

    with pytest.raises(ParquetUnreadable):
        _ = memo.footer(path)
    with pytest.raises(ParquetUnreadable):
        _ = memo.footer(path)

    assert parsed == [path, path]


def test_an_absent_file_raises_from_the_stat(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        _ = ParquetFooterMemo().footer(tmp_path / "never.parquet")


def test_files_of_one_schema_share_their_column_tuple(tmp_path: Path) -> None:
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    _write(first, {"x": [1], "y": [2]})
    _write(second, {"x": [3, 4], "y": [5, 6]})
    memo = ParquetFooterMemo()

    footer_a = memo.footer(first)
    footer_b = memo.footer(second)

    assert footer_a.columns is footer_b.columns
    assert footer_a.n_rows == 1
    assert footer_b.n_rows == 2
