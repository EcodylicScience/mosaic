"""The memo of parquet footers that a run and an inventory scan share."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline import parquet_reader
from mosaic.core.pipeline.parquet_reader import (
    ParquetFooterMemo,
    ParquetUnreadable,
    memoized_footers,
    output_n_rows,
    read_footer,
    read_parquet_column_names,
)
from tests.helpers import record_path_calls


def _write(path: Path, columns: dict[str, list[int]]) -> None:
    pd.DataFrame(columns).to_parquet(path, index=False)


def test_two_footer_calls_on_one_file_parse_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1, 2, 3]})
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")
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
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")
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
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")
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
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

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
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")
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


def _read_three_ways(path: Path) -> None:
    _ = read_footer(path)
    _ = read_parquet_column_names(path)
    _ = output_n_rows(path)


def test_the_reads_of_one_file_parse_it_once_under_a_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1, 2, 3]})
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    with memoized_footers(ParquetFooterMemo()):
        _read_three_ways(path)
        assert read_parquet_column_names(path) == ("x",)
        assert output_n_rows(path) == 3

    assert parsed == [path]


def test_each_read_parses_with_no_memo_entered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1, 2, 3]})
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    _read_three_ways(path)

    assert parsed == [path, path, path]


def test_a_read_after_the_block_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1]})
    memo = ParquetFooterMemo()
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    with memoized_footers(memo):
        _ = read_footer(path)
    _ = read_footer(path)
    _ = read_footer(path)

    assert parsed == [path, path, path]


def test_a_block_that_raises_resets_the_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1]})
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    with pytest.raises(RuntimeError), memoized_footers(ParquetFooterMemo()):
        _ = read_footer(path)
        raise RuntimeError

    _ = read_footer(path)

    assert parsed == [path, path]


def test_a_nested_block_restores_the_outer_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1]})
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    with memoized_footers(ParquetFooterMemo()):
        _ = read_footer(path)
        with memoized_footers(ParquetFooterMemo()):
            _ = read_footer(path)
        _ = read_footer(path)

    assert parsed == [path, path]


def test_a_new_thread_parses_what_the_block_has_memoized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new thread starts without the context on CPython 3.13.

    A free-threaded 3.14 build with ``thread_inherit_context`` would differ.
    """
    path = tmp_path / "a.parquet"
    _write(path, {"x": [1]})
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    with memoized_footers(ParquetFooterMemo()):
        _ = read_footer(path)
        assert parsed == [path]

        worker = threading.Thread(target=read_footer, args=(path,))
        worker.start()
        worker.join()
        assert parsed == [path, path]

        with ThreadPoolExecutor(max_workers=1) as pool:
            _ = pool.submit(read_footer, path).result()
        assert parsed == [path, path, path]

        _ = read_footer(path)
        assert parsed == [path, path, path]


def _assert_absent_file_is_unreadable(path: Path) -> None:
    assert read_parquet_column_names(path) is None
    with pytest.raises(ParquetUnreadable):
        _ = read_footer(path)
    with pytest.raises(ParquetUnreadable):
        _ = output_n_rows(path)


def test_an_absent_file_is_unreadable_with_no_memo(tmp_path: Path) -> None:
    _assert_absent_file_is_unreadable(tmp_path / "never.parquet")


def test_an_absent_file_is_unreadable_under_a_memo(tmp_path: Path) -> None:
    with memoized_footers(ParquetFooterMemo()):
        _assert_absent_file_is_unreadable(tmp_path / "never.parquet")


def test_a_torn_file_under_a_memo_raises_and_is_parsed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    _ = path.write_bytes(b"torn")
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    with memoized_footers(ParquetFooterMemo()):
        with pytest.raises(ParquetUnreadable):
            _ = read_footer(path)
        with pytest.raises(ParquetUnreadable):
            _ = read_footer(path)
        assert read_parquet_column_names(path) is None

    assert parsed == [path, path, path]


@pytest.fixture
def unsearchable_file(tmp_path: Path) -> Iterator[Path]:
    """A parquet whose directory denies search, so that its path cannot be examined."""
    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("directory permissions do not deny this user")
    folder = tmp_path / "closed"
    folder.mkdir()
    path = folder / "a.parquet"
    _write(path, {"x": [1]})
    folder.chmod(0o600)
    try:
        yield path
    finally:
        folder.chmod(0o700)


def test_a_path_that_cannot_be_examined_raises_the_same_with_and_without_a_memo(
    unsearchable_file: Path,
) -> None:
    with pytest.raises(PermissionError):
        _ = read_parquet_column_names(unsearchable_file)

    with memoized_footers(ParquetFooterMemo()), pytest.raises(PermissionError):
        _ = read_parquet_column_names(unsearchable_file)


def test_read_footer_reports_a_path_that_cannot_be_examined_as_unreadable(
    unsearchable_file: Path,
) -> None:
    with pytest.raises(ParquetUnreadable):
        _ = read_footer(unsearchable_file)

    with memoized_footers(ParquetFooterMemo()), pytest.raises(ParquetUnreadable):
        _ = read_footer(unsearchable_file)
