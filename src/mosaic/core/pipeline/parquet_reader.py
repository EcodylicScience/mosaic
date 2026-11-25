"""Typed reads of one parquet file through pyarrow: its footer, numeric summaries and head.

pyarrow does not ship type stubs, and its compiled modules leave every attribute
unknown to the type checker. This module holds every footer read of the
pipeline and the preview's reads, each returning plain Python values, and the
release of pyarrow's unused memory after an entry, so its callers type-check in
full. The file-level directives below cover the unknown types of pyarrow alone.

A footer read opens the file, unless the thread entered :func:`memoized_footers`,
which reads each file's footer once until the file changes. A memo hands the
footers it parsed to another thread or process as :class:`FooterRecord` values,
and a memo built from them trusts each footer while its file's stamp holds.
Every other read opens the file and reads only the columns it names. None of
them builds a pandas frame. A file that pyarrow cannot read raises
:class:`ParquetUnreadable`.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from typing import Any

    import numpy.typing as npt

__all__ = [
    "ArrowKind",
    "FooterColumn",
    "FooterRecord",
    "NumericSummary",
    "ParquetFooter",
    "ParquetFooterMemo",
    "ParquetUnreadable",
    "iter_column_batches",
    "memoized_footers",
    "output_n_rows",
    "parquet_is_readable",
    "parse_footer",
    "read_footer",
    "read_head",
    "read_parquet_column_names",
    "release_arrow_memory",
    "summarize_numeric_columns",
]

type ArrowKind = Literal["integer", "floating", "boolean", "string", "other"]
"""The family of a column's Arrow type. A dictionary column takes its values' family."""


class ParquetUnreadable(Exception):
    """A file whose footer or data pyarrow cannot read, as an interrupted write leaves it."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"{path} is not a readable parquet file: {cause}")
        self.path: Path = path


@dataclass(frozen=True, slots=True)
class FooterColumn:
    """One column of a parquet footer's schema.

    Attributes:
        name: The column name.
        kind: The family of its Arrow type.
        arrow_type: The Arrow type as pyarrow writes it, such as ``int64``.
    """

    name: str
    kind: ArrowKind
    arrow_type: str


@dataclass(frozen=True, slots=True)
class ParquetFooter:
    """A parquet file's schema and row count, from its footer.

    Attributes:
        columns: Every column, in table order.
        n_rows: The file's row count.
    """

    columns: tuple[FooterColumn, ...]
    n_rows: int


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """One numeric column reduced over every row of the file.

    Attributes:
        min: The smallest finite value, exact in the column's own type: an
            ``int`` for an integer or boolean column. ``None`` without one.
        max: The largest finite value, under the same rule.
        total: The sum of the finite values, accumulated in float64.
        n_finite: How many values are finite.
        n_missing: How many values are NaN or null.
    """

    min: int | float | None
    max: int | float | None
    total: float
    n_finite: int
    n_missing: int


@contextmanager
def _reading(path: Path) -> Generator[None]:
    """Raise :class:`ParquetUnreadable` for an error pyarrow raises reading *path*."""
    try:
        yield
    except (OSError, pa.ArrowException) as error:
        raise ParquetUnreadable(path, error) from error


def parse_footer(path: Path) -> ParquetFooter:
    """Read the schema and the row count from the footer of the file at *path*.

    The read always opens the file, whatever memo is entered. :func:`read_footer`
    calls it, and so does :class:`ParquetFooterMemo`, which is the caller to
    use where a file may be read more than once.
    """
    with _reading(path):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        n_rows = int(parquet.metadata.num_rows)
    columns = tuple(
        FooterColumn(
            name=str(field.name),
            kind=_kind(field.type),
            arrow_type=str(field.type),
        )
        for field in schema
    )
    return ParquetFooter(columns=columns, n_rows=n_rows)


def read_footer(path: Path) -> ParquetFooter:
    """Return the schema and the row count that *path*'s footer records.

    Inside :func:`memoized_footers` the footer is parsed only when the file moved
    since the memo last read it. A file that cannot be examined raises
    :class:`ParquetUnreadable`, as one that cannot be opened does.
    """
    memo = _MEMO.get()
    if memo is None:
        return parse_footer(path)
    with _reading(path):
        return memo.footer(path)


def parquet_is_readable(path: Path) -> bool:
    """Is this parquet whole? Footer-only, so it costs one seek rather than a read.

    A parquet's footer is written last, so a file torn by an interrupted write
    has no readable metadata. That is cheap enough to run per file on a gate
    while still catching the failure presence alone cannot.
    """
    try:
        # Opening the file parses its footer, which is all that is asked.
        pq.ParquetFile(path).close()
    except Exception:
        return False
    return True


@dataclass(frozen=True, slots=True)
class _Known:
    """What the memo knows of one file: its stamp, and its footer once read.

    Attributes:
        inode: The file's inode.
        size: The file's size in bytes.
        mtime_ns: The file's modification time in nanoseconds.
        parsed: Whether the footer parsed.
        footer: The parsed footer, or ``None`` where only a scan has seen the
            file or the footer did not parse.
    """

    inode: int
    size: int
    mtime_ns: int
    parsed: bool
    footer: ParquetFooter | None

    def stamped(self, inode: int, size: int, mtime_ns: int) -> bool:
        """Does a file with this inode, size and modification time match the entry?"""
        return (self.inode, self.size, self.mtime_ns) == (inode, size, mtime_ns)


@dataclass(frozen=True, slots=True)
class FooterRecord:
    """The hand-over of one parsed footer to a worker, with the stamp it was read at.

    A memo built from records (:meth:`ParquetFooterMemo.from_records`) returns
    the footer without a parse while the file's stamp matches. A file whose
    stamp moved is parsed again.

    Attributes:
        path: The file, as the memo that parsed it was asked for it.
        inode: The file's inode when the footer was read.
        size: The file's size in bytes when the footer was read.
        mtime_ns: The file's modification time in nanoseconds when the footer
            was read.
        footer: The parsed footer.
    """

    path: Path
    inode: int
    size: int
    mtime_ns: int
    footer: ParquetFooter


class ParquetFooterMemo:
    """The parquet files of a directory that parsed, and their footers, by stamp.

    A feature run holds one output per entry, and reading every footer again on
    each scan of a dataset of thousands of entries costs seconds. A file is known
    by its inode, size and modification time. A file that was rewritten or
    replaced has another stamp and is read again, so a torn output is still seen.
    Each directory keeps what its latest scan found, so a deleted output is
    forgotten. A holder keeps one memo for as long as it wants the footers
    remembered and drops it to forget them.

    A file rewritten in place with the same inode and size within one tick of a
    coarse filesystem clock, whole seconds on some network shares, is not seen as
    moved.
    """

    def __init__(self) -> None:
        self._known: dict[Path, dict[str, _Known]] = {}
        self._columns: dict[tuple[FooterColumn, ...], tuple[FooterColumn, ...]] = {}

    @classmethod
    def from_records(cls, records: Iterable[FooterRecord]) -> ParquetFooterMemo:
        """A memo that knows each record's footer by the stamp the record holds.

        :meth:`footer` returns a record's footer without a parse while the
        file's inode, size and modification time match the record's.
        """
        memo = cls()
        for record in records:
            memo._known.setdefault(record.path.parent, {})[record.path.name] = _Known(
                inode=record.inode,
                size=record.size,
                mtime_ns=record.mtime_ns,
                parsed=True,
                footer=record.footer,
            )
        return memo

    def records(self, paths: Iterable[Path]) -> tuple[FooterRecord, ...]:
        """The records of the footers this memo parsed among *paths*, in their order.

        A path the memo only scanned, failed to parse or never saw has no
        record. A path matches only as spelled when the memo read it, so
        another spelling of the same file has no record. No file is read.
        """
        found: list[FooterRecord] = []
        for path in paths:
            known = self._known.get(path.parent, {}).get(path.name)
            if known is None or known.footer is None:
                continue
            found.append(
                FooterRecord(
                    path=path,
                    inode=known.inode,
                    size=known.size,
                    mtime_ns=known.mtime_ns,
                    footer=known.footer,
                )
            )
        return tuple(found)

    def whole_stems(self, root: Path) -> set[str]:
        """The stems of the parquet files of *root* whose footer parses.

        Each file's footer is read only when its stamp is not the one the
        previous scan of *root* saw. A file whose stamp did not move keeps its
        entry, parsed footer included.
        """
        before = self._known.get(root, {})
        after: dict[str, _Known] = {}
        stems: set[str] = set()
        with os.scandir(root) as listing:
            for item in listing:
                if not item.name.endswith(".parquet"):
                    continue
                try:
                    state = item.stat()
                    inode = item.inode()
                except OSError:
                    continue
                seen = before.get(item.name)
                if seen is not None and seen.stamped(
                    inode, state.st_size, state.st_mtime_ns
                ):
                    known = seen
                else:
                    known = _Known(
                        inode=inode,
                        size=state.st_size,
                        mtime_ns=state.st_mtime_ns,
                        parsed=parquet_is_readable(Path(item.path)),
                        footer=None,
                    )
                after[item.name] = known
                if known.parsed:
                    stems.add(Path(item.name).stem)
        self._known[root] = after
        return stems

    def footer(self, path: Path) -> ParquetFooter:
        """The footer of *path*, parsed only when the file moved since the last read.

        Raises:
            OSError: If *path* cannot be examined.
            ParquetUnreadable: If the footer does not parse. A failed read stores
                nothing, so the next call reads the file again.
        """
        state = path.stat()
        inode, size, mtime_ns = state.st_ino, state.st_size, state.st_mtime_ns
        seen = self._known.get(path.parent, {}).get(path.name)
        if (
            seen is not None
            and seen.footer is not None
            and seen.stamped(inode, size, mtime_ns)
        ):
            return seen.footer
        parsed = parse_footer(path)
        footer = ParquetFooter(
            columns=self._columns.setdefault(parsed.columns, parsed.columns),
            n_rows=parsed.n_rows,
        )
        self._known.setdefault(path.parent, {})[path.name] = _Known(
            inode=inode, size=size, mtime_ns=mtime_ns, parsed=True, footer=footer
        )
        return footer


_MEMO: Final[ContextVar[ParquetFooterMemo | None]] = ContextVar(
    "mosaic_parquet_footer_memo", default=None
)


@contextmanager
def memoized_footers(memo: ParquetFooterMemo) -> Generator[None]:
    """Route :func:`read_footer` through *memo* for the block, in this thread or task.

    The memo belongs to the context that entered it. A thread that the block
    starts does not share it and reads each file. The block restores the memo
    found on entry when it exits.
    """
    token = _MEMO.set(memo)
    try:
        yield
    finally:
        _MEMO.reset(token)


def read_parquet_column_names(path: Path) -> tuple[str, ...] | None:
    """The column names of a parquet, read from its footer and not its data.

    ``None`` for an absent file and for one whose footer does not parse, as an
    interrupted write leaves it. A caller asking what a table contains treats
    that as unknown. A path that cannot be examined for another reason, such as a
    directory that denies search, raises the :class:`OSError` that
    :meth:`pathlib.Path.exists` raises for it, with or without a memo.
    """
    memo = _MEMO.get()
    if memo is None and not path.exists():
        return None
    try:
        footer = read_footer(path)
    except ParquetUnreadable:
        if memo is not None:
            # The memo's stat replaced the check above. Ask Path.exists about
            # the failed path so that it raises what it raises without a memo.
            _ = path.exists()
        return None
    return tuple(column.name for column in footer.columns)


def output_n_rows(out_path: Path) -> int:
    """Footer-only row count for an existing output parquet (fast)."""
    return read_footer(out_path).n_rows


def summarize_numeric_columns(
    path: Path, names: list[str]
) -> dict[str, NumericSummary]:
    """Return a :class:`NumericSummary` of each named numeric column of *path*.

    Each column is read alone, one row group at a time, so the memory a summary
    takes is one column's row group, whatever the width or length of the table.
    """
    if not names:
        return {}
    with _reading(path):
        parquet = pq.ParquetFile(path)
        return {name: _summarize(parquet, name) for name in names}


def read_head(path: Path, names: list[str], limit: int) -> list[tuple[object, ...]]:
    """Return the first *limit* rows of the named columns, in file order, as Python values."""
    if limit <= 0 or not names:
        return []
    columns: list[list[object]] = [[] for _ in names]
    count = 0
    with _reading(path):
        batches = pq.ParquetFile(path).iter_batches(batch_size=limit, columns=names)
        for batch in batches:
            taken = batch.slice(0, limit - count)
            for cells, name in zip(columns, names, strict=True):
                cells.extend(taken.column(name).to_pylist())
            count += int(taken.num_rows)
            if count >= limit:
                break
    return list(zip(*columns, strict=True))


def iter_column_batches(
    path: Path, names: Sequence[str], batch_size: int
) -> Generator[dict[str, npt.NDArray[Any]]]:
    """Yield the named columns of *path* as numpy arrays, *batch_size* rows at a time.

    Only the named columns are read, and a batch holds at most *batch_size* rows
    of them. The batch size does not bound the memory of the read: pyarrow decodes
    a whole row group of the named columns, about twice their float64 size, so a
    caller that reads a long file names few columns at a time. A column that
    holds a null arrives as floating point with NaN for it, and a boolean column
    as ``bool``. The batches follow the file's row order. *names* must not repeat
    a column.
    """
    if not names:
        return
    with _reading(path):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(names)):
            yield {
                name: batch.column(name).to_numpy(zero_copy_only=False)
                for name in names
            }


def release_arrow_memory() -> None:
    """Return the memory pyarrow's default pool holds unused to the system.

    Called after each entry's table or output is released, so a long run's
    process does not keep the largest entry's buffers.
    """
    pa.default_memory_pool().release_unused()


def _summarize(parquet: pq.ParquetFile, name: str) -> NumericSummary:
    """Reduce column *name* of *parquet* batch by batch to a :class:`NumericSummary`."""
    low: int | float | None = None
    high: int | float | None = None
    total = 0.0
    n_finite = 0
    n_missing = 0
    for batch in parquet.iter_batches(columns=[name]):
        values = batch.column(0)
        if pa.types.is_dictionary(values.type):
            values = values.dictionary_decode()
        if pa.types.is_boolean(values.type):
            values = values.cast(pa.int8())
        n_missing += int(values.null_count)
        if pa.types.is_floating(values.type):
            n_missing += int(_compute("sum", _compute("is_nan", values)).as_py() or 0)
            finite = _compute("filter", values, _compute("is_finite", values))
        else:
            finite = _compute("drop_null", values)
        if not len(finite):
            continue
        extremes = _compute("min_max", finite)
        low = _lower(low, _number(extremes["min"].as_py()))
        high = _higher(high, _number(extremes["max"].as_py()))
        total += float(_compute("sum", finite.cast(pa.float64(), safe=False)).as_py())
        n_finite += len(finite)
    return NumericSummary(
        min=low, max=high, total=total, n_finite=n_finite, n_missing=n_missing
    )


def _compute(function: str, *arguments: object):
    """Call the Arrow compute *function*, which ``pyarrow.compute`` binds only at runtime."""
    return pc.call_function(function, list(arguments))


def _number(value: object) -> int | float:
    if isinstance(value, int | float):
        return value
    msg = f"a numeric column's minimum or maximum read as {value!r}"
    raise TypeError(msg)


def _lower(current: int | float | None, value: int | float) -> int | float:
    return value if current is None else min(current, value)


def _higher(current: int | float | None, value: int | float) -> int | float:
    return value if current is None else max(current, value)


def _kind(arrow_type: pa.DataType) -> ArrowKind:
    if pa.types.is_dictionary(arrow_type):
        return _kind(arrow_type.value_type)
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_integer(arrow_type):
        return "integer"
    if pa.types.is_floating(arrow_type):
        return "floating"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    return "other"
