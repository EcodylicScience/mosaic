"""Typed reads of one parquet file through pyarrow: its footer, numeric summaries and head.

pyarrow does not ship type stubs, and its compiled modules leave every attribute
unknown to the type checker. This module holds every footer read of the
pipeline and the preview's reads, each returning plain Python values, so its
callers type-check in full. The file-level directives below cover the unknown
types of pyarrow alone.

Each read opens the file and reads only the columns it names. None of them
builds a pandas frame. A file that pyarrow cannot read raises
:class:`ParquetUnreadable`.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path
    from typing import Any

    import numpy.typing as npt

__all__ = [
    "ArrowKind",
    "FooterColumn",
    "NumericSummary",
    "ParquetFooter",
    "ParquetUnreadable",
    "iter_column_batches",
    "output_n_rows",
    "read_footer",
    "read_head",
    "read_parquet_column_names",
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


def read_footer(path: Path) -> ParquetFooter:
    """Return the schema and the row count that *path*'s footer records."""
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


def read_parquet_column_names(path: Path) -> tuple[str, ...] | None:
    """The column names of a parquet, read from its footer and not its data.

    ``None`` for an absent file and for one whose footer does not parse, as an
    interrupted write leaves it. A caller asking what a table contains treats
    that as unknown.
    """
    if not path.exists():
        return None
    try:
        footer = read_footer(path)
    except ParquetUnreadable:
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
