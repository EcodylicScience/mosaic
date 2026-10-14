"""One entry's table of a feature run or a tracks variant, previewed from its parquet file.

:func:`preview_artifact` returns an :class:`ArtifactPreview`: the file's footer
schema, its row count and frame extent, a head of at most
:data:`MAX_PREVIEW_ROWS` rows, and statistics of each numeric column read. The
CLI's ``mosaic preview --json`` is that model dumped.

The shapes follow the table reads of mosaic-api. A column is typed ``int``,
``float`` or ``text``, a boolean column is ``int`` with cells 0 and 1, and rows
are positional in the order of :attr:`ArtifactPreview.read_columns`. Every
non-finite float is ``null``. The model therefore dumps as strict JSON.

A preview reads the indexes and the one parquet file and writes nothing. It
does not convert raw tracks, does not take ``index_lock`` and does not create a
root. It reads only the columns asked for, plus ``frame`` for the extent,
through :mod:`~mosaic.core.pipeline.parquet_reader`, and does not build a pandas
frame of the table.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.inventory.model import FeatureRunRef, TracksVariantRef
from mosaic.core.pipeline.inventory.wire import ref_address
from mosaic.core.pipeline.parquet_reader import (
    ArrowKind,
    FooterColumn,
    NumericSummary,
    ParquetUnreadable,
    read_footer,
    read_head,
    summarize_numeric_columns,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue

__all__ = [
    "DEFAULT_PREVIEW_ROWS",
    "MAX_PREVIEW_ROWS",
    "ArtifactPreview",
    "ColumnStatistics",
    "PreviewColumn",
    "PreviewInvalid",
    "PreviewNotFound",
    "PreviewRef",
    "PreviewUnreadable",
    "preview_artifact",
    "preview_ref",
]

DEFAULT_PREVIEW_ROWS: Final = 20
"""The row count of a preview's head when the caller does not name a limit."""

MAX_PREVIEW_ROWS: Final = 50
"""The largest row count of a preview's head. A larger limit is refused."""

FRAME_COLUMN: Final = "frame"
"""The column whose smallest and largest values are a preview's frame extent."""

type PreviewRef = FeatureRunRef | TracksVariantRef
"""The artifacts that store one parquet table per entry."""

type PreviewColumnType = Literal["int", "float", "text", "other"]
"""A column's type as the table reads of mosaic-api name it.

``int`` covers integer and boolean columns and ``text`` string columns. A
dictionary column takes the type of its values. ``other`` covers every remaining
type, such as a timestamp or a list, whose cells the head writes as text.
"""

type PreviewCell = int | float | str | None
type Statistic = int | float | None


class PreviewInvalid(ClassifiedError, ValueError):
    """A preview asked with a malformed ref, an unknown column or a limit out of range."""

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field: str = field

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the argument that is invalid."""
        return {"field": self.field}


class PreviewNotFound(ClassifiedError, LookupError):
    """A ref and entry without an indexed table, or an indexed table without its file."""

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, message: str, *, ref: str, entry: Entry) -> None:
        super().__init__(message)
        self.ref: str = ref
        self.entry: Entry = entry

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the ref and the entry that were looked up."""
        return _entry_detail(self.ref, self.entry)


class PreviewUnreadable(ClassifiedError):
    """An indexed table whose file is not a readable parquet, as an interrupted write leaves it."""

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, message: str, *, ref: str, entry: Entry) -> None:
        super().__init__(message)
        self.ref: str = ref
        self.entry: Entry = entry

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the ref and the entry whose table is unreadable."""
        return _entry_detail(self.ref, self.entry)


def _entry_detail(ref: str, entry: Entry) -> dict[str, JsonValue]:
    group, sequence = entry
    return {"ref": ref, "group": group, "sequence": sequence}


class ColumnStatistics(StrictModel):
    """One numeric column summarized over every row of the table.

    Attributes:
        min: The smallest finite value, or ``None`` when the column has none.
        max: The largest finite value, or ``None`` when the column has none.
        mean: The mean of the finite values, or ``None`` when the column has none.
        nan_fraction: The fraction of rows whose value is NaN or null, or
            ``None`` for a table without rows.
    """

    min: Statistic
    max: Statistic
    mean: float | None
    nan_fraction: float | None


class PreviewColumn(StrictModel):
    """One column of the file's footer schema.

    Attributes:
        name: The column name.
        type: The column's :data:`PreviewColumnType`.
        arrow_type: The Arrow type that the footer records, such as ``int64``.
        statistics: The column's statistics when it was read and is numeric,
            else ``None``.
    """

    name: str
    type: PreviewColumnType
    arrow_type: str
    statistics: ColumnStatistics | None


class ArtifactPreview(StrictModel):
    """One entry's table of a feature run or a tracks variant, previewed.

    Attributes:
        ref: The artifact's address, as ``mosaic inventory`` prints it.
        group: The entry's group.
        sequence: The entry's sequence.
        path: The absolute path of the whole parquet file.
        columns: Every column of the file's footer schema, in table order.
        n_rows: The table's row count, from the footer.
        frame_min: The smallest ``frame``, or ``None`` when the table has no
            ``frame`` column or no frame value.
        frame_max: The largest ``frame``, under the same rule.
        read_columns: The columns that were read, in the order of each row's cells.
        rows: The table's first rows in file order, each positional in the order
            of ``read_columns``.
        has_more: Whether the table contains more rows than ``rows`` lists.
    """

    ref: str
    group: str
    sequence: str
    path: str
    columns: tuple[PreviewColumn, ...]
    n_rows: int
    frame_min: int | None
    frame_max: int | None
    read_columns: tuple[str, ...]
    rows: tuple[tuple[PreviewCell, ...], ...]
    has_more: bool


def preview_ref(address: str) -> PreviewRef:
    """Return the ref that *address* names, as :func:`~.inventory.wire.ref_address` writes it.

    ``feature/<storage>/<run_id>`` names a feature run and
    ``tracks-variant/<run_id>`` a tracks variant, where ``tracks-variant/``
    names the unlabeled tables.

    Raises:
        PreviewInvalid: The address names another kind or has the wrong number
            of parts.
    """
    kind, _, rest = address.partition("/")
    if kind == FeatureRunRef.kind:
        name, separator, run_id = rest.partition("/")
        if separator and name and run_id and "/" not in run_id:
            return FeatureRunRef(name=name, run_id=run_id)
    elif kind == TracksVariantRef.kind and "/" in address and "/" not in rest:
        return TracksVariantRef(run_id=rest)
    forms = f"'{FeatureRunRef.kind}/<storage>/<run_id>' or '{TracksVariantRef.kind}/<run_id>'"
    raise PreviewInvalid(
        f"cannot preview {address!r}: a preview reads {forms}", field="ref"
    )


def preview_artifact(
    ds: Dataset,
    ref: PreviewRef,
    entry: Entry,
    *,
    columns: list[str] | None = None,
    limit: int = DEFAULT_PREVIEW_ROWS,
) -> ArtifactPreview:
    """Return a preview of *entry*'s table under *ref*.

    Args:
        ds: The dataset, which may be opened without its roots.
        ref: The feature run or tracks variant.
        entry: The ``(group, sequence)`` whose table to read.
        columns: The columns to read for the head and the statistics, in that
            order, or ``None`` for every column. A repeated name is read once.
        limit: The head's row count, from 0 to :data:`MAX_PREVIEW_ROWS`.

    Raises:
        PreviewInvalid: *limit* is out of range or *columns* names a column the
            table lacks.
        PreviewNotFound: The index lacks a parquet table for *entry* under
            *ref*, or its file is absent.
        PreviewUnreadable: The indexed file is not a readable parquet.
    """
    if not 0 <= limit <= MAX_PREVIEW_ROWS:
        raise PreviewInvalid(
            f"limit {limit} is outside 0 to {MAX_PREVIEW_ROWS}", field="limit"
        )
    address = ref_address(ref)
    path = _table_path(ds, ref, entry, address)
    try:
        footer = read_footer(path)
        kinds: dict[str, ArrowKind] = {c.name: c.kind for c in footer.columns}
        read_columns = _read_columns(kinds, columns)
        numeric = [name for name in read_columns if kinds[name] in _NUMERIC_KINDS]
        measured = (
            [*numeric, FRAME_COLUMN]
            if kinds.get(FRAME_COLUMN) in _NUMERIC_KINDS and FRAME_COLUMN not in numeric
            else numeric
        )
        summaries = summarize_numeric_columns(path, measured)
        head = tuple(
            tuple(_cell(value) for value in row)
            for row in read_head(path, list(read_columns), limit)
        )
    except ParquetUnreadable as error:
        raise PreviewUnreadable(str(error), ref=address, entry=entry) from error
    statistics = {name: _statistics(summaries[name], footer.n_rows) for name in numeric}
    frame = summaries.get(FRAME_COLUMN)
    frame_min, frame_max = (
        (None, None) if frame is None else (_frame(frame.min), _frame(frame.max))
    )
    group, sequence = entry
    return ArtifactPreview(
        ref=address,
        group=group,
        sequence=sequence,
        path=str(path),
        columns=tuple(_column(column, statistics) for column in footer.columns),
        n_rows=footer.n_rows,
        frame_min=frame_min,
        frame_max=frame_max,
        read_columns=read_columns,
        rows=head,
        has_more=footer.n_rows > len(head),
    )


def _table_path(ds: Dataset, ref: PreviewRef, entry: Entry, address: str) -> Path:
    """Return the resolved parquet path that the index records for *entry* under *ref*."""
    group, sequence = entry
    stored: list[str]
    match ref:
        case FeatureRunRef():
            index = feature_index(feature_index_path(ds, ref.name))
            try:
                rows = index.read(
                    run_id=ref.run_id,
                    filter_ext=".parquet",
                    scope=Scope(entries=[entry]),
                )
            except FileNotFoundError:
                stored = []
            else:
                stored = [str(cell) for cell in rows["abs_path"]]
        case TracksVariantRef():
            rows = read_tracks_index(ds)
            mask = (
                (rows["run_id"] == ref.run_id)
                & (rows["group"] == group)
                & (rows["sequence"] == sequence)
            )
            stored = [str(cell) for cell in rows.loc[mask, "abs_path"]]
    if not stored:
        raise PreviewNotFound(
            f"{address} lacks a table for {group}:{sequence}",
            ref=address,
            entry=entry,
        )
    path = ds.resolve_path(stored[0])
    if not path.is_file():
        raise PreviewNotFound(
            f"{address} records {path} for {group}:{sequence}, which is absent",
            ref=address,
            entry=entry,
        )
    return path


def _read_columns(
    kinds: dict[str, ArrowKind], columns: list[str] | None
) -> tuple[str, ...]:
    """Return the columns to read, each once, refusing a name the table lacks."""
    if columns is None:
        return tuple(kinds)
    unknown = sorted({name for name in columns if name not in kinds})
    if unknown:
        named = ", ".join(map(repr, unknown))
        message = f"the table lacks {named}. Its columns are {', '.join(kinds)}."
        raise PreviewInvalid(message, field="columns")
    return tuple(dict.fromkeys(columns))


_NUMERIC_KINDS: Final[frozenset[ArrowKind]] = frozenset(
    {"integer", "floating", "boolean"}
)

_COLUMN_TYPES: Final[dict[ArrowKind, PreviewColumnType]] = {
    "integer": "int",
    "boolean": "int",
    "floating": "float",
    "string": "text",
    "other": "other",
}


def _column(
    column: FooterColumn, statistics: dict[str, ColumnStatistics]
) -> PreviewColumn:
    return PreviewColumn(
        name=column.name,
        type=_COLUMN_TYPES[column.kind],
        arrow_type=column.arrow_type,
        statistics=statistics.get(column.name),
    )


def _statistics(summary: NumericSummary, n_rows: int) -> ColumnStatistics:
    """Return the statistics of a column of *n_rows* rows from its summary."""
    return ColumnStatistics(
        min=summary.min,
        max=summary.max,
        mean=summary.total / summary.n_finite if summary.n_finite else None,
        nan_fraction=summary.n_missing / n_rows if n_rows else None,
    )


def _frame(value: int | float | None) -> int | None:
    """Return a frame extent's bound as an integer."""
    return None if value is None else int(value)


def _cell(value: object) -> PreviewCell:
    """Return *value* as a JSON-safe cell, a value of an ``other`` column as text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)
