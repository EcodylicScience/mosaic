"""One entry's table of a feature run, a tracks variant or a labels variant, previewed.

:func:`preview_artifact` returns an :class:`ArtifactPreview`: the file's footer
schema, its row count and frame extent, a head of at most
:data:`MAX_PREVIEW_ROWS` rows, and statistics of each numeric column read. The
CLI's ``mosaic preview --json`` is that model dumped.

A wide table is bounded by ``max_columns``, which reads the identity and
bookkeeping columns first and names the rest in ``columns_omitted``. A column
may be asked for by an ``fnmatch`` pattern. A labels variant's ``.npz`` file is
previewed as its events, with :class:`LabelsPreview` counting each class.

The shapes follow the table reads of mosaic-api. A column is typed ``int``,
``float`` or ``text``, a boolean column is ``int`` with cells 0 and 1, and rows
are positional in the order of :attr:`ArtifactPreview.read_columns`. Every
non-finite float is ``null``. The model therefore dumps as strict JSON.

A preview reads the indexes and the one parquet file and writes nothing. It
does not convert raw tracks, does not take ``index_lock`` and does not create a
root. It reads only the columns asked for, plus ``frame`` for the extent,
through :mod:`~mosaic.core.pipeline.parquet_reader`, and does not build a pandas
frame of the table. A labels file is loaded without unpickling.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast

from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.inventory.model import (
    FeatureRunRef,
    LabelsVariantRef,
    TracksVariantRef,
)
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
from mosaic.core.pipeline.types.data_config import BOOKKEEPING_COLS, META_COLS
from mosaic.core.scope import Scope
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    import numpy as np
    from numpy.typing import NDArray

    from mosaic.core.json_value import JsonValue

__all__ = [
    "DEFAULT_PREVIEW_ROWS",
    "MAX_OMITTED_NAMES",
    "MAX_PREVIEW_ROWS",
    "ArtifactPreview",
    "ColumnStatistics",
    "LabelClassCount",
    "LabelsPreview",
    "PreviewListed",
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

MAX_OMITTED_NAMES: Final = 100
"""The most column names :attr:`ArtifactPreview.columns_omitted` lists."""

MAX_NAMED_UNKNOWN: Final = 20
"""The most unknown columns an invalid-columns message names."""

MAX_DETAILED_UNKNOWN: Final = 10
"""The most unknown columns whose close names an invalid-columns error lists."""

FRAME_COLUMN: Final = "frame"
"""The column whose smallest and largest values are a preview's frame extent."""

LEADING_COLUMNS: Final = META_COLS | BOOKKEEPING_COLS
"""The columns that ``max_columns`` reads before any other: who and when a row is."""

PAIR_FORMAT: Final = "individual_pair_v1"
"""The labels format of events, each naming a frame, a label and a pair of ids."""

type PreviewRef = FeatureRunRef | TracksVariantRef | LabelsVariantRef
"""The artifacts that store one table per entry."""

type PreviewListed = Literal["all", "read"]
"""Which columns :attr:`ArtifactPreview.columns` lists: the file's, or those read."""

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

    def __init__(
        self,
        message: str,
        *,
        field: str,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.field: str = field
        self.extra: dict[str, JsonValue] = dict(extra or {})

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the argument that is invalid, and what else the refusal knows."""
        return {"field": self.field, **self.extra}


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


class LabelClassCount(StrictModel):
    """One label class of a labels table, counted.

    Attributes:
        id: The class id the events carry.
        name: The class name the labels index records, ``""`` when it records
            none for this id.
        events: The events of this class, a mirrored pair event counted once.
            For a dense table, the frames that carry it.
        fraction: ``events`` over the events of every class, ``None`` when
            there are none.
    """

    id: int
    name: str
    events: int
    fraction: float | None


class LabelsPreview(StrictModel):
    """What one entry's labels table holds, class by class.

    Attributes:
        label_kind: The kind of labels, such as ``behavior``.
        label_format: ``individual_pair_v1`` for events, ``dense`` for one
            label per frame.
        n_events: The events counted, a mirrored pair event once. For a dense
            table, its frames.
        frames_labeled: The distinct frames that carry an event. For a dense
            table, its length.
        n_frames: The entry's frames as the labels index records them,
            ``None`` when unrecorded.
        coverage: ``frames_labeled`` over ``n_frames``, ``None`` when
            ``n_frames`` is unrecorded.
        n_pairs: The distinct pairs of ids that the events name, ``None`` for a
            dense table.
        symmetric_deduplicated: Whether the table stores pair events in both
            orders, ``[i, j]`` and ``[j, i]``, and each was counted once.
        classes: Each class that the index names or an event carries, by id.
    """

    label_kind: str
    label_format: str
    n_events: int
    frames_labeled: int
    n_frames: int | None
    coverage: float | None
    n_pairs: int | None
    symmetric_deduplicated: bool
    classes: tuple[LabelClassCount, ...]


class ArtifactPreview(StrictModel):
    """One entry's table of a feature run, a tracks variant or a labels variant.

    Attributes:
        ref: The artifact's address, as ``mosaic inventory`` prints it.
        group: The entry's group.
        sequence: The entry's sequence.
        path: The absolute path of the whole file: a parquet table, or the
            ``.npz`` file of a labels variant.
        columns: Every column of the table in table order, or under
            ``listed="read"`` only the columns read. A labels table's columns
            are those of its events: ``frame``, ``label``, ``id1`` and ``id2``.
        n_rows: The table's row count, from the footer. A labels table's events
            as stored.
        frame_min: The smallest ``frame``, or ``None`` when the table has no
            ``frame`` column or no frame value.
        frame_max: The largest ``frame``, under the same rule.
        read_columns: The columns that were read, in the order of each row's cells.
        rows: The table's first rows in file order, each positional in the order
            of ``read_columns``.
        has_more: Whether the table contains more rows than ``rows`` lists.
        n_columns: The table's column count.
        columns_omitted: The columns that ``max_columns`` left unread, in table
            order, at most :data:`MAX_OMITTED_NAMES`.
        columns_omitted_has_more: Whether more columns were left unread than
            ``columns_omitted`` names.
        labels: The class counts and coverage of a labels table, ``None`` for
            any other.
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
    n_columns: int = 0
    columns_omitted: tuple[str, ...] = ()
    columns_omitted_has_more: bool = False
    labels: LabelsPreview | None = None


def preview_ref(address: str) -> PreviewRef:
    """Return the ref that *address* names, as :func:`~.inventory.wire.ref_address` writes it.

    ``feature/<storage>/<run_id>`` names a feature run,
    ``tracks-variant/<run_id>`` a tracks variant, where ``tracks-variant/``
    names the unlabeled tables, and ``labels-variant/<kind>/<run_id>`` a labels
    variant.

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
    elif kind == LabelsVariantRef.kind:
        label_kind, separator, run_id = rest.partition("/")
        if separator and label_kind and "/" not in run_id:
            return LabelsVariantRef(label_kind=label_kind, run_id=run_id)
    forms = (
        f"'{FeatureRunRef.kind}/<storage>/<run_id>', "
        f"'{TracksVariantRef.kind}/<run_id>' or "
        f"'{LabelsVariantRef.kind}/<kind>/<run_id>'"
    )
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
    max_columns: int | None = None,
    listed: PreviewListed = "all",
) -> ArtifactPreview:
    """Return a preview of *entry*'s table under *ref*.

    Args:
        ds: The dataset, which may be opened without its roots.
        ref: The feature run, tracks variant or labels variant.
        entry: The ``(group, sequence)`` whose table to read.
        columns: The columns to read for the head and the statistics, in that
            order, or ``None`` for every column. An entry that names a column
            is that column. Otherwise an entry containing ``*``, ``?`` or ``[``
            is an ``fnmatch`` pattern, which reads its matches in table order.
            A column asked for twice is read once.
        limit: The head's row count, from 0 to :data:`MAX_PREVIEW_ROWS`.
        max_columns: The most columns to read, ``None`` for no bound. With
            ``columns=None`` the identity and bookkeeping columns are read
            first. The columns left unread are named in ``columns_omitted``.
        listed: ``all`` lists every column of the table in
            :attr:`ArtifactPreview.columns`, and ``read`` only those read.

    Raises:
        PreviewInvalid: *limit* or *max_columns* is out of range, or *columns*
            names a column, or holds a pattern, that the table lacks.
        PreviewNotFound: The index lacks a table for *entry* under *ref*, or
            its file is absent.
        PreviewUnreadable: The indexed file is not a readable parquet, or not a
            labels file that loads without unpickling.
    """
    if not 0 <= limit <= MAX_PREVIEW_ROWS:
        raise PreviewInvalid(
            f"limit {limit} is outside 0 to {MAX_PREVIEW_ROWS}", field="limit"
        )
    if max_columns is not None and max_columns < 1:
        raise PreviewInvalid(
            f"max_columns {max_columns} is below 1", field="max_columns"
        )
    address = ref_address(ref)
    if isinstance(ref, LabelsVariantRef):
        return _preview_labels(
            ds,
            ref,
            entry,
            address,
            _ColumnRequest(columns, max_columns, listed),
            limit,
        )
    path = _table_path(ds, ref, entry, address)
    try:
        footer = read_footer(path)
        kinds: dict[str, ArrowKind] = {c.name: c.kind for c in footer.columns}
        read_columns, omitted = _read_columns(kinds, columns, max_columns)
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
    read = frozenset(read_columns)
    return ArtifactPreview(
        ref=address,
        group=group,
        sequence=sequence,
        path=str(path),
        columns=tuple(
            _column(column, statistics)
            for column in footer.columns
            if listed == "all" or column.name in read
        ),
        n_rows=footer.n_rows,
        frame_min=frame_min,
        frame_max=frame_max,
        read_columns=read_columns,
        rows=head,
        has_more=footer.n_rows > len(head),
        n_columns=len(footer.columns),
        columns_omitted=omitted[:MAX_OMITTED_NAMES],
        columns_omitted_has_more=len(omitted) > MAX_OMITTED_NAMES,
    )


def _table_path(
    ds: Dataset, ref: FeatureRunRef | TracksVariantRef, entry: Entry, address: str
) -> Path:
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
    return _indexed_file(ds, stored, entry, address)


def _indexed_file(ds: Dataset, stored: list[str], entry: Entry, address: str) -> Path:
    """Return the first indexed path of *entry*, refusing none or an absent file."""
    group, sequence = entry
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


_PATTERN_CHARACTERS: Final = frozenset("*?[")


def _read_columns(
    kinds: Mapping[str, ArrowKind],
    columns: list[str] | None,
    max_columns: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the columns to read, each once, and those that *max_columns* left out.

    Both are in table order when *columns* is ``None``, the identity and
    bookkeeping columns chosen first. Otherwise the read columns follow
    *columns*, a pattern expanding in table order.

    Raises:
        PreviewInvalid: A name or pattern in *columns* matches no column.
    """
    if columns is None:
        leading = [name for name in kinds if name in LEADING_COLUMNS]
        rest = [name for name in kinds if name not in LEADING_COLUMNS]
        chosen = frozenset([*leading, *rest][:max_columns])
        return (
            tuple(name for name in kinds if name in chosen),
            tuple(name for name in kinds if name not in chosen),
        )
    asked: list[str] = []
    unknown: list[str] = []
    for item in columns:
        if item in kinds:
            asked.append(item)
            continue
        matches = (
            [name for name in kinds if fnmatchcase(name, item)]
            if _PATTERN_CHARACTERS & set(item)
            else []
        )
        if not matches:
            unknown.append(item)
        asked.extend(matches)
    if unknown:
        raise _unknown_columns(list(dict.fromkeys(unknown)), list(kinds))
    ordered = tuple(dict.fromkeys(asked))
    return ordered[:max_columns], ordered[max_columns:] if max_columns else ()


def _unknown_columns(unknown: list[str], names: list[str]) -> PreviewInvalid:
    """Refuse *unknown*, naming a bounded number of them with close column names."""
    named = ", ".join(map(repr, unknown[:MAX_NAMED_UNKNOWN]))
    if len(unknown) > MAX_NAMED_UNKNOWN:
        named += f" and {len(unknown) - MAX_NAMED_UNKNOWN} more"
    close: dict[str, JsonValue] = {
        item: list(difflib.get_close_matches(item, names, n=3))
        for item in unknown[:MAX_DETAILED_UNKNOWN]
    }
    hints = "; ".join(
        f"{item!r} is close to {', '.join(map(repr, found))}"
        for item, found in close.items()
        if isinstance(found, list) and found
    )
    message = f"the table lacks {named}. It has {len(names)} columns"
    message += f"; {hints}." if hints else "."
    return PreviewInvalid(
        message,
        field="columns",
        extra={
            "unknown": list(unknown[:MAX_DETAILED_UNKNOWN]),
            "close": close,
            "n_columns": len(names),
        },
    )


@dataclass(frozen=True, slots=True)
class _ColumnRequest:
    """The column arguments of a preview, passed on to the labels reader."""

    columns: list[str] | None
    max_columns: int | None
    listed: PreviewListed


def _preview_labels(
    ds: Dataset,
    ref: LabelsVariantRef,
    entry: Entry,
    address: str,
    request: _ColumnRequest,
    limit: int,
) -> ArtifactPreview:
    """Return a preview of *entry*'s labels table under *ref*, counted by class.

    The events are the table's rows, and :class:`LabelsPreview` counts them.
    The class names come from the index row, which may leave them blank.
    """
    import numpy as np

    from mosaic.core.helpers import detect_label_format, load_labels_auto
    from mosaic.core.pipeline.labels_index import (
        read_labels_index,
        select_label_variant_rows,
    )

    group, sequence = entry
    indexed = select_label_variant_rows(
        read_labels_index(ds, ref.label_kind), ref.run_id
    )
    mask = (indexed["group"] == group) & (indexed["sequence"] == sequence)
    matched = indexed.loc[mask]
    path = _indexed_file(
        ds, [str(cell) for cell in matched["abs_path"]], entry, address
    )
    row = matched.iloc[0]
    try:
        with np.load(path, allow_pickle=False) as npz:
            label_format = detect_label_format(npz)
        loaded = load_labels_auto(path, return_format="sparse", allow_pickle=False)
    except (OSError, ValueError, KeyError) as error:
        message = f"{path} is not a labels file that loads without unpickling: {error}"
        raise PreviewUnreadable(message, ref=address, entry=entry) from error
    frames, labels, ids = cast("tuple[NDArray[np.int64], ...]", loaded)
    paired = label_format == PAIR_FORMAT
    table: dict[str, NDArray[np.int64]] = {"frame": frames, "label": labels}
    if paired:
        table |= {"id1": ids[:, 0].astype(np.int64), "id2": ids[:, 1].astype(np.int64)}
    kinds: dict[str, ArrowKind] = dict.fromkeys(table, "integer")
    read_columns, omitted = _read_columns(kinds, request.columns, request.max_columns)
    n_rows = len(labels)
    head = tuple(
        tuple(int(table[name][position]) for name in read_columns)
        for position in range(min(limit, n_rows))
    )
    read = frozenset(read_columns)
    return ArtifactPreview(
        ref=address,
        group=group,
        sequence=sequence,
        path=str(path),
        columns=tuple(
            PreviewColumn(name=name, type="int", arrow_type="int64", statistics=None)
            for name in table
            if request.listed == "all" or name in read
        ),
        n_rows=n_rows,
        frame_min=int(frames.min()) if n_rows else None,
        frame_max=int(frames.max()) if n_rows else None,
        read_columns=read_columns,
        rows=head,
        has_more=n_rows > len(head),
        n_columns=len(table),
        columns_omitted=omitted,
        labels=_count_labels(
            ref.label_kind,
            label_format,
            table,
            n_frames=_cell_int(row["n_frames"]),
            names=_class_names(str(row["label_ids"]), str(row["label_names"])),
        ),
    )


def _count_labels(
    label_kind: str,
    label_format: str,
    table: Mapping[str, NDArray[np.int64]],
    *,
    n_frames: int | None,
    names: Mapping[int, str],
) -> LabelsPreview:
    """Count a labels table's events by class, a mirrored pair event once.

    A converter that stores each pair event as ``[i, j]`` and ``[j, i]``
    (BORIS's) would count it twice, so pair events are counted once per
    frame, label and unordered pair of ids, as ``load_labels_for_feature_frames``
    reads them.
    """
    import numpy as np

    frames = table["frame"]
    labels = table["label"]
    paired = "id1" in table
    deduplicated = False
    n_pairs: int | None = None
    if paired:
        low = np.minimum(table["id1"], table["id2"])
        high = np.maximum(table["id1"], table["id2"])
        events = np.unique(np.stack([frames, labels, low, high], axis=1), axis=0)
        deduplicated = len(events) < len(frames)
        frames, labels = events[:, 0], events[:, 1]
        n_pairs = len(np.unique(events[:, 2:], axis=0))
    n_events = len(labels)
    frames_labeled = len(np.unique(frames)) if paired else n_events
    found, counts = np.unique(labels, return_counts=True)
    events_of = {int(i): int(n) for i, n in zip(found, counts, strict=True)}
    return LabelsPreview(
        label_kind=label_kind,
        label_format=label_format,
        n_events=n_events,
        frames_labeled=frames_labeled,
        n_frames=n_frames,
        coverage=frames_labeled / n_frames if n_frames else None,
        n_pairs=n_pairs,
        symmetric_deduplicated=deduplicated,
        classes=tuple(
            LabelClassCount(
                id=class_id,
                name=names.get(class_id, ""),
                events=events_of.get(class_id, 0),
                fraction=events_of.get(class_id, 0) / n_events if n_events else None,
            )
            for class_id in sorted(set(names) | set(events_of))
        ),
    )


def _class_names(ids: str, names: str) -> dict[int, str]:
    """Return the class names an index row records, an id with no name left out."""
    id_cells = [cell.strip() for cell in ids.split(",")] if ids.strip() else []
    name_cells = names.split(",") if names.strip() else []
    named: dict[int, str] = {}
    for position, cell in enumerate(id_cells):
        try:
            class_id = int(cell)
        except ValueError:
            continue
        named[class_id] = (
            name_cells[position].strip() if position < len(name_cells) else ""
        )
    return named


def _cell_int(value: object) -> int | None:
    """Return a positive count read off an index cell, ``None`` when blank or zero."""
    try:
        count = int(str(value))
    except ValueError:
        return None
    return count if count > 0 else None


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
