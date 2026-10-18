"""Read one finished feature run as a row index and an array.

``load_values`` takes one ``ResultColumn`` per column, so a 546-column wavelet table
takes 546 sources, and it returns one float64 frame of every entry. It has no
perspective filter either. An analysis script that wanted a model's columns read the
parquet files itself and aligned the rows by hand.

:func:`load_run_matrix` reads the run instead: the identity of each row in a small
frame, the numbers in one array of the dtype asked for, and the column names. It sizes
the array from the parquet footers before it reads a value, refuses a read that would
not fit, and then writes each entry into the array in place, a batch of rows at a
time. A perspective filter applies to each batch, so a pair run holds only the rows
it returns.

A batch size does not bound what pyarrow holds. It decodes a whole row group of the
columns it is asked for, and a feature's output is one row group up to a million rows.
A file whose columns would take more than :data:`BLOCK_BYTES` to decode is therefore
read a block of columns at a time, and the estimate that a read is checked against
counts the largest block.

``mosaic.core.pipeline`` binds the names of this module on access.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.entry import Entry
from mosaic.core.scope import Scope

from .dataset_indexes import feature_storages
from .index import feature_index, feature_index_path, missing_outputs_error
from .memory_watch import host_memory_bytes
from .parquet_reader import ParquetFooter, iter_column_batches, read_footer
from .track_universe import current_run_id
from .types import BOOKKEEPING_COLS, META_COLS, Result

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset

__all__ = ["RunMatrix", "load_run_matrix"]

BATCH_ROWS: Final = 65_536
"""The most rows of a file that one batch holds."""

BATCH_BYTES: Final = 64 * 1024**2
"""The most memory that one batch may take, counted as float64 cells.

A wide table takes fewer rows per batch than :data:`BATCH_ROWS`, so a 546-column run
reads as many bytes per batch as a 5-column one.
"""

BLOCK_BYTES: Final = 256 * 1024**2
"""The most memory that pyarrow may hold to decode the columns of one file at a time.

A file that would take more is read a block of columns at a time, down to one column.
"""

DECODE_FACTOR: Final = 2
"""How many times the float64 size of its cells pyarrow holds while it decodes a row group.

Measured at 1.9 for a row group of 400,000 rows, whatever the batch size.
"""

MEMORY_HEADROOM: Final = 0.20
"""The share of available memory that a read leaves free, as ``temporal-stack`` does."""

_IDENTITY_COLUMNS: Final = ("frame", "id", "id1", "id2", "perspective")
"""The columns of a row's identity, after ``group`` and ``sequence``, in index order."""

_NUMERIC_KINDS: Final = frozenset({"integer", "floating", "boolean"})

_LISTED: Final = 12
"""How many names a message lists before it counts the rest."""


@dataclass(frozen=True, slots=True, eq=False)
class RunMatrix:
    """The rows of a feature run, as an identity frame and an array.

    Attributes:
        feature: The storage name of the feature that wrote the run.
        run_id: The run that was read. An unpinned ``Result`` reads the run
            :func:`~mosaic.core.pipeline.track_universe.current_run_id` names.
        entries: The ``(group, sequence)`` pairs that were read, sorted.
        index: One row per row of *values*, in the same order: ``group``,
            ``sequence``, and each of ``frame``, ``id``, ``id1``, ``id2`` and
            ``perspective`` that the run writes.
        values: The cells, shape ``(len(index), len(columns))``, in the dtype
            asked for. A missing value is NaN.
        columns: The name of each column of *values*.
        missing: The entries in the scope that the run has no readable output
            for. Empty unless ``allow_partial`` let the read go on without them.
        finished: Whether the run is marked finished. ``False`` only where
            ``allow_partial`` let the read go on.
    """

    feature: str
    run_id: str
    entries: tuple[Entry, ...]
    index: pd.DataFrame
    values: npt.NDArray[np.floating[Any]]
    columns: tuple[str, ...]
    missing: tuple[Entry, ...] = ()
    finished: bool = True


@dataclass(frozen=True, slots=True)
class _Outputs:
    """What the index lists for the scope: the files to read and what is lacking."""

    found: tuple[tuple[Entry, Path], ...]
    unreadable: tuple[Entry, ...]
    absent: tuple[Entry, ...]
    unfinished: int
    listed: int


@dataclass(frozen=True, slots=True)
class _Layout:
    """What a read of the run returns, from its footers and before any value is read."""

    columns: tuple[str, ...]
    identity: tuple[str, ...]
    entry_rows: tuple[int, ...]
    file_rows: tuple[int, ...]


def load_run_matrix(
    ds: Dataset,
    result: Result[str],
    *,
    scope: Scope | None = None,
    perspective: int | None = None,
    columns: str | Sequence[str] | None = None,
    dtype: npt.DTypeLike = np.float32,
    allow_partial: bool = False,
) -> RunMatrix:
    """Read the rows of one finished feature run into an identity frame and an array.

    The entries come in order of ``(group, sequence)``, whatever order the index
    lists them in, and the rows of an entry in the order of its file. Nothing is
    written, so a dataset opened with ``ensure_roots=False`` reads as well. The
    memory taken is the array, the identity frame, and what pyarrow holds to decode
    the largest block of columns of one file.

    Args:
        ds: The dataset holding the run.
        result: The run to read. A ``Result`` whose ``run_id`` is ``None`` reads the
            run that an unpinned reference resolves to everywhere else.
        scope: The entries to read. ``None`` reads every entry that the run holds.
        perspective: Keep only the rows of this ``perspective``, when the run writes
            one, and read all rows when it does not.
        columns: The columns to return. A name, a glob (``"w*"``) or a list of
            either; a list keeps its order and a column once. A name or a pattern
            matches the numeric columns that are not part of a row's identity,
            bookkeeping columns (``fps``, ``label``, ``nn_id`` and the like)
            among them. ``None`` returns every numeric column except identity and
            bookkeeping.
        dtype: The floating-point dtype of the array. An integer or a boolean column
            is converted, and a missing value is NaN.
        allow_partial: Read what the run holds when the scope names entries that it
            does not hold, when an output file listed in its index is gone, or when
            the run is not finished. ``RunMatrix.missing`` and ``finished`` then
            report it. The default refuses all three.

    Returns:
        The rows of the run.

    Raises:
        FileNotFoundError: The feature has no runs here, the run is not in its index,
            or an output it lists is gone. The last also covers every output being
            gone, as when a dataset moved and its paths did not.
        ValueError: The scope selects no output of the run, names an entry it does
            not hold, or the run is not finished. A pattern in *columns* matches no
            numeric column, two entries write different columns, or *dtype* is not
            floating point.
        MemoryError: The array would not fit in the memory available. Narrow the
            read with *scope*, *perspective* or *columns*, or choose a smaller
            *dtype*.
    """
    value_dtype = _floating_dtype(dtype)
    selector = scope if scope is not None else Scope()
    run_id = _resolve_run_id(ds, result)
    outputs = _run_outputs(ds, result.feature, run_id, selector)
    _refuse_a_partial_run(result.feature, run_id, outputs, allow_partial)
    patterns = (columns,) if isinstance(columns, str) else columns
    layout = _plan(
        result.feature,
        run_id,
        outputs.found,
        None if patterns is None else tuple(patterns),
        perspective,
    )
    batch_rows = _batch_rows(len(layout.columns) + len(layout.identity))
    _check_memory(result.feature, run_id, layout, value_dtype)
    index, values = _read(outputs.found, layout, perspective, value_dtype, batch_rows)
    return RunMatrix(
        feature=result.feature,
        run_id=run_id,
        entries=tuple(entry for entry, _ in outputs.found),
        index=index,
        values=values,
        columns=layout.columns,
        missing=tuple(sorted({*outputs.unreadable, *outputs.absent})),
        finished=outputs.unfinished == 0,
    )


def _floating_dtype(dtype: npt.DTypeLike) -> np.dtype[Any]:
    """The dtype asked for, which has to hold NaN."""
    resolved = np.dtype(dtype)
    if resolved.kind != "f":
        msg = (
            f"dtype={resolved} cannot hold a missing value. Pass a floating-point "
            f"dtype such as numpy.float32 or numpy.float64."
        )
        raise ValueError(msg)
    return resolved


def _resolve_run_id(ds: Dataset, result: Result[str]) -> str:
    """The run to read: the pinned one, or the one the unpinned rule names."""
    path = feature_index_path(ds, result.feature)
    if not path.exists():
        with_runs = [
            name
            for name in feature_storages(ds)
            if feature_index_path(ds, name).exists()
        ]
        msg = (
            f"{result.feature!r} has no runs in this dataset: {path} does not "
            f"exist. A feature is stored under the name its run was given, such "
            f"as 'speed-angvel__from__tracks'. The features that have runs are: "
            f"{_listed(with_runs)}."
        )
        raise FileNotFoundError(msg)
    if result.run_id is not None:
        return result.run_id
    return current_run_id(ds, result.feature)


def _run_outputs(ds: Dataset, feature: str, run_id: str, selector: Scope) -> _Outputs:
    """Which files of the run the scope reaches, and what it reaches that is lacking.

    The index is the universe: an entry that a run never wrote has no row, and only
    an entry that the scope names outright can be told from one the scope did not
    ask for.
    """
    rows = feature_index(feature_index_path(ds, feature)).read(
        run_id=run_id, filter_ext=".parquet", scope=selector
    )
    finished_at = rows["finished_at"] if "finished_at" in rows.columns else None
    found: list[tuple[Entry, Path]] = []
    unreadable: list[tuple[Entry, Path]] = []
    for row in rows.itertuples(index=False):
        entry = (str(row.group), str(row.sequence))
        path = ds.resolve_path(str(row.abs_path))
        (found if path.exists() else unreadable).append((entry, path))
    named = selector.entry_pairs
    listed = {entry for entry, _ in found} | {entry for entry, _ in unreadable}
    absent = sorted(named - listed) if named is not None else []
    if not found and not absent and not unreadable:
        msg = (
            f"{selector!r} selects no output of {feature} run {run_id}. Check the "
            f"group and sequence names against the entries that the run holds."
        )
        raise ValueError(msg)
    if unreadable and not found:
        raise missing_outputs_error(
            feature, run_id, [path for _, path in unreadable], len(rows)
        )
    return _Outputs(
        found=tuple(sorted(found, key=lambda item: item[0])),
        unreadable=tuple(sorted(entry for entry, _ in unreadable)),
        absent=tuple(absent),
        unfinished=0 if finished_at is None else int((finished_at == "").sum()),
        listed=len(rows),
    )


def _refuse_a_partial_run(
    feature: str, run_id: str, outputs: _Outputs, allow_partial: bool
) -> None:
    """Raise unless the run is whole over the scope, or the caller accepted less."""
    if not outputs.found:
        msg = (
            f"{feature} run {run_id} holds no output for the entries the scope "
            f"names: {_listed(_names(outputs.absent))}."
        )
        raise ValueError(msg)
    if allow_partial:
        return
    if outputs.unreadable:
        msg = (
            f"{feature} run {run_id} lists outputs that are gone from disk: "
            f"{_listed(_names(outputs.unreadable))}. Recompute those entries, "
            f"drop their stale index rows with ds.reindex_features(), or read the "
            f"others with scope=, or with allow_partial=True."
        )
        raise FileNotFoundError(msg)
    if outputs.absent:
        msg = (
            f"The scope names entries that {feature} run {run_id} holds no output "
            f"for: {_listed(_names(outputs.absent))}. Compute them, drop them from "
            f"the scope, or pass allow_partial=True."
        )
        raise ValueError(msg)
    if outputs.unfinished:
        msg = (
            f"{feature} run {run_id} is not finished: {outputs.unfinished} of "
            f"{outputs.listed} outputs in scope have no finish time. A run that is "
            f"still going, or that lost an entry, stays unfinished. Wait for it, "
            f"resubmit it to compute the lost entries, or pass allow_partial=True "
            f"to read the entries it has."
        )
        raise ValueError(msg)


def _plan(
    feature: str,
    run_id: str,
    found: tuple[tuple[Entry, Path], ...],
    patterns: tuple[str, ...] | None,
    perspective: int | None,
) -> _Layout:
    """Fix the columns and count the rows of every entry, from footers alone.

    A perspective filter costs one more read per entry, of the ``perspective``
    column alone, so that the count is of the rows returned and not of the rows in
    the files.
    """
    first: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    first_entry = found[0][0]
    entry_rows: list[int] = []
    file_rows: list[int] = []
    for entry, path in found:
        footer = read_footer(path)
        chosen = _select_columns(footer, patterns, f"{feature} run {run_id}", entry)
        identity = tuple(
            name for name in _IDENTITY_COLUMNS if _has_column(footer, name)
        )
        if first is None:
            first = (chosen, identity)
        elif set(chosen + identity) != set(first[0] + first[1]):
            msg = (
                f"{feature} run {run_id}: entry {_name(entry)} writes different "
                f"columns from entry {_name(first_entry)}"
                f"{_difference(first[0] + first[1], chosen + identity)}. A matrix "
                f"needs one set of columns, so narrow the read with scope=."
            )
            raise ValueError(msg)
        entry_rows.append(_rows_returned(path, footer, perspective))
        file_rows.append(footer.n_rows)
    assert first is not None
    return _Layout(first[0], first[1], tuple(entry_rows), tuple(file_rows))


def _has_column(footer: ParquetFooter, name: str) -> bool:
    return any(column.name == name for column in footer.columns)


def _rows_returned(path: Path, footer: ParquetFooter, perspective: int | None) -> int:
    """The rows of one file that survive the perspective filter."""
    if perspective is None or not _has_column(footer, "perspective"):
        return footer.n_rows
    return sum(
        int(np.count_nonzero(batch["perspective"] == perspective))
        for batch in iter_column_batches(path, ["perspective"], BATCH_ROWS)
    )


def _select_columns(
    footer: ParquetFooter,
    patterns: tuple[str, ...] | None,
    where: str,
    entry: Entry,
) -> tuple[str, ...]:
    """The columns of one entry that *patterns* select, in the order they return.

    Only a numeric column that is not part of a row's identity can be a value. A
    pattern that is the exact name of such a column takes it literally, so a name
    holding glob characters still selects itself.
    """
    candidates = [
        column.name
        for column in footer.columns
        if column.kind in _NUMERIC_KINDS
        and column.name not in META_COLS
        and not column.name.startswith("__index_level_")
    ]
    if patterns is None:
        chosen = tuple(name for name in candidates if name not in BOOKKEEPING_COLS)
        if not chosen:
            msg = (
                f"{where}: entry {_name(entry)} has no numeric payload column. Name "
                f"the columns to read with columns=. Its numeric columns are: "
                f"{_listed(candidates)}."
            )
            raise ValueError(msg)
        return chosen
    taken: list[str] = []
    for pattern in patterns:
        matched = [name for name in candidates if name == pattern] or [
            name for name in candidates if fnmatch.fnmatchcase(name, pattern)
        ]
        if not matched:
            msg = (
                f"{where}: columns={pattern!r} matches no numeric column of entry "
                f"{_name(entry)}. An identity column (frame, time, id, id1, id2, "
                f"perspective) is not a value, and nor is a string column. The "
                f"numeric columns are: {_listed(candidates)}."
            )
            raise ValueError(msg)
        taken.extend(name for name in matched if name not in taken)
    return tuple(taken)


def _batch_rows(n_columns: int) -> int:
    """The rows per batch, so that a batch of float64 cells fits :data:`BATCH_BYTES`."""
    return max(1, min(BATCH_ROWS, BATCH_BYTES // (8 * max(1, n_columns))))


def _block_width(file_rows: int) -> int:
    """How many columns of a file of *file_rows* rows one read may ask pyarrow for.

    At least one, so that a file too long for :data:`BLOCK_BYTES` is still read, and
    its estimate counts the one column.
    """
    return max(1, BLOCK_BYTES // (DECODE_FACTOR * 8 * max(1, file_rows)))


def _blocks(columns: tuple[str, ...], file_rows: int) -> list[tuple[str, ...]]:
    """The columns of one file, in the blocks that are read one after another.

    A block is one pass over the file. One block holds every column of a file that
    is short enough, as the read has always gone. No columns is one empty block,
    which still reads the identity.
    """
    width = _block_width(file_rows)
    return [
        columns[start : start + width] for start in range(0, len(columns), width)
    ] or [()]


def _check_memory(
    feature: str,
    run_id: str,
    layout: _Layout,
    dtype: np.dtype[Any],
) -> None:
    """Raise if the read would leave less than the headroom free.

    The estimate is the array, the identity frame (counted twice, because the frame
    is built from copies of its pieces), and what pyarrow holds to decode the largest
    block of columns of one file. Where the host reports no memory, as without
    ``psutil``, the read goes ahead.
    """
    _, available = host_memory_bytes()
    if available <= 0:
        return
    n_rows = sum(layout.entry_rows)
    n_columns, n_identity = len(layout.columns), len(layout.identity)
    identity_bytes = n_rows * 8 * (n_identity + 2)
    decoding = max(
        (
            DECODE_FACTOR * 8 * rows * (min(n_columns, _block_width(rows)) + n_identity)
            for rows in layout.file_rows
        ),
        default=0,
    )
    estimated = n_rows * n_columns * dtype.itemsize + 2 * identity_bytes + decoding
    if estimated <= available * (1.0 - MEMORY_HEADROOM):
        return
    msg = (
        f"Reading {feature} run {run_id} would take about "
        f"{estimated / 1024**3:.1f} GB for {n_rows:,} rows x {n_columns} columns "
        f"as {dtype}, and only {available / 1024**3:.1f} GB of memory is "
        f"available. Narrow the read with scope=, perspective= or columns=, or "
        f"pass a smaller dtype=."
    )
    raise MemoryError(msg)


def _read(
    found: tuple[tuple[Entry, Path], ...],
    layout: _Layout,
    perspective: int | None,
    dtype: np.dtype[Any],
    batch_rows: int,
) -> tuple[pd.DataFrame, npt.NDArray[np.floating[Any]]]:
    """Write every entry into one array, a batch at a time, and build the identity.

    The first block of an entry's columns is read with its identity columns, and a
    later block with the ``perspective`` column alone beside it, to choose its rows.
    """
    values = np.empty((sum(layout.entry_rows), len(layout.columns)), dtype=dtype)
    filtering = perspective is not None and "perspective" in layout.identity
    later = ("perspective",) if filtering else ()
    column_of = {name: position for position, name in enumerate(layout.columns)}
    groups: list[npt.NDArray[np.object_]] = []
    sequences: list[npt.NDArray[np.object_]] = []
    parts: dict[str, list[npt.NDArray[Any]]] = {name: [] for name in layout.identity}
    start = 0
    for ((group, sequence), path), file_rows, expected in zip(
        found, layout.file_rows, layout.entry_rows, strict=True
    ):
        for block, block_columns in enumerate(_blocks(layout.columns, file_rows)):
            beside = layout.identity if block == 0 else later
            names = [*beside, *block_columns]
            position = start
            for batch in iter_column_batches(path, names, batch_rows):
                keep = batch["perspective"] == perspective if filtering else None
                n_batch = len(batch[names[0]]) if keep is None else int(keep.sum())
                if block == 0:
                    for name in layout.identity:
                        cells = batch[name]
                        parts[name].append(cells if keep is None else cells[keep])
                for name in block_columns:
                    cells = batch[name]
                    values[position : position + n_batch, column_of[name]] = (
                        cells if keep is None else cells[keep]
                    )
                position += n_batch
            if position - start != expected:
                msg = f"{_name((group, sequence))} changed while it was being read"
                raise RuntimeError(msg)
        start += expected
        groups.append(np.full(expected, group, dtype=object))
        sequences.append(np.full(expected, sequence, dtype=object))
    identity: dict[str, npt.NDArray[Any]] = {
        "group": _joined(groups),
        "sequence": _joined(sequences),
    }
    identity.update({name: _joined(pieces) for name, pieces in parts.items()})
    return pd.DataFrame(identity), values


def _joined(pieces: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """The pieces end to end, or an empty array where a read returned no row."""
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=object)


def _name(entry: Entry) -> str:
    return f"{entry[0]}/{entry[1]}"


def _names(entries: Sequence[Entry]) -> list[str]:
    return [_name(entry) for entry in entries]


def _listed(names: Sequence[str]) -> str:
    """The first names, and a count of the rest."""
    shown = ", ".join(names[:_LISTED])
    rest = len(names) - _LISTED
    return shown if rest <= 0 else f"{shown}, and {rest} more"


def _difference(expected: tuple[str, ...], actual: tuple[str, ...]) -> str:
    """What one entry lacks and adds against the first, as a clause for a message."""
    lacks = [name for name in expected if name not in actual]
    adds = [name for name in actual if name not in expected]
    clauses = [
        text
        for text in (
            f"lacks {_listed(lacks)}" if lacks else "",
            f"adds {_listed(adds)}" if adds else "",
        )
        if text
    ]
    return f" ({'; '.join(clauses)})" if clauses else ""
