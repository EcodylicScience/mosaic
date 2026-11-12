"""Describe one feature run's rows by group: counts and statistics per value column.

:func:`summarize_run` plans the read with
:func:`~mosaic.core.pipeline.run_matrix.plan_run_read`, refuses a per-frame run and
keys whose types disagree between entries from the plan alone, and only then reads
the values. Each group is one combination of key values that rows hold, and each
statistic is taken over the finite cells of one column of one group.

The reader and scipy are imported inside the functions that use them, so a module
that names :class:`RunSummary` loads neither.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .error_kind import ClassifiedError, ErrorKind
from .run_errors import RunError, entry_label

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.scope import Scope

    from .run_matrix import KeyKind, RunRead
    from .types import Result

__all__ = [
    "MAX_PAIRWISE_GROUPS",
    "KeyValue",
    "RunSummary",
    "SummaryConflict",
    "SummaryInvalid",
    "SummaryReason",
    "SummaryTest",
    "summarize_run",
]

type KeyValue = str | int | float | bool
"""One value of a group key."""

type SummaryTest = Literal["kruskal", "mannwhitney"]
"""A test between groups: Kruskal-Wallis over all groups, or Mann-Whitney per pair."""

type SummaryReason = Literal["too_few_groups", "no_values", "no_variation"]
"""Why a test row has no statistic: fewer than two groups with values, a side of a
pair without values, or every value the test reads equal."""

type _KeyType = Literal["string", "integer", "floating", "boolean", "null"]
"""The one type a key takes across entries, ``null`` when it holds no value."""

_RESERVED: Final = (
    "column",
    "n",
    "n_rows",
    "n_entries",
    "n_missing",
    "mean",
    "sd",
    "sem",
    "median",
)
"""The summary's own column names, which no key may take."""

_TESTS: Final[tuple[SummaryTest, ...]] = ("kruskal", "mannwhitney")

MAX_PAIRWISE_GROUPS: Final = 12
"""The most groups that ``mannwhitney`` compares pairwise, 66 pairs per column."""

_PER_FRAME: Final = ("frame", "time")
"""Columns that mark a run as one row per frame."""

_EXACT: Final = 2**53
"""The least integer magnitude that float64 does not tell from its neighbor."""

_KRUSKAL_COLUMNS: Final = {
    "column": "object",
    "k": "int64",
    "n": "int64",
    "statistic": "float64",
    "df": "float64",
    "p": "float64",
    "reason": "object",
}
"""The columns of a Kruskal-Wallis row, with their dtypes."""

_MANN_WHITNEY_COLUMNS: Final = {
    "n_a": "int64",
    "n_b": "int64",
    "statistic": "float64",
    "p": "float64",
    "p_holm": "float64",
    "comparisons": "int64",
    "reason": "object",
}
"""The columns of a Mann-Whitney row after ``column`` and the keys of both groups."""

type _KruskalRow = tuple[str, int, int, float, float, float, SummaryReason | None]
"""``column``, ``k``, ``n``, ``statistic``, ``df``, ``p`` and ``reason``."""

type _PairRow = tuple[int, int, float, float, SummaryReason | None]
"""``n_a``, ``n_b``, ``statistic``, ``p`` and ``reason`` of one pair."""

_SINGLE: Final[dict[str, _KeyType]] = {
    "string": "string",
    "integer": "integer",
    "floating": "floating",
    "boolean": "boolean",
}
"""The type of a key stored under one kind in every entry that holds a value."""

_DTYPES: Final[dict[_KeyType, str]] = {
    "string": "object",
    "integer": "int64",
    "floating": "float64",
    "boolean": "bool",
    "null": "object",
}
"""The numpy dtype that holds a key's cells while groups are formed."""

_COLUMN_DTYPES: Final[dict[_KeyType, str]] = {**_DTYPES, "string": "str"}
"""The dtype of a key's column in a summary frame, whether or not it has rows.

``str`` is the dtype pandas infers for text, so a frame without rows has the
dtype of one with rows.
"""


class SummaryInvalid(RunError, ClassifiedError, ValueError):
    """The arguments state no summary, or the run cannot be summarized as asked."""

    error_kind: ClassVar[ErrorKind] = "invalid"


class SummaryConflict(RunError, ClassifiedError, ValueError):
    """The run's entries store a key under types that do not merge."""

    error_kind: ClassVar[ErrorKind] = "conflict"


@dataclass(frozen=True, slots=True, eq=False)
class RunSummary:
    """The rows of one run, counted and described by group.

    Attributes:
        feature: The storage name of the feature that wrote the run.
        run_id: The run that was read.
        by: The group keys, in order.
        columns: The value columns summarized, in order.
        entries: The ``(group, sequence)`` pairs that were read, sorted.
        missing: The entries in the scope that the run has no readable output for.
        finished: Whether the run is marked finished.
        entries_without_rows: The entries read whose output holds no row.
        rows_without_key: The rows left out because a ``by`` or ``where`` key is
            null on them, counted before ``where``.
        max_rows_per_entry: The most rows that one entry contributes to the groups.
        groups: One row per group, sorted by the keys: the ``by`` keys,
            ``n_rows`` and ``n_entries``, the distinct ``(group, sequence)`` of
            the group's rows.
        statistics: One row per group and column, groups first: the ``by`` keys,
            ``column``, ``n``, ``n_missing``, ``mean``, ``sd``, ``sem`` and
            ``median``. A statistic without a value is NaN.
        kruskal: One row per column, through ``scipy.stats.kruskal`` over the
            groups with a finite cell: ``column``, ``k`` groups entered, their
            total ``n``, ``statistic`` (H), ``df`` (``k - 1``), ``p`` and
            ``reason``. Empty when the test was not asked.
        mann_whitney: One row per column and pair of groups, columns first and
            pairs in group order: ``column``, ``<key>_a`` and ``<key>_b`` for
            each key, ``n_a``, ``n_b``, ``statistic`` (U of the first group),
            ``p`` (two-sided), ``p_holm``, ``comparisons`` and ``reason``.
            ``p_holm`` is Holm's step-down adjustment over the pairs of one
            column that have a ``p``, and ``comparisons`` their count. Columns
            are not adjusted against each other. Empty when the test was not
            asked.

    A test row without a statistic names its :data:`SummaryReason`, and its
    ``statistic``, ``p`` and ``p_holm`` are NaN, as is ``df`` with fewer than two
    groups. ``reason`` is ``None`` on a tested row.
    """

    feature: str
    run_id: str
    by: tuple[str, ...]
    columns: tuple[str, ...]
    entries: tuple[Entry, ...]
    missing: tuple[Entry, ...]
    finished: bool
    entries_without_rows: tuple[Entry, ...]
    rows_without_key: int
    max_rows_per_entry: int
    groups: pd.DataFrame
    statistics: pd.DataFrame
    kruskal: pd.DataFrame
    mann_whitney: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _Key:
    """One key's resolved type and its value on every row read, ``None`` for null."""

    kind: _KeyType
    cells: tuple[KeyValue | None, ...]


def summarize_run(
    ds: Dataset,
    run: Result[str],
    *,
    by: Sequence[str] = (),
    columns: str | Sequence[str] | None = None,
    tests: Sequence[SummaryTest] = (),
    where: Mapping[str, Sequence[KeyValue]] | None = None,
    scope: Scope | None = None,
    allow_partial: bool = False,
) -> RunSummary:
    """Count and describe the rows of one run by group, per value column.

    A row of the run's table is one observation: an entry for
    ``sequence-aggregate``, an individual for ``individual-aggregate``. A run with
    one row per frame, whose table has a ``frame`` or a ``time`` column, is
    refused before a value is read, since frames of one recording are not
    independent. Rows of one entry are not independent either: when
    ``max_rows_per_entry`` exceeds 1 and a key is constant within entries, as a
    strain is, each individual of a recording counts as a replicate of it.
    ``n_entries`` beside ``n_rows`` shows how many recordings stand behind a group.

    Each key takes one type across entries. A key null in an entry's whole file
    fits any type, and an integer key stored as floating point in one entry
    merges with the integers of another, reported as integers. An integer label
    that has a null in every entry is stored as floating point, so its groups read
    1.0, 2.0. Write such a label without nulls or as text to get integers. Text,
    booleans and numbers do not merge. The refusals of
    :func:`~mosaic.core.pipeline.run_matrix.plan_run_read` and
    :func:`~mosaic.core.pipeline.run_matrix.read_run` pass through unchanged,
    among them a key that no entry writes.

    Args:
        ds: The dataset holding the run.
        run: The run to summarize, as for
            :func:`~mosaic.core.pipeline.run_matrix.load_run_matrix`.
        by: The group keys: ``group``, ``sequence``, an identity column the run
            writes, or any other column it writes, such as a kept label. No key
            groups every row into one group.
        columns: The value columns, as for ``load_run_matrix``. A ``by`` or
            ``where`` key is never among them.
        tests: The tests between groups to run: ``kruskal``, ``mannwhitney`` or
            both, each over the finite cells of one column. A test reads no
            value of a group without one, and calls no scipy function when
            every value it reads is equal. Their frames are empty with their
            columns when left out.
        where: For each key named, the values a row's key must take to be kept.
            A key need not be in *by*. A ``where`` column is read as a key and is
            never a value.
        scope: The entries to read.
        allow_partial: Read what the run holds where it is partial.

    Returns:
        The summary, with groups sorted by the keys in order, each ascending in
        its type.

    Raises:
        SummaryInvalid: A ``by`` or ``where`` key among the names in
            *columns*; a ``by`` key named ``column``, ``n``, ``n_rows``,
            ``n_entries``, ``n_missing``, ``mean``, ``sd``, ``sem`` or
            ``median``; a ``where`` key with no value; a test name other than
            ``kruskal`` and ``mannwhitney``; ``mannwhitney`` over more than
            :data:`MAX_PAIRWISE_GROUPS` groups, refused before any test runs;
            a per-frame run; or a key that is
            not text, an integer, floating point or a boolean (a ``ValueError``,
            kind ``invalid``).
        SummaryConflict: A key that is text, a boolean or a number in different
            entries, or a key mixing integers and floating point that holds a
            value of magnitude at least 2**53 (a ``ValueError``, kind
            ``conflict``).

    """
    from .run_matrix import plan_run_read, read_run

    keys_by = tuple(dict.fromkeys(by))
    kept = {name: tuple(values) for name, values in (where or {}).items()}
    _refuse_arguments(run, keys_by, columns, kept, tests)
    names = tuple(dict.fromkeys((*keys_by, *kept)))
    read = plan_run_read(
        ds,
        run,
        scope=scope,
        columns=columns,
        keys=names,
        allow_partial=allow_partial,
    )
    _refuse_per_frame(read)
    kinds: dict[str, _KeyType | Literal["mixed"]] = {
        name: _resolve_kind(read, name) for name in names
    }
    matrix = read_run(read, np.float64)
    keys = {
        name: _typed(read, name, kinds[name], matrix.index[name].tolist())
        for name in names
    }
    found = read.outputs.found
    entries_without_rows = tuple(
        entry
        for (entry, _), n_rows in zip(found, read.layout.entry_rows, strict=True)
        if n_rows == 0
    )
    n_read = len(matrix.index)
    keyed = [
        all(keys[name].cells[row] is not None for name in names)
        for row in range(n_read)
    ]
    wanted = {
        name: {_token(value) for value in values} for name, values in kept.items()
    }
    rows = [
        row
        for row in range(n_read)
        if keyed[row]
        and all(
            _token(_present(keys[name].cells[row])) in tokens
            for name, tokens in wanted.items()
        )
    ]
    entry_of = list(
        zip(
            matrix.index["group"].astype(str).tolist(),
            matrix.index["sequence"].astype(str).tolist(),
            strict=True,
        )
    )
    group_keys, members = _groups(keys_by, keys, rows)
    _refuse_pairwise(read, keys_by, tests, len(members))
    values = np.asarray(matrix.values, dtype=np.float64)
    samples = {
        column: [_finite(values[group_rows, position]) for group_rows in members]
        for position, column in enumerate(matrix.columns)
    }
    return RunSummary(
        feature=matrix.feature,
        run_id=matrix.run_id,
        by=keys_by,
        columns=matrix.columns,
        entries=matrix.entries,
        missing=matrix.missing,
        finished=matrix.finished,
        entries_without_rows=entries_without_rows,
        rows_without_key=keyed.count(False),
        max_rows_per_entry=max(
            Counter(entry_of[row] for row in rows).values(), default=0
        ),
        groups=_groups_frame(keys_by, keys, group_keys, members, entry_of),
        statistics=_statistics_frame(
            keys_by, keys, group_keys, members, matrix.columns, values
        ),
        kruskal=_kruskal_frame(samples if "kruskal" in tests else {}),
        mann_whitney=_mann_whitney_frame(
            keys_by, keys, group_keys, samples if "mannwhitney" in tests else {}
        ),
    )


def _refuse_arguments(
    run: Result[str],
    by: tuple[str, ...],
    columns: str | Sequence[str] | None,
    where: Mapping[str, tuple[KeyValue, ...]],
    tests: Sequence[SummaryTest],
) -> None:
    """Raise for arguments that state no summary, before the run is read."""
    named = (
        ()
        if columns is None
        else (columns,)
        if isinstance(columns, str)
        else tuple(columns)
    )
    where_shown = {name: list(values) for name, values in where.items()}
    stated: dict[str, tuple[object, tuple[str, ...]]] = {
        "by": (list(by), by),
        "where": (where_shown, tuple(where)),
    }
    clashes = {
        argument: (shown, [name for name in keys if name in named])
        for argument, (shown, keys) in stated.items()
    }
    valued = list(
        dict.fromkeys(name for _, names in clashes.values() for name in names)
    )
    reserved = [name for name in by if name in _RESERVED]
    empty = [name for name, values in where.items() if not values]
    unknown = [test for test in tests if test not in _TESTS]
    if valued:
        named_twice = " and ".join(
            f"{argument}={shown!r} names {names!r}"
            for argument, (shown, names) in clashes.items()
            if names
        )
        msg = (
            f"{named_twice}, also among columns={list(named)!r}. "
            f"A column is a group key or a value, not both."
        )
    elif reserved:
        msg = (
            f"by={list(by)!r} names {reserved!r}, a column the summary writes "
            f"itself ({', '.join(_RESERVED)}). Group by another column."
        )
    elif empty:
        msg = (
            f"where={where_shown!r} lists no value for {empty!r}, so it keeps no "
            f"row. List the values to keep, or leave the key out."
        )
    elif unknown:
        msg = (
            f"tests={list(tests)!r} names {unknown!r}. The tests are "
            f"{', '.join(_TESTS)}."
        )
    else:
        return
    raise SummaryInvalid(
        msg,
        feature=run.feature,
        run_id=run.run_id,
        columns=[*valued, *reserved, *empty],
    )


def _refuse_pairwise(
    read: RunRead, by: tuple[str, ...], tests: Sequence[SummaryTest], groups: int
) -> None:
    """Raise for ``mannwhitney`` over more than :data:`MAX_PAIRWISE_GROUPS` groups."""
    if "mannwhitney" not in tests or groups <= MAX_PAIRWISE_GROUPS:
        return
    pairs = MAX_PAIRWISE_GROUPS * (MAX_PAIRWISE_GROUPS - 1) // 2
    msg = (
        f"{read.feature} run {read.run_id}: mannwhitney compares every pair of the "
        f"{groups} groups that by={list(by)!r} forms, above the limit of "
        f"{MAX_PAIRWISE_GROUPS} groups ({pairs} pairs per column). Narrow the "
        f"groups with where= or scope=, group by fewer keys, or run kruskal alone."
    )
    raise SummaryInvalid(msg, feature=read.feature, run_id=read.run_id, columns=by)


def _refuse_per_frame(read: RunRead) -> None:
    """Raise for a run with one row per frame, from its first footer alone."""
    present = [name for name, _ in read.footer_columns if name in _PER_FRAME]
    if not present:
        return
    msg = (
        f"{read.feature} run {read.run_id} holds one row per frame (its table has "
        f"{' and '.join(repr(name) for name in present)}), and frames of one "
        f"recording are not independent observations. Reduce it first with "
        f"sequence-aggregate or individual-aggregate, keeping the group labels "
        f"with keep_columns, and summarize that run."
    )
    raise SummaryInvalid(msg, feature=read.feature, run_id=read.run_id, columns=present)


def _resolve_kind(read: RunRead, name: str) -> _KeyType | Literal["mixed"]:
    """The one type of key *name* across the entries read.

    ``mixed`` is a key stored as integers in some entries and floating point in
    others, whose values decide whether it reports integers.
    """
    if name in ("group", "sequence"):
        return "string"
    by_entry: list[tuple[Entry, KeyKind]] = []
    if name in read.keys:
        for (entry, _), kinds in zip(read.outputs.found, read.key_kinds, strict=True):
            by_entry.append((entry, dict(kinds)[name]))
    else:
        by_entry.append((read.outputs.found[0][0], dict(read.footer_columns)[name]))
    present = {kind for _, kind in by_entry if kind != "null"}
    if "other" in present:
        msg = (
            f"{read.feature} run {read.run_id}: the key {name!r} holds values that "
            f"are not text, integers, floating point or booleans, such as "
            f"timestamps or lists. Group by a column of one of those types."
        )
        raise SummaryInvalid(
            msg, feature=read.feature, run_id=read.run_id, columns=[name]
        )
    if not present:
        return "null"
    if present == {"integer", "floating"}:
        return "mixed"
    if len(present) == 1:
        (kind,) = present
        return _SINGLE[kind]
    first = {kind: entry_label(entry) for entry, kind in reversed(by_entry)}
    stored = ", ".join(f"{kind} in {first[kind]}" for kind in sorted(present))
    msg = (
        f"{read.feature} run {read.run_id}: the key {name!r} is stored as {stored}. "
        f"Text, booleans and numbers do not merge into one key, so write it as "
        f"one type in every entry, or narrow the read with scope=."
    )
    raise SummaryConflict(msg, feature=read.feature, run_id=read.run_id, columns=[name])


def _typed(
    read: RunRead,
    name: str,
    kind: _KeyType | Literal["mixed"],
    raw: Sequence[object],
) -> _Key:
    """The cells of one key cast to its type, NaN and ``None`` alike as null."""
    cells = [_scalar(cell) for cell in raw]
    if kind == "mixed":
        numbers = [cell for cell in cells if isinstance(cell, int | float)]
        if any(abs(number) >= _EXACT for number in numbers):
            msg = (
                f"{read.feature} run {read.run_id}: the key {name!r} is stored as "
                f"integers in some entries and floating point in others, and holds "
                f"a value of magnitude at least 2**53, which floating point does "
                f"not hold exactly. Write it as one type in every entry."
            )
            raise SummaryConflict(
                msg, feature=read.feature, run_id=read.run_id, columns=[name]
            )
        integral = all(float(number).is_integer() for number in numbers)
        kind = "integer" if integral else "floating"
    cast = {
        "string": str,
        "integer": int,
        "floating": float,
        "boolean": bool,
        "null": str,
    }[kind]
    return _Key(kind, tuple(None if cell is None else cast(cell) for cell in cells))


def _scalar(cell: object) -> KeyValue | None:
    """One cell, as ``Series.tolist`` gives it, or ``None`` where it is null."""
    if isinstance(cell, float) and math.isnan(cell):
        return None
    if isinstance(cell, str | int | float):
        return cell
    return None


def _present(value: KeyValue | None) -> KeyValue:
    assert value is not None
    return value


def _token(value: KeyValue) -> tuple[str, KeyValue]:
    """A value tagged by its family, so ``True`` matches no number and ``1`` matches ``1.0``."""
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        return ("string", value)
    return ("number", value)


def _groups(
    by: tuple[str, ...], keys: Mapping[str, _Key], rows: list[int]
) -> tuple[list[tuple[npt.NDArray[np.generic], ...]], list[list[int]]]:
    """The groups that the kept rows form, sorted by the keys in order.

    Returns:
        Each group's value of each key, and each group's rows in read order.
    """
    if not by:
        return ([()], [rows]) if rows else ([], [])
    uniques: list[npt.NDArray[np.generic]] = []
    codes: list[npt.NDArray[np.intp]] = []
    for name in by:
        key = keys[name]
        cells = np.array([key.cells[row] for row in rows], dtype=_DTYPES[key.kind])
        unique, inverse = np.unique(cells, return_inverse=True)
        uniques.append(unique)
        codes.append(inverse.reshape(-1))
    members: dict[tuple[int, ...], list[int]] = {}
    for position, row in enumerate(rows):
        code = tuple(int(column[position]) for column in codes)
        members.setdefault(code, []).append(row)
    ordered = sorted(members)
    group_keys = [
        tuple(uniques[k][code[k] : code[k] + 1] for k in range(len(by)))
        for code in ordered
    ]
    return group_keys, [members[code] for code in ordered]


def _key_columns(
    by: tuple[str, ...],
    keys: Mapping[str, _Key],
    group_keys: list[tuple[npt.NDArray[np.generic], ...]],
    repeat: int,
) -> dict[str, pd.Series]:
    """Each key's value per group, each value repeated *repeat* times, in its type."""
    return {
        name: pd.Series(
            np.concatenate(
                [np.repeat(group[k], repeat) for group in group_keys]
                or [np.empty(0, dtype=_DTYPES[keys[name].kind])]
            ),
            dtype=_COLUMN_DTYPES[keys[name].kind],
        )
        for k, name in enumerate(by)
    }


def _groups_frame(
    by: tuple[str, ...],
    keys: Mapping[str, _Key],
    group_keys: list[tuple[npt.NDArray[np.generic], ...]],
    members: list[list[int]],
    entry_of: list[tuple[str, str]],
) -> pd.DataFrame:
    """One row per group: its keys, its rows and its distinct entries."""
    return pd.DataFrame(
        {
            **_key_columns(by, keys, group_keys, 1),
            "n_rows": pd.Series([len(rows) for rows in members], dtype="int64"),
            "n_entries": pd.Series(
                [len({entry_of[row] for row in rows}) for rows in members],
                dtype="int64",
            ),
        }
    )


def _statistics_frame(
    by: tuple[str, ...],
    keys: Mapping[str, _Key],
    group_keys: list[tuple[npt.NDArray[np.generic], ...]],
    members: list[list[int]],
    columns: tuple[str, ...],
    values: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """One row per group and column of the statistics over its finite cells."""
    described = [
        _describe(values[rows, position])
        for rows in members
        for position in range(len(columns))
    ]
    return pd.DataFrame(
        {
            **_key_columns(by, keys, group_keys, len(columns)),
            "column": pd.Series(list(columns) * len(members), dtype="object"),
            "n": pd.Series([n for n, *_ in described], dtype="int64"),
            "n_missing": pd.Series([row[1] for row in described], dtype="int64"),
            **{
                name: pd.Series([row[position] for row in described], dtype="float64")
                for position, name in enumerate(("mean", "sd", "sem", "median"), 2)
            },
        }
    )


def _describe(
    cells: npt.NDArray[np.float64],
) -> tuple[int, int, float, float, float, float]:
    """``n``, ``n_missing``, ``mean``, ``sd``, ``sem`` and ``median`` of the finite cells."""
    finite = _finite(cells)
    n = len(finite)
    nan = math.nan
    mean = float(np.mean(finite)) if n else nan
    median = float(np.median(finite)) if n else nan
    sd = float(np.std(finite, ddof=1)) if n >= 2 else nan
    sem = sd / math.sqrt(n) if n >= 2 else nan
    return n, len(cells) - n, mean, sd, sem, median


def _finite(cells: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The finite cells of *cells*, in order."""
    return cells[np.isfinite(cells)]


def _varies(*samples: npt.NDArray[np.float64]) -> bool:
    """Whether the cells of *samples*, none of them empty, hold two values."""
    joined = np.concatenate(samples)
    return bool(joined.min() != joined.max())


def _frame(dtypes: Mapping[str, str], rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    """A frame of *rows*, one value per column of *dtypes* in order."""
    return pd.DataFrame(
        {
            name: pd.Series([row[position] for row in rows], dtype=dtype)
            for position, (name, dtype) in enumerate(dtypes.items())
        }
    )


def _kruskal_frame(
    samples: Mapping[str, list[npt.NDArray[np.float64]]],
) -> pd.DataFrame:
    """One Kruskal-Wallis row per column of *samples*, each group's finite cells."""
    rows = [_kruskal(column, cells) for column, cells in samples.items()]
    return _frame(_KRUSKAL_COLUMNS, rows)


def _kruskal(column: str, cells: list[npt.NDArray[np.float64]]) -> _KruskalRow:
    """Kruskal-Wallis across the groups of one column that hold a finite cell."""
    from scipy import stats

    entered = [group for group in cells if len(group)]
    k, n = len(entered), sum(len(group) for group in entered)
    if k < 2:
        return (column, k, n, math.nan, math.nan, math.nan, "too_few_groups")
    if not _varies(*entered):
        return (column, k, n, math.nan, k - 1, math.nan, "no_variation")
    tested = stats.kruskal(*entered)
    return (column, k, n, float(tested.statistic), k - 1, float(tested.pvalue), None)


def _mann_whitney_frame(
    by: tuple[str, ...],
    keys: Mapping[str, _Key],
    group_keys: list[tuple[npt.NDArray[np.generic], ...]],
    samples: Mapping[str, list[npt.NDArray[np.float64]]],
) -> pd.DataFrame:
    """One Mann-Whitney row per column of *samples* and pair of groups."""
    pairs = list(combinations(range(len(group_keys)), 2))
    named: list[str] = []
    tested: list[_PairRow] = []
    adjusted: list[float] = []
    counts: list[int] = []
    for column, cells in samples.items():
        rows = [_mann_whitney(cells[a], cells[b]) for a, b in pairs]
        held = _holm([row[3] for row in rows])
        named.extend([column] * len(rows))
        tested.extend(rows)
        adjusted.extend(held)
        counts.extend([sum(not math.isnan(p) for p in held)] * len(rows))
    sides = [pair for _ in samples for pair in pairs]
    per_side = {
        side: _key_columns(by, keys, [group_keys[pair[end]] for pair in sides], 1)
        for end, side in enumerate("ab")
    }
    named_keys = pd.DataFrame(
        {
            "column": pd.Series(named, dtype="object"),
            **{f"{name}_{side}": per_side[side][name] for name in by for side in "ab"},
        }
    )
    results = _frame(
        _MANN_WHITNEY_COLUMNS,
        [
            (n_a, n_b, statistic, p, p_holm, count, reason)
            for (n_a, n_b, statistic, p, reason), p_holm, count in zip(
                tested, adjusted, counts, strict=True
            )
        ],
    )
    return pd.concat([named_keys, results], axis=1)


def _mann_whitney(
    first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
) -> _PairRow:
    """Two-sided Mann-Whitney of one pair, ``statistic`` being U of *first*."""
    from scipy import stats

    n_a, n_b = len(first), len(second)
    if not n_a or not n_b:
        return (n_a, n_b, math.nan, math.nan, "no_values")
    if not _varies(first, second):
        return (n_a, n_b, math.nan, math.nan, "no_variation")
    tested = stats.mannwhitneyu(first, second, alternative="two-sided", method="auto")
    return (n_a, n_b, float(tested.statistic), float(tested.pvalue), None)


def _holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment of the p values that are not NaN.

    The smallest of *m* is multiplied by *m*, the next by *m - 1*, and so on; each
    is raised to the one before it, so the order is kept, and capped at 1. Equal p
    values are taken in their order in *p_values*. A NaN stays NaN.
    """
    ranked = sorted(
        (p, position) for position, p in enumerate(p_values) if not math.isnan(p)
    )
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, position) in enumerate(ranked):
        running = max(running, min(1.0, (len(ranked) - rank) * p))
        adjusted[position] = running
    return adjusted
