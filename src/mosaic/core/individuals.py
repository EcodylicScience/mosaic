"""Splits a tracks table into its individuals, pairs them, and refuses repeated rows.

The feature library and ``core.pipeline.resample_tracks`` both read a tracks
table one individual at a time. ``core`` does not import ``behavior``, and this
leaf below both defines the rule once. :func:`individual_pairs` is the one rule
for which individuals a feature relates: two individuals of one arena, or any
two in a table without ``arena``. It imports only the standard library, numpy
and pandas.

The id and frame columns are named as in the standard track schemas of
``core.schema``, because this module cannot read ``COLUMNS`` from
``core.pipeline``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import Scalar

__all__ = [
    "id_arenas",
    "individual_pairs",
    "iter_individuals",
    "refuse_repeated_rows",
    "same_arena_mask",
]

_ID_COLUMN = "id"
_FRAME_COLUMN = "frame"
_ARENA_COLUMN = "arena"


def iter_individuals(
    df: pd.DataFrame, order_col: str = _FRAME_COLUMN
) -> Iterator[tuple[Scalar | None, pd.DataFrame]]:
    """Yield the rows of each individual in *df*, ordered by *order_col*.

    Individuals come in the order in which their ids first appear in *df*. A
    table without an id column is one individual, yielded with ``None``. A row
    whose id is missing belongs to no individual and is not yielded.

    Args:
        df: The rows to split. Left unchanged.
        order_col: The column that orders each individual's rows.

    Yields:
        ``(id, rows)`` per individual, the rows sorted by *order_col* and indexed
        from 0. Rows with equal *order_col* values keep their order in *df*.
    """
    if _ID_COLUMN not in df.columns:
        yield None, df.sort_values(order_col, kind="stable").reset_index(drop=True)
        return
    for individual, rows in df.groupby(_ID_COLUMN, sort=False):
        yield (
            individual,
            rows.sort_values(order_col, kind="stable").reset_index(drop=True),
        )


def refuse_repeated_rows(
    df: pd.DataFrame, who: str, order_col: str = _FRAME_COLUMN, *, remedy: str = ""
) -> None:
    """Refuse *df* when an individual has two rows of one *order_col* value.

    A table without an id column is one individual, as in :func:`iter_individuals`.

    Args:
        df: The rows to check.
        who: The feature or the table that the refusal names.
        order_col: The column that orders each individual's rows.
        remedy: A sentence the refusal ends with, saying what to change.

    Raises:
        ValueError: Naming *who*, the first repeated *order_col* value and its id.
    """
    keys = [order_col]
    if _ID_COLUMN in df.columns:
        keys.append(_ID_COLUMN)
    repeated = np.flatnonzero(df.duplicated(keys).to_numpy())
    if repeated.size == 0:
        return
    first = int(repeated[0])
    where = f"{order_col} {df[order_col].iloc[first]}"
    if _ID_COLUMN in keys:
        where += f" of id {df[_ID_COLUMN].iloc[first]}"
    msg = f"{who}: {where} appears in more than one row"
    if remedy:
        msg = f"{msg}. {remedy}"
    raise ValueError(msg)


def id_arenas(df: pd.DataFrame) -> dict[Hashable, str] | None:
    """Return the arena of each id of *df*, ``None`` when *df* has no ``arena``.

    A row without an arena names none for its id.

    Raises:
        ValueError: If one id appears in two arenas, which ``assign-arenas``
            never writes.
    """
    if _ARENA_COLUMN not in df.columns or _ID_COLUMN not in df.columns:
        return None
    named = df[[_ID_COLUMN, _ARENA_COLUMN]].dropna().drop_duplicates()
    repeated = named[_ID_COLUMN][named[_ID_COLUMN].duplicated()]
    if not repeated.empty:
        identity = repeated.iloc[0]
        message = (
            f"id {identity} appears in two arenas. Assign the arenas again with "
            "assign-arenas, which keeps each id in one."
        )
        raise ValueError(message)
    return {
        identity: str(arena)
        for identity, arena in zip(
            named[_ID_COLUMN].tolist(), named[_ARENA_COLUMN].tolist(), strict=True
        )
    }


def same_arena_mask(
    first: Sequence[Hashable],
    second: Sequence[Hashable],
    arenas: Mapping[Hashable, str] | None,
) -> npt.NDArray[np.bool_]:
    """Return which pairs ``(first[i], second[i])`` a feature may relate.

    Without arenas every pair. With them, the pairs of two individuals of one
    arena, and an individual without an arena is related to none.
    """
    if len(first) != len(second):
        message = f"{len(first)} first ids against {len(second)} second ids"
        raise ValueError(message)
    if arenas is None:
        return np.ones(len(first), dtype=bool)
    codes = {name: code for code, name in enumerate(sorted(set(arenas.values())))}

    def code_of(identity: Hashable) -> int:
        arena = arenas.get(identity)
        return -1 if arena is None else codes[arena]

    first_codes = np.array([code_of(a) for a in first], dtype=np.int64)
    second_codes = np.array([code_of(b) for b in second], dtype=np.int64)
    return (first_codes >= 0) & (first_codes == second_codes)


def individual_pairs[T: Hashable](
    ids: Iterable[T],
    arenas: Mapping[Hashable, str] | None,
    *,
    ordered: bool = False,
) -> list[tuple[T, T]]:
    """Return the pairs of *ids* that a feature relates, in the order of *ids*.

    Without arenas these are ``combinations(ids, 2)``, or ``permutations``
    when *ordered*. With them, only the pairs within one arena
    (:func:`same_arena_mask`).

    Args:
        ids: The individuals, in the order the pairs follow.
        arenas: Each id's arena, from :func:`id_arenas`.
        ordered: Whether to return both orderings of each pair.
    """
    candidates = list(permutations(ids, 2) if ordered else combinations(ids, 2))
    keep = same_arena_mask(
        [a for a, _ in candidates], [b for _, b in candidates], arenas
    )
    return [pair for pair, kept in zip(candidates, keep.tolist(), strict=True) if kept]
