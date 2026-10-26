"""Splits a tracks table into its individuals and refuses an individual's repeated rows.

The feature library and ``core.pipeline.resample_tracks`` both read a tracks
table one individual at a time. ``core`` does not import ``behavior``, and this
leaf below both defines the rule once. It imports only the standard library,
numpy and pandas.

The id and frame columns are named as in the standard track schemas of
``core.schema``, because this module cannot read ``COLUMNS`` from
``core.pipeline``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import Scalar

__all__ = ["iter_individuals", "refuse_repeated_rows"]

_ID_COLUMN = "id"
_FRAME_COLUMN = "frame"


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
