"""Assign each row of a tracks table to the arena that contains its body center.

The ``assign-arenas`` op (``tracking/ops/assign_arenas.py``) applies
:func:`assign_arenas` to each entry of a tracks variant. A row is placed by its
``X`` and ``Y``, the body center in source pixels, against the half-open
rectangles of an :class:`~mosaic.core.arenas.ArenaLayout`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.arenas import ArenaLayout, Rect
from mosaic.core.schema import ARENA_COLUMN

__all__ = [
    "DROP_REASONS",
    "ArenaCapacityError",
    "AssignmentReport",
    "DropReason",
    "assign_arenas",
]

type DropReason = Literal["no_position", "outside", "excluded", "over_capacity"]
"""Why a row was dropped: no body center, in no arena, in an excluded rect, or
above its arena's capacity in its frame."""

DROP_REASONS: Final[tuple[DropReason, ...]] = (
    "no_position",
    "outside",
    "excluded",
    "over_capacity",
)
"""Every :data:`DropReason`, in report order."""

_CONFIDENCE: Final = "confidence"


class ArenaCapacityError(ValueError):
    """An arena holds more rows than its capacity in one frame, and none can be ranked."""


@dataclass(frozen=True, slots=True)
class AssignmentReport:
    """What :func:`assign_arenas` kept and dropped.

    Attributes:
        rows: The rows of the input table.
        kept: The rows kept in each arena, in the layout's order.
        dropped: The rows dropped for each reason, in :data:`DROP_REASONS` order.
    """

    rows: int
    kept: Mapping[str, int]
    dropped: Mapping[DropReason, int]


def _inside(
    rect: Rect, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Return which points lie in the half-open *rect*."""
    return (x >= rect.x0) & (x < rect.x1) & (y >= rect.y0) & (y < rect.y1)


def _over_capacity(
    df: pd.DataFrame,
    ordinal: npt.NDArray[np.int64],
    keep: npt.NDArray[np.bool_],
    layout: ArenaLayout,
) -> npt.NDArray[np.bool_]:
    """Return the kept rows above their arena's capacity in their frame.

    The rows with the highest ``confidence`` stay. A missing confidence ranks
    last, and rows of equal confidence keep their table order.

    Raises:
        ArenaCapacityError: If an arena is above capacity and the table has no
            ``confidence`` column.
    """
    over = np.zeros(len(df), dtype=bool)
    frames = df["frame"].to_numpy()
    confidence = (
        pd.to_numeric(df[_CONFIDENCE], errors="coerce").to_numpy(dtype=float)
        if _CONFIDENCE in df.columns
        else None
    )
    for index, arena in enumerate(layout.arenas):
        if arena.capacity is None:
            continue
        rows = np.flatnonzero(keep & (ordinal == index))
        if rows.size == 0:
            continue
        counts = pd.Series(frames[rows]).value_counts()
        if not (counts > arena.capacity).any():
            continue
        if confidence is None:
            frame = counts[counts > arena.capacity].index.min()
            message = (
                f"arena {arena.name!r} holds more than its capacity of "
                f"{arena.capacity} at frame {frame}, and the table has no "
                f"'{_CONFIDENCE}' column to keep the likeliest rows by."
            )
            raise ArenaCapacityError(message)
        score = np.nan_to_num(confidence[rows], nan=-np.inf)
        ranked = pd.DataFrame({"frame": frames[rows], "score": score, "row": rows})
        ranked = ranked.sort_values(
            ["frame", "score", "row"], ascending=[True, False, True], kind="stable"
        )
        rank = ranked.groupby("frame", sort=False).cumcount().to_numpy()
        over[ranked["row"].to_numpy()[rank >= arena.capacity]] = True
    return over


def _assigned_ids(
    df: pd.DataFrame, ordinal: npt.NDArray[np.int64], layout: ArenaLayout
) -> npt.NDArray[np.int64]:
    """Return each kept row's id, unique in the entry and within one arena.

    When every arena holds one individual, an id is its arena's ordinal, the
    same in every sequence of a recording. Otherwise ids number the
    ``(ordinal, source id)`` pairs in sorted order.
    """
    if all(arena.capacity == 1 for arena in layout.arenas):
        return ordinal.astype(np.int64)
    pairs = pd.DataFrame({"ordinal": ordinal, "source": df["id"].to_numpy()})
    numbers = pairs.groupby(["ordinal", "source"], sort=True).ngroup()
    return np.asarray(numbers, dtype=np.int64)


def assign_arenas(
    df: pd.DataFrame, layout: ArenaLayout
) -> tuple[pd.DataFrame, AssignmentReport]:
    """Return *df*'s rows that lie in an arena, with ``arena`` and new ids.

    A row is in the arena whose region contains its body center (``X``,
    ``Y``). A row without a center, in no region, or in its arena's ``exclude``
    rects is dropped. Above an arena's capacity in one frame the rows with the
    highest ``confidence`` stay. ``id`` is renumbered so that each id stays in
    one arena (:func:`_assigned_ids`). Frames and every other column are
    unchanged, and rows keep their order. Writes nothing.

    Args:
        df: One entry's tracks table in source pixels.
        layout: The entry's arena layout.

    Returns:
        The kept rows, and what was kept and dropped.

    Raises:
        ValueError: If *df* lacks ``X``, ``Y``, ``frame`` or ``id``.
        ArenaCapacityError: If an arena is above capacity and *df* has no
            ``confidence`` column.
    """
    missing = [
        column for column in ("frame", "id", "X", "Y") if column not in df.columns
    ]
    if missing:
        message = f"assigning arenas needs the columns {missing}, which the table lacks"
        raise ValueError(message)
    x = pd.to_numeric(df["X"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype=float)
    no_position = np.isnan(x) | np.isnan(y)
    ordinal = np.full(len(df), -1, dtype=np.int64)
    excluded = np.zeros(len(df), dtype=bool)
    for index, arena in enumerate(layout.arenas):
        inside = _inside(arena.region, x, y)
        ordinal[inside] = index
        for rect in arena.exclude:
            excluded |= inside & _inside(rect, x, y)
    outside = ~no_position & (ordinal < 0)
    keep = ~no_position & ~outside & ~excluded
    over = _over_capacity(df, ordinal, keep, layout)
    keep &= ~over

    kept = df.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    kept_ordinal = ordinal[keep]
    kept["id"] = _assigned_ids(kept, kept_ordinal, layout)
    names = np.array(layout.names, dtype=object)
    kept[ARENA_COLUMN] = names[kept_ordinal] if len(kept) else pd.Series(dtype=object)
    counts = np.bincount(kept_ordinal, minlength=len(layout.arenas))
    report = AssignmentReport(
        rows=len(df),
        kept={
            name: int(count) for name, count in zip(layout.names, counts, strict=True)
        },
        dropped={
            "no_position": int(no_position.sum()),
            "outside": int(outside.sum()),
            "excluded": int(excluded.sum()),
            "over_capacity": int(over.sum()),
        },
    )
    return kept, report
