"""Merges the index rows of a dataset and of the datasets it links.

A dataset reads one frame per kind of index, and each link adds another. The
merge is one pure function over those frames, so the lookups of every kind
share one rule and nothing in this module knows what a dataset is: it imports
pandas and the standard library alone, and writes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

__all__ = ["LINK_ID_COLUMN", "merge_linked_rows"]

LINK_ID_COLUMN = "link_id"
"""The column that names the link of a merged row, empty for an own row."""


def _anchored(cell: object, base_dir: Path) -> object:
    """Returns *cell* made absolute against *base_dir* when it is relative.

    A cell that is not a string, is empty, is already absolute or holds a JSON
    array of paths is returned as stored. The array is a multi-clip camera's
    cell and names several files, which one anchor cannot place.
    """
    if not isinstance(cell, str) or not cell or cell.startswith("["):
        return cell
    if Path(cell).is_absolute():
        return cell
    return str(base_dir / cell)


def _linked_frame(
    link_id: str, base_dir: Path, frame: pd.DataFrame, path_columns: Sequence[str]
) -> pd.DataFrame:
    """Returns a copy of *frame* anchored at *base_dir* with *link_id* set."""
    anchored = frame.copy()
    for column in path_columns:
        if column in anchored.columns:
            anchored[column] = anchored[column].map(
                lambda cell: _anchored(cell, base_dir)
            )
    return anchored.assign(**{LINK_ID_COLUMN: link_id})


def merge_linked_rows(
    frames: Sequence[tuple[str, Path, pd.DataFrame]],
    keys: Sequence[str],
    path_columns: Sequence[str],
) -> pd.DataFrame:
    """Merges the index frames of a dataset and of its links into one.

    Each key keeps the rows of the first frame that holds it, so the dataset's
    own rows shadow a linked one and an earlier link shadows a later one. Rows
    that share a key within one frame are all kept as stored. A relative
    path in a linked row is anchored at its own dataset's base directory, which
    makes it name the file from the reading dataset. A path in the dataset's
    own rows stays as stored, because the reading dataset anchors it at its own
    base directory the same way. The frames are not modified.

    Args:
        frames: One ``(link_id, base_dir, frame)`` per dataset in lookup order.
            The first is the reading dataset's own, with an empty ``link_id``.
        keys: The columns that identify a row.
        path_columns: The columns whose cells name one file each. A column that
            a frame lacks is skipped in that frame.

    Returns:
        The merged rows with a fresh index, the columns of the own frame first
        and then each column a link adds in the order met, and a ``link_id``
        column last. A cell of a column that a frame lacks is ``""``. When
        every linked frame is empty the result is the own frame with
        ``link_id`` added and its dtypes unchanged.

    Raises:
        ValueError: *frames* is empty.
    """
    if not frames:
        raise ValueError("merge_linked_rows needs the own frame first")
    own_id, _, own_frame = frames[0]
    own = own_frame.assign(**{LINK_ID_COLUMN: own_id})
    linked = [
        _linked_frame(link_id, base_dir, frame, path_columns)
        for link_id, base_dir, frame in frames[1:]
        if not frame.empty
    ]
    if not linked:
        return own.reset_index(drop=True)

    columns = [
        column
        for column in dict.fromkeys(
            column for frame in (own, *linked) for column in frame.columns
        )
        if column != LINK_ID_COLUMN
    ]
    present = [frame for frame in (own, *linked) if not frame.empty]
    filled = [
        frame.assign(
            **{column: "" for column in columns if column not in frame.columns}
        )
        for frame in present
    ]
    merged = pd.concat(filled, ignore_index=True)
    frame_order = pd.Series(
        [order for order, frame in enumerate(filled) for _ in range(len(frame))]
    )
    first_frame = frame_order.groupby(
        [merged[key] for key in keys], sort=False, dropna=False
    ).transform("min")
    merged = merged[frame_order == first_frame]
    return merged.reindex(columns=[*columns, LINK_ID_COLUMN]).reset_index(drop=True)
