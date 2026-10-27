"""The source tables of an op that reads one tracks variant and writes another.

``resample-tracks`` and ``assign-arenas`` chain: each reads a tracks variant and
writes a new one whose identity carries the source variant as its ``upstream``.
They resolve their source, refuse a scope spanning two variants and refuse to
leave a partial variant by the same rules, defined here once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.entry import Entry
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    row_in_scope,
    select_variant_rows,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "SOURCE_TRACKS_RUN_ID_DESCRIPTION",
    "chained_source_rows",
    "one_source_variant",
    "refuse_incomplete_variant",
]

SOURCE_TRACKS_RUN_ID_DESCRIPTION: Final = (
    "Which tracks variant to read, e.g. 'trex.0.1-abc123def0'. Unset resolves "
    "whichever variant the scope's entries hold, and refuses when they hold two."
)


def chained_source_rows(
    ds: Dataset,
    source_tracks_run_id: str | None,
    entries: list[Entry] | None,
    *,
    producer: str,
) -> pd.DataFrame:
    """Return the source tables a chained op reads, one row per entry.

    Rows that *producer* wrote are excluded before the variant is resolved,
    unless the caller names a variant. A chained op does not read its own
    output: once a run has written its variant, an unpinned re-run would see
    two variants per entry and refuse, and a run resumed after a partial
    failure could not resolve its source. Excluding by producer also covers a
    second run of the op under other params. Chaining the op onto its own
    output is named with ``source_tracks_run_id``.
    """
    index = read_tracks_index(ds)
    if index.empty:
        return index
    if source_tracks_run_id is None and "producer" in index.columns:
        index = index[
            [text_cell(value) != producer for value in index["producer"]]
        ].reset_index(drop=True)
    resolved = select_variant_rows(index, source_tracks_run_id)
    if resolved.empty:
        return resolved
    keep = [
        i
        for i, (_, row) in enumerate(resolved.iterrows())
        if row_in_scope(row, entries)
    ]
    return resolved.iloc[keep].reset_index(drop=True)


def one_source_variant(rows: pd.DataFrame, kind: str) -> str:
    """Return the single variant that *rows* were produced by.

    A chained op's identity carries one ``upstream``, so a scope spanning two
    variants would name two recipes by one identifier.

    Raises:
        ValueError: If *rows* hold more than one variant.
    """
    variants = sorted({text_cell(value) for value in rows["run_id"]})
    if len(variants) > 1:
        message = (
            f"{kind} reads one tracks variant and this scope resolves "
            f"{len(variants)} ({variants}). Its identity carries the variant it "
            "chains from, so a run over two under one name would claim they were "
            "one recipe. Name one with source_tracks_run_id=, or narrow the scope."
        )
        raise ValueError(message)
    return variants[0]


def refuse_incomplete_variant(
    kind: str, variant: str, failed: Sequence[str], total: int
) -> None:
    """Refuse a run that left entries of its variant unwritten.

    A chained variant replaces its source as what every entry resolves to, so
    an entry missing from it would disappear once the source is retired. A
    re-run resumes, because what was written is kept.

    Raises:
        RuntimeError: Naming the first unwritten entries, when *failed* is not
            empty.
    """
    if not failed:
        return
    shown = f"{list(failed[:5])}{'...' if len(failed) > 5 else ''}"
    message = (
        f"{kind} left {len(failed)} of {total} entries unwritten ({shown}). "
        f"Variant {variant} is incomplete; fix the named entries and re-run to "
        "resume."
    )
    raise RuntimeError(message)
