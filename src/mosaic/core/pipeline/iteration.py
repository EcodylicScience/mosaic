from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.pipeline.tracks_index import read_tracks_index, select_variant_rows

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Helpers ---


def _filter_index(
    df_idx: pd.DataFrame,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Filter index rows by group, sequence, and/or allowed (group, sequence) pairs."""
    mask = pd.Series(True, index=df_idx.index)
    if groups is not None:
        mask &= df_idx["group"].isin(set(groups))
    if sequences is not None:
        mask &= df_idx["sequence"].isin(set(sequences))
    if allowed_pairs is not None:
        pair_mask = pd.Series(
            [
                (row["group"], row["sequence"]) in allowed_pairs
                for _, row in df_idx.iterrows()
            ],
            index=df_idx.index,
        )
        mask &= pair_mask
    return df_idx[mask]


# --- Sequence iteration ---


def yield_sequences(
    ds: Dataset,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
    run_id: str | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """
    Yield (group, sequence, df) for standardized tracks present in tracks/index.csv,
    filtered by groups and/or sequences if provided.

    ``run_id`` names one tracks variant. Both iterators here walk *rows*, so
    without the shared variant selector an entry carrying two of them would be
    yielded twice -- silently doubling a sequence rather than failing.
    """
    df_idx = _filter_index(
        select_variant_rows(read_tracks_index(ds), run_id),
        groups,
        sequences,
        allowed_pairs,
    )

    for _, row in df_idx.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        p = ds.resolve_path(row["abs_path"])
        if not p.exists():
            raise FileNotFoundError(f"Stale tracks index: ({g},{s}) -> {p}")
        yield g, s, pd.read_parquet(p)
