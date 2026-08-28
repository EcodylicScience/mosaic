from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.pipeline.tracks_index import read_tracks_index, select_variant_rows
from mosaic.core.scope import Scope

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Helpers ---


def _filter_index(df_idx: pd.DataFrame, scope: Scope | None = None) -> pd.DataFrame:
    """Keep the index rows *scope* names.

    ``None`` and an unset selector both keep every row. A camera-addressed
    selection narrows to its ``(group, sequence)`` pairs, since an index row is
    keyed without a camera.
    """
    selector = scope if scope is not None else Scope()
    mask = pd.Series(True, index=df_idx.index)
    if selector.groups is not None:
        mask &= df_idx["group"].isin(set(selector.groups))
    if selector.sequences is not None:
        mask &= df_idx["sequence"].isin(set(selector.sequences))
    pairs = selector.entry_pairs
    if pairs is not None:
        mask &= pd.Series(
            [(row["group"], row["sequence"]) in pairs for _, row in df_idx.iterrows()],
            index=df_idx.index,
        )
    return df_idx[mask]


# --- Sequence iteration ---


def yield_sequences(
    ds: Dataset,
    scope: Scope | None = None,
    run_id: str | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield ``(group, sequence, df)`` for the standardized tracks *scope* names.

    Reads ``tracks/index.csv``. ``None`` and an unset selector both cover every
    entry it names.

    ``run_id`` names one tracks variant. This walks *rows*, and without the
    shared variant selector an entry answering to two of them would be yielded
    twice under one name.
    """
    df_idx = _filter_index(select_variant_rows(read_tracks_index(ds), run_id), scope)

    for _, row in df_idx.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        p = ds.resolve_path(row["abs_path"])
        if not p.exists():
            raise FileNotFoundError(f"Stale tracks index: ({g},{s}) -> {p}")
        yield g, s, pd.read_parquet(p)
