"""The index reads that every step of one plan shares.

A plan resolves each feature step's identity and checks the tracks rows of each
step that follows ids. Both read indexes whose content does not depend on the
step: each root's ``sequences.csv``, and the tracks index. Read once for each
step, a recipe of many steps reads each file as many times. A plan builds one
:class:`PlanReads` and hands it to every step, so each file is read at most once
for all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .manifest import tracks_variants_for
from .sequence_index import recorded_compositions
from .tracks_index import AmbiguousTracksVariant, read_tracks_index

if TYPE_CHECKING:
    import pandas as pd

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry

__all__ = ["PlanReads"]


class PlanReads:
    """Index reads made on first use and kept for the rest of one plan.

    A plan writes nothing, so what it reads once holds for every step. Each
    answer is the one its function gives, read when a step first asks.

    Args:
        ds: The dataset whose indexes are read.
    """

    def __init__(self, ds: Dataset) -> None:
        self._ds: Dataset = ds
        self._compositions: dict[Entry, dict[str, str]] | None = None
        self._tracks_index: pd.DataFrame | None = None
        self._variants: tuple[str, ...] | AmbiguousTracksVariant | None = None

    def compositions(self) -> dict[Entry, dict[str, str]]:
        """Every entry's :func:`~.sequence_index.recorded_compositions`."""
        if self._compositions is None:
            self._compositions = recorded_compositions(self._ds)
        return self._compositions

    def tracks_index(self) -> pd.DataFrame:
        """The tracks index, as :func:`~.tracks_index.read_tracks_index` reads it."""
        if self._tracks_index is None:
            self._tracks_index = read_tracks_index(self._ds)
        return self._tracks_index

    def tracks_variants(self) -> tuple[str, ...]:
        """What :func:`~.manifest.tracks_variants_for` answers with no variant named.

        Raises:
            AmbiguousTracksVariant: An entry holds two variants. Raised again on
                each call, as the function raises it.
        """
        if self._variants is None:
            try:
                self._variants = tracks_variants_for(self._ds, None)
            except AmbiguousTracksVariant as ambiguous:
                self._variants = ambiguous
        if isinstance(self._variants, AmbiguousTracksVariant):
            raise self._variants
        return self._variants
