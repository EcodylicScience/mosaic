from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol, final

import pandas as pd

from mosaic.core.pipeline.types.inputs import InputsLike
from mosaic.core.pipeline.types.params import Params

DependencyLookup = dict[tuple[str, str], Path]


@final
class InputStream:
    """Factory for fit() input iterators, with entry count metadata.

    Wraps a callable that produces ``(entry_key, DataFrame)`` iterators.
    Each call creates a fresh iterator over the manifest entries.
    ``n_entries`` exposes the total number of entries so features can
    make exact allocation decisions (e.g. train/test split counts)
    without an extra data pass.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[tuple[str, pd.DataFrame]]],
        n_entries: int,
    ) -> None:
        self._factory = factory
        self._n_entries = n_entries

    @property
    def n_entries(self) -> int:
        return self._n_entries

    def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        return self._factory()


class Feature(Protocol):
    """Feature protocol -- 5 attributes, 4 methods."""

    name: str
    version: str
    parallelizable: bool
    scope_dependent: bool
    consumed_roots: tuple[str, ...]
    """The **source roots this feature opens directly**, outside its inputs.

    The obvious reading is wrong, so read this one. It is not "the roots my data
    came from": a feature consuming ``"tracks"`` declares ``()``, because
    ``tracks/`` is a *derived* root and the manifest already hands it the tables.
    It is the roots the feature reaches past its inputs to read -- today, the two
    that open video through ``resolve_media`` / ``MultiVideoReader``.

    **Why a tracks-consuming feature declares nothing.** If it carried
    ``tracks_raw``'s composition, a change under ``tracks_raw`` would move its
    identifier without the tracks parquet having changed a byte -- a false
    invalidation, and precisely the "couple every per-sequence feature to the
    whole dataset" hazard the media-storage decision note warns about.

    **Not transitive through the variant identity, which this used to claim.** A
    change under ``tracks_raw`` re-produces the tracks table but does *not* move the
    tracks variant identity: that identity is params-only, and ``tracks_identity``
    says so -- "this names the recipe, not the input". So the ``_tracks`` hash term
    is byte-identical across a re-conversion of changed bytes, and nothing about the
    identifier notices.

    What closes it is a recorded comparison rather than an identifier: the tracks row
    already carries the source composition its table was converted from, and a
    feature row carries that value forward in ``consumed_tracks_composition``, so the
    per-entry cache check compares the two. Declaring ``("tracks_raw",)`` would have
    closed it by false invalidation instead, which is the wrong trade -- an honest
    miss beats a confident wrong value, but a *spurious* miss is neither.
    """

    @property
    def inputs(self) -> InputsLike: ...

    @property
    def params(self) -> Params: ...

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool: ...

    def fit(self, inputs: InputStream) -> None: ...

    def save_state(self, run_root: Path) -> None: ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
