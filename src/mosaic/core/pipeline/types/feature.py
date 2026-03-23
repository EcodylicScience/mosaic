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
    """Feature protocol -- 4 attributes, 4 methods."""

    name: str
    version: str
    parallelizable: bool
    scope_dependent: bool

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
