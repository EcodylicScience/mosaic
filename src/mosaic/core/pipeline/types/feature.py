from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import pandas as pd

from mosaic.core.pipeline.types.inputs import InputsLike
from mosaic.core.pipeline.types.params import Params


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
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool: ...

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None: ...

    def save_state(self, run_root: Path) -> None: ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
