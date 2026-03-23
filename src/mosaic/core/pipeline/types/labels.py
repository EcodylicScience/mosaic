from __future__ import annotations

from typing import Generic, Literal

from pydantic import Field
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import LoadSpec, NpzLoadSpec, StrictModel
from mosaic.core.pipeline.types.results import ResultColumn

K = TypeVar("K", bound=str, default=str)


class LabelsSource(StrictModel, Generic[K]):
    """Base class for dataset label dependencies.

    Resolved by _resolve_dependency_paths (Task 9) to
    <dataset_root>/labels/<kind>/.
    """

    kind: K


class GroundTruthLabelsSource(LabelsSource[Literal["behavior"]]):
    """Labels loaded from labels/<kind>/index.csv."""

    source: Literal["labels"] = "labels"
    kind: Literal["behavior"] = "behavior"
    load: LoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))
    pattern: str | None = None


LabelsSourceSpec = ResultColumn | GroundTruthLabelsSource
