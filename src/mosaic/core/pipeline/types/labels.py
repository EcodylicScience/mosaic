from __future__ import annotations

from typing import Annotated, Generic, Literal

from pydantic import Field
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import (
    LoadSpec,
    NpzLoadSpec,
)
from mosaic.core.params import Declared
from mosaic.core.strict_model import StrictModel
from mosaic.core.pipeline.types.results import ResultColumn

K = TypeVar("K", bound=str, default=str)

_KIND_DESCRIPTION = "Which labels/<kind> subdirectory this dependency resolves to."

_GROUND_TRUTH_KIND_DESCRIPTION = (
    "Fixed to behavior. Ground-truth labels always resolve to labels/behavior/."
)

_SOURCE_DESCRIPTION = "Fixed tag naming this as a labels/ dependency."

_SOURCE_UNWIRED = (
    "no code path reads this field -- a GroundTruthLabelsSource is told apart "
    "from a ResultColumn by isinstance, not by this tag"
)

_LOAD_DESCRIPTION = "How to load the matched label file."

_LOAD_UNWIRED = (
    "no code path reads this field -- the resolved label file is read "
    "through load_labels_for_feature_frames, not through load_from_spec, so "
    "this load specification is never consulted"
)

_PATTERN_DESCRIPTION = "Glob pattern for the label file within labels/<kind>/."

_PATTERN_UNWIRED = (
    "no code path reads this field -- the label file is the abs_path "
    "recorded in labels/<kind>/index.csv, not a file matched by a pattern"
)


class LabelsSource(StrictModel, Generic[K]):
    """Base class for dataset label dependencies.

    Resolved to <dataset_root>/labels/<kind>/ by _build_labels_lookup and
    resolve_labels_variants in core/pipeline/run.py.
    """

    kind: Annotated[K, Declared(_KIND_DESCRIPTION)]


class GroundTruthLabelsSource(LabelsSource[Literal["behavior"]]):
    """Labels loaded from labels/<kind>/index.csv."""

    source: Annotated[
        Literal["labels"], Declared(_SOURCE_DESCRIPTION, unwired=_SOURCE_UNWIRED)
    ] = "labels"
    kind: Annotated[Literal["behavior"], Declared(_GROUND_TRUTH_KIND_DESCRIPTION)] = (
        "behavior"
    )
    load: Annotated[LoadSpec, Declared(_LOAD_DESCRIPTION, unwired=_LOAD_UNWIRED)] = (
        Field(default_factory=lambda: NpzLoadSpec(key="labels"))
    )
    pattern: Annotated[
        str | None, Declared(_PATTERN_DESCRIPTION, unwired=_PATTERN_UNWIRED)
    ] = None


LabelsSourceSpec = ResultColumn | GroundTruthLabelsSource
