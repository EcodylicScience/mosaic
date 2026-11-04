from __future__ import annotations

from typing import Annotated, Generic, Literal, Self, TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import (
    LoadSpec,
    NpzLoadSpec,
)
from mosaic.core.helpers import UNLABELED
from mosaic.core.params import Declared, NestedParams
from mosaic.core.pipeline.types.results import ResultColumn

K = TypeVar("K", bound=str, default=str)

_KIND_DESCRIPTION = "Which labels/<kind> subdirectory this dependency resolves to."

_GROUND_TRUTH_KIND_DESCRIPTION = (
    "Which labels/<kind> subdirectory the ground-truth labels resolve to. "
    "Defaults to behavior; another kind resolves the same way, through "
    "labels/<kind>/index.csv."
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


_REMAP_DESCRIPTION = (
    "Maps a class id of the label file to the id every reader sees, as {file_id: id}."
)

_MASK_DESCRIPTION = (
    "Class ids whose frames are unlabeled, so no step trains on or scores them."
)

LabelId: TypeAlias = Annotated[int, Field(ge=0)]
"""A class id of a label file, 0 or more."""

LabelRemap: TypeAlias = Annotated[
    dict[LabelId, LabelId] | None, Declared(_REMAP_DESCRIPTION)
]
"""A labels source's ``remap``: a class id of the file to the id readers see."""

LabelMask: TypeAlias = Annotated[list[LabelId] | None, Declared(_MASK_DESCRIPTION)]
"""A labels source's ``mask``: the class ids whose frames become unlabeled."""


class LabelsSource(NestedParams, Generic[K]):
    """Base class for dataset label dependencies.

    Resolved to <dataset_root>/labels/<kind>/ by _build_labels_lookup and
    resolve_labels_variants in core/pipeline/run.py.
    """

    kind: Annotated[K, Declared(_KIND_DESCRIPTION)]


class GroundTruthLabelsSource(LabelsSource[str]):
    """Labels loaded from labels/<kind>/index.csv."""

    source: Annotated[
        Literal["labels"], Declared(_SOURCE_DESCRIPTION, unwired=_SOURCE_UNWIRED)
    ] = "labels"
    # min_length: the resolver skips a source with an empty kind, so "" would
    # resolve to no labels at all and fail every entry for want of a file.
    kind: Annotated[
        str, Field(min_length=1), Declared(_GROUND_TRUTH_KIND_DESCRIPTION)
    ] = "behavior"
    load: Annotated[LoadSpec, Declared(_LOAD_DESCRIPTION, unwired=_LOAD_UNWIRED)] = (
        Field(default_factory=lambda: NpzLoadSpec(key="labels"))
    )
    pattern: Annotated[
        str | None, Declared(_PATTERN_DESCRIPTION, unwired=_PATTERN_UNWIRED)
    ] = None
    remap: LabelRemap = None
    mask: LabelMask = None

    @field_validator("remap")
    @classmethod
    def _canonical_remap(cls, remap: dict[int, int] | None) -> dict[int, int] | None:
        """Drop the pairs that map an id to itself and sort the keys; empty is ``None``."""
        kept = {k: v for k, v in sorted((remap or {}).items()) if k != v}
        return kept or None

    @field_validator("mask")
    @classmethod
    def _canonical_mask(cls, mask: list[int] | None) -> list[int] | None:
        """Sort the ids without repeats; empty is ``None``."""
        return sorted(set(mask or ())) or None

    @model_validator(mode="after")
    def _one_control_per_id(self) -> Self:
        """Refuse an id both masked and remapped.

        With the two field validators, every spelling of one control is one
        identity.
        """
        both = sorted(set(self.remap or {}) & set(self.mask or ()))
        if both:
            msg = (
                f"class ids {both} are both masked and remapped. A masked frame "
                "is unlabeled, so give each id to one of mask and remap"
            )
            raise ValueError(msg)
        return self

    def identity_dump(self) -> dict[str, object]:
        """The ``model_dump()``, without ``remap`` and ``mask`` while they are unset.

        A source without controls dumps as it did before they existed, so the
        run of every labels-consuming feature keeps its identifier.
        """
        dumped = self.model_dump()
        for name in ("remap", "mask"):
            if dumped[name] is None:
                del dumped[name]
        return dumped

    def relabel(
        self, labels: npt.ArrayLike, *, unlabeled: int = UNLABELED
    ) -> npt.NDArray[np.int64]:
        """*labels* after ``mask`` and ``remap``, as a new int64 array.

        Each masked id becomes *unlabeled*, then every remapped id takes its
        target at once, so ``{1: 2, 2: 1}`` swaps two classes. A value equal to
        *unlabeled* stays, and a row the file's background class fills is
        remapped like any other.

        Args:
            labels: Class ids, one per row.
            unlabeled: The value a masked id takes.
        """
        source = np.asarray(labels, dtype=np.int64)
        relabeled = source.copy()
        if self.remap:
            for old, new in self.remap.items():
                relabeled[source == old] = new
        relabeled[self.masked(source)] = unlabeled
        return relabeled

    def masked(self, ids: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Whether each of *ids* is a masked class id."""
        values = np.asarray(ids, dtype=np.int64)
        if not self.mask:
            return np.zeros(values.shape, dtype=np.bool_)
        return np.isin(values, np.asarray(self.mask, dtype=np.int64))


LabelsSourceSpec = ResultColumn | GroundTruthLabelsSource
