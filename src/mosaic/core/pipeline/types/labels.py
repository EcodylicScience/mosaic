from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Annotated, Final, Generic, Self, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import Field, ValidationError, field_validator, model_validator

from mosaic.core.pipeline._loaders import NpzLoadSpec
from mosaic.core.helpers import UNLABELED
from mosaic.core.narrowing import is_str_mapping
from mosaic.core.pipeline.label_controls import (
    LABEL_CONTROL_FIELDS,
    LabelMask,
    LabelRemap,
)
from mosaic.core.params import Declared, NestedParams
from mosaic.core.pipeline.types.results import ResultColumn

K = TypeVar("K", bound=str, default=str)

_KIND_DESCRIPTION = "Which labels/<kind> subdirectory this dependency resolves to."

_GROUND_TRUTH_KIND_DESCRIPTION = (
    "Which labels/<kind> subdirectory the ground-truth labels resolve to. "
    "Defaults to behavior; another kind resolves the same way, through "
    "labels/<kind>/index.csv."
)

# The values of the withdrawn ``source``, ``load`` and ``pattern`` fields, which
# every labels source held. ``identity_dump`` keeps writing them so that runs
# computed before the fields were withdrawn keep their identifiers, and a
# recorded params file spelling one of them still validates.
_RETIRED_IDENTITY: Final[Mapping[str, object]] = {
    "source": "labels",
    "load": {"kind": "npz", "key": "labels", "transpose": False},
    "pattern": None,
}


def _is_retired_value(name: str, value: object) -> bool:
    """Whether *value* spells the old value of the withdrawn field *name*.

    A ``load`` counts in any spelling ``NpzLoadSpec`` reads as the old one,
    defaults omitted or a model instance included.
    """
    if name != "load":
        return value == _RETIRED_IDENTITY[name]
    try:
        spec = NpzLoadSpec.model_validate(value)
    except ValidationError:
        return False
    return spec.model_dump() == _RETIRED_IDENTITY["load"]


class LabelsSource(NestedParams, Generic[K]):
    """Base class for dataset label dependencies.

    Resolved to <dataset_root>/labels/<kind>/ by _build_labels_lookup and
    resolve_labels_variants in core/pipeline/run.py.
    """

    kind: Annotated[K, Declared(_KIND_DESCRIPTION)]


class GroundTruthLabelsSource(LabelsSource[str]):
    """Labels loaded from labels/<kind>/index.csv."""

    # min_length: the resolver skips a source with an empty kind, so "" would
    # resolve to no labels at all and fail every entry for want of a file.
    kind: Annotated[
        str, Field(min_length=1), Declared(_GROUND_TRUTH_KIND_DESCRIPTION)
    ] = "behavior"
    remap: LabelRemap = None
    mask: LabelMask = None

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_keys(cls, data: object) -> object:
        """Drop ``source``, ``load`` and ``pattern`` at their old values.

        Every params file recorded before the fields were withdrawn spells them,
        and ``mosaic reconcile`` rebuilds a run from that file. Any other value
        asked for something no labels source does, and is refused.
        """
        if not is_str_mapping(data):
            return data
        kept = dict(data)
        for name in _RETIRED_IDENTITY:
            if name in kept and not _is_retired_value(name, kept.pop(name)):
                msg = (
                    f"{name!r} is withdrawn from a labels source and is accepted "
                    f"only at its old value {_RETIRED_IDENTITY[name]!r}. The labels "
                    "of a kind are the files labels/<kind>/index.csv lists"
                )
                raise ValueError(msg)
        return kept

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

        The dump also holds ``source``, ``load`` and ``pattern`` at the values
        every source held before those fields were withdrawn. A source without
        controls dumps as it did before they existed, so the run of every
        labels-consuming feature keeps its identifier.
        """
        dumped: dict[str, object] = self.model_dump()
        dumped.update(copy.deepcopy(dict(_RETIRED_IDENTITY)))
        for name in LABEL_CONTROL_FIELDS:
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
