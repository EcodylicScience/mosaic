"""The types of a labels source's ``remap`` and ``mask``, and the controls found on runs.

A leaf module, so that the recipe model can type a labels reference by them
without loading the pipeline type library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final, TypeAlias

from pydantic import Field

from mosaic.core.params import Declared

_REMAP_DESCRIPTION = (
    "Maps a class id of the label file to the id every reader sees, as {file_id: id}."
)

_MASK_DESCRIPTION = (
    "Class ids whose frames are unlabeled, so no step trains on or scores them."
)

LABEL_CONTROL_FIELDS: Final = ("remap", "mask")
"""The keys a labels source takes beside its kind: ``remap`` and ``mask``."""

LabelId: TypeAlias = Annotated[int, Field(ge=0)]
"""A class id of a label file, 0 or more."""

LabelRemap: TypeAlias = Annotated[
    dict[LabelId, LabelId] | None, Declared(_REMAP_DESCRIPTION)
]
"""A labels source's ``remap``: a class id of the file to the id readers see."""

LabelMask: TypeAlias = Annotated[list[LabelId] | None, Declared(_MASK_DESCRIPTION)]
"""A labels source's ``mask``: the class ids whose frames become unlabeled."""


@dataclass(frozen=True, slots=True)
class LabelControls:
    """The ``remap`` pairs and ``mask`` ids found on a set of runs.

    True when either is non-empty.
    """

    remap: frozenset[tuple[int, int]] = frozenset()
    mask: frozenset[int] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.remap or self.mask)
