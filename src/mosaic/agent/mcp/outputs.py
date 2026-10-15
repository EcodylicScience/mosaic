"""The tool outputs built from the editor wires, which load pandas on import.

:mod:`~mosaic.agent.mcp.spec` names these models by
:class:`~mosaic.agent.mcp.spec.ModelRef`, which keeps them out of its import
budget.
"""

from __future__ import annotations

from mosaic.core.pipeline.graph import (
    Attachment,
    CatalogIndexEntry,
    InputCandidate,
    StillNeeded,
)
from mosaic.core.strict_model import StrictModel

from .spec import SuggestMode

__all__ = ["StepSuggestions", "StepTypePage"]


class StepTypePage(StrictModel):
    """One page of the step types that a search matches, most relevant first.

    A search without words lists every step type, ordered by name.

    Attributes:
        schema_version: The recipe format this installation reads.
        mosaic_version: The installed ``mosaic-behavior`` version.
        entries: The page, each step type without its params schema.
        total: How many step types the search matches.
        next_cursor: The cursor of the next page, ``None`` on the last.
    """

    schema_version: int
    mosaic_version: str
    entries: tuple[CatalogIndexEntry, ...]
    total: int
    next_cursor: str | None


class StepSuggestions(StrictModel):
    """What may be added to a recipe at one step, the allowed choices only.

    Attributes:
        mode: The question answered.
        step: The step asked about, ``None`` for the top of the recipe.
        steps: Under ``attach`` and ``insert``, each allowed step type by name,
            with the step to add.
        inputs: Under ``input``, each allowed input by step id or ``tracks``.
        children: Under ``insert``, the steps that would read the inserted step.
        still_needs: Under ``input``, the step's requirements that its inputs do
            not guarantee.
        refused: How many choices were refused.
    """

    mode: SuggestMode
    step: str | None
    steps: dict[str, Attachment]
    inputs: dict[str, InputCandidate]
    children: tuple[str, ...]
    still_needs: tuple[StillNeeded, ...]
    refused: int
