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
    StepTypeDetail,
    StillNeeded,
)
from mosaic.core.strict_model import StrictModel

from .spec import StepTypeDetailLevel, SuggestMode

__all__ = ["StepSuggestions", "StepTypes"]


class StepTypes(StrictModel):
    """Step types found by a search, or described by name.

    A search returns one page of index rows, most relevant first, and a search
    without words lists every step type by name. Names return one row or one
    detail per known name, in the order asked.

    Attributes:
        schema_version: The recipe format this installation reads.
        mosaic_version: The installed ``mosaic-behavior`` version.
        detail: How much each step type carries: ``summary`` its index row,
            ``fields`` its params table too, ``schema`` its params JSON Schema
            too.
        entries: Under ``summary``, the index rows.
        details: Under ``fields`` and ``schema``, the step types in detail.
        unknown: The names asked for that no step type has.
        total: How many step types the search matches, or the names described.
        next_cursor: The cursor of the next page, ``None`` on the last and for
            names.
    """

    schema_version: int
    mosaic_version: str
    detail: StepTypeDetailLevel
    entries: tuple[CatalogIndexEntry, ...] = ()
    details: tuple[StepTypeDetail, ...] = ()
    unknown: tuple[str, ...] = ()
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
