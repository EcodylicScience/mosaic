"""The JSON a pipeline editor reads, as pydantic models.

One model per answer an editor asks for: the step types on offer
(:class:`EditorCatalog`), what is wrong with a recipe (:class:`ValidationReport`),
what may follow a step (:class:`AttachReport`) and what a recipe would do to one
dataset (:class:`PlanWire`). The CLI's ``--json`` output and mosaic-api's
responses are these models dumped, so the two cannot disagree about a shape.

Models only. The functions that fill them live beside what they read --
:func:`~mosaic.core.pipeline.graph.resolve.editor_catalog`,
:func:`~mosaic.core.pipeline.graph.validate.validation_report`,
:func:`~mosaic.core.pipeline.graph.validate.attach_candidates` and
:func:`~mosaic.core.pipeline.graph.serialize.plan_wire` -- and each of those
modules imports this one. This module imports none of them, which keeps the
import graph acyclic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import Field

from mosaic.core.json_value import JsonValue

from ..inventory.model import ArtifactStatus
from ..ops import ScopeTakes
from .compatibility import Declaration
from .model import GraphModel

__all__ = [
    "MISSING_SAMPLE_WIRE",
    "AttachReport",
    "Attachment",
    "CatalogEntry",
    "CoverageShortWire",
    "CoverageWire",
    "DepsIncompleteWire",
    "EditorCatalog",
    "HeldOnParentsWire",
    "IdentityUnresolvedWire",
    "PlanWire",
    "PlannedStepWire",
    "Problem",
    "ReasonWire",
    "ReferenceKind",
    "StalledWire",
    "ValidationReport",
    "WaitingOnResourceWire",
    "render_location",
]

MISSING_SAMPLE_WIRE: Final = 20
"""How many missing entries a plan sends per step.

A full entry list for a 2,000-sequence dataset under a ten-step graph is
megabytes on every preview. The sample names enough entries to act on, and
``has_more`` says whether it is the whole list.
"""


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a recipe, at the place it is wrong.

    Attributes:
        step: The step id, or ``""`` for a fault of the graph as a whole.
        where: Which part of the step -- ``feature``, ``kind``, ``inputs``,
            ``tracks``, ``params.templates``, ``steps``.
        message: What is wrong, phrased so it reads on its own.
    """

    step: str
    where: str
    message: str

    def __str__(self) -> str:
        at = f"step {self.step!r} " if self.step else ""
        return f"{at}{self.where}: {self.message}"


def render_location(location: Sequence[int | str]) -> str:
    """Render a pydantic error location as a path an editor can point at.

    Names are joined with ``.`` and indices render as ``[i]``, so
    ``("params", "keypoint_names", 0)`` reads ``params.keypoint_names[0]``.
    """
    rendered = ""
    for part in location:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = part
    return rendered


type ReferenceKind = Literal["result", "identifier"]
"""What a params field accepts in place of a value: a run reference, or a bare id.

``result`` takes ``{"step": ...}`` substituted as ``{feature, run_id}``;
``identifier`` takes one substituted as the upstream's bare identifier, which is
how a model is named. See ``params_reference_site``.
"""


class CatalogEntry(GraphModel):
    """One step type an editor may offer.

    Attributes:
        name: What a recipe writes to run it: a feature slug, an op kind, or
            ``tracks``.
        type: Which of the three it is.
        version: The version its producer declares, ``""`` for ``tracks``.
        category: Its feature or op category, for grouping.
        summary: The first line of its class docstring.
        declaration: What it produces and consumes, as validation reads it.
        params_schema: Its params model as JSON Schema, ``None`` for ``tracks``.
        reference_fields: The top-level params fields that accept a step
            reference, and how each substitutes. A JSON Schema cannot say this.
    """

    name: str
    type: Literal["feature", "op", "tracks"]
    version: str
    category: str
    summary: str
    declaration: Declaration
    params_schema: dict[str, JsonValue] | None
    reference_fields: dict[str, ReferenceKind]


class EditorCatalog(GraphModel):
    """Every step type a recipe may run on this installation.

    Attributes:
        schema_version: The recipe format this build reads.
        mosaic_version: The installed ``mosaic-behavior`` version, ``""`` when
            unknown.
        entries: One per step type, sorted by name.
    """

    schema_version: int
    mosaic_version: str
    entries: tuple[CatalogEntry, ...]


# ``Declaration.scope_takes`` is annotated with ``ScopeTakes``, which
# ``compatibility`` imports for type checking only so a declaration's readers do
# not load the op registry. This module loads it anyway, so pydantic resolves the
# name here.
_ = CatalogEntry.model_rebuild(_types_namespace={"ScopeTakes": ScopeTakes})


class ValidationReport(GraphModel):
    """Everything wrong with one recipe document.

    Attributes:
        valid: Whether it has no problems.
        digest: Its ``recipe_digest`` when it parsed, else ``None``.
        problems: Every problem, in topological order after any structural ones.
    """

    valid: bool
    digest: str | None
    problems: tuple[Problem, ...]


class Attachment(GraphModel):
    """Whether one step type may be added below a step, and the step to add.

    Attributes:
        allowed: Whether it may.
        reason: Why not, ``""`` when allowed.
        template: The new step complete except for its ``id``, wired to the step
            above it. ``None`` when refused.
    """

    allowed: bool
    reason: str
    template: dict[str, JsonValue] | None


class AttachReport(GraphModel):
    """What may be added below one step, or at the top of a recipe.

    Attributes:
        after: The step asked about, ``None`` for the top level.
        candidates: One attachment per step type, keyed by name.
    """

    after: str | None
    candidates: dict[str, Attachment]


class CoverageWire(GraphModel):
    """Which of its planned entries a step already holds, summarized.

    Attributes:
        covers_all: A single artifact answers for every entry, as a global fit or
            a trained model does.
        covered: How many planned entries it holds.
        target: How many it was planned over.
        missing_sample: Up to :data:`MISSING_SAMPLE_WIRE` missing entries, sorted.
        has_more: Whether more are missing than the sample lists.
    """

    covers_all: bool
    covered: int
    target: int
    missing_sample: tuple[tuple[str, str], ...]
    has_more: bool


class DepsIncompleteWire(GraphModel):
    """Something this step reads does not hold everything this step wants."""

    kind: Literal["deps_incomplete"] = "deps_incomplete"
    blocking: tuple[str, ...]


class CoverageShortWire(GraphModel):
    """This step holds some of its scope and not the rest."""

    kind: Literal["coverage_short"] = "coverage_short"
    covered: int
    target: int
    missing_sample: tuple[tuple[str, str], ...]


class HeldOnParentsWire(GraphModel):
    """Its parents are running, or waiting to."""

    kind: Literal["held_on_parents"] = "held_on_parents"
    parents: tuple[str, ...]


class WaitingOnResourceWire(GraphModel):
    """Its lane's resource class is at capacity."""

    kind: Literal["waiting_on_resource"] = "waiting_on_resource"
    resource_class: str
    in_use: int
    capacity: int


class StalledWire(GraphModel):
    """Its last attempt succeeded and produced nothing new."""

    kind: Literal["stalled"] = "stalled"
    covered: int
    target: int
    missing_sample: tuple[tuple[str, str], ...]


class IdentityUnresolvedWire(GraphModel):
    """This step, or one above it, cannot say what it will be called yet."""

    kind: Literal["identity_unresolved"] = "identity_unresolved"
    step: str
    because: str


type ReasonWire = Annotated[
    DepsIncompleteWire
    | CoverageShortWire
    | HeldOnParentsWire
    | WaitingOnResourceWire
    | StalledWire
    | IdentityUnresolvedWire,
    Field(discriminator="kind"),
]
"""Why a step is not simply running, tagged by ``kind``. One per ``Reason``."""


class PlannedStepWire(GraphModel):
    """One step of a plan, as an editor renders it.

    Attributes:
        step: The step's id in the recipe.
        type: Whether it runs a feature or an op.
        runs: The feature slug or op kind.
        run_id: What its run will be called, ``None`` when that cannot be said
            yet.
        status: What its coverage and consistency add up to.
        finished: Whether its artifact records having finished.
        coverage: Which planned entries it already holds.
        asked: How many entries it would be asked to compute.
        drift: How many entries' recorded sources have moved under the run.
        lane: Which queue its work is offered to.
        parents: The steps it references, ``after`` included.
        storage_name: Where a feature step's outputs live.
        tracks_variant: What an op step's tables will be named by.
        model_run_id: What names a training step's model.
        reason: Why it is not simply running, or ``None``.
    """

    step: str
    type: Literal["feature", "op"]
    runs: str
    run_id: str | None
    status: ArtifactStatus
    finished: bool
    coverage: CoverageWire
    asked: int
    drift: int
    lane: str
    parents: tuple[str, ...]
    storage_name: str
    tracks_variant: str
    model_run_id: str
    reason: ReasonWire | None


class PlanWire(GraphModel):
    """What a recipe would do to one dataset, bounded in size.

    Attributes:
        recipe_digest: What names the recipe.
        scope_size: How many entries the graph is planned over.
        complete: Whether every step is already computed over its scope.
        steps: Every step, in topological order.
    """

    recipe_digest: str
    scope_size: int
    complete: bool
    steps: tuple[PlannedStepWire, ...]
