"""Ordering a recipe, and reading its shape.

The walk ``plan_pipeline`` makes is topological, and it is the whole mechanism by
which every step's identity resolves: step A's is a function of its params,
step B's of its params plus A's identity, step C's of B's. Nothing waits on
execution, so nothing here reads a dataset.

**Edges are derived, never declared.** A recipe carries its cross-step references
at the sites they substitute, so :func:`edges` reads them back out rather than
being a second list that could disagree with the bodies. The one exception is
``after``, which is ordering-only and corresponds to nothing in any payload --
it appears here as an edge with no substitution site.

Order among ready steps is **declaration order**, not sorted. It is equally
deterministic, and it means a plan reads in the order its author wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model import Recipe, Step, references_of

__all__ = [
    "Edge",
    "RecipeCycle",
    "ancestors_of",
    "children_of",
    "descendants_of",
    "edges",
    "parents_of",
    "topological_order",
]


class RecipeCycle(ValueError):
    """The graph has a cycle, so no step in it can be ordered."""


type EdgeSite = Literal["inputs", "tracks", "params", "after"]


@dataclass(frozen=True, slots=True)
class Edge:
    """One reference from *consumer* back to *producer*, and where it sits.

    A read-only view for anything that draws or explains the graph. ``where`` is
    the exact site -- ``inputs[1]``, ``params.templates`` -- because that is what
    decides how the reference is substituted, and a canvas showing "these two are
    connected" without saying how has hidden the only interesting part.
    """

    producer: str
    consumer: str
    site: EdgeSite
    where: str


def _site_of(where: str) -> EdgeSite:
    """Which kind of site a reference location names."""
    if where.startswith("inputs"):
        return "inputs"
    if where.startswith("params."):
        return "params"
    if where.startswith("after"):
        return "after"
    return "tracks"


def edges(recipe: Recipe) -> tuple[Edge, ...]:
    """Every reference in *recipe*, in declaration order."""
    found: list[Edge] = []
    for step in recipe.steps:
        for producer, where in references_of(step):
            found.append(
                Edge(
                    producer=producer,
                    consumer=step.id,
                    site=_site_of(where),
                    where=where,
                )
            )
    return tuple(found)


def parents_of(recipe: Recipe, step_id: str) -> tuple[str, ...]:
    """The steps *step_id* references, deduplicated, in declaration order.

    ``after`` is included: an ordering-only edge is still a parent, and a step
    held on its parents is held on that one too.
    """
    seen: list[str] = []
    for producer, _ in references_of(recipe.step(step_id)):
        if producer not in seen:
            seen.append(producer)
    return tuple(seen)


def children_of(recipe: Recipe, step_id: str) -> tuple[str, ...]:
    """The steps that reference *step_id*, in declaration order."""
    return tuple(
        step.id
        for step in recipe.steps
        if step_id in {producer for producer, _ in references_of(step)}
    )


def _walk(recipe: Recipe, start: str, forward: bool) -> frozenset[str]:
    """Everything reachable from *start*, inclusive, in one direction."""
    _ = recipe.step(start)
    seen: set[str] = set()
    queue = [start]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(
            children_of(recipe, current) if forward else parents_of(recipe, current)
        )
    return frozenset(seen)


def descendants_of(recipe: Recipe, step_id: str) -> frozenset[str]:
    """*step_id* and everything transitively downstream of it."""
    return _walk(recipe, step_id, forward=True)


def ancestors_of(recipe: Recipe, step_id: str) -> frozenset[str]:
    """*step_id* and everything transitively upstream of it."""
    return _walk(recipe, step_id, forward=False)


def topological_order(recipe: Recipe) -> tuple[Step, ...]:
    """*recipe*'s steps, every step after every step it references.

    Kahn's algorithm over declaration order, so the result is deterministic and
    reads as its author wrote it wherever the graph does not force otherwise.

    Raises:
        RecipeCycle: Naming the steps that could not be ordered, which is the set
            the cycle runs through.
    """
    remaining = {step.id: set(parents_of(recipe, step.id)) for step in recipe.steps}
    ordered: list[Step] = []
    while remaining:
        ready = [
            step.id
            for step in recipe.steps
            if step.id in remaining and not remaining[step.id]
        ]
        if not ready:
            raise RecipeCycle(
                "this recipe has a cycle; these steps cannot be ordered: "
                + ", ".join(sorted(remaining))
            )
        for step_id in ready:
            ordered.append(recipe.step(step_id))
            del remaining[step_id]
        done = set(ready)
        for pending in remaining.values():
            pending -= done
    return tuple(ordered)
