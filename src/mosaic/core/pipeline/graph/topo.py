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

**A label conversion is ordered after every step that writes tracks.** A converter
of events in seconds places its labels on the tracks that exist when it runs, and no
reference says so, so :func:`parents_of`, and with it every ordering and every walk
up or down the graph, adds those parents. Every label conversion is ordered so,
whatever its converter, because labels follow the tracks they annotate and a status
read walks the graph without reading a registry. :func:`edges` lists references only.

Order among ready steps is **declaration order**, not sorted. It is equally
deterministic, and it means a plan reads in the order its author wrote it.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Final, Literal

from .model import OpStepSpec, Recipe, Step, references_of

__all__ = [
    "Edge",
    "RecipeCycle",
    "ancestors_of",
    "children_of",
    "descendants_of",
    "edges",
    "parents_of",
    "topological_order",
    "tracks_before_labels",
]


_CONVERT_LABELS: Final = "convert-labels"
"""The op kind that converts raw labels into a labels variant."""

TRACK_WRITING_OPS: Final = frozenset(
    {
        "assign-arenas",
        "assign-identities",
        "convert-tracks",
        "infer-localizer",
        "infer-points",
        "infer-pose",
        "litpose",
        "resample-tracks",
        "sleap",
        "trex",
        "ultralytics",
    }
)
"""The op kinds that write a tracks variant, which a label conversion runs after.

Listed here so that ordering a recipe reads no registry. It names exactly the
registered ops whose ``writes_tracks`` holds
(``tests/meta/test_track_writing_ops_listed.py``), and a new op that writes tracks
is added to it.
"""


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
    for producer in tracks_before_labels(recipe, step_id):
        if producer not in seen:
            seen.append(producer)
    return tuple(seen)


def tracks_before_labels(recipe: Recipe, step_id: str) -> tuple[str, ...]:
    """The op steps that write tracks, when *step_id* converts labels, else none.

    A writer that reads the labels step, through any reference, is left out: the
    labels are then converted before it and cannot wait on it.
    """
    step = recipe.step(step_id)
    if not isinstance(step, OpStepSpec) or step.kind != _CONVERT_LABELS:
        return ()
    below = _walk_references(recipe, step_id)
    return tuple(
        other.id
        for other in recipe.steps
        if isinstance(other, OpStepSpec)
        and other.kind in TRACK_WRITING_OPS
        and other.id != step_id
        and other.id not in below
    )


def _walk_references(recipe: Recipe, start: str) -> frozenset[str]:
    """The steps that reference *start*, directly or through others, by reference alone."""
    found: set[str] = set()
    queue = [start]
    while queue:
        current = queue.pop()
        for step in recipe.steps:
            if step.id not in found and any(
                producer == current for producer, _ in references_of(step)
            ):
                found.add(step.id)
                queue.append(step.id)
    return frozenset(found)


def children_of(recipe: Recipe, step_id: str) -> tuple[str, ...]:
    """The steps that reference *step_id*, in declaration order."""
    return tuple(
        step.id for step in recipe.steps if step_id in parents_of(recipe, step.id)
    )


def _walk(
    recipe: Recipe, start: str, forward: bool, cut: Collection[str]
) -> frozenset[str]:
    """Everything reachable from *start*, inclusive, in one direction.

    A step in *cut* other than *start* is neither entered nor walked past.
    """
    _ = recipe.step(start)
    seen: set[str] = set()
    queue = [start]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(
            step
            for step in (
                children_of(recipe, current) if forward else parents_of(recipe, current)
            )
            if step not in cut
        )
    return frozenset(seen)


def descendants_of(
    recipe: Recipe, step_id: str, *, cut: Collection[str] = ()
) -> frozenset[str]:
    """*step_id* and everything transitively downstream of it.

    A step in *cut*, such as one a request binds to a run, reads nothing above
    it, so the walk does not enter it or pass through it.
    """
    return _walk(recipe, step_id, forward=True, cut=cut)


def ancestors_of(
    recipe: Recipe, step_id: str, *, cut: Collection[str] = ()
) -> frozenset[str]:
    """*step_id* and everything transitively upstream of it.

    A step in *cut*, such as one a request binds to a run, stands for nothing
    above it, so the walk does not enter it or pass through it.
    """
    return _walk(recipe, step_id, forward=False, cut=cut)


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
