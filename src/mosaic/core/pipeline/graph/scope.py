"""The entry set a graph is planned against, before any of it has run.

The one thing a dependent walk could not previously supply. Prediction routed
through ``build_manifest(on_missing_run="empty")``, which reports an *empty*
scope when the upstream index does not exist -- so a cold ``scope_dependent``
step hashed ``[]`` where execution would hash the real entries, and predicted an
identifier it then did not produce. That value exists for display, and acting on
it is a documented pitfall.

**It turns out not to need a propagation mechanism.** The scope term is a sorted
list of ``(group, sequence)`` plus, for a feature that declares source roots, a
per-entry composition digest. Every ``scope_dependent`` feature today declares
none, so the term reduces to the entry names alone; and where roots *are*
declared, the compositions are read from the source roots, which exist before
the pipeline runs. Nothing in the term comes from an intermediate output. So the
entry set a step *will* see is knowable now, and supplying it makes the resolved
identity exact.

**Where the entries come from, in order.** An explicit narrowing if the caller
gave one -- that is the submission saying what it wants. A selector naming
groups or sequences is enumerated against the universe below rather than taken
as an entry list, because only the index knows which entries those names cover.
Otherwise the dataset's tracks universe, which is what every feature reads. And
when that is empty *because the graph itself produces the tracks*, the media
universe, since a tracker turns videos into tracks one entry at a time and those
videos are on disk before anything is planned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.core.scope import Scope

from .._utils import ResolvedScope
from ..inventory.scan import narrow_target, reportable_universe
from .model import Recipe
from .topo import topological_order

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.dataset import Dataset


__all__ = ["graph_writes_tracks", "intended_scope", "media_universe"]


def media_universe(ds: Dataset) -> frozenset[tuple[str, str]]:
    """Every ``(group, sequence)`` this dataset has media for.

    The cameras of one recording collapse back to one entry here, because a
    tracks table is addressed by ``(group, sequence)`` and a graph is planned in
    those terms. The camera axis matters to what a *frame run* covers, and this
    is not that.

    **A dataset with no media index has no media, which is an answer.** The
    accessor raises for it, correctly, because a caller about to *read* media
    needs to be told to index first. This caller is deciding what a graph would
    cover, and a graph over a dataset with neither tracks nor media covers
    nothing -- so refusing here would take a whole plan down over the honest
    empty case, including the plan that would have said which steps could not
    run.
    """
    try:
        scope = ds.resolve_media_scope(None)
    except FileNotFoundError:
        return frozenset()
    return frozenset((entry.group, entry.sequence) for entry in scope)


def graph_writes_tracks(recipe: Recipe, produces_tracks: Iterable[str]) -> bool:
    """Does *recipe* contain a step that writes a tracks variant?

    Asked with the set of kinds that bridge into ``tracks/`` rather than looking
    them up, so this stays free of the op registry -- a planner already has the
    declarations and hands the answer in.
    """
    writers = set(produces_tracks)
    return any(
        getattr(step, "kind", "") in writers for step in topological_order(recipe)
    )


def intended_scope(
    ds: Dataset,
    recipe: Recipe,
    scope: Scope | None = None,
    *,
    tracks_run_id: str | None = None,
    produces_tracks: Iterable[str] = (),
) -> ResolvedScope:
    """The entries this graph is planned over, beside the selector they came from.

    Args:
        ds: The dataset being planned against. Read only.
        recipe: The graph, consulted only for whether it produces its own tracks.
        scope: What the submission asked for. ``None`` and an unset selector
            both mean everything the dataset can process.
        tracks_run_id: Which tracks variant defines the universe. Pass the one
            the graph reads; measuring against the whole index would widen the
            scope past what any step will actually see.
        produces_tracks: The op kinds that bridge into ``tracks/``, so a graph
            that makes its own tracks can be recognised without this module
            importing the op registry.

    Returns:
        A :class:`~mosaic.core.pipeline._utils.ResolvedScope` whose ``entries``
        every ``scope_dependent`` step in this graph will hash, and whose
        ``selector`` is what was asked. An empty resolution is a real answer --
        a dataset with neither tracks nor media has nothing to plan over -- and
        the step that cares reports the shortfall.
    """
    selector = scope if scope is not None else Scope()
    if not selector.is_unset:
        # An explicit narrowing is the submission speaking, and it is not
        # widened back out by anything found on disk. A selector naming groups
        # or sequences enumerates against the tracks universe, the one every
        # feature step reads.
        universe = reportable_universe(ds, tracks_run_id)
        if not universe and graph_writes_tracks(recipe, produces_tracks):
            universe = media_universe(ds)
        named = selector.entry_pairs
        entries = (
            frozenset(named) if named is not None else narrow_target(universe, selector)
        )
        return ResolvedScope(entries=set(entries), selector=selector)

    # ``reportable_universe`` rather than ``entry_universe``: an entry carrying
    # two genuine tracks recipes makes the strict resolver raise, correctly,
    # because *executing* against it has no defensible default. Planning is not
    # executing, and refusing to say what a dataset holds would take the whole
    # plan down over an ambiguity that only matters when a step comes to read.
    universe = reportable_universe(ds, tracks_run_id)
    if universe:
        return ResolvedScope(entries=set(narrow_target(universe)), selector=selector)

    # No tracks yet. If this graph is what produces them, the entries it will
    # cover are the ones there is media for -- which is on disk now, so the
    # answer is exact rather than a guess.
    if graph_writes_tracks(recipe, produces_tracks):
        return ResolvedScope(entries=set(media_universe(ds)), selector=selector)
    return ResolvedScope(selector=selector)
