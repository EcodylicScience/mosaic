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
gave one -- that is the submission saying what it wants. Otherwise the dataset's
tracks universe, which is what every feature reads. And when that is empty
*because the graph itself produces the tracks*, the media universe, since a
tracker turns videos into tracks one entry at a time and those videos are on
disk before anything is planned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..inventory.scan import narrow_target, reportable_universe
from .model import Recipe
from .topo import topological_order

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaic.core.dataset import Dataset

    from ..inventory.model import Entry

__all__ = ["graph_writes_tracks", "intended_scope", "media_universe"]


def media_universe(ds: Dataset) -> frozenset[tuple[str, str]]:
    """Every ``(group, sequence)`` this dataset has media for.

    The cameras of one recording collapse back to one entry here, because a
    tracks table is addressed by ``(group, sequence)`` and a graph is planned in
    those terms. The camera axis matters to what a *frame run* covers, and this
    is not that.
    """
    return frozenset(
        (entry.group, entry.sequence) for entry in ds.resolve_media_scope(None, None)
    )


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
    intended_entries: Iterable[tuple[str, str]] | None = None,
    *,
    tracks_run_id: str | None = None,
    produces_tracks: Iterable[str] = (),
) -> frozenset[Entry]:
    """The entries this graph is planned over.

    Args:
        ds: The dataset being planned against. Read only.
        recipe: The graph, consulted only for whether it produces its own tracks.
        intended_entries: What the submission asked for, or ``None`` for
            everything the dataset can process.
        tracks_run_id: Which tracks variant defines the universe. Pass the one
            the graph reads; measuring against the whole index would widen the
            scope past what any step will actually see.
        produces_tracks: The op kinds that bridge into ``tracks/``, so a graph
            that makes its own tracks can be recognised without this module
            importing the op registry.

    Returns:
        The ``(group, sequence)`` set every ``scope_dependent`` step in this
        graph will hash. Empty is a real answer -- a dataset with neither tracks
        nor media has nothing to plan over -- and is not a failure here; the
        step that cares reports the shortfall.
    """
    if intended_entries is not None:
        # An explicit narrowing is the submission speaking, and it is not
        # widened back out by anything found on disk.
        return frozenset(intended_entries)

    # ``reportable_universe`` rather than ``entry_universe``: an entry carrying
    # two genuine tracks recipes makes the strict resolver raise, correctly,
    # because *executing* against it has no defensible default. Planning is not
    # executing, and refusing to say what a dataset holds would take the whole
    # plan down over an ambiguity that only matters when a step comes to read.
    universe = reportable_universe(ds, tracks_run_id)
    if universe:
        return narrow_target(universe)

    # No tracks yet. If this graph is what produces them, the entries it will
    # cover are the ones there is media for -- which is on disk now, so the
    # answer is exact rather than a guess.
    if graph_writes_tracks(recipe, produces_tracks):
        return media_universe(ds)
    return frozenset()
