"""Holding an inventory across calls, and noticing when it has moved.

The second cache layer. The first lives inside one scan (``_read.IndexReader``)
and stops a single call reading one index three times; this one lets a
long-lived process -- an API worker, a notebook kernel, a loop deciding what to
run next -- keep an inventory between calls and refresh only what changed.

**In RAM, never on disk.** A materialized ``.mosaic/inventory.json`` would buy
only surviving a restart, and would cost the thing that makes any of this safe:
it would sit beside the indexes looking equally authoritative, and the first
time the two disagreed somebody would have to decide which to believe. Truth for
artifacts is the index files plus the files themselves; everything here is a view
over them and is thrown away rather than reconciled.

**Stale is safe, which is why polling is enough.** A view behind reality causes
redundant or delayed work, never wrong work: a step thought incomplete is
re-dispatched and hits its cache, and a step thought complete fails loudly
downstream rather than reading a phantom. So there is no filesystem watcher --
inotify does not work over NFS, which is in the portability story -- and no
event channel to lose. :meth:`InventoryCache.revalidate` stats the index files,
which is tens of syscalls and no parsing.

**Several holders do not coordinate, and must not.** Two processes each keep
their own view, each independently stale, agreeing about nothing. Any design
where they had to agree would need the shared store the first paragraph rules
out.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.pipeline.dataset_indexes import iter_dataset_indexes
from mosaic.core.pipeline.identity_scheme import FEATURE_IDENTITY_SCHEME
from mosaic.core.pipeline.index_stamp import IndexStamp
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME
from mosaic.core.pipeline.tracks_identity import TRACKS_IDENTITY_SCHEME
from mosaic.core.scope import Scope

from .contributors import registered_inventory_kinds
from .model import ArtifactKind, DatasetInventory
from ._read import ParquetFooterMemo
from .scan import inventory

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mosaic.core.dataset import Dataset

__all__ = ["InventoryCache", "RevalidationReport", "identity_schemes"]


def identity_schemes() -> tuple[str, str, str]:
    """The hashing contracts a held inventory was built under.

    A scheme change is not staleness -- the artifacts did not move, their *names*
    did, because ``mosaic reconcile`` re-addresses them. Every key in a held
    inventory is then wrong, and refreshing the files that changed would leave a
    view half in each scheme. So this is compared, and a difference forces a full
    rebuild rather than an incremental one.
    """
    return (FEATURE_IDENTITY_SCHEME, OP_IDENTITY_SCHEME, TRACKS_IDENTITY_SCHEME)


@dataclass(frozen=True, slots=True)
class RevalidationReport:
    """What a refresh found, so a caller can see it working rather than assume it."""

    stat_count: int
    changed: tuple[Path, ...] = ()
    full_rebuild: bool = False
    reason: str = ""

    @property
    def stale(self) -> bool:
        """Did anything move?"""
        return self.full_rebuild or bool(self.changed)


@dataclass
class InventoryCache:
    """One dataset's inventory, held and refreshed rather than rebuilt.

    Not thread-safe and not shared: one holder per process, per dataset. Two
    holders are not an error -- they simply do not coordinate, which is the
    design rather than a limitation.
    """

    ds: Dataset
    _held: DatasetInventory | None = field(default=None, init=False)
    _stamps: dict[Path, IndexStamp] = field(default_factory=dict, init=False)
    _root_keys: dict[Path, frozenset[str]] = field(default_factory=dict, init=False)
    _schemes: tuple[str, str, str] = field(default=("", "", ""), init=False)
    _kinds: frozenset[ArtifactKind] = field(default=frozenset(), init=False)
    _generation: int = field(default=0, init=False)
    _footers: ParquetFooterMemo = field(default_factory=ParquetFooterMemo, init=False)

    @property
    def generation(self) -> int:
        """How many views this holder has built or dropped.

        A result computed from the held view is current while this number is the
        one it was computed under: a rebuild after an index moved, and
        :meth:`invalidate`, each advance it. Call :meth:`revalidate` first, because
        a view that has not been checked still carries the number of the one the
        files have since left.
        """
        return self._generation

    def get(
        self,
        *,
        kinds: Iterable[ArtifactKind] | None = None,
        scope: Scope | None = None,
        tracks_run_id: str | None = None,
    ) -> DatasetInventory:
        """The held inventory, building it if there is none.

        Does **not** revalidate: call :meth:`revalidate` when you want to know
        whether the dataset moved. Keeping the two apart is what lets a caller
        decide how often to pay for the stats, and what stops a read looking like
        a refresh.
        """
        if self._held is None:
            self._rebuild(kinds=kinds, scope=scope, tracks_run_id=tracks_run_id)
        held = self._held
        assert held is not None
        return held

    def revalidate(self) -> RevalidationReport:
        """Refresh what moved. Stats the index files and re-reads only if needed.

        Returns:
            What was checked and what changed. An unchanged dataset costs one
            ``stat`` per index -- tens of syscalls, no parsing -- which is what
            makes calling this on a timer reasonable.
        """
        if self._held is None:
            return RevalidationReport(stat_count=0, full_rebuild=True, reason="unheld")

        if self._schemes != identity_schemes():
            self._rebuild(kinds=self._kinds)
            return RevalidationReport(
                stat_count=0, full_rebuild=True, reason="identity scheme changed"
            )
        if self._kinds != _kinds_available(self._kinds):
            # A producer was imported after the view was built, so a kind that
            # honestly read as unavailable is now answerable.
            self._rebuild(kinds=None)
            return RevalidationReport(
                stat_count=0, full_rebuild=True, reason="a contributor registered"
            )

        current, root_keys = self._stamp_indexes()
        changed = tuple(
            sorted(
                path
                for path, stamp in current.items()
                if self._stamps.get(path) != stamp
            )
        )
        gone = tuple(sorted(set(self._stamps) - set(current)))
        if changed or gone:
            moved = changed + gone
            kinds = self._kinds_of(moved, {**self._root_keys, **root_keys})
            if kinds is None:
                self._rebuild(kinds=self._kinds, stamped=(current, root_keys))
            else:
                self._rebuild_kinds(kinds, (current, root_keys))
        return RevalidationReport(stat_count=len(current), changed=changed + gone)

    def invalidate(self) -> None:
        """Drop the held view, so the next :meth:`get` reads from disk.

        The escape hatch for the one thing stamps cannot see: ``params.json`` is
        not stamped -- stamping every run's sidecar would blow the tens-of-files
        budget on the first large dataset -- so a pass that rewrites provenance
        under an unchanged index goes unnoticed. ``mosaic reconcile`` is that
        pass, and it builds its own inventory rather than sharing a holder.

        A feature run's status reads the entries its ``params.json`` records as
        asked for. An invocation that changes them without writing an index
        row, such as one whose every entry failed, leaves the cached view stale
        until this is called.
        """
        self._held = None
        self._stamps = {}
        self._root_keys = {}
        self._generation += 1

    def _stamp_indexes(
        self,
    ) -> tuple[dict[Path, IndexStamp], dict[Path, frozenset[str]]]:
        """Stat every index now, with the root keys that declare each one.

        One file can be declared by two roots when a root sits inside another.
        """
        stamps: dict[Path, IndexStamp] = {}
        keys: dict[Path, set[str]] = {}
        for index in iter_dataset_indexes(self.ds):
            stamps[index.path] = IndexStamp.of(index.path)
            keys.setdefault(index.path, set()).add(index.root_key)
        return stamps, {path: frozenset(found) for path, found in keys.items()}

    def _kinds_of(
        self, moved: Iterable[Path], root_keys: Mapping[Path, frozenset[str]]
    ) -> frozenset[ArtifactKind] | None:
        """The kinds that a move of the indexes *moved* can change, or ``None`` for all.

        A feature run's index is read by the ``feature`` kind alone: the tracks
        variants, the label variants and the media do not read it, and a run is
        measured against tracks that did not move. Any other index may be read by
        any kind, so a move of one rebuilds every kind. An index is a feature
        index by the root that declares it, and by that alone.
        """
        features = frozenset({"features"})
        if all(root_keys.get(path) == features for path in moved):
            return frozenset({"feature"})
        return None

    def _rebuild_kinds(
        self,
        kinds: frozenset[ArtifactKind],
        stamped: tuple[dict[Path, IndexStamp], dict[Path, frozenset[str]]],
    ) -> None:
        """Read *kinds* again and keep the other records as they are.

        *stamped* is what the indexes looked like before the scan began, so an
        index that moves during it is seen on the next revalidation.
        """
        held = self._held
        assert held is not None
        again = kinds & self._kinds
        if again:
            fresh = inventory(
                self.ds,
                kinds=again,
                scope=held.scope.selector,
                tracks_run_id=held.scope.tracks_run_id,
                footers=self._footers,
            )
            order = sorted(held.scope.kinds)
            self._held = replace(
                held,
                records=tuple(
                    record
                    for kind in order
                    for record in (fresh if kind in again else held).of_kind(kind)
                ),
                errors=(
                    *(
                        error
                        for error in held.errors
                        if error.split(":", 1)[0] not in again
                    ),
                    *fresh.errors,
                ),
            )
        self._generation += 1
        self._stamps, self._root_keys = stamped

    def _rebuild(
        self,
        *,
        kinds: Iterable[ArtifactKind] | None = None,
        scope: Scope | None = None,
        tracks_run_id: str | None = None,
        stamped: tuple[dict[Path, IndexStamp], dict[Path, frozenset[str]]]
        | None = None,
    ) -> None:
        held = self._held
        # Stamped before the scan, so that an index that moves while it runs is
        # seen as moved by the next revalidation.
        stamps, root_keys = self._stamp_indexes() if stamped is None else stamped
        built = inventory(
            self.ds,
            kinds=kinds,
            scope=scope
            if scope is not None
            else (held.scope.selector if held else None),
            tracks_run_id=tracks_run_id
            if tracks_run_id is not None
            else (held.scope.tracks_run_id if held else None),
            footers=self._footers,
        )
        self._held = built
        self._generation += 1
        self._kinds = built.scope.kinds
        self._schemes = identity_schemes()
        self._stamps = stamps
        self._root_keys = root_keys


def _kinds_available(previous: frozenset[ArtifactKind]) -> frozenset[ArtifactKind]:
    """What a rebuild would cover now, to compare against what it covered then."""
    from .scan import CORE_KINDS

    widened = CORE_KINDS | registered_inventory_kinds()
    # Only widening counts. A caller that asked for one kind is not stale because
    # other kinds exist; it is stale when the set it *would* get has grown.
    return widened if previous >= CORE_KINDS else previous
