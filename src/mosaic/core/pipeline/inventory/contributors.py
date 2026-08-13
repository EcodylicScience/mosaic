"""How the ops half of the inventory arrives without inverting the layering.

``core`` does not import ``tracking`` -- the constraint ``provenance`` states and
the reason its walk leaves extracted frames out -- and tracker runs, frame runs
and trained models all live behind that line, with their row classes defined
there. Naming them here would invert the layering for the sake of an import.

So each producer registers a contributor when its module is imported, which is
the seam ``register_reconcilable_index``, ``FEATURES``, ``TRACK_CONVERTERS`` and
``OPS`` already use. The registration is a plain call at the bottom of the
producing module, not a decorator: it returns nothing, and there is nothing to
decorate.

**A kind with no contributor is reported, never silently empty.** A caller that
imported only ``mosaic.core`` has not imported the modules that fill this
registry, and answering "zero tracker runs" would be a wrong answer where
"nobody can tell you about tracker runs" is a true one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final

from .model import AnyRecord, ArtifactKind, InventoryScope

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

    from ._read import IndexReader

__all__ = [
    "InventoryContributor",
    "inventory_contributor",
    "register_inventory_contributor",
    "registered_inventory_kinds",
]

type InventoryContributor = Callable[
    ["Dataset", InventoryScope, "IndexReader"], Sequence[AnyRecord]
]
"""What a producer supplies: finished records for its own kind.

One signature covering every kind, with no ``Any`` and no cast, because a
callable's return type is covariant and ``Sequence`` is covariant in its
element: a ``Callable[..., Sequence[EntryRecord]]`` *is* a
``Callable[..., Sequence[AnyRecord]]``. Each producer keeps its precise return
type where it is written, and the consumer narrows on the ref.

Contributors are handed the shared :class:`~._read.IndexReader` rather than
opening files themselves, so a scan reads each index once however many kinds
want it.
"""

_CONTRIBUTORS: Final[dict[ArtifactKind, InventoryContributor]] = {}


def register_inventory_contributor(
    kind: ArtifactKind, contributor: InventoryContributor
) -> None:
    """Declare who can report artifacts of *kind*.

    Called at module scope by the producer, so importing the producer is what
    makes its kind available. Registering twice replaces, which is what a
    reloaded module during development should do.

    Args:
        kind: The artifact kind this reports.
        contributor: Takes ``(dataset, scope, reader)`` and returns its records.
    """
    _CONTRIBUTORS[kind] = contributor


def inventory_contributor(kind: ArtifactKind) -> InventoryContributor | None:
    """The registered contributor for *kind*, or ``None`` if nothing registered one."""
    return _CONTRIBUTORS.get(kind)


def registered_inventory_kinds() -> frozenset[ArtifactKind]:
    """Every kind some imported module can report on."""
    return frozenset(_CONTRIBUTORS)
