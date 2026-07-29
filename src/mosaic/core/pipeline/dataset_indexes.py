"""Every index in a dataset, enumerated once -- item 6.1's second paragraph.

Three passes want the same list and each had its own copy: the two path passes
(``rewrite_index_paths`` and ``make_portable``) enumerated it in parallel sets of
local closures, and the reconciler could not exist without a third. Three copies
of "which files are indexes" is three answers to one question, and the way they
diverge is silent -- a root missing from one pass is a root that stops being
portable, or stops being reconciled, with nothing failing.

**A declarative table, not a walk.** An index is found by asking the dataset for
a root, not by globbing for ``index.csv``: a glob would reach a file under
``_tracking`` that a sweep is about to delete, and would miss a declared root
that has not been created yet. Roots resolve through ``get_root`` rather than out
of ``ds.roots`` directly, because a stored root is relative to the dataset and
``Path(root)`` would resolve it against the process working directory instead.

**Two shapes, and the distinction is not cosmetic.** Some roots hold one
``index.csv`` at the top (``tracks``, ``media``, a tracker root); some hold one
per subdirectory (``features/<name>/``, ``labels/<kind>/``, ``frames/<method>/``).
``features`` and ``labels`` are both -- they carry a root-level index *and*
per-child ones -- which is why this is a table of (root, shape) pairs rather than
a list of roots.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

import pandas as pd

from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "DatasetIndex",
    "IndexShape",
    "ReconcilableIndex",
    "iter_dataset_indexes",
    "reconcilable_index",
    "register_reconcilable_index",
]

IndexShape = Literal["root", "per_subdir"]
"""Whether a root holds one ``index.csv`` or one per child directory."""


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    """One index file, and enough about it to rewrite or reconcile it."""

    root_key: str
    path: Path
    path_columns: tuple[str, ...]


# Which roots hold indexes, and in which shape. Ordered for a stable report:
# sources, then derived, then the generated roots under `_tracking`.
_ROOT_SHAPES: Final[tuple[tuple[str, IndexShape], ...]] = (
    ("media_raw", "root"),
    ("tracks_raw", "root"),
    ("media", "root"),
    ("tracks", "root"),
    ("models", "root"),
    ("features", "root"),
    ("features", "per_subdir"),
    ("labels", "root"),
    ("labels", "per_subdir"),
    ("frames", "per_subdir"),
    *((key, "root") for key in TRACKING_ROOTS),
)


def iter_dataset_indexes(
    ds: Dataset,
    path_columns: Mapping[str, Sequence[str]] | None = None,
) -> list[DatasetIndex]:
    """Every index file this dataset declares a root for, in a stable order.

    Only roots that are *set and present on disk* are visited -- a root may be
    declared in a manifest and never created, and a pass over a dataset that has
    only been indexed must not raise on the parts that do not exist yet.

    An index file that a root declares but disk does not hold is still returned:
    the passes each decide what an absent file means, and for the reconciler it
    means "nothing to reconcile" rather than "error".

    *path_columns* is a mapping rather than a callback so that a root it does not
    mention yields ``()`` instead of whatever ``Mapping.get`` hands back. Passing
    the bare ``.get`` was the first spelling here, and it returns ``None``.
    """
    lookup = path_columns or {}
    found: list[DatasetIndex] = []
    for key, shape in _ROOT_SHAPES:
        if not ds.has_root(key):
            continue
        root = ds.get_root(key)
        if not root.exists():
            continue
        columns = tuple(lookup.get(key, ()))
        if shape == "root":
            found.append(DatasetIndex(key, root / "index.csv", columns))
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                found.append(DatasetIndex(key, child / "index.csv", columns))
    return found


# --- Reconciling: dropping rows whose file is gone ---------------------------


class ReconcilableIndex(Protocol):
    """What the reconciler needs from an index, and nothing more.

    A structural type rather than ``IndexCSV[SomeRow]``, because ``IndexCSV`` is
    generic in its row and invariant, so no single parameterization names the
    features index and the tracker indexes at once. What they share is this one
    method, which is what the reconciler actually calls.
    """

    def prune_missing(
        self, resolver: Callable[[str], Path], *, dry_run: bool = False
    ) -> pd.DataFrame: ...


_FACTORIES: Final[dict[str, Callable[[Path], ReconcilableIndex]]] = {}


def register_reconcilable_index(
    root_key: str, factory: Callable[[Path], ReconcilableIndex]
) -> None:
    """Declare how to open *root_key*'s index for reconciliation.

    Registration rather than import, for the reason this package states
    elsewhere: ``core`` does not import ``tracking``, and the tracker row classes
    live there. Each producer registers itself when its module is imported, the
    same seam ``FEATURES``, ``TRACK_CONVERTERS`` and ``OPS`` already use.

    A root with no factory is simply not reconciled, and that is deliberate --
    ``media_raw``, ``media`` and ``tracks_raw`` are raw-pandas indexes with no
    ``IndexCSV`` behind them, so they need their own pass rather than a wrong one.
    """
    _FACTORIES[root_key] = factory


def reconcilable_index(root_key: str) -> Callable[[Path], ReconcilableIndex] | None:
    """The registered opener for *root_key*, or ``None`` if it has none."""
    return _FACTORIES.get(root_key)
