from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mosaic.core.helpers import make_entry_key

# ``now_iso`` / ``new_execution_id`` now live in the dependency-light leaf
# ``mosaic.runlog`` (so external readers can import them without the heavy pipeline).
# Re-exported here to keep the historical ``from ._utils import ...`` call sites working.
from mosaic.runlog import (
    new_execution_id as new_execution_id,
    now_iso as now_iso,
)

# --- Pipeline data types ---


@dataclass
class Scope:
    """What a feature run resolved to, before it computes anything.

    ``entries`` is the source of truth for *which sequences* -- a set of
    (group, sequence) tuples, from which ``groups``, ``sequences``,
    ``entry_keys`` and ``entry_map`` are all derived.

    ``tracks_variants`` is the other half of "what this run reads": which tracks
    recipes produced the tables behind those entries, sorted and deduplicated,
    empty when nothing said. It rides here because ``build_manifest`` is what
    learns it and ``compute_run_id`` already takes a ``Scope``, so no third
    channel is needed between them and no state has to be stashed on the
    feature's ``Inputs`` -- which is a module-level shared default in two dozen
    feature modules and would leak a pin across datasets and across tests.

    **It is deliberately computed before the groups/sequences/entries narrowing**
    (see ``_resolve_tracks``). A scope-free feature gets one identifier for every
    scope, and a term that moved with the narrowing would quietly end that,
    minting several identifiers for one computation and leaving ``Pipeline.clean``
    to delete all but the one it predicted.
    """

    entries: set[tuple[str, str]] = field(default_factory=set)
    frame_start: int | None = None
    frame_end: int | None = None
    tracks_variants: tuple[str, ...] = ()

    @property
    def groups(self) -> set[str]:
        return {group for group, _ in self.entries}

    @property
    def sequences(self) -> set[str]:
        return {seq for _, seq in self.entries}

    @property
    def entry_keys(self) -> set[str]:
        return {make_entry_key(group, seq) for group, seq in self.entries}

    @property
    def entry_map(self) -> dict[str, tuple[str, str]]:
        return {make_entry_key(group, seq): (group, seq) for group, seq in self.entries}


@dataclass
class FeatureMeta:
    """Output metadata for a single (group, sequence) within run_feature."""

    group: str
    sequence: str
    out_path: Path


# --- Utility functions ---


def _ordering_key(value: object) -> str:
    """Total, deterministic order for already-serialized values.

    ``sorted()`` on a set of mixed or unorderable elements (dicts, or ints
    beside strings) raises. Everything reaching here has been through the
    conversion below, so it is plain JSON data and re-serializing is total.
    """
    return json.dumps(value, sort_keys=True)


def _ready(obj: object, *, strict: bool) -> object:
    """Recursively make *obj* JSON-serializable.

    ``strict`` selects the identity contract over the provenance one; see
    ``identity_ready`` and ``json_ready``.
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _ready(v, strict=strict) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # Ordered, because a set has none of its own: iteration order varies
        # with PYTHONHASHSEED, and ``json.dumps(sort_keys=True)`` orders dict
        # keys only, so an unordered set in identity hashes differently in every
        # process. Sorted on the serialized form so heterogeneous sets work.
        return sorted((_ready(v, strict=strict) for v in obj), key=_ordering_key)
    if isinstance(obj, (list, tuple)):
        # NOT ordered. Sequence order is semantic wherever it is hashed today:
        # ``Inputs`` is a tuple whose order feeds both the digest and
        # ``storage_suffix()``, ``_frame_range`` is ``[start, end]``, and the
        # per-camera composition planned for item 4.4 is mixed -- its outer list
        # sorted, its inner uid list never. Sorting here would move identifiers
        # and corrupt that composition.
        return [_ready(v, strict=strict) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        if strict:
            raise TypeError(
                f"{type(obj).__name__} cannot be represented in an identity "
                f"payload. Every value of an unrepresentable type would hash "
                f"alike, so the collision would present as a cache hit serving "
                f"another run's outputs. Convert it to a JSON-representable "
                f"value (see JsonValue), or exclude it from identity with "
                f"HASH_EXCLUDE if it is a throughput knob."
            )
        return f"<{type(obj).__name__}>"
    return obj


def json_ready(obj: object) -> object:
    """Recursively make an object JSON-serializable, degrading if it cannot.

    The **provenance** contract. Used by the writers that record what ran --
    ``params.json`` and the two ``run_params.json`` files -- each of which
    writes best-effort inside a ``try/except`` that prints and continues. An
    unrepresentable value becomes a ``"<TypeName>"`` placeholder, because a
    lossy record of what ran beats no record at all.

    Never use this for identity: the placeholder is a constant, so every value
    of that type collapses to one digest. Use ``identity_ready``.
    """
    return _ready(obj, strict=False)


def identity_ready(obj: object) -> object:
    """Recursively make an object JSON-serializable, raising if it cannot.

    The **identity** contract, and the input to ``hash_params``. Differs from
    ``json_ready`` in exactly two ways, both of which are correctness rather
    than taste:

    - a set is ordered before hashing, so the digest does not vary by process
    - an unrepresentable value raises instead of collapsing to a constant

    Raises:
        TypeError: if any value cannot be represented in JSON.
    """
    return _ready(obj, strict=True)


def hash_params(d: object) -> str:
    """The 40-bit identity digest over *d*.

    **The width is fixed at 10 hex characters and does not change.** This
    function mints more than feature identity: the frame-extraction identifier,
    the TREx run id, the transcode run id, and every tracking-op id all come
    from here. The frame-extraction one is pinned outside mosaic -- the control
    plane writes it to ``AnnotationFrame.run_id`` and embeds it mid-string in
    ``image_path``, on version-controlled rows carrying keypoint annotation
    labor -- so moving it orphans annotation work rather than costing a
    recompute.

    Collision headroom is not the constraint: the namespace is one
    ``features/<name>/`` directory, where the birthday bound is 4.5e-7 at a
    thousand distinct parameter sets and reaching a tenth of a percent takes
    roughly 47,000. If a family ever does need more, widen it inside that
    family's own minter behind an identity-scheme bump, not here.

    ``json.dumps`` is called without a ``default=`` fallback on purpose: a
    fallback would silently stringify whatever ``identity_ready`` was meant to
    reject, making the raise unreachable.
    """
    s = json.dumps(identity_ready(d), sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


# Process umask, read once at import (single-threaded startup). Used to restore
# sensible permissions on temp files, which ``mkstemp`` creates mode 0600.
_UMASK = os.umask(0)
os.umask(_UMASK)


def atomic_write(final_path: Path, write_fn: Callable[[Path], object]) -> None:
    """Write *final_path* atomically: *write_fn* fills a temp file, then rename.

    ``write_fn`` receives a temp path in the same directory and must write the
    full contents there. On success the temp is ``os.replace``-d onto
    *final_path* (an atomic same-filesystem rename); a concurrent reader never
    sees a partial file, and a failed/interrupted write never clobbers a
    pre-existing *final_path*. The temp is removed if anything raises. The temp
    name is a leading-dot, ``.tmp``-suffixed hidden file so an orphan left by a
    hard kill never matches ``*.parquet`` output filters.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.stem}-", suffix=".tmp"
    )
    os.close(fd)  # mkstemp returns an open fd; the writer reopens tmp by path
    tmp_path = Path(tmp)
    try:
        os.chmod(tmp_path, 0o666 & ~_UMASK)  # mkstemp is 0600; restore umask perms
        write_fn(tmp_path)
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def coerce_np(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def derive_storage_name(feature_name: str, inputs_suffix: str | None) -> str:
    """Compute on-disk directory name for a feature run.

    If the feature reads from upstream features, the directory includes
    a ``__from__`` suffix (e.g. ``speed__from__tracks``).
    """
    if inputs_suffix is not None:
        return f"{feature_name}__from__{inputs_suffix}"
    return feature_name
