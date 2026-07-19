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
    """Resolved scope for feature computation.

    ``entries`` is the source of truth -- a set of (group, sequence) tuples.
    All other identifiers are derived from it.
    """

    entries: set[tuple[str, str]] = field(default_factory=set)
    frame_start: int | None = None
    frame_end: int | None = None

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


def json_ready(obj: object) -> object:
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return f"<{type(obj).__name__}>"
    return obj


def hash_params(d: object) -> str:
    d = json_ready(d)
    s = json.dumps(d, sort_keys=True, default=str)
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
