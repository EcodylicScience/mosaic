"""Shared, dependency-light helpers for tracking ops.

Factored out of ``ops/train.py`` so the training ops and the ``convert-points`` op
share one copy of the models-root guard and the copy-stable dataset fingerprint used
in content ``run_id`` computation. Behavior is identical to the original private
helpers -- training ``run_id``s are unchanged by the move.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.pipeline._utils import hash_params

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def ensure_models_root(ds: Dataset) -> None:
    """Ensure the dataset has a ``models`` root (default ``models/``)."""
    if not ds.has_root("models"):
        ds.set_root("models", "models")


def fingerprint_dataset(path: Path) -> str:
    """Cheap, copy-stable digest of a training/converted dataset (file text + size listing).

    Uses relative paths + file sizes (not mtimes) so a copied/moved dataset with
    identical contents fingerprints identically -- keeping run_ids deterministic
    across machines.
    """
    path = Path(path)
    parts: dict[str, object] = {}
    if path.is_file():
        parts["file"] = path.name
        try:
            parts["text"] = path.read_text(errors="ignore")
        except Exception:
            parts["text"] = ""
        base = path.parent
    else:
        base = path
    listing: list[str] = []
    if base.exists():
        for f in sorted(base.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                listing.append(f"{f.relative_to(base).as_posix()}:{size}")
    parts["listing"] = listing
    return hash_params(parts)
