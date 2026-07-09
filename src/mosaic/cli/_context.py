"""Dataset context helpers for the CLI (the library analogue of a DB session).

The library takes an explicit ``--manifest <dataset.yaml>`` -- there is no
``MOSAIC_DATA_ROOT`` here (that is a mosaic-api concept). Heavy imports stay
lazy so ``--help`` and the read-only commands never pull in the feature/tracking
stacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.cli._io import fail

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def load_dataset(manifest: Path) -> "Dataset":
    """Load the dataset at *manifest* (``Dataset(manifest_path=...).load()``)."""
    from mosaic.core.dataset import Dataset

    if not manifest.exists():
        fail(f"Manifest not found: {manifest}")
    try:
        return Dataset(manifest_path=manifest).load()
    except Exception as exc:  # noqa: BLE001 - surface any load failure cleanly
        fail(f"Failed to load dataset from {manifest}: {exc}")


def features_db_path(ds: "Dataset") -> Path:
    """Return the ``.mosaic.db`` path for *ds* (the status/progress bridge)."""
    try:
        return ds.get_root("features") / ".mosaic.db"
    except KeyError:
        fail("Dataset has no 'features' root configured.")
