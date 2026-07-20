"""Resolve a model reference (weights path or prior training run_id) to weights + lineage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.pipeline.models import model_index_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def resolve_model(ds: "Dataset", ref: str, kind: str) -> tuple[Path, str]:
    """Resolve a model reference to ``(best_weights_path, base_run_id)``.

    *ref* is either a filesystem path to weights (returns ``(path, "")`` -- no
    lineage) or a prior training ``run_id`` in ``models/<kind>/index.csv``
    (returns the recorded ``best_model_path`` and the run_id as lineage). This
    powers retrain-from-existing-model and the trained-model -> TREx
    ``detect_model`` handoff.
    """
    p = Path(ref)
    if p.exists():
        return p, ""

    idx_path = model_index_path(ds, kind)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Model reference '{ref}' is not a path and {idx_path} does not "
            f"exist; cannot resolve as a run_id."
        )
    df = pd.read_csv(idx_path)
    match = df[df["run_id"].astype(str) == ref]
    if match.empty:
        raise KeyError(f"No model run_id '{ref}' found in {idx_path}")
    best = str(match.iloc[0]["best_model_path"])
    return ds.resolve_path(best), ref
