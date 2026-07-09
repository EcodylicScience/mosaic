"""Filesystem path helpers for trained-model artifact storage.

These resolve locations under the Dataset ``models/`` root — a model's run
directory (``models/<name>/<run_id>/``) and its ``index.csv``. They are shared,
domain-agnostic path helpers, currently used by the tracking model-training ops
(:mod:`mosaic.tracking.ops.train`) to lay out trained pose / point / localizer
models and their run index.

Note:
    The legacy ``train_model`` scaffold and its behavior-model index
    (``ModelIndexRow`` / ``model_index``, plus ``load_model_config`` /
    ``write_model_config``) that previously lived here were removed. Behavior
    model training now runs as a *global* fit-then-apply feature via
    :func:`mosaic.core.pipeline.run.run_feature` (which persists artifacts under
    the ``features/`` root and is covered by the Job Contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def model_run_root(ds: Dataset, model_name: str, run_id: str) -> Path:
    """Return the run directory for a trained model (``models/<name>/<run_id>/``)."""
    return ds.get_root("models") / model_name / run_id


def model_index_path(ds: Dataset, model_name: str) -> Path:
    """Return the index CSV path for a model (``models/<name>/index.csv``)."""
    return ds.get_root("models") / model_name / "index.csv"
