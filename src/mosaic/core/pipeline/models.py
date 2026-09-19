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

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


PREPARED_DATA_KINDS: Final[frozenset[str]] = frozenset(
    {"convert-points", "prepare-training-data"}
)
"""The ``models/<kind>/`` directories that hold training *data*, not a model.

A prepared dataset lives under ``models/`` because that root is what a trainer
reads and writes. It carries no weights, so anything that judges a directory
there as a trained model has to be told which ones are not. Named here, once,
because the two readers that need it -- the trained-model inventory and the
prepared-dataset one -- would otherwise each keep a list that drifts.
"""


PREPARED_ARTIFACT_COLUMNS: Final = ("artifact_path", "data_yaml")
"""Where a prepared-data row names what a trainer is handed, newest first.

``convert-points`` records ``data_yaml``. ``prepare-training-data`` records
``artifact_path``, because its artifact is a ``data.yaml``, a ``.slp`` file or a
Lightning Pose project depending on the target.
"""


def prepared_artifact_cell(row: Mapping[str, str]) -> str:
    """The stored path of what a trainer is handed, from one prepared-data row."""
    for column in PREPARED_ARTIFACT_COLUMNS:
        stored = row.get(column, "").strip()
        if stored:
            return stored
    return ""


def model_run_root(ds: Dataset, model_name: str, run_id: str) -> Path:
    """Return the run directory for a trained model (``models/<name>/<run_id>/``)."""
    return ds.get_root("models") / model_name / run_id


def model_index_path(ds: Dataset, model_name: str) -> Path:
    """Return the index CSV path for a model (``models/<name>/index.csv``)."""
    return ds.get_root("models") / model_name / "index.csv"
