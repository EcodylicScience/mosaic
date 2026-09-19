"""What the inventory says about prepared training data.

``models/`` holds two different things: the models a trainer wrote, and the
training data a trainer reads. Both belong there, because that root is a contract
rather than a provenance, but only one of them has weights. Reported as a model,
every prepared dataset read as a finished row whose artifact was missing.

This module reports them under their own kind. Which directories those are is
:data:`~mosaic.core.pipeline.models.PREPARED_DATA_KINDS`, the same list the
trained-model inventory reads to leave them alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.pipeline.inventory._read import IndexReader
from mosaic.core.pipeline.inventory.contributors import register_inventory_contributor
from mosaic.core.pipeline.inventory.model import ArtifactRecord, InventoryScope
from mosaic.core.pipeline.models import (
    PREPARED_DATA_KINDS,
    model_index_path,
    model_run_root,
    prepared_artifact_cell,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset

__all__ = ["prepared_dataset_is_complete"]


def _read(path: Path) -> pd.DataFrame:
    """The index at *path* as text cells, or an empty frame when it is absent."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def prepared_dataset_is_complete(ds: Dataset, row: dict[str, str]) -> bool:
    """Did this preparation finish, and is what it wrote still there?

    Both halves, for the reason a trained model needs both: the row is the only
    thing on disk that means the writer returned, and a row can outlive the
    directory it names.
    """
    if row.get("status", "") != "finished":
        return False
    stored = prepared_artifact_cell(row)
    return bool(stored) and ds.resolve_path(stored).exists()


def _prepared_dataset_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[str]]:
    """Every prepared training dataset under ``models/<kind>/<run_id>/``.

    One artifact per run, like a model, so its coverage is itself. ``scope`` does
    not narrow it: a prepared dataset is built from annotation sets, which span
    sequences, and filtering by entry would drop every one from a scoped query
    rather than report the ones that exist.
    """
    from mosaic.core.pipeline.dataset_indexes import root_subdirectories
    from mosaic.core.pipeline.index_csv import index_records
    from mosaic.core.pipeline.inventory.model import (
        Coverage,
        PreparedDatasetRef,
        classify,
    )

    _ = scope
    records: list[ArtifactRecord[str]] = []
    # Through ``root_subdirectories``, which answers an undeclared or absent
    # ``models`` root with nothing. Asking for the root directly raises on a
    # dataset that never declared one, and having no models is not an error.
    present = root_subdirectories(ds, "models")
    for kind in (name for name in present if name in PREPARED_DATA_KINDS):
        index_path = model_index_path(ds, kind)
        reader.note(index_path)
        frame = reader.frame(index_path, lambda p=index_path: _read(p))
        if frame.empty or "run_id" not in frame.columns:
            continue
        seen: dict[str, dict[str, str]] = {}
        for record in index_records(frame):
            seen.setdefault(record.get("run_id", ""), record)
        for run_id in sorted(seen):
            row = seen[run_id]
            covered = prepared_dataset_is_complete(ds, row)
            records.append(
                ArtifactRecord[str](
                    ref=PreparedDatasetRef(op_kind=kind, run_id=run_id),
                    name=kind,
                    run_id=run_id,
                    coverage=Coverage(
                        target=frozenset({run_id}),
                        present=frozenset({run_id} if covered else ()),
                    ),
                    status=classify(
                        satisfied=covered,
                        any_covered=covered,
                        orphan_rows=not covered and row.get("status", "") == "finished",
                        orphan_files=False,
                        drifted=False,
                        finished=bool(row.get("finished_at", "")),
                    ),
                    run_root=model_run_root(ds, kind, run_id),
                    index_path=index_path,
                    rows=frozenset({run_id}),
                    started_at=row.get("started_at", ""),
                    finished_at=row.get("finished_at", ""),
                    upstreams=(),
                )
            )
    return records


register_inventory_contributor("prepared-dataset", _prepared_dataset_records)
