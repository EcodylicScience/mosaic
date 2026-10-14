"""The trained-model index, and what the inventory says about trained models.

Each training op records one row per run in models/<kind>/index.csv. This
module reads and writes that index and decides whether a run finished. It
imports no tracker, trainer or extraction code, and loads only the media modules
that ``mosaic.core.pipeline.inventory`` loads, so an inventory reads trained
models without loading what trains them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase, index_records
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    Coverage,
    InventoryScope,
    TrainedModelRef,
    classify,
)
from mosaic.core.pipeline.models import (
    holds_trained_models,
    model_index_path,
    model_run_root,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.inventory._read import IndexReader

__all__ = [
    "TRAINED_MODEL_INDEX_COLUMNS",
    "TrainedModelIndexRow",
    "adopt_trained_model_columns",
    "trained_model_index",
    "trained_model_records",
    "training_is_complete",
]


@dataclass(frozen=True, slots=True)
class TrainedModelIndexRow(RunIndexRowBase):
    """Typed row for a trained-model index CSV (``models/<kind>/index.csv``).

    ``best_model_path`` and ``metrics_path`` are stored dataset-root-relative so
    the index survives a move. A new path column here must also be added to
    ``_INDEX_PATH_COLUMNS["models"]`` in ``core/dataset.py``, or the two
    path-repair passes will not see it and it silently stops being portable.
    """

    kind: str
    base_model: str
    base_run_id: str  # lineage: the prior run_id this was retrained from ("" if none)
    best_model_path: str
    metrics_path: str
    n_epochs: int
    status: str
    # The base weights' content digest, recorded whether or not they had a run to
    # name them. Provenance, never identity: ``base_run_id`` already carries what
    # enters the hash, and this says which exact bytes were behind it. Defaulted
    # so every existing construction site keeps working, and not path-bearing, so
    # no portability rewrite list.
    base_digest: str = ""
    # What shape of artifact this run produced -- a ``ModelShape``. Empty means a
    # single weights file, which is what every row written before a model could
    # be a directory describes, so an old index keeps resolving unchanged.
    artifact_shape: str = ""
    # The artifact root, when it is not the weights file: a Lightning Pose model
    # directory, or the ordered directory a SLEAP run leaves behind. Empty falls
    # back to ``best_model_path``. Root-relative, and listed in the ``models``
    # entry of ``_INDEX_PATH_COLUMNS`` so it survives a dataset move.
    artifact_path: str = ""
    # Which head or architecture the artifact carries, read from its config.
    # Provenance, never identity -- the tracker rows already record this and the
    # trained-model row had nowhere to put it.
    model_type: str = ""
    # What the run was trained on: the data reference as the run resolved it,
    # root-relative when it lies inside the dataset, and the content fingerprint
    # that entered the identifier. The first link of the provenance chain from a
    # model back to the annotations behind it. Before these columns the data
    # path survived only in whatever submitted the run, so nothing reading the
    # dataset alone could say what a model had seen. ``data_path`` is listed in
    # the ``models`` entry of ``_INDEX_PATH_COLUMNS``.
    data_path: str = ""
    data_fingerprint: str = ""
    # Which linked library served the base weights, as ``<link id>@<uuid>``.
    # Empty when they were local or a bare path. Provenance, never identity.
    base_origin: str = ""


TRAINED_MODEL_INDEX_COLUMNS: list[str] = [
    field.name for field in fields(TrainedModelIndexRow)
]


def adopt_trained_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a trained-model index read off disk up to the current schema.

    ``trained_model_index`` had no ``adopt`` hook. Survivable while the schema
    was fixed; not survivable the moment it grows a column, because an absent
    one concatenated against a real row widens ``n_epochs`` and ``100`` reaches
    disk as ``100.0``. Every column built with an explicit ``object`` dtype, the
    same shape as ``tracks_index.adopt_legacy_columns``.
    """
    out = pd.DataFrame(index=df.index)
    for column in TRAINED_MODEL_INDEX_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    return out


def trained_model_index(path: Path) -> IndexCSV[TrainedModelIndexRow]:
    return IndexCSV(
        path,
        TrainedModelIndexRow,
        dedup_keys=["run_id"],
        adopt=adopt_trained_model_columns,
    )


def training_is_complete(ds: Dataset, kind: str, run_id: str) -> bool:
    """Has this exact training run already finished and left its artifact?

    The completion evidence is the ``models/<kind>/index.csv`` row, because
    ``finalize_training`` writes it only after the trainer returns. Deliberately
    **not** the artifact alone: Ultralytics writes ``best.pt`` progressively, so a
    run killed mid-training leaves a plausible-looking weights file, and adopting
    that would hand a caller a half-trained model under a finished run's id. The
    row is the only thing on disk that means "the trainer returned".

    Both halves are required. The row proves the trainer finished; the artifact
    check catches the row outliving what it points at -- a swept run root, a
    hand-deleted directory -- in which case the honest answer is to train again.

    Absent index, absent row, unreadable index: not complete. A false negative
    costs a recompute, a false positive ships the wrong model.
    """
    index_path = model_index_path(ds, kind)
    if not index_path.exists():
        return False
    try:
        rows = trained_model_index(index_path).read(run_id=run_id)
    except (OSError, ValueError, KeyError):
        return False
    if rows.empty:
        return False
    if text_cell(rows.iloc[-1].get("status", "")) != "finished":
        return False
    recorded = text_cell(rows.iloc[-1].get("artifact_path", "")) or text_cell(
        rows.iloc[-1].get("best_model_path", "")
    )
    if not recorded:
        return False
    return ds.resolve_path(recorded).exists()


def trained_model_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[str]]:
    """Every trained model under ``models/<kind>/<run_id>/``.

    A model is one artifact rather than a per-entry set, so its coverage is
    itself: covered or not. Covered means what :func:`training_is_complete`
    already means -- an index row that says finished *and* an artifact that
    resolves -- because Ultralytics writes ``best.pt`` progressively, so the file
    alone is not evidence the trainer returned, and a row can outlive what it
    points at.

    ``scope.entries`` does not narrow this: a model is fitted over a scope
    recorded elsewhere, and filtering it by entry would silently drop every model
    from a scoped query rather than reporting the ones that exist.
    """
    records: list[ArtifactRecord[str]] = []
    for kind in root_subdirectories(ds, "models"):
        # ``models/`` also holds prepared training data, which has no weights.
        # Judged as a model, every such directory read as a finished row whose
        # artifact was missing -- damage that was never there.
        if not holds_trained_models(kind):
            continue
        index_path = model_index_path(ds, kind)
        reader.note(index_path)
        frame = reader.frame(
            index_path, lambda p=index_path: trained_model_index(p).read()
        )
        if frame.empty or "run_id" not in frame.columns:
            continue
        seen: dict[str, dict[str, str]] = {}
        for record in index_records(frame):
            seen.setdefault(record.get("run_id", ""), record)
        for run_id in sorted(seen):
            row = seen[run_id]
            covered = training_is_complete(ds, kind, run_id)
            coverage = Coverage(
                target=frozenset({run_id}),
                present=frozenset({run_id} if covered else ()),
            )
            records.append(
                ArtifactRecord[str](
                    ref=TrainedModelRef(op_kind=kind, run_id=run_id),
                    name=kind,
                    run_id=run_id,
                    coverage=coverage,
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
                    upstreams=tuple(
                        value for value in (row.get("base_run_id", ""),) if value
                    ),
                )
            )
    return records
