from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.registry import FeatureRegistry, open_registry

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def feature_run_root(ds: Dataset, feature_name: str, run_id: str) -> Path:
    return ds.get_root("features") / feature_name / run_id


def feature_index_path(ds: Dataset, feature_name: str) -> Path:
    return ds.get_root("features") / feature_name / "index.csv"


# --- Feature Index ---


@dataclass(frozen=True, slots=True)
class FeatureIndexRow(RunIndexRowBase):
    """Typed row for the feature index CSV."""

    feature: str
    version: str
    group: str
    sequence: str
    params_hash: str
    n_rows: int = 0


def feature_index(path: Path) -> IndexCSV[FeatureIndexRow]:
    """Factory: return an IndexCSV configured for the feature index schema."""
    return IndexCSV(
        path,
        FeatureIndexRow,
        dedup_keys=["run_id", "group", "sequence"],
    )


def list_feature_runs(ds: Dataset, feature_name: str) -> pd.DataFrame:
    return feature_index(feature_index_path(ds, feature_name)).list_runs()


def latest_feature_run_root(ds: Dataset, feature_name: str) -> tuple[str, Path]:
    idx = feature_index(feature_index_path(ds, feature_name))
    run_id = idx.latest_run_id()
    return run_id, feature_run_root(ds, feature_name, run_id)


def feature_registry(ds: Dataset, *, migrate_csv: bool = True) -> FeatureRegistry:
    """Open (or create) the SQLite feature registry for a dataset."""
    return open_registry(ds.get_root("features"), migrate_csv=migrate_csv)
