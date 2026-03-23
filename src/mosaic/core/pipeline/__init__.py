"""Computation pipeline for Dataset: feature runs, frame extraction, model training."""

from __future__ import annotations

from .index import (
    FeatureIndexRow,
    feature_index_path,
    feature_run_root,
    latest_feature_run_root,
    list_feature_runs,
)
from .index_csv import IndexCSV
from .manifest import (
    FileSpecs,
    FilterFactory,
    Manifest,
    ManifestEntry,
    build_manifest,
    iter_manifest,
)
from .models import model_run_root
from .types import (
    Feature,
    Inputs,
    InputsLike,
    Params,
    Result,
)
from .writers import trim_feature_output, write_output

__all__ = [
    "Feature",
    "Inputs",
    "InputsLike",
    "Params",
    "Result",
    "FeatureIndexRow",
    "IndexCSV",
    "feature_index_path",
    "feature_run_root",
    "FileSpecs",
    "FilterFactory",
    "Manifest",
    "ManifestEntry",
    "build_manifest",
    "iter_manifest",
    "latest_feature_run_root",
    "list_feature_runs",
    "model_run_root",
    "trim_feature_output",
    "write_output",
]
