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
from .iteration import resolve_input_scope, yield_sequences
from .manifest import Manifest, build_manifest, iter_manifest
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
    "Manifest",
    "build_manifest",
    "iter_manifest",
    "resolve_input_scope",
    "latest_feature_run_root",
    "list_feature_runs",
    "model_run_root",
    "trim_feature_output",
    "write_output",
    "yield_sequences",
]
