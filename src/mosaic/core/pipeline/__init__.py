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
from .iteration import inputset_from_inputs, yield_sequences
from .models import model_run_root
from .writers import trim_feature_output, write_output

__all__ = [
    "FeatureIndexRow",
    "IndexCSV",
    "feature_index_path",
    "feature_run_root",
    "inputset_from_inputs",
    "latest_feature_run_root",
    "list_feature_runs",
    "model_run_root",
    "trim_feature_output",
    "write_output",
    "yield_sequences",
]
