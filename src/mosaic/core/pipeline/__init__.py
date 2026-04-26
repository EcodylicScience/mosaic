"""Computation pipeline for Dataset: feature runs, frame extraction, model training."""

from __future__ import annotations

from .index import (
    FeatureIndexRow,
    feature_index_path,
    feature_registry,
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
from .pipeline import CallbackStep, FeatureStep, Pipeline
from .registry import FeatureRegistry, open_registry
from .types import (
    Feature,
    Inputs,
    InputsLike,
    Params,
    Result,
)
from .viz import show_pipeline_diagram, show_pipeline_tree
from .writers import trim_feature_output, write_output

__all__ = [
    "CallbackStep",
    "Feature",
    "FeatureIndexRow",
    "FeatureRegistry",
    "FeatureStep",
    "FileSpecs",
    "FilterFactory",
    "IndexCSV",
    "Inputs",
    "InputsLike",
    "Manifest",
    "ManifestEntry",
    "Params",
    "Pipeline",
    "Result",
    "build_manifest",
    "feature_index_path",
    "feature_registry",
    "feature_run_root",
    "iter_manifest",
    "latest_feature_run_root",
    "list_feature_runs",
    "model_run_root",
    "open_registry",
    "show_pipeline_diagram",
    "show_pipeline_tree",
    "trim_feature_output",
    "write_output",
]
