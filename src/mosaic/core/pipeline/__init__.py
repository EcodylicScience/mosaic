"""Computation pipeline for Dataset: feature runs, frame extraction.

This file binds every public name on access. Python executes it before any
``mosaic.core.pipeline.*`` submodule, and every consumer of a leaf below pays
for each name bound at module scope. The modules that define these names import
pandas, and the media op modules import cv2 and av.
``tests/meta/test_pipeline_lazy_exports.py`` checks that importing the package
loads none of them.

Each name is also declared under ``TYPE_CHECKING`` for type checkers and editors.

The media ops register when their modules are imported. This file imports none
of those modules. :func:`~mosaic.core.pipeline.ops.registered_op` and
``mosaic.tracking.register_ops`` import them through
:func:`~mosaic.core.pipeline.ops.ensure_core_ops_registered`, and
:func:`~mosaic.core.pipeline.dataset_indexes.reconcilable_index` imports the
modules that register the core indexes.
"""

from typing import TYPE_CHECKING, Final

from mosaic.lazy_exports import lazy_export

if TYPE_CHECKING:
    from .index import (
        FeatureIndexRow as FeatureIndexRow,
        feature_index_path as feature_index_path,
        feature_run_root as feature_run_root,
        latest_feature_run_root as latest_feature_run_root,
        list_feature_runs as list_feature_runs,
    )
    from .index_csv import IndexCSV as IndexCSV
    from .loading import (
        alignment_verdict as alignment_verdict,
        entity_level_of as entity_level_of,
    )
    from .manifest import (
        FileSpecs as FileSpecs,
        FilterFactory as FilterFactory,
        Manifest as Manifest,
        ManifestEntry as ManifestEntry,
        build_manifest as build_manifest,
        iter_manifest as iter_manifest,
    )
    from .pipeline import (
        CallbackStep as CallbackStep,
        FeatureStep as FeatureStep,
        Pipeline as Pipeline,
    )
    from .run import load_values as load_values
    from .types import (
        Feature as Feature,
        Inputs as Inputs,
        InputsLike as InputsLike,
        Result as Result,
    )
    from .viz import (
        show_pipeline_diagram as show_pipeline_diagram,
        show_pipeline_tree as show_pipeline_tree,
    )
    from .writers import (
        trim_feature_output as trim_feature_output,
        write_output as write_output,
    )

_MODULE_OF: Final[dict[str, str]] = {
    "FeatureIndexRow": ".index",
    "feature_index_path": ".index",
    "feature_run_root": ".index",
    "latest_feature_run_root": ".index",
    "list_feature_runs": ".index",
    "IndexCSV": ".index_csv",
    "alignment_verdict": ".loading",
    "entity_level_of": ".loading",
    "FileSpecs": ".manifest",
    "FilterFactory": ".manifest",
    "Manifest": ".manifest",
    "ManifestEntry": ".manifest",
    "build_manifest": ".manifest",
    "iter_manifest": ".manifest",
    "CallbackStep": ".pipeline",
    "FeatureStep": ".pipeline",
    "Pipeline": ".pipeline",
    "load_values": ".run",
    "Feature": ".types",
    "Inputs": ".types",
    "InputsLike": ".types",
    "Result": ".types",
    "show_pipeline_diagram": ".viz",
    "show_pipeline_tree": ".viz",
    "trim_feature_output": ".writers",
    "write_output": ".writers",
}
"""The submodule that defines each public name."""


def __getattr__(name: str) -> object:
    """Import the submodule that defines *name* on first access."""
    return lazy_export(__name__, _MODULE_OF, name)


__all__ = [
    "entity_level_of",
    "alignment_verdict",
    "CallbackStep",
    "Feature",
    "FeatureIndexRow",
    "FeatureStep",
    "FileSpecs",
    "FilterFactory",
    "IndexCSV",
    "Inputs",
    "InputsLike",
    "Manifest",
    "ManifestEntry",
    "Pipeline",
    "Result",
    "build_manifest",
    "feature_index_path",
    "feature_run_root",
    "iter_manifest",
    "latest_feature_run_root",
    "list_feature_runs",
    "load_values",
    "show_pipeline_diagram",
    "show_pipeline_tree",
    "trim_feature_output",
    "write_output",
]
