"""What a dataset holds: every computed artifact, its identity and its coverage.

Nothing in the stack answered "what has been computed in this dataset, with what
params, over which sequences". ``mosaic sequences`` is a tracks listing wearing a
sequences name, ``mosaic runs``/``status`` are job-log surfaces reporting
*attempts*, and ``mosaic features list`` is the registry -- what the installation
knows how to compute, not what this dataset has.

This package is the answer, and it is deliberately a **read**. Truth for
artifacts lives on disk: the ``index.csv`` files plus the files themselves, on
any filesystem. Every view built here is a cache, is never authoritative, and is
never written anywhere it could be mistaken for truth.

This module is the only public import path; the submodules are its
implementation.
"""

from __future__ import annotations

from .cache import InventoryCache, RevalidationReport
from .contributors import (
    InventoryContributor,
    inventory_contributor,
    register_inventory_contributor,
    registered_inventory_kinds,
)
from .model import (
    ARTIFACT_KINDS,
    ARTIFACT_STATUSES,
    ArtifactKind,
    ArtifactRecord,
    ArtifactRef,
    ArtifactStatus,
    Coverage,
    DatasetInventory,
    FeatureRunRef,
    FrameRunRef,
    InventoryScope,
    LabelSeriesRef,
    LabelsVariantRef,
    MediaDerivativeRef,
    MediaVariantRef,
    PreparedDatasetRef,
    TrackerRunRef,
    TracksVariantRef,
    TrainedModelRef,
    classify,
    is_artifact_kind,
    is_artifact_status,
    reclassify,
)
from .scan import (
    entry_universe,
    existing_tracks_records,
    inventory,
    narrow_target,
    reportable_universe,
    run_covers,
)
from .params import (
    LABELS_EDGE,
    TRACKS_EDGE,
    ParamsState,
    ResolvedRef,
    RunParams,
    RunParamsRead,
    ScopeBlock,
    read_run_params,
    run_tracks_variants,
)
from .wire import (
    MISSING_SAMPLE_WIRE,
    ArtifactDetailWire,
    ArtifactNotFound,
    ArtifactWire,
    CoverageWire,
    InventoryQueryInvalid,
    InventoryWire,
    KeySampleWire,
    ResolvedWire,
    coverage_wire,
    inventory_wire,
    key_sample,
    ref_address,
)

__all__ = [
    "ARTIFACT_KINDS",
    "ARTIFACT_STATUSES",
    "MISSING_SAMPLE_WIRE",
    "ArtifactDetailWire",
    "ArtifactKind",
    "ArtifactNotFound",
    "ArtifactWire",
    "CoverageWire",
    "InventoryQueryInvalid",
    "InventoryWire",
    "KeySampleWire",
    "ResolvedWire",
    "InventoryCache",
    "InventoryContributor",
    "ArtifactRecord",
    "ArtifactRef",
    "ArtifactStatus",
    "Coverage",
    "DatasetInventory",
    "FeatureRunRef",
    "FrameRunRef",
    "InventoryScope",
    "LABELS_EDGE",
    "LabelSeriesRef",
    "LabelsVariantRef",
    "MediaDerivativeRef",
    "MediaVariantRef",
    "PreparedDatasetRef",
    "ParamsState",
    "ResolvedRef",
    "RunParams",
    "RunParamsRead",
    "ScopeBlock",
    "TRACKS_EDGE",
    "TrackerRunRef",
    "TracksVariantRef",
    "TrainedModelRef",
    "classify",
    "coverage_wire",
    "inventory_wire",
    "is_artifact_kind",
    "is_artifact_status",
    "key_sample",
    "reclassify",
    "ref_address",
    "run_tracks_variants",
    "entry_universe",
    "existing_tracks_records",
    "inventory",
    "inventory_contributor",
    "narrow_target",
    "RevalidationReport",
    "read_run_params",
    "register_inventory_contributor",
    "registered_inventory_kinds",
    "reportable_universe",
    "run_covers",
]
