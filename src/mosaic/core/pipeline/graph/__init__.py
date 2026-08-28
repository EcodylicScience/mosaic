"""The pipeline as a file: a recipe, a request beside it, and what to run.

A **recipe** is the pipeline -- a JSON graph identified by content digest,
diffable, shareable, portable across datasets, and never holding a resolved
``run_id``, because those are dataset state. A **plan** is what
:func:`~mosaic.core.pipeline.graph.plan.plan_pipeline` returns for a given
dataset and recipe: what to run, under what identity, and what is already done.

Deliberately separate from the live-object
:class:`~mosaic.core.pipeline.pipeline.Pipeline`, which holds feature *classes*
and a ``CallbackStep`` wrapping a live callable and therefore has no wire form.
The two do not fork on substance: both resolve identity through
:func:`~mosaic.core.pipeline.run.resolve_feature_identity`, so there is one
answer to "what ``run_id`` will this step have" -- and the notebooks drive
``Pipeline``, so a second answer would be the one users meet first.

**Only :mod:`~mosaic.core.pipeline.graph.resolve` may import ``FEATURES``.**
Importing the feature library costs seconds of wall clock, and parsing a recipe,
ordering it, listing a step's parents, deciding a lane and rendering a status
view must all work without paying it -- otherwise a release gate that runs far
more often than a submit does acquires that floor, and so do the read endpoints
and cancel. ``tests/test_graph_imports.py`` holds the line.

This module is the only public import path; the submodules are its
implementation.
"""

from __future__ import annotations

from .claims import FileFailureStore, claims_root
from .compatibility import (
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    EntityLevel,
    ProducerDecl,
    TRACKS_DECLARATION,
    Verdict,
    can_connect,
    can_join,
    compatible_consumers,
    compatible_producers,
    possible_connections,
    resolve_emits,
)
from .digest import canonical_json, canonical_recipe, recipe_digest
from .failures import (
    QUARANTINE_AFTER,
    EntryFailureKey,
    Exclusions,
    FailureRecord,
    FailureStore,
    StepFailureKey,
)
from .lanes import (
    DEFAULT_LANE,
    GPU_INFER_LANE,
    GPU_TRAIN_LANE,
    lane_for,
    resource_class_of,
)
from .plan import (
    COMPLETE_STATUSES,
    MISSING_SAMPLE,
    ArtifactView,
    CoverageShort,
    DepsIncomplete,
    HeldOnParents,
    IdentityUnresolved,
    Plan,
    PlannedStep,
    Reason,
    Stalled,
    WaitingOnResource,
    coverage_against,
    is_stalled,
    plan_pipeline,
)
from .preflight import (
    REFUSED_EXIT_CODE,
    CoverageShortfall,
    RefusalReason,
    StepRefused,
    preflight,
    refuse_mixed_schemas,
)
from .request import (
    SubmittedRequest,
    load_recipe_for_request,
    step_argv,
    submit_request,
)
from .model import (
    SCHEMA_VERSION,
    BoundRef,
    FeatureStepSpec,
    OpStepSpec,
    Recipe,
    Request,
    Step,
    StepRef,
    StepRun,
    TRACKS_INPUT,
    params_step_refs,
)
from .resolve import (
    ReferenceSite,
    ResolvedStep,
    StepBuildError,
    StepSpec,
    build_feature,
    build_op_params,
    build_step_feature,
    build_step_op_params,
    declaration_catalog,
    declared_version,
    feature_class_for_slug,
    op_class_for_kind,
    params_reference_site,
    resolve_step_spec,
)
from .rollup import RequestRollup, RequestStatus, StepAttempt, request_rollup
from .run import PipelineRun, run_pipeline
from .step import STEP_RUN_KIND, StepOutcome, asked_of, execute_step
from .scope import graph_writes_tracks, intended_scope, media_universe
from .storage import storage_name_of
from .store import (
    load_recipe,
    load_request,
    pipelines_root,
    recipe_path,
    request_path,
    requests_root,
    save_recipe,
    save_request,
)
from .topo import (
    Edge,
    RecipeCycle,
    ancestors_of,
    children_of,
    descendants_of,
    edges,
    parents_of,
    topological_order,
)
from .validate import (
    EXCLUDED_KINDS,
    OVERWRITE_PARAM,
    Problem,
    RecipeInvalid,
    check_recipe,
    reject_unless_valid,
)

__all__ = [
    "ArtifactView",
    "BoundRef",
    "COMPLETE_STATUSES",
    "ConsumerDecl",
    "CoverageShort",
    "CoverageShortfall",
    "DEFAULT_LANE",
    "Declaration",
    "DeclarationCatalog",
    "DepsIncomplete",
    "EXCLUDED_KINDS",
    "Edge",
    "EntityLevel",
    "EntryFailureKey",
    "Exclusions",
    "FailureRecord",
    "FailureStore",
    "FeatureStepSpec",
    "FileFailureStore",
    "GPU_INFER_LANE",
    "GPU_TRAIN_LANE",
    "HeldOnParents",
    "IdentityUnresolved",
    "MISSING_SAMPLE",
    "OVERWRITE_PARAM",
    "OpStepSpec",
    "PipelineRun",
    "Plan",
    "PlannedStep",
    "Problem",
    "ProducerDecl",
    "QUARANTINE_AFTER",
    "REFUSED_EXIT_CODE",
    "Reason",
    "Recipe",
    "RecipeCycle",
    "RecipeInvalid",
    "ReferenceSite",
    "RefusalReason",
    "Request",
    "RequestRollup",
    "RequestStatus",
    "ResolvedStep",
    "SCHEMA_VERSION",
    "STEP_RUN_KIND",
    "Stalled",
    "Step",
    "StepAttempt",
    "StepBuildError",
    "StepFailureKey",
    "StepOutcome",
    "StepRef",
    "StepRefused",
    "StepRun",
    "StepSpec",
    "SubmittedRequest",
    "TRACKS_DECLARATION",
    "TRACKS_INPUT",
    "Verdict",
    "WaitingOnResource",
    "ancestors_of",
    "asked_of",
    "build_feature",
    "build_op_params",
    "build_step_feature",
    "build_step_op_params",
    "can_connect",
    "can_join",
    "canonical_json",
    "canonical_recipe",
    "check_recipe",
    "children_of",
    "claims_root",
    "compatible_consumers",
    "compatible_producers",
    "coverage_against",
    "declaration_catalog",
    "declared_version",
    "descendants_of",
    "edges",
    "execute_step",
    "feature_class_for_slug",
    "graph_writes_tracks",
    "intended_scope",
    "is_stalled",
    "lane_for",
    "load_recipe",
    "load_recipe_for_request",
    "load_request",
    "media_universe",
    "op_class_for_kind",
    "params_reference_site",
    "params_step_refs",
    "parents_of",
    "pipelines_root",
    "plan_pipeline",
    "possible_connections",
    "preflight",
    "recipe_digest",
    "recipe_path",
    "refuse_mixed_schemas",
    "reject_unless_valid",
    "request_path",
    "request_rollup",
    "requests_root",
    "resolve_emits",
    "resolve_step_spec",
    "resource_class_of",
    "run_pipeline",
    "save_recipe",
    "save_request",
    "step_argv",
    "storage_name_of",
    "submit_request",
    "topological_order",
]
