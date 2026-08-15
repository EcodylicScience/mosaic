"""The pipeline as a file: a recipe, a request beside it, and what to run.

A **recipe** is the pipeline -- a JSON graph identified by content digest,
diffable, shareable, portable across datasets, and never holding a resolved
``run_id``, because those are dataset state. A **plan** is what
:func:`~mosaic.core.pipeline.graph.plan.plan_pipeline` returns for a given
dataset and recipe: what to run, under what identity, and what is already done.

Deliberately separate from the live-object
:class:`~mosaic.core.pipeline.pipeline.Pipeline`, which holds feature *classes*
and a ``CallbackStep`` wrapping a live callable and therefore has no wire form.
The two do not fork on substance: ``Pipeline`` is a thin caller of
``plan_pipeline`` over a recipe built in memory, so there is one answer to "what
``run_id`` will this step have".

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
from .lanes import (
    DEFAULT_LANE,
    GPU_INFER_LANE,
    GPU_TRAIN_LANE,
    lane_for,
    resource_class_of,
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
    ResolvedStep,
    StepSpec,
    declaration_catalog,
    feature_class_for_slug,
    resolve_step_spec,
)
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

__all__ = [
    "BoundRef",
    "ConsumerDecl",
    "DEFAULT_LANE",
    "Declaration",
    "DeclarationCatalog",
    "Edge",
    "EntityLevel",
    "FeatureStepSpec",
    "GPU_INFER_LANE",
    "GPU_TRAIN_LANE",
    "OpStepSpec",
    "ProducerDecl",
    "Recipe",
    "RecipeCycle",
    "Request",
    "ResolvedStep",
    "SCHEMA_VERSION",
    "Step",
    "StepRef",
    "StepRun",
    "StepSpec",
    "TRACKS_DECLARATION",
    "TRACKS_INPUT",
    "Verdict",
    "ancestors_of",
    "can_connect",
    "can_join",
    "canonical_json",
    "canonical_recipe",
    "children_of",
    "compatible_consumers",
    "compatible_producers",
    "declaration_catalog",
    "descendants_of",
    "edges",
    "feature_class_for_slug",
    "graph_writes_tracks",
    "intended_scope",
    "lane_for",
    "load_recipe",
    "load_request",
    "media_universe",
    "params_step_refs",
    "parents_of",
    "pipelines_root",
    "possible_connections",
    "recipe_digest",
    "recipe_path",
    "request_path",
    "requests_root",
    "resolve_emits",
    "resolve_step_spec",
    "resource_class_of",
    "save_recipe",
    "save_request",
    "storage_name_of",
    "topological_order",
]
