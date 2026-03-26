# Pipeline

The composable pipeline infrastructure that orchestrates feature extraction,
frame sampling, and model training. See the
[Pipeline Architecture](../../pipeline.md) guide for design details and the
[Pipeline Guide](../../guide-pipeline.md) for practical usage.

## Pipeline Orchestrator

Declarative multi-step feature pipeline with automatic caching, staleness
detection, and dependency chaining.

::: mosaic.core.pipeline.pipeline.Pipeline
    options:
      show_source: true
      members_order: source

::: mosaic.core.pipeline.pipeline.FeatureStep
    options:
      show_source: true

::: mosaic.core.pipeline.pipeline.CallbackStep
    options:
      show_source: true

## Feature Registry

SQLite-backed registry replacing per-feature CSV indices. Enables cross-feature
queries, pending-work detection, dependency tracking, and API-friendly reads.

::: mosaic.core.pipeline.registry.FeatureRegistry
    options:
      show_source: true
      members_order: source

::: mosaic.core.pipeline.registry.open_registry
    options:
      show_source: true

## Feature Protocol

Features implement four methods plus four class attributes:

```python
class MyFeature:
    name: str                  # Unique feature name
    version: str               # Semantic version string
    parallelizable: bool       # Whether apply() can run in parallel
    scope_dependent: bool      # Whether run_id includes manifest scope

    def load_state(self, run_root, artifact_paths, dependency_lookups) -> bool: ...
    def fit(self, inputs: InputStream) -> None: ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def save_state(self, run_root) -> None: ...
```

## Pipeline Types

::: mosaic.core.pipeline.types
    options:
      show_source: true
      members_order: source

## Pipeline Runner

::: mosaic.core.pipeline.run
    options:
      show_source: true
      members:
        - run_feature
