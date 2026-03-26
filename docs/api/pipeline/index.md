# Pipeline

The composable pipeline infrastructure that orchestrates feature extraction,
frame sampling, and model training. See the
[Pipeline Architecture](../../pipeline.md) guide for design details.

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
