# Pipeline Guide

The `Pipeline` class orchestrates multi-step feature computation with
automatic caching and dependency tracking. Instead of manually chaining
`ds.run_feature()` calls, you declare a graph of named steps and let the
pipeline handle execution order, caching, and staleness propagation.

## Quick start

```python
from mosaic.core.pipeline import Pipeline, FeatureStep, CallbackStep
from mosaic.behavior.feature_library import (
    TrajectorySmooth, SpeedAngvel, FFGroups, Inputs,
)

pipe = Pipeline(default_run_kwargs={"parallel_workers": 8})

pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5}))
pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))
pipe.add(FeatureStep("ff", FFGroups, {"window_size": 20}, ["smooth"]))

pipe.status(dataset)         # check what's cached
results = pipe.run(dataset)  # execute — cached steps are skipped
```

## Concepts

### Steps

A pipeline is a sequence of **steps**, each with a unique name. There are
two kinds:

- **`FeatureStep`** — wraps a Feature class. Produces a `Result` that
  downstream steps can reference by name.
- **`CallbackStep`** — runs an arbitrary function between feature layers
  (e.g. computing labels or writing summaries).

### Dependency wiring

`FeatureStep.input_names` lists the upstream step names whose `Result`
objects are wired into the feature's `Inputs`. An empty list (the default)
means the feature reads directly from tracks:

```python
# Reads from tracks (no upstream dependency)
pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5}))

# Reads from the "smooth" step's output
pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))

# Multi-input: reads from smooth + ff + speed
pipe.add(FeatureStep("metrics", FFGroupsMetrics, {}, ["smooth", "ff", "speed"]))
```

### Caching

Each step's `run_id` is a deterministic hash of its parameters, inputs,
and frame range. If the output directory already exists with parquet files,
the step is skipped. Staleness propagates: if any upstream step is not
cached, all downstream steps are treated as stale regardless of whether
their own files exist.

### Run kwargs

`default_run_kwargs` are passed to every `ds.run_feature()` call. Per-step
overrides are set via `FeatureStep.run_kwargs`:

```python
pipe = Pipeline(default_run_kwargs={
    "parallel_workers": 8,
    "parallel_mode": "process",
    "filter_start_time": 3600.0,  # skip first hour
})

# This step overrides the worker count and removes the time filter
pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5},
                     run_kwargs={"parallel_workers": 10, "filter_start_time": None}))
```

## API

### `pipe.add(step)`

Register a step. Returns `self` for chaining. Validates that the step name
is unique and that all `input_names` / `depends_on` reference earlier steps.

### `pipe.status(dataset)`

Returns a DataFrame showing each step's cache state:

| step | feature | run_id | n_seq | runs | cached |
|------|---------|--------|-------|------|--------|
| smooth | trajectory-smooth | 0.1-a3f2b1c8e9 | 41 | 1 | True |
| speed | speed-angvel | 0.1-7e2c4f9a01 | 41 | 1 | True |
| ff | ff-groups | 0.1-b8d3e5f120 | 0 | 0 | False |

### `pipe.load(dataset)`

Populate `pipe.results` from cached runs on disk without executing
anything. Returns `dict[str, Result]`. Useful for loading prior results
in analysis notebooks.

### `pipe.run(dataset, force_from=None)`

Execute the pipeline. Cached steps are skipped. Returns `dict[str, Result]`.

Use `force_from="step_name"` to force recomputation from a specific step
and all its downstream dependents:

```python
# Recompute everything from "speed" onward
results = pipe.run(dataset, force_from="speed")
```

### `pipe.dag()`

Returns an adjacency dict `{step_name: [upstream_names]}` for
visualization or analysis.

## Callbacks

Use `CallbackStep` to run arbitrary code between feature layers — for
example, computing labels that downstream features need:

```python
def compute_labels(dataset, results):
    """Read metrics output, compute quantile labels, save to dataset."""
    ...

pipe.add(FeatureStep("metrics", FFGroupsMetrics, {}, ["smooth", "ff", "speed"]))
pipe.add(CallbackStep("labels", compute_labels, depends_on=["metrics"]))
pipe.add(FeatureStep("tagged", IdTagColumns, {"label_kind": "iso"}, ["smooth"],
                     run_kwargs={"parallel_workers": 1}))
```

The callback receives `(dataset, results_so_far)` where `results_so_far`
is the dict of all `Result` objects from previously executed steps.

## Feature Registry

When `Pipeline.run()` executes, it automatically creates a SQLite registry
at `features/.mosaic.db` within the dataset. This registry tracks all
feature runs, entries, and dependencies in a single queryable database.

You can also use the registry directly:

```python
from mosaic.core.pipeline import feature_registry

reg = feature_registry(dataset)

# What features have been computed?
reg.list_features()

# What entries exist for a feature?
reg.list_entries("speed-angvel__from__tracks")

# What's missing?
all_tracks = {("group1", "seq1"), ("group1", "seq2"), ("group1", "seq3")}
reg.pending_entries("speed-angvel__from__tracks", "0.1-abc123", all_tracks)

# Dependency lineage
reg.lineage("ff-groups-metrics__from__tracks+ff-groups+speed-angvel", "0.1-xyz789")

reg.close()
```

The registry replaces per-feature `index.csv` files with a single SQLite
database. Existing CSV indices are automatically migrated on first access.
The database uses WAL mode for safe concurrent reads.
