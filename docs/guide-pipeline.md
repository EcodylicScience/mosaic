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

## What-ran & run status

There is no database on the dataset filesystem. What-ran is recorded in plain
files — the permanent source of truth that works on any filesystem (NFS, HPC,
external drives):

- **Results & what-ran** live in each feature's `features/<name>/index.csv` (one
  row per computed entry: `run_id`, `version`, `params_hash`, `group`,
  `sequence`, `abs_path`, `finished_at`) plus the parquet outputs themselves.
  Cache-hit and completeness are decided by globbing those parquet files on disk,
  never by a status flag.

```python
from mosaic.core.pipeline import list_feature_runs

# What runs exist for a feature? (reads index.csv)
list_feature_runs(dataset, "speed-angvel__from__tracks")
```

- **Attempt status & progress** live in one append-only JSONL run-log per
  attempt, under `<dataset_root>/.mosaic/runs/<execution_id>.jsonl`. Each unit of
  compute that runs under the Job Contract (features, tracking ops, TREx, and
  future payloads) records its lifecycle (`started` → `finished` / `failed` /
  `cancelled`), a liveness heartbeat, and coarse per-entry / per-epoch progress
  there. It is job-kind-agnostic (the `kind` is a field in the log, not part of
  the path) and NFS-safe (one writer, append-only). These files are ephemeral —
  bounded by the work and safe to age out.

Read attempt status from the CLI or the stdlib-only reader helpers:

```bash
mosaic runs   -m dataset.yaml --kind feature --json
mosaic status -m dataset.yaml --execution-id <ULID> --progress --json
```

```python
from mosaic.core.pipeline.run_log import read_runs, read_run, run_log_dir

run_dir = run_log_dir(dataset.base_dir)
read_runs(run_dir, kind="feature")          # newest-first attempt snapshots
read_run(run_dir, "<execution_id>")         # one attempt's status + progress counters
```
