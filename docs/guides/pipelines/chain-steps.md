# Chain steps into a recipe

A **step** is one unit of work with an identity. It may be a **feature**, or an
**op**: a tracker run, a conversion, a transcode. Both kinds are ordinary members of
one graph, so a recipe can run a whole workflow from video to embedding.

There are two ways to spell the same graph.

## A recipe file, when the graph outlives the session

JSON, portable across datasets, checked before it runs, and drivable from
`mosaic pipeline`. This is the spelling that can hold op steps.

```json
{
  "schema_version": 1,
  "name": "trex to global tsne",
  "steps": [
    {"id": "transcode", "type": "op", "kind": "transcode",
     "params": {"target": "analysis"}},

    {"id": "trex", "type": "op", "kind": "trex",
     "params": {"track_max_individuals": 4, "cm_per_pixel": 0.05},
     "after": ["transcode"]},

    {"id": "speed", "type": "feature", "feature": "speed-angvel",
     "inputs": ["tracks"], "tracks": {"step": "trex"}},

    {"id": "templates", "type": "feature", "feature": "extract-templates",
     "inputs": [{"step": "speed"}], "params": {"n_templates": 2000}},

    {"id": "tsne", "type": "feature", "feature": "global-tsne",
     "inputs": [{"step": "speed"}],
     "params": {"templates": {"step": "templates"},
                "perplexity": 50}}
  ]
}
```

Five steps spanning all three stages: transcode the originals, track the derivative
with TRex, derive speeds from the tracks that produced, sample templates, fit an
embedding. The `tsne` step names no `pattern`: its `templates` field declares which
file it reads.

```bash
mosaic pipeline validate -m dataset.yaml --recipe @recipe.json
mosaic pipeline plan     -m dataset.yaml --recipe @recipe.json
mosaic pipeline run      -m dataset.yaml --recipe @recipe.json
```

`validate` runs before the dataset is touched and reports every problem rather than
the first. `plan` says what each step will be called and what is already done, without
submitting anything. On first use the recipe is copied to
`<dataset>/.mosaic/pipelines/<digest>.json`, so the dataset records which pipelines
were applied to it.

### The four ways a step references another

| Spelling | Means |
| --- | --- |
| `"inputs": [{"step": "speed"}]` | Read that step's feature output |
| `"tracks": {"step": "trex"}` | Read the tracks variant that op produced |
| `"params": {"f": {"step": "x"}}` | Take that step's artifact, named by the field |
| `"after": ["transcode"]` | Ordering only, no data reference |

A reference sits at the exact place it substitutes, so there is no separate edge list
to drift from the step bodies.

**`pattern` is optional, because the consumer's params field names the file.** A
producer's run directory holds one per-entry output parquet per sequence beside its
named artifacts, so a reference that resolved by glob would take whichever sorts
first — usually one of those per-entry outputs rather than the artifact you meant.
Nothing about that is visible downstream, so it is refused at both ends instead: a
reference that still resolves by glob is rejected by `validate`, before the producing
step has run, and a pattern matching more than one file is refused when it is
resolved. Spelling `pattern` stays legal and is how you reach a producer's other
artifacts — `template_provenance.parquet`, say, rather than `templates.parquet`.

`after` is for the case where one step must precede another without reading anything
from it — transcode before trex is the standard example: the tracker reads the
derivative from the media index, not from the transcode step's return value.

### `overwrite` is refused

A step's `params` may not carry `overwrite`. Validation rejects it on presence,
before the dataset is touched, because overwriting mutates content under a stable
address: a downstream reader gets a mixed read that its own `run_id` records nothing
about. Change the params instead, which gives the new work its own address.

## `Pipeline`, when the graph is code you are editing

A live object holding feature *classes*, with a `CallbackStep` escape hatch for a
plain function between layers. Reach for it in a notebook. It cannot express op steps
— for those, use a recipe file.

```python
from mosaic.core.pipeline import Pipeline, FeatureStep
from mosaic.behavior.feature_library import TrajectorySmooth, SpeedAngvel, FFGroups

pipe = Pipeline(default_run_kwargs={"parallel_workers": 8})
pipe.add(FeatureStep("smooth", TrajectorySmooth, {"savgol_window": 5}))
pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))
pipe.add(FeatureStep("ff", FFGroups, {"window_size": 20}, ["smooth"]))

pipe.status(dataset)         # what is already cached
results = pipe.run(dataset)  # execute; cached steps are skipped
```

The third argument lists upstream step names whose `Result` objects are wired into the
feature's `Inputs`. An empty list — the default — means the feature reads tracks
directly.

```python
pipe.add(FeatureStep("metrics", FFGroupsMetrics, {}, ["smooth", "ff", "speed"]))
```

`pipe.run(dataset, force_from="speed")` recomputes from a step onward, and
`pipe.load(dataset)` populates results from cached runs without executing anything —
which is how a notebook picks up where a previous session left off.

Every method and signature is documented in the class's own docstrings; see
[`core/pipeline/pipeline.py`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/pipeline/pipeline.py).

### Per-step overrides

`default_run_kwargs` reach every `run_feature` call; a step overrides them
individually.

```python
pipe = Pipeline(default_run_kwargs={
    "parallel_workers": 8,
    "parallel_mode": "process",
    "filter_start_time": 3600.0,   # skip the first hour
})

pipe.add(FeatureStep("smooth", TrajectorySmooth, {"savgol_window": 5},
                     run_kwargs={"parallel_workers": 10, "filter_start_time": None}))
```

### A function between layers

`CallbackStep` runs arbitrary code between feature layers — computing labels a
downstream feature needs, for instance. It receives `(dataset, results_so_far)`.

```python
def compute_labels(dataset, results):
    """Read metrics output, compute quantile labels, save to dataset."""
    ...

pipe.add(FeatureStep("metrics", FFGroupsMetrics, {}, ["smooth", "ff", "speed"]))
pipe.add(CallbackStep("labels", compute_labels, depends_on=["metrics"]))
```

## Caching across a chain

Each step's `run_id` is a deterministic hash of its parameters, inputs and frame
range, so a step whose output is already on disk is skipped. Staleness propagates: if
an upstream step is not cached, everything downstream is treated as stale regardless
of whether its own files exist.

That is why a chain is cheap to re-run and why changing one parameter re-runs only
what depended on it. [Reproducibility, run_id and
caching](../../concepts/reproducibility.md) has what this does and does not promise.

## Which spelling to reach for

Use `Pipeline` while you are still deciding what the analysis is. Move to a recipe
file when it needs to be reviewed, handed to somebody else, run on a machine with no
notebook, or extended with the tracking and media steps `Pipeline` cannot hold.
[Pipelines as documents](../../concepts/pipelines.md) explains why the file form is
the more capable of the two.
