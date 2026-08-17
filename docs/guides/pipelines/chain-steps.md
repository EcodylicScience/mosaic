# Chain steps into a recipe

A **step** is one unit of work with an identity. It may be a **feature** or it may be
an **op** — a tracker run, a conversion, a transcode. That is the whole point of this
section: a recipe is not an analysis chain with tracking bolted on the front, it is
one graph in which both kinds of step are ordinary members.

There are two ways to spell the same graph, and they resolve identity through one
shared site so they cannot disagree about what a step will be called.

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
     "params": {"templates": {"step": "templates",
                              "pattern": "templates.parquet"},
                "perplexity": 50}}
  ]
}
```

Five steps spanning all three stages: transcode the originals, track the derivative
with TRex, derive speeds from the tracks that produced, sample templates, fit an
embedding.

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
| `"params": {"f": {"step": "x", "pattern": "*.parquet"}}` | Take a named artifact from that step |
| `"after": ["transcode"]` | Ordering only, no data reference |

A reference sits at the exact place it substitutes, so there is no separate edge list
to drift from the step bodies.

`after` is for the case where one step must precede another without reading anything
from it — transcode before trex is the standard example: the tracker reads the
derivative from the media index, not from the transcode step's return value.

## `Pipeline`, when the graph is code you are editing

A live object holding feature *classes*, with a `CallbackStep` escape hatch for a
plain function between layers. Reach for it in a notebook. It cannot express op steps
— for those, use a recipe file.

```python
from mosaic.core.pipeline import Pipeline, FeatureStep
from mosaic.behavior.feature_library import TrajectorySmooth, SpeedAngvel, FFGroups

pipe = Pipeline(default_run_kwargs={"parallel_workers": 8})
pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5}))
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

See [the Pipeline API](../../api/pipeline/index.md) for every method and signature.

### Per-step overrides

`default_run_kwargs` reach every `run_feature` call; a step overrides them
individually.

```python
pipe = Pipeline(default_run_kwargs={
    "parallel_workers": 8,
    "parallel_mode": "process",
    "filter_start_time": 3600.0,   # skip the first hour
})

pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5},
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
