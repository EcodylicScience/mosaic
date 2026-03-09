# Migration guide: mosaic feature API

This covers changes to `mosaic.behavior.feature_library` and how notebook/script
code needs to update. Reference implementation: `notebooks/calms21-template.ipynb`.

## Imports

Features and params are now re-exported from `__init__.py`:

```python
# before
from mosaic.behavior import feature_library
feature_library.pairposedistancepca.PairPoseDistancePCA(params=...)
feature_library.global_tsne.GlobalTSNE(params=...)

# after
from mosaic.behavior.feature_library import (
    GlobalTSNE, GlobalKMeansClustering, PairWavelet, Inputs, ...
)
GlobalTSNE(Inputs(...), params=...)
```

Same for models and visualizations:

```python
from mosaic.behavior.model_library import BehaviorXGBoostModel
from mosaic.behavior.visualization_library import VizGlobalColored
```

## Inputs replace input_kind/input_feature/input_run_id

The old `run_feature()` keyword arguments `input_kind`, `input_feature`,
`input_run_id` are replaced by typed `Inputs` passed to the feature constructor.

```python
# before
feat = PairWavelet(params=wavelet_params)
run = dataset.run_feature(feat, input_kind="feature",
                          input_feature="pair-posedistance-pca",
                          input_run_id=pose_run, ...)

# after
feat = PairWavelet(Inputs((pose_result,)), params=wavelet_params)
result = dataset.run_feature(feat, ...)
```

`Inputs` accepts a tuple of:
- `"tracks"` -- raw track data (the default for single-track features)
- `Result(feature=..., run_id=...)` -- output of a previous `run_feature()` call

Features that only accept tracks default to `Inputs(("tracks",))`. Features that
load their own data (global-ward, viz features) default to `Inputs(())`.

### Multiple inputs replace inputsets

The old `save_inputset()` + `input_kind="inputset"` pattern is replaced by
passing multiple Results:

```python
# before
save_inputset(dataset, "social+ego", [...])
tsne = GlobalTSNE(params={"inputset": "social+ego", ...})
run = dataset.run_feature(tsne, input_kind="inputset",
                          input_feature="social+ego", ...)

# after
tsne = GlobalTSNE(
    Inputs((social_wave_result, ego_wave_result)),
    params={"total_templates": 2000, "perplexity": 50},
)
result = dataset.run_feature(tsne, ...)
```

Feature names and run directories are derived automatically:
`global-tsne__from__pair-wavelet__from__pair-posedistance-pca+pair-wavelet__from__pair-egocentric`.

### use_latest()

To reference the latest run of a feature (same as `run_id=None` before):

```python
Inputs((pose_result.use_latest(),))
```

## Artifacts replace raw dicts

Features that produce non-per-frame outputs (models, templates, cluster centers)
define typed artifact classes. These replace hand-written dicts with `feature`,
`run_id`, `pattern`, `load` keys.

```python
# before
params={
    "artifact": {
        "feature": tsne_feature,
        "run_id": tsne_run,
        "pattern": "global_templates_features.npz",
        "load": {"kind": "npz", "key": "templates"},
    },
    "assign": {"scaler": {"feature": tsne_feature, ...}},
}

# after
params={
    "templates": GlobalTSNE.TemplatesArtifact.from_result(tsne_result),
    "scaler": GlobalTSNE.ScalerArtifact.from_result(tsne_result),
}
```

Each artifact class defines its own `pattern` and `load` spec, so the caller
only needs to specify which run to load from.

## Label sources

Visualization features accept typed label sources instead of raw dicts:

```python
# before -- ground truth labels
"labels": {"source": "labels", "kind": "behavior",
           "load": {"kind": "npz", "key": "labels"}}

# after
from mosaic.behavior.feature_library import GroundTruthLabelsSource
"labels": GroundTruthLabelsSource()  # defaults match the above
```

```python
# before -- labels from a feature artifact
"labels": {"feature": kmeans_feature, "run_id": kmeans_run_id,
           "pattern": "global_kmeans_labels_seq=*.npz",
           "load": {"kind": "npz", "key": "labels"}}

# after
"labels": GlobalKMeansClustering.SeqLabelsArtifact.from_result(k_result)
```

## Params are typed Pydantic models

Feature params are now `Params` subclasses (inheriting from `DictModel`).
Constructors accept `dict` overrides that get validated:

```python
# still works -- dict overrides merged with field defaults
feat = PairWavelet(Inputs((pose_result,)), params={"f_min": 0.2, ...})

# params object is a Pydantic model, not a dict
feat.params.f_min  # attribute access
feat.params["f_min"]  # dict-style also works (DictModel compatibility)
feat.params.model_dump()  # serialize to dict
```

Nested model fields support partial overrides:

```python
# only override pose_n, other PoseConfig fields keep defaults
PairPoseDistancePCA(params={"pose": {"pose_n": 7}})
```

## run_feature() return value

`run_feature()` now returns a `Result` dataclass instead of a raw `run_id` string.
This is why the "after" examples throughout this guide use `result = dataset.run_feature(...)`
instead of `run = ...` -- the variable holds a `Result`, not a plain run ID.

```python
# before
run = dataset.run_feature(feat, ...)  # str (a run_id)
feature_name = f"pair-posedistance-pca"

# after
result = dataset.run_feature(feat, ...)  # Result
result.feature  # "pair-posedistance-pca"
result.run_id   # "0.1-b1933f9f3d"
```

Results can be passed directly to `Inputs()` and `from_result()`.

## Frame/time filtering is now on run_feature()

The old `save_inputset()` accepted `filter_start_frame`, `filter_end_frame`,
`filter_start_time`, `filter_end_time` kwargs that were stored in the inputset
JSON. These filters are now direct parameters on `run_feature()`:

```python
# before -- filters embedded in inputset metadata
save_inputset(dataset, "social+ego", [...],
              filter_start_frame=100, filter_end_frame=5000)
run = dataset.run_feature(tsne, input_kind="inputset",
                          input_feature="social+ego", ...)

# after -- direct params on run_feature()
result = dataset.run_feature(feat, groups=GROUP_SCOPE,
                             filter_start_frame=100,
                             filter_end_frame=5000)
```

Time-based filters work the same way and require `fps_default` in the dataset
manifest metadata:

```python
result = dataset.run_feature(feat,
                             filter_start_time=1.0,
                             filter_end_time=50.0)
```

Frame and time filters are mutually exclusive per boundary -- you can't set both
`filter_start_frame` and `filter_start_time` (raises `ValueError`). Mixing is
fine: `filter_start_frame=100, filter_end_time=50.0`.

Semantics: `start` is inclusive (`>=`), `end` is exclusive (`<`).

### Nearest-neighbor pair filter moved to params

The `pair_filter` dict that was stored in inputset metadata is now a typed
parameter on feature `Params`. Unlike frame/time filters (which set data
scope), pair filtering controls *which* pairs are used in the embedding -- an
algorithmic choice, not a data range, so it belongs in feature params.

Features with `pair_filter: NNResult | None` on their `Params`:
`GlobalTSNE`, `GlobalKMeansClustering`, `GlobalWard`, `WardAssign`,
`TemporalStacking`.

```python
# before -- pair_filter in inputset JSON metadata
save_inputset(dataset, "social+ego", [...],
              pair_filter={"type": "nearest_neighbor",
                           "feature": "nearest-neighbor",
                           "run_id": nn_run})

# after -- typed NNResult on feature params
from mosaic.behavior.feature_library import NNResult

tsne = GlobalTSNE(
    Inputs((social_wave_result, ego_wave_result)),
    params={
        "pair_filter": NNResult(run_id=nn_result.run_id),
    },
)
```

## XGBoost model: undersample changes

`use_undersample` is removed. Use `foreground_samples` and `undersample_ratio`:

```python
# before
"use_undersample": True,
"undersample_ratio": 3.0,

# after
"foreground_samples": 500,   # cap minority class to N samples (None = all)
"undersample_ratio": 3.0,    # majority/minority ratio (1.0 = no undersampling)
```
