# Migration guide: mosaic feature API

This covers changes to `mosaic.behavior.feature_library` and how notebook/script
code needs to update. Reference implementation: `notebooks/calms21-template.ipynb`.

## Imports

Features and types are re-exported from `__init__.py`:

```python
# before
from mosaic.behavior import feature_library
feature_library.pairposedistancepca.PairPoseDistancePCA(params=...)
feature_library.global_tsne.GlobalTSNE(params=...)

# after
from mosaic.behavior.feature_library import (
    GlobalTSNE, GlobalKMeansClustering, PairWavelet,
    ExtractTemplates, GlobalScaler, ExtractLabeledTemplates, XgboostFeature,
    Inputs, Result, ResultColumn, GroundTruthLabelsSource, ...
)
```

`BehaviorXGBoostModel` is deprecated. Use `XgboostFeature` instead (see below).
`VizTimeline` and `VizGlobalColored` are removed. Use `load_values()` instead.

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

Features that only accept tracks default to `Inputs(("tracks",))`.

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
    params={"perplexity": 50, ...},
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

## Global features are composable stages

GlobalTSNE no longer handles template extraction and scaling internally.
These are now separate features chained together:

```python
# 1. Extract templates from per-sequence features
templates = ExtractTemplates(
    Inputs((social_wave_result, ego_wave_result)),
    params={"n_templates": 2000},
)
templates_result = dataset.run_feature(templates)

# 2. Fit scaler on templates, scale per-sequence data
scaler = GlobalScaler(
    Inputs((social_wave_result, ego_wave_result)),
    params={
        "templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result),
    },
)
scaler_result = dataset.run_feature(scaler)

# 3. Extract templates from scaled data (for embedding/clustering)
scaled_templates = ExtractTemplates(
    Inputs((scaler_result,)),
    params={"n_templates": 2000, "strategy": "farthest_first"},
)
scaled_templates_result = dataset.run_feature(scaled_templates)

# 4. Fit t-SNE on scaled templates
tsne = GlobalTSNE(
    Inputs((scaled_templates_result,)),
    params={
        "perplexity": 50,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
)
tsne_result = dataset.run_feature(tsne)
```

Each stage caches independently. Re-running with different t-SNE perplexity
reuses the existing scaler and templates.

### Reusing a fitted model (skip fit)

Global features that extend `GlobalModelParams` accept either `templates`
(fit from scratch) or `model` (load a previously fitted model). These are
mutually exclusive -- provide exactly one.

```python
# Fit from templates (default workflow above)
tsne = GlobalTSNE(
    Inputs((scaled_templates_result,)),
    params={
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
        "perplexity": 50,
    },
)
tsne_result = dataset.run_feature(tsne)

# Reuse the fitted model on different data (skip fit, apply only)
tsne_reuse = GlobalTSNE(
    Inputs((other_scaler_result,)),
    params={
        "model": GlobalTSNE.TSNEModelArtifact().from_result(tsne_result),
        "perplexity": 50,  # used during apply for mapping new points
    },
)
dataset.run_feature(tsne_reuse)
```

This works for all global features: `GlobalScaler` (`ScalerModelArtifact`),
`GlobalTSNE` (`TSNEModelArtifact`), `GlobalKMeansClustering`
(`KMeansModelArtifact`), `GlobalWardClustering` (`WardModelArtifact`),
and `XgboostFeature` (`XgboostModelArtifact`).

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
}

# after
params={
    "templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result),
}
```

Each artifact class defines its own `pattern` and `load` spec, so the caller
only needs to specify which run to load from.

## Label sources

Ground truth labels use a typed source:

```python
# before
"labels": {"source": "labels", "kind": "behavior",
           "load": {"kind": "npz", "key": "labels"}}

# after
from mosaic.behavior.feature_library import GroundTruthLabelsSource
GroundTruthLabelsSource()  # defaults match the above
```

## Params are typed Pydantic models

Feature params are now `Params` subclasses (Pydantic `BaseModel` with `extra="forbid"`).
Constructors accept `dict` overrides that get validated:

```python
# still works -- dict overrides merged with field defaults
feat = PairWavelet(Inputs((pose_result,)), params={"f_min": 0.2, ...})

# params object is a Pydantic model
feat.params.f_min         # attribute access
feat.params.model_dump()  # serialize to dict
```

Nested model fields support partial overrides:

```python
# only override pose_n, other PoseConfig fields keep defaults
PairPoseDistancePCA(params={"pose": {"pose_n": 7}})
```

## run_feature() return value

`run_feature()` returns a `Result` dataclass instead of a raw `run_id` string:

```python
# before
run = dataset.run_feature(feat, ...)  # str (a run_id)

# after
result = dataset.run_feature(feat, ...)  # Result
result.feature  # "pair-posedistance-pca"
result.run_id   # "0.1-b1933f9f3d"
```

Results can be passed directly to `Inputs()` and `from_result()`.

## Frame/time filtering is now on run_feature()

Frame/time filters are direct parameters on `run_feature()`:

```python
# before -- filters embedded in inputset metadata
save_inputset(dataset, "social+ego", [...],
              filter_start_frame=100, filter_end_frame=5000)
run = dataset.run_feature(tsne, input_kind="inputset",
                          input_feature="social+ego", ...)

# after
result = dataset.run_feature(feat,
                             filter_start_frame=100,
                             filter_end_frame=5000)
```

Frame and time filters are mutually exclusive per boundary -- you can't set both
`filter_start_frame` and `filter_start_time` (raises `ValueError`). Mixing is
fine: `filter_start_frame=100, filter_end_time=50.0`.

Semantics: `start` is inclusive (`>=`), `end` is exclusive (`<`).

### Nearest-neighbor pair filter moved to params

The `pair_filter` dict that was stored in inputset metadata is now a typed
parameter on feature `Params`:

```python
# before -- pair_filter in inputset JSON metadata
save_inputset(dataset, "social+ego", [...],
              pair_filter={"type": "nearest_neighbor", ...})

# after -- typed NNResult on feature params
tsne = GlobalTSNE(
    Inputs((social_wave_result, ego_wave_result)),
    params={
        "pair_filter": nn_result,
    },
)
```

`NNResult` is a `Result` subclass narrowed to only accept `"nearest-neighbor"`
feature results. You can pass an existing result directly, pin a specific
`run_id` with `NNResult(run_id="...")`, or use `nn_result.use_latest()` to
always resolve to the most recent run.

Features with `pair_filter` on their Params: `GlobalKMeansClustering`,
`GlobalWardClustering`, `TemporalStackingFeature`.

## load_values() replaces visualization features

`VizTimeline` and `VizGlobalColored` are removed. Use `load_values()` to load
any combination of feature columns, track columns, and ground truth labels into
a single DataFrame:

```python
from mosaic.core.pipeline import load_values

df = load_values(
    dataset,
    [
        ResultColumn(column="tsne_x").from_result(tsne_result),
        ResultColumn(column="tsne_y").from_result(tsne_result),
        ResultColumn(column="cluster").from_result(k_result),
        GroundTruthLabelsSource(),
    ],
)
# df has columns: group, sequence, frame, id1, id2, tsne_x, tsne_y, cluster, labels-behavior
```

All visualization and analysis happens in notebook code using standard pandas/matplotlib.

## XGBoost: model_library to feature_library

`BehaviorXGBoostModel` + `ModelPredictFeature` are replaced by two composable
features: `ExtractLabeledTemplates` and `XgboostFeature`.

```python
# before
from mosaic.behavior.model_library import BehaviorXGBoostModel

xgb_model = BehaviorXGBoostModel()
xgb_model.bind_dataset(dataset)
xgb_model.configure({
    "feature": ts_stack_result.feature,
    "feature_run_id": ts_stack_result.run_id,
    "label_kind": "behavior",
    "train_sequences": train_seqs,
    "test_sequences": test_seqs,
    "standardize": True,
    "foreground_samples": 500,
    "undersample_ratio": 3.0,
    "xgb_params": {"n_estimators": 10},
}, run_root)
xgb_model.train()

# prediction required a separate ModelPredictFeature wrapper
predict_feat = ModelPredictFeature(
    Inputs((ts_stack_result,)),
    params={"model_class": "...BehaviorXGBoostModel", ...},
)
pred_result = dataset.run_feature(predict_feat)

# after
from mosaic.behavior.feature_library import (
    ExtractLabeledTemplates, XgboostFeature,
    GroundTruthLabelsSource, Inputs,
)

# 1. Extract labeled templates (handles label alignment + train/test split)
labeled_templates = ExtractLabeledTemplates(
    Inputs((ts_stack_result,)),
    params={
        "labels": GroundTruthLabelsSource(),
        "n_per_class": 500,
        "test_fraction": 0.2,
    },
)
labeled_templates_result = dataset.run_feature(labeled_templates)

# 2. Train + predict in one step (fit on templates, apply per-sequence)
xgb = XgboostFeature(
    Inputs((ts_stack_result,)),
    params={
        "templates": ExtractLabeledTemplates.LabeledTemplatesArtifact()
            .from_result(labeled_templates_result),
        "strategy": "multiclass",
        "default_class": 3,
        "n_estimators": 10,
        "max_depth": 3,
    },
)
xgb_result = dataset.run_feature(xgb)
```

Key differences:
- No manual train/test sequence lists. `ExtractLabeledTemplates` assigns splits
  automatically (deterministic, guarantees at least 1 test sequence).
- No internal scaling. Use `GlobalScaler` upstream.
- No `ModelPredictFeature` wrapper. `XgboostFeature.apply()` runs inference
  directly during `run_feature()`.
- `ExtractLabeledTemplates.apply()` adds `label` and `split` columns to
  per-sequence outputs, so you can filter to test-only data:

```python
df = load_values(dataset, [
    ResultColumn(column="predicted_label").from_result(xgb_result),
    ResultColumn(column="split").from_result(labeled_templates_result),
    GroundTruthLabelsSource(),
])
df_test = df[df["split"] == "test"]
```
