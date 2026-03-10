# Mosaic Feature System Migration Guide

Summary of the major refactor to the mosaic feature system (commits `e788a2c` through `8c6f4b0`).

---

## What Changed at a Glance

| Aspect | Old | New |
|--------|-----|-----|
| **Feature creation** | `feature_library.module.Class(params={...})` | `Class(Inputs(...), params={...})` |
| **Input routing** | `run_feature(..., input_kind=, input_feature=, input_run_id=)` | `Inputs((Result(...),))` embedded on the feature |
| **InputSets** | `save_inputset()` + `input_kind="inputset"` | Multiple `Result` objects in an `Inputs` tuple |
| **Params** | Untyped dicts | Typed `Params` (Pydantic models) |
| **Artifacts** | Hand-written dicts | Typed `ArtifactSpec` subclasses |
| **run_feature() return** | String `run_id` | `Result(feature=..., run_id=...)` |
| **Frame/time filtering** | Not available | `filter_start_frame`, `filter_end_frame`, `filter_start_time`, `filter_end_time` |

---

## 1. Imports

Features and types are now re-exported from the top-level package:

```python
# Old
from mosaic.behavior import feature_library
feat = feature_library.speed_angvel.SpeedAngvel(params={...})

# New
from mosaic.behavior.feature_library import SpeedAngvel, Inputs, Result
feat = SpeedAngvel(params={...})
```

Other importable types: `ArtifactSpec`, `Params`, `COLUMNS`, `FeatureLabelsSource`, `GroundTruthLabelsSource`, `GlobalTSNE`, etc.

---

## 2. Input Routing (replaces `input_kind`, `input_feature`, `input_run_id`)

Inputs are now declared on the feature itself via an `Inputs` object, not passed as kwargs to `run_feature()`.

### Tracks input (simplest case, unchanged behavior)

```python
# Old
feat = SpeedAngvel(params={...})
run_id = dataset.run_feature(feat)

# New — tracks is the default for features that only accept tracks
feat = SpeedAngvel(params={...})
result = dataset.run_feature(feat)
```

### Single feature input

```python
# Old
feat = PairWavelet(params={"f_min": 0.2})
run_id = dataset.run_feature(
    feat,
    input_kind="feature",
    input_feature="pair-posedistance-pca",
    input_run_id=pose_run_id,
)

# New — pass the upstream Result directly
pose_result = dataset.run_feature(PairPoseDistancePCA(params=...))
feat = PairWavelet(
    Inputs((pose_result,)),
    params={"f_min": 0.2},
)
result = dataset.run_feature(feat)
```

### Multiple feature inputs (replaces InputSets)

```python
# Old — save_inputset + input_kind="inputset"
dataset.save_inputset("my_set", {...})
run_id = dataset.run_feature(feat, input_kind="inputset", input_set="my_set")

# New — just pass multiple Results in the Inputs tuple
tsne = GlobalTSNE(
    Inputs((social_wave_result, ego_wave_result)),
    params={"total_templates": 2000},
)
result = dataset.run_feature(tsne)
# Storage name auto-derived: "global-tsne__from__pair-wavelet+pair-egocentric"
```

**InputSets are removed.** All multi-input routing is handled by `Inputs`.

---

## 3. `run_feature()` Return Value

```python
# Old — returns a string run_id
run_id = dataset.run_feature(feat)  # "0.1-abc123"

# New — returns a Result dataclass
result = dataset.run_feature(feat)
result.feature   # "speed-angvel"
result.run_id    # "0.1-abc123"
```

`Result` objects are passed directly into `Inputs(...)` for downstream features, creating a natural pipeline chain.

---

## 4. `Inputs` System

`Inputs` is a typed, validated container (Pydantic `RootModel`).

### Validation via `_require`

Each feature's inner `Inputs` class declares what inputs it expects:

| `_require` value | Meaning | Example |
|------------------|---------|---------|
| `"nonempty"` (default) | At least one input required | Most features |
| `"empty"` | Must have no inputs (feature loads its own data) | Self-loading global features |
| `"any"` | Both empty and non-empty valid | Features with fit + assign modes |

```python
class MyFeature:
    # Only accepts tracks
    class Inputs(Inputs[TrackInput]):
        pass

class MySelfLoader:
    # Loads its own data, no pipeline inputs
    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "empty"

class MyKMeans:
    # fit needs data, assign can re-use saved model
    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"
```

### Useful properties

```python
inputs.has_tracks         # bool
inputs.is_single_tracks   # bool
inputs.is_single_feature  # bool
inputs.is_multi           # bool
inputs.feature_inputs     # tuple[Result, ...]
inputs.storage_suffix()   # e.g. "pair-wavelet+pair-egocentric"
```

---

## 5. Typed Params

Feature params are now Pydantic models with defaults and validation, replacing untyped dicts.

```python
class SpeedAngvel:
    class Params(Params):
        step_size: int | None = None

# Usage — dict overrides still work
feat = SpeedAngvel(params={"step_size": 3})
feat.params.step_size  # 3
feat.params["step_size"]  # also works (DictModel compatibility)
```

`Params.from_overrides(dict)` merges user overrides with field defaults, supporting 1-level deep merge.

---

## 6. Typed Artifacts (`ArtifactSpec`)

Artifacts are now typed spec objects instead of hand-written dicts.

```python
# Old
{"feature": "global-kmeans", "pattern": "model.joblib",
 "load": {"kind": "joblib"}, "run_id": "0.1-abc"}

# New
class ModelArtifact(ArtifactSpec):
    feature: str = "global-kmeans"
    pattern: str = "model.joblib"
    load: LoadSpec = Field(default_factory=JoblibLoadSpec)
```

Load specs: `NpzLoadSpec(key=...)`, `ParquetLoadSpec(...)`, `JoblibLoadSpec(key=...)`.

---

## 7. Time/Frame Filtering (New)

`run_feature()` now supports filtering data to a frame or time range:

```python
# By frame number
result = dataset.run_feature(feat, filter_start_frame=100, filter_end_frame=500)

# By time in seconds (requires fps_default in dataset metadata)
result = dataset.run_feature(feat, filter_start_time=5.0, filter_end_time=10.0)
```

- Frame and time filters are mutually exclusive per boundary (raises `ValueError` if both set).
- Frame range is included in the run_id hash for reproducibility.

---

## 8. COLUMNS Global Config

Centralized column name configuration replaces scattered string literals:

```python
from mosaic.behavior.feature_library import COLUMNS

COLUMNS.id_col           # "id"
COLUMNS.seq_col          # "sequence"
COLUMNS.group_col        # "group"
COLUMNS.frame_col        # "frame"
COLUMNS.time_col         # "time"
COLUMNS.x_col            # "X"
COLUMNS.y_col            # "Y"
COLUMNS.orientation_col  # "ANGLE"
COLUMNS.order_by         # "frames" | "time"
```

---

## 9. Feature Protocol (for implementing new features)

Features must satisfy this protocol:

```python
@final
@register_feature
class MyFeature:
    name = "my-feature"
    version = "0.1"
    parallelizable = True
    output_type: OutputType = "per_frame"  # "per_frame" | "global" | "summary" | "viz" | None

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        my_param: float = 1.0

    def __init__(self, inputs=Inputs(("tracks",)), params=None):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ds = None
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

    def bind_dataset(self, ds): ...
    def set_scope_filter(self, scope): ...
    def needs_fit(self) -> bool: ...
    def supports_partial_fit(self) -> bool: ...
    def loads_own_data(self) -> bool: ...
    def fit(self, X_iter): ...
    def partial_fit(self, df): ...
    def finalize_fit(self): ...
    def save_model(self, path): ...
    def load_model(self, path): ...
    def transform(self, df) -> pd.DataFrame: ...
```

---

## 10. Full Pipeline Example (Before → After)

### Before

```python
from mosaic.behavior import feature_library

# Step 1: speed
speed_feat = feature_library.speed_angvel.SpeedAngvel(params={})
speed_run = dataset.run_feature(speed_feat)  # "0.1-abc"

# Step 2: wavelet on speed
wave_feat = feature_library.pair_wavelet.PairWavelet(params={"f_min": 0.2})
wave_run = dataset.run_feature(
    wave_feat,
    input_kind="feature",
    input_feature="speed-angvel",
    input_run_id=speed_run,
)

# Step 3: multi-input t-SNE via inputset
dataset.save_inputset("my_set", {
    "inputs": [
        {"feature": "pair-wavelet", "run_id": wave_run},
        {"feature": "pair-egocentric", "run_id": ego_run},
    ]
})
tsne_feat = feature_library.global_tsne.GlobalTSNE(params={"total_templates": 2000})
tsne_run = dataset.run_feature(tsne_feat, input_kind="inputset", input_set="my_set")
```

### After

```python
from mosaic.behavior.feature_library import (
    SpeedAngvel, PairWavelet, PairEgocentric, GlobalTSNE, Inputs, Result
)

# Step 1: speed
speed_result = dataset.run_feature(SpeedAngvel())

# Step 2: wavelet on speed — pass Result directly
wave_result = dataset.run_feature(
    PairWavelet(Inputs((speed_result,)), params={"f_min": 0.2})
)

# Step 3: multi-input t-SNE — no inputset needed
tsne_result = dataset.run_feature(
    GlobalTSNE(
        Inputs((wave_result, ego_result)),
        params={"total_templates": 2000},
    )
)

# Results chain naturally
tsne_result.feature  # "global-tsne__from__pair-wavelet+pair-egocentric"
tsne_result.run_id   # "0.1-xyz789"
```
