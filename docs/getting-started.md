# Getting Started

This guide walks through a typical mosaic workflow from raw data to feature extraction.

## Installation

```bash
conda create -n mosaic python=3.12 -y
conda activate mosaic
conda install -c conda-forge ffmpeg -y
pip install -e ".[recommended]"
```

`ffmpeg` provides `ffprobe`, used by media indexing and raw H.264 support.
The `recommended` extra bundles wavelets + YOLO pose + PyTorch localizer; see
the [project README](https://github.com/EcodylicScience/mosaic#installation)
for finer-grained options.

## Create a dataset

A mosaic dataset is a directory with a YAML or JSON manifest pointing to
named roots for media, tracks, labels, features, and models.

```python
from mosaic.core.dataset import Dataset

ds = Dataset(manifest_path="my_project/dataset.yaml")
ds.set_root("media_raw", "my_project/videos")
ds.set_root("tracks_raw", "my_project/tracking_output")
ds.save()
```

## Index media

Scan directories for video files and collect metadata via ffprobe:

```python
ds.index_media(
    search_dirs=["my_project/videos"],
    extensions=(".mp4", ".avi", ".h264"),
)
```

This writes `media/index.csv` with columns for group, sequence, path,
width, height, fps, and codec.


## Index and convert tracks

```python
# Index raw tracking files
ds.index_tracks_raw(
    search_dirs=["my_project/tracking_output"],
    src_format="calms21_npy",
)

# Convert to standardized parquet format
ds.convert_all_tracks()
```

### Tracking videos with TRex (optional)

If you don't already have tracks, mosaic can drive [TRex](https://trex.run) to
detect and track animals, producing per-id `.npz` you then convert with
`src_format="trex_npz"`:

```python
from mosaic.tracking.trex import run_trex_convert, run_trex_track

conv = run_trex_convert("video.mp4", "out/", detect_model="yolo.pt",
                        track_max_individuals=4, trex_conda_env="track", display=":99")
trk  = run_trex_track(conv.pv_path, "out/", track_max_individuals=4,
                      trex_conda_env="track", display=":99")
```

**Two-env setup (recommended).** TRex's conda package pins `python=3.11` /
`numpy=1.26`, so install it in its **own** env rather than the mosaic env:

```bash
conda create -n track -c conda-forge -c trexing trex      # dedicated TRex env (py3.11)
pip install ultralytics torch                              # into `track`, for YOLO detection
```

Then point the mosaic caller at it with `trex_conda_env="track"` (or set
`MOSAIC_TREX_CONDA_ENV=track`); use `trex_bin=`/`MOSAIC_TREX_BIN` for an explicit
binary, or omit both for a `trex` already on `$PATH`. TRex needs an OpenGL/GLFW
display even headless — run **one** persistent virtual framebuffer and pass its
display (don't wrap `trex` in `xvfb-run`, which fork-bombs since TRex relaunches
itself):

```bash
Xvfb :99 -screen 0 1280x1024x24 &     # one persistent display; pass display=":99"
```

## Run features

Features are composable pipeline stages. Each produces per-sequence
parquet files versioned by `run_id`:

```python
from mosaic.behavior.feature_library import (
    SpeedAngvel, PairEgocentric, Inputs,
)

# Basic kinematic features (reads from tracks by default)
speed = SpeedAngvel()
speed_result = ds.run_feature(speed)

# Pair-egocentric features
ego = PairEgocentric()
ego_result = ds.run_feature(ego)
```

### Chain features together

```python
from mosaic.behavior.feature_library import (
    ExtractTemplates, GlobalScaler, GlobalTSNE,
)

# 1. Extract templates from upstream features
templates = ExtractTemplates(
    Inputs((speed_result, ego_result)),
    params={"n_templates": 2000},
)
templates_result = ds.run_feature(templates)

# 2. Fit scaler on those templates, apply per-sequence
scaler = GlobalScaler(
    Inputs((speed_result, ego_result)),
    params={"templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result)},
)
scaler_result = ds.run_feature(scaler)

# 3. Re-extract templates from the scaled output (farthest-first for coverage)
scaled_templates = ExtractTemplates(
    Inputs((scaler_result,)),
    params={"n_templates": 2000, "strategy": "farthest_first"},
)
scaled_templates_result = ds.run_feature(scaled_templates)

# 4. Fit t-SNE on the scaled templates, map every sequence
tsne = GlobalTSNE(
    Inputs((scaled_templates_result,)),
    params={
        "perplexity": 50,
        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
    },
)
tsne_result = ds.run_feature(tsne)
```

The full pattern (with k-means/Ward clustering, ground-truth alignment, and
XGBoost training on top) is shown in the
[CalMS21 template notebook](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb).

### Declarative pipelines

For multi-step workflows, use the `Pipeline` class instead of manual
chaining. It handles caching, staleness detection, and dependency tracking
automatically:

```python
from mosaic.core.pipeline import Pipeline, FeatureStep
from mosaic.behavior.feature_library import TrajectorySmooth, SpeedAngvel

pipe = Pipeline(default_run_kwargs={"parallel_workers": 8})
pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5}))
pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))

pipe.status(ds)         # check what's cached
results = pipe.run(ds)  # execute — cached steps are skipped
```

See the [Pipeline Guide](guide-pipeline.md) for the full API and examples.

## Next steps

- See the [Pipeline Guide](guide-pipeline.md) for declarative multi-step pipelines
